import os
import json
import base64
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union
from types import SimpleNamespace
from aisuite.framework.chat_completion_response import ChatCompletionResponse, Choice, ChoiceDelta, StreamChoice
from aisuite.framework.message import Message, ChatCompletionMessageToolCall, Function, ReasoningContent
from aisuite.framework.replay_payload import (
    ProviderReplayCapabilities,
    ReplayBuildResult,
    ReplayCaptureResult,
    ReplayDiagnostic,
    ReplayValidationResult,
    build_replay_payload,
    get_replay_payload,
    unwrap_replay_payload,
)
from aisuite.framework.stop_reason import stop_reason_manager
from aisuite.provider import Provider, LLMError

# Import Google GenAI SDK lazily so helper functions remain importable when the
# optional dependency is absent in the current environment.
_GEMINI_IMPORT_ERROR = None
try:
    from google import genai
    from google.genai import types
except ModuleNotFoundError as exc:
    _GEMINI_IMPORT_ERROR = exc
    genai = SimpleNamespace(Client=None)
    types = SimpleNamespace()


logger = logging.getLogger(__name__)

def _normalize_gemini_usage(usage_obj):
    """Normalize Gemini usage/usageMetadata to AISuite standard dict.

    Maps promptTokenCount/prompt_token_count -> prompt_tokens,
    responseTokenCount/response_token_count (or candidatesTokenCount/candidates_token_count) -> completion_tokens,
    totalTokenCount/total_token_count -> total_tokens,
    cachedContentTokenCount/cached_content_token_count -> cache_read_input_tokens.
    """
    if not usage_obj:
        return None

    # Try to extract as plain dict
    if hasattr(usage_obj, "model_dump"):
        data = usage_obj.model_dump()
    elif isinstance(usage_obj, dict):
        data = usage_obj
    else:
        data = {}
        for attr in (
            "prompt_token_count",
            "cached_content_token_count",
            "response_token_count",
            "candidates_token_count",
            "total_token_count",
            "promptTokenCount",
            "cachedContentTokenCount",
            "responseTokenCount",
            "candidatesTokenCount",
            "totalTokenCount",
        ):
            if hasattr(usage_obj, attr):
                data[attr] = getattr(usage_obj, attr)

    prompt_tokens = data.get("prompt_token_count")
    if prompt_tokens is None:
        prompt_tokens = data.get("promptTokenCount")
    cached_content_tokens = data.get("cached_content_token_count")
    if cached_content_tokens is None:
        cached_content_tokens = data.get("cachedContentTokenCount")
    completion_tokens = data.get("response_token_count")
    if completion_tokens is None:
        completion_tokens = data.get("responseTokenCount")
    if completion_tokens is None:
        completion_tokens = data.get("candidates_token_count")
    if completion_tokens is None:
        completion_tokens = data.get("candidatesTokenCount")
    total_tokens = (
        data.get("total_token_count")
        or data.get("totalTokenCount")
        or (
            (prompt_tokens or 0) + (completion_tokens or 0)
            if prompt_tokens is not None and completion_tokens is not None
            else None
        )
    )

    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None

    return {
        "prompt_tokens": int(prompt_tokens) if prompt_tokens is not None else None,
        "completion_tokens": int(completion_tokens) if completion_tokens is not None else None,
        "total_tokens": int(total_tokens) if total_tokens is not None else None,
        "cache_read_input_tokens": int(cached_content_tokens) if cached_content_tokens is not None else 0,
        "cache_write_input_tokens": 0,
        "cache_write_by_ttl": {
            "ephemeral_5m_input_tokens": 0,
            "ephemeral_1h_input_tokens": 0,
        },
    }


GEMINI_PROVIDER_NAME = "gemini"
GEMINI_SIGNATURE_PLACEHOLDER = "skip_thought_signature_validator"
GEMINI_TOOL_CALL_REPLAY_KIND = "gemini_tool_call"
GEMINI_THOUGHT_PARTS_REPLAY_KIND = "gemini_thought_parts"

_TOOL_CALL_ERROR_FINISH_REASONS = {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS"}
_PROTOCOL_ERROR_FINISH_REASONS = {"MISSING_THOUGHT_SIGNATURE", "MALFORMED_RESPONSE"}


def _normalize_finish_reason(reason: Any) -> Optional[str]:
    if reason is None:
        return None
    if isinstance(reason, str):
        return reason
    # Prefer enum name/value when available
    name = getattr(reason, "name", None)
    if isinstance(name, str) and name:
        return name
    value = getattr(reason, "value", None)
    if isinstance(value, str) and value:
        return value
    text = str(reason)
    if "." in text:
        # e.g. "FinishReason.MALFORMED_FUNCTION_CALL" -> "MALFORMED_FUNCTION_CALL"
        return text.rsplit(".", 1)[-1]
    return text


def _extract_provider_call_id(source: Any) -> Optional[str]:
    if source is None:
        return None

    if isinstance(source, dict):
        for key in ("id", "call_id", "callId"):
            value = source.get(key)
            if value:
                return str(value)
        payload = source.get("payload")
        if payload is not None:
            nested = _extract_provider_call_id(payload)
            if nested:
                return nested
        return None

    for attr in ("id", "call_id", "callId"):
        value = getattr(source, attr, None)
        if value:
            return str(value)

    if hasattr(source, "model_dump"):
        try:
            dumped = source.model_dump()
            if isinstance(dumped, dict):
                return _extract_provider_call_id(dumped)
        except Exception:
            pass

    return None


def _extract_thought_signature(source: Any) -> Optional[str]:
    """Best-effort extraction of thought_signature / thoughtSignature."""
    if source is None:
        return None

    # Handle list of possible containers
    if isinstance(source, (list, tuple)):
        for item in source:
            sig = _extract_thought_signature(item)
            if sig:
                return sig
        return None

    # Handle dict-based structures
    if isinstance(source, dict):
        for key in ("thought_signature", "thoughtSignature"):
            sig = source.get(key)
            if sig:
                return sig
        google_meta = source.get("google") or {}
        if isinstance(google_meta, dict):
            for key in ("thought_signature", "thoughtSignature"):
                sig = google_meta.get(key)
                if sig:
                    return sig
        # Recurse into nested metadata
        for nested_key in ("metadata", "additional_kwargs", "payload", "meta"):
            nested = source.get(nested_key)
            sig = _extract_thought_signature(nested)
            if sig:
                return sig
        return None

    # Attribute-based objects (pydantic / SDK models)
    for attr in ("thought_signature", "thoughtSignature"):
        sig = getattr(source, attr, None)
        if sig:
            return sig

    # Some SDK objects carry metadata/additional kwargs
    for attr in ("metadata", "additional_kwargs"):
        nested = getattr(source, attr, None)
        sig = _extract_thought_signature(nested)
        if sig:
            return sig

    # Fallback: model_dump() if available
    if hasattr(source, "model_dump"):
        try:
            data = source.model_dump()
            if isinstance(data, dict):
                return _extract_thought_signature(data)
        except Exception:
            pass

    return None


def _build_tool_call_extra(
    signature: Optional[str],
    *,
    provider_call_id: Optional[str] = None,
    provider_function_name: Optional[str] = None,
    replay_mode: str = "provider_exact_turn",
    placeholder: bool = False,
    degraded: bool = False,
    degraded_reason: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build replay metadata for a Gemini tool call."""
    normalized_signature = None
    if signature is not None:
        if isinstance(signature, bytes):
            normalized_signature = "base64:" + base64.b64encode(signature).decode("ascii")
        else:
            normalized_signature = str(signature)

    if not normalized_signature and not placeholder and not provider_call_id and not provider_function_name:
        return None

    if (
        not normalized_signature
        and not placeholder
        and (provider_call_id or provider_function_name)
        and not degraded
    ):
        degraded = True
        degraded_reason = degraded_reason or "missing_thought_signature"
        if replay_mode == "provider_exact_turn":
            replay_mode = "degraded_legacy_turn"

    payload: Dict[str, Any] = {
        "provider_call_id": provider_call_id,
        "provider_function_name": provider_function_name,
        "replay_mode": replay_mode,
        "thought_signature": normalized_signature,
    }
    meta: Dict[str, Any] = {}
    if placeholder:
        meta["placeholder"] = True
    if degraded:
        meta["degraded"] = True
    if degraded_reason:
        meta["degraded_reason"] = degraded_reason

    legacy_google_meta: Dict[str, Any] = {}
    if normalized_signature:
        legacy_google_meta["thought_signature"] = normalized_signature
    if placeholder:
        legacy_google_meta["placeholder"] = True

    legacy_fields: Dict[str, Any] = {
        "provider": GEMINI_PROVIDER_NAME,
        "provider_call_id": provider_call_id,
        "provider_function_name": provider_function_name,
        "replay_mode": replay_mode,
    }
    if legacy_google_meta:
        legacy_fields["google"] = legacy_google_meta

    return build_replay_payload(
        GEMINI_PROVIDER_NAME,
        GEMINI_TOOL_CALL_REPLAY_KIND,
        payload,
        meta=meta,
        legacy_fields=legacy_fields,
    )


def _extract_gemini_tool_call_metadata(tool_call: Any) -> Dict[str, Any]:
    extra = _get_tool_call_extra(tool_call)
    if not isinstance(extra, dict):
        return {}

    envelope = get_replay_payload(extra)
    if envelope and envelope.get("provider") == GEMINI_PROVIDER_NAME:
        payload = unwrap_replay_payload(extra)
        meta = envelope.get("meta")
        result = payload.copy() if isinstance(payload, dict) else {}
        if isinstance(meta, dict):
            result["_meta"] = meta
        return result

    google_meta = extra.get("google")
    metadata: Dict[str, Any] = {
        "provider_call_id": extra.get("provider_call_id"),
        "provider_function_name": extra.get("provider_function_name"),
        "replay_mode": extra.get("replay_mode"),
    }
    if isinstance(google_meta, dict):
        metadata["thought_signature"] = google_meta.get("thought_signature")
        metadata["_meta"] = {
            "placeholder": bool(google_meta.get("placeholder")),
        }
    elif "thought_signature" in extra:
        metadata["thought_signature"] = extra.get("thought_signature")

    return {k: v for k, v in metadata.items() if v is not None or k == "_meta"}


def _is_degraded_gemini_tool_call_metadata(metadata: Dict[str, Any]) -> bool:
    if not metadata:
        return False

    meta = metadata.get("_meta")
    if isinstance(meta, dict) and meta.get("degraded"):
        return True

    replay_mode = metadata.get("replay_mode")
    return isinstance(replay_mode, str) and replay_mode not in {"", "provider_exact_turn"}


def _is_exact_gemini_tool_call_metadata(metadata: Dict[str, Any]) -> bool:
    return bool(metadata) and not _is_degraded_gemini_tool_call_metadata(metadata)


def _build_gemini_thought_parts_raw_data(thought_parts: List[Dict[str, Any]], thinking_text: str) -> Dict[str, Any]:
    payload = {
        "thought_parts": thought_parts,
        "thinking_text": thinking_text,
    }
    return build_replay_payload(
        GEMINI_PROVIDER_NAME,
        GEMINI_THOUGHT_PARTS_REPLAY_KIND,
        payload,
        legacy_fields=payload,
    )


def _extract_gemini_reasoning_payload(raw_data: Any) -> Dict[str, Any]:
    envelope = get_replay_payload(raw_data)
    if envelope and envelope.get("provider") == GEMINI_PROVIDER_NAME:
        payload = unwrap_replay_payload(raw_data)
        if isinstance(payload, dict):
            return payload
        return {}
    if isinstance(raw_data, dict):
        return raw_data
    return {}


def _get_tool_call_extra(tool_call: Any) -> Optional[Dict[str, Any]]:
    """Extract the extra_content dict from a tool_call structure."""
    if tool_call is None:
        return None
    if isinstance(tool_call, ChatCompletionMessageToolCall):
        return getattr(tool_call, "extra_content", None)
    if isinstance(tool_call, dict):
        extra = tool_call.get("extra_content")
        if extra:
            return extra
    return None


def _normalize_thinking_config(value: Any) -> Dict[str, Any]:
    """Normalize thinking_config input into SDK-friendly snake_case keys."""
    if value is None:
        return {}

    data: Dict[str, Any] = {}
    if isinstance(value, types.ThinkingConfig):
        try:
            data = value.model_dump()
        except Exception:
            try:
                data = value.dict()
            except Exception:
                data = {}
    elif isinstance(value, dict):
        data = value.copy()
    elif hasattr(value, "__dict__"):
        data = dict(value.__dict__)

    if not data:
        return {}

    normalized: Dict[str, Any] = {}
    remap = {
        "includeThoughts": "include_thoughts",
        "thinkingLevel": "thinking_level",
        "thinkingBudget": "thinking_budget",
    }
    for key, val in data.items():
        normalized[remap.get(key, key)] = val

    return normalized


def _filter_thinking_config_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Drop thinking_config keys unsupported by the installed SDK."""
    if not data:
        return {}

    allowed = None
    if hasattr(types.ThinkingConfig, "model_fields"):
        allowed = set(types.ThinkingConfig.model_fields.keys())
    elif hasattr(types.ThinkingConfig, "__fields__"):
        allowed = set(types.ThinkingConfig.__fields__.keys())

    if not allowed:
        return data

    return {key: value for key, value in data.items() if key in allowed}


class GeminiMessageConverter:

    @staticmethod
    def to_gemini_request(conversation):
        """
        Convert AISuite conversation (list of messages) to Gemini API request format.
        """
        system_instruction = None
        messages = conversation

        # If the first message is a system role, use it as systemInstruction for Gemini
        if messages and messages[0].get("role") == "system":
            system_instruction = messages[0]["content"]
            messages = messages[1:]  # remove system message from main history

        # Build Gemini 'contents' list from remaining messages
        contents = []
        for msg in messages:
            role = msg.get("role")
            content_text = msg.get("content", "")
            # Map AISuite role to Gemini role (Gemini expects "user" or "model")
            if role == "assistant":
                role = "model"
            elif role == "user":
                role = "user"
            else:
                # Other roles (if any) can be treated as user by default
                role = "user"
            # Each content entry has a role and parts (here just one text part)
            content_entry = {
                "role": role,
                "parts": [ {"text": content_text} ]
            }
            contents.append(content_entry)

        # Construct the request payload for Gemini API
        request_payload = {"contents": contents}
        if system_instruction:
            # Gemini expects system instructions separately (as Content object)
            request_payload["systemInstruction"] = {
                "parts": [ {"text": system_instruction} ]
            }
        return request_payload

    @staticmethod
    def from_gemini_response(response):
        """
        将 Gemini API 响应转换为 AISuite 的 ChatCompletionResponse 格式。

        Args:
            response: Gemini API 的响应对象
        Returns:
            ChatCompletionResponse 对象
        """
        # Extract tool calls and reasoning content if present
        tool_calls = None
        content = response.text
        reasoning_content = None

        candidate = response.candidates[0] if response.candidates else None
        candidate_content = getattr(candidate, "content", None)
        candidate_parts = getattr(candidate_content, "parts", None)

        if candidate_parts:
            reasoning_text_parts = []
            reasoning_part_data = []
            content_text_parts = []

            for part in candidate_parts:
                # Check if the part is a thought and has text
                if getattr(part, 'thought', False) and getattr(part, 'text', None):
                    reasoning_text_parts.append(part.text)
                    if hasattr(part, 'model_dump'):
                        reasoning_part_data.append(part.model_dump())
                    elif hasattr(part, 'dict'):
                        reasoning_part_data.append(part.dict())
                    else:
                        reasoning_part_data.append({
                            "thought": True,
                            "text": part.text,
                        })
                # Check if the part is a function call
                elif hasattr(part, 'function_call') and part.function_call:
                    try:
                        provider_call_id = _extract_provider_call_id(part.function_call)
                        signature = _extract_thought_signature(part) or _extract_thought_signature(part.function_call)
                        function = Function(
                            name=part.function_call.name,
                            arguments=json.dumps(part.function_call.args) if hasattr(part.function_call, 'args') else "{}"
                        )

                        tool_call_obj = ChatCompletionMessageToolCall(
                            id=provider_call_id or f"call_{part.function_call.name}_{hash(str(part.function_call.args)) % 10000}",
                            function=function,
                            type="function",
                            extra_content=_build_tool_call_extra(
                                signature,
                                provider_call_id=provider_call_id,
                                provider_function_name=part.function_call.name,
                            )
                        )

                        if tool_calls is None:
                            tool_calls = []
                        tool_calls.append(tool_call_obj)
                    except Exception:
                        # If there's any error processing the function call, skip it
                        pass
                # Else, if it's not a thought but has text, it's regular content
                elif getattr(part, 'text', None):
                    content_text_parts.append(part.text)

            # Combine reasoning parts if any
            if reasoning_text_parts:
                combined_thinking = "".join(reasoning_text_parts)
                reasoning_content = ReasoningContent(
                    thinking=combined_thinking,
                    provider="gemini",
                    raw_data=_build_gemini_thought_parts_raw_data(reasoning_part_data, combined_thinking),
                )

            # Use combined content text or fallback to response.text
            if content_text_parts:
                content = "".join(content_text_parts)

        # Get finish_reason for stop_info creation
        finish_reason = _normalize_finish_reason(response.candidates[0].finish_reason) if response.candidates else None

        # Create stop_info (we need to import stop_reason_manager at the top level)
        stop_info = None
        if finish_reason:
            choice_data = {
                "content": {
                    "parts": [{"text": content}] if content else []
                }
            }
            # Add tool calls to choice_data if present
            if tool_calls:
                choice_data["content"]["parts"].extend([{"functionCall": tc} for tc in tool_calls])

            # Import stop_reason_manager here to avoid circular imports
            from aisuite.framework.stop_reason import stop_reason_manager
            stop_info = stop_reason_manager.map_stop_reason("gemini", finish_reason, {
                "has_content": bool(content or tool_calls),
                "content_length": len(content) if content else 0,
                "tool_calls_count": len(tool_calls) if tool_calls else 0,
                "finish_reason": finish_reason,
                "model": response.model_version,
                "provider": "gemini"
            })

        # 提取 usage 元数据（token 统计）
        usage_metadata = getattr(response, "usage_metadata", None) or getattr(response, "usageMetadata", None)
        usage = _normalize_gemini_usage(usage_metadata)

        # 创建 ChatCompletionResponse 对象
        metadata = {
            "model": response.model_version,  # 模型名称
            # Gemini API 目前不会返回这些字段
            "id": None,
            "created": None,
        }
        if usage:
            metadata["usage"] = usage

        return ChatCompletionResponse(
            choices=[
                Choice(
                    index=0,  # Gemini 通常只返回一个选项
                    message=Message(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls,
                        refusal=None,
                        reasoning_content=reasoning_content,
                    ),  # 使用 Message 对象包装响应内容
                    finish_reason=finish_reason,
                    stop_info=stop_info,
                )
            ],
            metadata=metadata,
        )



class GeminiProvider(Provider):
    def get_replay_capabilities(self, model: str | None = None) -> ProviderReplayCapabilities:
        return ProviderReplayCapabilities(
            needs_exact_turn_replay=True,
            needs_provider_call_id_binding=True,
            needs_reasoning_raw_replay=True,
            supports_canonical_only_history=False,
            empty_actionless_stop_is_retryable=True,
        )

    def capture_response(self, response, model: str | None = None, **kwargs):
        if not response or not getattr(response, "choices", None):
            return ReplayCaptureResult()
        choice = response.choices[0]
        message = getattr(choice, "message", None)
        if not message:
            return ReplayCaptureResult(stop_info=getattr(choice, "stop_info", None))
        reasoning_content = getattr(message, "reasoning_content", None)
        return ReplayCaptureResult(
            canonical_message=message,
            stop_info=getattr(choice, "stop_info", None),
            replay_metadata={
                "tool_calls": getattr(message, "tool_calls", None),
                "reasoning_content": getattr(reasoning_content, "raw_data", None) if reasoning_content else None,
            },
            protocol_diagnostics=(),
        )

    def _convert_reasoning_content(self, thinking_text, parts):
        """Convert Gemini thinking content to ReasoningContent object."""
        if not thinking_text:
            return None

        # Extract raw thought parts for reconstruction
        thought_parts = []
        for part in parts:
            if getattr(part, 'thought', False):
                part_data = {}
                if hasattr(part, 'model_dump'):
                    part_data = part.model_dump()
                elif hasattr(part, 'dict'):
                    part_data = part.dict()
                else:
                    part_data = {
                        'thought': True,
                        'text': getattr(part, 'text', '')
                    }
                thought_parts.append(part_data)

        return ReasoningContent(
            thinking=thinking_text,
            provider="gemini",
            raw_data=_build_gemini_thought_parts_raw_data(thought_parts, thinking_text)
        )

    def __init__(self, **kwargs):
        """Initialize the Gemini provider with API key and client."""
        if getattr(genai, "Client", None) is None:
            raise RuntimeError(
                "google-genai is required for GeminiProvider"
            ) from _GEMINI_IMPORT_ERROR
        api_key = os.environ.get("GEMINI_API_KEY") or kwargs.get("api_key")
        if api_key is None:
            raise RuntimeError("GEMINI_API_KEY is required for GeminiProvider")
        # Initialize the GenAI client for Gemini (non-Vertex usage)
        self.client = genai.Client(api_key=api_key)

        # State for accumulating streaming tool calls
        self._streaming_tool_calls = {}

        # Track accumulated content for streaming responses
        # Used to provide accurate metadata in stop_info
        self._stream_content_length = 0
        self._stream_tool_calls_count = 0

    def _is_gemini_3_model(self, model_id: str) -> bool:
        """Heuristic check for Gemini 3 series models.

        We use a simple substring match so it works for both plain
        "gemini-3-..." and provider-prefixed names like "google/gemini-3-...".
        """
        if not model_id:
            return False
        return "gemini-3" in model_id

    def _is_gemini_3_flash_model(self, model_id: str) -> bool:
        """Heuristic check for Gemini 3 Flash models."""
        if not model_id:
            return False
        return "gemini-3" in model_id and "flash" in model_id

    def _resolve_tool_call_signature(self, model_id: str, tool_call: Any) -> Tuple[Optional[str], bool]:
        """
        Determine which signature should be used for a tool_call when replaying history.

        Returns:
            (signature, is_placeholder)
        """
        metadata = _extract_gemini_tool_call_metadata(tool_call)
        signature = metadata.get("thought_signature") or _extract_thought_signature(tool_call)
        placeholder = False

        # Gemini 3 requires a thought_signature for replaying Gemini-originated
        # tool calls. Placeholder fallback is reserved for degraded/legacy
        # histories that do not carry Gemini-native replay metadata.
        if not signature and self._is_gemini_3_model(model_id) and self._should_use_placeholder_signature(tool_call, metadata):
            signature = GEMINI_SIGNATURE_PLACEHOLDER
            placeholder = True

        return signature, placeholder

    def _should_use_placeholder_signature(self, tool_call: Any, metadata: Dict[str, Any]) -> bool:
        if not metadata:
            return True

        if _is_degraded_gemini_tool_call_metadata(metadata):
            return True

        replay_mode = metadata.get("replay_mode")
        provider_call_id = metadata.get("provider_call_id")
        provider_function_name = metadata.get("provider_function_name")

        provider = None
        extra = _get_tool_call_extra(tool_call)
        if isinstance(extra, dict):
            provider = extra.get("provider")
            envelope = get_replay_payload(extra)
            if envelope:
                provider = envelope.get("provider") or provider

        is_gemini_history = (
            provider == GEMINI_PROVIDER_NAME
            or bool(provider_call_id)
            or bool(provider_function_name)
            or replay_mode == "provider_exact_turn"
        )
        return not is_gemini_history

    def _attach_thought_signature(self, part: Any, signature: Optional[str]) -> None:
        """Attach thought_signature metadata to a Part/function_call if possible."""
        if signature is None or part is None:
            return

        if isinstance(signature, bytes):
            sig_bytes: bytes = signature
        elif isinstance(signature, str) and signature.startswith("base64:"):
            try:
                sig_bytes = base64.b64decode(signature[len("base64:"):], validate=False)
            except Exception:
                sig_bytes = signature.encode("utf-8", errors="replace")
        else:
            sig_bytes = str(signature).encode("utf-8", errors="replace")

        try:
            setattr(part, "thought_signature", sig_bytes)
        except Exception:
            pass
        try:
            setattr(part, "thoughtSignature", sig_bytes)
        except Exception:
            pass

        function_call = getattr(part, "function_call", None)
        if function_call:
            try:
                setattr(function_call, "thought_signature", sig_bytes)
            except Exception:
                pass
            try:
                setattr(function_call, "thoughtSignature", sig_bytes)
            except Exception:
                pass

    def _build_protocol_diagnostic(
        self,
        code: str,
        message: str,
        *,
        severity: str = "error",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReplayDiagnostic:
        return ReplayDiagnostic(
            code=code,
            message=message,
            severity=severity,
            provider=GEMINI_PROVIDER_NAME,
            metadata=metadata or {},
        )

    def _normalize_message_dict(self, msg: Any) -> Dict[str, Any]:
        if isinstance(msg, dict):
            return msg
        if hasattr(msg, "model_dump"):
            try:
                return msg.model_dump()
            except Exception:
                pass
        return {}

    def _find_replay_tool_call_metadata(self, messages: List[Any], tool_call_id: str) -> Dict[str, Any]:
        for msg in reversed(messages):
            msg_dict = self._normalize_message_dict(msg)
            if msg_dict.get("role") != "assistant":
                continue
            tool_calls = msg_dict.get("tool_calls") or []
            for tool_call in reversed(tool_calls):
                if isinstance(tool_call, dict):
                    current_id = tool_call.get("id")
                else:
                    current_id = getattr(tool_call, "id", None)
                if current_id == tool_call_id:
                    metadata = _extract_gemini_tool_call_metadata(tool_call)
                    if metadata:
                        return metadata
                    function_name = None
                    if isinstance(tool_call, dict):
                        function_name = (tool_call.get("function") or {}).get("name")
                    elif getattr(tool_call, "function", None):
                        function_name = getattr(tool_call.function, "name", None)
                    if function_name:
                        return {
                            "provider_function_name": function_name,
                            "_legacy_fallback": True,
                            "_exact_metadata": False,
                        }
        return {}

    def _resolve_tool_response_replay_metadata(
        self,
        model_id: str,
        msg: Dict[str, Any],
        replay_messages: Optional[List[Any]] = None,
    ) -> Tuple[Dict[str, Any], Optional[ReplayDiagnostic]]:
        tool_call_id = msg.get("tool_call_id")
        if not tool_call_id:
            return {}, None

        replay_metadata = self._find_replay_tool_call_metadata(replay_messages or [], tool_call_id)
        if replay_metadata:
            replay_metadata = dict(replay_metadata)

        provider_function_name = replay_metadata.get("provider_function_name")
        provider_call_id = replay_metadata.get("provider_call_id")
        exact_metadata = (
            bool(replay_metadata)
            and not replay_metadata.get("_legacy_fallback", False)
            and not _is_degraded_gemini_tool_call_metadata(replay_metadata)
        )
        replay_metadata["_exact_metadata"] = exact_metadata

        if self._is_gemini_3_model(model_id):
            if not provider_function_name:
                diagnostic = self._build_protocol_diagnostic(
                    "missing_provider_function_name",
                    "Gemini tool replay metadata is missing provider_function_name.",
                    metadata={"tool_call_id": tool_call_id},
                )
                return replay_metadata, diagnostic
            if exact_metadata and not provider_call_id:
                diagnostic = self._build_protocol_diagnostic(
                    "missing_provider_call_id",
                    "Gemini tool replay metadata is missing provider_call_id.",
                    metadata={"tool_call_id": tool_call_id},
                )
                return replay_metadata, diagnostic
            return replay_metadata, None

        if not provider_function_name:
            function_name = "unknown_function"
            if tool_call_id.startswith("call_"):
                parts = tool_call_id.split("_")
                if len(parts) >= 3:
                    function_name = "_".join(parts[1:-1])
            replay_metadata["provider_function_name"] = function_name

        return replay_metadata, None

    def _create_function_response_part(
        self,
        *,
        function_name: str,
        content: Any,
        provider_call_id: Optional[str] = None,
    ):
        response_payload = {"result": content}
        function_response_cls = getattr(types, "FunctionResponse", None)
        part_cls = getattr(types, "Part", None)

        if function_response_cls is not None and part_cls is not None:
            try:
                kwargs = {
                    "name": function_name,
                    "response": response_payload,
                }
                if provider_call_id:
                    kwargs["id"] = provider_call_id
                function_response = function_response_cls(**kwargs)
                return part_cls(function_response=function_response)
            except Exception:
                pass

        return types.Part.from_function_response(
            name=function_name,
            response=response_payload,
        )

    def _create_function_call_part(
        self,
        *,
        function_name: str,
        parsed_args: Dict[str, Any],
        provider_call_id: Optional[str] = None,
    ):
        function_call_cls = getattr(types, "FunctionCall", None)
        part_cls = getattr(types, "Part", None)

        if function_call_cls is not None and part_cls is not None:
            try:
                kwargs = {
                    "name": function_name,
                    "args": parsed_args,
                }
                if provider_call_id:
                    kwargs["id"] = provider_call_id
                function_call = function_call_cls(**kwargs)
                return part_cls(function_call=function_call)
            except Exception:
                pass

        return types.Part.from_function_call(
            name=function_name,
            args=parsed_args,
        )

    def _build_tool_response_parts(
        self,
        model_id: str,
        msg: Dict[str, Any],
        replay_messages: Optional[List[Any]] = None,
    ):
        tool_call_id = msg.get("tool_call_id")
        if not tool_call_id:
            return None, None

        replay_metadata, diagnostic = self._resolve_tool_response_replay_metadata(
            model_id,
            msg,
            replay_messages=replay_messages,
        )
        if diagnostic:
            return None, diagnostic

        provider_function_name = replay_metadata.get("provider_function_name")
        provider_call_id = replay_metadata.get("provider_call_id")

        part = self._create_function_response_part(
            function_name=provider_function_name,
            content=msg.get("content"),
            provider_call_id=provider_call_id,
        )
        return [part], None

    def validate_replay_window(self, model: str, messages: list, **kwargs) -> ReplayValidationResult:
        diagnostics: List[ReplayDiagnostic] = []
        degraded = False

        if not self._is_gemini_3_model(model):
            return ReplayValidationResult(ok=True)

        normalized_messages = [self._normalize_message_dict(msg) for msg in messages]

        for msg in normalized_messages:
            role = msg.get("role")
            if role == "assistant" and isinstance(msg.get("tool_calls"), list):
                for tool_call in msg["tool_calls"]:
                    metadata = _extract_gemini_tool_call_metadata(tool_call)
                    if not metadata:
                        degraded = True
                        continue
                    is_degraded_metadata = _is_degraded_gemini_tool_call_metadata(metadata)
                    if is_degraded_metadata:
                        degraded = True
                    if not metadata.get("provider_call_id"):
                        diagnostics.append(
                            self._build_protocol_diagnostic(
                                "missing_provider_call_id",
                                "Gemini replay metadata is missing provider_call_id.",
                                metadata={"tool_call_id": getattr(tool_call, "id", None) if not isinstance(tool_call, dict) else tool_call.get("id")},
                            )
                        )
                    if not metadata.get("provider_function_name"):
                        diagnostics.append(
                            self._build_protocol_diagnostic(
                                "missing_provider_function_name",
                                "Gemini replay metadata is missing provider_function_name.",
                                metadata={"tool_call_id": getattr(tool_call, "id", None) if not isinstance(tool_call, dict) else tool_call.get("id")},
                            )
                        )
                    if not metadata.get("thought_signature") and _is_exact_gemini_tool_call_metadata(metadata):
                        diagnostics.append(
                            self._build_protocol_diagnostic(
                                "missing_thought_signature",
                                "Gemini replay metadata is missing thought_signature.",
                                metadata={"tool_call_id": getattr(tool_call, "id", None) if not isinstance(tool_call, dict) else tool_call.get("id")},
                            )
                        )

            if role == "tool" and msg.get("tool_call_id"):
                replay_metadata, diagnostic = self._resolve_tool_response_replay_metadata(
                    model,
                    msg,
                    replay_messages=normalized_messages,
                )
                if diagnostic:
                    diagnostics.append(diagnostic)
                    continue
                if replay_metadata and not replay_metadata.get("_exact_metadata", True):
                    degraded = True

        return ReplayValidationResult(
            ok=not any(diag.severity == "error" for diag in diagnostics),
            degraded=degraded,
            diagnostics=tuple(diagnostics),
        )

    def build_replay_view(self, model: str, messages: list, **kwargs):
        validation = self.validate_replay_window(model, messages, **kwargs)
        if not validation.ok:
            error_codes = ", ".join(diag.code for diag in validation.diagnostics if diag.severity == "error")
            raise LLMError(f"Gemini replay window validation failed: {error_codes}")
        normalized_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                normalized_messages.append(msg)
            elif hasattr(msg, "model_dump"):
                normalized_messages.append(msg.model_dump())
            else:
                normalized_messages.append(msg)
        replay_mode = "provider_exact_turn"
        if validation.degraded:
            replay_mode = "degraded_legacy_turn"
        return ReplayBuildResult(
            request_view=self._preprocess_messages_for_gemini(normalized_messages),
            replay_mode=replay_mode,
            degraded=validation.degraded,
            diagnostics=validation.diagnostics,
        )

    def _parse_tool_call_args(self, args: Any) -> Dict[str, Any]:
        """Parse tool call arguments into a dictionary."""
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            try:
                return json.loads(args)
            except json.JSONDecodeError:
                return {}
        return {}

    def _convert_assistant_tool_message_to_parts(self, model_id: str, msg: Dict[str, Any]) -> Optional[list]:
        """
        Convert an assistant message containing tool_calls into Gemini parts,
        ensuring thought_signature is attached when available (or placeholder when required).
        """
        parts = []

        if msg.get("content"):
            parts.append(types.Part.from_text(text=msg["content"]))

        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            return parts if parts else None

        for tool_call in tool_calls:
            func_name = None
            raw_args: Any = "{}"
            provider_call_id = None

            if isinstance(tool_call, ChatCompletionMessageToolCall):
                func_name = tool_call.function.name
                raw_args = tool_call.function.arguments
            elif isinstance(tool_call, dict):
                func = tool_call.get("function", {})
                func_name = func.get("name")
                raw_args = func.get("arguments", "{}")

            if not func_name:
                continue

            metadata = _extract_gemini_tool_call_metadata(tool_call)
            provider_call_id = metadata.get("provider_call_id")
            parsed_args = self._parse_tool_call_args(raw_args)
            signature, placeholder = self._resolve_tool_call_signature(model_id, tool_call)

            if (
                self._is_gemini_3_model(model_id)
                and _is_exact_gemini_tool_call_metadata(metadata)
                and not signature
            ):
                raise LLMError(
                    f"Gemini replay metadata for tool_call {getattr(tool_call, 'id', None) if not isinstance(tool_call, dict) else tool_call.get('id')} is missing thought_signature"
                )

            function_call_part = self._create_function_call_part(
                function_name=func_name,
                parsed_args=parsed_args,
                provider_call_id=provider_call_id,
            )
            if signature:
                self._attach_thought_signature(function_call_part, signature)
            parts.append(function_call_part)

        return parts if parts else None

    def _create_stop_info(self, finish_reason: str, choice_data: dict = None, model: str = None) -> dict:
        """Create StopInfo from Gemini finish_reason."""
        if not finish_reason:
            return None

        # Analyze choice data to determine content presence
        has_content = False
        content_length = 0
        tool_calls_count = 0

        if choice_data:
            # Check content from Gemini candidate
            content = choice_data.get("content")
            if content:
                # Gemini content has parts array
                parts = content.get("parts", [])
                for part in parts:
                    if part.get("text"):
                        has_content = True
                        content_length += len(part.get("text", ""))
                    elif part.get("functionCall"):
                        tool_calls_count += 1
                        has_content = True  # Tool calls count as content

        metadata = {
            "has_content": has_content,
            "content_length": content_length,
            "tool_calls_count": tool_calls_count,
            "finish_reason": finish_reason,
            "model": model,
            "provider": "gemini"
        }

        # Map Gemini finish_reason to standard StopReason
        return stop_reason_manager.map_stop_reason("gemini", finish_reason, metadata)



    def _convert_tool_spec(self, openai_tools):
        """
        Convert OpenAI tools format to Gemini function_declarations format.

        OpenAI format:
        {
          "tools": [
            {
              "type": "function",
              "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {...}
              }
            }
          ]
        }

        Gemini format:
        {
          "tools": [
            {
              "function_declarations": [
                {
                  "name": "get_weather",
                  "description": "Get weather information",
                  "parameters": {...}
                }
              ]
            }
          ]
        }

        Args:
            openai_tools: List of tools in OpenAI format

        Returns:
            List of tools in Gemini format, or None if no valid tools
        """
        if not openai_tools:
            return None

        function_declarations = []

        for tool in openai_tools:
            # Check if this is a valid OpenAI function tool
            if not isinstance(tool, dict):
                continue

            if tool.get("type") != "function":
                continue

            if "function" not in tool:
                continue

            func_def = tool["function"]

            # Extract required fields for Gemini
            if "name" not in func_def:
                continue

            # Build Gemini function declaration
            gemini_func = {
                "name": func_def["name"],
                "description": func_def.get("description", ""),
            }

            # Add parameters if present
            if "parameters" in func_def:
                gemini_func["parameters"] = func_def["parameters"]

            function_declarations.append(gemini_func)

        if not function_declarations:
            return None

        # Create Gemini tools format
        gemini_tools = [{
            "function_declarations": function_declarations
        }]

        return gemini_tools

    def _preprocess_messages_for_gemini(self, messages: list) -> list:
        """
        Preprocess messages to ensure compatibility with Gemini's strict message ordering requirements.

        Gemini requires that function calls (assistant messages with tool_calls) must come
        immediately after either:
        - A user message
        - A tool/function response message

        This method reorganizes messages to meet these requirements when switching from other models.
        """
        if not messages:
            return messages

        processed = []
        i = 0

        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")

            # Check if this is a problematic sequence: assistant (without tool_calls) -> user
            if (role == "assistant" and
                "tool_calls" not in msg and
                i + 1 < len(messages) and
                messages[i + 1].get("role") == "user"):

                # Look ahead to see if there's a tool interaction pattern
                # If the previous message was a tool response, we might need to consolidate
                if i > 0 and messages[i - 1].get("role") == "tool":
                    # This is a pattern: tool -> assistant (summary) -> user
                    # We can merge the assistant summary into the next user message
                    next_user = messages[i + 1].copy()
                    assistant_content = msg.get("content", "")

                    # Add context about the previous assistant response
                    if assistant_content:
                        # Prepend the assistant's summary to the user message
                        original_content = next_user.get("content", "")
                        next_user["content"] = f"[Assistant's previous response: {assistant_content}]\n\n{original_content}"

                    # Skip the problematic assistant message
                    i += 1  # Skip assistant
                    processed.append(next_user)
                    i += 1  # Move past user message
                    continue

            # For other messages, add them as-is
            processed.append(msg)
            i += 1

        return processed



    async def chat_completions_create(self, model: str, messages: list, **kwargs) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionResponse, None]]:
        """Create a chat completion (single-turn or streaming) using a Gemini model."""

        replay_request_view = kwargs.pop("_replay_request_view", None)
        replay_mode = kwargs.pop("_replay_mode", None)

        if replay_request_view is not None and replay_mode in {"provider_exact_turn", "degraded_legacy_turn"}:
            messages = replay_request_view
        else:
            # Validate and preprocess replay window before building provider-native history
            replay_build = self.build_replay_view(model, messages, **kwargs)
            messages = replay_build.request_view

        # Determine if streaming
        stream = kwargs.get("stream", False)
        if "stream" in kwargs:
            kwargs.pop("stream")
        # Map model name to proper format
        model_id = model
        # Separate system message (if present) for config
        config_kwargs = {}
        thinking_config_fields = _normalize_thinking_config(kwargs.pop("thinking_config", None))
        # Default thinking_config for 2.5 series models (thought summaries)
        if "2.5" in model_id:  # Heuristic check for 2.5 series models
            if "include_thoughts" not in thinking_config_fields:
                thinking_config_fields["include_thoughts"] = True

        # Reasoning / thinking control for Gemini 3 models
        # Accept OpenAI-style reasoning_effort / reasoning, and direct thinking_level
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        thinking_level = kwargs.pop("thinking_level", None)

        # Also support OpenAI-style `reasoning={"effort": "low"}` for convenience
        reasoning = kwargs.pop("reasoning", None)
        if reasoning is not None and reasoning_effort is None:
            if isinstance(reasoning, dict) and "effort" in reasoning:
                reasoning_effort = reasoning["effort"]
            elif isinstance(reasoning, str):
                reasoning_effort = reasoning

        if self._is_gemini_3_model(model_id):
            # Map to Gemini 3 thinking_level (Flash supports minimal/medium/high; Pro supports low/high)
            level = None
            is_flash = self._is_gemini_3_flash_model(model_id)
            if isinstance(thinking_level, str) and thinking_level:
                level = thinking_level.lower()
            elif isinstance(reasoning_effort, str) and reasoning_effort:
                eff = reasoning_effort.lower()
                if eff in {"minimal", "none", "disable"}:
                    level = "minimal" if is_flash else "low"
                elif eff == "low":
                    level = "low"
                elif eff == "medium":
                    level = "medium" if is_flash else "high"
                elif eff == "high":
                    level = "high"

            if level and not is_flash:
                if level == "minimal":
                    level = "low"
                elif level == "medium":
                    level = "high"

            thinking_requested = bool(thinking_config_fields) or bool(level) or isinstance(reasoning_effort, str)
            if thinking_requested:
                if level and "thinking_level" not in thinking_config_fields:
                    thinking_config_fields["thinking_level"] = level
                if "include_thoughts" not in thinking_config_fields:
                    thinking_config_fields["include_thoughts"] = True

            # For Gemini 3, default temperature to 1.0 if not explicitly set
            if "temperature" not in kwargs and "temperature" not in config_kwargs:
                config_kwargs["temperature"] = 1.0

        if thinking_config_fields:
            filtered_thinking_config = _filter_thinking_config_fields(thinking_config_fields)
            if filtered_thinking_config:
                try:
                    config_kwargs["thinking_config"] = types.ThinkingConfig(**filtered_thinking_config)
                except Exception:
                    config_kwargs["thinking_config"] = filtered_thinking_config

        if messages and messages[0]['role'] == "system":
            config_kwargs["system_instruction"] = messages[0]['content']
            messages = messages[1:]
        # Map max_tokens to max_output_tokens for Google SDK
        if "max_tokens" in kwargs or "max_output_tokens" in kwargs:
            max_toks = kwargs.pop("max_output_tokens", None) or kwargs.pop("max_tokens", None)
            config_kwargs["max_output_tokens"] = max_toks
        # Pass through other known generation parameters
        for param in ["temperature", "top_p", "top_k", "candidate_count",
                      "presence_penalty", "frequency_penalty", "seed"]:
            if param in kwargs:
                config_kwargs[param] = kwargs.pop(param)
        # Handle stop sequences (stop or stop_sequences key)
        stop_seq = None
        if "stop_sequences" in kwargs or "stop" in kwargs:
            stop_seq = kwargs.pop("stop_sequences", None) or kwargs.pop("stop", None)
        if stop_seq:
            # Ensure stop sequences is a list
            if isinstance(stop_seq, str):
                config_kwargs["stop_sequences"] = [stop_seq]
            elif isinstance(stop_seq, list):
                config_kwargs["stop_sequences"] = stop_seq
        # Handle tools parameter - convert OpenAI format to Gemini format
        tool_names_for_log: list[str] = []
        if "tools" in kwargs:
            openai_tools = kwargs.pop("tools")
            gemini_tools = self._convert_tool_spec(openai_tools)
            if gemini_tools:
                config_kwargs["tools"] = gemini_tools
            if isinstance(openai_tools, list):
                for tool in openai_tools:
                    if isinstance(tool, dict) and tool.get("type") == "function":
                        func = tool.get("function") or {}
                        name = func.get("name")
                        if isinstance(name, str) and name:
                            tool_names_for_log.append(name)

        # (Ignore any remaining kwargs that are not applicable for now)
        # Create config object if any config parameters were specified
        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        # Prepare conversation history (all messages except the final prompt)
        history_msgs = []
        last_user_message = None
        if messages:
            # Handle different conversation scenarios
            if messages[-1]["role"] == "user":
                # Standard case: last message is from user
                last_user_message = messages[-1]["content"]
                convo_history = messages[:-1]
            elif messages[-1]["role"] in ["tool", "assistant"]:
                # Agent scenario: last message is tool result or assistant message
                # According to Gemini API docs, we should include all messages as history
                # and let the model continue naturally without adding "continue"
                convo_history = messages
                last_user_message = None  # No additional user message needed
            else:
                # Other cases: treat all as history with empty continuation
                convo_history = messages
                last_user_message = ""  # Empty message to continue conversation
            # Convert history messages to Content objects
            for msg in convo_history:
                role = msg["role"]

                # Handle different message types
                if role == "system":
                    # System messages are already handled separately
                    continue
                elif role == "tool":
                    # Tool messages should use function response format for Gemini
                    # Create proper function response part
                    if "tool_call_id" in msg:
                        replay_parts, diagnostic = self._build_tool_response_parts(
                            model_id,
                            msg,
                            replay_messages=convo_history,
                        )
                        if diagnostic:
                            raise LLMError(diagnostic.message)
                        if replay_parts:
                            history_msgs.append(types.Content(role="user", parts=replay_parts))
                    else:
                        # Fallback to text format if no tool_call_id
                        tool_content = f"Tool result: {msg['content']}"
                        part = types.Part.from_text(text=tool_content)
                        history_msgs.append(types.Content(role="user", parts=[part]))
                elif role in ("user", "assistant"):
                    # Map AISuite role to Gemini role (Gemini expects "user" or "model")
                    gemini_role = "model" if role == "assistant" else "user"

                    # Handle tool calls in assistant messages
                    if role == "assistant" and isinstance(msg.get("tool_calls"), list):
                        # Assistant message with tool calls - convert to function call parts
                        parts = self._convert_assistant_tool_message_to_parts(model_id, msg)
                        if parts:
                            history_msgs.append(types.Content(role=gemini_role, parts=parts))
                    else:
                        # Regular text message
                        if msg.get("content"):
                            part = types.Part.from_text(text=msg["content"])
                            history_msgs.append(types.Content(role=gemini_role, parts=[part]))
                else:
                    # Skip unknown message types
                    continue



        # Create a new chat session with history and config (if any)
        chat = self.client.chats.create(model=model_id, config=config, history=history_msgs if history_msgs else None)

        # Handle case where we don't need to send a new user message
        if last_user_message is None:
            # For agent scenarios where the last message is a tool result or assistant message,
            # we can use the model's generate_content method directly with the full conversation
            if stream:
                # Reset streaming state
                self._streaming_tool_calls = {}
                self._stream_content_length = 0
                self._stream_tool_calls_count = 0

                # For streaming, we need to handle this differently
                # Use the client's models.generate_content_stream method
                contents = []
                for msg in messages:
                    role = msg["role"]
                    if role == "system":
                        continue  # Already handled in config
                    elif role == "tool":
                        # Use function response format
                        if "tool_call_id" in msg:
                            replay_parts, diagnostic = self._build_tool_response_parts(
                                model_id,
                                msg,
                                replay_messages=messages,
                            )
                            if diagnostic:
                                raise LLMError(diagnostic.message)
                            if replay_parts:
                                contents.append(types.Content(role="user", parts=replay_parts))
                        else:
                            tool_content = f"Tool result: {msg['content']}"
                            part = types.Part.from_text(text=tool_content)
                            contents.append(types.Content(role="user", parts=[part]))
                    elif role in ("user", "assistant"):
                        gemini_role = "model" if role == "assistant" else "user"

                        if role == "assistant" and isinstance(msg.get("tool_calls"), list):
                            parts = self._convert_assistant_tool_message_to_parts(model_id, msg)
                            if parts:
                                contents.append(types.Content(role=gemini_role, parts=parts))
                        else:
                            if msg.get("content"):
                                part = types.Part.from_text(text=msg["content"])
                                contents.append(types.Content(role=gemini_role, parts=[part]))





                # Use streaming generation
                async def stream_generator():
                    response_id = None
                    stream_usage = None
                    pending_response = None
                    for chunk in self.client.models.generate_content_stream(
                        model=model_id,
                        contents=contents,
                        config=config
                    ):
                        if response_id is None:
                            potential_id = getattr(chunk, 'response_id', None)
                            if potential_id is not None:
                                response_id = potential_id

                        current_chunk_text = ""
                        current_chunk_reasoning = None
                        reasoning_text_parts = []
                        content_text_parts = []
                        tool_calls = None

                        candidate = chunk.candidates[0] if chunk.candidates else None
                        candidate_content = getattr(candidate, "content", None)
                        candidate_parts = getattr(candidate_content, "parts", None)

                        if candidate_parts:
                            for part in candidate_parts:
                                if getattr(part, 'thought', False) and getattr(part, 'text', None):
                                    reasoning_text_parts.append(part.text)
                                elif hasattr(part, 'function_call') and part.function_call:
                                    function_call_obj = part.function_call
                                    signature = _extract_thought_signature(part) or _extract_thought_signature(function_call_obj)
                                    if signature and function_call_obj:
                                        self._attach_thought_signature(function_call_obj, signature)
                                    mock_delta = type('MockDelta', (), {
                                        'function_call': function_call_obj
                                    })()
                                    tool_calls = self._accumulate_and_convert_tool_calls(mock_delta)
                                elif getattr(part, 'text', None):
                                    content_text_parts.append(part.text)

                        if reasoning_text_parts:
                            current_chunk_reasoning = "".join(reasoning_text_parts)

                        if content_text_parts:
                            current_chunk_text = "".join(content_text_parts)
                            # Accumulate content length for accurate stop_info metadata
                            self._stream_content_length += len(current_chunk_text)

                        # Accumulate tool calls count
                        if tool_calls:
                            self._stream_tool_calls_count += len(tool_calls)

                        # Capture usage metadata from chunk if available
                        usage_metadata = getattr(chunk, "usage_metadata", None) or getattr(chunk, "usageMetadata", None)
                        if usage_metadata:
                            normalized_usage = _normalize_gemini_usage(usage_metadata)
                            if normalized_usage:
                                stream_usage = normalized_usage

                        # Check for finish_reason in chunk
                        finish_reason = None
                        stop_info = None
                        if chunk.candidates and chunk.candidates[0].finish_reason:
                            finish_reason = _normalize_finish_reason(chunk.candidates[0].finish_reason)
                            # Use accumulated values for accurate stop_info metadata
                            stop_metadata = {
                                "has_content": self._stream_content_length > 0 or self._stream_tool_calls_count > 0,
                                "content_length": self._stream_content_length,
                                "tool_calls_count": self._stream_tool_calls_count,
                                "finish_reason": finish_reason,
                                "model": model_id,
                                "provider": "gemini"
                            }
                            if finish_reason in _TOOL_CALL_ERROR_FINISH_REASONS | _PROTOCOL_ERROR_FINISH_REASONS:
                                try:
                                    roles_tail = [getattr(c, "role", None) for c in contents][-12:]
                                except Exception:
                                    roles_tail = []
                                logger.warning(
                                    "Gemini abnormal finish_reason=%s model=%s tools=%s roles_tail=%s",
                                    finish_reason,
                                    model_id,
                                    tool_names_for_log,
                                    roles_tail,
                                )
                            stop_info = stop_reason_manager.map_stop_reason("gemini", finish_reason or "", stop_metadata)

                        response_metadata = {
                            'id': response_id,
                            'created': None,
                            'model': model_id
                        }

                        current_response = ChatCompletionResponse(
                            choices=[
                                StreamChoice(
                                    index=0,
                                    delta=ChoiceDelta(
                                        content=current_chunk_text if current_chunk_text else None,
                                        role="assistant" if current_chunk_text or tool_calls else None,
                                        tool_calls=tool_calls,
                                        reasoning_content=current_chunk_reasoning
                                    ),
                                    finish_reason=finish_reason,
                                    stop_info=stop_info
                                )
                            ],
                            metadata=response_metadata
                        )

                        if pending_response is not None:
                            yield pending_response

                        pending_response = current_response

                    if pending_response is not None:
                        if stream_usage:
                            pending_response.metadata["usage"] = stream_usage
                        yield pending_response
                return stream_generator()
            else:
                # For non-streaming, use generate_content directly
                contents = []
                for msg in messages:
                    role = msg["role"]
                    if role == "system":
                        continue  # Already handled in config
                    elif role == "tool":
                        if "tool_call_id" in msg:
                            replay_parts, diagnostic = self._build_tool_response_parts(
                                model_id,
                                msg,
                                replay_messages=messages,
                            )
                            if diagnostic:
                                raise LLMError(diagnostic.message)
                            if replay_parts:
                                contents.append(types.Content(role="user", parts=replay_parts))
                        else:
                            tool_content = f"Tool result: {msg['content']}"
                            part = types.Part.from_text(text=tool_content)
                            contents.append(types.Content(role="user", parts=[part]))
                    elif role in ("user", "assistant"):
                        gemini_role = "model" if role == "assistant" else "user"

                        if role == "assistant" and isinstance(msg.get("tool_calls"), list):
                            parts = self._convert_assistant_tool_message_to_parts(model_id, msg)
                            if parts:
                                contents.append(types.Content(role=gemini_role, parts=parts))
                        else:
                            if msg.get("content"):
                                part = types.Part.from_text(text=msg["content"])
                                contents.append(types.Content(role=gemini_role, parts=[part]))





                response = self.client.models.generate_content(
                    model=model_id,
                    contents=contents,
                    config=config
                )
                return GeminiMessageConverter.from_gemini_response(response)

        # Send the last user message and get response (streaming or full)
        if stream:
            # Reset streaming state
            self._streaming_tool_calls = {}
            self._stream_content_length = 0
            self._stream_tool_calls_count = 0

            # Streaming response: return a generator yielding ChatCompletionResponse objects
            async def stream_generator():
                response_id = None  # We'll use the first valid chunk's id for all chunks
                stream_usage = None
                pending_response = None
                for chunk in chat.send_message_stream(last_user_message):
                    if response_id is None:
                        potential_id = getattr(chunk, 'response_id', None)
                        if potential_id is not None:
                            response_id = potential_id

                    current_chunk_text = ""
                    current_chunk_reasoning = None
                    reasoning_text_parts = []
                    content_text_parts = []
                    tool_calls = None

                    candidate = chunk.candidates[0] if chunk.candidates else None
                    candidate_content = getattr(candidate, "content", None)
                    candidate_parts = getattr(candidate_content, "parts", None)

                    if candidate_parts: # Ensure candidates and parts exist
                        for part in candidate_parts:
                            # Check if the part is a thought and has text
                            if getattr(part, 'thought', False) and getattr(part, 'text', None):
                                reasoning_text_parts.append(part.text)
                            # Check if the part is a function call
                            elif hasattr(part, 'function_call') and part.function_call:
                                # Create a mock delta object for consistency with other providers
                                function_call_obj = part.function_call
                                signature = _extract_thought_signature(part) or _extract_thought_signature(function_call_obj)
                                if signature and function_call_obj:
                                    self._attach_thought_signature(function_call_obj, signature)
                                mock_delta = type('MockDelta', (), {
                                    'function_call': function_call_obj
                                })()
                                tool_calls = self._accumulate_and_convert_tool_calls(mock_delta)
                            # Else, if it's not a thought but has text, it's regular content
                            elif getattr(part, 'text', None):
                                content_text_parts.append(part.text)

                    if reasoning_text_parts:
                        current_chunk_reasoning = "".join(reasoning_text_parts)

                    if content_text_parts:
                        current_chunk_text = "".join(content_text_parts)
                        # Accumulate content length for accurate stop_info metadata
                        self._stream_content_length += len(current_chunk_text)

                    # Accumulate tool calls count
                    if tool_calls:
                        self._stream_tool_calls_count += len(tool_calls)

                    # Capture usage metadata from chunk if available
                    usage_metadata = getattr(chunk, "usage_metadata", None) or getattr(chunk, "usageMetadata", None)
                    if usage_metadata:
                        normalized_usage = _normalize_gemini_usage(usage_metadata)
                        if normalized_usage:
                            stream_usage = normalized_usage

                    # Check for finish_reason in chunk
                    finish_reason = None
                    stop_info = None
                    if chunk.candidates and chunk.candidates[0].finish_reason:
                        finish_reason = _normalize_finish_reason(chunk.candidates[0].finish_reason)
                        # Use accumulated values for accurate stop_info metadata
                        stop_metadata = {
                            "has_content": self._stream_content_length > 0 or self._stream_tool_calls_count > 0,
                            "content_length": self._stream_content_length,
                            "tool_calls_count": self._stream_tool_calls_count,
                            "finish_reason": finish_reason,
                            "model": model_id,
                            "provider": "gemini"
                        }
                        if finish_reason in _TOOL_CALL_ERROR_FINISH_REASONS | _PROTOCOL_ERROR_FINISH_REASONS:
                            try:
                                roles_tail = [getattr(c, "role", None) for c in (history_msgs or [])][-12:]
                            except Exception:
                                roles_tail = []
                            logger.warning(
                                "Gemini abnormal finish_reason=%s model=%s tools=%s last_user_message_len=%s roles_tail=%s",
                                finish_reason,
                                model_id,
                                tool_names_for_log,
                                len(last_user_message or ""),
                                roles_tail,
                            )
                        stop_info = stop_reason_manager.map_stop_reason("gemini", finish_reason or "", stop_metadata)

                    response_metadata = {
                        'id': response_id, # Use the captured ID (or None if never found)
                        'created': None,  # Gemini doesn't provide timestamp
                        'model': model_id
                    }

                    current_response = ChatCompletionResponse(
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=ChoiceDelta(
                                    content=current_chunk_text if current_chunk_text else None,
                                    role="assistant" if current_chunk_text or tool_calls else None,
                                    tool_calls=tool_calls,
                                    reasoning_content=current_chunk_reasoning
                                ),
                                finish_reason=finish_reason,
                                stop_info=stop_info
                            )
                        ],
                        metadata=response_metadata
                    )

                    if pending_response is not None:
                        yield pending_response

                    pending_response = current_response

                if pending_response is not None:
                    if stream_usage:
                        pending_response.metadata["usage"] = stream_usage
                    yield pending_response
            return stream_generator()
        else:
            # Single-turn completion: get the full response
            response = chat.send_message(message=last_user_message)
            return GeminiMessageConverter.from_gemini_response(response)  # The response object with .text, .candidates, etc.



    def _accumulate_and_convert_tool_calls(self, delta):
        """
        Accumulate tool call chunks and convert to unified format when complete.

        For Gemini, function calls are typically complete in a single chunk,
        but we maintain the same interface as other providers for consistency.

        Args:
            delta: The delta object from streaming response (mock object for Gemini)

        Returns:
            List of converted tool calls if any are complete, None otherwise
        """
        # Check if delta has function_call (Gemini format)
        if hasattr(delta, 'function_call') and delta.function_call:
            return self._process_gemini_function_call_direct(delta.function_call)

        # Check if delta has tool_calls (standard format, for future compatibility)
        if not hasattr(delta, 'tool_calls') or not delta.tool_calls:
            return None

        # Standard tool_calls processing (for future Gemini API changes)
        return self._process_standard_tool_calls(delta.tool_calls)

    def _process_gemini_function_call_direct(self, function_call):
        """
        Process Gemini function call directly (current Gemini format).

        Args:
            function_call: Gemini function call object

        Returns:
            List of converted tool calls if complete, None otherwise
        """
        if not function_call:
            return None

        try:
            provider_call_id = _extract_provider_call_id(function_call)
            signature = _extract_thought_signature(function_call)
            # Gemini function calls are typically complete in a single chunk
            function = Function(
                name=function_call.name,
                arguments=json.dumps(function_call.args) if hasattr(function_call, 'args') else "{}"
            )

            tool_call_obj = ChatCompletionMessageToolCall(
                id=provider_call_id or f"call_{function_call.name}_{hash(str(function_call.args)) % 10000}",
                function=function,
                type="function",
                extra_content=_build_tool_call_extra(
                    signature,
                    provider_call_id=provider_call_id,
                    provider_function_name=function_call.name,
                )
            )

            return [tool_call_obj]

        except Exception:
            logger.exception("Failed to process Gemini function_call (name=%s)", getattr(function_call, "name", None))
            return None

    def _process_standard_tool_calls(self, tool_calls):
        """
        Process standard tool_calls format (for future compatibility).

        This method handles the standard streaming tool_calls format,
        similar to OpenAI and other providers.

        Args:
            tool_calls: List of tool call delta objects

        Returns:
            List of converted tool calls if any are complete, None otherwise
        """
        if not tool_calls:
            return None

        # Accumulate tool call chunks (similar to other providers)
        for tool_call_delta in tool_calls:
            index = getattr(tool_call_delta, 'index', 0)

            # Initialize tool call accumulator if not exists
            if index not in self._streaming_tool_calls:
                self._streaming_tool_calls[index] = {
                    "id": "",
                    "type": "function",
                    "function": {
                        "name": "",
                        "arguments": ""
                    },
                    "extra_content": None
                }

            tool_call = self._streaming_tool_calls[index]

            # Accumulate id
            if hasattr(tool_call_delta, 'id') and tool_call_delta.id:
                tool_call["id"] += tool_call_delta.id

            # Accumulate function data
            if hasattr(tool_call_delta, 'function') and tool_call_delta.function:
                signature = _extract_thought_signature(tool_call_delta.function)
                provider_call_id = tool_call["id"] or _extract_provider_call_id(tool_call_delta.function)
                provider_function_name = getattr(tool_call_delta.function, 'name', None) or tool_call["function"]["name"] or None
                if signature or provider_call_id or provider_function_name:
                    tool_call["extra_content"] = _build_tool_call_extra(
                        signature,
                        provider_call_id=provider_call_id,
                        provider_function_name=provider_function_name,
                    )

                if hasattr(tool_call_delta.function, 'name') and tool_call_delta.function.name:
                    tool_call["function"]["name"] += tool_call_delta.function.name

                if hasattr(tool_call_delta.function, 'arguments') and tool_call_delta.function.arguments:
                    tool_call["function"]["arguments"] += tool_call_delta.function.arguments

            # Set type if provided
            if hasattr(tool_call_delta, 'type') and tool_call_delta.type:
                tool_call["type"] = tool_call_delta.type

            # Some SDKs may attach signature at the delta level
            if not tool_call.get("extra_content"):
                signature = _extract_thought_signature(tool_call_delta)
                provider_call_id = tool_call["id"] or _extract_provider_call_id(tool_call_delta)
                provider_function_name = tool_call["function"]["name"] or None
                if signature or provider_call_id or provider_function_name:
                    tool_call["extra_content"] = _build_tool_call_extra(
                        signature,
                        provider_call_id=provider_call_id,
                        provider_function_name=provider_function_name,
                    )

        # Check for complete tool calls and convert them
        complete_tool_calls = []
        for index, tool_call_data in list(self._streaming_tool_calls.items()):
            if (tool_call_data["id"] and
                tool_call_data["function"]["name"] and
                tool_call_data["function"]["arguments"]):

                try:
                    # Try to parse arguments as JSON to ensure completeness
                    json.loads(tool_call_data["function"]["arguments"])

                    # Convert to framework format
                    function = Function(
                        name=tool_call_data["function"]["name"],
                        arguments=tool_call_data["function"]["arguments"]
                    )

                    tool_call_obj = ChatCompletionMessageToolCall(
                        id=tool_call_data["id"],
                        function=function,
                        type="function",
                        extra_content=tool_call_data.get("extra_content")
                    )

                    complete_tool_calls.append(tool_call_obj)

                    # Remove completed tool call from accumulator
                    del self._streaming_tool_calls[index]

                except json.JSONDecodeError:
                    # Arguments are not complete yet, continue accumulating
                    continue

        return complete_tool_calls if complete_tool_calls else None
