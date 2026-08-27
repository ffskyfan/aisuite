import copy
import os
from typing import Any, Dict, List, Optional

from aisuite.framework.content import (
    MultimodalCapabilities,
    content_text,
    has_image_content,
    is_image_part,
)
from aisuite.framework.replay_payload import (
    ReplayBuildResult,
    ReplayDiagnostic,
    ReplayValidationResult,
)
from aisuite.provider import LLMError
from aisuite.providers.deepseek_provider import DeepseekProvider


class QwenProvider(DeepseekProvider):
    """Qwen provider using Alibaba Cloud Model Studio's OpenAI-compatible API."""

    PROVIDER_NAME = "qwen"
    TOOL_IMAGE_FORWARD_TEXT = "[visual tool result attached in following user message]"

    TOKEN_PLAN_BASE_URL = (
        "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    )
    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    _OPENAI_EXTRA_BODY_FIELDS = (
        "enable_search",
        "enable_thinking",
        "preserve_thinking",
        "search_options",
        "thinking_budget",
        "vl_high_resolution_images",
    )

    def __init__(self, **config):
        token_plan_api_key = os.getenv("BAILIAN_TOKEN_PLAN_API_KEY")
        api_key = (
            config.get("api_key")
            or os.getenv("QWEN_API_KEY")
            or token_plan_api_key
            or os.getenv("DASHSCOPE_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "Qwen API key is missing. Provide api_key or set QWEN_API_KEY, "
                "DASHSCOPE_API_KEY, or BAILIAN_TOKEN_PLAN_API_KEY."
            )

        uses_token_plan = str(api_key).startswith("sk-sp-") or (
            token_plan_api_key is not None and api_key == token_plan_api_key
        )
        default_base_url = (
            self.TOKEN_PLAN_BASE_URL if uses_token_plan else self.DASHSCOPE_BASE_URL
        )

        client_config = dict(config)
        client_config["api_key"] = api_key
        client_config["base_url"] = (
            client_config.get("base_url")
            or os.getenv("QWEN_BASE_URL")
            or os.getenv("BAILIAN_TOKEN_PLAN_BASE_URL")
            or os.getenv("DASHSCOPE_BASE_URL")
            or default_base_url
        )
        super().__init__(**client_config)

    def get_multimodal_capabilities(
        self, model: str | None = None
    ) -> MultimodalCapabilities:
        normalized_model = (model or "").lower()
        vision_models = (
            "qwen3.8-max",
            "qwen3.7-plus",
            "qwen3.6-plus",
            "qwen3.7-flash",
            "qwen3.6-flash",
            "qwen3.5-plus",
            "qwen3.5-flash",
            "qwen3-vl",
            "qwen-vl",
        )
        user_images = (
            "supported"
            if any(marker in normalized_model for marker in vision_models)
            else "unknown"
        )

        # Alibaba Cloud requires role=tool content to be a string. The adapter
        # preserves canonical tool-result images by projecting them into a
        # following user vision message before the wire request is sent.
        return MultimodalCapabilities(
            user_images=user_images,
            tool_result_images=user_images,
        )

    def _project_tool_result_images(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Keep Qwen tool messages string-only without discarding screenshots."""

        projected: List[Dict[str, Any]] = []
        pending_visual_parts: List[Dict[str, Any]] = []

        def flush_visual_parts() -> None:
            if not pending_visual_parts:
                return
            projected.append(
                {
                    "role": "user",
                    "content": copy.deepcopy(pending_visual_parts),
                }
            )
            pending_visual_parts.clear()

        for raw_message in messages:
            message = copy.deepcopy(self._normalize_message(raw_message))
            if message.get("role") != "tool":
                flush_visual_parts()
                projected.append(message)
                continue

            content = message.get("content")
            if not has_image_content(content):
                projected.append(message)
                continue

            tool_call_id = str(message.get("tool_call_id") or "unknown")
            text = content_text(content).strip()
            message["content"] = text or self.TOOL_IMAGE_FORWARD_TEXT
            projected.append(message)

            pending_visual_parts.append(
                {
                    "type": "text",
                    "text": (
                        "The following image is visual output from tool result "
                        f"{tool_call_id}. Inspect it as part of that tool result."
                    ),
                }
            )
            pending_visual_parts.extend(
                copy.deepcopy(part) for part in content if is_image_part(part)
            )

        flush_visual_parts()
        return projected

    def _build_reasoning_replay_payload(self, reasoning_content: str) -> None:
        # Qwen replays the original thinking text directly; no raw copy is needed.
        return None

    def _extract_reasoning_input(self, reasoning_content: Any) -> Optional[str]:
        """Read the sole canonical text verbatim, without raw-payload fallbacks."""
        if isinstance(reasoning_content, str):
            thinking = reasoning_content
        elif isinstance(reasoning_content, dict):
            thinking = reasoning_content.get("thinking")
        else:
            thinking = getattr(reasoning_content, "thinking", None)
        return thinking if isinstance(thinking, str) and thinking else None

    def _has_reasoning_input(self, reasoning_content: Any) -> bool:
        value = self._extract_reasoning_input(reasoning_content)
        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def _normalize_message(message: Any) -> Dict[str, Any]:
        if isinstance(message, dict):
            return message
        if hasattr(message, "model_dump"):
            return message.model_dump()
        return {}

    @staticmethod
    def _extra_body(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        extra_body = kwargs.get("extra_body")
        return extra_body if isinstance(extra_body, dict) else {}

    def _is_qwen38_model(self, model: str) -> bool:
        return (model or "").lower().startswith("qwen3.8-max")

    def _preserve_thinking_enabled(self, model: str, kwargs: Dict[str, Any]) -> bool:
        direct_value = kwargs.get("preserve_thinking")
        if direct_value is not None:
            return bool(direct_value)
        extra_value = self._extra_body(kwargs).get("preserve_thinking")
        if extra_value is not None:
            return bool(extra_value)
        # qwen3.8-max defaults preserve_thinking to true in the official API.
        return self._is_qwen38_model(model)

    def _is_thinking_enabled_for_model(
        self, model: str, kwargs: Dict[str, Any]
    ) -> bool:
        direct_value = kwargs.get("enable_thinking")
        if direct_value is not None:
            return bool(direct_value)
        extra_value = self._extra_body(kwargs).get("enable_thinking")
        if extra_value is not None:
            return bool(extra_value)

        thinking = kwargs.get("thinking")
        if isinstance(thinking, dict):
            thinking_type = thinking.get("type")
            if thinking_type == "disabled":
                return False
            if thinking_type in {"enabled", "adaptive"}:
                return True
        elif thinking is not None:
            return bool(thinking)

        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort is None:
            reasoning = kwargs.get("reasoning")
            reasoning_effort = (
                reasoning.get("effort") if isinstance(reasoning, dict) else reasoning
            )
        if isinstance(reasoning_effort, str):
            return reasoning_effort.lower() not in {"none", "off"}
        return self._is_qwen38_model(model)

    def _prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        projected_messages = self._project_tool_result_images(messages)
        prepared_messages = super()._prepare_messages(projected_messages)
        for message in prepared_messages:
            # is_error is canonical AISuite metadata, not part of Qwen's wire schema.
            message.pop("is_error", None)
        return prepared_messages

    def validate_replay_window(
        self, model: str, messages: list, **kwargs
    ) -> ReplayValidationResult:
        diagnostics: List[ReplayDiagnostic] = []
        normalized_messages = [self._normalize_message(message) for message in messages]
        preserve_thinking = self._preserve_thinking_enabled(model, kwargs)
        thinking_enabled = self._is_thinking_enabled_for_model(model, kwargs)

        expected_tool_call_ids = set()
        observed_tool_result_ids = set()
        for message in normalized_messages:
            role = message.get("role")
            if role == "assistant" and message.get("tool_calls"):
                for tool_call in message.get("tool_calls") or []:
                    tool_call_data = self._normalize_message(tool_call)
                    function = tool_call_data.get("function") or {}
                    if not isinstance(function, dict) and hasattr(
                        function, "model_dump"
                    ):
                        function = function.model_dump()
                    tool_call_id = tool_call_data.get("id")
                    if not tool_call_id:
                        diagnostics.append(
                            ReplayDiagnostic(
                                code="missing_tool_call_id",
                                message="Qwen assistant tool call is missing id.",
                                provider=self.PROVIDER_NAME,
                            )
                        )
                    else:
                        expected_tool_call_ids.add(tool_call_id)
                    if not function.get("name"):
                        diagnostics.append(
                            ReplayDiagnostic(
                                code="missing_tool_function_name",
                                message="Qwen assistant tool call is missing function name.",
                                provider=self.PROVIDER_NAME,
                                metadata={"tool_call_id": tool_call_id},
                            )
                        )
                    if function.get("arguments") is None:
                        diagnostics.append(
                            ReplayDiagnostic(
                                code="missing_tool_arguments",
                                message="Qwen assistant tool call is missing arguments.",
                                provider=self.PROVIDER_NAME,
                                metadata={"tool_call_id": tool_call_id},
                            )
                        )

                if (
                    preserve_thinking
                    and thinking_enabled
                    and not self._has_reasoning_input(message.get("reasoning_content"))
                ):
                    diagnostics.append(
                        ReplayDiagnostic(
                            code="missing_reasoning_content",
                            message=(
                                "Qwen preserve_thinking is enabled, but an assistant "
                                "tool-call message has no replayable reasoning_content."
                            ),
                            severity="warning",
                            provider=self.PROVIDER_NAME,
                            metadata={"role": "assistant"},
                        )
                    )

            if role == "tool":
                tool_call_id = message.get("tool_call_id")
                if not tool_call_id:
                    diagnostics.append(
                        ReplayDiagnostic(
                            code="missing_tool_call_id",
                            message="Qwen tool result replay requires tool_call_id.",
                            provider=self.PROVIDER_NAME,
                            metadata={"role": "tool"},
                        )
                    )
                else:
                    observed_tool_result_ids.add(tool_call_id)

            if (
                role == "assistant"
                and message.get("reasoning_content") is not None
                and not self._has_reasoning_input(message.get("reasoning_content"))
            ):
                diagnostics.append(
                    ReplayDiagnostic(
                        code="missing_reasoning_raw_replay",
                        message=(
                            "Qwen preserved thinking requires the original thinking text."
                        ),
                        severity="warning",
                        provider=self.PROVIDER_NAME,
                        metadata={"role": "assistant"},
                    )
                )

        for tool_call_id in observed_tool_result_ids - expected_tool_call_ids:
            diagnostics.append(
                ReplayDiagnostic(
                    code="orphan_tool_result",
                    message="Qwen tool result has no matching assistant tool call.",
                    provider=self.PROVIDER_NAME,
                    metadata={"tool_call_id": tool_call_id},
                )
            )

        return ReplayValidationResult(
            ok=not any(item.severity == "error" for item in diagnostics),
            degraded=any(item.severity == "warning" for item in diagnostics),
            diagnostics=tuple(diagnostics),
        )

    def build_replay_view(self, model: str, messages: list, **kwargs):
        validation = self.validate_replay_window(model, messages, **kwargs)
        if not validation.ok:
            error_codes = ", ".join(
                item.code for item in validation.diagnostics if item.severity == "error"
            )
            raise LLMError(f"Qwen replay window validation failed: {error_codes}")
        return ReplayBuildResult(
            request_view=self._prepare_messages(messages),
            replay_mode="canonical_with_reasoning",
            degraded=validation.degraded,
            diagnostics=validation.diagnostics,
        )

    @staticmethod
    def _normalize_reasoning_effort(reasoning_effort: Any) -> Optional[str]:
        if reasoning_effort is None:
            return None
        value = str(reasoning_effort).lower()
        aliases = {
            "minimal": "low",
            "high": "xhigh",
            "max": "xhigh",
        }
        value = aliases.get(value, value)
        if value in {"none", "off"}:
            return "none"
        if value not in {"low", "medium", "xhigh"}:
            raise ValueError(
                "Qwen reasoning_effort must be one of low, medium, xhigh, "
                "or a documented alias (minimal, high, max, none)."
            )
        return value

    def _prepare_request_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        prepared = dict(kwargs)
        extra_body = dict(prepared.pop("extra_body", None) or {})

        for field in self._OPENAI_EXTRA_BODY_FIELDS:
            if field in prepared:
                extra_body[field] = prepared.pop(field)

        thinking = prepared.pop("thinking", None)
        if thinking is not None and "enable_thinking" not in extra_body:
            if isinstance(thinking, dict):
                thinking_type = thinking.get("type")
                if thinking_type in {"enabled", "adaptive"}:
                    extra_body["enable_thinking"] = True
                elif thinking_type == "disabled":
                    extra_body["enable_thinking"] = False
                else:
                    raise ValueError(
                        "Qwen thinking.type must be enabled, adaptive, or disabled."
                    )
            else:
                extra_body["enable_thinking"] = bool(thinking)

        reasoning = prepared.pop("reasoning", None)
        if reasoning is not None and "reasoning_effort" not in prepared:
            prepared["reasoning_effort"] = (
                reasoning.get("effort") if isinstance(reasoning, dict) else reasoning
            )

        reasoning_effort = self._normalize_reasoning_effort(
            prepared.get("reasoning_effort")
        )
        if reasoning_effort == "none":
            extra_body["enable_thinking"] = False
            prepared.pop("reasoning_effort", None)
            extra_body.pop("thinking_budget", None)
        elif reasoning_effort is not None:
            prepared["reasoning_effort"] = reasoning_effort

        if extra_body.get("enable_thinking") is False:
            prepared.pop("reasoning_effort", None)
            extra_body.pop("thinking_budget", None)

        if (
            prepared.get("reasoning_effort") is not None
            and extra_body.get("thinking_budget") is not None
        ):
            raise ValueError(
                "Qwen does not allow reasoning_effort and thinking_budget in the "
                "same request."
            )

        if prepared.get("tools") and prepared.get("n") not in {None, 1}:
            raise ValueError("Qwen Function Calling requires n=1.")

        if extra_body:
            prepared["extra_body"] = extra_body
        return prepared

    def _normalize_usage(self, usage_obj):
        normalized = super()._normalize_usage(usage_obj)
        if normalized is None:
            return None

        if hasattr(usage_obj, "model_dump"):
            data = usage_obj.model_dump()
        elif isinstance(usage_obj, dict):
            data = usage_obj
        else:
            data = {}

        def _deep_get(container, *path):
            current = container
            for key in path:
                if current is None:
                    return None
                if isinstance(current, dict):
                    current = current.get(key)
                else:
                    current = getattr(current, key, None)
            return current

        prompt_details = data.get("prompt_tokens_details") or getattr(
            usage_obj, "prompt_tokens_details", None
        )
        completion_details = data.get("completion_tokens_details") or getattr(
            usage_obj, "completion_tokens_details", None
        )

        cache_creation = _deep_get(prompt_details, "cache_creation")
        cache_write_tokens = _deep_get(cache_creation, "cache_creation_input_tokens")
        cache_write_5m = _deep_get(cache_creation, "ephemeral_5m_input_tokens")
        if cache_write_tokens is not None:
            normalized["cache_write_input_tokens"] = max(0, int(cache_write_tokens))
        if cache_write_5m is not None:
            normalized["cache_write_by_ttl"]["ephemeral_5m_input_tokens"] = max(
                0, int(cache_write_5m)
            )

        optional_usage_fields = {
            "reasoning_tokens": _deep_get(completion_details, "reasoning_tokens"),
            "input_text_tokens": _deep_get(prompt_details, "text_tokens"),
            "input_image_tokens": _deep_get(prompt_details, "image_tokens"),
            "input_video_tokens": _deep_get(prompt_details, "video_tokens"),
            "output_text_tokens": _deep_get(completion_details, "text_tokens"),
        }
        for key, value in optional_usage_fields.items():
            if value is not None:
                normalized[key] = max(0, int(value))

        return normalized

    async def chat_completions_create(
        self, model, messages, stream: bool = False, **kwargs
    ):
        if self._is_qwen38_model(model):
            extra_body = kwargs.get("extra_body")
            has_explicit_preserve = "preserve_thinking" in kwargs or (
                isinstance(extra_body, dict) and "preserve_thinking" in extra_body
            )
            if not has_explicit_preserve:
                # Match qwen3.8-max's documented default explicitly so complete
                # reasoning replay remains stable across compatible endpoints.
                kwargs["preserve_thinking"] = True

        return await super().chat_completions_create(
            model=model,
            messages=messages,
            stream=stream,
            **kwargs,
        )
