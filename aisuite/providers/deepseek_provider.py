import copy
import openai
import os
import json
import inspect
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from aisuite.provider import Provider, LLMError
from aisuite.framework.chat_completion_response import (
    ChatCompletionResponse,
    Choice,
    ChoiceDelta,
    StreamChoice,
)
from aisuite.framework.message import (
    Message,
    ChatCompletionMessageToolCall,
    Function,
    ReasoningContent,
)
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
from aisuite.framework.stop_reason import StopInfo, StopReason, stop_reason_manager
from aisuite.framework.content import (
    MultimodalCapabilities,
    content_text,
    has_image_content,
    is_image_part,
)


class DeepseekProvider(Provider):
    PROVIDER_NAME = "deepseek"
    REASONING_REPLAY_KIND = "deepseek_reasoning_text"
    TOOL_IMAGE_FORWARD_TEXT = "[visual tool result attached in following user message]"

    def get_multimodal_capabilities(
        self, model: str | None = None
    ) -> MultimodalCapabilities:
        normalized_model = (model or "").lower().removeprefix("deepseek:")
        # V4.1 Flash is natively multimodal. Both retired Flash API names
        # already route to it; do not infer vision support for unknown models.
        supports_vision = normalized_model in {
            "deepseek-flash",
            "deepseek-v4-flash",
            "deepseek-v4-flash-vision-exp",
        }
        # Chat Completions accepts images only in user messages. Tool images
        # remain supported through the provider-local projection below.
        # https://api-docs.deepseek.com/guides/vision/
        image_support = "supported" if supports_vision else "unsupported"
        return MultimodalCapabilities(
            user_images=image_support,
            tool_result_images=image_support,
        )

    def __init__(self, **config):
        """
        Initialize the DeepSeek provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("DEEPSEEK_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "DeepSeek API key is missing. Please provide it in the config or set the DEEPSEEK_API_KEY environment variable."
            )
        config["base_url"] = (
            config.get("base_url")
            or os.getenv("DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com"
        )

        # NOTE: We could choose to remove above lines for api_key since OpenAI will automatically
        # infer certain values from the environment variables.
        # Eg: OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_PROJECT_ID. Except for OPEN_AI_BASE_URL which has to be the deepseek url

        self._http_client = config.get("http_client")
        self._owns_http_client = self._http_client is None
        if self._http_client is None:
            # Pass an explicit client so the OpenAI SDK does not create an
            # AsyncHttpxClientWrapper that relies on __del__ for cleanup.
            self._http_client = openai.DefaultAsyncHttpxClient()
            config["http_client"] = self._http_client

        # Pass the entire config to the OpenAI client constructor
        self.client = openai.AsyncOpenAI(**config)

        # State for accumulating streaming tool calls
        self._streaming_tool_calls = {}
        self._streaming_reasoning = ""

        # Track accumulated content for streaming responses
        # Used to provide accurate metadata in stop_info
        self._stream_content_length = 0
        self._stream_tool_calls_count = 0

    async def aclose(self):
        try:
            is_closed = getattr(self.client, "is_closed", None)
            if callable(is_closed):
                is_closed = is_closed()
            if not is_closed:
                await self.client.close()
        finally:
            await self._close_owned_http_client()

    async def _close_owned_http_client(self) -> None:
        if not self._owns_http_client or self._http_client is None:
            return

        is_closed = getattr(self._http_client, "is_closed", None)
        if callable(is_closed):
            is_closed = is_closed()
        if is_closed:
            return

        await self._http_client.aclose()

    async def _close_stream_response(self, response) -> None:
        close = getattr(response, "aclose", None) or getattr(response, "close", None)
        if not callable(close):
            return

        result = close()
        if inspect.isawaitable(result):
            await result

    def get_replay_capabilities(
        self, model: str | None = None
    ) -> ProviderReplayCapabilities:
        return ProviderReplayCapabilities(
            needs_exact_turn_replay=False,
            needs_provider_call_id_binding=True,
            needs_reasoning_raw_replay=True,
            supports_canonical_only_history=True,
            empty_actionless_stop_is_retryable=False,
        )

    def _is_thinking_enabled(self, kwargs: Dict[str, Any]) -> bool:
        thinking = kwargs.get("thinking")
        extra_body = kwargs.get("extra_body")
        if thinking is None and isinstance(extra_body, dict):
            thinking = extra_body.get("thinking")
        if isinstance(thinking, dict):
            return thinking.get("type") == "enabled"
        return bool(thinking)

    def validate_replay_window(
        self, model: str, messages: list, **kwargs
    ) -> ReplayValidationResult:
        diagnostics: list[ReplayDiagnostic] = []
        normalized_messages: list[dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, dict):
                normalized_messages.append(msg)
            elif hasattr(msg, "model_dump"):
                normalized_messages.append(msg.model_dump())

        for msg in normalized_messages:
            role = msg.get("role")
            if role == "tool" and not msg.get("tool_call_id"):
                diagnostics.append(
                    ReplayDiagnostic(
                        code="missing_tool_call_id",
                        message="DeepSeek tool result replay requires tool_call_id.",
                        provider="deepseek",
                        metadata={"role": "tool"},
                    )
                )
            if role == "assistant" and msg.get("tool_calls"):
                for tool_call in msg.get("tool_calls", []):
                    function = (
                        tool_call.get("function", {})
                        if isinstance(tool_call, dict)
                        else getattr(tool_call, "function", None)
                    )
                    tool_call_id = (
                        tool_call.get("id")
                        if isinstance(tool_call, dict)
                        else getattr(tool_call, "id", None)
                    )
                    function_name = None
                    if isinstance(function, dict):
                        function_name = function.get("name")
                    elif function is not None:
                        function_name = getattr(function, "name", None)
                    if not tool_call_id:
                        diagnostics.append(
                            ReplayDiagnostic(
                                code="missing_tool_call_id",
                                message="DeepSeek assistant tool call is missing id.",
                                provider="deepseek",
                            )
                        )
                    if not function_name:
                        diagnostics.append(
                            ReplayDiagnostic(
                                code="missing_tool_function_name",
                                message="DeepSeek assistant tool call is missing function name.",
                                provider="deepseek",
                                metadata={"tool_call_id": tool_call_id},
                            )
                        )
        return ReplayValidationResult(
            ok=not any(diag.severity == "error" for diag in diagnostics),
            diagnostics=tuple(diagnostics),
        )

    def capture_response(self, response, model: str | None = None, **kwargs):
        if not response or not getattr(response, "choices", None):
            return ReplayCaptureResult()
        choice = response.choices[0]
        message = getattr(choice, "message", None)
        if not message:
            return ReplayCaptureResult(stop_info=getattr(choice, "stop_info", None))

        replay_metadata = {
            "tool_calls": getattr(message, "tool_calls", None),
            "reasoning_content": None,
        }
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content is not None:
            replay_metadata["reasoning_content"] = getattr(
                reasoning_content, "raw_data", None
            )
        return ReplayCaptureResult(
            canonical_message=message,
            stop_info=getattr(choice, "stop_info", None),
            replay_metadata=replay_metadata,
            protocol_diagnostics=(),
        )

    def _build_reasoning_replay_payload(
        self, reasoning_content: str
    ) -> Optional[Dict[str, Any]]:
        """Return optional provider replay data in addition to canonical thinking."""
        return build_replay_payload(
            "deepseek",
            self.REASONING_REPLAY_KIND,
            {"reasoning_content": reasoning_content},
            legacy_fields={"reasoning_content": reasoning_content},
        )

    def _extract_reasoning_input(self, reasoning_content: Any) -> Optional[str]:
        if reasoning_content is None:
            return None

        if isinstance(reasoning_content, ReasoningContent):
            raw_data = reasoning_content.raw_data or {}
            envelope = get_replay_payload(raw_data)
            if envelope and envelope.get("provider") == "deepseek":
                payload = unwrap_replay_payload(raw_data)
                if isinstance(payload, dict):
                    value = payload.get("reasoning_content")
                    if isinstance(value, str):
                        return value
            if isinstance(raw_data, dict):
                value = raw_data.get("reasoning_content")
                if isinstance(value, str):
                    return value
            return reasoning_content.thinking

        if hasattr(reasoning_content, "thinking"):
            raw_data = getattr(reasoning_content, "raw_data", None) or {}
            envelope = get_replay_payload(raw_data)
            if envelope and envelope.get("provider") == "deepseek":
                payload = unwrap_replay_payload(raw_data)
                if isinstance(payload, dict):
                    value = payload.get("reasoning_content")
                    if isinstance(value, str):
                        return value
            if isinstance(raw_data, dict):
                value = raw_data.get("reasoning_content")
                if isinstance(value, str):
                    return value
            thinking = getattr(reasoning_content, "thinking", None)
            return thinking if isinstance(thinking, str) else None

        if isinstance(reasoning_content, dict):
            raw_data = reasoning_content.get("raw_data") or {}
            envelope = get_replay_payload(raw_data)
            if envelope and envelope.get("provider") == "deepseek":
                payload = unwrap_replay_payload(raw_data)
                if isinstance(payload, dict):
                    value = payload.get("reasoning_content")
                    if isinstance(value, str):
                        return value
            if isinstance(raw_data, dict):
                value = raw_data.get("reasoning_content")
                if isinstance(value, str):
                    return value
            for key in ("reasoning_content", "thinking", "text"):
                value = reasoning_content.get(key)
                if isinstance(value, str):
                    return value
            return None

        return str(reasoning_content)

    def _accumulate_reasoning_content(self, reasoning_text: str) -> None:
        if reasoning_text:
            self._streaming_reasoning += reasoning_text

    def _get_accumulated_thinking(self) -> Dict[str, Any]:
        thinking_text = self._streaming_reasoning
        self._streaming_reasoning = ""
        return {
            "thinking": thinking_text,
            "raw_data": self._build_reasoning_replay_payload(thinking_text),
        }

    def _project_tool_result_images(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Project screenshots for APIs whose tool messages must be text-only.

        Shared by DeepSeek vision and Qwen. Flush only after the entire
        contiguous tool-result group so parallel call/result pairs stay intact.
        Canonical history is never modified or given synthetic user turns.
        """
        projected: List[Dict[str, Any]] = []
        pending_visual_parts: List[Dict[str, Any]] = []

        def flush_visual_parts() -> None:
            if pending_visual_parts:
                projected.append(
                    {"role": "user", "content": list(pending_visual_parts)}
                )
                pending_visual_parts.clear()

        for raw_message in messages:
            message = copy.deepcopy(
                raw_message.model_dump()
                if hasattr(raw_message, "model_dump")
                else dict(raw_message)
            )
            # AISuite-only tool metadata is not part of the wire schema.
            message.pop("is_error", None)
            if message.get("role") != "tool":
                flush_visual_parts()
                projected.append(message)
                continue

            content = message.get("content")
            if not has_image_content(content):
                if isinstance(content, list):
                    message["content"] = content_text(content)
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
            pending_visual_parts.extend(part for part in content if is_image_part(part))

        flush_visual_parts()
        return projected

    def _prepare_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        fill_empty_tool_reasoning: bool = False,
    ) -> List[Dict[str, Any]]:
        prepared_messages = []

        for message in messages:
            prepared = (
                message.model_dump()
                if hasattr(message, "model_dump")
                else dict(message)
            )
            prepared.pop("refusal", None)

            is_thinking_tool_call = bool(
                fill_empty_tool_reasoning
                and prepared.get("role") == "assistant"
                and prepared.get("tool_calls")
            )
            # DeepSeek V4 can emit an explicit empty reasoning_content for a
            # tool-call turn. Keep that state replayable instead of dropping
            # the valid assistant/tool protocol block from history.
            reasoning_content = prepared.get("reasoning_content")
            if reasoning_content is not None:
                extracted_reasoning = self._extract_reasoning_input(reasoning_content)
                if extracted_reasoning is not None:
                    prepared["reasoning_content"] = extracted_reasoning
                elif is_thinking_tool_call:
                    prepared["reasoning_content"] = ""
                else:
                    prepared["reasoning_content"] = None
            elif is_thinking_tool_call:
                prepared["reasoning_content"] = ""

            prepared_messages.append(prepared)

        return prepared_messages

    def build_replay_view(self, model: str, messages: list, **kwargs):
        validation = self.validate_replay_window(model, messages, **kwargs)
        if not validation.ok:
            error_codes = ", ".join(
                diag.code for diag in validation.diagnostics if diag.severity == "error"
            )
            raise LLMError(f"DeepSeek replay window validation failed: {error_codes}")

        prepared_messages = self._prepare_messages(
            messages,
            fill_empty_tool_reasoning=self._is_thinking_enabled(kwargs),
        )
        if self.get_multimodal_capabilities(model).tool_result_images == "supported":
            prepared_messages = self._project_tool_result_images(prepared_messages)

        return ReplayBuildResult(
            request_view=prepared_messages,
            replay_mode="canonical_with_reasoning",
            degraded=validation.degraded,
            diagnostics=validation.diagnostics,
        )

    def _create_stop_info(
        self, finish_reason: str, choice_data: dict = None, model: str = None
    ) -> dict:
        """Create StopInfo from OpenAI-compatible finish_reason."""
        if not finish_reason:
            return None

        # Analyze choice data to determine content presence
        has_content = False
        content_length = 0
        tool_calls_count = 0

        if choice_data:
            stream_content_length = choice_data.get("stream_content_length")
            stream_tool_calls_count = choice_data.get("stream_tool_calls_count")

            # Check message content
            message = choice_data.get("message") or choice_data.get("delta")
            if message:
                content = getattr(message, "content", None) or (
                    message.get("content") if isinstance(message, dict) else None
                )
                if content:
                    has_content = True
                    content_length = len(content)

                # Check tool calls
                tool_calls = getattr(message, "tool_calls", None) or (
                    message.get("tool_calls") if isinstance(message, dict) else None
                )
                if tool_calls:
                    tool_calls_count = len(tool_calls)
                    if tool_calls_count > 0:
                        has_content = True  # Tool calls count as content

            if isinstance(stream_content_length, int) and stream_content_length > 0:
                content_length = max(content_length, stream_content_length)
                has_content = True

            if isinstance(stream_tool_calls_count, int) and stream_tool_calls_count > 0:
                # The terminal chunk can repeat tool calls or contain an empty
                # delta. Use the accumulated count without double-counting the
                # calls visible in the current delta.
                tool_calls_count = max(tool_calls_count, stream_tool_calls_count)
                has_content = True

        metadata = {
            "has_content": has_content,
            "content_length": content_length,
            "tool_calls_count": tool_calls_count,
            "finish_reason": finish_reason,
            "model": model,
            "provider": self.PROVIDER_NAME,
        }

        # Use OpenAI mapping since DeepSeek is OpenAI-compatible, but preserve DeepSeek provider identity
        stop_info = stop_reason_manager.map_stop_reason(
            "openai", finish_reason, metadata
        )
        # Override provider in metadata to correctly identify as DeepSeek
        stop_info.metadata["provider"] = self.PROVIDER_NAME
        return stop_info

    def _normalize_usage(self, usage_obj):
        """Normalize DeepSeek/OpenAI usage objects to a standard dict.

        Supports both Chat Completions usage (prompt_tokens/completion_tokens)
        and possible input/output token naming variants.
        """
        if not usage_obj:
            return None

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

        # Try to get a plain dict from pydantic model or mapping
        if hasattr(usage_obj, "model_dump"):
            data = usage_obj.model_dump()
        elif isinstance(usage_obj, dict):
            data = usage_obj
        else:
            data = {}
            for attr in (
                "input_tokens",
                "output_tokens",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "prompt_cache_hit_tokens",
                "prompt_cache_miss_tokens",
            ):
                if hasattr(usage_obj, attr):
                    data[attr] = getattr(usage_obj, attr)

        prompt_cache_hit_tokens = data.get("prompt_cache_hit_tokens")
        prompt_cache_miss_tokens = data.get("prompt_cache_miss_tokens")

        if prompt_cache_hit_tokens is not None or prompt_cache_miss_tokens is not None:
            cache_read_input_tokens = int(prompt_cache_hit_tokens or 0)
            prompt_tokens = cache_read_input_tokens + int(prompt_cache_miss_tokens or 0)
        else:
            prompt_tokens = data.get("input_tokens")
            if prompt_tokens is None:
                prompt_tokens = data.get("prompt_tokens")
            cached_tokens = _deep_get(data, "prompt_tokens_details", "cached_tokens")
            if cached_tokens is None:
                cached_tokens = _deep_get(data, "input_tokens_details", "cached_tokens")
            if cached_tokens is None:
                cached_tokens = _deep_get(
                    usage_obj, "prompt_tokens_details", "cached_tokens"
                )
            if cached_tokens is None:
                cached_tokens = _deep_get(
                    usage_obj, "input_tokens_details", "cached_tokens"
                )
            cache_read_input_tokens = (
                int(cached_tokens) if cached_tokens is not None else 0
            )

        if cache_read_input_tokens < 0:
            cache_read_input_tokens = 0

        completion_tokens = data.get("output_tokens")
        if completion_tokens is None:
            completion_tokens = data.get("completion_tokens")

        if prompt_tokens is None and hasattr(usage_obj, "prompt_tokens"):
            prompt_tokens = getattr(usage_obj, "prompt_tokens")
        if completion_tokens is None and hasattr(usage_obj, "completion_tokens"):
            completion_tokens = getattr(usage_obj, "completion_tokens")

        if prompt_tokens is None or completion_tokens is None:
            return None

        total_tokens = data.get("total_tokens")
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        return {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(total_tokens),
            "cache_read_input_tokens": cache_read_input_tokens,
            "cache_write_input_tokens": 0,
            "cache_write_by_ttl": {
                "ephemeral_5m_input_tokens": 0,
                "ephemeral_1h_input_tokens": 0,
            },
        }

    def _prepare_request_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize DeepSeek-specific request controls for the OpenAI SDK."""
        prepared = dict(kwargs)

        thinking = prepared.pop("thinking", None)
        if thinking is not None:
            extra_body = dict(prepared.get("extra_body") or {})
            extra_body["thinking"] = thinking
            prepared["extra_body"] = extra_body

        reasoning = prepared.pop("reasoning", None)
        if reasoning is not None and "reasoning_effort" not in prepared:
            if isinstance(reasoning, dict):
                reasoning_effort = reasoning.get("effort")
            else:
                reasoning_effort = reasoning
            if reasoning_effort:
                prepared["reasoning_effort"] = reasoning_effort

        extra_body = prepared.get("extra_body") or {}
        thinking_config = (
            extra_body.get("thinking") if isinstance(extra_body, dict) else None
        )
        thinking_type = (
            thinking_config.get("type")
            if isinstance(thinking_config, dict)
            else getattr(thinking_config, "type", None)
        )
        if thinking_type == "disabled":
            prepared.pop("reasoning_effort", None)
        elif thinking_type == "enabled":
            for parameter in (
                "temperature",
                "top_p",
                "presence_penalty",
                "frequency_penalty",
            ):
                prepared.pop(parameter, None)

        return prepared

    @staticmethod
    def _truncate_string(value: str, limit: int = 500) -> str:
        if len(value) <= limit:
            return value
        return value[:limit] + f"...<truncated {len(value) - limit} chars>"

    def _summarize_pending_tool_call_errors(self) -> List[Dict[str, Any]]:
        pending: List[Dict[str, Any]] = []
        for index, tool_call in sorted(
            self._streaming_tool_calls.items(), key=lambda item: str(item[0])
        ):
            function = tool_call.get("function") or {}
            arguments = function.get("arguments") or ""
            parse_error = None
            if arguments:
                try:
                    json.loads(arguments)
                except json.JSONDecodeError as exc:
                    parse_error = f"{exc.msg} at position {exc.pos}"
            else:
                parse_error = "missing function.arguments"

            pending.append(
                {
                    "index": index,
                    "id": tool_call.get("id") or None,
                    "type": tool_call.get("type") or None,
                    "function_name": function.get("name") or None,
                    "arguments_len": len(arguments),
                    "arguments_preview": self._truncate_string(arguments, 240),
                    "parse_error": parse_error,
                }
            )
        return pending

    def _build_pending_tool_call_error_stop_info(
        self,
        *,
        finish_reason: str,
        model: str,
    ) -> StopInfo:
        pending_tool_calls = self._summarize_pending_tool_call_errors()
        metadata = {
            "has_content": self._stream_content_length > 0,
            "content_length": self._stream_content_length,
            "tool_calls_count": self._stream_tool_calls_count,
            "pending_tool_calls_count": len(pending_tool_calls),
            "pending_tool_calls": pending_tool_calls,
            "finish_reason": finish_reason,
            "model": model,
            "provider": self.PROVIDER_NAME,
            "error_class": "malformed_streaming_tool_call_arguments",
            "error_message": "模型生成了工具调用，但工具参数不是合法 JSON",
            "retryable": True,
        }
        return StopInfo(
            reason=StopReason.TOOL_CALL_ERROR,
            original_reason=finish_reason,
            metadata=metadata,
        )

    async def chat_completions_create(
        self, model, messages, stream: bool = False, **kwargs
    ) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionResponse, None]]:
        # Any exception raised by OpenAI will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        replay_request_view = kwargs.pop("_replay_request_view", None)
        replay_mode = kwargs.pop("_replay_mode", None)
        if (
            replay_request_view is not None
            and replay_mode == "canonical_with_reasoning"
        ):
            prepared_messages = replay_request_view
        else:
            replay_build = self.build_replay_view(model, messages, **kwargs)
            prepared_messages = replay_build.request_view

        request_kwargs = self._prepare_request_kwargs(kwargs)

        if stream:
            # Reset streaming state
            self._streaming_tool_calls = {}
            self._streaming_reasoning = ""
            self._stream_content_length = 0
            self._stream_tool_calls_count = 0

            # Ensure usage is included in streaming responses
            stream_kwargs = request_kwargs.copy()
            stream_options = stream_kwargs.get("stream_options") or {}
            if "include_usage" not in stream_options:
                stream_options["include_usage"] = True
            stream_kwargs["stream_options"] = stream_options

            response = await self.client.chat.completions.create(
                model=model,
                messages=prepared_messages,
                stream=True,
                **stream_kwargs,  # Pass any additional arguments to the OpenAI API
            )

            async def stream_generator():
                try:
                    stream_usage = None
                    chunk_index = 0
                    async for chunk in response:
                        chunk_index += 1
                        # Capture usage from streaming chunks (only appears on final chunk)
                        if hasattr(chunk, "usage") and chunk.usage:
                            stream_usage = (
                                self._normalize_usage(chunk.usage) or stream_usage
                            )

                        # OpenAI-compatible APIs can emit usage in a trailing
                        # chunk whose choices list is empty. Forward that chunk
                        # so downstream accounting does not lose token usage.
                        if not chunk.choices and stream_usage:
                            yield ChatCompletionResponse(
                                choices=[],
                                metadata={
                                    "id": getattr(chunk, "id", None),
                                    "created": getattr(chunk, "created", None),
                                    "model": getattr(chunk, "model", None) or model,
                                    "usage": stream_usage,
                                },
                            )
                            continue

                        if chunk.choices:
                            # Create choices with stop_info
                            choices = []
                            for choice in chunk.choices:
                                # Accumulate content and tool calls for accurate metadata
                                if choice.delta.content:
                                    self._stream_content_length += len(
                                        choice.delta.content
                                    )
                                reasoning_content = getattr(
                                    choice.delta, "reasoning_content", None
                                )
                                if reasoning_content:
                                    self._accumulate_reasoning_content(
                                        reasoning_content
                                    )

                                accumulated_tool_calls = (
                                    self._accumulate_and_convert_tool_calls(
                                        choice.delta
                                    )
                                )
                                if accumulated_tool_calls:
                                    self._stream_tool_calls_count += len(
                                        accumulated_tool_calls
                                    )

                                # Create stop_info if finish_reason is present
                                stop_info = None
                                if choice.finish_reason:
                                    # Use accumulated values for final stop_info
                                    if choice.finish_reason == "stop":
                                        metadata = {
                                            "has_content": self._stream_content_length
                                            > 0
                                            or self._stream_tool_calls_count > 0,
                                            "content_length": self._stream_content_length,
                                            "tool_calls_count": self._stream_tool_calls_count,
                                            "finish_reason": choice.finish_reason,
                                            "model": chunk.model,
                                            "provider": self.PROVIDER_NAME,
                                        }
                                        stop_info = stop_reason_manager.map_stop_reason(
                                            "openai", choice.finish_reason, metadata
                                        )
                                        stop_info.metadata["provider"] = (
                                            self.PROVIDER_NAME
                                        )
                                    elif (
                                        choice.finish_reason == "tool_calls"
                                        and not accumulated_tool_calls
                                        and self._streaming_tool_calls
                                    ):
                                        stop_info = self._build_pending_tool_call_error_stop_info(
                                            finish_reason=choice.finish_reason,
                                            model=chunk.model,
                                        )
                                    else:
                                        choice_data = {
                                            "delta": choice.delta,
                                            "stream_content_length": self._stream_content_length,
                                            "stream_tool_calls_count": self._stream_tool_calls_count,
                                        }
                                        stop_info = self._create_stop_info(
                                            choice.finish_reason,
                                            choice_data,
                                            chunk.model,
                                        )

                                choices.append(
                                    StreamChoice(
                                        index=choice.index,
                                        delta=ChoiceDelta(
                                            content=choice.delta.content,
                                            role=choice.delta.role,
                                            tool_calls=accumulated_tool_calls,
                                            reasoning_content=reasoning_content,
                                        ),
                                        finish_reason=choice.finish_reason,
                                        stop_info=stop_info,
                                    )
                                )

                            # Determine if this is the final chunk
                            is_final_chunk = (
                                any(c.finish_reason for c in chunk.choices)
                                or stream_usage is not None
                            )
                            metadata = {
                                "id": chunk.id,
                                "created": chunk.created,
                                "model": chunk.model,
                            }
                            if is_final_chunk and stream_usage:
                                metadata["usage"] = stream_usage

                            yield ChatCompletionResponse(
                                choices=choices,
                                metadata=metadata,
                            )
                finally:
                    await self._close_stream_response(response)

            return stream_generator()
        else:
            response = await self.client.chat.completions.create(
                model=model,
                messages=prepared_messages,
                stream=False,
                **request_kwargs,  # Pass any additional arguments to the OpenAI API
            )

            # Create choices with stop_info
            choices = []
            for choice in response.choices:
                # Create stop_info
                choice_data = {"message": choice.message}
                finish_reason = getattr(choice, "finish_reason", None)
                stop_info = self._create_stop_info(finish_reason, choice_data, model)
                tool_calls = (
                    self._convert_tool_calls(choice.message.tool_calls)
                    if hasattr(choice.message, "tool_calls")
                    and choice.message.tool_calls
                    else None
                )

                choices.append(
                    Choice(
                        index=choice.index,
                        message=Message(
                            content=choice.message.content,
                            role=choice.message.role,
                            tool_calls=tool_calls,
                            refusal=None,
                            reasoning_content=self._convert_reasoning_content(
                                getattr(choice.message, "reasoning_content", None),
                                preserve_empty=bool(tool_calls),
                            ),
                        ),
                        finish_reason=finish_reason,
                        stop_info=stop_info,
                    )
                )

            usage = getattr(response, "usage", None)
            usage_dict = self._normalize_usage(usage)

            metadata = {
                "id": response.id,
                "created": response.created,
                "model": model,
            }
            if usage_dict:
                metadata["usage"] = usage_dict

            return ChatCompletionResponse(
                choices=choices,
                metadata=metadata,
            )

    def _convert_reasoning_content(
        self,
        reasoning_content,
        *,
        preserve_empty: bool = False,
    ):
        """Convert DeepSeek reasoning_content to ReasoningContent object."""
        if reasoning_content is None:
            if not preserve_empty:
                return None
            reasoning_content = ""
        elif not isinstance(reasoning_content, str):
            reasoning_content = str(reasoning_content)

        if not reasoning_content and not preserve_empty:
            return None

        return ReasoningContent(
            thinking=reasoning_content,
            provider=self.PROVIDER_NAME,
            raw_data=self._build_reasoning_replay_payload(reasoning_content),
        )

    def _accumulate_and_convert_tool_calls(self, delta):
        """
        Accumulate tool call chunks and convert to unified format when complete.

        Args:
            delta: The delta object from streaming response

        Returns:
            List of converted tool calls if any are complete, None otherwise
        """
        if not hasattr(delta, "tool_calls") or not delta.tool_calls:
            return None

        # Accumulate tool call chunks
        for tool_call_delta in delta.tool_calls:
            index = getattr(tool_call_delta, "index", 0)
            function_delta = getattr(tool_call_delta, "function", None)
            has_payload = bool(
                getattr(tool_call_delta, "id", None)
                or getattr(function_delta, "name", None)
                or getattr(function_delta, "arguments", None)
            )

            # Some OpenAI-compatible endpoints repeat an empty tool-call item in
            # the terminal chunk. It is only a protocol placeholder: creating a
            # fresh accumulator for it would turn a completed call into a false
            # malformed_streaming_tool_call_arguments error.
            if not has_payload:
                continue

            # Initialize tool call accumulator if not exists
            if index not in self._streaming_tool_calls:
                self._streaming_tool_calls[index] = {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }

            tool_call = self._streaming_tool_calls[index]

            # Accumulate id
            if hasattr(tool_call_delta, "id") and tool_call_delta.id:
                tool_call["id"] += tool_call_delta.id

            # Accumulate function data
            if function_delta:
                if hasattr(function_delta, "name") and function_delta.name:
                    tool_call["function"]["name"] += function_delta.name

                if hasattr(function_delta, "arguments") and function_delta.arguments:
                    tool_call["function"]["arguments"] += function_delta.arguments

            # Set type if provided
            if hasattr(tool_call_delta, "type") and tool_call_delta.type:
                tool_call["type"] = tool_call_delta.type

        # Check for complete tool calls and convert them
        complete_tool_calls = []
        for index, tool_call_data in list(self._streaming_tool_calls.items()):
            if (
                tool_call_data["id"]
                and tool_call_data["function"]["name"]
                and tool_call_data["function"]["arguments"]
            ):

                try:
                    # Try to parse arguments as JSON to ensure completeness
                    json.loads(tool_call_data["function"]["arguments"])

                    # Create mock object for _convert_tool_calls
                    class MockToolCall:
                        def __init__(self, id, function_name, function_args):
                            self.id = id
                            self.function = MockFunction(function_name, function_args)

                    class MockFunction:
                        def __init__(self, name, arguments):
                            self.name = name
                            self.arguments = arguments

                    mock_tool_call = MockToolCall(
                        tool_call_data["id"],
                        tool_call_data["function"]["name"],
                        tool_call_data["function"]["arguments"],
                    )

                    # Use existing conversion logic
                    converted = self._convert_tool_calls([mock_tool_call])
                    if converted:
                        complete_tool_calls.extend(converted)

                    # Remove completed tool call from accumulator
                    del self._streaming_tool_calls[index]

                except json.JSONDecodeError:
                    # Arguments are not complete yet, continue accumulating
                    continue

        return complete_tool_calls if complete_tool_calls else None

    def _convert_tool_calls(self, tool_calls):
        """Convert tool calls to the framework's format."""
        if not tool_calls:
            return None

        converted_tool_calls = []
        for tool_call in tool_calls:
            function = Function(
                name=tool_call.function.name, arguments=tool_call.function.arguments
            )
            tool_call_obj = ChatCompletionMessageToolCall(
                id=tool_call.id, function=function, type="function"
            )
            converted_tool_calls.append(tool_call_obj)
        return converted_tool_calls
