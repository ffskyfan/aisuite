import json
import os
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Optional, Union

import httpx
from zai import ZhipuAiClient
from zai.core import drop_prefix_image_data

from aisuite.framework.chat_completion_response import (
    ChatCompletionResponse,
    Choice,
    ChoiceDelta,
    StreamChoice,
)
from aisuite.framework.message import (
    ChatCompletionMessageToolCall,
    Function,
    Message,
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
from aisuite.framework.stop_reason import stop_reason_manager
from aisuite.provider import LLMError, Provider


class GlmProvider(Provider):
    REASONING_REPLAY_KIND = "glm_reasoning_text"

    def __init__(self, **config):
        api_key = (
            config.get("api_key")
            or os.getenv("ZAI_API_KEY")
            or os.getenv("ZHIPUAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "GLM-5 API key is missing. Please provide it in the config or set "
                "the ZAI_API_KEY or ZHIPUAI_API_KEY environment variable."
            )

        client_config = dict(config)
        client_config["api_key"] = api_key
        self.client = ZhipuAiClient(**client_config)

        self._streaming_tool_calls: Dict[int, Dict[str, Any]] = {}
        self._streaming_reasoning = ""
        self._stream_content_length = 0
        self._stream_tool_calls_count = 0

    @staticmethod
    def _to_namespace(value: Any) -> Any:
        if isinstance(value, Mapping):
            return SimpleNamespace(
                **{key: GlmProvider._to_namespace(item) for key, item in value.items()}
            )
        if isinstance(value, list):
            return [GlmProvider._to_namespace(item) for item in value]
        return value

    def _build_http_body(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        request_kwargs = dict(kwargs)
        temperature = request_kwargs.get("temperature")
        if temperature is not None:
            if temperature <= 0:
                request_kwargs["do_sample"] = False
                request_kwargs["temperature"] = 0.01
            elif temperature >= 1:
                request_kwargs["temperature"] = 0.99

        top_p = request_kwargs.get("top_p")
        if top_p is not None:
            if top_p >= 1:
                request_kwargs["top_p"] = 0.99
            elif top_p <= 0:
                request_kwargs["top_p"] = 0.01

        prepared_messages = []
        for message in messages:
            prepared = dict(message)
            if prepared.get("content"):
                prepared["content"] = drop_prefix_image_data(prepared["content"])
            prepared_messages.append(prepared)

        body = {
            "model": model,
            "messages": prepared_messages,
            "stream": stream,
            **request_kwargs,
        }
        return self.client._prepare_json_data(body)

    def _request_url(self, path: str) -> str:
        return str(self.client._prepare_url(path))

    def _request_headers(self) -> Dict[str, str]:
        return dict(self.client._default_headers)

    async def _async_post_json(self, body: Dict[str, Any]) -> Any:
        async with httpx.AsyncClient(
            timeout=self.client.timeout,
            limits=self.client._limits,
        ) as client:
            response = await client.post(
                self._request_url("/chat/completions"),
                headers=self._request_headers(),
                json=body,
            )
            response.raise_for_status()
            return self._to_namespace(response.json())

    async def _iter_sse_json(self, response: httpx.Response) -> AsyncIterator[Any]:
        event = None
        data_lines: List[str] = []

        async for line in response.aiter_lines():
            line = line.rstrip("\n")
            if not line:
                if not data_lines and event is None:
                    continue

                data = "\n".join(data_lines)
                event = None
                data_lines = []

                if data.startswith("[DONE]"):
                    break

                payload = json.loads(data)
                if isinstance(payload, Mapping) and payload.get("error"):
                    raise LLMError(
                        f"An error occurred during GLM streaming: {payload['error']}"
                    )
                yield self._to_namespace(payload)
                continue

            if line.startswith(":"):
                continue

            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]

            if field == "event":
                event = value
            elif field == "data":
                data_lines.append(value)

    async def _async_stream_json(self, body: Dict[str, Any]) -> AsyncIterator[Any]:
        async with httpx.AsyncClient(
            timeout=self.client.timeout,
            limits=self.client._limits,
        ) as client:
            request = client.build_request(
                "POST",
                self._request_url("/chat/completions"),
                headers=self._request_headers(),
                json=body,
            )
            response = await client.send(request, stream=True)
            stream_iter = self._iter_sse_json(response)
            try:
                response.raise_for_status()
                async for chunk in stream_iter:
                    yield chunk
            finally:
                await stream_iter.aclose()
                await response.aclose()

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
                        message="GLM tool result replay requires tool_call_id.",
                        provider="glm",
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
                                message="GLM assistant tool call is missing id.",
                                provider="glm",
                            )
                        )
                    if not function_name:
                        diagnostics.append(
                            ReplayDiagnostic(
                                code="missing_tool_function_name",
                                message="GLM assistant tool call is missing function name.",
                                provider="glm",
                                metadata={"tool_call_id": tool_call_id},
                            )
                        )
            if role == "assistant" and msg.get("reasoning_content") is not None and not self._has_valid_reasoning_replay_payload(
                msg.get("reasoning_content")
            ):
                diagnostics.append(
                    ReplayDiagnostic(
                        code="missing_reasoning_raw_replay",
                        message=(
                            "GLM preserved thinking requires assistant reasoning_content "
                            "to carry provider-native raw replay payload."
                        ),
                        severity="warning",
                        provider="glm",
                        metadata={"role": "assistant"},
                    )
                )

        return ReplayValidationResult(
            ok=not any(diag.severity == "error" for diag in diagnostics),
            degraded=any(diag.severity == "warning" for diag in diagnostics),
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

    def _build_reasoning_replay_payload(self, reasoning_content: str) -> Dict[str, Any]:
        return build_replay_payload(
            "glm",
            self.REASONING_REPLAY_KIND,
            {"reasoning_content": reasoning_content},
            legacy_fields={"reasoning_content": reasoning_content},
        )

    def _prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared_messages = []

        for message in messages:
            prepared = (
                message.model_dump()
                if hasattr(message, "model_dump")
                else dict(message)
            )
            prepared.pop("refusal", None)

            reasoning_content = prepared.get("reasoning_content")
            if reasoning_content is not None:
                extracted_reasoning = self._extract_reasoning_input(reasoning_content)
                if extracted_reasoning is None:
                    prepared.pop("reasoning_content", None)
                else:
                    prepared["reasoning_content"] = extracted_reasoning

            prepared_messages.append(prepared)

        return prepared_messages

    def _prepare_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        prepared = dict(kwargs)

        if "reasoning" in prepared and "thinking" not in prepared:
            prepared["thinking"] = prepared.pop("reasoning")
        else:
            prepared.pop("reasoning", None)

        # GLM models default thinking to enabled if omitted. Normalize omission to
        # an explicit disabled state so provider behavior matches AISuite's
        # cross-provider expectation that "not requested" means "off".
        if prepared.get("thinking") is None:
            prepared["thinking"] = {"type": "disabled"}

        if "max_completion_tokens" in prepared and "max_tokens" not in prepared:
            prepared["max_tokens"] = prepared.pop("max_completion_tokens")

        # OpenAI-style stream options are not supported by the Zhipu SDK.
        prepared.pop("stream_options", None)
        prepared.pop("reasoning_effort", None)
        prepared.pop("verbosity", None)

        return prepared

    def _extract_reasoning_input(self, reasoning_content: Any) -> Optional[str]:
        return self._extract_reasoning_replay_text(reasoning_content)

    def _extract_reasoning_replay_text(self, reasoning_content: Any) -> Optional[str]:
        if reasoning_content is None:
            return None

        raw_data = None
        if isinstance(reasoning_content, ReasoningContent):
            raw_data = reasoning_content.raw_data or {}
        elif isinstance(reasoning_content, dict):
            raw_data = reasoning_content.get("raw_data") or {}
        else:
            return None

        envelope = get_replay_payload(raw_data)
        if envelope and envelope.get("provider") == "glm":
            payload = unwrap_replay_payload(raw_data)
            if isinstance(payload, dict) and payload.get("reasoning_content"):
                return payload["reasoning_content"]

        legacy_reasoning_content = raw_data.get("reasoning_content")
        if isinstance(legacy_reasoning_content, str) and legacy_reasoning_content:
            return legacy_reasoning_content

        return None

    def _has_valid_reasoning_replay_payload(self, reasoning_content: Any) -> bool:
        return self._extract_reasoning_replay_text(reasoning_content) is not None

    def _accumulate_reasoning_content(self, reasoning_text: str) -> None:
        if reasoning_text:
            self._streaming_reasoning += reasoning_text

    def _get_accumulated_thinking(self) -> Dict[str, Any]:
        thinking_text = self._streaming_reasoning
        self._streaming_reasoning = ""
        if not thinking_text:
            return {"thinking": "", "raw_data": None}
        return {
            "thinking": thinking_text,
            "raw_data": self._build_reasoning_replay_payload(thinking_text),
        }

    def build_replay_view(self, model: str, messages: list, **kwargs):
        validation = self.validate_replay_window(model, messages, **kwargs)
        if not validation.ok:
            error_codes = ", ".join(
                diag.code for diag in validation.diagnostics if diag.severity == "error"
            )
            raise LLMError(f"GLM replay window validation failed: {error_codes}")

        return ReplayBuildResult(
            request_view=self._prepare_messages(messages),
            replay_mode="canonical_with_reasoning",
            degraded=validation.degraded,
            diagnostics=validation.diagnostics,
        )

    def _raise_for_terminal_error_finish_reason(
        self,
        finish_reason: Optional[str],
        *,
        model: Optional[str],
        stream: bool,
    ) -> None:
        if finish_reason != "network_error":
            return

        mode = "streaming" if stream else "non-streaming"
        raise LLMError(
            f"GLM returned finish_reason=network_error during {mode} completion"
            f" (model={model or 'unknown'})"
        )

    def _create_stop_info(
        self, finish_reason: Optional[str], choice_data: dict = None, model: str = None
    ):
        if not finish_reason:
            return None

        has_content = False
        content_length = 0
        tool_calls_count = 0

        if choice_data:
            stream_content_length = choice_data.get("stream_content_length")
            stream_tool_calls_count = choice_data.get("stream_tool_calls_count")
            message = choice_data.get("message") or choice_data.get("delta")
            if message:
                content = getattr(message, "content", None)
                if isinstance(message, dict):
                    content = content or message.get("content")
                if content:
                    has_content = True
                    content_length = len(content)

                tool_calls = getattr(message, "tool_calls", None)
                if isinstance(message, dict):
                    tool_calls = tool_calls or message.get("tool_calls")
                if tool_calls:
                    tool_calls_count = len(tool_calls)
                    has_content = True

            if isinstance(stream_content_length, int) and stream_content_length > 0:
                content_length = max(content_length, stream_content_length)
                has_content = True

            if isinstance(stream_tool_calls_count, int) and stream_tool_calls_count > 0:
                tool_calls_count = max(tool_calls_count, stream_tool_calls_count)
                has_content = True

        metadata = {
            "has_content": has_content,
            "content_length": content_length,
            "tool_calls_count": tool_calls_count,
            "finish_reason": finish_reason,
            "model": model,
            "provider": "glm",
        }

        stop_info = stop_reason_manager.map_stop_reason(
            "openai", finish_reason, metadata
        )
        stop_info.metadata["provider"] = "glm"
        return stop_info

    def _normalize_usage(self, usage_obj: Any) -> Optional[Dict[str, Any]]:
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

        if hasattr(usage_obj, "model_dump"):
            data = usage_obj.model_dump()
        elif isinstance(usage_obj, dict):
            data = usage_obj
        else:
            data = {}
            for attr in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if hasattr(usage_obj, attr):
                    data[attr] = getattr(usage_obj, attr)

        prompt_tokens = data.get("prompt_tokens")
        completion_tokens = data.get("completion_tokens")

        if prompt_tokens is None and hasattr(usage_obj, "prompt_tokens"):
            prompt_tokens = getattr(usage_obj, "prompt_tokens")
        if completion_tokens is None and hasattr(usage_obj, "completion_tokens"):
            completion_tokens = getattr(usage_obj, "completion_tokens")

        if prompt_tokens is None or completion_tokens is None:
            return None

        cached_tokens = _deep_get(data, "prompt_tokens_details", "cached_tokens")
        if cached_tokens is None:
            cached_tokens = _deep_get(
                usage_obj, "prompt_tokens_details", "cached_tokens"
            )
        cache_read_input_tokens = int(cached_tokens) if cached_tokens is not None else 0

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

    def _convert_reasoning_content(self, reasoning_content: Optional[str]):
        if not reasoning_content:
            return None

        return ReasoningContent(
            thinking=reasoning_content,
            provider="glm",
            raw_data=self._build_reasoning_replay_payload(reasoning_content),
        )

    def _convert_tool_calls(
        self, tool_calls
    ) -> Optional[List[ChatCompletionMessageToolCall]]:
        if not tool_calls:
            return None

        converted_tool_calls = []
        for tool_call in tool_calls:
            converted_tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=tool_call.id,
                    function=Function(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    ),
                    type="function",
                )
            )

        return converted_tool_calls

    def _accumulate_and_convert_tool_calls(self, delta):
        if not hasattr(delta, "tool_calls") or not delta.tool_calls:
            return None

        for tool_call_delta in delta.tool_calls:
            index = getattr(tool_call_delta, "index", 0)
            if index not in self._streaming_tool_calls:
                self._streaming_tool_calls[index] = {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }

            tool_call = self._streaming_tool_calls[index]

            if getattr(tool_call_delta, "id", None):
                tool_call["id"] += tool_call_delta.id

            if getattr(tool_call_delta, "function", None):
                function = tool_call_delta.function
                if getattr(function, "name", None):
                    tool_call["function"]["name"] += function.name
                if getattr(function, "arguments", None):
                    tool_call["function"]["arguments"] += function.arguments

            if getattr(tool_call_delta, "type", None):
                tool_call["type"] = tool_call_delta.type

        complete_tool_calls = []
        for index, tool_call_data in list(self._streaming_tool_calls.items()):
            if not (
                tool_call_data["id"]
                and tool_call_data["function"]["name"]
                and tool_call_data["function"]["arguments"]
            ):
                continue

            try:
                json.loads(tool_call_data["function"]["arguments"])
            except json.JSONDecodeError:
                continue

            class MockFunction:
                def __init__(self, name, arguments):
                    self.name = name
                    self.arguments = arguments

            class MockToolCall:
                def __init__(self, tool_call_id, function_name, function_arguments):
                    self.id = tool_call_id
                    self.function = MockFunction(function_name, function_arguments)

            converted = self._convert_tool_calls(
                [
                    MockToolCall(
                        tool_call_data["id"],
                        tool_call_data["function"]["name"],
                        tool_call_data["function"]["arguments"],
                    )
                ]
            )
            if converted:
                complete_tool_calls.extend(converted)
                del self._streaming_tool_calls[index]

        return complete_tool_calls if complete_tool_calls else None

    async def chat_completions_create(
        self, model, messages, stream: bool = False, **kwargs
    ) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionResponse, None]]:
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
        prepared_kwargs = self._prepare_kwargs(kwargs)

        try:
            if stream:
                self._streaming_tool_calls = {}
                self._streaming_reasoning = ""
                self._stream_content_length = 0
                self._stream_tool_calls_count = 0

                body = self._build_http_body(
                    model=model,
                    messages=prepared_messages,
                    stream=True,
                    kwargs=prepared_kwargs,
                )

                async def stream_generator():
                    stream_usage = None

                    async for chunk in self._async_stream_json(body):
                        if getattr(chunk, "usage", None):
                            stream_usage = (
                                self._normalize_usage(chunk.usage) or stream_usage
                            )

                        if not getattr(chunk, "choices", None):
                            continue

                        choices = []
                        for choice in chunk.choices:
                            delta = getattr(choice, "delta", SimpleNamespace())
                            if getattr(delta, "content", None):
                                self._stream_content_length += len(delta.content)
                            if getattr(delta, "reasoning_content", None):
                                self._accumulate_reasoning_content(
                                    delta.reasoning_content
                                )

                            converted_tool_calls = (
                                self._accumulate_and_convert_tool_calls(delta)
                            )
                            if converted_tool_calls:
                                self._stream_tool_calls_count += len(
                                    converted_tool_calls
                                )

                            stop_info = None
                            finish_reason = getattr(choice, "finish_reason", None)
                            if finish_reason:
                                chunk_model = getattr(chunk, "model", None) or model
                                self._raise_for_terminal_error_finish_reason(
                                    finish_reason,
                                    model=chunk_model,
                                    stream=True,
                                )
                                if finish_reason == "stop":
                                    stop_info = stop_reason_manager.map_stop_reason(
                                        "openai",
                                        finish_reason,
                                        {
                                            "has_content": self._stream_content_length
                                            > 0
                                            or self._stream_tool_calls_count > 0,
                                            "content_length": self._stream_content_length,
                                            "tool_calls_count": self._stream_tool_calls_count,
                                            "finish_reason": finish_reason,
                                            "model": chunk_model,
                                            "provider": "glm",
                                        },
                                    )
                                    stop_info.metadata["provider"] = "glm"
                                else:
                                    stop_info = self._create_stop_info(
                                        finish_reason,
                                        {
                                            "delta": delta,
                                            "stream_content_length": self._stream_content_length,
                                            "stream_tool_calls_count": self._stream_tool_calls_count,
                                        },
                                        chunk_model,
                                    )

                            choices.append(
                                StreamChoice(
                                    index=getattr(choice, "index", 0),
                                    delta=ChoiceDelta(
                                        content=getattr(delta, "content", None),
                                        role=getattr(delta, "role", None),
                                        reasoning_content=getattr(
                                            delta, "reasoning_content", None
                                        ),
                                        tool_calls=converted_tool_calls,
                                    ),
                                    finish_reason=finish_reason,
                                    stop_info=stop_info,
                                )
                            )

                        metadata = {
                            "id": getattr(chunk, "id", None),
                            "created": getattr(chunk, "created", None),
                            "model": getattr(chunk, "model", None) or model,
                        }
                        if (
                            any(
                                getattr(choice, "finish_reason", None)
                                for choice in chunk.choices
                            )
                            and stream_usage
                        ):
                            metadata["usage"] = stream_usage

                        yield ChatCompletionResponse(choices=choices, metadata=metadata)

                return stream_generator()

            body = self._build_http_body(
                model=model,
                messages=prepared_messages,
                stream=False,
                kwargs=prepared_kwargs,
            )
            response = await self._async_post_json(body)

            choices = []
            for choice in response.choices:
                finish_reason = getattr(choice, "finish_reason", None)
                self._raise_for_terminal_error_finish_reason(
                    finish_reason,
                    model=model,
                    stream=False,
                )
                choices.append(
                    Choice(
                        index=choice.index,
                        message=Message(
                            content=choice.message.content,
                            role=choice.message.role,
                            tool_calls=(
                                self._convert_tool_calls(choice.message.tool_calls)
                                if getattr(choice.message, "tool_calls", None)
                                else None
                            ),
                            refusal=None,
                            reasoning_content=self._convert_reasoning_content(
                                getattr(choice.message, "reasoning_content", None)
                            ),
                        ),
                        finish_reason=finish_reason,
                        stop_info=self._create_stop_info(
                            finish_reason, {"message": choice.message}, model
                        ),
                    )
                )

            metadata = {
                "id": getattr(response, "id", None),
                "created": getattr(response, "created", None),
                "model": getattr(response, "model", None) or model,
            }
            usage = self._normalize_usage(getattr(response, "usage", None))
            if usage:
                metadata["usage"] = usage

            return ChatCompletionResponse(choices=choices, metadata=metadata)
        except Exception as exc:
            raise LLMError(f"An error occurred while calling GLM-5: {exc}") from exc
