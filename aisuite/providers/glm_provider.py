import json
import os
from typing import Any, Dict, Generator, List, Optional, Union

from zai import ZhipuAiClient

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
from aisuite.framework.stop_reason import stop_reason_manager
from aisuite.provider import LLMError, Provider


class GlmProvider(Provider):
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
        self._stream_content_length = 0
        self._stream_tool_calls_count = 0

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
                prepared["reasoning_content"] = self._extract_reasoning_input(
                    reasoning_content
                )

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
        if reasoning_content is None:
            return None

        if isinstance(reasoning_content, ReasoningContent):
            raw_data = reasoning_content.raw_data or {}
            return raw_data.get("reasoning_content") or reasoning_content.thinking

        if isinstance(reasoning_content, dict):
            raw_data = reasoning_content.get("raw_data") or {}
            return (
                raw_data.get("reasoning_content")
                or reasoning_content.get("thinking")
                or reasoning_content.get("text")
                or str(reasoning_content)
            )

        return str(reasoning_content)

    def _create_stop_info(
        self, finish_reason: Optional[str], choice_data: dict = None, model: str = None
    ):
        if not finish_reason:
            return None

        has_content = False
        content_length = 0
        tool_calls_count = 0

        if choice_data:
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
            raw_data={"reasoning_content": reasoning_content},
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

    def chat_completions_create(
        self, model, messages, stream: bool = False, **kwargs
    ) -> Union[ChatCompletionResponse, Generator[ChatCompletionResponse, None, None]]:
        prepared_messages = self._prepare_messages(messages)
        prepared_kwargs = self._prepare_kwargs(kwargs)

        try:
            if stream:
                self._streaming_tool_calls = {}
                self._stream_content_length = 0
                self._stream_tool_calls_count = 0

                response = self.client.chat.completions.create(
                    model=model,
                    messages=prepared_messages,
                    stream=True,
                    **prepared_kwargs,
                )

                def stream_generator():
                    stream_usage = None

                    for chunk in response:
                        if getattr(chunk, "usage", None):
                            stream_usage = (
                                self._normalize_usage(chunk.usage) or stream_usage
                            )

                        if not getattr(chunk, "choices", None):
                            continue

                        choices = []
                        for choice in chunk.choices:
                            if getattr(choice.delta, "content", None):
                                self._stream_content_length += len(choice.delta.content)

                            converted_tool_calls = (
                                self._accumulate_and_convert_tool_calls(choice.delta)
                            )
                            if converted_tool_calls:
                                self._stream_tool_calls_count += len(
                                    converted_tool_calls
                                )

                            stop_info = None
                            if choice.finish_reason:
                                if choice.finish_reason == "stop":
                                    stop_info = stop_reason_manager.map_stop_reason(
                                        "openai",
                                        choice.finish_reason,
                                        {
                                            "has_content": self._stream_content_length
                                            > 0
                                            or self._stream_tool_calls_count > 0,
                                            "content_length": self._stream_content_length,
                                            "tool_calls_count": self._stream_tool_calls_count,
                                            "finish_reason": choice.finish_reason,
                                            "model": chunk.model or model,
                                            "provider": "glm",
                                        },
                                    )
                                    stop_info.metadata["provider"] = "glm"
                                else:
                                    stop_info = self._create_stop_info(
                                        choice.finish_reason,
                                        {"delta": choice.delta},
                                        chunk.model or model,
                                    )

                            choices.append(
                                StreamChoice(
                                    index=choice.index,
                                    delta=ChoiceDelta(
                                        content=choice.delta.content,
                                        role=choice.delta.role,
                                        reasoning_content=getattr(
                                            choice.delta, "reasoning_content", None
                                        ),
                                        tool_calls=converted_tool_calls,
                                    ),
                                    finish_reason=choice.finish_reason,
                                    stop_info=stop_info,
                                )
                            )

                        metadata = {
                            "id": getattr(chunk, "id", None),
                            "created": getattr(chunk, "created", None),
                            "model": getattr(chunk, "model", None) or model,
                        }
                        if any(c.finish_reason for c in chunk.choices) and stream_usage:
                            metadata["usage"] = stream_usage

                        yield ChatCompletionResponse(choices=choices, metadata=metadata)

                return stream_generator()

            response = self.client.chat.completions.create(
                model=model,
                messages=prepared_messages,
                stream=False,
                **prepared_kwargs,
            )

            choices = []
            for choice in response.choices:
                finish_reason = getattr(choice, "finish_reason", None)
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
