import openai
import os
import json
from typing import AsyncGenerator, Union

from aisuite.provider import Provider, LLMError
from aisuite.framework.chat_completion_response import ChatCompletionResponse, Choice, ChoiceDelta, StreamChoice
from aisuite.framework.message import Message, ChatCompletionMessageToolCall, Function, ReasoningContent
from aisuite.framework.stop_reason import stop_reason_manager


class DeepseekProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the DeepSeek provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("DEEPSEEK_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "DeepSeek API key is missing. Please provide it in the config or set the OPENAI_API_KEY environment variable."
            )
        config["base_url"] = "https://api.deepseek.com"

        # NOTE: We could choose to remove above lines for api_key since OpenAI will automatically
        # infer certain values from the environment variables.
        # Eg: OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_PROJECT_ID. Except for OPEN_AI_BASE_URL which has to be the deepseek url

        # Pass the entire config to the OpenAI client constructor
        self.client = openai.AsyncOpenAI(**config)

        # State for accumulating streaming tool calls
        self._streaming_tool_calls = {}

        # Track accumulated content for streaming responses
        # Used to provide accurate metadata in stop_info
        self._stream_content_length = 0
        self._stream_tool_calls_count = 0

    def _create_stop_info(self, finish_reason: str, choice_data: dict = None, model: str = None) -> dict:
        """Create StopInfo from OpenAI-compatible finish_reason."""
        if not finish_reason:
            return None

        # Analyze choice data to determine content presence
        has_content = False
        content_length = 0
        tool_calls_count = 0

        if choice_data:
            # Check message content
            message = choice_data.get("message") or choice_data.get("delta")
            if message:
                content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else None)
                if content:
                    has_content = True
                    content_length = len(content)

                # Check tool calls
                tool_calls = getattr(message, "tool_calls", None) or (message.get("tool_calls") if isinstance(message, dict) else None)
                if tool_calls:
                    tool_calls_count = len(tool_calls)
                    if tool_calls_count > 0:
                        has_content = True  # Tool calls count as content

        metadata = {
            "has_content": has_content,
            "content_length": content_length,
            "tool_calls_count": tool_calls_count,
            "finish_reason": finish_reason,
            "model": model,
            "provider": "deepseek"
        }

        # Use OpenAI mapping since DeepSeek is OpenAI-compatible, but preserve DeepSeek provider identity
        stop_info = stop_reason_manager.map_stop_reason("openai", finish_reason, metadata)
        # Override provider in metadata to correctly identify as DeepSeek
        stop_info.metadata["provider"] = "deepseek"
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
            for attr in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens", "total_tokens"):
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
                cached_tokens = _deep_get(usage_obj, "prompt_tokens_details", "cached_tokens")
            if cached_tokens is None:
                cached_tokens = _deep_get(usage_obj, "input_tokens_details", "cached_tokens")
            cache_read_input_tokens = int(cached_tokens) if cached_tokens is not None else 0

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


    async def chat_completions_create(self, model, messages, stream: bool = False, **kwargs) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionResponse, None]]:
        # Any exception raised by OpenAI will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        if stream:
            # Reset streaming state
            self._streaming_tool_calls = {}
            self._stream_content_length = 0
            self._stream_tool_calls_count = 0

            # Ensure usage is included in streaming responses
            stream_kwargs = kwargs.copy()
            stream_options = stream_kwargs.get("stream_options") or {}
            if "include_usage" not in stream_options:
                stream_options["include_usage"] = True
            stream_kwargs["stream_options"] = stream_options

            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **stream_kwargs,  # Pass any additional arguments to the OpenAI API
            )

            async def stream_generator():
                stream_usage = None
                async for chunk in response:
                    # Capture usage from streaming chunks (only appears on final chunk)
                    if hasattr(chunk, "usage") and chunk.usage:
                        stream_usage = self._normalize_usage(chunk.usage) or stream_usage

                    if chunk.choices:
                        # Create choices with stop_info
                        choices = []
                        for choice in chunk.choices:
                            # Accumulate content and tool calls for accurate metadata
                            if choice.delta.content:
                                self._stream_content_length += len(choice.delta.content)

                            accumulated_tool_calls = self._accumulate_and_convert_tool_calls(choice.delta)
                            if accumulated_tool_calls:
                                self._stream_tool_calls_count += len(accumulated_tool_calls)

                            # Create stop_info if finish_reason is present
                            stop_info = None
                            if choice.finish_reason:
                                # Use accumulated values for final stop_info
                                if choice.finish_reason == "stop":
                                    metadata = {
                                        "has_content": self._stream_content_length > 0 or self._stream_tool_calls_count > 0,
                                        "content_length": self._stream_content_length,
                                        "tool_calls_count": self._stream_tool_calls_count,
                                        "finish_reason": choice.finish_reason,
                                        "model": chunk.model,
                                        "provider": "deepseek",
                                    }
                                    stop_info = stop_reason_manager.map_stop_reason("openai", choice.finish_reason, metadata)
                                    stop_info.metadata["provider"] = "deepseek"
                                else:
                                    choice_data = {"delta": choice.delta}
                                    stop_info = self._create_stop_info(choice.finish_reason, choice_data, chunk.model)

                            choices.append(
                                StreamChoice(
                                    index=choice.index,
                                    delta=ChoiceDelta(
                                        content=choice.delta.content,
                                        role=choice.delta.role,
                                        tool_calls=accumulated_tool_calls,
                                        reasoning_content=getattr(choice.delta, "reasoning_content", None),
                                    ),
                                    finish_reason=choice.finish_reason,
                                    stop_info=stop_info,
                                )
                            )

                        # Determine if this is the final chunk
                        is_final_chunk = any(c.finish_reason for c in chunk.choices) or stream_usage is not None
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

            return stream_generator()
        else:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                **kwargs,  # Pass any additional arguments to the OpenAI API
            )

            # Create choices with stop_info
            choices = []
            for choice in response.choices:
                # Create stop_info
                choice_data = {"message": choice.message}
                finish_reason = getattr(choice, "finish_reason", None)
                stop_info = self._create_stop_info(finish_reason, choice_data, model)

                choices.append(
                    Choice(
                        index=choice.index,
                        message=Message(
                            content=choice.message.content,
                            role=choice.message.role,
                            tool_calls=self._convert_tool_calls(choice.message.tool_calls)
                            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls
                            else None,
                            refusal=None,
                            reasoning_content=self._convert_reasoning_content(getattr(choice.message, "reasoning_content", None)),
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

    def _convert_reasoning_content(self, reasoning_content):
        """Convert DeepSeek reasoning_content to ReasoningContent object."""
        if not reasoning_content:
            return None

        return ReasoningContent(
            thinking=reasoning_content,
            provider="deepseek",
            raw_data={"reasoning_content": reasoning_content}
        )

    def _accumulate_and_convert_tool_calls(self, delta):
        """
        Accumulate tool call chunks and convert to unified format when complete.

        Args:
            delta: The delta object from streaming response

        Returns:
            List of converted tool calls if any are complete, None otherwise
        """
        if not hasattr(delta, 'tool_calls') or not delta.tool_calls:
            return None

        # Accumulate tool call chunks
        for tool_call_delta in delta.tool_calls:
            index = getattr(tool_call_delta, 'index', 0)

            # Initialize tool call accumulator if not exists
            if index not in self._streaming_tool_calls:
                self._streaming_tool_calls[index] = {
                    "id": "",
                    "type": "function",
                    "function": {
                        "name": "",
                        "arguments": ""
                    }
                }

            tool_call = self._streaming_tool_calls[index]

            # Accumulate id
            if hasattr(tool_call_delta, 'id') and tool_call_delta.id:
                tool_call["id"] += tool_call_delta.id

            # Accumulate function data
            if hasattr(tool_call_delta, 'function') and tool_call_delta.function:
                if hasattr(tool_call_delta.function, 'name') and tool_call_delta.function.name:
                    tool_call["function"]["name"] += tool_call_delta.function.name

                if hasattr(tool_call_delta.function, 'arguments') and tool_call_delta.function.arguments:
                    tool_call["function"]["arguments"] += tool_call_delta.function.arguments

            # Set type if provided
            if hasattr(tool_call_delta, 'type') and tool_call_delta.type:
                tool_call["type"] = tool_call_delta.type

        # Check for complete tool calls and convert them
        complete_tool_calls = []
        for index, tool_call_data in list(self._streaming_tool_calls.items()):
            if (tool_call_data["id"] and
                tool_call_data["function"]["name"] and
                tool_call_data["function"]["arguments"]):

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
                        tool_call_data["function"]["arguments"]
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
                name=tool_call.function.name,
                arguments=tool_call.function.arguments
            )
            tool_call_obj = ChatCompletionMessageToolCall(
                id=tool_call.id,
                function=function,
                type="function"
            )
            converted_tool_calls.append(tool_call_obj)
        return converted_tool_calls
