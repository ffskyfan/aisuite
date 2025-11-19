import openai
import os
import json
from typing import AsyncGenerator, Union

from aisuite.provider import Provider, LLMError
from aisuite.framework.chat_completion_response import ChatCompletionResponse, Choice, ChoiceDelta, StreamChoice
from aisuite.framework.message import Message, ChatCompletionMessageToolCall, Function, ReasoningContent


class VercelProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the Vercel AI Gateway provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("AI_GATEWAY_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "AI Gateway API key is missing. Please provide it in the config or set the AI_GATEWAY_API_KEY environment variable."
            )
        config["base_url"] = "https://ai-gateway.vercel.sh/v1"

        # Pass the entire config to the OpenAI client constructor
        self.client = openai.AsyncOpenAI(**config)

        # State for accumulating streaming tool calls
        self._streaming_tool_calls = {}

    def _normalize_usage(self, usage_obj):
        """Normalize OpenAI-style usage objects to a standard dict.

        Supports both Chat Completions usage (prompt_tokens/completion_tokens)
        and possible input/output token naming variants.
        """
        if not usage_obj:
            return None

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

        prompt_tokens = data.get("input_tokens") or data.get("prompt_tokens")
        completion_tokens = data.get("output_tokens") or data.get("completion_tokens")

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
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }


    async def chat_completions_create(self, model, messages, stream: bool = False, **kwargs) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionResponse, None]]:
        # Any exception raised by OpenAI will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        if stream:
            # Reset streaming tool calls state
            self._streaming_tool_calls = {}

            # Clean messages parameter, remove ReasoningContent objects (following OpenAI provider pattern)
            cleaned_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    cleaned_msg = msg.copy()
                    # Remove reasoning_content field as Vercel AI Gateway doesn't need it as input
                    if "reasoning_content" in cleaned_msg:
                        cleaned_msg.pop("reasoning_content")
                    cleaned_messages.append(cleaned_msg)
                else:
                    # If it's a Message object, convert to dict and remove reasoning_content
                    if hasattr(msg, "model_dump"):
                        cleaned_msg = msg.model_dump()
                        if "reasoning_content" in cleaned_msg:
                            cleaned_msg.pop("reasoning_content")
                        cleaned_messages.append(cleaned_msg)
                    else:
                        cleaned_messages.append(msg)

            # Ensure usage is included in streaming responses
            stream_kwargs = kwargs.copy()
            stream_options = stream_kwargs.get("stream_options") or {}
            if "include_usage" not in stream_options:
                stream_options["include_usage"] = True
            stream_kwargs["stream_options"] = stream_options

            response = await self.client.chat.completions.create(
                model=model,
                messages=cleaned_messages,  # Use cleaned messages
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
                        choices = []
                        for choice in chunk.choices:
                            choices.append(
                                StreamChoice(
                                    index=choice.index,
                                    delta=ChoiceDelta(
                                        content=choice.delta.content,
                                        role=choice.delta.role,
                                        tool_calls=self._accumulate_and_convert_tool_calls(choice.delta),
                                        reasoning_content=getattr(
                                            choice.delta,
                                            "reasoning_content",
                                            None,
                                        ),  # Keep for OpenAI compatibility, though Vercel AI Gateway may not support it yet
                                    ),
                                    finish_reason=choice.finish_reason,
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
            # For non-streaming calls, also need to clean messages (following OpenAI provider pattern)
            cleaned_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    cleaned_msg = msg.copy()
                    if "reasoning_content" in cleaned_msg:
                        cleaned_msg.pop("reasoning_content")
                    cleaned_messages.append(cleaned_msg)
                else:
                    if hasattr(msg, "model_dump"):
                        cleaned_msg = msg.model_dump()
                        if "reasoning_content" in cleaned_msg:
                            cleaned_msg.pop("reasoning_content")
                        cleaned_messages.append(cleaned_msg)
                    else:
                        cleaned_messages.append(msg)

            response = await self.client.chat.completions.create(
                model=model,
                messages=cleaned_messages,  # Use cleaned messages
                stream=False,
                **kwargs,  # Pass any additional arguments to the OpenAI API
            )

            usage = getattr(response, "usage", None)
            usage_dict = self._normalize_usage(usage)

            return ChatCompletionResponse(
                choices=[
                    Choice(
                        index=choice.index,
                        message=Message(
                            content=choice.message.content,
                            role=choice.message.role,
                            tool_calls=self._convert_tool_calls(choice.message.tool_calls)
                            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls
                            else None,
                            refusal=None,
                            reasoning_content=self._convert_reasoning_content(getattr(choice.message, "reasoning_content", None)),  # Keep for OpenAI compatibility
                        ),
                        finish_reason=getattr(choice, "finish_reason", None),
                    )
                    for choice in response.choices
                ],
                metadata={
                    "id": response.id,
                    "created": response.created,
                    "model": model,
                    **({"usage": usage_dict} if usage_dict else {}),
                },
            )

    def _convert_reasoning_content(self, reasoning_content):
        """
        Convert Vercel AI Gateway reasoning_content to ReasoningContent object.

        Note: As of current documentation review, Vercel AI Gateway's OpenAI-compatible API
        does not explicitly support reasoning content parameters. However, this method is
        kept for OpenAI compatibility and potential future support.
        """
        if not reasoning_content:
            return None

        return ReasoningContent(
            thinking=reasoning_content,
            provider="vercel",
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

            # Accumulate function name and arguments
            if hasattr(tool_call_delta, 'function') and tool_call_delta.function:
                if hasattr(tool_call_delta.function, 'name') and tool_call_delta.function.name:
                    tool_call["function"]["name"] += tool_call_delta.function.name
                if hasattr(tool_call_delta.function, 'arguments') and tool_call_delta.function.arguments:
                    tool_call["function"]["arguments"] += tool_call_delta.function.arguments

        # Check for complete tool calls
        complete_tool_calls = []
        for index, tool_call_data in list(self._streaming_tool_calls.items()):
            if tool_call_data["id"] and tool_call_data["function"]["name"] and tool_call_data["function"]["arguments"]:

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
