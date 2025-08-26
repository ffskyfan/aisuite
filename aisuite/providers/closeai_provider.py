import os
import json
from typing import Union, AsyncGenerator
from openai import AsyncOpenAI

from aisuite.provider import Provider, LLMError
from aisuite.framework.chat_completion_response import ChatCompletionResponse, Choice, ChoiceDelta, StreamChoice
from aisuite.framework.message import Message, ReasoningContent, ChatCompletionMessageToolCall, Function


class CloseaiProvider(Provider):
    """
    CloseAI provider for aisuite.
    
    CloseAI is an OpenAI-compatible API proxy service that provides access to multiple AI models
    through a unified interface. It supports all OpenAI parameters and automatically handles
    protocol conversion between different model providers.
    
    Key features:
    - Full OpenAI API compatibility
    - Multi-model aggregation interface
    - Automatic protocol conversion (ChatCompletion ↔ Response, Anthropic, Gemini)
    - Extended timeout support for reasoning models (up to 20 minutes)
    - Load balancing across multiple accounts
    
    Limitations:
    - Does not support stateful interfaces (file, fine-tune, assistants)
    - Response API supports stateless usage only
    """

    def __init__(self, **config):
        """
        Initialize the CloseAI provider.
        
        Args:
            api_key (str, optional): CloseAI API key. If not provided, will look for CLOSEAI_API_KEY environment variable.
            base_url (str, optional): Base URL for CloseAI API. Defaults to https://api.openai-proxy.org/v1
            **config: Additional configuration passed to AsyncOpenAI client
        """
        # Get API key from config or environment
        api_key = config.get('api_key') or os.getenv("CLOSEAI_API_KEY")
        if not api_key:
            raise ValueError("CloseAI API key is required. Set CLOSEAI_API_KEY environment variable or pass api_key parameter.")
        
        # Set base URL - CloseAI uses OpenAI-compatible endpoint
        base_url = config.get('base_url', 'https://api.openai-proxy.org/v1')
        
        # Initialize OpenAI client with CloseAI configuration
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            **{k: v for k, v in config.items() if k not in ['api_key', 'base_url']}
        )
        
        # Track streaming tool calls for accumulation
        self._streaming_tool_calls = {}

    def _supports_reasoning(self, model: str) -> bool:
        """
        Check if the model supports reasoning parameters.
        
        CloseAI supports all OpenAI reasoning models and automatically handles
        protocol conversion for models that require it.
        """
        # o1 series models support reasoning_effort
        if model.startswith('o1-'):
            return True
        # o3 series models support reasoning_effort (including o3, o3-mini, etc.)
        if model.startswith('o3') or model.startswith('o3-'):
            return True
        # GPT-5 series models support reasoning parameter
        if model.startswith('gpt-5'):
            return True
        # Regular GPT models (gpt-4o, gpt-4, etc.) do not support reasoning
        return False

    def _prepare_reasoning_kwargs(self, model: str, kwargs: dict) -> dict:
        """
        Prepare reasoning-related kwargs based on model type.

        CloseAI automatically handles protocol conversion and supports reasoning models
        through ChatCompletion interface. According to CloseAI docs, they automatically
        convert ChatCompletion requests to Response API for models that need it.
        """
        prepared_kwargs = kwargs.copy()

        # If model doesn't support reasoning, remove reasoning-related parameters
        if not self._supports_reasoning(model):
            # Remove reasoning parameters that would cause API errors
            prepared_kwargs.pop('reasoning', None)
            prepared_kwargs.pop('reasoning_effort', None)
            return prepared_kwargs

        # For reasoning models, handle special parameter requirements
        if (model.startswith('gpt-5') or
            model.startswith('o1-') or
            model.startswith('o3') or
            model.startswith('o3-')):
            # These models don't support max_tokens, use max_completion_tokens instead
            if 'max_tokens' in prepared_kwargs:
                max_tokens_value = prepared_kwargs.pop('max_tokens')
                prepared_kwargs['max_completion_tokens'] = max_tokens_value

            # GPT-5 only supports default temperature (1.0)
            if model.startswith('gpt-5'):
                if 'temperature' in prepared_kwargs and prepared_kwargs['temperature'] != 1.0:
                    prepared_kwargs.pop('temperature')  # Remove non-default temperature

        # CloseAI handles reasoning models through ChatCompletion interface
        # Remove reasoning parameters as CloseAI will handle the conversion automatically
        # The reasoning content will be returned in the standard message format

        # Remove reasoning parameters that are not supported by ChatCompletion API
        prepared_kwargs.pop('reasoning', None)
        prepared_kwargs.pop('reasoning_effort', None)
        prepared_kwargs.pop('verbosity', None)

        return prepared_kwargs

    async def chat_completions_create(self, model, messages, stream: bool = False, **kwargs) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionResponse, None]]:
        """
        Create chat completions using CloseAI's multi-model aggregation interface.
        
        CloseAI automatically handles protocol conversion for different model types,
        so we can use the standard ChatCompletion interface for all models.
        """
        # Prepare kwargs based on model capabilities
        prepared_kwargs = self._prepare_reasoning_kwargs(model, kwargs)
        
        if stream:
            # Reset streaming tool calls state
            self._streaming_tool_calls = {}

            # Clean messages parameter, remove ReasoningContent objects
            cleaned_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    cleaned_msg = msg.copy()
                    # Remove reasoning_content field as CloseAI API doesn't need it as input
                    if 'reasoning_content' in cleaned_msg:
                        cleaned_msg.pop('reasoning_content')
                    cleaned_messages.append(cleaned_msg)
                else:
                    # If it's a Message object, convert to dict and remove reasoning_content
                    if hasattr(msg, 'model_dump'):
                        cleaned_msg = msg.model_dump()
                        if 'reasoning_content' in cleaned_msg:
                            cleaned_msg.pop('reasoning_content')
                        cleaned_messages.append(cleaned_msg)
                    else:
                        cleaned_messages.append(msg)

            response = await self.client.chat.completions.create(
                model=model,
                messages=cleaned_messages,  # Use cleaned messages
                stream=True,
                **prepared_kwargs  # Use prepared kwargs that are compatible with the model
            )
            
            async def stream_generator():
                async for chunk in response:
                    if chunk.choices:
                        yield ChatCompletionResponse(
                            choices=[
                                StreamChoice(
                                    index=choice.index,
                                    delta=ChoiceDelta(
                                        content=choice.delta.content,
                                        role=choice.delta.role,
                                        tool_calls=self._accumulate_and_convert_tool_calls(choice.delta),
                                        reasoning_content=getattr(choice.delta, 'reasoning_content', None)  # Keep original format in streaming
                                    ),
                                    finish_reason=choice.finish_reason
                                )
                                for choice in chunk.choices
                            ],
                            metadata={
                                'id': chunk.id,
                                'created': chunk.created,
                                'model': chunk.model
                            }
                        )
            return stream_generator()
        else:
            # For non-streaming calls, also need to clean messages
            cleaned_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    cleaned_msg = msg.copy()
                    if 'reasoning_content' in cleaned_msg:
                        cleaned_msg.pop('reasoning_content')
                    cleaned_messages.append(cleaned_msg)
                else:
                    if hasattr(msg, 'model_dump'):
                        cleaned_msg = msg.model_dump()
                        if 'reasoning_content' in cleaned_msg:
                            cleaned_msg.pop('reasoning_content')
                        cleaned_messages.append(cleaned_msg)
                    else:
                        cleaned_messages.append(msg)

            response = await self.client.chat.completions.create(
                model=model,
                messages=cleaned_messages,  # Use cleaned messages
                stream=False,
                **prepared_kwargs  # Use prepared kwargs that are compatible with the model
            )
            return ChatCompletionResponse(
                choices=[
                    Choice(
                        index=choice.index,
                        message=Message(
                            content=choice.message.content,
                            role=choice.message.role,
                            tool_calls=self._convert_tool_calls(choice.message.tool_calls) if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls else None,
                            refusal=None,
                            reasoning_content=self._convert_reasoning_content(getattr(choice.message, 'reasoning_content', None))
                        ),
                        finish_reason=choice.finish_reason
                    )
                    for choice in response.choices
                ],
                metadata={
                    "id": response.id,
                    "created": response.created,
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
            )

    def _convert_reasoning_content(self, reasoning_content):
        """Convert CloseAI reasoning_content to ReasoningContent object."""
        if not reasoning_content:
            return None

        return ReasoningContent(
            thinking=reasoning_content,
            provider="closeai",
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
