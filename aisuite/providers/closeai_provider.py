import os
import json
from typing import Union, AsyncGenerator, List, Dict, Any, Optional
from openai import AsyncOpenAI

from aisuite.provider import Provider, LLMError
from aisuite.framework.chat_completion_response import ChatCompletionResponse, Choice, ChoiceDelta, StreamChoice
from aisuite.framework.message import Message, ReasoningContent, ChatCompletionMessageToolCall, Function


class CloseaiProvider(Provider):
    """
    CloseAI provider for aisuite with enhanced Responses API support.

    CloseAI is an OpenAI-compatible API proxy service that provides access to multiple AI models
    through a unified interface. It supports all OpenAI parameters and automatically handles
    protocol conversion between different model providers.

    Key features:
    - Full OpenAI API compatibility (Chat Completions + Responses API)
    - Multi-model aggregation interface
    - Automatic protocol conversion (ChatCompletion ↔ Response, Anthropic, Gemini)
    - Extended timeout support for reasoning models (up to 20 minutes)
    - Load balancing across multiple accounts
    - Intelligent API selection (Responses API for reasoning models, Chat Completions for others)
    - Reasoning summaries conversion to aisuite ReasoningContent format
    - Multi-turn reasoning context support for stateless conversations

    Reasoning Model Support:
    - GPT-5 series: Automatic Responses API usage with reasoning summaries
    - o4 series: Enhanced reasoning support with context preservation
    - o3 series: Traditional reasoning support
    - o1 series: Traditional reasoning support

    Limitations:
    - Does not support stateful interfaces (file, fine-tune, assistants)
    - Response API supports stateless usage only (but with context preservation in ReasoningContent)
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
        # o4 series models support reasoning_effort (including o4-mini, etc.)
        if model.startswith('o4') or model.startswith('o4-'):
            return True
        # GPT-5 series models support reasoning parameter
        if model.startswith('gpt-5'):
            return True
        # Regular GPT models (gpt-4o, gpt-4, etc.) do not support reasoning
        return False

    def _should_use_responses_api(self, model: str, kwargs: Dict[str, Any]) -> bool:
        """
        Determine if we should use Responses API instead of Chat Completions API.

        According to OpenAI docs, GPT-5 and newer reasoning models work better with
        Responses API, especially when using reasoning parameters or when we need
        to pass reasoning items for multi-turn conversations.
        """
        # GPT-5 models are recommended to use Responses API
        if model.startswith('gpt-5'):
            return True

        # o4 series models benefit from Responses API for reasoning summaries
        if model.startswith('o4') or model.startswith('o4-'):
            return True

        # Use Responses API if reasoning parameters are present
        if any(key in kwargs for key in ['reasoning', 'verbosity', 'reasoning_effort', 'reasoning_summary']):
            return True

        # Use Responses API if we have reasoning items in messages (for multi-turn)
        messages = kwargs.get('messages', [])
        for msg in messages:
            if isinstance(msg, dict) and msg.get('reasoning_content'):
                return True
            elif hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                return True

        return False

    def _prepare_reasoning_kwargs(self, model: str, kwargs: Dict[str, Any], use_responses_api: bool = False) -> Dict[str, Any]:
        """
        Prepare reasoning-related kwargs based on model type and API choice.

        CloseAI automatically handles protocol conversion and supports reasoning models
        through both ChatCompletion and Responses interfaces.
        """
        prepared_kwargs = kwargs.copy()

        # If model doesn't support reasoning, remove reasoning-related parameters
        if not self._supports_reasoning(model):
            # Remove reasoning parameters that would cause API errors
            prepared_kwargs.pop('reasoning', None)
            prepared_kwargs.pop('reasoning_effort', None)
            prepared_kwargs.pop('reasoning_summary', None)
            prepared_kwargs.pop('verbosity', None)
            return prepared_kwargs

        # For reasoning models, handle special parameter requirements
        if (model.startswith('gpt-5') or
            model.startswith('o1-') or
            model.startswith('o3') or
            model.startswith('o3-') or
            model.startswith('o4') or
            model.startswith('o4-')):

            # Handle token limit parameters - convert to appropriate format based on API
            max_tokens_value = None

            # Check for max_tokens first (direct from user)
            if 'max_tokens' in prepared_kwargs:
                max_tokens_value = prepared_kwargs.pop('max_tokens')
            # Check for max_completion_tokens (already converted by LLMUnit)
            elif 'max_completion_tokens' in prepared_kwargs:
                max_tokens_value = prepared_kwargs.pop('max_completion_tokens')

            # Apply the appropriate parameter based on API type
            if max_tokens_value is not None:
                if use_responses_api:
                    prepared_kwargs['max_output_tokens'] = max_tokens_value
                else:
                    prepared_kwargs['max_completion_tokens'] = max_tokens_value

            # GPT-5 only supports default temperature (1.0)
            if model.startswith('gpt-5'):
                if 'temperature' in prepared_kwargs and prepared_kwargs['temperature'] != 1.0:
                    prepared_kwargs.pop('temperature')  # Remove non-default temperature

        if use_responses_api:
            # For Responses API, convert Chat Completions parameters to Responses format
            self._convert_to_responses_format(prepared_kwargs)
        else:
            # For Chat Completions API, remove Responses-specific parameters
            prepared_kwargs.pop('reasoning', None)
            prepared_kwargs.pop('reasoning_effort', None)
            prepared_kwargs.pop('reasoning_summary', None)
            prepared_kwargs.pop('verbosity', None)

        return prepared_kwargs

    def _convert_to_responses_format(self, kwargs: Dict[str, Any]) -> None:
        """
        Convert Chat Completions parameters to Responses API format.
        """
        # Convert reasoning_effort to reasoning.effort format
        if 'reasoning_effort' in kwargs:
            effort = kwargs.pop('reasoning_effort')
            if 'reasoning' not in kwargs:
                kwargs['reasoning'] = {}
            kwargs['reasoning']['effort'] = effort

        # Convert reasoning_summary to reasoning.summary format
        if 'reasoning_summary' in kwargs:
            summary = kwargs.pop('reasoning_summary')
            if 'reasoning' not in kwargs:
                kwargs['reasoning'] = {}
            kwargs['reasoning']['summary'] = summary

        # Convert verbosity to text.verbosity format
        if 'verbosity' in kwargs:
            verbosity = kwargs.pop('verbosity')
            kwargs['text'] = {'verbosity': verbosity}

    async def _responses_create(self, model: str, messages: List[Dict[str, Any]], stream: bool = False, **kwargs) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionResponse, None]]:
        """
        Create responses using CloseAI's Responses API.

        This method handles the newer Responses API which is better for reasoning models
        and supports passing reasoning items between turns.
        """
        # Prepare input format for Responses API
        input_items = []

        # Convert messages to Responses API input format
        for msg in messages:
            if isinstance(msg, dict):
                msg_dict = msg.copy()
                # For multi-turn context, we don't need to extract reasoning items
                # Just remove reasoning_content from the message as it's not part of the API input
                if 'reasoning_content' in msg_dict:
                    msg_dict.pop('reasoning_content')

                input_items.append(msg_dict)
            else:
                # Handle Message objects
                if hasattr(msg, 'model_dump'):
                    msg_dict = msg.model_dump()
                    if 'reasoning_content' in msg_dict:
                        msg_dict.pop('reasoning_content')
                    input_items.append(msg_dict)
                else:
                    input_items.append(msg)

        # Note: For CloseAI, we don't need to manually pass reasoning items
        # The service handles reasoning context automatically

        # Prepare kwargs for Responses API
        responses_kwargs = kwargs.copy()
        responses_kwargs.pop('messages', None)  # Remove messages as we use input

        # Convert tools format for Responses API if present
        if 'tools' in responses_kwargs:
            responses_kwargs['tools'] = self._convert_tools_for_responses_api(responses_kwargs['tools'])

        if stream:
            # For streaming, we need to handle the response differently
            response = await self.client.responses.create(
                model=model,
                input=input_items,
                stream=True,
                **responses_kwargs
            )

            async def stream_generator():
                async for chunk in response:
                    # Convert Responses API streaming format to ChatCompletionResponse
                    yield self._convert_responses_stream_chunk(chunk)

            return stream_generator()
        else:
            # Non-streaming response
            response = await self.client.responses.create(
                model=model,
                input=input_items,
                stream=False,
                **responses_kwargs
            )

            return self._convert_responses_response(response)

    def _convert_tools_for_responses_api(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert tools from Chat Completions format to Responses API format.

        Based on the error, Responses API still needs the 'type' field but with different structure.
        Let's try to keep the original format but ensure all required fields are present.
        """
        converted_tools = []

        for tool in tools:
            if isinstance(tool, dict):
                if tool.get('type') == 'function' and 'function' in tool:
                    # Keep the original format but ensure all required fields
                    function_def = tool['function']
                    converted_tool = {
                        'type': 'function',
                        'name': function_def.get('name'),
                        'description': function_def.get('description', ''),
                        'parameters': function_def.get('parameters', {})
                    }
                    converted_tools.append(converted_tool)
                else:
                    # Keep as is for other formats
                    converted_tools.append(tool)
            else:
                # Non-dict tool, keep as is
                converted_tools.append(tool)

        return converted_tools

    async def chat_completions_create(self, model, messages, stream: bool = False, **kwargs) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionResponse, None]]:
        """
        Create chat completions using CloseAI's multi-model aggregation interface.

        CloseAI automatically handles protocol conversion for different model types.
        For newer reasoning models, we use the Responses API for better performance.
        """
        # Check if we should use Responses API
        use_responses_api = self._should_use_responses_api(model, kwargs)

        if use_responses_api:
            # Use Responses API for better reasoning model support
            prepared_kwargs = self._prepare_reasoning_kwargs(model, kwargs, use_responses_api=True)
            return await self._responses_create(model, messages, stream, **prepared_kwargs)

        # Use Chat Completions API for regular models
        prepared_kwargs = self._prepare_reasoning_kwargs(model, kwargs, use_responses_api=False)

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

    def _convert_responses_response(self, response) -> ChatCompletionResponse:
        """
        Convert Responses API response to ChatCompletionResponse format.
        """
        # Extract the main message content and reasoning
        message_content = None
        reasoning_content = None
        finish_reason = "stop"

        # First, try to get content from output_text (most reliable)
        if hasattr(response, 'output_text') and response.output_text:
            message_content = response.output_text

        # Parse the response output for additional information
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'type'):
                    if item.type == 'message':
                        # Extract message content if we don't have it yet
                        if not message_content and hasattr(item, 'content') and item.content:
                            for content_item in item.content:
                                if hasattr(content_item, 'type') and content_item.type == 'output_text':
                                    message_content = content_item.text
                                    break
                        # Get finish reason
                        if hasattr(item, 'status'):
                            finish_reason = "stop" if item.status == "completed" else "incomplete"
                    elif item.type == 'reasoning':
                        # Extract reasoning summary
                        if hasattr(item, 'summary') and item.summary:
                            reasoning_text = ""
                            for summary_item in item.summary:
                                if hasattr(summary_item, 'text'):
                                    reasoning_text += summary_item.text + "\n"

                            if reasoning_text.strip():
                                # Create ReasoningContent with full response data for multi-turn support
                                reasoning_content = ReasoningContent(
                                    thinking=reasoning_text.strip(),
                                    provider="closeai",
                                    raw_data={
                                        "reasoning_items": [item],  # Store reasoning items for next turn
                                        "output": response.output,  # Store full output for context
                                        "response_id": getattr(response, 'id', None)
                                    }
                                )
                        else:
                            # Even if no summary, create reasoning content to indicate reasoning was used
                            reasoning_content = ReasoningContent(
                                thinking="[推理过程已完成，但未提供摘要]",
                                provider="closeai",
                                raw_data={
                                    "reasoning_items": [item],
                                    "output": response.output,
                                    "response_id": getattr(response, 'id', None)
                                }
                            )

        return ChatCompletionResponse(
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        content=message_content,
                        role="assistant",
                        tool_calls=None,  # TODO: Handle tool calls in Responses API
                        refusal=None,
                        reasoning_content=reasoning_content
                    ),
                    finish_reason=finish_reason
                )
            ],
            metadata={
                "id": getattr(response, 'id', None),
                "created": getattr(response, 'created', None),
                "model": getattr(response, 'model', None),
                "usage": self._extract_usage_from_responses(response)
            }
        )

    def _convert_responses_stream_chunk(self, chunk) -> ChatCompletionResponse:
        """
        Convert Responses API streaming chunk to ChatCompletionResponse format.
        """
        content = None
        reasoning_content = None
        finish_reason = None

        # Extract content from streaming chunk
        if hasattr(chunk, 'output') and chunk.output:
            for item in chunk.output:
                if hasattr(item, 'type'):
                    if item.type == 'message' and hasattr(item, 'content'):
                        for content_item in item.content:
                            if hasattr(content_item, 'type') and content_item.type == 'output_text':
                                content = content_item.text
                                break
                    elif item.type == 'reasoning' and hasattr(item, 'summary'):
                        reasoning_text = ""
                        for summary_item in item.summary:
                            if hasattr(summary_item, 'text'):
                                reasoning_text += summary_item.text
                        if reasoning_text:
                            reasoning_content = reasoning_text

        return ChatCompletionResponse(
            choices=[
                StreamChoice(
                    index=0,
                    delta=ChoiceDelta(
                        content=content,
                        role="assistant",
                        tool_calls=None,
                        reasoning_content=reasoning_content
                    ),
                    finish_reason=finish_reason
                )
            ],
            metadata={
                'id': getattr(chunk, 'id', None),
                'created': getattr(chunk, 'created', None),
                'model': getattr(chunk, 'model', None)
            }
        )

    def _extract_usage_from_responses(self, response) -> Optional[Dict[str, Any]]:
        """
        Extract usage information from Responses API response.
        """
        if hasattr(response, 'usage'):
            usage = response.usage
            result = {}

            if hasattr(usage, 'input_tokens'):
                result['prompt_tokens'] = usage.input_tokens
            if hasattr(usage, 'output_tokens'):
                result['completion_tokens'] = usage.output_tokens
            if hasattr(usage, 'total_tokens'):
                result['total_tokens'] = usage.total_tokens
            elif 'prompt_tokens' in result and 'completion_tokens' in result:
                result['total_tokens'] = result['prompt_tokens'] + result['completion_tokens']

            # Add reasoning token details if available
            if hasattr(usage, 'output_tokens_details'):
                details = usage.output_tokens_details
                if hasattr(details, 'reasoning_tokens'):
                    result['reasoning_tokens'] = details.reasoning_tokens

            return result
        return None

    def _convert_reasoning_content(self, reasoning_content):
        """
        Convert CloseAI reasoning_content to ReasoningContent object.

        This handles both Chat Completions API reasoning_content (simple string)
        and Responses API reasoning summaries (structured format).
        """
        if not reasoning_content:
            return None

        # Handle simple string format (from Chat Completions API)
        if isinstance(reasoning_content, str):
            return ReasoningContent(
                thinking=reasoning_content,
                provider="closeai",
                raw_data={"reasoning_content": reasoning_content}
            )

        # Handle structured format (from Responses API)
        if isinstance(reasoning_content, dict):
            thinking_text = ""

            # Extract thinking text from various possible formats
            if 'summary' in reasoning_content:
                if isinstance(reasoning_content['summary'], list):
                    for item in reasoning_content['summary']:
                        if isinstance(item, dict) and 'text' in item:
                            thinking_text += item['text'] + "\n"
                        elif isinstance(item, str):
                            thinking_text += item + "\n"
                elif isinstance(reasoning_content['summary'], str):
                    thinking_text = reasoning_content['summary']
            elif 'thinking' in reasoning_content:
                thinking_text = reasoning_content['thinking']
            elif 'text' in reasoning_content:
                thinking_text = reasoning_content['text']

            return ReasoningContent(
                thinking=thinking_text.strip() if thinking_text else str(reasoning_content),
                provider="closeai",
                raw_data=reasoning_content
            )

        # Fallback: convert to string
        return ReasoningContent(
            thinking=str(reasoning_content),
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
