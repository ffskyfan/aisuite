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

        # Map Responses function_call item.id (fc_*) to call_id (call_*) for next-turn outputs
        self._responses_tool_call_ids: Dict[str, str] = {}

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
                role = msg_dict.get('role')

                # If this message carries Responses raw output via reasoning_content, append it directly
                rc = msg_dict.get('reasoning_content')
                raw_output = None
                if rc is not None:
                    try:
                        if hasattr(rc, 'raw_data') and isinstance(rc.raw_data, dict):
                            raw_output = rc.raw_data.get('output')
                        elif isinstance(rc, dict):
                            raw_output = rc.get('raw_data', {}).get('output')
                    except Exception:
                        raw_output = None
                if raw_output:
                    # Append the exact previous Responses output items (reasoning/message/function_call, etc.)
                    input_items.extend(raw_output)
                    # Do not add a separate assistant message to avoid duplication
                    continue

                # Convert assistant tool_calls into Responses function_call items (Chat-style)
                if role == 'assistant' and 'tool_calls' in msg_dict and msg_dict['tool_calls']:
                    # Preserve assistant textual content if present
                    text = msg_dict.get('content')
                    if text:
                        input_items.append({'role': 'assistant', 'content': text})
                    # Convert each tool_call
                    for tc in msg_dict['tool_calls']:
                        fn = tc.get('function', {})
                        call_id = tc.get('id')
                        name = fn.get('name')
                        arguments = fn.get('arguments', '')
                        input_items.append({
                            'type': 'function_call',
                            'call_id': call_id,
                            'name': name,
                            'arguments': arguments
                        })
                    continue  # Done with this assistant message

                # Convert tool messages into function_call_output items
                if role == 'tool':
                    input_items.append({
                        'type': 'function_call_output',
                        'call_id': msg_dict.get('tool_call_id'),
                        'output': msg_dict.get('content', '')
                    })
                    continue

                # Skip function role messages (unsupported)
                if role == 'function':
                    continue

                # Remove fields that Responses API doesn't support on message objects
                for field in ['reasoning_content', 'tool_calls', 'tool_call_id']:
                    msg_dict.pop(field, None)
                input_items.append(msg_dict)

            else:
                # Handle Message objects
                if hasattr(msg, 'model_dump'):
                    msg_dict = msg.model_dump()
                    role = msg_dict.get('role')

                    # If this message carries Responses raw output via reasoning_content, append it directly
                    rc = msg_dict.get('reasoning_content')
                    raw_output = None
                    if rc is not None:
                        try:
                            if hasattr(rc, 'raw_data') and isinstance(rc.raw_data, dict):
                                raw_output = rc.raw_data.get('output')
                            elif isinstance(rc, dict):
                                raw_output = rc.get('raw_data', {}).get('output')
                        except Exception:
                            raw_output = None
                    if raw_output:
                        input_items.extend(raw_output)
                        continue

                    if role == 'assistant' and 'tool_calls' in msg_dict and msg_dict['tool_calls']:
                        text = msg_dict.get('content')
                        if text:
                            input_items.append({'role': 'assistant', 'content': text})
                        for tc in msg_dict['tool_calls']:
                            fn = tc.get('function', {})
                            call_id = tc.get('id')
                            name = fn.get('name')
                            arguments = fn.get('arguments', '')
                            input_items.append({
                                'type': 'function_call',
                                'call_id': call_id,
                                'name': name,
                                'arguments': arguments
                            })
                        continue

                    if role == 'tool':
                        input_items.append({
                            'type': 'function_call_output',
                            'call_id': msg_dict.get('tool_call_id'),
                            'output': msg_dict.get('content', '')
                        })
                        continue

                    if role == 'function':
                        continue

                    for field in ['reasoning_content', 'tool_calls', 'tool_call_id']:
                        msg_dict.pop(field, None)
                    input_items.append(msg_dict)
                else:
                    # Fallback: append as-is if not a dict-like message
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
            # Reset streaming tool calls state for Responses API too
            self._streaming_tool_calls = {}

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
                    # 透传 strict（若用户提供）
                    if 'strict' in function_def:
                        converted_tool['strict'] = function_def['strict']
                        # 若 strict 为 True，自动补充 additionalProperties=false 以满足严格模式要求
                        try:
                            params = converted_tool.get('parameters') or {}
                            if function_def['strict'] and isinstance(params, dict) and 'additionalProperties' not in params:
                                params['additionalProperties'] = False
                                converted_tool['parameters'] = params
                        except Exception:
                            pass
                    converted_tools.append(converted_tool)
                else:
                    # Keep as is for other formats
                    converted_tools.append(tool)
            else:
                # Non-dict tool, keep as is
                converted_tools.append(tool)

        return converted_tools

    def _accumulate_responses_tool_calls(self, chunk) -> Optional[List[ChatCompletionMessageToolCall]]:
        """
        Accumulate tool call arguments from Responses API streaming events.

        This handles response.function_call_arguments.delta events and builds up
        the complete tool call arguments incrementally.
        """
        if not hasattr(chunk, 'item_id') or not hasattr(chunk, 'delta'):
            return None

        item_id = chunk.item_id
        delta = chunk.delta

        # Initialize tool call data if not exists
        if item_id not in self._streaming_tool_calls:
            # We need to get the function name from a previous event
            # For now, we'll initialize with placeholder and update later
            self._streaming_tool_calls[item_id] = {
                "id": item_id,
                "type": "function",
                "function": {
                    "name": "",  # Will be filled from response.output_item.added
                    "arguments": ""
                }
            }

        # Accumulate arguments
        self._streaming_tool_calls[item_id]["function"]["arguments"] += delta

        # Don't return anything yet - wait for completion
        return None

    def _handle_responses_output_item_added(self, chunk) -> None:
        """
        Handle response.output_item.added events to extract function names.
        """
        if (hasattr(chunk, 'item') and
            hasattr(chunk.item, 'type') and
            chunk.item.type == 'function_call' and
            hasattr(chunk.item, 'name')):

            # Get function name and ID
            item_name = chunk.item.name
            item_id = getattr(chunk.item, 'id', None)

            # Initialize or update tool call data
            if item_id:
                if item_id not in self._streaming_tool_calls:
                    self._streaming_tool_calls[item_id] = {
                        "id": item_id,
                        "type": "function",
                        "function": {
                            "name": item_name,
                            "arguments": ""
                        }
                    }
                else:
                    # Update existing entry with function name
                    self._streaming_tool_calls[item_id]["function"]["name"] = item_name

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

        Based on OpenAI Responses API streaming documentation:
        - response.created: Response started
        - response.output_text.delta: Text content delta (key event for content)
        - response.function_call_arguments.delta: Function call arguments delta
        - response.completed: Response finished
        """
        content = None
        reasoning_content = None
        finish_reason = None
        role = None
        tool_calls = None

        # Handle different event types based on official documentation
        chunk_type = getattr(chunk, 'type', None)

        if chunk_type == 'response.output_text.delta':
            # Text delta event - contains the actual content increments
            if hasattr(chunk, 'delta'):
                content = chunk.delta
                role = "assistant"

        elif chunk_type == 'response.output_item.added':
            # Output item added - set role for message items and handle function calls
            if hasattr(chunk, 'item'):
                item = chunk.item
                if hasattr(item, 'role'):
                    role = item.role
                elif hasattr(item, 'type') and item.type == 'message':
                    role = "assistant"
                elif hasattr(item, 'type') and item.type == 'function_call':
                    # Handle function call item added
                    self._handle_responses_output_item_added(chunk)
                    role = "assistant"

        elif chunk_type == 'response.function_call_arguments.delta':
            # Function call arguments delta - handle tool calls streaming
            if hasattr(chunk, 'delta') and hasattr(chunk, 'item_id'):
                # Accumulate function call arguments for Responses API
                tool_calls = self._accumulate_responses_tool_calls(chunk)
                if tool_calls:
                    # Return accumulated tool calls when complete
                    return ChatCompletionResponse(
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=ChoiceDelta(
                                    content=None,
                                    role="assistant",
                                    tool_calls=tool_calls,
                                    reasoning_content=None
                                ),
                                finish_reason=None
                            )
                        ],
                        metadata={
                            'id': getattr(chunk, 'response_id', None),
                            'created': getattr(chunk, 'created', None),
                            'model': getattr(chunk, 'model', None)
                        }
                    )

        elif chunk_type == 'response.function_call_arguments.done':
            # Function call arguments completed - finalize tool call
            if hasattr(chunk, 'item_id') and hasattr(chunk, 'arguments'):
                # Force completion of this tool call
                item_id = chunk.item_id
                if item_id in self._streaming_tool_calls:
                    tool_call_data = self._streaming_tool_calls[item_id]
                    tool_call_data["function"]["arguments"] = chunk.arguments

                    # Convert to final format
                    try:
                        mock_tool_call = type('MockToolCall', (), {
                            'id': tool_call_data["id"],
                            'type': tool_call_data["type"],
                            'function': type('MockFunction', (), {
                                'name': tool_call_data["function"]["name"],
                                'arguments': tool_call_data["function"]["arguments"]
                            })()
                        })()

                        converted = self._convert_tool_calls([mock_tool_call])
                        if converted:
                            # Remove completed tool call
                            del self._streaming_tool_calls[item_id]

                            return ChatCompletionResponse(
                                choices=[
                                    StreamChoice(
                                        index=0,
                                        delta=ChoiceDelta(
                                            content=None,
                                            role="assistant",
                                            tool_calls=converted,
                                            reasoning_content=None
                                        ),
                                        finish_reason=None
                                    )
                                ],
                                metadata={
                                    'id': getattr(chunk, 'response_id', None),
                                    'created': getattr(chunk, 'created', None),
                                    'model': getattr(chunk, 'model', None)
                                }
                            )
                    except Exception as e:
                        # If conversion fails, clean up and continue
                        if item_id in self._streaming_tool_calls:
                            del self._streaming_tool_calls[item_id]

        elif chunk_type == 'response.output_item.done':
            # Output item completed
            if hasattr(chunk, 'item') and hasattr(chunk.item, 'status'):
                if chunk.item.status == 'completed':
                    finish_reason = 'stop'

        elif chunk_type in ['response.completed', 'response.done']:
            # Response completed
            finish_reason = 'stop'

        elif chunk_type == 'error':
            # Error occurred
            finish_reason = 'error'

        # Return streaming chunk - only include non-None values to avoid overwriting
        return ChatCompletionResponse(
            choices=[
                StreamChoice(
                    index=0,
                    delta=ChoiceDelta(
                        content=content,
                        role=role,
                        tool_calls=tool_calls,
                        reasoning_content=reasoning_content
                    ),
                    finish_reason=finish_reason
                )
            ],
            metadata={
                'id': getattr(chunk, 'response_id', None) or getattr(chunk, 'item_id', None),
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
