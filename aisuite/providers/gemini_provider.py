import os
import json
from typing import AsyncGenerator, Union
from aisuite.framework.chat_completion_response import ChatCompletionResponse, Choice, ChoiceDelta, StreamChoice
from aisuite.framework.message import Message, ChatCompletionMessageToolCall, Function, ReasoningContent
from aisuite.provider import Provider 

# Import Google GenAI SDK
from google import genai
from google.genai import types

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

        if response.candidates and response.candidates[0].content.parts:
            reasoning_text_parts = []
            content_text_parts = []

            for part in response.candidates[0].content.parts:
                # Check if the part is a thought and has text
                if getattr(part, 'thought', False) and getattr(part, 'text', None):
                    reasoning_text_parts.append(part.text)
                # Check if the part is a function call
                elif hasattr(part, 'function_call') and part.function_call:
                    try:
                        function = Function(
                            name=part.function_call.name,
                            arguments=json.dumps(part.function_call.args) if hasattr(part.function_call, 'args') else "{}"
                        )

                        tool_call_obj = ChatCompletionMessageToolCall(
                            id=f"call_{part.function_call.name}_{hash(str(part.function_call.args)) % 10000}",
                            function=function,
                            type="function"
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
                reasoning_content = "".join(reasoning_text_parts)

            # Use combined content text or fallback to response.text
            if content_text_parts:
                content = "".join(content_text_parts)

        # 创建 ChatCompletionResponse 对象
        return ChatCompletionResponse(
            choices=[
                Choice(
                    index=0,  # Gemini 通常只返回一个选项
                    message=Message(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls,
                        refusal=None,
                        reasoning_content=reasoning_content
                    ),  # 使用 Message 对象包装响应内容
                    finish_reason=response.candidates[0].finish_reason if response.candidates else None
                )
            ],
            metadata={
                "model": response.model_version,  # 模型名称
                # Gemini API 可能不提供这些字段，所以我们设置为 None
                "id": None,
                "created": None,
                "usage": None  # Gemini API 目前不提供 token 使用统计
            }
        )



class GeminiProvider(Provider):

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
            raw_data={
                "thought_parts": thought_parts,
                "thinking_text": thinking_text
            }
        )
    def __init__(self, **kwargs):
        """Initialize the Gemini provider with API key and client."""
        api_key = os.environ.get("GEMINI_API_KEY") or kwargs.get("api_key")
        if api_key is None:
            raise RuntimeError("GEMINI_API_KEY is required for GeminiProvider")
        # Initialize the GenAI client for Gemini (non-Vertex usage)
        self.client = genai.Client(api_key=api_key)

        # State for accumulating streaming tool calls
        self._streaming_tool_calls = {}



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

        # Preprocess messages to ensure Gemini compatibility
        messages = self._preprocess_messages_for_gemini(messages)

        # Determine if streaming
        stream = kwargs.get("stream", False)
        if "stream" in kwargs:
            kwargs.pop("stream")
        # Map model name to proper format
        model_id = model
        # Separate system message (if present) for config
        config_kwargs = {}
        # Add this for thinking_config for 2.5 series models
        if "2.5" in model_id:  # Heuristic check for 2.5 series models
            # Ensure that types.ThinkingConfig and types.GenerateContentConfig are correctly referenced/imported
            # Assuming 'types' is already imported from google.genai
            config_kwargs["thinking_config"] = types.ThinkingConfig(include_thoughts=True)

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
        if "tools" in kwargs:
            openai_tools = kwargs.pop("tools")
            gemini_tools = self._convert_tool_spec(openai_tools)
            if gemini_tools:
                config_kwargs["tools"] = gemini_tools

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
                        # Extract function name from tool_call_id if possible
                        # Format: call_function_name_hash -> function_name
                        tool_call_id = msg["tool_call_id"]
                        function_name = "unknown_function"
                        if tool_call_id.startswith("call_"):
                            parts = tool_call_id.split("_")
                            if len(parts) >= 3:
                                function_name = "_".join(parts[1:-1])  # Everything between "call_" and the hash

                        # Create function response part
                        function_response_part = types.Part.from_function_response(
                            name=function_name,
                            response={"result": msg["content"]}
                        )
                        history_msgs.append(types.Content(role="user", parts=[function_response_part]))
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
                        parts = []

                        # Add text content if present
                        if msg.get("content"):
                            parts.append(types.Part.from_text(text=msg["content"]))

                        # Add function calls
                        for tool_call in msg["tool_calls"]:
                            if tool_call.get("type") == "function" and "function" in tool_call:
                                func = tool_call["function"]
                                # Parse arguments if they're a string
                                args = func.get("arguments", "{}")
                                if isinstance(args, str):
                                    try:
                                        import json
                                        args = json.loads(args)
                                    except json.JSONDecodeError:
                                        args = {}

                                # Create function call part
                                function_call_part = types.Part.from_function_call(
                                    name=func["name"],
                                    args=args
                                )
                                parts.append(function_call_part)

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
                            tool_call_id = msg["tool_call_id"]
                            function_name = "unknown_function"
                            if tool_call_id.startswith("call_"):
                                parts = tool_call_id.split("_")
                                if len(parts) >= 3:
                                    function_name = "_".join(parts[1:-1])

                            function_response_part = types.Part.from_function_response(
                                name=function_name,
                                response={"result": msg["content"]}
                            )
                            contents.append(types.Content(role="user", parts=[function_response_part]))
                        else:
                            tool_content = f"Tool result: {msg['content']}"
                            part = types.Part.from_text(text=tool_content)
                            contents.append(types.Content(role="user", parts=[part]))
                    elif role in ("user", "assistant"):
                        gemini_role = "model" if role == "assistant" else "user"

                        if role == "assistant" and isinstance(msg.get("tool_calls"), list):
                            parts = []
                            if msg.get("content"):
                                parts.append(types.Part.from_text(text=msg["content"]))

                            for tool_call in msg["tool_calls"]:
                                if tool_call.get("type") == "function" and "function" in tool_call:
                                    func = tool_call["function"]
                                    args = func.get("arguments", "{}")
                                    if isinstance(args, str):
                                        try:
                                            import json
                                            args = json.loads(args)
                                        except json.JSONDecodeError:
                                            args = {}

                                    function_call_part = types.Part.from_function_call(
                                        name=func["name"],
                                        args=args
                                    )
                                    parts.append(function_call_part)

                            if parts:
                                contents.append(types.Content(role=gemini_role, parts=parts))
                        else:
                            if msg.get("content"):
                                part = types.Part.from_text(text=msg["content"])
                                contents.append(types.Content(role=gemini_role, parts=[part]))





                # Use streaming generation
                async def stream_generator():
                    response_id = None
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

                        if chunk.candidates and chunk.candidates[0].content.parts:
                            for part in chunk.candidates[0].content.parts:
                                if getattr(part, 'thought', False) and getattr(part, 'text', None):
                                    reasoning_text_parts.append(part.text)
                                elif hasattr(part, 'function_call') and part.function_call:
                                    mock_delta = type('MockDelta', (), {
                                        'function_call': part.function_call
                                    })()
                                    tool_calls = self._accumulate_and_convert_tool_calls(mock_delta)
                                elif getattr(part, 'text', None):
                                    content_text_parts.append(part.text)

                        if reasoning_text_parts:
                            current_chunk_reasoning = "".join(reasoning_text_parts)

                        if content_text_parts:
                            current_chunk_text = "".join(content_text_parts)

                        yield ChatCompletionResponse(
                            choices=[
                                StreamChoice(
                                    index=0,
                                    delta=ChoiceDelta(
                                        content=current_chunk_text if current_chunk_text else None,
                                        role="assistant" if current_chunk_text or tool_calls else None,
                                        tool_calls=tool_calls,
                                        reasoning_content=current_chunk_reasoning
                                    ),
                                    finish_reason=None
                                )
                            ],
                            metadata={
                                'id': response_id,
                                'created': None,
                                'model': model_id
                            }
                        )
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
                            tool_call_id = msg["tool_call_id"]
                            function_name = "unknown_function"
                            if tool_call_id.startswith("call_"):
                                parts = tool_call_id.split("_")
                                if len(parts) >= 3:
                                    function_name = "_".join(parts[1:-1])

                            function_response_part = types.Part.from_function_response(
                                name=function_name,
                                response={"result": msg["content"]}
                            )
                            contents.append(types.Content(role="user", parts=[function_response_part]))
                        else:
                            tool_content = f"Tool result: {msg['content']}"
                            part = types.Part.from_text(text=tool_content)
                            contents.append(types.Content(role="user", parts=[part]))
                    elif role in ("user", "assistant"):
                        gemini_role = "model" if role == "assistant" else "user"

                        if role == "assistant" and isinstance(msg.get("tool_calls"), list):
                            parts = []
                            if msg.get("content"):
                                parts.append(types.Part.from_text(text=msg["content"]))

                            for tool_call in msg["tool_calls"]:
                                if tool_call.get("type") == "function" and "function" in tool_call:
                                    func = tool_call["function"]
                                    args = func.get("arguments", "{}")
                                    if isinstance(args, str):
                                        try:
                                            import json
                                            args = json.loads(args)
                                        except json.JSONDecodeError:
                                            args = {}

                                    function_call_part = types.Part.from_function_call(
                                        name=func["name"],
                                        args=args
                                    )
                                    parts.append(function_call_part)

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
            # Reset streaming tool calls state
            self._streaming_tool_calls = {}

            # Streaming response: return a generator yielding ChatCompletionResponse objects
            async def stream_generator():
                response_id = None  # We'll use the first valid chunk's id for all chunks
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

                    if chunk.candidates and chunk.candidates[0].content.parts: # Ensure candidates and parts exist
                        for part in chunk.candidates[0].content.parts:
                            # Check if the part is a thought and has text
                            if getattr(part, 'thought', False) and getattr(part, 'text', None):
                                reasoning_text_parts.append(part.text)
                            # Check if the part is a function call
                            elif hasattr(part, 'function_call') and part.function_call:
                                # Create a mock delta object for consistency with other providers
                                mock_delta = type('MockDelta', (), {
                                    'function_call': part.function_call
                                })()
                                tool_calls = self._accumulate_and_convert_tool_calls(mock_delta)
                            # Else, if it's not a thought but has text, it's regular content
                            elif getattr(part, 'text', None):
                                content_text_parts.append(part.text)

                    if reasoning_text_parts:
                        current_chunk_reasoning = "".join(reasoning_text_parts)

                    if content_text_parts:
                        current_chunk_text = "".join(content_text_parts)

                    yield ChatCompletionResponse(
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=ChoiceDelta(
                                    content=current_chunk_text if current_chunk_text else None,
                                    role="assistant" if current_chunk_text or tool_calls else None,
                                    tool_calls=tool_calls,
                                    reasoning_content=current_chunk_reasoning
                                ),
                                finish_reason=None  # Gemini doesn't provide per-chunk finish reason
                            )
                        ],
                        metadata={
                            'id': response_id, # Use the captured ID (or None if never found)
                            'created': None,  # Gemini doesn't provide timestamp
                            'model': model_id
                        }
                    )
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
            # Gemini function calls are typically complete in a single chunk
            function = Function(
                name=function_call.name,
                arguments=json.dumps(function_call.args) if hasattr(function_call, 'args') else "{}"
            )

            tool_call_obj = ChatCompletionMessageToolCall(
                id=f"call_{function_call.name}_{hash(str(function_call.args)) % 10000}",  # Generate a unique ID
                function=function,
                type="function"
            )

            return [tool_call_obj]

        except Exception:
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

                    # Convert to framework format
                    function = Function(
                        name=tool_call_data["function"]["name"],
                        arguments=tool_call_data["function"]["arguments"]
                    )

                    tool_call_obj = ChatCompletionMessageToolCall(
                        id=tool_call_data["id"],
                        function=function,
                        type="function"
                    )

                    complete_tool_calls.append(tool_call_obj)

                    # Remove completed tool call from accumulator
                    del self._streaming_tool_calls[index]

                except json.JSONDecodeError:
                    # Arguments are not complete yet, continue accumulating
                    continue

        return complete_tool_calls if complete_tool_calls else None
