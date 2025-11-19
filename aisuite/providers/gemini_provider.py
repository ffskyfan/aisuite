import os
import json
from typing import AsyncGenerator, Union
from aisuite.framework.chat_completion_response import ChatCompletionResponse, Choice, ChoiceDelta, StreamChoice
from aisuite.framework.message import Message, ChatCompletionMessageToolCall, Function, ReasoningContent
from aisuite.framework.stop_reason import stop_reason_manager
from aisuite.provider import Provider

# Import Google GenAI SDK
from google import genai
from google.genai import types


def _normalize_gemini_usage(usage_obj):
    """Normalize Gemini usage/usageMetadata to AISuite standard dict.

    Maps promptTokenCount/prompt_token_count -> prompt_tokens,
    candidatesTokenCount/candidates_token_count -> completion_tokens,
    totalTokenCount/total_token_count -> total_tokens.
    """
    if not usage_obj:
        return None

    # Try to extract as plain dict
    if hasattr(usage_obj, "model_dump"):
        data = usage_obj.model_dump()
    elif isinstance(usage_obj, dict):
        data = usage_obj
    else:
        data = {}
        for attr in (
            "prompt_token_count",
            "candidates_token_count",
            "total_token_count",
            "promptTokenCount",
            "candidatesTokenCount",
            "totalTokenCount",
        ):
            if hasattr(usage_obj, attr):
                data[attr] = getattr(usage_obj, attr)

    prompt_tokens = data.get("prompt_token_count") or data.get("promptTokenCount")
    completion_tokens = data.get("candidates_token_count") or data.get("candidatesTokenCount")
    total_tokens = (
        data.get("total_token_count")
        or data.get("totalTokenCount")
        or (
            (prompt_tokens or 0) + (completion_tokens or 0)
            if prompt_tokens is not None and completion_tokens is not None
            else None
        )
    )

    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None

    return {
        "prompt_tokens": int(prompt_tokens) if prompt_tokens is not None else None,
        "completion_tokens": int(completion_tokens) if completion_tokens is not None else None,
        "total_tokens": int(total_tokens) if total_tokens is not None else None,
    }


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

        candidate = response.candidates[0] if response.candidates else None
        candidate_content = getattr(candidate, "content", None)
        candidate_parts = getattr(candidate_content, "parts", None)

        if candidate_parts:
            reasoning_text_parts = []
            content_text_parts = []

            for part in candidate_parts:
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

        # Get finish_reason for stop_info creation
        finish_reason = response.candidates[0].finish_reason if response.candidates else None

        # Create stop_info (we need to import stop_reason_manager at the top level)
        stop_info = None
        if finish_reason:
            choice_data = {
                "content": {
                    "parts": [{"text": content}] if content else []
                }
            }
            # Add tool calls to choice_data if present
            if tool_calls:
                choice_data["content"]["parts"].extend([{"functionCall": tc} for tc in tool_calls])

            # Import stop_reason_manager here to avoid circular imports
            from aisuite.framework.stop_reason import stop_reason_manager
            stop_info = stop_reason_manager.map_stop_reason("gemini", finish_reason, {
                "has_content": bool(content or tool_calls),
                "content_length": len(content) if content else 0,
                "tool_calls_count": len(tool_calls) if tool_calls else 0,
                "finish_reason": finish_reason,
                "model": response.model_version,
                "provider": "gemini"
            })

        # 提取 usage 元数据（token 统计）
        usage_metadata = getattr(response, "usage_metadata", None) or getattr(response, "usageMetadata", None)
        usage = _normalize_gemini_usage(usage_metadata)

        # 创建 ChatCompletionResponse 对象
        metadata = {
            "model": response.model_version,  # 模型名称
            # Gemini API 目前不会返回这些字段
            "id": None,
            "created": None,
        }
        if usage:
            metadata["usage"] = usage

        return ChatCompletionResponse(
            choices=[
                Choice(
                    index=0,  # Gemini 通常只返回一个选项
                    message=Message(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls,
                        refusal=None,
                        reasoning_content=reasoning_content,
                    ),  # 使用 Message 对象包装响应内容
                    finish_reason=finish_reason,
                    stop_info=stop_info,
                )
            ],
            metadata=metadata,
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

        # Track accumulated content for streaming responses
        # Used to provide accurate metadata in stop_info
        self._stream_content_length = 0
        self._stream_tool_calls_count = 0

    def _is_gemini_3_model(self, model_id: str) -> bool:
        """Heuristic check for Gemini 3 series models.

        We use a simple substring match so it works for both plain
        "gemini-3-..." and provider-prefixed names like "google/gemini-3-...".
        """
        if not model_id:
            return False
        return "gemini-3" in model_id


    def _create_stop_info(self, finish_reason: str, choice_data: dict = None, model: str = None) -> dict:
        """Create StopInfo from Gemini finish_reason."""
        if not finish_reason:
            return None

        # Analyze choice data to determine content presence
        has_content = False
        content_length = 0
        tool_calls_count = 0

        if choice_data:
            # Check content from Gemini candidate
            content = choice_data.get("content")
            if content:
                # Gemini content has parts array
                parts = content.get("parts", [])
                for part in parts:
                    if part.get("text"):
                        has_content = True
                        content_length += len(part.get("text", ""))
                    elif part.get("functionCall"):
                        tool_calls_count += 1
                        has_content = True  # Tool calls count as content

        metadata = {
            "has_content": has_content,
            "content_length": content_length,
            "tool_calls_count": tool_calls_count,
            "finish_reason": finish_reason,
            "model": model,
            "provider": "gemini"
        }

        # Map Gemini finish_reason to standard StopReason
        return stop_reason_manager.map_stop_reason("gemini", finish_reason, metadata)



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
        # Default thinking_config for 2.5 series models (thought summaries)
        if "2.5" in model_id:  # Heuristic check for 2.5 series models
            config_kwargs["thinking_config"] = types.ThinkingConfig(include_thoughts=True)

        # Reasoning / thinking control for Gemini 3 models
        # Accept OpenAI-style reasoning_effort / reasoning, and direct thinking_level
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        thinking_level = kwargs.pop("thinking_level", None)

        # Also support OpenAI-style `reasoning={"effort": "low"}` for convenience
        reasoning = kwargs.pop("reasoning", None)
        if reasoning is not None and reasoning_effort is None:
            if isinstance(reasoning, dict) and "effort" in reasoning:
                reasoning_effort = reasoning["effort"]
            elif isinstance(reasoning, str):
                reasoning_effort = reasoning

        if self._is_gemini_3_model(model_id):
            # Map to Gemini 3 thinking_level = "low" | "high"
            level = None
            if isinstance(thinking_level, str) and thinking_level:
                level = thinking_level.lower()
            elif isinstance(reasoning_effort, str) and reasoning_effort:
                eff = reasoning_effort.lower()
                if eff in {"minimal", "low", "none", "disable"}:
                    level = "low"
                elif eff in {"medium", "high"}:
                    level = "high"

            if level in ("low", "high"):
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level)

            # For Gemini 3, default temperature to 1.0 if not explicitly set
            if "temperature" not in kwargs and "temperature" not in config_kwargs:
                config_kwargs["temperature"] = 1.0

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
                # Reset streaming state
                self._streaming_tool_calls = {}
                self._stream_content_length = 0
                self._stream_tool_calls_count = 0

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
                    stream_usage = None
                    pending_response = None
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

                        candidate = chunk.candidates[0] if chunk.candidates else None
                        candidate_content = getattr(candidate, "content", None)
                        candidate_parts = getattr(candidate_content, "parts", None)

                        if candidate_parts:
                            for part in candidate_parts:
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
                            # Accumulate content length for accurate stop_info metadata
                            self._stream_content_length += len(current_chunk_text)

                        # Accumulate tool calls count
                        if tool_calls:
                            self._stream_tool_calls_count += len(tool_calls)

                        # Capture usage metadata from chunk if available
                        usage_metadata = getattr(chunk, "usage_metadata", None) or getattr(chunk, "usageMetadata", None)
                        if usage_metadata:
                            normalized_usage = _normalize_gemini_usage(usage_metadata)
                            if normalized_usage:
                                stream_usage = normalized_usage

                        # Check for finish_reason in chunk
                        finish_reason = None
                        stop_info = None
                        if chunk.candidates and chunk.candidates[0].finish_reason:
                            finish_reason = chunk.candidates[0].finish_reason
                            # Use accumulated values for accurate stop_info metadata
                            stop_metadata = {
                                "has_content": self._stream_content_length > 0 or self._stream_tool_calls_count > 0,
                                "content_length": self._stream_content_length,
                                "tool_calls_count": self._stream_tool_calls_count,
                                "finish_reason": finish_reason,
                                "model": model_id,
                                "provider": "gemini"
                            }
                            stop_info = stop_reason_manager.map_stop_reason("gemini", finish_reason, stop_metadata)

                        response_metadata = {
                            'id': response_id,
                            'created': None,
                            'model': model_id
                        }

                        current_response = ChatCompletionResponse(
                            choices=[
                                StreamChoice(
                                    index=0,
                                    delta=ChoiceDelta(
                                        content=current_chunk_text if current_chunk_text else None,
                                        role="assistant" if current_chunk_text or tool_calls else None,
                                        tool_calls=tool_calls,
                                        reasoning_content=current_chunk_reasoning
                                    ),
                                    finish_reason=finish_reason,
                                    stop_info=stop_info
                                )
                            ],
                            metadata=response_metadata
                        )

                        if pending_response is not None:
                            yield pending_response

                        pending_response = current_response

                    if pending_response is not None:
                        if stream_usage:
                            pending_response.metadata["usage"] = stream_usage
                        yield pending_response
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
            # Reset streaming state
            self._streaming_tool_calls = {}
            self._stream_content_length = 0
            self._stream_tool_calls_count = 0

            # Streaming response: return a generator yielding ChatCompletionResponse objects
            async def stream_generator():
                response_id = None  # We'll use the first valid chunk's id for all chunks
                stream_usage = None
                pending_response = None
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

                    candidate = chunk.candidates[0] if chunk.candidates else None
                    candidate_content = getattr(candidate, "content", None)
                    candidate_parts = getattr(candidate_content, "parts", None)

                    if candidate_parts: # Ensure candidates and parts exist
                        for part in candidate_parts:
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
                        # Accumulate content length for accurate stop_info metadata
                        self._stream_content_length += len(current_chunk_text)

                    # Accumulate tool calls count
                    if tool_calls:
                        self._stream_tool_calls_count += len(tool_calls)

                    # Capture usage metadata from chunk if available
                    usage_metadata = getattr(chunk, "usage_metadata", None) or getattr(chunk, "usageMetadata", None)
                    if usage_metadata:
                        normalized_usage = _normalize_gemini_usage(usage_metadata)
                        if normalized_usage:
                            stream_usage = normalized_usage

                    # Check for finish_reason in chunk
                    finish_reason = None
                    stop_info = None
                    if chunk.candidates and chunk.candidates[0].finish_reason:
                        finish_reason = chunk.candidates[0].finish_reason
                        # Use accumulated values for accurate stop_info metadata
                        stop_metadata = {
                            "has_content": self._stream_content_length > 0 or self._stream_tool_calls_count > 0,
                            "content_length": self._stream_content_length,
                            "tool_calls_count": self._stream_tool_calls_count,
                            "finish_reason": finish_reason,
                            "model": model_id,
                            "provider": "gemini"
                        }
                        stop_info = stop_reason_manager.map_stop_reason("gemini", finish_reason, stop_metadata)

                    response_metadata = {
                        'id': response_id, # Use the captured ID (or None if never found)
                        'created': None,  # Gemini doesn't provide timestamp
                        'model': model_id
                    }

                    current_response = ChatCompletionResponse(
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=ChoiceDelta(
                                    content=current_chunk_text if current_chunk_text else None,
                                    role="assistant" if current_chunk_text or tool_calls else None,
                                    tool_calls=tool_calls,
                                    reasoning_content=current_chunk_reasoning
                                ),
                                finish_reason=finish_reason,
                                stop_info=stop_info
                            )
                        ],
                        metadata=response_metadata
                    )

                    if pending_response is not None:
                        yield pending_response

                    pending_response = current_response

                if pending_response is not None:
                    if stream_usage:
                        pending_response.metadata["usage"] = stream_usage
                    yield pending_response
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
