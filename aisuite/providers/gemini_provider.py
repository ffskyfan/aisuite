import os
import json
from typing import AsyncGenerator, Union
from aisuite.framework.chat_completion_response import ChatCompletionResponse, Choice, ChoiceDelta, StreamChoice
from aisuite.framework.message import Message, ChatCompletionMessageToolCall, Function
from aisuite.provider import Provider, LLMError

# Import Google GenAI SDK
from google import genai
from google.genai import types

class GeminiMessageConverter:

    @staticmethod
    def to_gemini_request(self, conversation):
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
    def __init__(self, **kwargs):
        """Initialize the Gemini provider with API key and client."""
        api_key = os.environ.get("GEMINI_API_KEY") or kwargs.get("api_key")
        if api_key is None:
            raise RuntimeError("GEMINI_API_KEY is required for GeminiProvider")
        # Initialize the GenAI client for Gemini (non-Vertex usage)
        self.client = genai.Client(api_key=api_key)

        # State for accumulating streaming tool calls
        self._streaming_tool_calls = {}
    
    
    async def chat_completions_create(self, model: str, messages: list, **kwargs) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionResponse, None]]:
        """Create a chat completion (single-turn or streaming) using a Gemini model."""
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
        # (Ignore any remaining kwargs that are not applicable for now)
        # Create config object if any config parameters were specified
        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        # Prepare conversation history (all messages except the final prompt)
        history_msgs = []
        last_user_message = None
        if messages:
            # Assume the last message is the user prompt we want to answer
            if messages[-1]["role"] == "user":
                last_user_message = messages[-1]["content"]
                convo_history = messages[:-1]
            else:
                # If last message is not user (edge case), treat all as history
                convo_history = messages
                last_user_message = None
            # Convert history messages to Content objects
            for msg in convo_history:
                role = msg["role"]
                if role not in ("user", "assistant"):
                    # Skip any non-user/assistant (e.g., system already handled)
                    continue
                # Map AISuite role to Gemini role (Gemini expects "user" or "model")
                gemini_role = "model" if role == "assistant" else "user"
                part = types.Part.from_text(text=msg["content"])
                history_msgs.append(types.Content(role=gemini_role, parts=[part]))
        # Create a new chat session with history and config (if any)
        chat = self.client.chats.create(model=model_id, config=config, history=history_msgs if history_msgs else None)
        if last_user_message is None:
            # No user prompt to send (no completion to generate)
            return None
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

                    if chunk.candidates: # Ensure candidates exist
                        for part in chunk.candidates[0].content.parts:
                            # Check if the part is a thought and has text
                            if getattr(part, 'thought', False) and getattr(part, 'text', None):
                                reasoning_text_parts.append(part.text)
                            # Check if the part is a function call
                            elif hasattr(part, 'function_call') and part.function_call:
                                tool_calls = self._process_gemini_function_call(part.function_call)
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
    
    def chat_create(self, model: str, messages: list = None, **kwargs):
        """Create a persistent chat session (ChatSession) for multi-turn conversations."""
        model_id = model
        # Separate system message for config if provided in messages
        config_kwargs = {}
        history_msgs = []
        if messages:
            if messages and messages[0].get("role") == "system":
                config_kwargs["system_instruction"] = messages[0]["content"]
                messages = messages[1:]
            # We can also extract generation params from kwargs for config (reuse logic)
        # Merge any generation params into config (reusing logic from chat_completions_create)
        if "max_tokens" in kwargs or "max_output_tokens" in kwargs:
            max_toks = kwargs.pop("max_output_tokens", None) or kwargs.pop("max_tokens", None)
            config_kwargs["max_output_tokens"] = max_toks
        for param in ["temperature", "top_p", "top_k", "candidate_count",
                      "presence_penalty", "frequency_penalty", "seed"]:
            if param in kwargs:
                config_kwargs[param] = kwargs.pop(param)
        if "stop_sequences" in kwargs or "stop" in kwargs:
            stop_seq = kwargs.pop("stop_sequences", None) or kwargs.pop("stop", None)
            if stop_seq:
                config_kwargs["stop_sequences"] = [stop_seq] if isinstance(stop_seq, str) else stop_seq
        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        # Convert any provided conversation messages to Content objects for history
        if messages:
            for msg in messages:
                if msg["role"] in ("user", "assistant"):
                    # Map AISuite role to Gemini role (Gemini expects "user" or "model")
                    gemini_role = "model" if msg["role"] == "assistant" else "user"
                    part = types.Part.from_text(text=msg["content"])
                    history_msgs.append(types.Content(role=gemini_role, parts=[part]))
        # Create and return the chat session object
        chat_session = self.client.chats.create(model=model_id, config=config, history=history_msgs if history_msgs else None)
        return chat_session

    def _process_gemini_function_call(self, function_call):
        """
        Process Gemini function call and convert to framework format.

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
            # If there's any error processing the function call, return None
            return None
