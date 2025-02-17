import os
from aisuite.providers import Provider
from typing import AsyncGenerator, Union
from aisuite.framework.chat_completion_response import ChatCompletionResponse, Choice, ChoiceDelta, StreamChoice
from aisuite.provider import Provider, LLMError

# Import Google GenAI SDK
from google import genai
from google.genai import types

class GeminiProvider(Provider):
    def __init__(self, **kwargs):
        """Initialize the Gemini provider with API key and client."""
        super().__init__(**kwargs)
        api_key = os.environ.get("GEMINI_API_KEY") or kwargs.get("api_key")
        if api_key is None:
            raise RuntimeError("GEMINI_API_KEY is required for GeminiProvider")
        # Initialize the GenAI client for Gemini (non-Vertex usage)
        self.client = genai.Client(api_key=api_key)
    
    
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
        if messages and messages[0].get("role") == "system":
            config_kwargs["system_instruction"] = messages[0]["content"]
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
                part = types.Part.from_text(text=msg["content"])
                history_msgs.append(types.Content(role=role, parts=[part]))
        # Create a new chat session with history and config (if any)
        chat = self.client.chats.create(model=model_id, config=config, history=history_msgs if history_msgs else None)
        if last_user_message is None:
            # No user prompt to send (no completion to generate)
            return None
        # Send the last user message and get response (streaming or full)
        if stream:
            # Streaming response: return a generator yielding ChatCompletionResponse objects
            async def stream_generator():
                response_id = None  # We'll use the first chunk's id for all chunks
                for chunk in chat.send_message_stream(last_user_message):
                    if response_id is None:
                        response_id = chunk.id
                    yield ChatCompletionResponse(
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=ChoiceDelta(
                                    content=chunk.text,
                                    role="assistant" if chunk.text else None
                                ),
                                finish_reason=None  # Gemini doesn't provide per-chunk finish reason
                            )
                        ],
                        metadata={
                            'id': response_id,
                            'created': None,  # Gemini doesn't provide timestamp
                            'model': model_id
                        }
                    )
            return stream_generator()
        else:
            # Single-turn completion: get the full response
            response = chat.send_message(message=last_user_message)
            return response  # The response object with .text, .candidates, etc.
    
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
                    part = types.Part.from_text(text=msg["content"])
                    history_msgs.append(types.Content(role=msg["role"], parts=[part]))
        # Create and return the chat session object
        chat_session = self.client.chats.create(model=model_id, config=config, history=history_msgs if history_msgs else None)
        return chat_session
