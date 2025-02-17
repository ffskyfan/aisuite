from google import generativeai as genai
import os
from typing import AsyncGenerator, Union

from aisuite.provider import Provider, LLMError
from aisuite.framework.chat_completion_response import ChatCompletionResponse, Choice, ChoiceDelta, StreamChoice


class GeminiProvider(Provider):
    def __init__(self, **config):
        # Initialize the Gemini client. API key is required.
        config.setdefault("api_key", os.getenv("GEMINI_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "Gemini API key is missing. Please provide it in the config or set the GEMINI_API_KEY environment variable."
            )
        
        # Configure the Gemini client
        genai.configure(api_key=config["api_key"])
        
        # Default generation config
        self.generation_config = {
            "temperature": config.get("temperature", 1),
            "top_p": config.get("top_p", 0.95),
            "top_k": config.get("top_k", 64),
            "max_output_tokens": config.get("max_output_tokens", 8192),
            "response_mime_type": "text/plain",
        }

    async def chat_completions_create(self, model, messages, stream: bool = False, **kwargs) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionResponse, None]]:
        # Check for testing environment
        if os.getenv("IS_TESTING"):
            return self._mock_response(stream, kwargs.get("response_text"), kwargs.get("response_chunks"))

        # Create the model with generation config and system instruction
        generation_model = genai.GenerativeModel(
            model_name=model,
            generation_config=self.generation_config
        )

        # Convert messages to Gemini chat history format
        history = []
        for msg in messages[:-1]:  # Process all messages except the last one as history
            history.append({
                "role": msg.role,
                "parts": [msg.content]
            })

        # Start chat session with history
        chat = generation_model.start_chat(history=history if history else None)
        
        # Send the last message
        last_message = messages[-1].content

        if stream:
            response = chat.send_message(last_message, stream=True)
            async def stream_generator():
                for chunk in response:
                    if chunk.text:  # Check for empty chunks
                        yield ChatCompletionResponse(
                            choices=[
                                StreamChoice(
                                    index=0,
                                    delta=ChoiceDelta(
                                        content=chunk.text,
                                        role="assistant"
                                    ),
                                    finish_reason=None
                                )
                            ],
                            metadata={
                                'model': model
                            }
                        )
            return stream_generator()
        else:
            response = chat.send_message(last_message)
            
            return ChatCompletionResponse(
                choices=[
                    Choice(
                        index=0,
                        message=response.text,
                        finish_reason=None
                    )
                ],
                metadata={
                    "model": model,
                }
            )
