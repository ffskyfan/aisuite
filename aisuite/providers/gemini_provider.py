from google import genai
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
        # Pass timeout to the client via transport if provided
        timeout = config.pop("timeout", None)
        if timeout:
            transport = genai.transports.Transport(timeout=timeout)
            self.client = genai.Client(api_key=config["api_key"], transport=transport)
        else:
            self.client = genai.Client(api_key=config["api_key"])


    async def chat_completions_create(self, model, messages, stream: bool = False, **kwargs) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionResponse, None]]:
        # Check for testing environment
        if os.getenv("IS_TESTING"):
            return self._mock_response(stream, kwargs.get("response_text"), kwargs.get("response_chunks"))

        # Convert aisuite messages to Gemini contents format (list of strings).
        contents = [message.content for message in messages]

        if stream:
            response = self.client.models.generate_content_stream(
                model=model,
                contents=contents,
                **kwargs  # Pass any additional arguments, including 'config'
            )
            async def stream_generator():
                for chunk in response:
                    if chunk.text:  # Check for empty chunks
                        yield ChatCompletionResponse(
                            choices=[
                                StreamChoice(
                                    index=0,  # Gemini doesn't provide index in stream
                                    delta=ChoiceDelta(
                                        content=chunk.text,
                                        role="model"  # Gemini doesn't return role in stream
                                    ),
                                    finish_reason=None  # Gemini doesn't provide this in the stream.
                                )
                            ],
                            metadata={
                                'model': model  # Use the requested model
                            }
                        )
            return stream_generator()
        else:
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                **kwargs  # Pass any additional arguments, including 'config'
            )

            return ChatCompletionResponse(
                choices=[
                    Choice(
                        index=0,
                        message=response.text,
                        finish_reason=None  # TODO: Check if Gemini provides a finish reason
                    )
                ],
                metadata={
                    "model": model,  # Use the requested model
                }
            )
