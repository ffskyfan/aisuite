import openai
import os
from typing import AsyncGenerator, Union

from aisuite.provider import Provider, LLMError
from aisuite.framework.chat_completion_response import ChatCompletionResponse, Choice, ChoiceDelta, StreamChoice


class OpenaiProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the OpenAI provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("OPENAI_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "OpenAI API key is missing. Please provide it in the config or set the OPENAI_API_KEY environment variable."
            )

        # NOTE: We could choose to remove above lines for api_key since OpenAI will automatically
        # infer certain values from the environment variables.
        # Eg: OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_PROJECT_ID, OPENAI_BASE_URL, etc.

        # Pass the entire config to the OpenAI client constructor
        self.client = openai.AsyncOpenAI(**config)

    async def chat_completions_create(self, model, messages, stream: bool = False, **kwargs) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionResponse, None]]:
        # Any exception raised by OpenAI will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        if stream:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **kwargs  # Pass any additional arguments to the OpenAI API
            )
            async def stream_generator():
                async for chunk in response:
                    if chunk.choices:
                        yield ChatCompletionResponse(
                            id=chunk.id,
                            created=chunk.created,
                            model=chunk.model,
                            choices=[
                                StreamChoice(
                                    index=choice.index,
                                    delta=ChoiceDelta(
                                        content=choice.delta.content,
                                        role=choice.delta.role
                                    ),
                                    finish_reason=choice.finish_reason
                                )
                                for choice in chunk.choices
                            ],
                            usage=None  # Usage is not provided in stream chunks
                        )
            return stream_generator()
        else:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                **kwargs  # Pass any additional arguments to the OpenAI API
            )

            return ChatCompletionResponse(
                id=response.id,
                created=response.created,
                model=response.model,
                choices=[
                    Choice(
                        index=choice.index,
                        message=choice.message.content,
                        finish_reason=choice.finish_reason
                    )
                    for choice in response.choices
                ],
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            )
