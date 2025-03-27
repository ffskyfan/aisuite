import openai
import os
import json
from typing import AsyncGenerator, Union

from aisuite.provider import Provider, LLMError
from aisuite.framework.chat_completion_response import ChatCompletionResponse, Choice, ChoiceDelta, StreamChoice
from aisuite.framework.message import Message, ChatCompletionMessageToolCall, Function


class DeepseekProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the DeepSeek provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("DEEPSEEK_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "DeepSeek API key is missing. Please provide it in the config or set the OPENAI_API_KEY environment variable."
            )
        config["base_url"] = "https://api.deepseek.com"

        # NOTE: We could choose to remove above lines for api_key since OpenAI will automatically
        # infer certain values from the environment variables.
        # Eg: OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_PROJECT_ID. Except for OPEN_AI_BASE_URL which has to be the deepseek url

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
                                choices=[
                                    StreamChoice(
                                        index=choice.index,
                                        delta=ChoiceDelta(
                                            content=choice.delta.content,
                                            role=choice.delta.role,
                                            reasoning_content=getattr(choice.delta, 'reasoning_content', None)
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
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                **kwargs  # Pass any additional arguments to the OpenAI API
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
                            reasoning_content=getattr(choice.message, 'reasoning_content', None)
                        ),
                        finish_reason=getattr(choice, 'finish_reason', None)
                    )
                    for choice in response.choices
                ],
                metadata={
                    "id": response.id,
                    "created": response.created,
                    "model": model
                }
            )

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
