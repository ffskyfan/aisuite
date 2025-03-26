# Anthropic provider
# Links:
# Tool calling docs - https://docs.anthropic.com/en/docs/build-with-claude/tool-use

import anthropic
import json
from typing import AsyncGenerator, Union
from aisuite.provider import Provider
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.message import Message, ChatCompletionMessageToolCall, Function
from aisuite.framework.chat_completion_response import Choice, ChoiceDelta, StreamChoice

# Define a constant for the default max_tokens value
DEFAULT_MAX_TOKENS = 4096


class AnthropicMessageConverter:
    # Role constants
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_TOOL = "tool"
    ROLE_SYSTEM = "system"

    # Finish reason mapping
    FINISH_REASON_MAPPING = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }

    def convert_request(self, messages):
        """Convert framework messages to Anthropic format."""
        system_message = self._extract_system_message(messages)
        converted_messages = [self._convert_single_message(msg) for msg in messages]
        return system_message, converted_messages

    def convert_response(self, response):
        """Normalize the response from the Anthropic API to match OpenAI's response format."""
        normalized_response = ChatCompletionResponse()
         # 创建一个Choice对象并添加到choices列表中，避免索引越界
        message = self._get_message(response)
        finish_reason = self._get_finish_reason(response)
        usage = self._get_usage_stats(response)
        
        # 创建Choice对象并添加到列表中
        from aisuite.framework.choice import Choice
        normalized_response.choices.append(Choice(
            index=0,
            message=message,
            finish_reason=finish_reason
        ))
        
        # 设置使用量统计
        normalized_response.metadata['usage'] = usage
        return normalized_response
        
    def convert_stream_response(self, chunk, model):
        """Convert a streaming response chunk from Anthropic to the framework's format."""
        if hasattr(chunk, 'delta') and chunk.delta and chunk.delta.text:
            return ChatCompletionResponse(
                choices=[
                    StreamChoice(
                        index=0,
                        delta=ChoiceDelta(
                            content=chunk.delta.text,
                            role="assistant" if chunk.delta.type == "text_delta" else None
                        ),
                        finish_reason=self._get_finish_reason(chunk) if hasattr(chunk, 'stop_reason') else None
                    )
                ],
                metadata={
                    'id': getattr(chunk, 'id', None),
                    'created': None,  # Anthropic doesn't provide timestamp in chunks
                    'model': model
                }
            )
        # For the initial chunk that might not have content
        return ChatCompletionResponse(
            choices=[
                StreamChoice(
                    index=0,
                    delta=ChoiceDelta(
                        content="",
                        role="assistant"
                    ),
                    finish_reason=None
                )
            ],
            metadata={
                'id': getattr(chunk, 'id', None),
                'created': None,
                'model': model
            }
        )

    def _convert_single_message(self, msg):
        """Convert a single message to Anthropic format."""
        if isinstance(msg, dict):
            return self._convert_dict_message(msg)
        return self._convert_message_object(msg)

    def _convert_dict_message(self, msg):
        """Convert a dictionary message to Anthropic format."""
        if msg["role"] == self.ROLE_TOOL:
            return self._create_tool_result_message(msg["tool_call_id"], msg["content"])
        elif msg["role"] == self.ROLE_ASSISTANT and "tool_calls" in msg:
            return self._create_assistant_tool_message(
                msg["content"], msg["tool_calls"]
            )
        return {"role": msg["role"], "content": msg["content"]}

    def _convert_message_object(self, msg):
        """Convert a Message object to Anthropic format."""
        if msg.role == self.ROLE_TOOL:
            return self._create_tool_result_message(msg.tool_call_id, msg.content)
        elif msg.role == self.ROLE_ASSISTANT and msg.tool_calls:
            return self._create_assistant_tool_message(msg.content, msg.tool_calls)
        return {"role": msg.role, "content": msg.content}

    def _create_tool_result_message(self, tool_call_id, content):
        """Create a tool result message in Anthropic format."""
        return {
            "role": self.ROLE_USER,
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content,
                }
            ],
        }

    def _create_assistant_tool_message(self, content, tool_calls):
        """Create an assistant message with tool calls in Anthropic format."""
        message_content = []
        if content:
            message_content.append({"type": "text", "text": content})

        for tool_call in tool_calls:
            tool_input = (
                tool_call["function"]["arguments"]
                if isinstance(tool_call, dict)
                else tool_call.function.arguments
            )
            message_content.append(
                {
                    "type": "tool_use",
                    "id": (
                        tool_call["id"] if isinstance(tool_call, dict) else tool_call.id
                    ),
                    "name": (
                        tool_call["function"]["name"]
                        if isinstance(tool_call, dict)
                        else tool_call.function.name
                    ),
                    "input": json.loads(tool_input),
                }
            )

        return {"role": self.ROLE_ASSISTANT, "content": message_content}

    def _extract_system_message(self, messages):
        """Extract system message if present, otherwise return empty list."""
        # TODO: This is a temporary solution to extract the system message.
        # User can pass multiple system messages, which can mingled with other messages.
        # This needs to be fixed to handle this case.
        if messages and messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages.pop(0)
            return system_message
        return []

    def _get_finish_reason(self, response):
        """Get the normalized finish reason."""
        return self.FINISH_REASON_MAPPING.get(response.stop_reason, "stop")

    def _get_usage_stats(self, response):
        """Get the usage statistics."""
        return {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

    def _get_message(self, response):
        """Get the appropriate message based on response type."""
        if response.stop_reason == "tool_use":
            tool_message = self.convert_response_with_tool_use(response)
            if tool_message:
                return tool_message

        return Message(
            content=response.content[0].text,
            role="assistant",
            tool_calls=None,
            refusal=None,
        )

    def convert_response_with_tool_use(self, response):
        """Convert Anthropic tool use response to the framework's format."""
        tool_call = next(
            (content for content in response.content if content.type == "tool_use"),
            None,
        )

        if tool_call:
            function = Function(
                name=tool_call.name, arguments=json.dumps(tool_call.input)
            )
            tool_call_obj = ChatCompletionMessageToolCall(
                id=tool_call.id, function=function, type="function"
            )
            text_content = next(
                (
                    content.text
                    for content in response.content
                    if content.type == "text"
                ),
                "",
            )

            return Message(
                content=text_content or None,
                tool_calls=[tool_call_obj] if tool_call else None,
                role="assistant",
                refusal=None,
            )
        return None

    def convert_tool_spec(self, openai_tools):
        """Convert OpenAI tool specification to Anthropic format."""
        anthropic_tools = []

        for tool in openai_tools:
            if tool.get("type") != "function":
                continue

            function = tool["function"]
            anthropic_tool = {
                "name": function["name"],
                "description": function["description"],
                "input_schema": {
                    "type": "object",
                    "properties": function["parameters"]["properties"],
                    "required": function["parameters"].get("required", []),
                },
            }
            anthropic_tools.append(anthropic_tool)

        return anthropic_tools


class AnthropicProvider(Provider):
    def __init__(self, **config):
        """Initialize the Anthropic provider with the given configuration."""
        self.client = anthropic.AsyncAnthropic(**config)
        self.converter = AnthropicMessageConverter()

    async def chat_completions_create(self, model, messages, stream: bool = False, **kwargs) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionResponse, None]]:
        """Create a chat completion using the Anthropic API."""
        kwargs = self._prepare_kwargs(kwargs)
        system_message, converted_messages = self.converter.convert_request(messages)

        if stream:
            response = await self.client.messages.create(
                model=model, 
                system=system_message, 
                messages=converted_messages, 
                stream=True,
                **kwargs
            )
            
            async def stream_generator():
                async for chunk in response:
                    yield self.converter.convert_stream_response(chunk, model)
            
            return stream_generator()
        else:
            response = await self.client.messages.create(
                model=model, 
                system=system_message, 
                messages=converted_messages, 
                **kwargs
            )
            return self.converter.convert_response(response)

    def _prepare_kwargs(self, kwargs):
        """Prepare kwargs for the API call."""
        kwargs = kwargs.copy()
        kwargs.setdefault("max_tokens", DEFAULT_MAX_TOKENS)

        if "tools" in kwargs:
            kwargs["tools"] = self.converter.convert_tool_spec(kwargs["tools"])

        return kwargs
