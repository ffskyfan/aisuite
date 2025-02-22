"""Interface to hold contents of api responses when they do not confirm to the OpenAI style response"""

from pydantic import BaseModel
from typing import Literal, Optional


class Function(BaseModel):
    arguments: str
    name: str


class ChatCompletionMessageToolCall(BaseModel):
    id: str
    function: Function
    type: Literal["function"]


class Message(BaseModel):
    content: Optional[str] = None
    tool_calls: Optional[list[ChatCompletionMessageToolCall]] = None
    role: Optional[Literal["user", "assistant", "system"]] = "assistant"
    refusal: Optional[str] = None
    reasoning_content: Optional[str] = None
