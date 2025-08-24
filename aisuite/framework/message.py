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


class ReasoningContent(BaseModel):
    """推理内容，支持不同provider的推理数据格式"""
    thinking: str                    # 推理文本内容
    signature: Optional[str] = None  # Claude等provider的签名
    provider: Optional[str] = None   # 标识来源provider
    raw_data: Optional[dict] = None  # 原始数据，用于完整重构


class Message(BaseModel):
    content: Optional[str] = None
    tool_calls: Optional[list[ChatCompletionMessageToolCall]] = None
    role: Optional[Literal["user", "assistant", "system"]] = "assistant"
    refusal: Optional[str] = None
    reasoning_content: Optional[ReasoningContent] = None
