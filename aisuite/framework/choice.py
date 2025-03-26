from aisuite.framework.message import Message
from typing import Literal, Optional, List


class Choice:
    def __init__(self, index: Optional[int] = None, message: Optional[Message] = None, 
                 finish_reason: Optional[Literal["stop", "tool_calls"]] = None):
        self.index = index
        self.finish_reason = finish_reason
        self.message = message if message is not None else Message(
            content=None, tool_calls=None, role="assistant", refusal=None
        )
        self.intermediate_messages: List[Message] = []
