from aisuite.framework.message import Message
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from aisuite.framework.stop_reason import StopInfo


class Choice:
    def __init__(self, index: Optional[int] = None, message: Optional[Message] = None,
                 finish_reason: Optional[str] = None, stop_info: Optional["StopInfo"] = None):
        self.index = index
        self.finish_reason = finish_reason  # Changed to str for flexibility
        self.stop_info = stop_info  # New enhanced stop information
        self.message = message if message is not None else Message(
            content=None, tool_calls=None, role="assistant", refusal=None
        )
        self.intermediate_messages: List[Message] = []
