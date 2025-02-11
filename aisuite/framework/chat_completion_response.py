from aisuite.framework.choice import Choice
from dataclasses import dataclass
from typing import List, Optional, Union

@dataclass
class ChoiceDelta:
    content: Optional[str] = None
    role: Optional[str] = None

@dataclass
class StreamChoice:
    index: int
    delta: ChoiceDelta
    finish_reason: Optional[str] = None

class ChatCompletionResponse:
    """Used to conform to the response model of OpenAI"""

    def __init__(self):
        self.choices = []  # Adjust the range as needed for more choices
