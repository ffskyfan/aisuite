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

    def __init__(self, id: str = None, created: int = None, model: str = None, choices: list = None, usage: dict = None):
        self.id = id
        self.created = created
        self.model = model
        self.choices = choices if choices is not None else []
        self.usage = usage
