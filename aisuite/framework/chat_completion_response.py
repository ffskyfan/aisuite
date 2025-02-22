from aisuite.framework.choice import Choice
from dataclasses import dataclass
from typing import List, Optional, Union

@dataclass
class ChoiceDelta:
    content: Optional[str] = None
    role: Optional[str] = None
    reasoning_content: Optional[str] = None

@dataclass
class StreamChoice:
    index: int
    delta: ChoiceDelta
    finish_reason: Optional[str] = None

class ChatCompletionResponse:
    """Standard response format for chat completions across all providers"""
    
    def __init__(self, choices: list = None, metadata: dict = None):
        self.choices = choices if choices is not None else []
        self.metadata = metadata or {}

    @property
    def id(self):
        return self.metadata.get('id')

    @property
    def created(self):
        return self.metadata.get('created')

    @property
    def model(self):
        return self.metadata.get('model')

    @property
    def usage(self):
        return self.metadata.get('usage')
