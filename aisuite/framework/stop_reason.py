"""
AISuite Framework Stop Reason Module

Standardizing stop reasons across different LLM providers.
Provides unified StopReason enumeration, StopInfo data structure, and mapping logic.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class StopReason(Enum):
    """Standardized stop reasons across all LLM providers"""
    
    # Normal completion
    COMPLETE = "complete"                    # Natural completion
    TOOL_CALL = "tool_call"                 # Tool call required
    
    # Limits
    LENGTH_LIMIT = "length_limit"           # Hit max_tokens limit
    CONTEXT_LIMIT = "context_limit"         # Context length limit
    
    # Control
    STOP_SEQUENCE = "stop_sequence"         # Hit custom stop sequence
    USER_INTERRUPT = "user_interrupt"       # User interrupted
    
    # Processing
    PROCESSING_PAUSE = "processing_pause"   # Processing pause (e.g., web search)
    
    # Safety/Content
    SAFETY_REFUSAL = "safety_refusal"       # Safety refusal
    CONTENT_FILTER = "content_filter"       # Content filtered
    LANGUAGE_UNSUPPORTED = "language_unsupported"  # Unsupported language

    # Tool/Function Call Issues
    TOOL_CALL_ERROR = "tool_call_error"     # Tool call error

    # Error
    ERROR = "error"                         # Model/request error
    UNKNOWN = "unknown"                     # Unknown reason


@dataclass
class StopInfo:
    """Enhanced stop information with standardized reason and metadata"""

    reason: StopReason                      # Standardized reason
    original_reason: str                    # Original provider-specific reason
    metadata: Dict[str, Any]                # Additional context

    def __post_init__(self):
        """Ensure metadata is always a dict"""
        if self.metadata is None:
            self.metadata = {}


class ProviderStopMapper:
    """Base class for provider-specific stop reason mapping"""
    
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
    
    def map_stop_reason(self, original_reason: str, metadata: Dict[str, Any]) -> StopInfo:
        """Map provider-specific stop reason to standardized StopInfo"""
        raise NotImplementedError("Subclasses must implement map_stop_reason")
    
    def _create_stop_info(self, reason: StopReason, original_reason: str,
                         metadata: Dict[str, Any]) -> StopInfo:
        """Helper to create StopInfo with provider metadata"""
        metadata = metadata or {}
        metadata["provider"] = self.provider_name

        return StopInfo(
            reason=reason,
            original_reason=original_reason,
            metadata=metadata
        )


class AnthropicStopMapper(ProviderStopMapper):
    """Anthropic Claude stop reason mapper"""
    
    def __init__(self):
        super().__init__("anthropic")
    
    def map_stop_reason(self, original_reason: str, metadata: Dict[str, Any]) -> StopInfo:
        """Map Anthropic stop reasons to standardized format"""
        
        if original_reason == "end_turn":
            return self._create_stop_info(
                StopReason.COMPLETE, original_reason, metadata
            )

        elif original_reason == "max_tokens":
            return self._create_stop_info(
                StopReason.LENGTH_LIMIT, original_reason, metadata
            )

        elif original_reason == "tool_use":
            return self._create_stop_info(
                StopReason.TOOL_CALL, original_reason, metadata
            )

        elif original_reason == "stop_sequence":
            return self._create_stop_info(
                StopReason.STOP_SEQUENCE, original_reason, metadata
            )

        elif original_reason == "pause_turn":
            return self._create_stop_info(
                StopReason.PROCESSING_PAUSE, original_reason, metadata
            )

        elif original_reason == "refusal":
            return self._create_stop_info(
                StopReason.SAFETY_REFUSAL, original_reason, metadata
            )
        
        else:
            logger.warning(f"Unknown Anthropic stop reason: {original_reason}")
            return self._create_stop_info(
                StopReason.UNKNOWN, original_reason, metadata
            )


class OpenAIStopMapper(ProviderStopMapper):
    """OpenAI stop reason mapper"""

    def __init__(self):
        super().__init__("openai")

    def map_stop_reason(self, original_reason: str, metadata: Dict[str, Any]) -> StopInfo:
        """Map OpenAI finish_reason to standardized format"""

        if original_reason == "stop":
            return self._create_stop_info(
                StopReason.COMPLETE, original_reason, metadata
            )

        elif original_reason == "length":
            return self._create_stop_info(
                StopReason.LENGTH_LIMIT, original_reason, metadata
            )

        elif original_reason in ["tool_calls", "function_call"]:
            return self._create_stop_info(
                StopReason.TOOL_CALL, original_reason, metadata
            )

        elif original_reason == "content_filter":
            return self._create_stop_info(
                StopReason.SAFETY_REFUSAL, original_reason, metadata
            )

        else:
            logger.warning(f"Unknown OpenAI stop reason: {original_reason}")
            return self._create_stop_info(
                StopReason.UNKNOWN, original_reason, metadata
            )


class GeminiStopMapper(ProviderStopMapper):
    """Google Gemini stop reason mapper"""

    def __init__(self):
        super().__init__("gemini")

    def map_stop_reason(self, original_reason: str, metadata: Dict[str, Any]) -> StopInfo:
        """Map Gemini finish_reason to standardized format"""

        if original_reason == "STOP":
            return self._create_stop_info(
                StopReason.COMPLETE, original_reason, metadata
            )

        elif original_reason == "MAX_TOKENS":
            return self._create_stop_info(
                StopReason.LENGTH_LIMIT, original_reason, metadata
            )

        elif original_reason in ["SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"]:
            return self._create_stop_info(
                StopReason.SAFETY_REFUSAL, original_reason, metadata
            )

        elif original_reason == "RECITATION":
            return self._create_stop_info(
                StopReason.CONTENT_FILTER, original_reason, metadata
            )

        elif original_reason == "LANGUAGE":
            return self._create_stop_info(
                StopReason.LANGUAGE_UNSUPPORTED, original_reason, metadata
            )

        elif original_reason in ["MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS"]:
            return self._create_stop_info(
                StopReason.TOOL_CALL_ERROR, original_reason, metadata
            )

        elif original_reason in ["OTHER", "FINISH_REASON_UNSPECIFIED"]:
            return self._create_stop_info(
                StopReason.UNKNOWN, original_reason, metadata
            )

        else:
            logger.warning(f"Unknown Gemini stop reason: {original_reason}")
            return self._create_stop_info(
                StopReason.UNKNOWN, original_reason, metadata
            )





class StopReasonManager:
    """Central manager for stop reason mapping and processing"""
    
    def __init__(self):
        self._mappers: Dict[str, ProviderStopMapper] = {}
        self._register_default_mappers()
    
    def _register_default_mappers(self):
        """Register default provider mappers"""
        self.register_mapper("anthropic", AnthropicStopMapper())
        self.register_mapper("openai", OpenAIStopMapper())
        self.register_mapper("gemini", GeminiStopMapper())
    
    def register_mapper(self, provider_name: str, mapper: ProviderStopMapper):
        """Register a provider-specific mapper"""
        self._mappers[provider_name] = mapper
        logger.debug(f"Registered stop mapper for provider: {provider_name}")
    
    def map_stop_reason(self, provider_name: str, original_reason: str, 
                       metadata: Dict[str, Any] = None) -> StopInfo:
        """Map provider-specific stop reason to standardized StopInfo"""
        
        metadata = metadata or {}
        
        if provider_name not in self._mappers:
            logger.warning(f"No mapper found for provider: {provider_name}")
            return StopInfo(
                reason=StopReason.UNKNOWN,
                original_reason=original_reason,
                metadata={**metadata, "provider": provider_name}
            )
        
        mapper = self._mappers[provider_name]
        return mapper.map_stop_reason(original_reason, metadata)


# Global instance
stop_reason_manager = StopReasonManager()
