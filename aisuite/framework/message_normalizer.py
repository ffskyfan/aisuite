"""
Message Normalizer for cross-provider compatibility
Handles format conversion and field cleaning between different providers
"""

from typing import Dict, List, Any, Optional, Union
from ..framework.message import Message, ReasoningContent

class MessageNormalizer:
    """Normalizes messages for cross-provider compatibility"""
    
    # Fields that should be preserved when passing between providers
    CORE_FIELDS = {'role', 'content', 'tool_calls', 'name'}
    
    # Fields that may cause compatibility issues
    OPTIONAL_FIELDS = {'refusal', 'reasoning_content', 'tool_call_id', 'function_call'}
    
    # Provider-specific field mappings
    PROVIDER_FIELD_MAP = {
        'gpt-5': {
            'remove_fields': ['refusal'],  # GPT-5 Response API doesn't accept these
            'preserve_reasoning': True
        },
        'claude': {
            'remove_fields': [],
            'preserve_reasoning': False  # Claude handles reasoning differently
        },
        'gemini': {
            'remove_fields': [],
            'preserve_reasoning': False
        }
    }
    
    @staticmethod
    def detect_provider_type(model: str) -> str:
        """Detect provider type from model string"""
        model_lower = model.lower()
        
        # Check for GPT-5 (most specific first)
        if 'gpt-5' in model_lower:
            return 'gpt-5'
        # Check for Claude/Anthropic
        elif 'claude' in model_lower or 'anthropic' in model_lower:
            return 'claude'
        # Check for Gemini
        elif 'gemini' in model_lower or 'google' in model_lower:
            return 'gemini'
        # Check for DeepSeek
        elif 'deepseek' in model_lower:
            return 'deepseek'
        # Check for GPT-4 or other OpenAI models
        elif 'gpt-4' in model_lower or 'gpt-3' in model_lower or 'openai' in model_lower:
            return 'openai'
        # Check for CloseAI (can be various models)
        elif 'closeai' in model_lower:
            # Further check the actual model after the colon
            parts = model.split(':')
            if len(parts) > 1:
                actual_model = parts[1].lower()
                if 'gpt-5' in actual_model:
                    return 'gpt-5'
                elif 'gpt-4' in actual_model or 'gpt-3' in actual_model:
                    return 'openai'
            return 'openai'  # Default for closeai
        else:
            return 'default'
    
    @classmethod
    def normalize_message(cls, message: Union[Dict, Message], target_provider: str = None) -> Dict[str, Any]:
        """
        Normalize a message for compatibility
        
        Args:
            message: The message to normalize (dict or Message object)
            target_provider: The target provider type (for provider-specific handling)
        
        Returns:
            Normalized message dictionary
        """
        # Convert Message object to dict if needed
        if hasattr(message, 'model_dump'):
            try:
                msg_dict = message.model_dump()
            except Exception:
                # Fallback for serialization issues
                msg_dict = cls._manual_serialize(message)
        elif isinstance(message, dict):
            msg_dict = message.copy()
        else:
            msg_dict = {'role': 'assistant', 'content': str(message)}
        
        # Clean up the message
        normalized = cls._clean_message(msg_dict, target_provider)
        
        # Ensure required fields
        normalized = cls._ensure_required_fields(normalized)
        
        return normalized
    
    @classmethod
    def _manual_serialize(cls, message: Message) -> Dict[str, Any]:
        """Manually serialize Message object when model_dump fails"""
        result = {}
        
        # Extract core fields
        result['role'] = getattr(message, 'role', 'assistant')
        result['content'] = getattr(message, 'content', None)
        
        # Handle None content (common with GPT-5)
        if result['content'] is None:
            result['content'] = ""
        
        # Extract optional fields
        if hasattr(message, 'tool_calls') and message.tool_calls:
            # Convert tool calls to serializable format
            tool_calls = []
            for tc in message.tool_calls:
                if hasattr(tc, 'model_dump'):
                    try:
                        tool_calls.append(tc.model_dump())
                    except Exception:
                        # Fallback serialization
                        tool_calls.append({
                            'id': getattr(tc, 'id', ''),
                            'type': getattr(tc, 'type', 'function'),
                            'function': {
                                'name': getattr(tc.function, 'name', ''),
                                'arguments': getattr(tc.function, 'arguments', '')
                            }
                        })
                else:
                    tool_calls.append({
                        'id': getattr(tc, 'id', ''),
                        'type': getattr(tc, 'type', 'function'),
                        'function': {
                            'name': getattr(tc.function, 'name', ''),
                            'arguments': getattr(tc.function, 'arguments', '')
                        }
                    })
            result['tool_calls'] = tool_calls
        
        # Handle reasoning content
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            rc = message.reasoning_content
            if isinstance(rc, ReasoningContent):
                result['reasoning_content'] = {
                    'thinking': rc.thinking,
                    'provider': rc.provider,
                    'raw_data': rc.raw_data
                }
            elif isinstance(rc, dict):
                result['reasoning_content'] = rc
            else:
                result['reasoning_content'] = {'thinking': str(rc)}
        
        return result
    
    @classmethod
    def _clean_message(cls, msg_dict: Dict[str, Any], target_provider: str = None) -> Dict[str, Any]:
        """Clean message based on target provider requirements"""
        cleaned = {}
        
        # Get provider-specific config
        provider_config = cls.PROVIDER_FIELD_MAP.get(target_provider, {})
        fields_to_remove = provider_config.get('remove_fields', [])
        preserve_reasoning = provider_config.get('preserve_reasoning', False)
        
        # ALWAYS remove these fields that cause compatibility issues
        # reasoning_content should NEVER be passed to any provider except GPT-5
        always_remove = {'refusal', 'reasoning_content'}
        if target_provider == 'gpt-5':
            always_remove = {'refusal'}  # GPT-5 can handle reasoning_content
        
        for key, value in msg_dict.items():
            # Skip None values except for content
            if value is None and key != 'content':
                continue
            
            # Always remove problematic fields
            if key in always_remove:
                continue
            
            # Skip fields that should be removed for target provider
            if key in fields_to_remove:
                continue
            
            # Handle reasoning content specially - NEVER pass to non-GPT-5 providers
            if key == 'reasoning_content':
                # Only preserve for GPT-5, remove for all others
                if target_provider == 'gpt-5' and preserve_reasoning and value:
                    cleaned[key] = value
                # Skip for all other providers - this is redundant but explicit
                continue
            
            # Keep core and allowed optional fields
            if key in cls.CORE_FIELDS or key in cls.OPTIONAL_FIELDS:
                cleaned[key] = value
        
        return cleaned
    
    @classmethod
    def _ensure_required_fields(cls, msg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure message has required fields"""
        # Ensure role
        if 'role' not in msg_dict:
            msg_dict['role'] = 'assistant'
        
        # Ensure content (convert None to empty string)
        if 'content' not in msg_dict or msg_dict['content'] is None:
            msg_dict['content'] = ""
        
        return msg_dict
    
    @classmethod
    def normalize_messages(cls, messages: List[Union[Dict, Message]], target_model: str = None) -> List[Dict[str, Any]]:
        """
        Normalize a list of messages for a target model
        
        Args:
            messages: List of messages to normalize
            target_model: The target model (e.g., "gpt-5", "claude-3-5-sonnet")
        
        Returns:
            List of normalized message dictionaries
        """
        target_provider = cls.detect_provider_type(target_model) if target_model else None
        normalized = []
        
        for msg in messages:
            # Skip system messages and user messages (usually don't have compatibility issues)
            if isinstance(msg, dict) and msg.get('role') in ['system', 'user', 'tool']:
                normalized.append(msg)
            else:
                normalized.append(cls.normalize_message(msg, target_provider))
        
        return normalized
    
    @classmethod
    def extract_reasoning_content(cls, message: Union[Dict, Message]) -> Optional[Dict[str, Any]]:
        """Extract reasoning content from a message"""
        if hasattr(message, 'reasoning_content'):
            return message.reasoning_content
        elif isinstance(message, dict):
            return message.get('reasoning_content')
        return None
    
    @classmethod
    def prepare_for_responses_api(cls, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare messages for GPT-5 Responses API format
        Handles the conversion from Chat format to Responses format
        """
        # This would be called by the provider when using Responses API
        # For now, just ensure compatibility
        prepared = []
        for msg in messages:
            cleaned = msg.copy()
            # Remove fields that Responses API doesn't accept
            for field in ['refusal', 'reasoning_content']:
                cleaned.pop(field, None)
            prepared.append(cleaned)
        return prepared