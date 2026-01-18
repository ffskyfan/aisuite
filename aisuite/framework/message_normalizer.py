"""
Message Normalizer for cross-provider compatibility
Handles format conversion and field cleaning between different providers
"""

from typing import Dict, List, Any, Optional, Union
from ..framework.message import Message, ReasoningContent
from ..framework.cache_config import CacheConfig, CacheType

class MessageNormalizer:
    """Normalizes messages for cross-provider compatibility"""
    
    # Fields that should be preserved when passing between providers
    CORE_FIELDS = {'role', 'content', 'tool_calls', 'name'}
    
    # Fields that may cause compatibility issues
    OPTIONAL_FIELDS = {'refusal', 'reasoning_content', 'tool_call_id', 'function_call', 'cache_control'}
    
    # Provider-specific field mappings
    PROVIDER_FIELD_MAP = {
        'openai': {
            'remove_fields': ['refusal'],  # Some OpenAI models don't accept these
            'preserve_reasoning': True  # GPT-5 can preserve reasoning
        },
        'anthropic': {
            'remove_fields': [],
            'preserve_reasoning': False  # Claude handles reasoning differently
        },
        'gemini': {
            'remove_fields': [],
            'preserve_reasoning': False
        }
    }
    
    # Provider cache support configuration
    PROVIDER_CACHE_SUPPORT = {
        'anthropic': {  # 使用正确的provider名称
            'supported': True,
            'field_name': 'cache_control',
            'format': lambda config: {
                "type": "ephemeral",
                "ttl": str(config.get('ttl', '5m')).replace('CacheTTL.', '').replace('MINUTES_5', '5m').replace('HOURS_1', '1h')
            } if config.get('type') in ['ephemeral', CacheType.EPHEMERAL, 'CacheType.EPHEMERAL'] else None,
            'min_tokens': {
                'claude-3-opus': 1024,
                'claude-3-5-sonnet': 2048,
                'claude-3-7-sonnet': 2048,
                'claude-3-haiku': 2048,
            }
        },
        'openai': {
            'supported': False,  # OpenAI (包括GPT-3/4/5) has automatic caching
            'auto_cache': True
        },
        'gemini': {
            'supported': False,  # Gemini has automatic caching
            'auto_cache': True
        },
        'google': {  # Google provider的别名
            'supported': False,  # Google Gemini has automatic caching
            'auto_cache': True
        },
        'deepseek': {
            'supported': False,  # DeepSeek has automatic caching
            'auto_cache': True
        },
        'default': {
            'supported': False
        }
    }
    
    @staticmethod
    def detect_provider_type(model: str) -> str:
        """
        Detect provider type from model string
        Returns the actual provider name, not the model name
        """
        model_lower = model.lower()
        
        # Extract provider from "provider:model" format if present
        if ':' in model:
            provider_part = model.split(':')[0].lower()
            # Direct provider mapping
            if provider_part in ['anthropic', 'claude']:
                return 'anthropic'
            elif provider_part in ['openai', 'closeai', 'vercel']:
                return 'openai'  # All these use OpenAI-compatible APIs
            elif provider_part in ['google', 'gemini']:
                return 'gemini'
            elif provider_part == 'deepseek':
                return 'deepseek'
        
        # Fallback to model name detection for backward compatibility
        # Check for Claude/Anthropic models
        if 'claude' in model_lower or 'anthropic' in model_lower:
            return 'anthropic'
        # Check for OpenAI models (including GPT-3/4/5)
        elif any(x in model_lower for x in ['gpt-3', 'gpt-4', 'gpt-5', 'openai']):
            return 'openai'
        # Check for Gemini/Google models
        elif 'gemini' in model_lower or 'google' in model_lower:
            return 'gemini'
        # Check for DeepSeek
        elif 'deepseek' in model_lower:
            return 'deepseek'
        else:
            return 'default'
    
    @classmethod
    def _convert_cache_config(cls, cache_config: Any, target_provider: str, model_name: str = None) -> Optional[Dict]:
        """
        Convert unified cache configuration to provider-specific format
        
        Args:
            cache_config: CacheConfig object or dictionary
            target_provider: Target provider
            model_name: Model name (for minimum token requirements)
        
        Returns:
            Provider-specific cache configuration, or None if not supported
        """
        if not cache_config:
            return None
            
        # Get provider configuration
        provider_config = cls.PROVIDER_CACHE_SUPPORT.get(target_provider, {})
        
        # If provider doesn't support cache marking, return None
        if not provider_config.get('supported', False):
            return None
            
        # Convert CacheConfig object to dictionary
        if isinstance(cache_config, CacheConfig):
            config_dict = cache_config.to_dict()
        else:
            config_dict = cache_config
            
        # Use provider's format function
        formatter = provider_config.get('format')
        if formatter:
            return formatter(config_dict)
            
        return None
    
    @classmethod
    def _apply_cache_marking(cls, message: Dict, target_provider: str, model_name: str = None) -> Dict:
        """Apply cache marking to message based on provider requirements"""
        # Check for _cache field
        cache_config = message.pop('_cache', None)
        
        if not cache_config:
            return message
            
        # Convert cache configuration
        provider_cache = cls._convert_cache_config(cache_config, target_provider, model_name)
        
        if provider_cache:
            # Get provider's field name
            provider_config = cls.PROVIDER_CACHE_SUPPORT.get(target_provider, {})
            field_name = provider_config.get('field_name')
            if field_name:
                message[field_name] = provider_cache
        
        return message
    
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
                            },
                            'extra_content': getattr(tc, 'extra_content', None)
                        })
                else:
                    tool_calls.append({
                        'id': getattr(tc, 'id', ''),
                        'type': getattr(tc, 'type', 'function'),
                        'function': {
                            'name': getattr(tc.function, 'name', ''),
                            'arguments': getattr(tc.function, 'arguments', '')
                        },
                        'extra_content': getattr(tc, 'extra_content', None)
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
        # reasoning_content should NEVER be passed to any provider except certain OpenAI models
        always_remove = {'refusal', 'reasoning_content'}
        # OpenAI provider with certain models can handle reasoning_content
        if target_provider == 'openai' and preserve_reasoning:
            always_remove = {'refusal'}  # Some OpenAI models can handle reasoning_content
        
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
            
            # Handle reasoning content specially - NEVER pass to non-OpenAI providers
            if key == 'reasoning_content':
                # Only preserve for OpenAI models that support it
                if target_provider == 'openai' and preserve_reasoning and value:
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
            # Always normalize messages (including system, user, tool) to handle cache marking
            if isinstance(msg, dict):
                # Apply cache marking if present
                normalized_msg = msg.copy()
                normalized_msg = cls._apply_cache_marking(normalized_msg, target_provider, target_model)
                
                # For assistant messages, also apply other normalizations
                if normalized_msg.get('role') == 'assistant':
                    normalized_msg = cls.normalize_message(normalized_msg, target_provider)
                else:
                    # For non-assistant messages, just ensure required fields
                    normalized_msg = cls._ensure_required_fields(normalized_msg)
                    
                normalized.append(normalized_msg)
            else:
                # Non-dict messages go through full normalization
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
