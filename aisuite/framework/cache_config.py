"""
统一的缓存配置标准
为各种LLM provider提供统一的缓存抽象
"""
from typing import Optional, Dict, Any, Union
from enum import Enum


class CacheType(str, Enum):
    """缓存类型枚举"""
    NONE = "none"               # 不缓存
    EPHEMERAL = "ephemeral"     # 临时缓存（Claude支持）
    AUTOMATIC = "automatic"      # 自动缓存（OpenAI/Gemini/DeepSeek）
    # PERSISTENT = "persistent"  # 未来可能支持的持久缓存


class CacheTTL(str, Enum):
    """缓存时长枚举"""
    MINUTES_5 = "5m"   # 5分钟缓存
    HOURS_1 = "1h"     # 1小时缓存
    AUTO = "auto"      # 自动（provider决定）


class CacheConfig:
    """
    统一的缓存配置类
    业务层使用此类来标记消息缓存策略
    """
    
    def __init__(
        self,
        cache_type: Union[CacheType, str] = CacheType.NONE,
        ttl: Union[CacheTTL, str] = CacheTTL.MINUTES_5,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化缓存配置
        
        Args:
            cache_type: 缓存类型
            ttl: 缓存时长（仅对支持的provider有效）
            priority: 优先级（0-100），用于智能缓存选择
            metadata: 额外的元数据
        """
        # 支持字符串输入
        if isinstance(cache_type, str):
            self.cache_type = cache_type
        else:
            self.cache_type = cache_type.value if hasattr(cache_type, 'value') else cache_type
            
        if isinstance(ttl, str):
            self.ttl = ttl
        else:
            self.ttl = ttl.value if hasattr(ttl, 'value') else ttl
            
        self.priority = priority
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "type": self.cache_type,
            "ttl": self.ttl,
            "priority": self.priority,
        }
        # 添加元数据
        if self.metadata:
            result.update(self.metadata)
        return result
    
    @classmethod
    def system_prompt(cls, ttl: Union[CacheTTL, str] = CacheTTL.HOURS_1) -> "CacheConfig":
        """系统提示词的推荐配置（高优先级，长缓存）"""
        return cls(
            cache_type=CacheType.EPHEMERAL,
            ttl=ttl,
            priority=100
        )
    
    @classmethod
    def tool_definition(cls, ttl: Union[CacheTTL, str] = CacheTTL.HOURS_1) -> "CacheConfig":
        """工具定义的推荐配置（高优先级，长缓存）"""
        return cls(
            cache_type=CacheType.EPHEMERAL,
            ttl=ttl,
            priority=95
        )
    
    @classmethod
    def history_message(cls, age_ratio: float = 0.5) -> "CacheConfig":
        """
        历史消息的推荐配置
        age_ratio: 0-1之间，表示消息在历史中的位置（0=最新，1=最旧）
        """
        # 越旧的消息越适合缓存
        if age_ratio > 0.7:
            return cls(
                cache_type=CacheType.EPHEMERAL,
                ttl=CacheTTL.HOURS_1,  # 很旧的消息用长缓存
                priority=int(60 + age_ratio * 20)
            )
        elif age_ratio > 0.3:
            return cls(
                cache_type=CacheType.EPHEMERAL,
                ttl=CacheTTL.MINUTES_5,  # 中等历史用短缓存
                priority=int(40 + age_ratio * 20)
            )
        else:
            # 最新的30%不缓存
            return cls(cache_type=CacheType.NONE)
    
    @classmethod
    def tool_result(cls, size_bytes: int = 0) -> "CacheConfig":
        """
        工具调用结果的推荐配置
        size_bytes: 结果大小，用于决定是否缓存
        """
        if size_bytes > 1000:  # 大于1KB的结果才缓存
            return cls(
                cache_type=CacheType.EPHEMERAL,
                ttl=CacheTTL.MINUTES_5,
                priority=70
            )
        return cls(cache_type=CacheType.NONE)
    
    def __repr__(self) -> str:
        return f"CacheConfig(type={self.cache_type}, ttl={self.ttl}, priority={self.priority})"