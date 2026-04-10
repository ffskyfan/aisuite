# Anthropic provider
# Links:
# Tool calling docs - https://docs.anthropic.com/en/docs/build-with-claude/tool-use

import anthropic
import json
import logging
from typing import AsyncGenerator, Union
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.message import Message, ChatCompletionMessageToolCall, Function, ReasoningContent
from aisuite.framework.chat_completion_response import Choice, ChoiceDelta, StreamChoice
from aisuite.framework.replay_payload import (
    ProviderReplayCapabilities,
    ReplayBuildResult,
    ReplayCaptureResult,
    ReplayDiagnostic,
    ReplayValidationResult,
    build_replay_payload,
    get_replay_payload,
    unwrap_replay_payload,
)
from aisuite.framework.stop_reason import stop_reason_manager

# 设置专门的logger用于调试stop reason
anthropic_logger = logging.getLogger("anthropic_stop_reason")
anthropic_logger.setLevel(logging.INFO)
if not anthropic_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[STOP_REASON] %(message)s')
    handler.setFormatter(formatter)
    anthropic_logger.addHandler(handler)

# Define a constant for the default max_tokens value
DEFAULT_MAX_TOKENS = 4096
ANTHROPIC_THINKING_REPLAY_KIND = "anthropic_thinking_block"


def _build_anthropic_thinking_replay_payload(thinking_block_data):
    return build_replay_payload(
        "anthropic",
        ANTHROPIC_THINKING_REPLAY_KIND,
        thinking_block_data,
    )


def _extract_anthropic_thinking_block(raw_data):
    envelope = get_replay_payload(raw_data)
    if envelope and envelope.get("provider") == "anthropic":
        return unwrap_replay_payload(raw_data)
    return raw_data


class AnthropicMessageConverter:
    # Role constants
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_TOOL = "tool"
    ROLE_SYSTEM = "system"

    # Finish reason mapping
    FINISH_REASON_MAPPING = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }

    CACHE_TTL_ALIASES = {
        "cachettl.minutes_5": "5m",
        "minutes_5": "5m",
        "5m": "5m",
        "cachettl.hours_1": "1h",
        "hours_1": "1h",
        "1h": "1h",
    }

    def convert_request(self, messages):
        """Convert framework messages to Anthropic format."""
        system_message = self._extract_system_message(messages)
        converted_messages = [self._convert_single_message(msg) for msg in messages]
        return system_message, converted_messages

    def convert_response(self, response):
        """Normalize the response from the Anthropic API to match OpenAI's response format."""
        normalized_response = ChatCompletionResponse()
         # 创建一个Choice对象并添加到choices列表中，避免索引越界
        message = self._get_message(response)
        finish_reason = self._get_finish_reason(response)
        usage = self._get_usage_stats(response)

        # Create enhanced stop info for non-streaming response
        stop_info = self._create_stop_info_from_response(response)

        # 创建Choice对象并添加到列表中
        normalized_response.choices.append(Choice(
            index=0,
            message=message,
            finish_reason=finish_reason,
            stop_info=stop_info
        ))

        # 设置使用量统计
        normalized_response.metadata['usage'] = usage
        return normalized_response
        
    def convert_stream_response(self, chunk, model, provider=None):
        """Convert a streaming response chunk from Anthropic to the framework's format."""

        content = ""
        tool_calls = None
        role = None
        reasoning_content = None
        finish_reason = None
        stop_info = None
        usage = None

        # 获取chunk类型
        chunk_type = getattr(chunk, 'type', 'unknown')

        # Handle Claude-specific state events and streaming chunks.
        if chunk_type == "content_block_stop":
            if provider:
                index = getattr(chunk, "index", None)
                content_block = getattr(chunk, "content_block", None)

                # Prefer the SDK's fully parsed tool block when available.
                # This is more reliable than reconstructing arguments from
                # streamed partial_json fragments, which may not always form a
                # valid JSON string when concatenated directly.
                tool_calls = provider._process_anthropic_tool_use(content_block)
                if tool_calls:
                    provider._discard_anthropic_tool_call(index)
                else:
                    tool_calls = provider._finalize_anthropic_tool_calls(index)

                if tool_calls:
                    provider._stream_tool_calls_count += len(tool_calls)
                    role = "assistant"
                else:
                    return None
            else:
                return None

        elif chunk_type in ["message_stop", "ping"]:
            return None  # Filter out Claude-specific state events

        # Capture input-side usage from message_start (input_tokens, cache fields)
        elif chunk_type == "message_start":
            if provider:
                usage_obj = None
                if hasattr(chunk, 'message') and chunk.message:
                    usage_obj = getattr(chunk.message, 'usage', None)
                if not usage_obj:
                    usage_obj = getattr(chunk, 'usage', None)
                if usage_obj:
                    provider._patch_stream_usage(usage_obj)
            return None  # Still filter out message_start from downstream

        # Handle content_block_start event
        elif chunk_type == "content_block_start":
            if (hasattr(chunk, 'content_block') and chunk.content_block and
                hasattr(chunk.content_block, 'type')):
                block_type = chunk.content_block.type

                if block_type in ["tool_use", "server_tool_use"]:
                    # Initialize tool call with id and name from content_block_start
                    if provider:
                        provider._initialize_tool_call_from_content_block(chunk.content_block, getattr(chunk, 'index', 0))
                    # Don't pass this event downstream, just initialize internal state
                    return None
                elif block_type in ["text", "thinking"]:
                    # These are content block starts, filter them out
                    # The actual content will come in content_block_delta events
                    return None

        # Handle content_block_delta event
        elif chunk_type == "content_block_delta":
            if hasattr(chunk, 'delta') and chunk.delta:
                delta_type = getattr(chunk.delta, 'type', 'unknown')

                # Handle text delta
                if (delta_type == "text_delta" and
                    hasattr(chunk.delta, 'text') and chunk.delta.text):
                    content = chunk.delta.text
                    role = "assistant"

                    # Track content in provider state
                    if provider:
                        provider._stream_has_content = True
                        provider._stream_content_length += len(content)

                # Handle thinking delta (reasoning content)
                elif (delta_type == "thinking_delta" and
                      hasattr(chunk.delta, 'thinking') and chunk.delta.thinking):
                    # Accumulate thinking content in provider
                    if provider:
                        provider._accumulate_thinking_content(chunk.delta.thinking)
                        provider._stream_reasoning_length += len(chunk.delta.thinking)
                    reasoning_content = chunk.delta.thinking
                    role = "assistant"

                # Handle signature delta (for thinking blocks)
                elif (delta_type == "signature_delta" and
                      hasattr(chunk.delta, 'signature') and chunk.delta.signature):
                    # Accumulate signature in provider
                    if provider:
                        provider._accumulate_thinking_signature(chunk.delta.signature)
                    # Don't pass signature deltas downstream
                    return None

                # Handle tool use delta (fine-grained streaming)
                elif (delta_type == "input_json_delta" and
                      hasattr(chunk.delta, 'partial_json')):
                    # This is part of tool input streaming
                    if provider:
                        tool_calls = provider._accumulate_anthropic_tool_calls(chunk)
                        if tool_calls:
                            provider._stream_tool_calls_count = len(tool_calls)
                    role = "assistant" if tool_calls else None

                else:
                    return None  # Filter out unknown delta types

        # Handle message_delta event
        elif chunk_type == "message_delta":
            if hasattr(chunk, 'delta') and chunk.delta:
                if hasattr(chunk.delta, 'stop_reason'):
                    original_stop_reason = chunk.delta.stop_reason
                    finish_reason = self._get_finish_reason(chunk.delta)

                    # Anthropic splits usage across two streaming events:
                    #   - message_start.message.usage → input_tokens, cache fields
                    #   - message_delta.usage → output_tokens
                    # We patch the output-side here, then build the final merged usage.
                    usage_obj = getattr(chunk, "usage", None)
                    if not usage_obj and hasattr(chunk.delta, "usage"):
                        usage_obj = chunk.delta.usage

                    if provider:
                        if usage_obj:
                            provider._patch_stream_usage(usage_obj)
                        normalized_usage = self._normalize_usage_obj(
                            provider._build_stream_usage()
                        )
                        if normalized_usage:
                            usage = normalized_usage

                    # Create enhanced stop info using stream-wide statistics
                    metadata = {
                        "has_content": provider._stream_has_content if provider else bool(content),
                        "content_length": provider._stream_content_length if provider else 0,
                        "tool_calls_count": provider._stream_tool_calls_count if provider else 0,
                        "reasoning_content_length": provider._stream_reasoning_length if provider else 0,
                        "chunk_type": chunk_type,
                        "model": model
                    }

                    stop_info = stop_reason_manager.map_stop_reason(
                        "anthropic", original_stop_reason, metadata
                    )

                    anthropic_logger.info(f"Stop Reason: {original_stop_reason} -> {stop_info.reason.value} (has_content: {metadata.get('has_content', False)})")
                    anthropic_logger.info(f"[PROVIDER] Created stop_info object: {stop_info}")

        # Handle error event
        elif chunk_type == "error":
            anthropic_logger.debug(f"CHUNK: error={getattr(chunk, 'error', 'unknown')}")
            # Create error stop info
            error_msg = getattr(chunk, 'error', 'unknown')
            stop_info = stop_reason_manager.map_stop_reason(
                "anthropic", "error", {"error_message": error_msg, "model": model}
            )

            return ChatCompletionResponse(
                choices=[
                    StreamChoice(
                        index=0,
                        delta=ChoiceDelta(content=None, role=None),
                        finish_reason="error",
                        stop_info=stop_info
                    )
                ],
                metadata={
                    'id': getattr(chunk, 'id', None),
                    'error': error_msg,
                    'model': model
                }
            )

        # Handle unknown event types
        else:
            anthropic_logger.debug(f"CHUNK: unknown_type={chunk_type}")
            return None  # Filter out unknown event types

        # Only pass downstream events that have actual content or important state changes
        if content or tool_calls or reasoning_content or finish_reason:
            anthropic_logger.debug(f"RESULT: content={bool(content)}, tools={len(tool_calls) if tool_calls else 0}, reasoning={bool(reasoning_content)}, finish={finish_reason}")

            # Log stop_info transmission
            if stop_info:
                anthropic_logger.info(f"[PROVIDER] Transmitting stop_info to downstream: {stop_info.reason.value}")
            else:
                anthropic_logger.info(f"[PROVIDER] No stop_info to transmit")

            metadata = {
                'id': getattr(chunk, 'id', None),
                'created': None,  # Anthropic doesn't provide timestamp in chunks
                'model': model,
            }
            if usage:
                metadata['usage'] = usage

            return ChatCompletionResponse(
                choices=[
                    StreamChoice(
                        index=0,
                        delta=ChoiceDelta(
                            content=content if content else None,
                            role=role,
                            tool_calls=tool_calls,
                            reasoning_content=reasoning_content,
                        ),
                        finish_reason=finish_reason,
                        stop_info=stop_info,  # Enhanced stop information
                    )
                ],
                metadata=metadata,
            )
        else:
            # No meaningful content to pass downstream
            anthropic_logger.debug(f"CHUNK: filtered_empty_content")
            return None

    def _convert_single_message(self, msg):
        """Convert a single message to Anthropic format."""
        if isinstance(msg, dict):
            return self._convert_dict_message(msg)
        return self._convert_message_object(msg)

    @classmethod
    def _normalize_cache_ttl(cls, ttl):
        if hasattr(ttl, "value"):
            ttl = ttl.value
        if ttl is None:
            return None
        ttl_str = str(ttl).strip()
        if not ttl_str:
            return None
        normalized = cls.CACHE_TTL_ALIASES.get(ttl_str.lower())
        return normalized or ttl_str

    @classmethod
    def _normalize_cache_control(cls, cache_control):
        if not isinstance(cache_control, dict):
            return cache_control

        normalized = cache_control.copy()
        if "type" in normalized and hasattr(normalized["type"], "value"):
            normalized["type"] = normalized["type"].value

        ttl = cls._normalize_cache_ttl(normalized.get("ttl"))
        if ttl is not None:
            normalized["ttl"] = ttl

        return normalized

    def _apply_cache_control_to_content(self, content, cache_control):
        """Apply Anthropic cache_control marking to message content blocks."""
        cache_control = self._normalize_cache_control(cache_control)
        if not cache_control:
            return content

        if isinstance(content, list):
            blocks = []
            for block in content:
                if isinstance(block, dict):
                    blocks.append(block.copy())
                else:
                    blocks.append(block)

            for index in range(len(blocks) - 1, -1, -1):
                block = blocks[index]
                if isinstance(block, dict):
                    block.setdefault("cache_control", cache_control)
                    break
            return blocks

        if content is None:
            content = ""
        return [{"type": "text", "text": str(content), "cache_control": cache_control}]

    def _convert_dict_message(self, msg):
        """Convert a dictionary message to Anthropic format."""
        cache_control = self._normalize_cache_control(msg.get("cache_control"))
        if msg["role"] == self.ROLE_TOOL:
            return self._create_tool_result_message(
                msg["tool_call_id"],
                msg.get("content"),
                cache_control=cache_control,
            )
        elif msg["role"] == self.ROLE_ASSISTANT and "tool_calls" in msg:
            reasoning_content = msg.get("reasoning_content")

            # 处理reasoning_content：可能是字符串（旧格式）或ReasoningContent对象（新格式）
            if reasoning_content:
                if isinstance(reasoning_content, dict):
                    reasoning_content = ReasoningContent(**reasoning_content)


            return self._create_assistant_tool_message(
                msg["content"],
                msg["tool_calls"],
                reasoning_content,  # 传递 reasoning_content（可能是ReasoningContent对象）
                cache_control=cache_control,
            )

        content = self._apply_cache_control_to_content(msg.get("content"), cache_control)
        return {"role": msg["role"], "content": content}

    def _convert_message_object(self, msg):
        """Convert a Message object to Anthropic format."""
        if msg.role == self.ROLE_TOOL:
            return self._create_tool_result_message(msg.tool_call_id, msg.content)
        elif msg.role == self.ROLE_ASSISTANT and msg.tool_calls:
            reasoning_content = msg.reasoning_content
            return self._create_assistant_tool_message(
                msg.content,
                msg.tool_calls,
                reasoning_content  # 传递 reasoning_content（ReasoningContent对象）
            )
        return {"role": msg.role, "content": msg.content}

    def _create_tool_result_message(self, tool_call_id, content, cache_control=None):
        """Create a tool result message in Anthropic format."""
        cache_control = self._normalize_cache_control(cache_control)
        tool_result = {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": content,
        }
        if cache_control:
            tool_result["cache_control"] = cache_control

        return {
            "role": self.ROLE_USER,
            "content": [
                tool_result
            ],
        }

    def _create_assistant_tool_message(self, content, tool_calls, reasoning_content=None, cache_control=None):
        """Create an assistant message with tool calls in Anthropic format."""
        cache_control = self._normalize_cache_control(cache_control)
        message_content = []

        # 1. 如果有ReasoningContent，使用其原始数据重构thinking块
        if reasoning_content and hasattr(reasoning_content, 'raw_data') and reasoning_content.raw_data:
            thinking_block = _extract_anthropic_thinking_block(reasoning_content.raw_data)
            if thinking_block:
                message_content.append(thinking_block)

        # 2. 添加文本内容
        if content:
            message_content.append({"type": "text", "text": content})

        # 3. 添加工具调用
        for tool_call in tool_calls:
            tool_input = (
                tool_call["function"]["arguments"]
                if isinstance(tool_call, dict)
                else tool_call.function.arguments
            )
            message_content.append(
                {
                    "type": "tool_use",
                    "id": (
                        tool_call["id"] if isinstance(tool_call, dict) else tool_call.id
                    ),
                    "name": (
                        tool_call["function"]["name"]
                        if isinstance(tool_call, dict)
                        else tool_call.function.name
                    ),
                    "input": json.loads(tool_input),
                }
            )

        if cache_control:
            message_content = self._apply_cache_control_to_content(message_content, cache_control)
        return {"role": self.ROLE_ASSISTANT, "content": message_content}

    def _extract_system_message(self, messages):
        """Extract system message if present, otherwise return empty list."""
        # TODO: This is a temporary solution to extract the system message.
        # User can pass multiple system messages, which can mingled with other messages.
        # This needs to be fixed to handle this case.
        if messages and messages[0].get("role") == "system":
            system_msg = messages.pop(0)
            system_content = system_msg.get("content")
            cache_control = system_msg.get("cache_control")
            if cache_control:
                return self._apply_cache_control_to_content(system_content, cache_control)
            return system_content
        return []

    def _get_finish_reason(self, response):
        """Get the normalized finish reason."""
        return self.FINISH_REASON_MAPPING.get(response.stop_reason, "stop")

    def _create_stop_info_from_response(self, response):
        """Create StopInfo from non-streaming response."""
        # Analyze response content to determine if it has content
        has_content = False
        content_length = 0
        tool_calls_count = 0
        reasoning_length = 0

        if hasattr(response, 'content') and response.content:
            for content_block in response.content:
                if hasattr(content_block, 'type'):
                    if content_block.type == 'text' and hasattr(content_block, 'text'):
                        has_content = True
                        content_length += len(content_block.text)
                    elif content_block.type == 'tool_use':
                        tool_calls_count += 1

        metadata = {
            "has_content": has_content,
            "content_length": content_length,
            "tool_calls_count": tool_calls_count,
            "reasoning_content_length": reasoning_length,
            "response_type": "non_streaming"
        }

        original_stop_reason = getattr(response, 'stop_reason', 'end_turn')
        return stop_reason_manager.map_stop_reason("anthropic", original_stop_reason, metadata)

    def _normalize_usage_obj(self, usage_obj):
        """Normalize Anthropic usage object to the unified AISuite format."""
        if not usage_obj:
            return None

        def _get_int(container, key, default=None):
            value = None
            if isinstance(container, dict):
                value = container.get(key)
            else:
                value = getattr(container, key, None)
            if value is None:
                return default
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _deep_get(container, *path):
            current = container
            for key in path:
                if current is None:
                    return None
                if isinstance(current, dict):
                    current = current.get(key)
                else:
                    current = getattr(current, key, None)
            return current

        # Anthropic usage object typically has `input_tokens` and `output_tokens`
        input_tokens = _get_int(usage_obj, "input_tokens")
        output_tokens = _get_int(usage_obj, "output_tokens")
        if input_tokens is None or output_tokens is None:
            return None

        cache_read_input_tokens = _get_int(usage_obj, "cache_read_input_tokens", 0) or 0
        cache_write_input_tokens = _get_int(usage_obj, "cache_creation_input_tokens", 0) or 0
        if cache_read_input_tokens < 0:
            cache_read_input_tokens = 0
        if cache_write_input_tokens < 0:
            cache_write_input_tokens = 0

        cache_creation = _deep_get(usage_obj, "cache_creation") or {}
        ephemeral_5m_input_tokens = _get_int(cache_creation, "ephemeral_5m_input_tokens", 0) or 0
        ephemeral_1h_input_tokens = _get_int(cache_creation, "ephemeral_1h_input_tokens", 0) or 0
        if ephemeral_5m_input_tokens < 0:
            ephemeral_5m_input_tokens = 0
        if ephemeral_1h_input_tokens < 0:
            ephemeral_1h_input_tokens = 0

        # Anthropic semantics: total input tokens = input_tokens + cache_read_input_tokens + cache_creation_input_tokens
        prompt_tokens = input_tokens + cache_read_input_tokens + cache_write_input_tokens

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
            "cache_write_input_tokens": cache_write_input_tokens,
            "cache_write_by_ttl": {
                "ephemeral_5m_input_tokens": ephemeral_5m_input_tokens,
                "ephemeral_1h_input_tokens": ephemeral_1h_input_tokens,
            },
        }

    def _get_usage_stats(self, response):
        """Get the usage statistics for non-streaming responses."""
        return self._normalize_usage_obj(getattr(response, "usage", None))

    def _get_message(self, response):
        """Get the appropriate message based on response type."""
        if response.stop_reason == "tool_use":
            tool_message = self.convert_response_with_tool_use(response)
            if tool_message:
                return tool_message

        # Extract thinking content if present
        reasoning_content = None
        text_content = None
        thinking_block_data = None

        for content_block in response.content:
            if hasattr(content_block, 'type'):
                if content_block.type == "thinking" and hasattr(content_block, 'thinking'):
                    # 保存完整的thinking块数据
                    thinking_block_data = {}
                    if hasattr(content_block, 'model_dump'):
                        thinking_block_data = content_block.model_dump()
                    elif hasattr(content_block, 'dict'):
                        thinking_block_data = content_block.dict()
                    else:
                        # 手动构建thinking块数据
                        thinking_block_data = {
                            'type': 'thinking',
                            'thinking': content_block.thinking
                        }
                        # 尝试获取signature
                        if hasattr(content_block, 'signature'):
                            thinking_block_data['signature'] = content_block.signature

                    # 创建ReasoningContent对象
                    reasoning_content = ReasoningContent(
                        thinking=content_block.thinking,
                        signature=thinking_block_data.get('signature'),
                        provider="anthropic",
                        raw_data=_build_anthropic_thinking_replay_payload(thinking_block_data)
                    )
                elif content_block.type == "text" and hasattr(content_block, 'text'):
                    text_content = content_block.text

        return Message(
            content=text_content or (response.content[0].text if response.content else None),
            role="assistant",
            tool_calls=None,
            refusal=None,
            reasoning_content=reasoning_content
        )

    def convert_response_with_tool_use(self, response):
        """Convert Anthropic tool use response to the framework's format."""
        tool_call = next(
            (content for content in response.content if content.type == "tool_use"),
            None,
        )

        if tool_call:
            function = Function(
                name=tool_call.name, arguments=json.dumps(tool_call.input)
            )
            tool_call_obj = ChatCompletionMessageToolCall(
                id=tool_call.id, function=function, type="function"
            )
            text_content = next(
                (
                    content.text
                    for content in response.content
                    if content.type == "text"
                ),
                "",
            )
            
            # Extract thinking content if present
            reasoning_content = None
            thinking_block_data = None

            for content_block in response.content:
                if hasattr(content_block, 'type') and content_block.type == "thinking" and hasattr(content_block, 'thinking'):
                    # 保存完整的thinking块数据
                    thinking_block_data = {}
                    if hasattr(content_block, 'model_dump'):
                        thinking_block_data = content_block.model_dump()
                    elif hasattr(content_block, 'dict'):
                        thinking_block_data = content_block.dict()
                    else:
                        # 手动构建thinking块数据
                        thinking_block_data = {
                            'type': 'thinking',
                            'thinking': content_block.thinking
                        }
                        # 尝试获取signature
                        if hasattr(content_block, 'signature'):
                            thinking_block_data['signature'] = content_block.signature

                    # 创建ReasoningContent对象
                    reasoning_content = ReasoningContent(
                        thinking=content_block.thinking,
                        signature=thinking_block_data.get('signature'),
                        provider="anthropic",
                        raw_data=_build_anthropic_thinking_replay_payload(thinking_block_data)
                    )
                    break

            return Message(
                content=text_content or None,
                tool_calls=[tool_call_obj] if tool_call else None,
                role="assistant",
                refusal=None,
                reasoning_content=reasoning_content
            )
        return None

    def convert_tool_spec(self, openai_tools):
        """Convert OpenAI tool specification to Anthropic format."""
        anthropic_tools = []

        for tool in openai_tools:
            if tool.get("type") != "function":
                continue

            function = tool["function"]
            anthropic_tool = {
                "name": function["name"],
                "description": function["description"],
                "input_schema": {
                    "type": "object",
                    "properties": function["parameters"]["properties"],
                    "required": function["parameters"].get("required", []),
                },
            }
            if tool.get("cache_control"):
                anthropic_tool["cache_control"] = self._normalize_cache_control(
                    tool["cache_control"]
                )
            anthropic_tools.append(anthropic_tool)

        return anthropic_tools


class AnthropicProvider(Provider):
    def __init__(self, **config):
        """Initialize the Anthropic provider with the given configuration."""
        self.client = anthropic.AsyncAnthropic(**config)
        self.converter = AnthropicMessageConverter()

        # State for accumulating streaming tool calls
        self._streaming_tool_calls = {}
        # Track thinking state for error recovery
        self._thinking_enabled = False
        # State for accumulating thinking content
        self._streaming_thinking = {
            "thinking": "",
            "signature": None
        }

        # State for tracking content across streaming session
        self._stream_has_content = False
        self._stream_content_length = 0
        self._stream_tool_calls_count = 0
        self._stream_reasoning_length = 0
        # Structured state for accumulating usage across streaming events
        self._stream_usage_state = self._make_empty_stream_usage_state()

    def get_replay_capabilities(self, model: str | None = None) -> ProviderReplayCapabilities:
        return ProviderReplayCapabilities(
            needs_exact_turn_replay=False,
            needs_provider_call_id_binding=True,
            needs_reasoning_raw_replay=True,
            supports_canonical_only_history=True,
            empty_actionless_stop_is_retryable=True,
        )

    def validate_replay_window(self, model: str, messages: list, **kwargs) -> ReplayValidationResult:
        diagnostics: list[ReplayDiagnostic] = []
        normalized_messages: list[dict] = []
        for msg in messages:
            if isinstance(msg, dict):
                normalized_messages.append(msg)
            elif hasattr(msg, "model_dump"):
                normalized_messages.append(msg.model_dump())

        for msg in normalized_messages:
            role = msg.get("role")
            if role == "tool" and not msg.get("tool_call_id"):
                diagnostics.append(
                    ReplayDiagnostic(
                        code="missing_tool_call_id",
                        message="Anthropic tool result replay requires tool_call_id.",
                        provider="anthropic",
                    )
                )
            if role == "assistant" and msg.get("tool_calls"):
                for tool_call in msg.get("tool_calls", []):
                    tool_call_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)
                    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else getattr(tool_call, "function", None)
                    function_name = function.get("name") if isinstance(function, dict) else getattr(function, "name", None)
                    if not tool_call_id:
                        diagnostics.append(
                            ReplayDiagnostic(
                                code="missing_tool_call_id",
                                message="Anthropic assistant tool call is missing id.",
                                provider="anthropic",
                            )
                        )
                    if not function_name:
                        diagnostics.append(
                            ReplayDiagnostic(
                                code="missing_tool_function_name",
                                message="Anthropic assistant tool call is missing function name.",
                                provider="anthropic",
                                metadata={"tool_call_id": tool_call_id},
                            )
                        )

        return ReplayValidationResult(
            ok=not any(diag.severity == "error" for diag in diagnostics),
            diagnostics=tuple(diagnostics),
        )

    def capture_response(self, response, model: str | None = None, **kwargs):
        if not response or not getattr(response, "choices", None):
            return ReplayCaptureResult()
        choice = response.choices[0]
        message = getattr(choice, "message", None)
        if not message:
            return ReplayCaptureResult(stop_info=getattr(choice, "stop_info", None))
        reasoning_content = getattr(message, "reasoning_content", None)
        return ReplayCaptureResult(
            canonical_message=message,
            stop_info=getattr(choice, "stop_info", None),
            replay_metadata={
                "tool_calls": getattr(message, "tool_calls", None),
                "reasoning_content": getattr(reasoning_content, "raw_data", None) if reasoning_content else None,
            },
            protocol_diagnostics=(),
        )

    def build_replay_view(self, model: str, messages: list, **kwargs):
        validation = self.validate_replay_window(model, messages, **kwargs)
        if not validation.ok:
            error_codes = ", ".join(diag.code for diag in validation.diagnostics if diag.severity == "error")
            raise LLMError(f"Anthropic replay window validation failed: {error_codes}")
        system_message, converted_messages = self.converter.convert_request(messages)
        return ReplayBuildResult(
            request_view={
                "system": system_message,
                "messages": converted_messages,
            },
            replay_mode="anthropic_messages",
            degraded=validation.degraded,
            diagnostics=validation.diagnostics,
        )

    # ---- Stream usage accumulation (patch / build) ----

    @staticmethod
    def _make_empty_stream_usage_state():
        """Return a fresh, zeroed-out stream usage state dict."""
        return {
            "input_tokens": None,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 0,
                "ephemeral_1h_input_tokens": 0,
            },
        }

    def _patch_stream_usage(self, usage_obj):
        """Patch *_stream_usage_state* with fields present in *usage_obj*.

        Called once for message_start (input side) and once for message_delta
        (output side).  Only non-None values overwrite the current state so that
        the two events complement each other.
        """
        def _get(key):
            if isinstance(usage_obj, dict):
                return usage_obj.get(key)
            return getattr(usage_obj, key, None)

        for key in ("input_tokens", "output_tokens",
                     "cache_read_input_tokens", "cache_creation_input_tokens"):
            val = _get(key)
            if val is not None:
                self._stream_usage_state[key] = int(val)

        # Nested cache_creation fields
        cache_creation_obj = _get("cache_creation")
        if cache_creation_obj:
            cc = self._stream_usage_state["cache_creation"]
            for sub_key in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"):
                sub_val = (cache_creation_obj.get(sub_key) if isinstance(cache_creation_obj, dict)
                           else getattr(cache_creation_obj, sub_key, None))
                if sub_val is not None:
                    cc[sub_key] = int(sub_val)

    def _build_stream_usage(self):
        """Build a merged usage dict from accumulated stream state.

        Returns a dict with both input_tokens and output_tokens guaranteed
        present (input_tokens defaults to 0 if message_start was somehow
        missed), so _normalize_usage_obj will not return None.
        """
        s = self._stream_usage_state
        input_tokens = s["input_tokens"] if s["input_tokens"] is not None else 0
        return {
            "input_tokens": input_tokens,
            "output_tokens": s["output_tokens"],
            "cache_read_input_tokens": s["cache_read_input_tokens"],
            "cache_creation_input_tokens": s["cache_creation_input_tokens"],
            "cache_creation": dict(s["cache_creation"]),
        }

    async def chat_completions_create(self, model, messages, stream: bool = False, **kwargs) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionResponse, None]]:
        """Create a chat completion using the Anthropic API."""
        replay_request_view = kwargs.pop("_replay_request_view", None)
        replay_mode = kwargs.pop("_replay_mode", None)
        kwargs = self._prepare_kwargs(kwargs)

        # Track thinking state for error recovery
        self._thinking_enabled = "thinking" in kwargs

        if replay_request_view is not None and replay_mode == "anthropic_messages":
            if isinstance(replay_request_view, dict):
                system_message = replay_request_view.get("system")
                converted_messages = replay_request_view.get("messages") or []
            else:
                system_message, converted_messages = self.converter.convert_request(replay_request_view)
        else:
            system_message, converted_messages = self.converter.convert_request(messages)



        # Fix thinking blocks proactively when thinking is enabled
        # This MUST be done before API call to prevent errors and resource waste
        if self._thinking_enabled:
            converted_messages = self._fix_thinking_messages(converted_messages)

            # If thinking was disabled during fix due to incompatible messages,
            # remove thinking from kwargs to ensure consistency
            if not self._thinking_enabled:
                anthropic_logger.info("Thinking was automatically disabled due to incompatible messages")
                kwargs = kwargs.copy()
                kwargs.pop("thinking", None)


        try:
            if stream:
                # Reset streaming tool calls state
                self._streaming_tool_calls = {}
                # Reset thinking accumulation state
                self._streaming_thinking = {
                    "thinking": "",
                    "signature": None
                }
                # Reset content tracking state
                self._stream_has_content = False
                self._stream_content_length = 0
                self._stream_tool_calls_count = 0
                self._stream_reasoning_length = 0
                self._stream_usage_state = self._make_empty_stream_usage_state()

                response = await self.client.messages.create(
                    model=model,
                    system=system_message,
                    messages=converted_messages,
                    stream=True,
                    **kwargs
                )

                async def stream_generator():
                    chunk_count = 0
                    yielded_count = 0
                    async for chunk in response:
                        chunk_count += 1
                        result = self.converter.convert_stream_response(chunk, model, self)
                        if result is not None:  # Only yield non-None results
                            yielded_count += 1
                            yield result
                    anthropic_logger.debug(f"STREAM_END: total_chunks={chunk_count}, yielded={yielded_count}")

                return stream_generator()
            else:
                response = await self.client.messages.create(
                    model=model,
                    system=system_message,
                    messages=converted_messages,
                    **kwargs
                )
                return self.converter.convert_response(response)

        except Exception as e:
            anthropic_logger.error(f"Error in chat_completions_create: {e}")
            # Do not retry - this would cause resource waste
            # All message fixes should be done before the first API call
            raise

    def _prepare_kwargs(self, kwargs):
        """Prepare kwargs for the API call."""
        kwargs = kwargs.copy()
        kwargs.setdefault("max_tokens", DEFAULT_MAX_TOKENS)

        if "tools" in kwargs:
            kwargs["tools"] = self.converter.convert_tool_spec(kwargs["tools"])

        return kwargs

    def _initialize_tool_call_from_content_block(self, content_block, index):
        """
        Initialize tool call from content_block_start event.

        Args:
            content_block: The tool_use content block containing id and name
            index: The index of the content block
        """
        if (
            not content_block
            or not hasattr(content_block, 'type')
            or content_block.type not in {"tool_use", "server_tool_use"}
        ):
            return

        initial_input = getattr(content_block, "input", None)
        serialized_input = ""
        seeded_from_start = False
        if initial_input is not None:
            try:
                serialized_input = json.dumps(initial_input)
                seeded_from_start = True
            except TypeError:
                serialized_input = ""
                seeded_from_start = False

        # Initialize tool call accumulator with id and name from content_block_start
        self._streaming_tool_calls[index] = {
            "id": getattr(content_block, 'id', ''),
            "name": getattr(content_block, 'name', ''),
            "input": serialized_input,
            "seeded_from_start": seeded_from_start,
            "received_delta": False,
        }

    def _discard_anthropic_tool_call(self, index):
        """Drop a pending streaming tool call accumulator entry."""
        if index is None:
            return
        self._streaming_tool_calls.pop(index, None)

    def _accumulate_anthropic_tool_calls(self, chunk):
        """
        Accumulate tool call chunks from Anthropic fine-grained streaming.

        Args:
            chunk: The streaming chunk containing tool input delta

        Returns:
            List of converted tool calls if any are complete, None otherwise
        """
        if not hasattr(chunk, 'delta') or not chunk.delta:
            return None

        # Get the tool use index from the chunk
        index = getattr(chunk, 'index', 0)

        # Tool call should already be initialized by _initialize_tool_call_from_content_block
        # If not, initialize with empty values (fallback)
        if index not in self._streaming_tool_calls:
            self._streaming_tool_calls[index] = {
                "id": "",
                "name": "",
                "input": "",
                "seeded_from_start": False,
                "received_delta": False,
            }

        tool_call = self._streaming_tool_calls[index]

        # Accumulate partial JSON input
        if hasattr(chunk.delta, 'partial_json'):
            if tool_call.get("seeded_from_start") and not tool_call.get("received_delta"):
                tool_call["input"] = ""
            tool_call["input"] += chunk.delta.partial_json
            tool_call["received_delta"] = True

        return self._finalize_anthropic_tool_calls()

    def _finalize_anthropic_tool_calls(self, index=None):
        """
        Finalize accumulated Anthropic tool calls.

        Args:
            index: Optional specific content block index to finalize.

        Returns:
            List of converted tool calls if any are complete, None otherwise.
        """
        complete_tool_calls = []
        for idx, tool_call_data in list(self._streaming_tool_calls.items()):
            if index is not None and idx != index:
                continue

            if tool_call_data["id"] and tool_call_data["name"]:
                raw_arguments = tool_call_data.get("input") or ""
                if not raw_arguments and not tool_call_data.get("received_delta"):
                    raw_arguments = "{}"

                if not raw_arguments:
                    continue

                try:
                    # Try to parse input as JSON to ensure completeness
                    json.loads(raw_arguments)

                    function = Function(
                        name=tool_call_data["name"],
                        arguments=raw_arguments
                    )

                    tool_call_obj = ChatCompletionMessageToolCall(
                        id=tool_call_data["id"],
                        function=function,
                        type="function"
                    )

                    complete_tool_calls.append(tool_call_obj)
                    anthropic_logger.debug(f"TOOL_COMPLETE: {tool_call_data['name']}")

                    # Remove completed tool call from accumulator
                    del self._streaming_tool_calls[idx]

                except json.JSONDecodeError:
                    # Input is not complete yet, continue accumulating
                    continue

        return complete_tool_calls if complete_tool_calls else None

    def _accumulate_thinking_content(self, thinking_text):
        """Accumulate thinking content from streaming chunks."""
        self._streaming_thinking["thinking"] += thinking_text

    def _accumulate_thinking_signature(self, signature):
        """Accumulate signature from streaming chunks."""
        self._streaming_thinking["signature"] = signature

    def _get_accumulated_thinking(self):
        """Get accumulated thinking content and reset state."""
        thinking_data = self._streaming_thinking.copy()
        # Reset for next stream
        self._streaming_thinking = {
            "thinking": "",
            "signature": None
        }
        return thinking_data



    def _fix_thinking_messages(self, messages):
        """
        Fix messages to ensure proper thinking blocks when thinking is enabled.

        Based on Claude documentation:
        - When thinking is enabled, final assistant message with tool_use must start with thinking block
        - Thinking blocks must be preserved during tool use for reasoning continuity
        - Only thinking blocks with signature (from Claude) are valid

        Strategy:
        1. Remove thinking blocks without signature (from other models)
        2. Keep thinking blocks with signature (from Claude)
        3. Find the actual final assistant message and ensure it has thinking if it has tool_use
        """
        if not self._thinking_enabled:
            return messages

        fixed_messages = []

        # First pass: clean up all messages and find valid thinking blocks
        valid_thinking_blocks = []

        for i, message in enumerate(messages):
            if (isinstance(message, dict) and
                message.get("role") == "assistant" and
                isinstance(message.get("content"), list)):

                content_blocks = message["content"]

                # Step 1: Filter out thinking blocks without signature
                filtered_blocks = []
                for block in content_blocks:
                    if block.get("type") in ["thinking", "redacted_thinking"]:
                        # Only keep thinking blocks that have a signature (from Claude)
                        if "signature" in block or "data" in block:  # data for redacted_thinking
                            anthropic_logger.debug("Keeping Claude-generated thinking block with signature")
                            filtered_blocks.append(block)
                            # Also collect for potential reuse
                            valid_thinking_blocks.append(block.copy())
                        else:
                            anthropic_logger.debug("Removing thinking block without signature (from other model)")
                            # Skip this block - it's from another model
                    else:
                        filtered_blocks.append(block)

                # Create the cleaned message
                if filtered_blocks != content_blocks:
                    fixed_message = message.copy()
                    fixed_message["content"] = filtered_blocks
                    fixed_messages.append(fixed_message)
                else:
                    fixed_messages.append(message)
            else:
                fixed_messages.append(message)

        # Second pass: find the actual final assistant message and fix it if needed
        final_assistant_index = -1
        for i in range(len(fixed_messages) - 1, -1, -1):
            if fixed_messages[i].get("role") == "assistant":
                final_assistant_index = i
                break

        if final_assistant_index >= 0:
            final_assistant = fixed_messages[final_assistant_index]
            if isinstance(final_assistant.get("content"), list):
                content = final_assistant["content"]
                has_tool_use = any(block.get("type") == "tool_use" for block in content)
                has_valid_thinking = any(
                    block.get("type") in ["thinking", "redacted_thinking"]
                    for block in content
                )

                # If final assistant has tool_use but no thinking, fix it
                if has_tool_use and not has_valid_thinking:
                    if valid_thinking_blocks:
                        # Use the most recent valid thinking block
                        thinking_block = valid_thinking_blocks[-1]
                        anthropic_logger.info(
                            f"Moving valid thinking block to final assistant message (index {final_assistant_index})"
                        )
                        new_content = [thinking_block] + content
                        fixed_messages[final_assistant_index] = {
                            **final_assistant,
                            "content": new_content
                        }
                    else:
                        anthropic_logger.warning(
                            "No valid thinking blocks found - disabling thinking to prevent API errors"
                        )
                        # Disable thinking to prevent API errors
                        self._thinking_enabled = False
                        return messages

        return fixed_messages

    def _process_anthropic_tool_use(self, content_block):
        """
        Process complete Anthropic tool use block.

        Args:
            content_block: The tool use content block

        Returns:
            List of converted tool calls
        """
        if (
            not content_block
            or not hasattr(content_block, 'type')
            or content_block.type not in {"tool_use", "server_tool_use"}
        ):
            return None

        try:
            tool_name = getattr(content_block, "name", None)
            tool_call_id = getattr(content_block, "id", None)
            if not tool_name or not tool_call_id:
                return None

            raw_input = getattr(content_block, "input", None)
            arguments = "{}" if raw_input is None else json.dumps(raw_input)

            function = Function(
                name=tool_name,
                arguments=arguments
            )

            tool_call_obj = ChatCompletionMessageToolCall(
                id=tool_call_id,
                function=function,
                type="function"
            )

            return [tool_call_obj]

        except Exception:
            # If there's any error processing the tool use, return None
            return None
