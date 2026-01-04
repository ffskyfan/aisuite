import openai
import os
import json
import hashlib
from typing import AsyncGenerator, Union, List, Dict, Any

from aisuite.provider import Provider, LLMError
from aisuite.framework.chat_completion_response import ChatCompletionResponse, Choice, ChoiceDelta, StreamChoice
from aisuite.framework.message import Message, ChatCompletionMessageToolCall, Function, ReasoningContent
from aisuite.framework.stop_reason import stop_reason_manager


class OpenaiProvider(Provider):
    # OpenAI finish_reason mapping to standard StopReason
    FINISH_REASON_MAPPING = {
        "stop": "complete",
        "length": "length_limit",
        "tool_calls": "tool_call",
        "function_call": "tool_call",  # Legacy function call
        "content_filter": "safety_refusal",
    }

    # OpenAI tool_call ID maximum length limit
    TOOL_CALL_ID_MAX_LENGTH = 40

    def __init__(self, **config):
        """
        Initialize the OpenAI provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("OPENAI_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "OpenAI API key is missing. Please provide it in the config or set the OPENAI_API_KEY environment variable."
            )

        # NOTE: We could choose to remove above lines for api_key since OpenAI will automatically
        # infer certain values from the environment variables.
        # Eg: OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_PROJECT_ID, OPENAI_BASE_URL, etc.

        # Pass the entire config to the OpenAI client constructor
        self.client = openai.AsyncOpenAI(**config)

        # State for accumulating streaming tool calls
        self._streaming_tool_calls = {}
        # Map Responses function_call item.id (fc_*) -> call_id (call_*) for streaming tool calls
        self._responses_tool_call_ids: Dict[str, str] = {}

        # Track accumulated content for streaming responses
        # Used to provide accurate metadata in stop_info
        self._stream_content_length = 0
        self._stream_tool_calls_count = 0

    def _create_stop_info(self, finish_reason: str, choice_data: dict = None, model: str = None) -> dict:
        """Create StopInfo from OpenAI finish_reason."""
        if not finish_reason:
            return None

        # Analyze choice data to determine content presence
        has_content = False
        content_length = 0
        tool_calls_count = 0

        if choice_data:
            # Check message content
            message = choice_data.get("message") or choice_data.get("delta")
            if message:
                content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else None)
                if content:
                    has_content = True
                    content_length = len(content)

                # Check tool calls
                tool_calls = getattr(message, "tool_calls", None) or (message.get("tool_calls") if isinstance(message, dict) else None)
                if tool_calls:
                    tool_calls_count = len(tool_calls)
                    if tool_calls_count > 0:
                        has_content = True  # Tool calls count as content

        metadata = {
            "has_content": has_content,
            "content_length": content_length,
            "tool_calls_count": tool_calls_count,
            "finish_reason": finish_reason,
            "model": model,
            "provider": "openai"
        }

        # Map OpenAI finish_reason to standard StopReason
        mapped_reason = self.FINISH_REASON_MAPPING.get(finish_reason, finish_reason)
        return stop_reason_manager.map_stop_reason("openai", finish_reason, metadata)

    def _truncate_tool_call_id(self, original_id: str) -> str:
        """
        Truncate tool_call ID to meet OpenAI's length requirement.
        Uses MD5 hash for consistent results within a single request.

        Args:
            original_id: The original tool_call ID

        Returns:
            Truncated ID that meets OpenAI's requirements
        """
        if len(original_id) <= self.TOOL_CALL_ID_MAX_LENGTH:
            return original_id

        # Use MD5 hash to generate a consistent short ID
        hash_obj = hashlib.md5(original_id.encode('utf-8'))
        return hash_obj.hexdigest()[:self.TOOL_CALL_ID_MAX_LENGTH]

    def _process_tool_calls_for_openai(self, tool_calls: List[Dict[str, Any]], id_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Process tool_calls to ensure IDs meet OpenAI's length requirements.

        Args:
            tool_calls: List of tool_call dictionaries
            id_mapping: Dictionary to store original_id -> truncated_id mapping

        Returns:
            Processed tool_calls with truncated IDs
        """
        if not tool_calls:
            return tool_calls

        processed_tool_calls = []
        for tc in tool_calls:
            tc_copy = tc.copy()
            if 'id' in tc_copy and tc_copy['id']:
                original_id = tc_copy['id']
                truncated_id = self._truncate_tool_call_id(original_id)
                tc_copy['id'] = truncated_id
                # Store mapping for later restoration
                id_mapping[truncated_id] = original_id
            processed_tool_calls.append(tc_copy)

        return processed_tool_calls

    def _clean_messages_for_openai(self, messages: List[Union[Dict, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Clean messages for OpenAI API, handling both reasoning_content removal and tool_call ID truncation.

        Args:
            messages: List of messages to clean

        Returns:
            Tuple of (cleaned_messages, id_mapping) where id_mapping maps truncated_id -> original_id
        """
        cleaned_messages = []
        id_mapping = {}  # truncated_id -> original_id

        for msg in messages:
            if isinstance(msg, dict):
                cleaned_msg = msg.copy()
            else:
                # Convert Message object to dict
                if hasattr(msg, 'model_dump'):
                    cleaned_msg = msg.model_dump()
                else:
                    cleaned_msg = msg

            # Remove reasoning_content field (existing logic)
            if 'reasoning_content' in cleaned_msg:
                cleaned_msg.pop('reasoning_content')

            # Process tool_calls in assistant messages
            if cleaned_msg.get('role') == 'assistant' and cleaned_msg.get('tool_calls'):
                cleaned_msg['tool_calls'] = self._process_tool_calls_for_openai(
                    cleaned_msg['tool_calls'], id_mapping
                )

            # Process tool_call_id in tool messages
            elif cleaned_msg.get('role') == 'tool' and cleaned_msg.get('tool_call_id'):
                original_id = cleaned_msg['tool_call_id']
                truncated_id = self._truncate_tool_call_id(original_id)
                cleaned_msg['tool_call_id'] = truncated_id
                # Store mapping for later restoration
                id_mapping[truncated_id] = original_id

            cleaned_messages.append(cleaned_msg)

        return cleaned_messages, id_mapping

    def _restore_tool_call_ids(self, tool_calls: List[ChatCompletionMessageToolCall], id_mapping: Dict[str, str]) -> List[ChatCompletionMessageToolCall]:
        """
        Restore original tool_call IDs in the response.

        Args:
            tool_calls: List of tool_call objects from OpenAI response
            id_mapping: Dictionary mapping truncated_id -> original_id

        Returns:
            Tool calls with restored original IDs
        """
        if not tool_calls or not id_mapping:
            return tool_calls

        restored_tool_calls = []
        for tc in tool_calls:
            # Create a new tool call with restored ID
            original_id = id_mapping.get(tc.id, tc.id)
            restored_tc = ChatCompletionMessageToolCall(
                id=original_id,
                function=tc.function,
                type=tc.type,
                extra_content=getattr(tc, "extra_content", None)
            )
            restored_tool_calls.append(restored_tc)

        return restored_tool_calls

    def _supports_reasoning(self, model: str) -> bool:
        """Check if the model supports reasoning parameters."""
        # o1 series models support reasoning_effort
        if model.startswith('o1-'):
            return True
        # o3 series models support reasoning_effort (including o3, o3-mini, etc.)
        if model.startswith('o3') or model.startswith('o3-'):
            return True
        # GPT-5 series models support reasoning parameter
        if model.startswith('gpt-5'):
            return True
        # Regular GPT models (gpt-4o, gpt-4, etc.) do not support reasoning
        return False

    def _prepare_reasoning_kwargs(self, model: str, kwargs: dict) -> dict:
        """Prepare reasoning-related kwargs based on model type."""
        prepared_kwargs = kwargs.copy()

        # If model doesn't support reasoning, remove reasoning-related parameters
        if not self._supports_reasoning(model):
            # Remove reasoning parameters that would cause API errors
            prepared_kwargs.pop('reasoning', None)
            prepared_kwargs.pop('reasoning_effort', None)
            return prepared_kwargs

        # For reasoning models, handle special parameter requirements
        if (model.startswith('gpt-5') or
            model.startswith('o1-') or
            model.startswith('o3') or
            model.startswith('o3-')):
            # These models don't support max_tokens, use max_completion_tokens instead
            if 'max_tokens' in prepared_kwargs:
                max_tokens_value = prepared_kwargs.pop('max_tokens')
                prepared_kwargs['max_completion_tokens'] = max_tokens_value

            # GPT-5 has specific parameter restrictions
            if model.startswith('gpt-5'):
                # GPT-5 may have temperature restrictions (based on CloseAI findings)
                # Remove temperature if it's not the default value to avoid potential issues
                if 'temperature' in prepared_kwargs and prepared_kwargs['temperature'] != 1.0:
                    # For safety, we'll keep the temperature but add a comment
                    # OpenAI's GPT-5 may be more flexible than CloseAI's implementation
                    pass  # Keep temperature for now, but monitor for issues

        # Handle reasoning parameters for supported models
        if 'reasoning' in kwargs:
            reasoning = kwargs['reasoning']
            if model.startswith('gpt-5'):
                # GPT-5 uses reasoning parameter with effort field
                prepared_kwargs['reasoning'] = reasoning
            elif model.startswith('o1-') or model.startswith('o3') or model.startswith('o3-'):
                # o1/o3 series use reasoning_effort parameter
                if isinstance(reasoning, dict) and 'effort' in reasoning:
                    prepared_kwargs['reasoning_effort'] = reasoning['effort']
                    prepared_kwargs.pop('reasoning', None)
                else:
                    # If reasoning is a string, use it as effort
                    prepared_kwargs['reasoning_effort'] = reasoning
                    prepared_kwargs.pop('reasoning', None)

        # Handle reasoning_effort parameter for o1/o3 models
        if 'reasoning_effort' in kwargs and (model.startswith('o1-') or model.startswith('o3') or model.startswith('o3-')):
            prepared_kwargs['reasoning_effort'] = kwargs['reasoning_effort']

        # Handle verbosity parameter for GPT-5 models
        if 'verbosity' in kwargs and model.startswith('gpt-5'):
            prepared_kwargs['verbosity'] = kwargs['verbosity']

        return prepared_kwargs

    def _normalize_usage(self, usage_obj):
        """Normalize various OpenAI usage objects to a standard dict.

        This helper can handle both Chat Completions usage (prompt_tokens/completion_tokens)
        and Responses API usage (input_tokens/output_tokens).
        """
        if not usage_obj:
            return None

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

        # Try to get a plain dict from pydantic model or mapping
        if hasattr(usage_obj, "model_dump"):
            data = usage_obj.model_dump()
        elif isinstance(usage_obj, dict):
            data = usage_obj
        else:
            # Fallback to attribute access
            data = {}
            for attr in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens", "total_tokens"):
                if hasattr(usage_obj, attr):
                    data[attr] = getattr(usage_obj, attr)

        prompt_tokens = data.get("input_tokens")
        if prompt_tokens is None:
            prompt_tokens = data.get("prompt_tokens")
        completion_tokens = data.get("output_tokens")
        if completion_tokens is None:
            completion_tokens = data.get("completion_tokens")

        if prompt_tokens is None and hasattr(usage_obj, "prompt_tokens"):
            prompt_tokens = getattr(usage_obj, "prompt_tokens")
        if completion_tokens is None and hasattr(usage_obj, "completion_tokens"):
            completion_tokens = getattr(usage_obj, "completion_tokens")

        if prompt_tokens is None or completion_tokens is None:
            return None

        cached_tokens = _deep_get(data, "prompt_tokens_details", "cached_tokens")
        if cached_tokens is None:
            cached_tokens = _deep_get(data, "input_tokens_details", "cached_tokens")
        if cached_tokens is None:
            cached_tokens = _deep_get(usage_obj, "prompt_tokens_details", "cached_tokens")
        if cached_tokens is None:
            cached_tokens = _deep_get(usage_obj, "input_tokens_details", "cached_tokens")
        cache_read_input_tokens = int(cached_tokens) if cached_tokens is not None else 0
        if cache_read_input_tokens < 0:
            cache_read_input_tokens = 0

        total_tokens = data.get("total_tokens")
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        return {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(total_tokens),
            "cache_read_input_tokens": cache_read_input_tokens,
            "cache_write_input_tokens": 0,
            "cache_write_by_ttl": {
                "ephemeral_5m_input_tokens": 0,
                "ephemeral_1h_input_tokens": 0,
            },
        }


    def _should_use_responses_api(self, model: str, kwargs: dict) -> bool:
        """
        是否使用 Responses API：
        - GPT-5 系列：默认使用 Responses（推荐）
        - 其他模型：走 Chat Completions
        """
        if model.startswith('gpt-5'):
            return True
        return False

    async def _responses_create(self, model: str, messages: List[Dict[str, Any]], stream: bool = False, **kwargs) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionResponse, None]]:
        """
        使用 OpenAI Responses API（与 CloseAI 对齐）的调用实现：
        - 将 Chat 风格的 assistant.tool_calls -> Responses function_call
        - 将 role:tool -> function_call_output
        - 若上一轮消息包含 reasoning_content.raw_data.output（Responses 原生 output），直接 extend 到本轮 input
        - 支持流式事件：response.output_text.delta / response.function_call_arguments.delta / done
        """
        # 准备 input
        input_items: List[Dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, dict):
                msg_dict = msg.copy()
                role = msg_dict.get('role')
                # 回传上一轮 Responses 原生 output（若可用）
                rc = msg_dict.get('reasoning_content')
                raw_output = None
                try:
                    if rc is not None:
                        if hasattr(rc, 'raw_data') and isinstance(rc.raw_data, dict):
                            raw_output = rc.raw_data.get('output')
                        elif isinstance(rc, dict):
                            raw_output = rc.get('raw_data', {}).get('output')
                except Exception:
                    raw_output = None
                if raw_output:
                    input_items.extend(raw_output)
                    continue
                # assistant.tool_calls -> function_call
                if role == 'assistant' and 'tool_calls' in msg_dict and msg_dict['tool_calls']:
                    text = msg_dict.get('content')
                    if text:
                        input_items.append({'role': 'assistant', 'content': text})
                    for tc in msg_dict['tool_calls']:
                        fn = tc.get('function', {})
                        input_items.append({
                            'type': 'function_call',
                            'call_id': tc.get('id'),
                            'name': fn.get('name'),
                            'arguments': fn.get('arguments', '')
                        })
                    continue
                # role: tool -> function_call_output
                if role == 'tool':
                    input_items.append({
                        'type': 'function_call_output',
                        'call_id': msg_dict.get('tool_call_id'),
                        'output': msg_dict.get('content', '')
                    })
                    continue
                # 清理不支持字段
                for field in ['reasoning_content', 'tool_calls', 'tool_call_id']:
                    msg_dict.pop(field, None)
                input_items.append(msg_dict)
            else:
                # Message 对象
                if hasattr(msg, 'model_dump'):
                    md = msg.model_dump()
                    role = md.get('role')
                    rc = md.get('reasoning_content')
                    raw_output = None
                    try:
                        if rc is not None:
                            if hasattr(rc, 'raw_data') and isinstance(rc.raw_data, dict):
                                raw_output = rc.raw_data.get('output')
                            elif isinstance(rc, dict):
                                raw_output = rc.get('raw_data', {}).get('output')
                    except Exception:
                        raw_output = None
                    if raw_output:
                        input_items.extend(raw_output)
                        continue
                    if role == 'assistant' and 'tool_calls' in md and md['tool_calls']:
                        text = md.get('content')
                        if text:
                            input_items.append({'role': 'assistant', 'content': text})
                        for tc in md['tool_calls']:
                            fn = tc.get('function', {})
                            input_items.append({
                                'type': 'function_call',
                                'call_id': tc.get('id'),
                                'name': fn.get('name'),
                                'arguments': fn.get('arguments', '')
                            })
                        continue
                    if role == 'tool':
                        input_items.append({
                            'type': 'function_call_output',
                            'call_id': md.get('tool_call_id'),
                            'output': md.get('content', '')
                        })
                        continue
                    for field in ['reasoning_content', 'tool_calls', 'tool_call_id']:
                        md.pop(field, None)
                    input_items.append(md)
                else:
                    input_items.append(msg)

        # 准备 kwargs：去掉 messages，处理 tools（透传）
        responses_kwargs = kwargs.copy()
        responses_kwargs.pop('messages', None)
        # 工具转换为 Responses API 期望的扁平格式（如存在 function 包裹则展开），并透传 strict
        if 'tools' in responses_kwargs:
            responses_kwargs['tools'] = self._convert_tools_for_responses_api(responses_kwargs['tools'])

        if stream:
            # 重置流式状态
            self._streaming_tool_calls = {}
            self._stream_content_length = 0
            self._stream_tool_calls_count = 0
            response = await self.client.responses.create(
                model=model,
                input=input_items,
                stream=True,
                **responses_kwargs
            )
            stream_usage = None

            async def stream_gen():
                nonlocal stream_usage
                async for chunk in response:
                    ctype = getattr(chunk, 'type', None)

                    # Capture usage information if available on this event
                    usage_obj = None
                    if hasattr(chunk, "usage"):
                        usage_obj = getattr(chunk, "usage")
                    elif hasattr(chunk, "response") and hasattr(chunk.response, "usage"):
                        usage_obj = chunk.response.usage
                    if usage_obj:
                        stream_usage = self._normalize_usage(usage_obj) or stream_usage

                    if ctype == 'response.output_text.delta' and hasattr(chunk, 'delta'):
                        # Accumulate content length
                        if chunk.delta:
                            self._stream_content_length += len(chunk.delta)
                        yield ChatCompletionResponse(
                            choices=[StreamChoice(index=0, delta=ChoiceDelta(content=chunk.delta, role="assistant", tool_calls=None, reasoning_content=None), finish_reason=None)],
                            metadata={'id': getattr(chunk, 'response_id', None) or getattr(chunk, 'id', None), 'model': getattr(chunk, 'model', None)}
                        )
                    elif ctype == 'response.function_call_arguments.delta' and hasattr(chunk, 'delta'):
                        # 复用 Chat 路径的积累器
                        mock_delta = type('MockDelta', (), {'tool_calls': [type('MTC', (), {'index': getattr(chunk, 'output_index', 0), 'id': chunk.item_id, 'function': type('MF', (), {'arguments': chunk.delta, 'name': ''})(), 'type': 'function'})()]})()
                        tool_calls = self._accumulate_and_convert_tool_calls(mock_delta)
                        if tool_calls:
                            # Accumulate tool calls count
                            self._stream_tool_calls_count += len(tool_calls)
                            yield ChatCompletionResponse(
                                choices=[StreamChoice(index=0, delta=ChoiceDelta(content=None, role="assistant", tool_calls=tool_calls, reasoning_content=None), finish_reason=None)],
                                metadata={'id': getattr(chunk, 'response_id', None), 'model': getattr(chunk, 'model', None)}
                            )
                    elif ctype in ['response.completed', 'response.done']:
                        # Create accurate stop_info with accumulated metadata
                        metadata = {
                            "has_content": self._stream_content_length > 0 or self._stream_tool_calls_count > 0,
                            "content_length": self._stream_content_length,
                            "tool_calls_count": self._stream_tool_calls_count,
                            "finish_reason": 'stop',
                            "model": getattr(chunk, 'model', None),
                            "provider": "openai"
                        }
                        stop_info = stop_reason_manager.map_stop_reason("openai", 'stop', metadata)
                        response_metadata = {
                            'id': getattr(chunk, 'response_id', None),
                            'model': getattr(chunk, 'model', None)
                        }
                        if stream_usage:
                            response_metadata['usage'] = stream_usage
                        yield ChatCompletionResponse(
                            choices=[StreamChoice(index=0, delta=ChoiceDelta(content=None, role=None, tool_calls=None, reasoning_content=None), finish_reason='stop', stop_info=stop_info)],
                            metadata=response_metadata
                        )
            return stream_gen()
        else:
            resp = await self.client.responses.create(
                model=model,
                input=input_items,
                stream=False,
                **responses_kwargs
            )
            # 非流式：提取content和reasoning
            content = getattr(resp, 'output_text', None)
            reasoning_content = None

            # 提取reasoning content
            if hasattr(resp, 'reasoning_items') and resp.reasoning_items:
                reasoning_content = ReasoningContent(
                    thinking="[推理内容已加密，暂未提供摘要]",
                    provider="openai",
                    raw_data={
                        'reasoning_items': resp.reasoning_items,
                        'output': resp.output if hasattr(resp, 'output') else None,
                        'response_id': getattr(resp, 'id', None)
                    }
                )

            # 更完整的content提取
            if not content and hasattr(resp, 'output') and resp.output:
                for item in resp.output:
                    if getattr(item, 'type', None) == 'message' and getattr(item, 'content', None):
                        for ci in item.content:
                            if getattr(ci, 'type', None) == 'output_text':
                                content = ci.text
                                break
                        if content:
                            break

            # 确保content不为None（GPT-5有时返回None）
            if content is None:
                content = ""

            usage = getattr(resp, 'usage', None)
            usage_dict = self._normalize_usage(usage)

            metadata = {
                'id': getattr(resp, 'id', None),
                'model': getattr(resp, 'model', None)
            }
            if usage_dict:
                metadata['usage'] = usage_dict

            return ChatCompletionResponse(
                choices=[
                    Choice(index=0, message=Message(
                        content=content,
                        role='assistant',
                        tool_calls=None,
                        refusal=None,
                        reasoning_content=reasoning_content
                    ), finish_reason='stop')
                ],
                metadata=metadata
            )

    def _convert_tools_for_responses_api(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted = []
        for t in tools or []:
            if not isinstance(t, dict):
                converted.append(t)
                continue
            if t.get('type') == 'function' and 'function' in t:
                fn = t['function'] or {}
                item = {
                    'type': 'function',
                    'name': fn.get('name'),
                    'description': fn.get('description', ''),
                    'parameters': fn.get('parameters', {})
                }
                if 'strict' in fn:
                    item['strict'] = fn['strict']
                    try:
                        params = item['parameters'] or {}
                        if fn['strict'] and isinstance(params, dict) and 'additionalProperties' not in params:
                            params['additionalProperties'] = False
                            item['parameters'] = params
                    except Exception:
                        pass
                converted.append(item)
            else:
                converted.append(t)
        return converted




    async def chat_completions_create(self, model, messages, stream: bool = False, **kwargs) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionResponse, None]]:
        # Prepare kwargs based on model capabilities
        prepared_kwargs = self._prepare_reasoning_kwargs(model, kwargs)

        # Check if we should use Responses API instead of Chat Completions API
        if self._should_use_responses_api(model, prepared_kwargs):
            # GPT-5 + reasoning 等场景：走 Responses API 路径（与 closeaide 实现对齐）
            return await self._responses_create(model, messages, stream=stream, **prepared_kwargs)

        # Any exception raised by OpenAI will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        if stream:
            # Reset streaming state
            self._streaming_tool_calls = {}
            self._stream_content_length = 0
            self._stream_tool_calls_count = 0

            # Clean messages for OpenAI API (remove reasoning_content and truncate tool_call IDs)
            cleaned_messages, id_mapping = self._clean_messages_for_openai(messages)

            # Ensure streaming usage is included in the OpenAI response
            stream_kwargs = prepared_kwargs.copy()
            stream_options = stream_kwargs.get("stream_options") or {}
            if "include_usage" not in stream_options:
                stream_options["include_usage"] = True
            stream_kwargs["stream_options"] = stream_options

            response = await self.client.chat.completions.create(
                model=model,
                messages=cleaned_messages,  # 使用清理后的messages
                stream=True,
                **stream_kwargs  # Use prepared kwargs that are compatible with the model
            )
            async def stream_generator():
                stream_usage = None
                async for chunk in response:
                    # Capture usage from the streaming chunk if available (typically on final chunk)
                    if hasattr(chunk, "usage") and chunk.usage:
                        stream_usage = self._normalize_usage(chunk.usage) or stream_usage

                    if chunk.choices:
                        # Create choices with stop_info
                        choices = []
                        for choice in chunk.choices:
                            # Accumulate content and tool calls for accurate metadata
                            if choice.delta.content:
                                self._stream_content_length += len(choice.delta.content)

                            # Process tool calls and restore original IDs
                            accumulated_tool_calls = self._accumulate_and_convert_tool_calls(choice.delta)
                            if accumulated_tool_calls and id_mapping:
                                accumulated_tool_calls = self._restore_tool_call_ids(accumulated_tool_calls, id_mapping)
                            if accumulated_tool_calls:
                                self._stream_tool_calls_count += len(accumulated_tool_calls)

                            # Create stop_info if finish_reason is present
                            stop_info = None
                            if choice.finish_reason:
                                # Use accumulated values for final stop_info
                                if choice.finish_reason == 'stop':
                                    metadata = {
                                        "has_content": self._stream_content_length > 0 or self._stream_tool_calls_count > 0,
                                        "content_length": self._stream_content_length,
                                        "tool_calls_count": self._stream_tool_calls_count,
                                        "finish_reason": choice.finish_reason,
                                        "model": chunk.model,
                                        "provider": "openai"
                                    }
                                    stop_info = stop_reason_manager.map_stop_reason("openai", choice.finish_reason, metadata)
                                else:
                                    choice_data = {"delta": choice.delta}
                                    stop_info = self._create_stop_info(choice.finish_reason, choice_data, chunk.model)

                            choices.append(StreamChoice(
                                index=choice.index,
                                delta=ChoiceDelta(
                                    content=choice.delta.content,
                                    role=choice.delta.role,
                                    tool_calls=accumulated_tool_calls,
                                    reasoning_content=getattr(choice.delta, 'reasoning_content', None)  # 流式时保持原始格式
                                ),
                                finish_reason=choice.finish_reason,
                                stop_info=stop_info
                            ))

                        # Determine if this is the final chunk (has finish_reason or usage)
                        is_final_chunk = any(c.finish_reason for c in chunk.choices) or stream_usage is not None

                        metadata = {
                            'id': chunk.id,
                            'created': chunk.created,
                            'model': chunk.model
                        }
                        if is_final_chunk and stream_usage:
                            metadata['usage'] = stream_usage

                        yield ChatCompletionResponse(
                            choices=choices,
                            metadata=metadata
                        )
            return stream_generator()
        else:
            # Clean messages for OpenAI API (remove reasoning_content and truncate tool_call IDs)
            cleaned_messages, id_mapping = self._clean_messages_for_openai(messages)

            response = await self.client.chat.completions.create(
                model=model,
                messages=cleaned_messages,  # 使用清理后的messages
                stream=False,
                **prepared_kwargs  # Use prepared kwargs that are compatible with the model
            )
            # Create choices with stop_info
            choices = []
            for choice in response.choices:
                # Create stop_info
                choice_data = {"message": choice.message}
                stop_info = self._create_stop_info(choice.finish_reason, choice_data, response.model)

                # Convert and restore tool calls
                converted_tool_calls = None
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                    converted_tool_calls = self._convert_tool_calls(choice.message.tool_calls)
                    if converted_tool_calls and id_mapping:
                        converted_tool_calls = self._restore_tool_call_ids(converted_tool_calls, id_mapping)

                choices.append(Choice(
                    index=choice.index,
                    message=Message(
                        content=choice.message.content,
                        role=choice.message.role,
                        tool_calls=converted_tool_calls,
                        refusal=None,
                        reasoning_content=self._convert_reasoning_content(getattr(choice.message, 'reasoning_content', None))
                    ),
                    finish_reason=choice.finish_reason,
                    stop_info=stop_info
                ))

            return ChatCompletionResponse(
                choices=choices,
                metadata={
                    "id": response.id,
                    "created": response.created,
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
            )

    def _convert_reasoning_content(self, reasoning_content):
        """Convert OpenAI reasoning_content to ReasoningContent object."""
        if not reasoning_content:
            return None

        return ReasoningContent(
            thinking=reasoning_content,
            provider="openai",
            raw_data={"reasoning_content": reasoning_content}
        )

    def _accumulate_and_convert_tool_calls(self, delta):
        """
        Accumulate tool call chunks and convert to unified format when complete.

        Args:
            delta: The delta object from streaming response

        Returns:
            List of converted tool calls if any are complete, None otherwise
        """
        if not hasattr(delta, 'tool_calls') or not delta.tool_calls:
            return None

        # Accumulate tool call chunks
        for tool_call_delta in delta.tool_calls:
            index = getattr(tool_call_delta, 'index', 0)

            # Initialize tool call accumulator if not exists
            if index not in self._streaming_tool_calls:
                self._streaming_tool_calls[index] = {
                    "id": "",
                    "type": "function",
                    "function": {
                        "name": "",
                        "arguments": ""
                    }
                }

            tool_call = self._streaming_tool_calls[index]

            # Accumulate id
            if hasattr(tool_call_delta, 'id') and tool_call_delta.id:
                tool_call["id"] += tool_call_delta.id

            # Accumulate function data
            if hasattr(tool_call_delta, 'function') and tool_call_delta.function:
                if hasattr(tool_call_delta.function, 'name') and tool_call_delta.function.name:
                    tool_call["function"]["name"] += tool_call_delta.function.name

                if hasattr(tool_call_delta.function, 'arguments') and tool_call_delta.function.arguments:
                    tool_call["function"]["arguments"] += tool_call_delta.function.arguments

            # Set type if provided
            if hasattr(tool_call_delta, 'type') and tool_call_delta.type:
                tool_call["type"] = tool_call_delta.type

        # Check for complete tool calls and convert them
        complete_tool_calls = []
        for index, tool_call_data in list(self._streaming_tool_calls.items()):
            if (tool_call_data["id"] and
                tool_call_data["function"]["name"] and
                tool_call_data["function"]["arguments"]):

                try:
                    # Try to parse arguments as JSON to ensure completeness
                    json.loads(tool_call_data["function"]["arguments"])

                    # Create mock object for _convert_tool_calls
                    class MockToolCall:
                        def __init__(self, id, function_name, function_args):
                            self.id = id
                            self.function = MockFunction(function_name, function_args)

                    class MockFunction:
                        def __init__(self, name, arguments):
                            self.name = name
                            self.arguments = arguments

                    mock_tool_call = MockToolCall(
                        tool_call_data["id"],
                        tool_call_data["function"]["name"],
                        tool_call_data["function"]["arguments"]
                    )

                    # Use existing conversion logic
                    converted = self._convert_tool_calls([mock_tool_call])
                    if converted:
                        complete_tool_calls.extend(converted)

                    # Remove completed tool call from accumulator
                    del self._streaming_tool_calls[index]

                except json.JSONDecodeError:
                    # Arguments are not complete yet, continue accumulating
                    continue

        return complete_tool_calls if complete_tool_calls else None

    def _convert_tool_calls(self, tool_calls):
        """Convert tool calls to the framework's format."""
        if not tool_calls:
            return None

        converted_tool_calls = []
        for tool_call in tool_calls:
            function = Function(
                name=tool_call.function.name,
                arguments=tool_call.function.arguments
            )
            tool_call_obj = ChatCompletionMessageToolCall(
                id=tool_call.id,
                function=function,
                type="function"
            )
            converted_tool_calls.append(tool_call_obj)
        return converted_tool_calls
