import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aisuite.framework.message import (
    ChatCompletionMessageToolCall,
    Function,
    Message,
    ReasoningContent,
)
from aisuite.framework.message_normalizer import MessageNormalizer
from aisuite.framework.replay_payload import ReplayBuildResult, ReplayCaptureResult
from aisuite.framework.replay_payload import build_replay_payload
from aisuite.framework.stop_reason import StopReason, stop_reason_manager
from aisuite.providers.anthropic_provider import (
    AnthropicProvider,
    AnthropicMessageConverter,
    _build_anthropic_thinking_replay_payload,
)
from aisuite.providers.openai_provider import OpenaiProvider


def test_stop_reason_maps_gemini_protocol_errors():
    stop_info = stop_reason_manager.map_stop_reason("gemini", "MISSING_THOUGHT_SIGNATURE", {})
    assert stop_info.reason == StopReason.PROTOCOL_ERROR
    assert stop_info.metadata["error_class"] == "protocol_error"

    stop_info = stop_reason_manager.map_stop_reason("gemini", "MALFORMED_RESPONSE", {})
    assert stop_info.reason == StopReason.PROTOCOL_ERROR
    assert stop_info.metadata["error_class"] == "protocol_error"


@patch("aisuite.providers.openai_provider.openai.AsyncOpenAI")
def test_openai_build_replay_view_reads_versioned_responses_payload(_mock_client_cls):
    provider = OpenaiProvider(api_key="test-openai-key")
    raw_output = [{"type": "message", "role": "assistant", "content": []}]
    replay_payload = provider._build_responses_replay_payload(
        output=raw_output,
        response_id="resp_123",
    )

    replay_view = provider.build_replay_view(
        "gpt-5.1",
        [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": {
                    "thinking": "hidden",
                    "provider": "openai",
                    "raw_data": replay_payload,
                },
            }
        ],
    )

    assert isinstance(replay_view, ReplayBuildResult)
    assert replay_view.request_view == raw_output
    assert replay_view.replay_mode == "responses_output"
    assert replay_view.degraded is False


def test_anthropic_converter_replays_versioned_thinking_block():
    converter = AnthropicMessageConverter()
    thinking_block = {
        "type": "thinking",
        "thinking": "I should call the tool.",
        "signature": "sig_123",
    }
    reasoning_content = ReasoningContent(
        thinking=thinking_block["thinking"],
        signature=thinking_block["signature"],
        provider="anthropic",
        raw_data=_build_anthropic_thinking_replay_payload(thinking_block),
    )

    result = converter._create_assistant_tool_message(
        "Checking tool.",
        [
            {
                "id": "tool_1",
                "function": {
                    "name": "lookup_weather",
                    "arguments": '{"city":"Paris"}',
                },
            }
        ],
        reasoning_content=reasoning_content,
    )

    assert result["content"][0] == thinking_block
    assert result["content"][1] == {"type": "text", "text": "Checking tool."}


@patch("aisuite.providers.anthropic_provider.anthropic.AsyncAnthropic")
def test_anthropic_build_replay_view_returns_structured_result(_mock_client_cls):
    provider = AnthropicProvider(api_key="test-anthropic-key")

    replay_view = provider.build_replay_view(
        "claude-sonnet",
        [{"role": "user", "content": "hello"}],
    )

    assert isinstance(replay_view, ReplayBuildResult)
    assert replay_view.replay_mode == "anthropic_messages"
    assert replay_view.request_view["messages"] == [{"role": "user", "content": "hello"}]


@patch("aisuite.providers.anthropic_provider.anthropic.AsyncAnthropic")
def test_anthropic_stream_tool_use_is_finalized_on_content_block_stop(_mock_client_cls):
    provider = AnthropicProvider(api_key="test-anthropic-key")

    start_chunk = SimpleNamespace(
        type="content_block_start",
        index=1,
        content_block=SimpleNamespace(
            type="tool_use",
            id="tool_1",
            name="lookup_weather",
            input={},
        ),
    )
    stop_chunk = SimpleNamespace(
        type="content_block_stop",
        index=1,
    )
    message_delta_chunk = SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(stop_reason="tool_use"),
        usage=None,
    )

    assert provider.converter.convert_stream_response(start_chunk, "claude-sonnet", provider) is None

    tool_result = provider.converter.convert_stream_response(stop_chunk, "claude-sonnet", provider)
    assert tool_result is not None
    assert tool_result.choices[0].delta.tool_calls is not None
    assert len(tool_result.choices[0].delta.tool_calls) == 1
    assert tool_result.choices[0].delta.tool_calls[0].id == "tool_1"
    assert tool_result.choices[0].delta.tool_calls[0].function.name == "lookup_weather"
    assert tool_result.choices[0].delta.tool_calls[0].function.arguments == "{}"

    stop_result = provider.converter.convert_stream_response(
        message_delta_chunk,
        "claude-sonnet",
        provider,
    )
    assert stop_result is not None
    assert stop_result.choices[0].stop_info.reason == StopReason.TOOL_CALL
    assert stop_result.choices[0].stop_info.metadata["tool_calls_count"] == 1


@patch("aisuite.providers.anthropic_provider.anthropic.AsyncAnthropic")
def test_anthropic_stream_tool_use_without_explicit_input_defaults_to_empty_object(_mock_client_cls):
    provider = AnthropicProvider(api_key="test-anthropic-key")

    start_chunk = SimpleNamespace(
        type="content_block_start",
        index=1,
        content_block=SimpleNamespace(
            type="tool_use",
            id="tool_2",
            name="set_task_list",
        ),
    )
    stop_chunk = SimpleNamespace(
        type="content_block_stop",
        index=1,
    )
    message_delta_chunk = SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(stop_reason="tool_use"),
        usage=None,
    )

    assert provider.converter.convert_stream_response(start_chunk, "claude-sonnet", provider) is None

    tool_result = provider.converter.convert_stream_response(stop_chunk, "claude-sonnet", provider)
    assert tool_result is not None
    assert tool_result.choices[0].delta.tool_calls is not None
    assert len(tool_result.choices[0].delta.tool_calls) == 1
    assert tool_result.choices[0].delta.tool_calls[0].id == "tool_2"
    assert tool_result.choices[0].delta.tool_calls[0].function.name == "set_task_list"
    assert tool_result.choices[0].delta.tool_calls[0].function.arguments == "{}"

    stop_result = provider.converter.convert_stream_response(
        message_delta_chunk,
        "claude-sonnet",
        provider,
    )
    assert stop_result is not None
    assert stop_result.choices[0].stop_info.reason == StopReason.TOOL_CALL
    assert stop_result.choices[0].stop_info.metadata["tool_calls_count"] == 1


@patch("aisuite.providers.anthropic_provider.anthropic.AsyncAnthropic")
def test_anthropic_stream_prefers_final_content_block_input_over_malformed_partial_json(_mock_client_cls):
    provider = AnthropicProvider(api_key="test-anthropic-key")

    expected_input = {
        "tasks": [
            {
                "title": '修复末尾笔误"讲"→"将"',
                "status": "NOT_STARTED",
            }
        ]
    }

    start_chunk = SimpleNamespace(
        type="content_block_start",
        index=2,
        content_block=SimpleNamespace(
            type="tool_use",
            id="tool_3",
            name="set_task_list",
        ),
    )
    delta_chunk_1 = SimpleNamespace(
        type="content_block_delta",
        index=2,
        delta=SimpleNamespace(
            type="input_json_delta",
            partial_json='{"tasks":[{"title":"修复末尾笔误',
        ),
    )
    delta_chunk_2 = SimpleNamespace(
        type="content_block_delta",
        index=2,
        delta=SimpleNamespace(
            type="input_json_delta",
            partial_json='"讲"→"将"","status":"NOT_STARTED"}]}',
        ),
    )
    stop_chunk = SimpleNamespace(
        type="content_block_stop",
        index=2,
        content_block=SimpleNamespace(
            type="tool_use",
            id="tool_3",
            name="set_task_list",
            input=expected_input,
        ),
    )
    message_delta_chunk = SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(stop_reason="tool_use"),
        usage=None,
    )

    assert provider.converter.convert_stream_response(start_chunk, "claude-sonnet", provider) is None
    assert provider.converter.convert_stream_response(delta_chunk_1, "claude-sonnet", provider) is None
    assert provider.converter.convert_stream_response(delta_chunk_2, "claude-sonnet", provider) is None

    tool_result = provider.converter.convert_stream_response(stop_chunk, "claude-sonnet", provider)
    assert tool_result is not None
    assert tool_result.choices[0].delta.tool_calls is not None
    assert len(tool_result.choices[0].delta.tool_calls) == 1
    tool_call = tool_result.choices[0].delta.tool_calls[0]
    assert tool_call.id == "tool_3"
    assert tool_call.function.name == "set_task_list"
    assert json.loads(tool_call.function.arguments) == expected_input

    stop_result = provider.converter.convert_stream_response(
        message_delta_chunk,
        "claude-sonnet",
        provider,
    )
    assert stop_result is not None
    assert stop_result.choices[0].stop_info.reason == StopReason.TOOL_CALL
    assert stop_result.choices[0].stop_info.metadata["tool_calls_count"] == 1


@patch("aisuite.providers.anthropic_provider.anthropic.AsyncAnthropic")
def test_anthropic_stream_dedupes_tool_call_emitted_from_delta_and_content_block_stop(_mock_client_cls):
    provider = AnthropicProvider(api_key="test-anthropic-key")

    start_chunk = SimpleNamespace(
        type="content_block_start",
        index=1,
        content_block=SimpleNamespace(
            type="tool_use",
            id="tool_dup_1",
            name="read_file",
        ),
    )
    delta_chunk = SimpleNamespace(
        type="content_block_delta",
        index=1,
        delta=SimpleNamespace(
            type="input_json_delta",
            partial_json='{"file_path":"chapter.aifc"}',
        ),
    )
    stop_chunk = SimpleNamespace(
        type="content_block_stop",
        index=1,
        content_block=SimpleNamespace(
            type="tool_use",
            id="tool_dup_1",
            name="read_file",
            input={"file_path": "chapter.aifc"},
        ),
    )
    message_delta_chunk = SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(stop_reason="tool_use"),
        usage=None,
    )

    assert provider.converter.convert_stream_response(start_chunk, "claude-sonnet", provider) is None

    delta_result = provider.converter.convert_stream_response(delta_chunk, "claude-sonnet", provider)
    assert delta_result is not None
    assert delta_result.choices[0].delta.tool_calls is not None
    assert len(delta_result.choices[0].delta.tool_calls) == 1
    assert delta_result.choices[0].delta.tool_calls[0].id == "tool_dup_1"

    stop_result = provider.converter.convert_stream_response(stop_chunk, "claude-sonnet", provider)
    assert stop_result is None

    final_stop_result = provider.converter.convert_stream_response(
        message_delta_chunk,
        "claude-sonnet",
        provider,
    )
    assert final_stop_result is not None
    assert final_stop_result.choices[0].stop_info.reason == StopReason.TOOL_CALL
    assert final_stop_result.choices[0].stop_info.metadata["tool_calls_count"] == 1


@patch("aisuite.providers.anthropic_provider.anthropic.AsyncAnthropic")
def test_anthropic_message_stop_recovers_tool_use_from_sdk_snapshot(_mock_client_cls):
    provider = AnthropicProvider(api_key="test-anthropic-key")

    message_stop_chunk = SimpleNamespace(
        type="message_stop",
        message=SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="tool_4",
                    name="edit_file",
                    input={"file_path": "chapter.aifc", "diff_content": ["@@\n-old\n+new"]},
                )
            ]
        ),
    )

    result = provider.converter.convert_stream_response(message_stop_chunk, "claude-sonnet", provider)

    assert result is not None
    assert provider._stream_tool_calls_count == 1
    tool_calls = result.choices[0].delta.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "edit_file"
    assert json.loads(tool_calls[0].function.arguments)["file_path"] == "chapter.aifc"


@patch("aisuite.providers.anthropic_provider.anthropic.AsyncAnthropic")
@pytest.mark.asyncio
async def test_anthropic_chat_stream_uses_sdk_stream_helper_and_final_message_recovery(_mock_client_cls):
    provider = AnthropicProvider(api_key="test-anthropic-key")

    final_message = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id="tool_5",
                name="edit_file",
                input={"file_path": "chapter.aifc", "diff_content": ["@@\n-old\n+new"]},
            )
        ]
    )

    class FakeAsyncStream:
        def __init__(self, final_message):
            self._final_message = final_message
            self.request_id = "req_test_123"

        def __aiter__(self):
            async def _iterate():
                if False:
                    yield None
            return _iterate()

        async def get_final_message(self):
            return self._final_message

    class FakeAsyncStreamManager:
        def __init__(self, stream):
            self._stream = stream

        async def __aenter__(self):
            return self._stream

        async def __aexit__(self, exc_type, exc, exc_tb):
            return None

    stream_mock = Mock(return_value=FakeAsyncStreamManager(FakeAsyncStream(final_message)))
    create_mock = AsyncMock()
    provider.client = SimpleNamespace(
        messages=SimpleNamespace(
            stream=stream_mock,
            create=create_mock,
        )
    )

    response_stream = await provider.chat_completions_create(
        "claude-sonnet",
        [{"role": "user", "content": "hello"}],
        stream=True,
    )

    streamed_responses = [chunk async for chunk in response_stream]

    assert create_mock.await_count == 0
    assert stream_mock.call_count == 1
    assert len(streamed_responses) == 1
    tool_calls = streamed_responses[0].choices[0].delta.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "tool_5"
    assert tool_calls[0].function.name == "edit_file"
    assert json.loads(tool_calls[0].function.arguments)["file_path"] == "chapter.aifc"


def test_message_normalizer_preserves_versioned_replay_payloads():
    tool_replay = build_replay_payload(
        "gemini",
        "gemini_tool_call",
        {
            "provider_call_id": "fc_123",
            "provider_function_name": "lookup_weather",
            "replay_mode": "provider_exact_turn",
            "thought_signature": "base64:c2ln",
        },
    )
    reasoning_replay = build_replay_payload(
        "openai",
        "responses_output",
        {"output": [{"type": "message"}], "response_id": "resp_1"},
    )

    message = Message(
        role="assistant",
        content="done",
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_local_1",
                function=Function(name="lookup_weather", arguments='{"city":"Paris"}'),
                type="function",
                extra_content=tool_replay,
            )
        ],
        reasoning_content=ReasoningContent(
            thinking="summary",
            provider="openai",
            raw_data=reasoning_replay,
        ),
    )

    serialized = MessageNormalizer._manual_serialize(message)

    assert serialized["tool_calls"][0]["extra_content"]["version"] == 1
    assert serialized["tool_calls"][0]["extra_content"]["payload"]["provider_call_id"] == "fc_123"
    assert serialized["reasoning_content"]["raw_data"]["version"] == 1
    assert serialized["reasoning_content"]["raw_data"]["payload"]["response_id"] == "resp_1"


@patch("aisuite.providers.openai_provider.openai.AsyncOpenAI")
def test_openai_capture_response_returns_structured_result(_mock_client_cls):
    provider = OpenaiProvider(api_key="test-openai-key")
    message = Message(content="done", role="assistant")
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=message,
                stop_info="stop-info",
            )
        ]
    )

    captured = provider.capture_response(response, model="gpt-5.1")

    assert isinstance(captured, ReplayCaptureResult)
    assert captured.canonical_message is message
    assert captured.stop_info == "stop-info"
    assert captured.replay_metadata["tool_calls"] is None
    assert captured.protocol_diagnostics == ()


@patch("aisuite.providers.openai_provider.openai.AsyncOpenAI")
def test_openai_validate_replay_window_reports_missing_tool_call_id(_mock_client_cls):
    provider = OpenaiProvider(api_key="test-openai-key")
    result = provider.validate_replay_window(
        "gpt-5.1",
        [{"role": "tool", "content": "done"}],
    )

    assert result.ok is False
    assert any(diag.code == "missing_tool_call_id" for diag in result.diagnostics)


@patch("aisuite.providers.openai_provider.openai.AsyncOpenAI")
@patch("aisuite.providers.openai_provider.OpenaiProvider.build_replay_view")
@pytest.mark.asyncio
async def test_openai_chat_completions_create_uses_replay_override_for_responses(
    mock_build_replay_view, _mock_client_cls
):
    provider = OpenaiProvider(api_key="test-openai-key")
    override_input = [{"type": "message", "role": "user", "content": []}]
    mock_build_replay_view.side_effect = AssertionError("should not rebuild replay view")

    response = SimpleNamespace(
        output_text="done",
        output=[],
        id="resp_1",
        model="gpt-5.1",
        usage=None,
    )

    with patch.object(
        provider.client.responses,
        "create",
        new=AsyncMock(return_value=response),
    ) as mock_create:
        result = await provider.chat_completions_create(
            "gpt-5.1",
            [{"role": "user", "content": "ignored"}],
            reasoning={"effort": "low"},
            _replay_request_view=override_input,
            _replay_mode="responses_output",
        )

    assert result.choices[0].message.content == "done"
    assert mock_create.await_args.kwargs["input"] == override_input


@patch("aisuite.providers.anthropic_provider.anthropic.AsyncAnthropic")
def test_anthropic_capture_response_returns_structured_result(_mock_client_cls):
    provider = AnthropicProvider(api_key="test-anthropic-key")
    reasoning = ReasoningContent(
        thinking="thinking",
        provider="anthropic",
        raw_data=_build_anthropic_thinking_replay_payload({"type": "thinking", "thinking": "thinking"}),
    )
    message = Message(content="done", role="assistant", reasoning_content=reasoning)
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=message,
                stop_info="stop-info",
            )
        ]
    )

    captured = provider.capture_response(response, model="claude-sonnet")

    assert isinstance(captured, ReplayCaptureResult)
    assert captured.canonical_message is message
    assert captured.stop_info == "stop-info"
    assert captured.replay_metadata["reasoning_content"]["provider"] == "anthropic"


@patch("aisuite.providers.anthropic_provider.anthropic.AsyncAnthropic")
def test_anthropic_validate_replay_window_reports_missing_tool_call_id(_mock_client_cls):
    provider = AnthropicProvider(api_key="test-anthropic-key")
    result = provider.validate_replay_window(
        "claude-sonnet",
        [{"role": "tool", "content": "done"}],
    )

    assert result.ok is False
    assert any(diag.code == "missing_tool_call_id" for diag in result.diagnostics)


@patch("aisuite.providers.anthropic_provider.anthropic.AsyncAnthropic")
@pytest.mark.asyncio
async def test_anthropic_chat_completions_create_uses_replay_override(_mock_client_cls):
    provider = AnthropicProvider(api_key="test-anthropic-key")
    replay_override = {
        "system": "system prompt",
        "messages": [{"role": "user", "content": "hello"}],
    }

    with patch.object(
        provider.converter,
        "convert_request",
        side_effect=AssertionError("should not convert canonical request"),
    ), patch.object(
        provider.converter,
        "convert_response",
        return_value="ok",
    ), patch.object(
        provider.client.messages,
        "create",
        new=AsyncMock(return_value=SimpleNamespace()),
    ) as mock_create:
        result = await provider.chat_completions_create(
            "claude-sonnet",
            [{"role": "user", "content": "ignored"}],
            _replay_request_view=replay_override,
            _replay_mode="anthropic_messages",
        )

    assert result == "ok"
    assert mock_create.await_args.kwargs["system"] == "system prompt"
    assert mock_create.await_args.kwargs["messages"] == [{"role": "user", "content": "hello"}]
