from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from aisuite.framework.message_normalizer import MessageNormalizer
from aisuite.framework.message import ReasoningContent
from aisuite.providers.deepseek_provider import DeepseekProvider


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


class _ClosableAsyncStream:
    def __init__(self, chunks):
        self.closed = False
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration

    async def aclose(self):
        self.closed = True


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-api-key")


@pytest.mark.asyncio
async def test_deepseek_provider_non_stream_replays_reasoning_content():
    provider = DeepseekProvider()
    mock_response = _ns(
        id="deepseek-response-id",
        created=1234567890,
        model="deepseek-v4-pro",
        usage=_ns(
            prompt_cache_hit_tokens=3,
            prompt_cache_miss_tokens=7,
            completion_tokens=5,
            total_tokens=15,
        ),
        choices=[
            _ns(
                index=0,
                finish_reason="stop",
                message=_ns(
                    content="done",
                    role="assistant",
                    reasoning_content="step-by-step",
                    tool_calls=None,
                ),
            )
        ],
    )

    with patch.object(
        provider.client.chat.completions,
        "create",
        new=AsyncMock(return_value=mock_response),
    ) as mock_create:
        response = await provider.chat_completions_create(
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": {
                        "thinking": "previous reasoning trace",
                        "provider": "deepseek",
                    },
                },
                {"role": "user", "content": "Hello!"},
            ],
            model="deepseek-v4-pro",
            temperature=0.2,
        )

    mock_create.assert_called_with(
        messages=[
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "previous reasoning trace",
            },
            {"role": "user", "content": "Hello!"},
        ],
        model="deepseek-v4-pro",
        stream=False,
        temperature=0.2,
    )
    assert response.choices[0].message.content == "done"
    assert response.choices[0].message.reasoning_content.thinking == "step-by-step"
    assert (
        response.choices[0].message.reasoning_content.raw_data["provider"] == "deepseek"
    )
    assert response.metadata["usage"]["cache_read_input_tokens"] == 3


@pytest.mark.asyncio
async def test_deepseek_provider_fills_and_preserves_empty_tool_reasoning():
    provider = DeepseekProvider(api_key="test-api-key")
    response_tool_call = _ns(
        id="call_2",
        type="function",
        function=_ns(name="write_file", arguments='{"path":"done.txt"}'),
    )
    mock_response = _ns(
        id="deepseek-response-id",
        created=1234567890,
        model="deepseek-v4-flash",
        usage=None,
        choices=[
            _ns(
                index=0,
                finish_reason="tool_calls",
                message=_ns(
                    content="",
                    role="assistant",
                    reasoning_content="",
                    tool_calls=[response_tool_call],
                ),
            )
        ],
    )

    previous_tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "read_file", "arguments": "{}"},
    }
    with patch.object(
        provider.client.chat.completions,
        "create",
        new=AsyncMock(return_value=mock_response),
    ) as mock_create:
        response = await provider.chat_completions_create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "user", "content": "continue"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [previous_tool_call],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "done"},
            ],
            thinking={"type": "enabled"},
        )

    sent_messages = mock_create.call_args.kwargs["messages"]
    assert sent_messages[1]["reasoning_content"] == ""
    assert response.choices[0].message.reasoning_content.thinking == ""
    assert (
        response.choices[0].message.reasoning_content.raw_data["payload"][
            "reasoning_content"
        ]
        == ""
    )


def test_deepseek_build_replay_view_preserves_reasoning_content():
    provider = DeepseekProvider(api_key="test-api-key")

    replay_build = provider.build_replay_view(
        "deepseek-v4-pro",
        [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": ReasoningContent(
                    thinking="cached reasoning",
                    provider="deepseek",
                    raw_data={
                        "provider": "deepseek",
                        "version": 1,
                        "kind": "deepseek_reasoning_text",
                        "payload": {"reasoning_content": "cached reasoning"},
                    },
                ),
            },
            {"role": "user", "content": "hi"},
        ],
    )

    assert replay_build.replay_mode == "canonical_with_reasoning"
    assert replay_build.request_view[0]["reasoning_content"] == "cached reasoning"


def test_deepseek_build_replay_view_accepts_duck_typed_reasoning_content():
    provider = DeepseekProvider(api_key="test-api-key")
    stored_reasoning = _ns(
        thinking="cached reasoning",
        provider="deepseek",
        raw_data={
            "provider": "deepseek",
            "version": 1,
            "kind": "deepseek_reasoning_text",
            "payload": {"reasoning_content": "cached reasoning"},
        },
    )

    replay_build = provider.build_replay_view(
        "deepseek-v4-pro",
        [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": stored_reasoning,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "user", "content": "continue"},
        ],
        extra_body={"thinking": {"type": "enabled"}},
    )

    assert replay_build.replay_mode == "canonical_with_reasoning"
    assert replay_build.request_view[0]["reasoning_content"] == "cached reasoning"


def test_deepseek_message_normalizer_preserves_reasoning_content():
    normalized = MessageNormalizer.normalize_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": {"thinking": "keep me"},
            }
        ],
        "deepseek:deepseek-v4-pro",
    )

    assert normalized[0]["reasoning_content"]["thinking"] == "keep me"


def test_deepseek_validate_replay_window_reports_missing_tool_call_id():
    provider = DeepseekProvider(api_key="test-api-key")

    result = provider.validate_replay_window(
        "deepseek-v4-flash",
        [{"role": "tool", "content": "result"}],
    )

    assert result.ok is False
    assert any(diag.code == "missing_tool_call_id" for diag in result.diagnostics)


def test_deepseek_build_replay_view_fills_empty_reasoning_for_thinking_tool_calls():
    provider = DeepseekProvider(api_key="test-api-key")

    replay_build = provider.build_replay_view(
        "deepseek-v4-pro",
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "user", "content": "continue"},
        ],
        extra_body={"thinking": {"type": "enabled"}},
    )

    assert replay_build.request_view[0]["reasoning_content"] == ""


def test_deepseek_validate_replay_window_allows_non_thinking_tool_calls_without_reasoning():
    provider = DeepseekProvider(api_key="test-api-key")

    result = provider.validate_replay_window(
        "deepseek-v4-pro",
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "user", "content": "continue"},
        ],
        extra_body={"thinking": {"type": "disabled"}},
    )

    assert result.ok is True


def test_deepseek_capture_response_returns_structured_result():
    provider = DeepseekProvider(api_key="test-api-key")
    response = _ns(
        choices=[
            _ns(
                stop_info=_ns(reason="complete"),
                message=_ns(
                    role="assistant",
                    content="done",
                    tool_calls=None,
                    reasoning_content=ReasoningContent(
                        thinking="think",
                        provider="deepseek",
                        raw_data={
                            "provider": "deepseek",
                            "version": 1,
                            "kind": "deepseek_reasoning_text",
                            "payload": {"reasoning_content": "think"},
                        },
                    ),
                ),
            )
        ]
    )

    captured = provider.capture_response(response, model="deepseek-v4-pro")

    assert captured.canonical_message.content == "done"
    assert captured.replay_metadata["reasoning_content"]["provider"] == "deepseek"


def test_deepseek_provider_preserves_custom_base_url():
    with patch("aisuite.providers.deepseek_provider.openai.AsyncOpenAI") as mock_client:
        DeepseekProvider(
            api_key="test-api-key", base_url="https://api.deepseek.com/beta"
        )

    assert mock_client.call_args.kwargs["base_url"] == "https://api.deepseek.com/beta"


@pytest.mark.asyncio
async def test_deepseek_provider_closes_async_client():
    created = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.closed = False
            created["http_client"] = kwargs["http_client"]

        def is_closed(self):
            return self.closed

        async def close(self):
            self.closed = True

    with patch(
        "aisuite.providers.deepseek_provider.openai.AsyncOpenAI",
        FakeAsyncOpenAI,
    ):
        provider = DeepseekProvider(api_key="test-api-key")

    assert created["http_client"] is provider._http_client
    assert created["http_client"].__class__.__name__ != "AsyncHttpxClientWrapper"
    assert provider._owns_http_client is True

    await provider.aclose()

    assert provider.client.closed is True
    assert created["http_client"].is_closed is True


@pytest.mark.asyncio
async def test_deepseek_provider_moves_thinking_to_extra_body():
    provider = DeepseekProvider(api_key="test-api-key")
    mock_response = _ns(
        id="deepseek-response-id",
        created=1234567890,
        model="deepseek-v4-flash",
        usage=None,
        choices=[
            _ns(
                index=0,
                finish_reason="stop",
                message=_ns(
                    content="done",
                    role="assistant",
                    reasoning_content=None,
                    tool_calls=None,
                ),
            )
        ],
    )

    with patch.object(
        provider.client.chat.completions,
        "create",
        new=AsyncMock(return_value=mock_response),
    ) as mock_create:
        await provider.chat_completions_create(
            messages=[{"role": "user", "content": "Hello!"}],
            model="deepseek-v4-flash",
            thinking={"type": "disabled"},
        )

    mock_create.assert_called_once()
    assert mock_create.call_args.kwargs["extra_body"] == {
        "thinking": {"type": "disabled"}
    }


def test_deepseek_enabled_thinking_removes_conflicting_sampling_controls():
    provider = DeepseekProvider(api_key="test-api-key")

    prepared = provider._prepare_request_kwargs(
        {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
            "temperature": 0.2,
            "top_p": 0.8,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.1,
            "max_completion_tokens": 4096,
        }
    )

    assert prepared == {
        "extra_body": {"thinking": {"type": "enabled"}},
        "reasoning_effort": "high",
        "max_completion_tokens": 4096,
    }


def test_deepseek_disabled_thinking_drops_reasoning_effort_only():
    provider = DeepseekProvider(api_key="test-api-key")

    prepared = provider._prepare_request_kwargs(
        {
            "thinking": {"type": "disabled"},
            "reasoning": {"effort": "high"},
            "temperature": 0.2,
            "top_p": 0.8,
        }
    )

    assert prepared == {
        "extra_body": {"thinking": {"type": "disabled"}},
        "temperature": 0.2,
        "top_p": 0.8,
    }


@pytest.mark.asyncio
async def test_deepseek_provider_streaming_accumulates_reasoning_content():
    provider = DeepseekProvider(api_key="test-api-key")

    chunks = [
        _ns(
            id="chunk-1",
            created=1,
            model="deepseek-v4-pro",
            usage=None,
            choices=[
                _ns(
                    index=0,
                    finish_reason=None,
                    delta=_ns(
                        content=None,
                        role="assistant",
                        reasoning_content="think-1",
                        tool_calls=None,
                    ),
                )
            ],
        ),
        _ns(
            id="chunk-2",
            created=2,
            model="deepseek-v4-pro",
            usage=None,
            choices=[
                _ns(
                    index=0,
                    finish_reason=None,
                    delta=_ns(
                        content="done",
                        role=None,
                        reasoning_content=" think-2",
                        tool_calls=None,
                    ),
                )
            ],
        ),
        _ns(
            id="chunk-3",
            created=3,
            model="deepseek-v4-pro",
            usage=_ns(prompt_tokens=11, completion_tokens=4, total_tokens=15),
            choices=[
                _ns(
                    index=0,
                    finish_reason="stop",
                    delta=_ns(
                        content=None,
                        role=None,
                        reasoning_content=None,
                        tool_calls=None,
                    ),
                )
            ],
        ),
    ]

    async def fake_response():
        for chunk in chunks:
            yield chunk

    with patch.object(
        provider.client.chat.completions,
        "create",
        new=AsyncMock(return_value=fake_response()),
    ):
        stream = await provider.chat_completions_create(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )
        streamed_chunks = [chunk async for chunk in stream]

    assert streamed_chunks[0].choices[0].delta.reasoning_content == "think-1"
    assert streamed_chunks[1].choices[0].delta.reasoning_content == " think-2"
    assert streamed_chunks[-1].metadata["usage"]["total_tokens"] == 15

    accumulated_thinking = provider._get_accumulated_thinking()
    assert accumulated_thinking["thinking"] == "think-1 think-2"
    assert accumulated_thinking["raw_data"]["provider"] == "deepseek"
    assert (
        accumulated_thinking["raw_data"]["payload"]["reasoning_content"]
        == "think-1 think-2"
    )


def test_deepseek_provider_empty_accumulated_thinking_keeps_replay_payload():
    provider = DeepseekProvider(api_key="test-api-key")

    accumulated_thinking = provider._get_accumulated_thinking()

    assert accumulated_thinking["thinking"] == ""
    assert accumulated_thinking["raw_data"]["provider"] == "deepseek"
    assert accumulated_thinking["raw_data"]["payload"]["reasoning_content"] == ""


@pytest.mark.asyncio
async def test_deepseek_provider_stream_forwards_trailing_usage_only_chunk():
    provider = DeepseekProvider(api_key="test-api-key")

    async def fake_response():
        yield _ns(
            id="chunk-1",
            created=1,
            model="deepseek-v4-pro",
            usage=None,
            choices=[
                _ns(
                    index=0,
                    finish_reason="stop",
                    delta=_ns(
                        content="done",
                        role="assistant",
                        reasoning_content=None,
                        tool_calls=None,
                    ),
                )
            ],
        )
        yield _ns(
            id="chunk-2",
            created=2,
            model="deepseek-v4-pro",
            usage=_ns(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
            choices=[],
        )

    with patch.object(
        provider.client.chat.completions,
        "create",
        new=AsyncMock(return_value=fake_response()),
    ):
        stream = await provider.chat_completions_create(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )
        streamed_chunks = [chunk async for chunk in stream]

    assert streamed_chunks[0].choices[0].delta.content == "done"
    assert streamed_chunks[-1].choices == []
    assert streamed_chunks[-1].usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cache_read_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "cache_write_by_ttl": {
            "ephemeral_5m_input_tokens": 0,
            "ephemeral_1h_input_tokens": 0,
        },
    }


@pytest.mark.asyncio
async def test_deepseek_provider_closes_stream_response_after_iteration():
    provider = DeepseekProvider(api_key="test-api-key")
    stream_response = _ClosableAsyncStream(
        [
            _ns(
                id="chunk-1",
                created=1,
                model="deepseek-v4-flash",
                usage=None,
                choices=[
                    _ns(
                        index=0,
                        finish_reason="stop",
                        delta=_ns(
                            content="done",
                            role="assistant",
                            reasoning_content=None,
                            tool_calls=None,
                        ),
                    )
                ],
            )
        ]
    )

    with patch.object(
        provider.client.chat.completions,
        "create",
        new=AsyncMock(return_value=stream_response),
    ):
        stream = await provider.chat_completions_create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )
        chunks = [chunk async for chunk in stream]

    assert chunks[0].choices[0].delta.content == "done"
    assert stream_response.closed is True


@pytest.mark.asyncio
async def test_deepseek_provider_streaming_marks_pending_malformed_tool_call_retryable():
    provider = DeepseekProvider(api_key="test-api-key")

    chunks = [
        _ns(
            id="chunk-1",
            created=1,
            model="deepseek-v4-pro",
            usage=None,
            choices=[
                _ns(
                    index=0,
                    finish_reason=None,
                    delta=_ns(
                        content=None,
                        role="assistant",
                        reasoning_content=None,
                        tool_calls=[
                            _ns(
                                index=0,
                                id="call_1",
                                type="function",
                                function=_ns(name="edit_file", arguments='{"path":'),
                            )
                        ],
                    ),
                )
            ],
        ),
        _ns(
            id="chunk-2",
            created=2,
            model="deepseek-v4-pro",
            usage=_ns(prompt_tokens=11, completion_tokens=4, total_tokens=15),
            choices=[
                _ns(
                    index=0,
                    finish_reason="tool_calls",
                    delta=_ns(
                        content=None,
                        role=None,
                        reasoning_content=None,
                        tool_calls=None,
                    ),
                )
            ],
        ),
    ]

    async def fake_response():
        for chunk in chunks:
            yield chunk

    with patch.object(
        provider.client.chat.completions,
        "create",
        new=AsyncMock(return_value=fake_response()),
    ):
        stream = await provider.chat_completions_create(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )
        streamed_chunks = [chunk async for chunk in stream]

    assert streamed_chunks[-1].choices[0].finish_reason == "tool_calls"
    assert streamed_chunks[-1].choices[0].delta.tool_calls is None
    assert streamed_chunks[-1].choices[0].stop_info.reason.value == "tool_call_error"
    assert (
        streamed_chunks[-1].choices[0].stop_info.metadata["error_class"]
        == "malformed_streaming_tool_call_arguments"
    )
    assert streamed_chunks[-1].choices[0].stop_info.metadata["retryable"] is True
    assert (
        streamed_chunks[-1]
        .choices[0]
        .stop_info.metadata["pending_tool_calls"][0]["function_name"]
        == "edit_file"
    )
    assert (
        streamed_chunks[-1]
        .choices[0]
        .stop_info.metadata["pending_tool_calls"][0]["parse_error"]
    )


@pytest.mark.asyncio
async def test_deepseek_provider_streaming_preserves_accumulated_multiple_tool_call_count():
    provider = DeepseekProvider(api_key="test-api-key")

    chunks = [
        _ns(
            id="chunk-1",
            created=1,
            model="deepseek-v4-pro",
            usage=None,
            choices=[
                _ns(
                    index=0,
                    finish_reason=None,
                    delta=_ns(
                        content=None,
                        role="assistant",
                        reasoning_content=None,
                        tool_calls=[
                            _ns(
                                index=0,
                                id="call_1",
                                type="function",
                                function=_ns(
                                    name="read_file",
                                    arguments='{"path":"brief.md"}',
                                ),
                            ),
                            _ns(
                                index=1,
                                id="call_2",
                                type="function",
                                function=_ns(
                                    name="read_file",
                                    arguments='{"path":"facts.json"}',
                                ),
                            ),
                        ],
                    ),
                )
            ],
        ),
        _ns(
            id="chunk-2",
            created=2,
            model="deepseek-v4-pro",
            usage=None,
            choices=[
                _ns(
                    index=0,
                    finish_reason="tool_calls",
                    delta=_ns(
                        content=None,
                        role=None,
                        reasoning_content=None,
                        tool_calls=None,
                    ),
                )
            ],
        ),
    ]

    async def fake_response():
        for chunk in chunks:
            yield chunk

    with patch.object(
        provider.client.chat.completions,
        "create",
        new=AsyncMock(return_value=fake_response()),
    ):
        stream = await provider.chat_completions_create(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "Read both files."}],
            stream=True,
        )
        streamed_chunks = [chunk async for chunk in stream]

    assert len(streamed_chunks[0].choices[0].delta.tool_calls) == 2
    final_choice = streamed_chunks[-1].choices[0]
    assert final_choice.finish_reason == "tool_calls"
    assert final_choice.delta.tool_calls is None
    assert final_choice.stop_info.reason.value == "tool_call"
    assert final_choice.stop_info.metadata["tool_calls_count"] == 2
    assert final_choice.stop_info.metadata["has_content"] is True


@pytest.mark.asyncio
async def test_deepseek_provider_streaming_ignores_empty_terminal_tool_call_placeholder():
    provider = DeepseekProvider(api_key="test-api-key")

    chunks = [
        _ns(
            id="chunk-1",
            created=1,
            model="deepseek-v4-pro",
            usage=None,
            choices=[
                _ns(
                    index=0,
                    finish_reason=None,
                    delta=_ns(
                        content=None,
                        role="assistant",
                        reasoning_content=None,
                        tool_calls=[
                            _ns(
                                index=0,
                                id="call_1",
                                type="function",
                                function=_ns(
                                    name="read_file",
                                    arguments='{"path":"brief.md"}',
                                ),
                            )
                        ],
                    ),
                )
            ],
        ),
        _ns(
            id="chunk-2",
            created=2,
            model="deepseek-v4-pro",
            usage=None,
            choices=[
                _ns(
                    index=0,
                    finish_reason="tool_calls",
                    delta=_ns(
                        content=None,
                        role=None,
                        reasoning_content=None,
                        tool_calls=[
                            _ns(
                                index=0,
                                id=None,
                                type="function",
                                function=_ns(name=None, arguments=None),
                            )
                        ],
                    ),
                )
            ],
        ),
    ]

    async def fake_response():
        for chunk in chunks:
            yield chunk

    with patch.object(
        provider.client.chat.completions,
        "create",
        new=AsyncMock(return_value=fake_response()),
    ):
        stream = await provider.chat_completions_create(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "Read the file."}],
            stream=True,
        )
        streamed_chunks = [chunk async for chunk in stream]

    assert len(streamed_chunks[0].choices[0].delta.tool_calls) == 1
    final_choice = streamed_chunks[-1].choices[0]
    assert final_choice.finish_reason == "tool_calls"
    assert final_choice.delta.tool_calls is None
    assert final_choice.stop_info.reason.value == "tool_call"
    assert final_choice.stop_info.metadata["tool_calls_count"] == 1
    assert "error_class" not in final_choice.stop_info.metadata


@pytest.mark.asyncio
async def test_deepseek_provider_streaming_does_not_double_count_terminal_tool_call():
    provider = DeepseekProvider(api_key="test-api-key")

    def tool_call_chunk(
        *,
        chunk_id,
        index,
        call_id,
        path,
        finish_reason=None,
    ):
        return _ns(
            id=chunk_id,
            created=index + 1,
            model="deepseek-v4-pro",
            usage=None,
            choices=[
                _ns(
                    index=0,
                    finish_reason=finish_reason,
                    delta=_ns(
                        content=None,
                        role="assistant" if index == 0 else None,
                        reasoning_content=None,
                        tool_calls=[
                            _ns(
                                index=index,
                                id=call_id,
                                type="function",
                                function=_ns(
                                    name="read_file",
                                    arguments=f'{{"path":"{path}"}}',
                                ),
                            )
                        ],
                    ),
                )
            ],
        )

    chunks = [
        tool_call_chunk(
            chunk_id="chunk-1",
            index=0,
            call_id="call_1",
            path="brief.md",
        ),
        tool_call_chunk(
            chunk_id="chunk-2",
            index=1,
            call_id="call_2",
            path="facts.json",
            finish_reason="tool_calls",
        ),
    ]

    async def fake_response():
        for chunk in chunks:
            yield chunk

    with patch.object(
        provider.client.chat.completions,
        "create",
        new=AsyncMock(return_value=fake_response()),
    ):
        stream = await provider.chat_completions_create(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "Read both files."}],
            stream=True,
        )
        streamed_chunks = [chunk async for chunk in stream]

    final_choice = streamed_chunks[-1].choices[0]
    assert len(final_choice.delta.tool_calls) == 1
    assert final_choice.stop_info.reason.value == "tool_call"
    assert final_choice.stop_info.metadata["tool_calls_count"] == 2
