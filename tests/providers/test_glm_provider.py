from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aisuite.framework.message_normalizer import MessageNormalizer
from aisuite.provider import LLMError
from aisuite.providers.glm_provider import GlmProvider


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


@pytest.mark.asyncio
async def test_glm_provider_non_stream(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")

    provider = GlmProvider()

    mock_response = _ns(
        id="glm-response-id",
        created=1234567890,
        model="glm-5",
        usage=_ns(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_tokens_details=_ns(cached_tokens=3),
            completion_tokens_details=_ns(reasoning_tokens=2),
        ),
        choices=[
            _ns(
                index=0,
                finish_reason="stop",
                message=_ns(
                    content="mocked-text-response-from-model",
                    role="assistant",
                    reasoning_content="step-by-step",
                    tool_calls=[
                        _ns(
                            id="call_1",
                            function=_ns(
                                name="lookup_weather",
                                arguments='{"city":"Shanghai"}',
                            ),
                        )
                    ],
                ),
            )
        ],
    )
    mock_post = AsyncMock(return_value=mock_response)
    monkeypatch.setattr(provider, "_async_post_json", mock_post)

    response = await provider.chat_completions_create(
        model="glm-5",
        messages=[
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": {
                    "thinking": "previous reasoning trace",
                    "provider": "glm",
                },
            },
            {"role": "user", "content": "Hello!"},
        ],
        temperature=0.2,
        reasoning={"type": "enabled"},
        stream_options={"include_usage": True},
    )

    mock_post.assert_awaited_once_with(
        {
            "model": "glm-5",
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "previous reasoning trace",
                },
                {"role": "user", "content": "Hello!"},
            ],
            "stream": False,
            "temperature": 0.2,
            "thinking": {"type": "enabled"},
        }
    )

    assert response.choices[0].message.content == "mocked-text-response-from-model"
    assert response.choices[0].message.reasoning_content.thinking == "step-by-step"
    assert response.choices[0].message.tool_calls[0].function.name == "lookup_weather"
    assert response.metadata["usage"]["cache_read_input_tokens"] == 3


@pytest.mark.asyncio
async def test_glm_provider_defaults_thinking_to_disabled(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")

    provider = GlmProvider()

    mock_response = _ns(
        id="glm-response-id",
        created=1234567890,
        model="glm-5",
        usage=_ns(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        choices=[
            _ns(
                index=0,
                finish_reason="stop",
                message=_ns(
                    content="ok",
                    role="assistant",
                    reasoning_content=None,
                    tool_calls=None,
                ),
            )
        ],
    )
    mock_post = AsyncMock(return_value=mock_response)
    monkeypatch.setattr(provider, "_async_post_json", mock_post)

    await provider.chat_completions_create(
        model="glm-5",
        messages=[{"role": "user", "content": "Hello!"}],
    )

    mock_post.assert_awaited_once_with(
        {
            "model": "glm-5",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": False,
            "thinking": {"type": "disabled"},
        }
    )


@pytest.mark.asyncio
async def test_glm_provider_streaming(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")

    provider = GlmProvider()

    chunks = [
        _ns(
            id="chunk-1",
            created=1,
            model="glm-5",
            usage=None,
            choices=[
                _ns(
                    index=0,
                    delta=_ns(
                        content="Hel",
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
            model="glm-5",
            usage=_ns(prompt_tokens=11, completion_tokens=4, total_tokens=15),
            choices=[
                _ns(
                    index=0,
                    finish_reason="stop",
                    delta=_ns(
                        content="lo",
                        role=None,
                        reasoning_content=None,
                        tool_calls=None,
                    ),
                )
            ],
        ),
    ]
    stream_bodies = []

    async def fake_stream_json(body):
        stream_bodies.append(body)
        for chunk in chunks:
            yield chunk

    monkeypatch.setattr(provider, "_async_stream_json", fake_stream_json)

    stream = await provider.chat_completions_create(
        model="glm-5",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    )
    streamed_chunks = [chunk async for chunk in stream]

    assert stream_bodies == [
        {
            "model": "glm-5",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": True,
            "thinking": {"type": "disabled"},
        }
    ]
    assert streamed_chunks[0].choices[0].delta.content == "Hel"
    assert streamed_chunks[0].choices[0].delta.reasoning_content == "think-1"
    assert streamed_chunks[-1].choices[0].finish_reason == "stop"
    assert streamed_chunks[-1].metadata["usage"]["total_tokens"] == 15
    assert streamed_chunks[-1].choices[0].stop_info.reason.value == "complete"


@pytest.mark.asyncio
async def test_glm_provider_streaming_preserves_accumulated_tool_call_count(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")

    provider = GlmProvider()

    chunks = [
        _ns(
            id="chunk-1",
            created=1,
            model="glm-5",
            usage=None,
            choices=[
                _ns(
                    index=0,
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
                                    name="lookup_weather",
                                    arguments='{"city":"Shanghai"}',
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
            model="glm-5",
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

    async def fake_stream_json(_body):
        for chunk in chunks:
            yield chunk

    monkeypatch.setattr(provider, "_async_stream_json", fake_stream_json)

    stream = await provider.chat_completions_create(
        model="glm-5",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    )
    streamed_chunks = [chunk async for chunk in stream]

    assert streamed_chunks[0].choices[0].delta.tool_calls is not None
    assert streamed_chunks[-1].choices[0].stop_info.reason.value == "tool_call"
    assert streamed_chunks[-1].choices[0].stop_info.metadata["tool_calls_count"] == 1


@pytest.mark.asyncio
async def test_glm_provider_streaming_raises_on_network_error(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")

    provider = GlmProvider()

    chunks = [
        _ns(
            id="chunk-1",
            created=1,
            model="glm-5",
            usage=None,
            choices=[
                _ns(
                    index=0,
                    finish_reason="network_error",
                    delta=_ns(
                        content=None,
                        role="assistant",
                        reasoning_content=None,
                        tool_calls=None,
                    ),
                )
            ],
        ),
    ]

    async def fake_stream_json(_body):
        for chunk in chunks:
            yield chunk

    monkeypatch.setattr(provider, "_async_stream_json", fake_stream_json)

    stream = await provider.chat_completions_create(
        model="glm-5",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    )

    with pytest.raises(LLMError, match="network_error"):
        async for _chunk in stream:
            pass


@pytest.mark.asyncio
async def test_glm_provider_parses_async_sse(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    provider = GlmProvider()

    class FakeResponse:
        async def aiter_lines(self):
            lines = [
                'data: {"id":"chunk-1","model":"glm-5","choices":[]}',
                "",
                "data: [DONE]",
                "",
            ]
            for line in lines:
                yield line

    chunks = [chunk async for chunk in provider._iter_sse_json(FakeResponse())]

    assert chunks[0].id == "chunk-1"
    assert chunks[0].model == "glm-5"


@pytest.mark.asyncio
async def test_glm_provider_closes_async_stream_on_aclose(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    provider = GlmProvider()

    class FakeResponse:
        def __init__(self):
            self.closed = False

        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            lines = [
                'data: {"id":"chunk-1","model":"glm-5","choices":[]}',
                "",
                'data: {"id":"chunk-2","model":"glm-5","choices":[]}',
                "",
            ]
            for line in lines:
                yield line

        async def aclose(self):
            self.closed = True

    fake_response = FakeResponse()
    client_closed = False

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            nonlocal client_closed
            client_closed = True

        def build_request(self, method, url, headers, json):
            return _ns(method=method, url=url, headers=headers, json=json)

        async def send(self, request, stream):
            assert stream is True
            return fake_response

    monkeypatch.setattr(
        "aisuite.providers.glm_provider.httpx.AsyncClient",
        FakeAsyncClient,
    )

    stream = provider._async_stream_json({"model": "glm-5", "messages": []})
    first_chunk = await anext(stream)

    assert first_chunk.id == "chunk-1"

    await stream.aclose()

    assert fake_response.closed is True
    assert client_closed is True


def test_glm_message_normalizer_preserves_reasoning_content():
    normalized = MessageNormalizer.normalize_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": {"thinking": "keep me"},
            }
        ],
        "glm:glm-5",
    )

    assert normalized[0]["reasoning_content"]["thinking"] == "keep me"


def test_glm_build_replay_view_preserves_reasoning_content(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")

    with patch("aisuite.providers.glm_provider.ZhipuAiClient"):
        provider = GlmProvider()

        replay_build = provider.build_replay_view(
            "glm-5",
            [
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": {
                        "thinking": "cached reasoning",
                        "provider": "glm",
                    },
                },
                {"role": "user", "content": "hi"},
            ],
        )

    assert replay_build.replay_mode == "canonical_with_reasoning"
    assert replay_build.request_view[0]["reasoning_content"] == "cached reasoning"


def test_glm_validate_replay_window_reports_missing_tool_call_id(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")

    with patch("aisuite.providers.glm_provider.ZhipuAiClient"):
        provider = GlmProvider()
        result = provider.validate_replay_window(
            "glm-5",
            [{"role": "tool", "content": "result"}],
        )

    assert result.ok is False
    assert any(diag.code == "missing_tool_call_id" for diag in result.diagnostics)


def test_glm_capture_response_returns_structured_result(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")

    with patch("aisuite.providers.glm_provider.ZhipuAiClient"):
        provider = GlmProvider()
        response = _ns(
            choices=[
                _ns(
                    stop_info=_ns(reason="complete"),
                    message=_ns(
                        role="assistant",
                        content="done",
                        tool_calls=None,
                        reasoning_content=provider._convert_reasoning_content("think"),
                    ),
                )
            ]
        )

        captured = provider.capture_response(response, model="glm-5")

    assert captured.canonical_message.content == "done"
    assert captured.replay_metadata["reasoning_content"]["provider"] == "glm"


@pytest.mark.asyncio
async def test_glm_provider_non_stream_raises_on_network_error(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")

    provider = GlmProvider()

    mock_response = _ns(
        id="glm-response-id",
        created=1234567890,
        model="glm-5",
        usage=_ns(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        choices=[
            _ns(
                index=0,
                finish_reason="network_error",
                message=_ns(
                    content="",
                    role="assistant",
                    reasoning_content=None,
                    tool_calls=None,
                ),
            )
        ],
    )
    mock_post = AsyncMock(return_value=mock_response)
    monkeypatch.setattr(provider, "_async_post_json", mock_post)

    with pytest.raises(LLMError, match="network_error"):
        await provider.chat_completions_create(
            model="glm-5",
            messages=[{"role": "user", "content": "Hello!"}],
        )
