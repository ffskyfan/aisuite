from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from aisuite import Client
from aisuite.framework.message import ReasoningContent
from aisuite.framework.message_normalizer import MessageNormalizer
from aisuite.providers.kimi_provider import KimiProvider


def _provider():
    with patch("aisuite.providers.kimi_provider.openai.AsyncOpenAI"):
        return KimiProvider(api_key="test-key")


def test_kimi_provider_uses_official_default_endpoint():
    with patch("aisuite.providers.kimi_provider.openai.AsyncOpenAI") as client:
        KimiProvider(api_key="test-key")
    assert client.call_args.kwargs["base_url"] == "https://api.moonshot.ai/v1"


def test_kimi_provider_prepares_fixed_k3_parameters():
    provider = _provider()
    prepared = provider._prepare_request_kwargs(
        {
            "temperature": 0.7,
            "top_p": 0.8,
            "thinking": {"type": "enabled"},
            "reasoning": {"effort": "high"},
            "max_tokens": 4096,
        }
    )
    assert prepared == {
        "max_completion_tokens": 4096,
        "reasoning_effort": "max",
    }


def test_kimi_message_normalizer_preserves_reasoning_content():
    normalized = MessageNormalizer.normalize_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": {"thinking": "keep me"},
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ],
        "kimi:kimi-k3",
    )

    assert MessageNormalizer.detect_provider_type("kimi:kimi-k3") == "kimi"
    assert normalized[0]["reasoning_content"]["thinking"] == "keep me"


@pytest.mark.asyncio
async def test_kimi_client_replays_reasoning_and_tool_call_id_on_second_create():
    reasoning_text = "I need to read the requested file before answering."
    tool_call_id = "call_read_brief"
    first_transport_response = SimpleNamespace(
        id="chat_1",
        created=1,
        model="kimi-k3",
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    role="assistant",
                    content="",
                    reasoning_content=reasoning_text,
                    tool_calls=[
                        SimpleNamespace(
                            id=tool_call_id,
                            type="function",
                            function=SimpleNamespace(
                                name="read_file",
                                arguments='{"path":"brief.md"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        usage=None,
    )
    second_transport_response = SimpleNamespace(
        id="chat_2",
        created=2,
        model="kimi-k3",
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason="stop",
                message=SimpleNamespace(
                    role="assistant",
                    content="done",
                    reasoning_content="The file result is available.",
                    tool_calls=None,
                ),
            )
        ],
        usage=None,
    )

    with patch("aisuite.providers.kimi_provider.openai.AsyncOpenAI"):
        client = Client(provider_configs={"kimi": {"api_key": "test-key"}})

    transport_create = AsyncMock(
        side_effect=[first_transport_response, second_transport_response]
    )
    client.providers["kimi"].client.chat.completions.create = transport_create

    messages = [{"role": "user", "content": "Read brief.md"}]
    first_response = await client.chat.completions.create(
        model="kimi:kimi-k3",
        messages=messages,
    )
    assistant_message = first_response.choices[0].message
    messages.extend(
        [
            assistant_message,
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": "File contents",
            },
        ]
    )

    second_response = await client.chat.completions.create(
        model="kimi:kimi-k3",
        messages=messages,
    )

    assert second_response.choices[0].message.content == "done"
    second_request_messages = transport_create.await_args_list[1].kwargs["messages"]
    replayed_assistant = second_request_messages[-2]
    replayed_tool_result = second_request_messages[-1]
    assert replayed_assistant["reasoning_content"] == reasoning_text
    assert replayed_assistant["tool_calls"][0]["id"] == tool_call_id
    assert replayed_tool_result["tool_call_id"] == tool_call_id


@pytest.mark.asyncio
async def test_kimi_client_replays_complete_tool_message_when_reasoning_is_absent():
    tool_call_id = "call_read_outline"
    first_transport_response = SimpleNamespace(
        id="chat_1",
        created=1,
        model="kimi-k3",
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    role="assistant",
                    content="",
                    reasoning_content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id=tool_call_id,
                            type="function",
                            function=SimpleNamespace(
                                name="read_file",
                                arguments='{"path":"deliverables/launch-outline.md"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        usage=None,
    )
    second_transport_response = SimpleNamespace(
        id="chat_2",
        created=2,
        model="kimi-k3",
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason="stop",
                message=SimpleNamespace(
                    role="assistant",
                    content="done",
                    reasoning_content=None,
                    tool_calls=None,
                ),
            )
        ],
        usage=None,
    )

    with patch("aisuite.providers.kimi_provider.openai.AsyncOpenAI"):
        client = Client(provider_configs={"kimi": {"api_key": "test-key"}})

    transport_create = AsyncMock(
        side_effect=[first_transport_response, second_transport_response]
    )
    client.providers["kimi"].client.chat.completions.create = transport_create

    messages = [{"role": "user", "content": "Read the outline"}]
    first_response = await client.chat.completions.create(
        model="kimi:kimi-k3",
        messages=messages,
    )
    messages.extend(
        [
            first_response.choices[0].message,
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": "Outline contents",
            },
        ]
    )

    second_response = await client.chat.completions.create(
        model="kimi:kimi-k3",
        messages=messages,
    )

    assert second_response.choices[0].message.content == "done"
    second_request_messages = transport_create.await_args_list[1].kwargs["messages"]
    replayed_assistant = second_request_messages[-2]
    replayed_tool_result = second_request_messages[-1]
    assert "reasoning_content" not in replayed_assistant
    assert replayed_assistant["tool_calls"][0]["id"] == tool_call_id
    assert replayed_tool_result["tool_call_id"] == tool_call_id


def test_kimi_provider_round_trips_reasoning_replay_payload():
    provider = _provider()
    payload = provider._build_reasoning_replay_payload("use the tool")
    reasoning = ReasoningContent(
        thinking="display text",
        provider="kimi",
        raw_data=payload,
    )
    assert provider._extract_reasoning_input(reasoning) == "use the tool"


def test_kimi_provider_accepts_tool_call_replay_when_upstream_omits_reasoning():
    provider = _provider()
    validation = provider.validate_replay_window(
        "kimi-k3",
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
        ],
    )
    assert validation.ok is True
    assert "missing_reasoning_content" not in {
        diagnostic.code for diagnostic in validation.diagnostics
    }


def test_kimi_provider_accepts_complete_tool_call_replay():
    provider = _provider()
    reasoning = ReasoningContent(
        thinking="use the tool",
        provider="kimi",
        raw_data=provider._build_reasoning_replay_payload("use the tool"),
    )
    validation = provider.validate_replay_window(
        "kimi-k3",
        [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": reasoning,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ],
    )
    assert validation.ok is True


def test_kimi_provider_normalizes_top_level_cached_tokens():
    provider = _provider()
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        cached_tokens=60,
    )
    assert provider._normalize_usage(usage) == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "cache_read_input_tokens": 60,
        "cache_write_input_tokens": 0,
        "cache_write_by_ttl": {
            "ephemeral_5m_input_tokens": 0,
            "ephemeral_1h_input_tokens": 0,
        },
    }


@pytest.mark.asyncio
async def test_kimi_provider_non_stream_preserves_reasoning_and_usage():
    provider = _provider()
    provider.client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            id="chat_1",
            created=1,
            model="kimi-k3",
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason="stop",
                    message=SimpleNamespace(
                        role="assistant",
                        content="done",
                        reasoning_content="thinking",
                        tool_calls=None,
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                cached_tokens=4,
            ),
        )
    )

    response = await provider.chat_completions_create(
        model="kimi-k3",
        messages=[{"role": "user", "content": "hello"}],
    )

    message = response.choices[0].message
    assert message.reasoning_content.provider == "kimi"
    assert message.reasoning_content.thinking == "thinking"
    assert response.metadata["usage"]["cache_read_input_tokens"] == 4


@pytest.mark.asyncio
async def test_kimi_provider_stream_forwards_trailing_usage_only_chunk():
    provider = _provider()

    async def fake_response():
        yield SimpleNamespace(
            id="chat_1",
            created=1,
            model="kimi-k3",
            usage=None,
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason="stop",
                    delta=SimpleNamespace(
                        role="assistant",
                        content="done",
                        reasoning_content=None,
                        tool_calls=None,
                    ),
                )
            ],
        )
        yield SimpleNamespace(
            id="chat_1",
            created=1,
            model="kimi-k3",
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                cached_tokens=4,
            ),
            choices=[],
        )

    provider.client.chat.completions.create = AsyncMock(
        return_value=fake_response()
    )

    stream = await provider.chat_completions_create(
        model="kimi-k3",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )
    chunks = [chunk async for chunk in stream]

    assert chunks[0].choices[0].delta.content == "done"
    assert chunks[-1].choices == []
    assert chunks[-1].usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cache_read_input_tokens": 4,
        "cache_write_input_tokens": 0,
        "cache_write_by_ttl": {
            "ephemeral_5m_input_tokens": 0,
            "ephemeral_1h_input_tokens": 0,
        },
    }


@pytest.mark.asyncio
async def test_kimi_provider_stream_preserves_inherited_multiple_tool_call_count():
    provider = _provider()

    async def fake_response():
        yield SimpleNamespace(
            id="chat_1",
            created=1,
            model="kimi-k3",
            usage=None,
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason=None,
                    delta=SimpleNamespace(
                        role="assistant",
                        content=None,
                        reasoning_content="I need both files.",
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_1",
                                type="function",
                                function=SimpleNamespace(
                                    name="read_file",
                                    arguments='{"path":"brief.md"}',
                                ),
                            ),
                            SimpleNamespace(
                                index=1,
                                id="call_2",
                                type="function",
                                function=SimpleNamespace(
                                    name="read_file",
                                    arguments='{"path":"facts.json"}',
                                ),
                            ),
                        ],
                    ),
                )
            ],
        )
        yield SimpleNamespace(
            id="chat_1",
            created=2,
            model="kimi-k3",
            usage=None,
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason="tool_calls",
                    delta=SimpleNamespace(
                        role=None,
                        content=None,
                        reasoning_content=None,
                        tool_calls=None,
                    ),
                )
            ],
        )

    provider.client.chat.completions.create = AsyncMock(
        return_value=fake_response()
    )

    stream = await provider.chat_completions_create(
        model="kimi-k3",
        messages=[{"role": "user", "content": "Read both files."}],
        stream=True,
    )
    chunks = [chunk async for chunk in stream]

    assert len(chunks[0].choices[0].delta.tool_calls) == 2
    final_choice = chunks[-1].choices[0]
    assert final_choice.delta.tool_calls is None
    assert final_choice.stop_info.reason.value == "tool_call"
    assert final_choice.stop_info.metadata["tool_calls_count"] == 2
    assert final_choice.stop_info.metadata["has_content"] is True
