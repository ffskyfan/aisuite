from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from aisuite.framework.message import ReasoningContent
from aisuite.providers.deepseek_provider import DeepseekProvider


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


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


def test_deepseek_validate_replay_window_reports_missing_tool_call_id():
    provider = DeepseekProvider(api_key="test-api-key")

    result = provider.validate_replay_window(
        "deepseek-v4-flash",
        [{"role": "tool", "content": "result"}],
    )

    assert result.ok is False
    assert any(diag.code == "missing_tool_call_id" for diag in result.diagnostics)


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
    assert (
        captured.replay_metadata["reasoning_content"]["provider"] == "deepseek"
    )


def test_deepseek_provider_preserves_custom_base_url():
    with patch("aisuite.providers.deepseek_provider.openai.AsyncOpenAI") as mock_client:
        DeepseekProvider(api_key="test-api-key", base_url="https://api.deepseek.com/beta")

    assert mock_client.call_args.kwargs["base_url"] == "https://api.deepseek.com/beta"


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
    assert mock_create.call_args.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
