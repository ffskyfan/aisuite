from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from aisuite import Client
from aisuite.framework.content import (
    filter_images_for_capabilities,
    make_image_data_url,
)
from aisuite.framework.message import ReasoningContent
from aisuite.framework.message_normalizer import MessageNormalizer
from aisuite.provider import ProviderFactory
from aisuite.providers.qwen_provider import QwenProvider


def _provider(**config):
    with patch("aisuite.providers.deepseek_provider.openai.AsyncOpenAI"):
        return QwenProvider(api_key="test-key", **config)


@pytest.fixture(autouse=True)
def clear_qwen_environment(monkeypatch):
    for name in (
        "QWEN_API_KEY",
        "QWEN_BASE_URL",
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "BAILIAN_TOKEN_PLAN_API_KEY",
        "BAILIAN_TOKEN_PLAN_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_qwen_provider_is_discovered_by_factory():
    assert "qwen" in ProviderFactory.get_supported_providers()


def test_qwen_provider_uses_token_plan_endpoint_for_subscription_key():
    with patch("aisuite.providers.deepseek_provider.openai.AsyncOpenAI") as mock_client:
        QwenProvider(api_key="sk-sp-subscription-key")

    assert mock_client.call_args.kwargs["base_url"] == QwenProvider.TOKEN_PLAN_BASE_URL


def test_qwen_provider_uses_dashscope_endpoint_for_standard_key():
    with patch("aisuite.providers.deepseek_provider.openai.AsyncOpenAI") as mock_client:
        QwenProvider(api_key="sk-standard-key")

    assert mock_client.call_args.kwargs["base_url"] == QwenProvider.DASHSCOPE_BASE_URL


def test_qwen_provider_uses_official_token_plan_environment(monkeypatch):
    monkeypatch.setenv("BAILIAN_TOKEN_PLAN_API_KEY", "token-plan-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "pay-as-you-go-key")
    with patch("aisuite.providers.deepseek_provider.openai.AsyncOpenAI") as mock_client:
        QwenProvider()

    assert mock_client.call_args.kwargs["api_key"] == "token-plan-key"
    assert mock_client.call_args.kwargs["base_url"] == QwenProvider.TOKEN_PLAN_BASE_URL


def test_qwen_provider_preserves_custom_base_url():
    with patch("aisuite.providers.deepseek_provider.openai.AsyncOpenAI") as mock_client:
        QwenProvider(
            api_key="test-key",
            base_url="https://workspace.example/compatible-mode/v1",
        )

    assert (
        mock_client.call_args.kwargs["base_url"]
        == "https://workspace.example/compatible-mode/v1"
    )


def test_qwen_message_normalizer_preserves_reasoning_content():
    normalized = MessageNormalizer.normalize_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": {"thinking": "keep me"},
            }
        ],
        "qwen:qwen3.8-max",
    )

    assert MessageNormalizer.detect_provider_type("qwen:qwen3.8-max") == "qwen"
    assert normalized[0]["reasoning_content"]["thinking"] == "keep me"


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("low", "low"),
        ("medium", "medium"),
        ("xhigh", "xhigh"),
        ("minimal", "low"),
        ("high", "xhigh"),
        ("max", "xhigh"),
    ],
)
def test_qwen_provider_normalizes_documented_reasoning_effort_aliases(
    requested, expected
):
    provider = _provider()

    assert provider._prepare_request_kwargs({"reasoning_effort": requested}) == {
        "reasoning_effort": expected
    }


def test_qwen_provider_maps_none_reasoning_effort_to_disabled_thinking():
    provider = _provider()

    assert provider._prepare_request_kwargs(
        {"reasoning_effort": "none", "thinking_budget": 4096}
    ) == {"extra_body": {"enable_thinking": False}}


def test_qwen_provider_moves_nonstandard_controls_to_extra_body():
    provider = _provider()

    prepared = provider._prepare_request_kwargs(
        {
            "enable_thinking": True,
            "preserve_thinking": True,
            "enable_search": True,
            "search_options": {"forced_search": True},
            "vl_high_resolution_images": True,
            "max_completion_tokens": 4096,
        }
    )

    assert prepared == {
        "max_completion_tokens": 4096,
        "extra_body": {
            "enable_thinking": True,
            "preserve_thinking": True,
            "enable_search": True,
            "search_options": {"forced_search": True},
            "vl_high_resolution_images": True,
        },
    }


def test_qwen_provider_maps_legacy_thinking_control():
    provider = _provider()

    assert provider._prepare_request_kwargs(
        {"thinking": {"type": "disabled"}, "reasoning_effort": "xhigh"}
    ) == {"extra_body": {"enable_thinking": False}}


def test_qwen_provider_rejects_reasoning_effort_with_thinking_budget():
    provider = _provider()

    with pytest.raises(ValueError, match="reasoning_effort and thinking_budget"):
        provider._prepare_request_kwargs(
            {"reasoning_effort": "medium", "thinking_budget": 4096}
        )


def test_qwen_provider_rejects_multiple_choices_with_tools():
    provider = _provider()

    with pytest.raises(ValueError, match="requires n=1"):
        provider._prepare_request_kwargs(
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "inspect_scene",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "n": 2,
            }
        )


def test_qwen38_declares_effective_user_and_tool_result_vision_support():
    provider = _provider()

    capabilities = provider.get_multimodal_capabilities("qwen3.8-max")

    assert capabilities.user_images == "supported"
    assert capabilities.tool_result_images == "supported"


def test_qwen_projects_tool_result_images_into_following_user_message():
    provider = _provider()
    image_url = make_image_data_url("image/png", b"frame")

    prepared = provider._prepare_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-frame",
                        "type": "function",
                        "function": {
                            "name": "inspect_frame",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-frame",
                "content": [
                    {"type": "text", "text": "Captured the current frame."},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "high"},
                    },
                ],
            },
        ]
    )

    assert prepared[1]["role"] == "tool"
    assert prepared[1]["content"] == "Captured the current frame."
    assert prepared[2]["role"] == "user"
    assert prepared[2]["content"][0]["type"] == "text"
    assert prepared[2]["content"][1]["image_url"]["url"] == image_url


@pytest.mark.asyncio
async def test_qwen_client_preserves_user_image_and_stringifies_tool_result():
    image_url = make_image_data_url("image/png", b"frame")
    with patch("aisuite.providers.deepseek_provider.openai.AsyncOpenAI"):
        client = Client(provider_configs={"qwen": {"api_key": "test-key"}})

    transport_create = AsyncMock(
        return_value=SimpleNamespace(
            id="chat_1",
            created=1,
            model="qwen3.8-max",
            usage=None,
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason="stop",
                    message=SimpleNamespace(
                        role="assistant",
                        content="done",
                        reasoning_content="checked the image",
                        tool_calls=None,
                    ),
                )
            ],
        )
    )
    client.providers["qwen"].client.chat.completions.create = transport_create
    reasoning = ReasoningContent(
        thinking="inspect the frame",
        provider="qwen",
        raw_data=client.providers["qwen"]._build_reasoning_replay_payload(
            "inspect the frame"
        ),
    )

    await client.chat.completions.create(
        model="qwen:qwen3.8-max",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "high"},
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": reasoning,
                "tool_calls": [
                    {
                        "id": "call_capture",
                        "type": "function",
                        "function": {"name": "capture", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_capture",
                "is_error": False,
                "content": [
                    {"type": "text", "text": "captured"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "high"},
                    },
                ],
            },
        ],
    )

    sent_messages = transport_create.await_args.kwargs["messages"]
    assert sent_messages[0]["content"][1]["image_url"]["url"] == image_url
    assert sent_messages[1]["reasoning_content"] == "inspect the frame"
    assert sent_messages[2]["content"] == "captured"
    assert "is_error" not in sent_messages[2]


@pytest.mark.asyncio
async def test_qwen_client_replays_reasoning_across_tool_round_trip():
    reasoning_text = "I should inspect the scene before editing it."
    tool_call_id = "call_inspect_scene"
    first_transport_response = SimpleNamespace(
        id="chat_1",
        created=1,
        model="qwen3.8-max",
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
                                name="inspect_scene",
                                arguments='{"scene":"main"}',
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
        model="qwen3.8-max",
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason="stop",
                message=SimpleNamespace(
                    role="assistant",
                    content="done",
                    reasoning_content="The scene is ready.",
                    tool_calls=None,
                ),
            )
        ],
        usage=None,
    )

    with patch("aisuite.providers.deepseek_provider.openai.AsyncOpenAI"):
        client = Client(provider_configs={"qwen": {"api_key": "test-key"}})
    transport_create = AsyncMock(
        side_effect=[first_transport_response, second_transport_response]
    )
    client.providers["qwen"].client.chat.completions.create = transport_create

    messages = [{"role": "user", "content": "Inspect the main scene"}]
    first_response = await client.chat.completions.create(
        model="qwen:qwen3.8-max",
        messages=messages,
    )
    messages.extend(
        [
            first_response.choices[0].message,
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": "Scene contents",
            },
        ]
    )

    second_response = await client.chat.completions.create(
        model="qwen:qwen3.8-max",
        messages=messages,
    )

    assert second_response.choices[0].message.content == "done"
    second_request_messages = transport_create.await_args_list[1].kwargs["messages"]
    assert second_request_messages[-2]["reasoning_content"] == reasoning_text
    assert second_request_messages[-2]["tool_calls"][0]["id"] == tool_call_id
    assert second_request_messages[-1]["tool_call_id"] == tool_call_id
    assert transport_create.await_args_list[0].kwargs["extra_body"] == {
        "preserve_thinking": True
    }
    assert transport_create.await_args_list[1].kwargs["extra_body"] == {
        "preserve_thinking": True
    }


def test_qwen38_replay_marks_missing_tool_reasoning_as_degraded():
    provider = _provider()

    validation = provider.validate_replay_window(
        "qwen3.8-max",
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
    assert validation.degraded is True
    assert any(
        diagnostic.code == "missing_reasoning_content"
        for diagnostic in validation.diagnostics
    )


def test_qwen_provider_normalizes_cache_and_multimodal_usage_details():
    provider = _provider()
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 30,
        "total_tokens": 130,
        "prompt_tokens_details": {
            "cached_tokens": 40,
            "text_tokens": 60,
            "image_tokens": 40,
            "video_tokens": 0,
            "cache_creation": {
                "cache_creation_input_tokens": 20,
                "ephemeral_5m_input_tokens": 20,
            },
        },
        "completion_tokens_details": {
            "reasoning_tokens": 10,
            "text_tokens": 20,
        },
    }

    normalized = provider._normalize_usage(usage)

    assert normalized["cache_read_input_tokens"] == 40
    assert normalized["cache_write_input_tokens"] == 20
    assert normalized["cache_write_by_ttl"]["ephemeral_5m_input_tokens"] == 20
    assert normalized["reasoning_tokens"] == 10
    assert normalized["input_text_tokens"] == 60
    assert normalized["input_image_tokens"] == 40
    assert normalized["input_video_tokens"] == 0
    assert normalized["output_text_tokens"] == 20


@pytest.mark.asyncio
async def test_qwen_stream_preserves_reasoning_tool_calls_and_trailing_usage():
    provider = _provider()

    async def fake_response():
        yield SimpleNamespace(
            id="chat_1",
            created=1,
            model="qwen3.8-max",
            usage=None,
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason=None,
                    delta=SimpleNamespace(
                        role="assistant",
                        content=None,
                        reasoning_content="inspect first",
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_1",
                                type="function",
                                function=SimpleNamespace(
                                    name="inspect_scene", arguments="{}"
                                ),
                            )
                        ],
                    ),
                )
            ],
        )
        yield SimpleNamespace(
            id="chat_1",
            created=2,
            model="qwen3.8-max",
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
        yield SimpleNamespace(
            id="chat_1",
            created=3,
            model="qwen3.8-max",
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                prompt_tokens_details=SimpleNamespace(cached_tokens=4),
                completion_tokens_details=None,
            ),
            choices=[],
        )

    provider.client.chat.completions.create = AsyncMock(return_value=fake_response())

    stream = await provider.chat_completions_create(
        model="qwen3.8-max",
        messages=[{"role": "user", "content": "Inspect the scene"}],
        stream=True,
    )
    chunks = [chunk async for chunk in stream]

    assert chunks[0].choices[0].delta.reasoning_content == "inspect first"
    assert chunks[0].choices[0].delta.tool_calls[0].id == "call_1"
    assert chunks[1].choices[0].stop_info.metadata["provider"] == "qwen"
    assert chunks[-1].choices == []
    assert chunks[-1].metadata["usage"]["cache_read_input_tokens"] == 4

    accumulated = provider._get_accumulated_thinking()
    assert accumulated["thinking"] == "inspect first"
    assert accumulated["raw_data"]["provider"] == "qwen"


def test_qwen_tool_result_image_without_text_is_projected_after_tool_marker():
    provider = _provider()
    image_url = make_image_data_url("image/png", b"frame")

    prepared, removed = filter_images_for_capabilities(
        [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "high"},
                    }
                ],
            }
        ],
        provider.get_multimodal_capabilities("qwen3.8-max"),
    )

    assert removed == 0
    wire_messages = provider._prepare_messages(prepared)
    assert wire_messages[0]["content"] == provider.TOOL_IMAGE_FORWARD_TEXT
    assert wire_messages[1]["role"] == "user"
    assert wire_messages[1]["content"][1]["image_url"]["url"] == image_url
