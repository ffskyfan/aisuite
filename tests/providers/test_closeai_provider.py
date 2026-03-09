from unittest.mock import patch

import pytest

from aisuite.framework.message_normalizer import MessageNormalizer
from aisuite.providers.closeai_provider import CloseaiProvider
from aisuite.providers.openai_provider import OpenaiProvider


class _FakeAsyncProvider:
    def __init__(self, protocol: str):
        self.protocol = protocol
        self.calls = []

    async def chat_completions_create(self, model, messages, stream=False, **kwargs):
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "stream": stream,
                "kwargs": kwargs,
            }
        )
        return {"protocol": self.protocol, "model": model, "kwargs": kwargs}


@pytest.mark.asyncio
async def test_closeai_provider_routes_to_explicit_anthropic_protocol():
    provider = CloseaiProvider(api_key="test-closeai-key")
    fake_provider = _FakeAsyncProvider("anthropic")
    provider._protocol_providers["anthropic"] = fake_provider

    response = await provider.chat_completions_create(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "hello"}],
        protocol="anthropic",
        thinking={"type": "enabled"},
    )

    assert response["protocol"] == "anthropic"
    assert fake_provider.calls[0]["model"] == "claude-sonnet-4-20250514"
    assert "protocol" not in fake_provider.calls[0]["kwargs"]
    assert fake_provider.calls[0]["kwargs"]["thinking"] == {"type": "enabled"}


@pytest.mark.asyncio
async def test_closeai_provider_routes_model_protocol_prefix():
    provider = CloseaiProvider(api_key="test-closeai-key")
    fake_provider = _FakeAsyncProvider("anthropic")
    provider._protocol_providers["anthropic"] = fake_provider

    response = await provider.chat_completions_create(
        model="anthropic/claude-3-7-sonnet-20250219",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )

    assert response["protocol"] == "anthropic"
    assert fake_provider.calls[0]["model"] == "claude-3-7-sonnet-20250219"
    assert fake_provider.calls[0]["stream"] is True


@pytest.mark.asyncio
async def test_closeai_provider_routes_from_model_protocol_mapping():
    provider = CloseaiProvider(
        api_key="test-closeai-key",
        model_protocols={"claude-": "anthropic"},
    )
    fake_provider = _FakeAsyncProvider("anthropic")
    provider._protocol_providers["anthropic"] = fake_provider

    response = await provider.chat_completions_create(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert response["protocol"] == "anthropic"
    assert fake_provider.calls[0]["model"] == "claude-sonnet-4-20250514"


@pytest.mark.asyncio
async def test_closeai_provider_infers_claude_protocol_by_default():
    provider = CloseaiProvider(api_key="test-closeai-key")
    fake_provider = _FakeAsyncProvider("anthropic")
    provider._protocol_providers["anthropic"] = fake_provider

    response = await provider.chat_completions_create(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert response["protocol"] == "anthropic"
    assert fake_provider.calls[0]["model"] == "claude-sonnet-4-6"


def test_closeai_provider_derives_anthropic_base_url_from_openai_endpoint():
    provider = CloseaiProvider(
        api_key="test-closeai-key",
        base_url="https://api.openai-proxy.org/v1",
    )

    with patch("aisuite.providers.anthropic_provider.anthropic.AsyncAnthropic") as mock_client_cls:
        provider._create_protocol_provider("anthropic")

    _, kwargs = mock_client_cls.call_args
    assert kwargs["api_key"] == "test-closeai-key"
    assert kwargs["base_url"] == "https://api.openai-proxy.org/anthropic"


def test_closeai_provider_reuses_openai_provider_for_openai_protocol():
    provider = CloseaiProvider(
        api_key="test-closeai-key",
        base_url="https://api.openai-proxy.org/v1",
    )

    openai_provider = provider._create_protocol_provider("openai")

    assert isinstance(openai_provider, OpenaiProvider)


def test_message_normalizer_detects_closeai_claude_as_anthropic():
    assert (
        MessageNormalizer.detect_provider_type(
            "closeai:claude-3-7-sonnet-20250219"
        )
        == "anthropic"
    )
    assert (
        MessageNormalizer.detect_provider_type(
            "closeai:anthropic/claude-sonnet-4-20250514"
        )
        == "anthropic"
    )
