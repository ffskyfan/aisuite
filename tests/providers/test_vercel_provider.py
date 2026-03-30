from unittest.mock import patch

import pytest

from aisuite.framework.message_normalizer import MessageNormalizer
from aisuite.providers.openai_provider import OpenaiProvider
from aisuite.providers.vercel_provider import VercelProvider


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
async def test_vercel_provider_routes_to_explicit_anthropic_protocol():
    provider = VercelProvider(api_key="test-vercel-key")
    fake_provider = _FakeAsyncProvider("anthropic")
    provider._protocol_providers["anthropic"] = fake_provider

    response = await provider.chat_completions_create(
        model="claude-sonnet-4.6",
        messages=[{"role": "user", "content": "hello"}],
        protocol="anthropic",
        thinking={"type": "enabled"},
    )

    assert response["protocol"] == "anthropic"
    assert fake_provider.calls[0]["model"] == "anthropic/claude-sonnet-4.6"
    assert "protocol" not in fake_provider.calls[0]["kwargs"]
    assert fake_provider.calls[0]["kwargs"]["thinking"] == {"type": "enabled"}


@pytest.mark.asyncio
async def test_vercel_provider_routes_model_protocol_prefix():
    provider = VercelProvider(api_key="test-vercel-key")
    fake_provider = _FakeAsyncProvider("anthropic")
    provider._protocol_providers["anthropic"] = fake_provider

    response = await provider.chat_completions_create(
        model="anthropic/claude-sonnet-4.6",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )

    assert response["protocol"] == "anthropic"
    assert fake_provider.calls[0]["model"] == "anthropic/claude-sonnet-4.6"
    assert fake_provider.calls[0]["stream"] is True


@pytest.mark.asyncio
async def test_vercel_provider_routes_from_model_protocol_mapping():
    provider = VercelProvider(
        api_key="test-vercel-key",
        model_protocols={"claude-": "anthropic"},
    )
    fake_provider = _FakeAsyncProvider("anthropic")
    provider._protocol_providers["anthropic"] = fake_provider

    response = await provider.chat_completions_create(
        model="claude-sonnet-4.6",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert response["protocol"] == "anthropic"
    assert fake_provider.calls[0]["model"] == "anthropic/claude-sonnet-4.6"


@pytest.mark.asyncio
async def test_vercel_provider_infers_claude_protocol_by_default():
    provider = VercelProvider(api_key="test-vercel-key")
    fake_provider = _FakeAsyncProvider("anthropic")
    provider._protocol_providers["anthropic"] = fake_provider

    response = await provider.chat_completions_create(
        model="claude-sonnet-4.6",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert response["protocol"] == "anthropic"
    assert fake_provider.calls[0]["model"] == "anthropic/claude-sonnet-4.6"


@pytest.mark.asyncio
async def test_vercel_provider_prefixes_openai_models_for_openai_protocol():
    provider = VercelProvider(api_key="test-vercel-key")
    fake_provider = _FakeAsyncProvider("openai")
    provider._protocol_providers["openai"] = fake_provider

    response = await provider.chat_completions_create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert response["protocol"] == "openai"
    assert fake_provider.calls[0]["model"] == "openai/gpt-5.2"


def test_vercel_provider_derives_anthropic_base_url_from_openai_endpoint():
    provider = VercelProvider(
        api_key="test-vercel-key",
        base_url="https://ai-gateway.vercel.sh/v1",
    )

    with patch(
        "aisuite.providers.anthropic_provider.anthropic.AsyncAnthropic"
    ) as mock_client_cls:
        provider._create_protocol_provider("anthropic")

    _, kwargs = mock_client_cls.call_args
    assert kwargs["api_key"] == "test-vercel-key"
    assert kwargs["base_url"] == "https://ai-gateway.vercel.sh"


def test_vercel_provider_reuses_openai_provider_for_openai_protocol():
    provider = VercelProvider(
        api_key="test-vercel-key",
        base_url="https://ai-gateway.vercel.sh",
    )

    openai_provider = provider._create_protocol_provider("openai")

    assert isinstance(openai_provider, OpenaiProvider)


def test_message_normalizer_detects_vercel_claude_as_anthropic():
    assert (
        MessageNormalizer.detect_provider_type("vercel:claude-sonnet-4.6")
        == "anthropic"
    )
    assert (
        MessageNormalizer.detect_provider_type(
            "vercel:anthropic/claude-sonnet-4.6"
        )
        == "anthropic"
    )
    assert (
        MessageNormalizer.detect_provider_type("vercel:openai/gpt-5.2")
        == "openai"
    )
