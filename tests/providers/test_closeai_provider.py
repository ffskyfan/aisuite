from unittest.mock import patch

import pytest

from aisuite.framework.message_normalizer import MessageNormalizer
from aisuite.framework.replay_payload import (
    ProviderReplayCapabilities,
    ReplayBuildResult,
    ReplayCaptureResult,
    ReplayDiagnostic,
    ReplayValidationResult,
)
from aisuite.providers.closeai_provider import CloseaiProvider
from aisuite.providers.openai_provider import OpenaiProvider


class _FakeAsyncProvider:
    def __init__(self, protocol: str):
        self.protocol = protocol
        self.calls = []
        self.validation_calls = []
        self.capture_calls = []
        self.build_calls = []

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

    def get_replay_capabilities(self, model=None):
        return ProviderReplayCapabilities(
            needs_provider_call_id_binding=self.protocol == "anthropic",
            supports_canonical_only_history=True,
        )

    def capture_response(self, response, model=None, **kwargs):
        self.capture_calls.append({"response": response, "model": model, "kwargs": kwargs})
        return ReplayCaptureResult(
            canonical_message={"role": "assistant", "content": "captured"},
            replay_metadata={"protocol": self.protocol},
        )

    def validate_replay_window(self, model, messages, **kwargs):
        self.validation_calls.append(
            {"model": model, "messages": messages, "kwargs": kwargs}
        )
        return ReplayValidationResult(
            ok=True,
            diagnostics=(
                ReplayDiagnostic(
                    code="delegated",
                    message="validated by fake provider",
                    provider=self.protocol,
                ),
            ),
        )

    def build_replay_view(self, model, messages, **kwargs):
        self.build_calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        return ReplayBuildResult(
            request_view={"protocol": self.protocol, "model": model, "messages": messages},
            replay_mode=f"{self.protocol}_replay",
        )


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


def test_closeai_replay_contract_delegates_to_anthropic_provider():
    provider = CloseaiProvider(api_key="test-closeai-key")
    fake_provider = _FakeAsyncProvider("anthropic")
    provider._protocol_providers["anthropic"] = fake_provider

    capabilities = provider.get_replay_capabilities("claude-sonnet-4-20250514")
    validation = provider.validate_replay_window(
        "claude-sonnet-4-20250514",
        [{"role": "tool", "tool_call_id": "tool_1", "content": "ok"}],
        protocol="anthropic",
    )
    replay_build = provider.build_replay_view(
        "claude-sonnet-4-20250514",
        [{"role": "user", "content": "hello"}],
        protocol="anthropic",
    )
    captured = provider.capture_response(
        {"response": "ok"},
        model="claude-sonnet-4-20250514",
        protocol="anthropic",
    )

    assert capabilities.needs_provider_call_id_binding is True
    assert validation.ok is True
    assert validation.diagnostics[0].provider == "anthropic"
    assert fake_provider.validation_calls[0]["model"] == "claude-sonnet-4-20250514"
    assert replay_build.replay_mode == "anthropic_replay"
    assert replay_build.request_view["model"] == "claude-sonnet-4-20250514"
    assert captured.replay_metadata["protocol"] == "anthropic"
    assert fake_provider.capture_calls[0]["model"] == "claude-sonnet-4-20250514"
