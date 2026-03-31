import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aisuite.framework.stop_reason import StopReason
from aisuite.framework.replay_payload import build_replay_payload
from aisuite.provider import LLMError


class _FakeThinkingConfig:
    model_fields = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeGenerateContentConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeFunctionCall:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeFunctionResponse:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakePart:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @staticmethod
    def from_text(text):
        return _FakePart(text=text)

    @staticmethod
    def from_function_call(name, args):
        return _FakePart(function_call=_FakeFunctionCall(name=name, args=args))

    @staticmethod
    def from_function_response(name, response):
        return _FakePart(function_response=_FakeFunctionResponse(name=name, response=response))


class _FakeContent:
    def __init__(self, role, parts):
        self.role = role
        self.parts = parts


fake_types = SimpleNamespace(
    ThinkingConfig=_FakeThinkingConfig,
    GenerateContentConfig=_FakeGenerateContentConfig,
    FunctionCall=_FakeFunctionCall,
    FunctionResponse=_FakeFunctionResponse,
    Part=_FakePart,
    Content=_FakeContent,
)

fake_google = ModuleType("google")
fake_genai = ModuleType("google.genai")
fake_genai.Client = MagicMock
fake_genai.types = fake_types
fake_google.genai = fake_genai
sys.modules.setdefault("google", fake_google)
sys.modules["google.genai"] = fake_genai

gemini_module = importlib.import_module("aisuite.providers.gemini_provider")
GeminiMessageConverter = gemini_module.GeminiMessageConverter
GeminiProvider = gemini_module.GeminiProvider
_build_tool_call_extra = gemini_module._build_tool_call_extra


@pytest.fixture(autouse=True)
def set_gemini_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")


@patch("aisuite.providers.gemini_provider.genai.Client")
def test_gemini_converter_preserves_provider_replay_metadata(_mock_client_cls):
    function_call = SimpleNamespace(
        id="fc_123",
        name="lookup_weather",
        args={"city": "Paris"},
        thought_signature="sig_abc",
    )
    part = SimpleNamespace(function_call=function_call, text=None)
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[part]),
        finish_reason="STOP",
    )
    response = SimpleNamespace(
        text="",
        candidates=[candidate],
        model_version="gemini-3.1-pro",
        usage_metadata=None,
    )

    converted = GeminiMessageConverter.from_gemini_response(response)
    tool_call = converted.choices[0].message.tool_calls[0]

    assert tool_call.id == "fc_123"
    assert tool_call.extra_content["version"] == 1
    assert tool_call.extra_content["kind"] == "gemini_tool_call"
    assert tool_call.extra_content["payload"]["provider_call_id"] == "fc_123"
    assert tool_call.extra_content["payload"]["provider_function_name"] == "lookup_weather"
    assert tool_call.extra_content["payload"]["thought_signature"] == "sig_abc"


@patch("aisuite.providers.gemini_provider.genai.Client")
def test_gemini_validate_replay_window_errors_on_missing_signature(_mock_client_cls):
    provider = GeminiProvider(api_key="test-gemini-key")
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_local_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                    "extra_content": build_replay_payload(
                        "gemini",
                        "gemini_tool_call",
                        {
                            "provider_call_id": "fc_123",
                            "provider_function_name": "lookup_weather",
                            "replay_mode": "provider_exact_turn",
                            "thought_signature": None,
                        },
                    ),
                }
            ],
        }
    ]

    result = provider.validate_replay_window("gemini-3.1-pro", messages)

    assert result.ok is False
    assert any(diag.code == "missing_thought_signature" for diag in result.diagnostics)


@patch("aisuite.providers.gemini_provider.genai.Client")
def test_gemini_validate_replay_window_degrades_partial_metadata_without_signature(_mock_client_cls):
    provider = GeminiProvider(api_key="test-gemini-key")
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_local_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                    "extra_content": _build_tool_call_extra(
                        None,
                        provider_call_id="fc_123",
                        provider_function_name="lookup_weather",
                    ),
                }
            ],
        }
    ]

    result = provider.validate_replay_window("gemini-3.1-pro", messages)

    assert result.ok is True
    assert result.degraded is True
    assert not any(diag.code == "missing_thought_signature" for diag in result.diagnostics)


@patch("aisuite.providers.gemini_provider.genai.Client")
def test_gemini_validate_replay_window_allows_degraded_legacy_history(_mock_client_cls):
    provider = GeminiProvider(api_key="test-gemini-key")
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "legacy_call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                }
            ],
        }
    ]

    result = provider.validate_replay_window("gemini-3.1-pro", messages)

    assert result.ok is True
    assert result.degraded is True


@patch("aisuite.providers.gemini_provider.genai.Client")
def test_gemini_build_replay_view_returns_structured_result_for_degraded_history(_mock_client_cls):
    provider = GeminiProvider(api_key="test-gemini-key")
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "legacy_call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                }
            ],
        }
    ]

    replay_build = provider.build_replay_view("gemini-3.1-pro", messages)

    assert replay_build.replay_mode == "degraded_legacy_turn"
    assert replay_build.degraded is True
    assert replay_build.request_view == messages


@patch("aisuite.providers.gemini_provider.genai.Client")
def test_gemini_build_tool_parts_uses_placeholder_for_degraded_partial_metadata(_mock_client_cls):
    provider = GeminiProvider(api_key="test-gemini-key")
    messages = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_local_1",
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "arguments": '{"city":"Paris"}',
                },
                "extra_content": _build_tool_call_extra(
                    None,
                    provider_call_id="fc_123",
                    provider_function_name="lookup_weather",
                ),
            }
        ],
    }

    parts = provider._convert_assistant_tool_message_to_parts("gemini-3.1-pro", messages)

    assert parts is not None
    assert len(parts) == 1
    assert getattr(parts[0], "thought_signature", None) is not None


@patch("aisuite.providers.gemini_provider.genai.Client")
def test_gemini_validate_replay_window_allows_degraded_legacy_tool_result_history(_mock_client_cls):
    provider = GeminiProvider(api_key="test-gemini-key")
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "legacy_call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "legacy_call_1",
            "content": "sunny",
        },
    ]

    result = provider.validate_replay_window("gemini-3.1-pro", messages)

    assert result.ok is True
    assert result.degraded is True
    assert not any(diag.severity == "error" for diag in result.diagnostics)


@patch("aisuite.providers.gemini_provider.genai.Client")
def test_gemini_build_tool_response_parts_allows_legacy_tool_result_history(_mock_client_cls):
    provider = GeminiProvider(api_key="test-gemini-key")
    provider._create_function_response_part = MagicMock(return_value="response-part")

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "legacy_call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "legacy_call_1",
            "content": "sunny",
        },
    ]

    parts, diagnostic = provider._build_tool_response_parts(
        "gemini-3.1-pro",
        messages[-1],
        replay_messages=messages,
    )

    assert diagnostic is None
    assert parts == ["response-part"]
    provider._create_function_response_part.assert_called_once_with(
        function_name="lookup_weather",
        content="sunny",
        provider_call_id=None,
    )


@patch("aisuite.providers.gemini_provider.genai.Client")
def test_gemini_build_tool_response_parts_uses_provider_metadata(_mock_client_cls):
    provider = GeminiProvider(api_key="test-gemini-key")
    provider._create_function_response_part = MagicMock(return_value="response-part")

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_local_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                    "extra_content": _build_tool_call_extra(
                        "sig_123",
                        provider_call_id="fc_123",
                        provider_function_name="lookup_weather",
                    ),
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_local_1",
            "content": "sunny",
        },
    ]

    parts, diagnostic = provider._build_tool_response_parts(
        "gemini-3.1-pro",
        messages[-1],
        replay_messages=messages,
    )

    assert diagnostic is None
    assert parts == ["response-part"]
    provider._create_function_response_part.assert_called_once_with(
        function_name="lookup_weather",
        content="sunny",
        provider_call_id="fc_123",
    )


@patch("aisuite.providers.gemini_provider.genai.Client")
def test_gemini_build_replay_view_raises_llm_error_for_exact_validation_failures(_mock_client_cls):
    provider = GeminiProvider(api_key="test-gemini-key")
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_local_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                    "extra_content": build_replay_payload(
                        "gemini",
                        "gemini_tool_call",
                        {
                            "provider_call_id": None,
                            "provider_function_name": "lookup_weather",
                            "replay_mode": "provider_exact_turn",
                            "thought_signature": "sig_123",
                        },
                    ),
                }
            ],
        }
    ]

    with pytest.raises(LLMError, match="missing_provider_call_id"):
        provider.build_replay_view("gemini-3.1-pro", messages)


def test_gemini_stop_reason_maps_protocol_error():
    stop_info = GeminiMessageConverter.from_gemini_response(
        SimpleNamespace(
            text="",
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[]),
                    finish_reason="MISSING_THOUGHT_SIGNATURE",
                )
            ],
            model_version="gemini-3.1-pro",
            usage_metadata=None,
        )
    ).choices[0].stop_info

    assert stop_info.reason == StopReason.PROTOCOL_ERROR
