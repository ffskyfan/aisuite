from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from aisuite.framework.message_normalizer import MessageNormalizer
from aisuite.providers.glm_provider import GlmProvider


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def test_glm_provider_non_stream(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")

    with patch("aisuite.providers.glm_provider.ZhipuAiClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

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
        mock_client.chat.completions.create.return_value = mock_response

        response = provider.chat_completions_create(
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

        mock_client.chat.completions.create.assert_called_with(
            model="glm-5",
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "previous reasoning trace",
                },
                {"role": "user", "content": "Hello!"},
            ],
            stream=False,
            temperature=0.2,
            thinking={"type": "enabled"},
        )

        assert response.choices[0].message.content == "mocked-text-response-from-model"
        assert response.choices[0].message.reasoning_content.thinking == "step-by-step"
        assert (
            response.choices[0].message.tool_calls[0].function.name == "lookup_weather"
        )
        assert response.metadata["usage"]["cache_read_input_tokens"] == 3


def test_glm_provider_defaults_thinking_to_disabled(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")

    with patch("aisuite.providers.glm_provider.ZhipuAiClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

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
        mock_client.chat.completions.create.return_value = mock_response

        provider.chat_completions_create(
            model="glm-5",
            messages=[{"role": "user", "content": "Hello!"}],
        )

        mock_client.chat.completions.create.assert_called_with(
            model="glm-5",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=False,
            thinking={"type": "disabled"},
        )


def test_glm_provider_streaming(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")

    with patch("aisuite.providers.glm_provider.ZhipuAiClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

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
                        finish_reason=None,
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
        mock_client.chat.completions.create.return_value = iter(chunks)

        stream = provider.chat_completions_create(
            model="glm-5",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        )
        streamed_chunks = list(stream)

        assert streamed_chunks[0].choices[0].delta.content == "Hel"
        assert streamed_chunks[0].choices[0].delta.reasoning_content == "think-1"
        assert streamed_chunks[-1].choices[0].finish_reason == "stop"
        assert streamed_chunks[-1].metadata["usage"]["total_tokens"] == 15
        assert streamed_chunks[-1].choices[0].stop_info.reason.value == "complete"


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
