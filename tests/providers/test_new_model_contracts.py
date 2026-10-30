import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aisuite.providers.anthropic_provider import AnthropicProvider
from aisuite.providers.glm_provider import GlmProvider
from aisuite.providers.kimi_provider import KimiProvider


@pytest.mark.parametrize("model", ["glm-5.3", "glm-5.3-flash"])
@pytest.mark.parametrize("effort", ["low", "high", "max"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_glm_5_3_native_effort_reaches_the_request_body(model, effort, stream):
    provider = GlmProvider(api_key="offline-test")
    bodies = []
    response = SimpleNamespace(
        id="test",
        created=1,
        model=model,
        choices=[],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    provider._async_post_json = AsyncMock(return_value=response)

    async def stream_json(body):
        bodies.append(body)
        yield response

    provider._async_stream_json = stream_json
    result = await provider.chat_completions_create(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        reasoning_effort=effort,
        stream=stream,
        max_tokens=64_000,
    )
    if stream:
        _ = [chunk async for chunk in result]
        assert bodies[0]["tool_stream"] is True
    else:
        bodies.append(provider._async_post_json.call_args.args[0])
    assert bodies[0]["model"] == model
    assert bodies[0]["thinking"] == {"type": "enabled"}
    assert bodies[0]["reasoning_effort"] == effort
    assert bodies[0]["max_tokens"] == 64_000


@pytest.mark.parametrize(
    "kwargs,expected", [({}, "max"), ({"thinking": {"type": "disabled"}}, "low")]
)
def test_glm_5_3_never_sends_disabled_thinking(kwargs, expected):
    provider = object.__new__(GlmProvider)
    result = provider._prepare_kwargs(kwargs, model="glm-5.3")
    assert result["thinking"]["type"] == "enabled"
    assert result["reasoning_effort"] == expected


@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_kimi_3_1_keeps_native_effort_and_fixed_sampling_contract(effort):
    provider = object.__new__(KimiProvider)
    result = provider._prepare_request_kwargs(
        {
            "reasoning_effort": effort,
            "temperature": 0.7,
            "top_p": 0.8,
            "thinking_config": {},
            "max_tokens": 64_000,
            "prompt_cache_options": {"mode": "implicit", "ttl": "1h"},
        },
        model="kimi-k3.1-pro",
    )
    assert result == {
        "reasoning_effort": effort,
        "max_completion_tokens": 64_000,
        "extra_body": {"prompt_cache_options": {"mode": "implicit", "ttl": "1h"}},
    }


@pytest.mark.parametrize(
    "provider_type,model",
    [(KimiProvider, "kimi-k3.1-pro"), (GlmProvider, "glm-5.3-flash")],
)
def test_new_vision_models_preserve_parallel_tool_results_reasoning_and_images(
    provider_type, model
):
    provider = object.__new__(provider_type)
    reasoning = {
        "thinking": "inspect frames",
        "raw_data": provider._build_reasoning_replay_payload("inspect frames"),
    }
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2nN8AAAAASUVORK5CYII="}}
    messages = [
        {"role": "user", "content": "Compare previews"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": reasoning,
            "tool_calls": [
                {
                    "id": key,
                    "type": "function",
                    "function": {"name": "preview", "arguments": "{}"},
                }
                for key in ("one", "two")
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "one",
            "content": [{"type": "text", "text": "first"}, image],
        },
        {"role": "tool", "tool_call_id": "two", "content": "second"},
    ]
    original = copy.deepcopy(messages)
    assert provider.validate_replay_window(model, messages).ok is True
    built = provider.build_replay_view(model, messages).request_view
    assert [m["role"] for m in built] == ["user", "assistant", "tool", "tool", "user"]
    assert built[1]["reasoning_content"] == "inspect frames"
    assert built[2]["content"] == "first"
    assert built[2]["tool_call_id"] == "one"
    assert built[3]["tool_call_id"] == "two"
    assert image in built[4]["content"]
    assert messages == original
    assert provider.get_multimodal_capabilities(model).tool_result_images == "supported"


def test_glm_5_3_text_model_does_not_advertise_vision():
    capabilities = object.__new__(GlmProvider).get_multimodal_capabilities("glm-5.3")
    assert capabilities.user_images == "unsupported"
    assert capabilities.tool_result_images == "unsupported"


@pytest.mark.parametrize("model", ["anthropic/claude-sonnet-5.5", "claude-sonnet-5-5"])
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_sonnet_5_5_preserves_effort_and_requests_reasoning_summaries(model, effort):
    provider = object.__new__(AnthropicProvider)
    original = {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": effort},
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 10,
    }
    snapshot = copy.deepcopy(original)
    prepared = provider._prepare_kwargs(original, model=model)
    assert prepared["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert prepared["output_config"] == {"effort": effort}
    assert not {"temperature", "top_p", "top_k"} & prepared.keys()
    assert original == snapshot


@pytest.mark.parametrize(
    "tool_choice", [{"type": "any"}, {"type": "tool", "name": "preview"}, "required"]
)
def test_sonnet_5_5_rejects_unsupported_forced_tools_before_calling_api(tool_choice):
    provider = object.__new__(AnthropicProvider)
    with pytest.raises(ValueError, match="auto/none"):
        provider._prepare_kwargs(
            {"tool_choice": tool_choice}, model="anthropic/claude-sonnet-5.5"
        )


def test_sonnet_5_5_legacy_disabled_uses_adaptive_low():
    prepared = object.__new__(AnthropicProvider)._prepare_kwargs(
        {"thinking": {"type": "disabled"}}, model="claude-sonnet-5-5"
    )
    assert prepared["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert prepared["output_config"]["effort"] == "low"
