import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai.types.responses import ResponseFunctionToolCall, ResponseReasoningItem

from aisuite.providers.openai_provider import OpenaiProvider


ALWAYS_REASONING_MODELS = ("gpt-6.1-sol", "gpt-6.2-sol", "gpt-6.2-astra")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ALWAYS_REASONING_MODELS + ("gpt-6-sol", "gpt-6.1-terra", "gpt-6-luna"))
async def test_gpt6_routes_through_responses_with_reasoning_and_tools(model):
    provider = object.__new__(OpenaiProvider)
    provider._responses_create = AsyncMock(return_value="responses-result")
    messages = [{"role": "user", "content": "hello"}]

    result = await provider.chat_completions_create(
        model=model,
        messages=messages,
        reasoning={"effort": "high"},
        max_tokens=2_048,
        tools=[{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
        temperature=0.7,
        top_p=0.8,
    )

    assert result == "responses-result"
    provider._responses_create.assert_awaited_once()
    args, kwargs = provider._responses_create.await_args
    assert args == (model, messages)
    assert kwargs["reasoning"] == {"effort": "high"}
    assert kwargs["max_completion_tokens"] == 2_048
    assert kwargs["tools"][0]["function"]["name"] == "lookup"
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert provider.get_replay_capabilities(model).needs_reasoning_raw_replay is True


def test_gpt6_none_effort_keeps_supported_sampling_parameters():
    provider = object.__new__(OpenaiProvider)

    prepared = provider._prepare_reasoning_kwargs(
        "openai/gpt-6-luna",
        {"reasoning": {"effort": "none"}, "temperature": 0.7, "top_p": 0.8},
    )

    assert prepared["reasoning"] == {"effort": "none"}
    assert prepared["temperature"] == 0.7
    assert prepared["top_p"] == 0.8
    assert provider._should_use_responses_api("openai/gpt-6-luna", prepared) is True


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
@pytest.mark.parametrize("model", ALWAYS_REASONING_MODELS)
def test_gpt6_preserves_native_efforts_and_filters_logprobs(model, effort):
    provider = object.__new__(OpenaiProvider)

    prepared = provider._prepare_reasoning_kwargs(
        model,
        {
            "reasoning": {"effort": effort},
            "include": ["reasoning.encrypted_content", "message.output_text.logprobs"],
            "temperature": 0.7,
            "top_p": 0.8,
            "top_logprobs": 3,
            "logprobs": True,
        },
    )

    assert prepared["reasoning"] == {"effort": effort}
    assert prepared["include"] == ["reasoning.encrypted_content"]
    for field in ("temperature", "top_p", "top_logprobs", "logprobs"):
        assert field not in prepared


@pytest.mark.parametrize("effort", ["none", "minimal"])
@pytest.mark.parametrize("model", ALWAYS_REASONING_MODELS)
def test_gpt6_unsupported_efforts_use_low_and_preserve_caller_config(model, effort):
    provider = object.__new__(OpenaiProvider)
    reasoning = {"effort": effort, "summary": "detailed"}

    prepared = provider._prepare_reasoning_kwargs(
        f"openai/{model}",
        {"reasoning": reasoning, "temperature": 0.7, "top_p": 0.8, "include": None},
    )

    assert prepared["reasoning"] == {"effort": "low", "summary": "detailed"}
    assert reasoning["effort"] == effort
    assert "temperature" not in prepared
    assert "top_p" not in prepared


@pytest.mark.parametrize("kwargs,expected", [
    ({"reasoning_effort": "none"}, {"effort": "low"}),
    ({"reasoning": "max"}, {"effort": "max"}),
    (
        {"reasoning_effort": "low", "reasoning": {"effort": "high", "summary": "auto"}},
        {"effort": "high", "summary": "auto"},
    ),
])
def test_gpt6_translates_chat_style_efforts_to_responses(kwargs, expected):
    provider = object.__new__(OpenaiProvider)
    assert provider._prepare_reasoning_kwargs("gpt-6.2-sol", kwargs) == {"reasoning": expected}


@pytest.mark.asyncio
async def test_non_streaming_responses_replay_encrypted_reasoning_and_bind_parallel_tool_results():
    provider = object.__new__(OpenaiProvider)
    output = [
        {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"},
        {"type": "function_call", "id": "fc_a", "call_id": "call_a", "name": "lookup", "arguments": '{"id":1}'},
        {"type": "function_call", "id": "fc_b", "call_id": "call_b", "name": "lookup", "arguments": '{"id":2}'},
    ]
    sdk_output = [ResponseReasoningItem(**output[0]), *(ResponseFunctionToolCall(**item) for item in output[1:])]
    create = AsyncMock(return_value=SimpleNamespace(id="resp_1", status="completed", output=sdk_output))
    provider.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    messages = [{"role": "user", "content": "Look up both items."}]
    response = await provider.chat_completions_create("gpt-6.2-astra", messages)
    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert json.loads(json.dumps(choice.message.reasoning_content.raw_data))["output"] == output
    assert [(tc.id, tc.function.name, tc.function.arguments) for tc in choice.message.tool_calls] == [
        ("call_a", "lookup", '{"id":1}'), ("call_b", "lookup", '{"id":2}'),
    ]
    messages.append(choice.message.model_dump(exclude_none=True))
    messages.extend([
        {"role": "tool", "tool_call_id": "call_a", "content": "red"},
        {"role": "tool", "tool_call_id": "call_b", "content": "blue"},
    ])
    await provider.chat_completions_create("gpt-6.2-astra", messages)
    assert create.await_args.kwargs["input"][1:] == [
        *output,
        {"type": "function_call_output", "call_id": "call_a", "output": "red"},
        {"type": "function_call_output", "call_id": "call_b", "output": "blue"},
    ]


@pytest.mark.asyncio
async def test_incomplete_responses_do_not_dispatch_partial_tool_arguments():
    provider = object.__new__(OpenaiProvider)
    partial = SimpleNamespace(
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
        output=[{"type": "function_call", "call_id": "call_a", "name": "lookup", "arguments": '{"id":'}],
    )
    create = AsyncMock(return_value=partial)
    provider.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    response = await provider.chat_completions_create("gpt-6.2-sol", [{"role": "user", "content": "Look up an item."}])
    assert response.choices[0].finish_reason == "length"
    assert response.choices[0].message.tool_calls is None
