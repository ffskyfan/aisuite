from unittest.mock import AsyncMock

import pytest

from aisuite.providers.openai_provider import OpenaiProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
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
