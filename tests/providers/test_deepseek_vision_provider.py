import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aisuite import Client
from aisuite.framework.content import (
    IMAGE_OMITTED_TEXT,
    filter_images_for_capabilities,
    make_image_data_url,
)
from aisuite.framework.message import Message, ReasoningContent
from aisuite.providers.deepseek_provider import DeepseekProvider
from aisuite.providers.qwen_provider import QwenProvider

MODEL = "deepseek-flash"
IMAGE = {
    "type": "image_url",
    "image_url": {
        "url": make_image_data_url("image/png", b"synthetic-frame"),
        "detail": "high",
    },
}


def _tool_call(call_id):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "capture_frame", "arguments": "{}"},
    }


def _history():
    return [
        {"role": "user", "content": "Inspect the game frame and diagnostics."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": ReasoningContent(
                thinking="Inspect before making changes.", provider="deepseek"
            ),
            "tool_calls": [
                _tool_call("frame-1"),
                _tool_call("frame-2"),
                _tool_call("state"),
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "frame-1",
            "is_error": False,
            "content": [{"type": "text", "text": "First frame."}, copy.deepcopy(IMAGE)],
        },
        {"role": "tool", "tool_call_id": "frame-2", "content": [copy.deepcopy(IMAGE)]},
        {
            "role": "tool",
            "tool_call_id": "state",
            "content": [{"type": "text", "text": "score=0"}],
        },
    ]


@pytest.mark.parametrize("model", [
    MODEL, f"deepseek:{MODEL}", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp"
])
def test_current_flash_and_its_provider_aliases_support_both_image_roles(model):
    provider = DeepseekProvider.__new__(DeepseekProvider)
    capabilities = provider.get_multimodal_capabilities(model)

    assert capabilities.user_images == "supported"
    assert capabilities.tool_result_images == "supported"
    messages = [{"role": "user", "content": [copy.deepcopy(IMAGE)]}, *_history()]
    filtered, removed = filter_images_for_capabilities(messages, capabilities)
    assert removed == 0
    assert filtered == messages


@pytest.mark.parametrize(
    "model",
    [
        None,
        "deepseek-chat",
        "deepseek-reasoner",
        "deepseek-v4-pro",
        "future-vision",
    ],
)
def test_text_models_keep_explicit_image_downgrade(model):
    provider = DeepseekProvider.__new__(DeepseekProvider)
    capabilities = provider.get_multimodal_capabilities(model)
    messages = [{"role": "user", "content": [copy.deepcopy(IMAGE)]}]
    original = copy.deepcopy(messages)

    filtered, removed = filter_images_for_capabilities(messages, capabilities)

    assert capabilities.user_images == "unsupported"
    assert capabilities.tool_result_images == "unsupported"
    assert removed == 1
    assert filtered[0]["content"] == IMAGE_OMITTED_TEXT
    assert messages == original


@pytest.mark.parametrize("provider_class", [DeepseekProvider, QwenProvider])
def test_projection_preserves_parallel_tool_group_and_canonical_history(provider_class):
    provider = provider_class.__new__(provider_class)
    messages = _history()
    messages.extend(
        [
            {
                "role": "assistant",
                "content": "The frames are visible.",
                "reasoning_content": "The playfield is centered.",
            },
            {"role": "user", "content": "Now continue."},
        ]
    )
    original = copy.deepcopy(messages)

    replay = provider.build_replay_view(
        MODEL if provider_class is DeepseekProvider else "qwen3.8-max",
        messages,
        thinking={"type": "enabled"},
    )
    wire = replay.request_view

    assert [m["role"] for m in wire] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "tool",
        "user",
        "assistant",
        "user",
    ]
    assert wire[1]["reasoning_content"] == "Inspect before making changes."
    assert wire[2]["content"] == "First frame."
    assert "is_error" not in wire[2]
    assert wire[3]["content"] == provider.TOOL_IMAGE_FORWARD_TEXT
    assert wire[4]["content"] == "score=0"
    assert [m["tool_call_id"] for m in wire[2:5]] == ["frame-1", "frame-2", "state"]
    assert "frame-1" in wire[5]["content"][0]["text"]
    assert "frame-2" in wire[5]["content"][2]["text"]
    assert wire[5]["content"][1] == IMAGE
    assert wire[5]["content"][3] == IMAGE
    assert wire[6]["reasoning_content"] == "The playfield is centered."
    assert messages == original
    assert provider._project_tool_result_images(wire) == wire  # Idempotent replay.
    wire[5]["content"][1]["image_url"]["url"] = "changed"
    assert messages == original  # Nested image blocks are detached too.


def test_projection_flushes_trailing_group_and_accepts_message_objects():
    provider = DeepseekProvider.__new__(DeepseekProvider)
    messages = _history()
    messages[2] = Message(**messages[2])

    replay = provider.build_replay_view(MODEL, messages)

    assert len(replay.request_view) == 6
    assert replay.request_view[-1]["role"] == "user"
    assert (
        replay.request_view[1]["reasoning_content"] == "Inspect before making changes."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("persist_history", [False, True])
async def test_client_sends_user_images_and_replays_tool_screenshots(persist_history):
    client = Client(provider_configs={"deepseek": {"api_key": "test-key"}})
    provider = client.providers["deepseek"]
    transport = AsyncMock(
        return_value=SimpleNamespace(
            id="vision-test",
            created=1,
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason="stop",
                    message=SimpleNamespace(
                        role="assistant",
                        content="Frame inspected.",
                        reasoning_content="The screenshot matches the diagnostics.",
                        tool_calls=None,
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_cache_hit_tokens=20,
                prompt_cache_miss_tokens=30,
                completion_tokens=10,
                total_tokens=60,
            ),
        )
    )
    provider.client.chat.completions.create = transport
    messages = _history()
    messages[0]["content"] = [
        {"type": "text", "text": "Inspect."},
        copy.deepcopy(IMAGE),
    ]
    if persist_history:
        messages = [
            json.loads(Message(**message).model_dump_json(exclude_none=True))
            for message in messages
        ]
    original = copy.deepcopy(messages)
    try:
        response = await client.chat.completions.create(
            model=f"deepseek:{MODEL}",
            messages=messages,
            thinking={"type": "enabled"},
            reasoning_effort="low",
        )
    finally:
        await client.aclose()

    request = transport.await_args.kwargs
    assert request["model"] == MODEL
    assert request["extra_body"] == {"thinking": {"type": "enabled"}}
    assert request["reasoning_effort"] == "low"
    assert request["messages"][0]["content"][1] == IMAGE
    assert [m["role"] for m in request["messages"]] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "tool",
        "user",
    ]
    assert (
        request["messages"][1]["reasoning_content"] == "Inspect before making changes."
    )
    assert request["messages"][-1]["content"][1] == IMAGE
    assert response.choices[0].message.content == "Frame inspected."
    assert response.metadata["usage"]["prompt_tokens"] == 50
    assert response.metadata["usage"]["cache_read_input_tokens"] == 20
    assert messages == original


@pytest.mark.asyncio
async def test_streaming_vision_request_keeps_images_and_trailing_usage():
    client = Client(provider_configs={"deepseek": {"api_key": "test-key"}})
    provider = client.providers["deepseek"]
    usage = SimpleNamespace(prompt_tokens=50, completion_tokens=10, total_tokens=60)

    async def stream_chunks():
        yield SimpleNamespace(
            id="vision-stream",
            created=1,
            model=MODEL,
            usage=None,
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason="stop",
                    delta=SimpleNamespace(
                        role="assistant",
                        content="Frame inspected.",
                        tool_calls=None,
                        reasoning_content="The frame matches the diagnostics.",
                    ),
                )
            ],
        )
        yield SimpleNamespace(
            id="vision-stream", created=1, model=MODEL, usage=usage, choices=[]
        )

    transport = AsyncMock(return_value=stream_chunks())
    provider.client.chat.completions.create = transport
    messages = _history()
    original = copy.deepcopy(messages)
    try:
        stream = await client.chat.completions.create(
            model=f"deepseek:{MODEL}",
            messages=messages,
            stream=True,
            thinking={"type": "enabled"},
            reasoning_effort="low",
        )
        chunks = [chunk async for chunk in stream]
    finally:
        await client.aclose()

    request = transport.await_args.kwargs
    assert request["stream_options"]["include_usage"] is True
    assert [m["role"] for m in request["messages"]] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "tool",
        "user",
    ]
    assert request["messages"][-1]["content"][1] == IMAGE
    assert chunks[0].choices[0].delta.content == "Frame inspected."
    assert chunks[-1].choices == []
    assert chunks[-1].metadata["usage"]["total_tokens"] == 60
    assert messages == original
