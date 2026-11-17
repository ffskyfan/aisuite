"""Exercise Gemini requests through the real SDK with an offline HTTP transport."""

import base64
import copy
import json

import httpx
import pytest
from google import genai
from google.genai import types

from aisuite.providers.gemini_provider import GeminiProvider


# Older supported SDKs forward new enum strings with a warning. Assert the
# serialized HTTP payload below, including xhigh, instead of mocking the enum.
pytestmark = pytest.mark.filterwarnings("ignore:xhigh is not a valid ThinkingLevel:UserWarning")

NEW_MODELS = ["gemini-4-hydrogen", "gemini-4-carbon"]


@pytest.fixture
def offline_gemini():
    clients = []

    def create(model, response_parts=None, *, split_parts=False):
        requests = []
        payload = {
            "candidates": [{
                "content": {"role": "model", "parts": response_parts or [{"text": "ready"}]},
                "finishReason": "STOP",
            }],
            "modelVersion": model,
            "usageMetadata": {
                "promptTokenCount": 1000,
                "cachedContentTokenCount": 200,
                "candidatesTokenCount": 200,
                "thoughtsTokenCount": 100,
                "totalTokenCount": 1300,
            },
        }

        def handle(request):
            requests.append(request)
            if "streamGenerateContent" in request.url.path:
                chunks = [payload]
                if split_parts:
                    chunks = []
                    for index, part in enumerate(payload["candidates"][0]["content"]["parts"]):
                        chunk = copy.deepcopy(payload)
                        chunk["candidates"][0]["content"]["parts"] = [part]
                        if index < len(payload["candidates"][0]["content"]["parts"]) - 1:
                            chunk["candidates"][0].pop("finishReason")
                            chunk.pop("usageMetadata")
                        chunks.append(chunk)
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks).encode(),
                )
            return httpx.Response(200, json=payload)

        client = genai.Client(
            api_key="offline-test",
            http_options=types.HttpOptions(
                client_args={"transport": httpx.MockTransport(handle), "trust_env": False}
            ),
        )
        clients.append(client)
        provider = object.__new__(GeminiProvider)
        provider.client = client
        provider._streaming_tool_calls = {}
        provider._stream_content_length = 0
        provider._stream_tool_calls_count = 0
        return provider, requests

    yield create
    for client in clients:
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", NEW_MODELS)
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
@pytest.mark.parametrize("entry", ["thinking_config", "thinking_level", "reasoning", "reasoning_effort"])
@pytest.mark.parametrize("stream", [False, True])
async def test_native_thinking_and_sampling_reach_http(offline_gemini, model, effort, entry, stream):
    provider, requests = offline_gemini(model)
    value = effort
    if entry == "thinking_config":
        value = {"thinkingLevel": effort}
    elif entry == "reasoning":
        value = {"effort": effort}
    result = await provider.chat_completions_create(
        model,
        [{"role": "user", "content": "hello"}],
        **{entry: value},
        stream=stream,
        max_tokens=64_000,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        candidate_count=2,
    )
    if stream:
        result = [chunk async for chunk in result][-1]
    body = json.loads(requests[0].content)
    expected = effort
    if effort in {"xhigh", "max"}:
        expected = "xhigh" if model == "gemini-4-hydrogen" else "high"
    config = body["generationConfig"]
    thinking = config["thinkingConfig"]
    assert thinking.get("thinkingLevel", thinking.get("thinking_level")).lower() == expected
    assert thinking.get("includeThoughts", thinking.get("include_thoughts")) is True
    assert config["maxOutputTokens"] == 64_000
    assert not {"temperature", "topP", "topK", "candidateCount"} & config.keys()
    assert model in requests[0].url.path
    assert len(requests) == 1
    usage = result.metadata["usage"]
    assert usage["completion_tokens"] == 300  # visible output plus billed thinking
    assert usage["reasoning_tokens"] == 100
    assert usage["cache_read_input_tokens"] == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("model", NEW_MODELS)
async def test_sdk_thinking_config_keeps_precedence_and_prefill_uses_continuation(offline_gemini, model):
    provider, requests = offline_gemini(model)
    config = types.ThinkingConfig(thinking_level="medium", include_thoughts=False)
    await provider.chat_completions_create(
        model,
        [
            {"role": "user", "content": "Write a scene."},
            {"role": "assistant", "content": "The door opened"},
        ],
        thinking_config=config,
        reasoning_effort="max",
        thinking_level="high",
    )
    body = json.loads(requests[0].content)
    thinking = body["generationConfig"]["thinkingConfig"]
    assert thinking.get("thinkingLevel", thinking.get("thinking_level")).lower() == "medium"
    assert thinking.get("includeThoughts", thinking.get("include_thoughts")) is False
    assert [message["role"] for message in body["contents"]] == ["user", "model", "user"]
    assert body["contents"][-1]["parts"] == [{"text": "Continue."}]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", NEW_MODELS)
@pytest.mark.parametrize("stream,split_parts", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("after_tool", [False, True])
@pytest.mark.parametrize("signature_mode", ["first_only", "each"])
async def test_parallel_tool_calls_keep_each_signature_and_image_result(
    offline_gemini, model, stream, split_parts, after_tool, signature_mode
):
    signatures = [base64.b64encode(value).decode() for value in (b"signature-one", b"signature-two")]
    parts = [
        {
            "functionCall": {"id": call_id, "name": "preview", "args": {"index": index}},
            "thoughtSignature": signature,
        }
        for index, (call_id, signature) in enumerate(zip(("one", "two"), signatures))
    ]
    if signature_mode == "first_only":
        parts[1].pop("thoughtSignature")
        signatures[1] = None
    provider, requests = offline_gemini(model, parts, split_parts=split_parts)
    messages = [{"role": "user", "content": "Compare previews"}]
    if after_tool:
        # Exercise generate_content[_stream] as well as the chat send path.
        messages.extend([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "previous",
                    "type": "function",
                    "function": {"name": "preview", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "previous", "content": "old preview"},
        ])
    result = await provider.chat_completions_create(model, messages, stream=stream)
    if stream:
        chunks = [chunk async for chunk in result]
        tool_calls = [call for chunk in chunks for call in chunk.choices[0].delta.tool_calls or []]
    else:
        tool_calls = result.choices[0].message.tool_calls
    assert [call.id for call in tool_calls] == ["one", "two"]

    assistant = {"role": "assistant", "content": "", "tool_calls": tool_calls}
    validation = provider.validate_replay_window(model, [assistant])
    assert validation.ok is True
    assert validation.degraded is False
    messages.append(assistant)
    image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2nN8AAAAASUVORK5CYII="
    messages.extend([
        {
            "role": "tool",
            "tool_call_id": "one",
            "content": [
                {"type": "text", "text": "first preview"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "two", "content": "second preview"},
    ])
    original = copy.deepcopy(messages)
    result = await provider.chat_completions_create(model, messages, stream=stream)
    if stream:
        _ = [chunk async for chunk in result]
    body = json.loads(requests[-1].content)
    replay_parts = [
        part for message in body["contents"] for part in message["parts"]
        if part.get("functionCall", {}).get("id") in {"one", "two"}
    ]
    assert [part.get("thoughtSignature") for part in replay_parts] == signatures
    responses = [
        part["functionResponse"] for message in body["contents"] for part in message["parts"]
        if part.get("functionResponse", {}).get("id") in {"one", "two"}
    ]
    assert [response["id"] for response in responses] == ["one", "two"]
    image_part = responses[0]["parts"][0]
    encoded_image = image_part.get("inlineData", image_part.get("inline_data"))["data"]
    assert base64.urlsafe_b64decode(encoded_image) == base64.b64decode(image_data)
    assert messages == original
