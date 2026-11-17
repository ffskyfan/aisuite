import asyncio
import json
import logging
from types import SimpleNamespace

import httpx
import pytest

from aisuite.framework.openai_stream_diagnostics import OpenAIStreamDiagnostics
from aisuite.providers.openai_provider import OpenaiProvider

LOGGER = "aisuite.framework.openai_stream_diagnostics"


def event_bytes(event):
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


class EventBody(httpx.AsyncByteStream):
    def __init__(self, events, *, block=False):
        self.frames = [b": keepalive\n\n", *(event_bytes(event) for event in events)]
        self.block = block
        self.blocked = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        for frame in self.frames:
            yield frame
        if self.block:
            self.blocked.set()
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


def provider_for(body, status=200):
    async def handle(request):
        return httpx.Response(status, headers={"x-request-id": "req_diagnostics"}, stream=body)

    return OpenaiProvider(
        api_key="PRIVATE_API_KEY",
        base_url="https://provider.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )


async def create_stream(provider):
    return await provider._responses_create(
        "gpt-6.1-sol",
        [],
        stream=True,
        _replay_request_view=[{"role": "user", "content": "PRIVATE_PROMPT"}],
        _replay_mode="responses_output",
        reasoning={"effort": "high"},
        max_output_tokens=64000,
    )


def summary_from(caplog):
    records = [record for record in caplog.records if record.name == LOGGER]
    assert len(records) == 1
    # Verify the actual plain-text log Celery/SLS will receive.
    message = logging.Formatter("%(message)s").format(records[0])
    assert "PRIVATE_" not in message
    return json.loads(message.split("summary: ", 1)[1])


CREATED = {
    "type": "response.created",
    "response": {
        "id": "resp_diagnostics",
        "model": "gpt-6.1-sol",
        "status": "in_progress",
    },
}
TOOL_DELTA = {
    "type": "response.function_call_arguments.delta",
    "item_id": "fc_1",
    "output_index": 0,
    "delta": '{"text":"PRIVATE_TOOL_ARGUMENTS"}',
}


@pytest.mark.asyncio
@pytest.mark.parametrize("events", [[], [CREATED], [CREATED, TOOL_DELTA]])
async def test_interrupted_stream_distinguishes_keepalives_events_and_buffered_tools(caplog, events):
    body = EventBody(events, block=True)
    provider = provider_for(body)
    try:
        with caplog.at_level(logging.INFO, logger=LOGGER):
            stream = await create_stream(provider)
            # Diagnostics must not activate the backend's different observed-stream timeouts.
            assert not hasattr(stream, "stream_activity")
            pending = asyncio.create_task(stream.__anext__())
            await asyncio.wait_for(body.blocked.wait(), timeout=2)
            assert not pending.done()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        summary = summary_from(caplog)
        assert summary["outcome"] == "cancelled"
        assert summary["provider_request_id"] == "req_diagnostics"
        assert summary["endpoint_host"] == "provider.test"
        assert summary["http_status"] == 200
        assert summary["network_observed"] is True
        assert summary["raw_bytes_received"] == sum(map(len, body.frames))
        assert summary["sdk_events_received"] == len(events)
        assert (summary["time_to_first_event_ms"] is not None) == bool(events)
        expected_tool_chars = len(TOOL_DELTA["delta"]) if TOOL_DELTA in events else 0
        assert summary["tool_input_chars_received"] == expected_tool_chars
        assert summary["has_model_progress"] == bool(expected_tool_chars)
        assert summary["text_chars_received"] == 0
        assert body.closed
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_success_logs_one_summary_without_prompt_reasoning_or_output(caplog):
    events = [
        CREATED,
        {"type": "response.reasoning_summary_text.delta", "delta": "PRIVATE_REASONING"},
        {"type": "response.output_text.delta", "delta": "PRIVATE_OUTPUT"},
        {
            "type": "response.completed",
            "response": {
                "id": "resp_diagnostics",
                "model": "gpt-6.1-sol",
                "status": "completed",
                "service_tier": "default",
                "output": [],
                "usage": None,
            },
        },
    ]
    body = EventBody(events)
    provider = provider_for(body)
    try:
        with caplog.at_level(logging.INFO, logger=LOGGER):
            stream = await create_stream(provider)
            chunks = [chunk async for chunk in stream]
        summary = summary_from(caplog)
        assert summary["outcome"] == "completed"
        assert summary["reasoning_effort"] == "high"
        assert summary["response_service_tier"] == "default"
        assert summary["provider_response_id"] == "resp_diagnostics"
        assert summary["reasoning_chars_received"] == len("PRIVATE_REASONING")
        assert summary["text_chars_received"] == len("PRIVATE_OUTPUT")
        assert summary["time_to_first_model_progress_ms"] >= 0
        assert chunks[0].choices[0].delta.reasoning_content == "PRIVATE_REASONING"
        assert chunks[1].choices[0].delta.content == "PRIVATE_OUTPUT"
        assert chunks[-1].choices[0].finish_reason == "stop"
        assert body.closed
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_request_error_logs_status_and_request_id_without_error_body(caplog):
    import openai

    body = EventBody([])
    provider = provider_for(body, status=429)
    try:
        with caplog.at_level(logging.INFO, logger=LOGGER):
            with pytest.raises(openai.RateLimitError):
                await create_stream(provider)
        summary = summary_from(caplog)
        assert summary["outcome"] == "request_failed"
        assert summary["http_status"] == 429
        assert summary["provider_request_id"] == "req_diagnostics"
        assert summary["sdk_events_received"] == 0
    finally:
        await provider.aclose()


def test_unknown_event_names_are_bounded_and_unobserved_network_is_explicit(caplog):
    diagnostics = OpenAIStreamDiagnostics("test-model", 1, {})
    for index in range(100):
        diagnostics.record_event(SimpleNamespace(type=f"future.event_{index}", delta="PRIVATE_CONTENT"))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        diagnostics.log_summary("completed")
        diagnostics.log_summary("completed")
    summary = summary_from(caplog)
    assert len(summary["event_counts"]) == 25
    assert sum(summary["event_counts"].values()) == 100
    assert summary["network_observed"] is False
    assert summary["outcome"] == "ended_without_completion"


def test_sdk_omitted_options_and_long_metadata_cannot_break_summary_logging(caplog):
    import openai

    diagnostics = OpenAIStreamDiagnostics(
        "test-model",
        1,
        {
            "tools": openai.NOT_GIVEN,
            "max_output_tokens": openai.NOT_GIVEN,
            "service_tier": openai.NOT_GIVEN,
        },
    )
    # Keep the final event recognizable even when the event histogram is already full.
    for index in range(30):
        diagnostics.record_event(SimpleNamespace(type=f"future.event_{index}"))
    diagnostics.record_event(
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(id="r" * 1000, status="completed"),
        )
    )
    with caplog.at_level(logging.INFO, logger=LOGGER):
        diagnostics.log_summary("completed")
    summary = summary_from(caplog)
    assert summary["outcome"] == "completed"
    assert summary["max_output_tokens"] is None
    assert summary["requested_service_tier"] is None
    assert len(summary["provider_response_id"]) == 160
    assert summary["last_event_type"] == "response.completed"
