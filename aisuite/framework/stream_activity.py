"""Out-of-band stream activity; never expose keepalives as assistant output."""

import time

import httpx


class StreamActivity:
    def __init__(self):
        self.started_at = time.monotonic()
        self.last_network_activity_at = self.started_at
        self.last_progress_at = None
        self.raw_chunks_received = 0
        self.raw_bytes_received = 0
        self.sdk_events_received = 0
        self.last_event_type = None
        self.text_chars_received = 0
        self.reasoning_chars_received = 0
        self.tool_input_chars_received = 0
        self.request_id = None

    def record_bytes(self, data):
        if data:
            self.last_network_activity_at = time.monotonic()
            self.raw_chunks_received += 1
            self.raw_bytes_received += len(data)

    def record_event(self, event):
        self.sdk_events_received += 1
        self.last_event_type = getattr(event, "type", "unknown")
        if self.last_event_type != "content_block_delta":
            return
        delta = getattr(event, "delta", None)
        field = {
            "text_delta": ("text", "text_chars_received"),
            "thinking_delta": ("thinking", "reasoning_chars_received"),
            "input_json_delta": ("partial_json", "tool_input_chars_received"),
        }.get(getattr(delta, "type", None))
        if field is None:
            return
        value = getattr(delta, field[0], None)
        if isinstance(value, str) and value:
            self.last_progress_at = time.monotonic()
            setattr(self, field[1], getattr(self, field[1]) + len(value))

    def snapshot(self):
        now = time.monotonic()
        result = {
            "raw_chunks_received": self.raw_chunks_received,
            "raw_bytes_received": self.raw_bytes_received,
            "sdk_events_received": self.sdk_events_received,
            "last_event_type": self.last_event_type,
            "text_chars_received": self.text_chars_received,
            "reasoning_chars_received": self.reasoning_chars_received,
            "tool_input_chars_received": self.tool_input_chars_received,
            "since_last_network_activity_ms": max(0, int((now - self.last_network_activity_at) * 1000)),
            "has_model_progress": self.last_progress_at is not None,
        }
        if self.last_progress_at is not None:
            result["since_last_model_progress_ms"] = max(0, int((now - self.last_progress_at) * 1000))
        if self.request_id:
            result["provider_request_id"] = self.request_id
        return result


class ActivityByteStream(httpx.AsyncByteStream):
    """Observe HTTP bytes before the SDK filters SSE ping/status events."""

    def __init__(self, stream, activity):
        self.stream = stream
        self.activity = activity

    async def __aiter__(self):
        async for data in self.stream:
            self.activity.record_bytes(data)
            yield data

    async def aclose(self):
        await self.stream.aclose()


class ObservedAsyncStream:
    def __init__(self, iterator, activity, close):
        self.iterator = iterator
        self.stream_activity = activity
        self._close = close
        self._closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._closed:
            raise StopAsyncIteration
        try:
            return await self.iterator.__anext__()
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self):
        if self._closed:
            return
        self._closed = True
        try:
            await self.iterator.aclose()
        finally:
            # Also close streams that were opened but never iterated.
            await self._close()
