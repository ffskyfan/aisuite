"""Bounded, content-free diagnostics for OpenAI Responses streams."""

import json
import logging
import time
import uuid

import httpx

from aisuite.framework.stream_activity import ActivityByteStream, StreamActivity

logger = logging.getLogger(__name__)


def _scalar(value):
    if isinstance(value, str):
        return value[:160]
    return value if value is None or isinstance(value, (int, float, bool)) else None


class OpenAIStreamDiagnostics(StreamActivity):
    """Observe traffic without changing the stream's output or timeout policy."""

    def __init__(self, model, input_item_count, options):
        super().__init__()
        reasoning = options.get("reasoning") or {}
        self.details = {
            "call_id": uuid.uuid4().hex,
            "model": _scalar(model),
            "input_item_count": input_item_count,
            "tool_count": len(options.get("tools") or []) if isinstance(options.get("tools", []), (list, tuple)) else None,
            "max_output_tokens": _scalar(options.get("max_output_tokens")),
            "reasoning_effort": _scalar(reasoning.get("effort")) if isinstance(reasoning, dict) else None,
            "requested_service_tier": _scalar(options.get("service_tier")),
            "network_observed": False,
        }
        self.event_counts = {}
        self.first_event_at = None
        self.last_event_at = None
        self.first_progress_at = None
        self._logged = False

    def attach_response(self, stream):
        self.details["response_headers_ms"] = int((time.monotonic() - self.started_at) * 1000)
        response = getattr(stream, "response", None)
        if isinstance(response, httpx.Response):
            self.request_id = _scalar(response.headers.get("x-request-id"))
            self.details["http_status"] = response.status_code
            try:
                self.details["endpoint_host"] = response.url.host
            except RuntimeError:
                pass  # Some custom clients return a response without its request.
            if isinstance(response.stream, httpx.AsyncByteStream) and not response.is_stream_consumed:
                response.stream = ActivityByteStream(response.stream, self)
                self.details["network_observed"] = True

    def record_event(self, event):
        super().record_event(event)
        now = time.monotonic()
        if self.first_event_at is None:
            self.first_event_at = now
        self.last_event_at = now
        event_type = str(getattr(event, "type", "unknown"))[:128]
        counter_key = event_type if event_type in self.event_counts or len(self.event_counts) < 24 else "other"
        self.event_counts[counter_key] = self.event_counts.get(counter_key, 0) + 1
        self.last_event_type = event_type
        if event_type in {"response.completed", "response.done", "response.failed", "response.incomplete", "error"}:
            self.details["terminal_event"] = event_type
        response = getattr(event, "response", None)
        for source, target in (
            ("id", "provider_response_id"),
            ("model", "response_model"),
            ("status", "response_status"),
            ("service_tier", "response_service_tier"),
        ):
            value = getattr(response, source, None)
            if isinstance(value, str):
                self.details[target] = value[:160]
        field = {
            "response.output_text.delta": "text_chars_received",
            "response.function_call_arguments.delta": "tool_input_chars_received",
            "response.reasoning_summary_text.delta": "reasoning_chars_received",
            "response.reasoning_text.delta": "reasoning_chars_received",
        }.get(event_type)
        delta = getattr(event, "delta", None)
        if field and isinstance(delta, str) and delta:
            self.last_progress_at = now
            if self.first_progress_at is None:
                self.first_progress_at = now
            setattr(self, field, getattr(self, field) + len(delta))

    def snapshot(self):
        now = time.monotonic()
        result = {**super().snapshot(), **self.details, "event_counts": dict(self.event_counts)}
        result["elapsed_ms"] = int((now - self.started_at) * 1000)
        for name, timestamp in (
            ("time_to_first_event_ms", self.first_event_at),
            ("time_to_first_model_progress_ms", self.first_progress_at),
        ):
            result[name] = int((timestamp - self.started_at) * 1000) if timestamp is not None else None
        result["since_last_sdk_event_ms"] = int((now - self.last_event_at) * 1000) if self.last_event_at is not None else None
        return result

    def log_summary(self, outcome, error=None):
        if self._logged:
            return
        self._logged = True
        summary = self.snapshot()
        if outcome == "completed":
            if summary.get("response_status") in {"failed", "incomplete", "cancelled"}:
                outcome = summary["response_status"]
            elif summary.get("terminal_event") not in {"response.completed", "response.done"}:
                outcome = "ended_without_completion"
        summary["outcome"] = outcome
        if error is not None:
            # Exception messages and response bodies can contain user content.
            summary["error_type"] = type(error).__name__
            if isinstance(error, httpx.HTTPError):
                summary["transport_error_type"] = type(error).__name__
            status = getattr(error, "status_code", None)
            request_id = getattr(error, "request_id", None)
            if isinstance(status, int):
                summary["http_status"] = status
            if isinstance(request_id, str):
                summary["provider_request_id"] = request_id[:160]
        level = logging.INFO if outcome == "completed" else logging.WARNING
        # Celery's plain formatter drops extra fields; keep this compact JSON in the message.
        logger.log(level, "OpenAI Responses stream summary: %s", json.dumps(summary, ensure_ascii=False))
