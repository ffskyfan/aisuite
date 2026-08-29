from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aisuite.providers.openai_provider import OpenaiProvider


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "details", "expected"),
    [
        ("incomplete", SimpleNamespace(reason="max_output_tokens"), "length"),
        ("incomplete", {"reason": "max_output_tokens"}, "length"),
        ("incomplete", SimpleNamespace(reason="content_filter"), "content_filter"),
        ("incomplete", None, "incomplete"),
        ("failed", None, "error"),
        ("cancelled", None, "error"),
        ("completed", None, "stop"),
        (None, None, "stop"),
    ],
)
async def test_non_streaming_responses_preserve_abnormal_finish_reason(status, details, expected):
    provider = OpenaiProvider.__new__(OpenaiProvider)
    raw_response = SimpleNamespace(
        status=status,
        incomplete_details=details,
        output_text="# State\n- Partial or complete memory.",
        output=[],
        usage=None,
    )
    provider.client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=raw_response)))
    response = await provider._responses_create(
        "test-model",
        [],
        stream=False,
        _replay_request_view=[],
        _replay_mode="responses_output",
    )
    assert response.choices[0].finish_reason == expected
    assert response.choices[0].message.content == raw_response.output_text
    provider.client.responses.create.assert_awaited_once()
