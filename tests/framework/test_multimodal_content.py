import pytest
from types import SimpleNamespace

from aisuite.framework.content import (
    IMAGE_OMITTED_TEXT,
    MultimodalCapabilities,
    MultimodalContentError,
    filter_images_for_capabilities,
    make_image_data_url,
    parse_image_data_url,
    to_anthropic_content,
    to_openai_responses_content,
    validate_message_content,
)
from aisuite.client import Completions


PNG_BYTES = b"\x89PNG\r\n\x1a\n"
PNG_URL = make_image_data_url("image/png", PNG_BYTES)


def image_part(detail="auto"):
    return {
        "type": "image_url",
        "image_url": {"url": PNG_URL, "detail": detail},
    }


def test_data_url_roundtrip_and_validation():
    assert parse_image_data_url(PNG_URL) == ("image/png", PNG_BYTES)
    assert validate_message_content(
        [{"type": "text", "text": "frame"}, image_part("high")],
        max_images=1,
        require_data_urls=True,
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/frame.png",
        "data:image/gif;base64,R0lGODlh",
        "data:image/png;base64,not-base64!",
    ],
)
def test_invalid_or_unsupported_data_url_is_rejected(url):
    with pytest.raises(MultimodalContentError):
        parse_image_data_url(url)


def test_unsupported_model_drops_images_but_preserves_tool_pairing():
    messages = [
        {"role": "user", "content": [image_part()]},
        {
            "role": "tool",
            "tool_call_id": "call_capture",
            "content": [
                {"type": "text", "text": "capture metadata"},
                image_part(),
            ],
        },
    ]

    filtered, removed = filter_images_for_capabilities(
        messages,
        MultimodalCapabilities(
            user_images="unsupported",
            tool_result_images="unsupported",
        ),
    )

    assert removed == 2
    assert filtered[0]["content"] == IMAGE_OMITTED_TEXT
    assert filtered[1]["tool_call_id"] == "call_capture"
    assert filtered[1]["content"] == "capture metadata"


def test_unknown_capability_keeps_images_for_optimistic_provider_attempt():
    messages = [{"role": "user", "content": [image_part()]}]
    filtered, removed = filter_images_for_capabilities(
        messages,
        MultimodalCapabilities(user_images="unknown"),
    )
    assert removed == 0
    assert filtered == messages


def test_supported_or_unknown_capability_validates_data_url_before_provider():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,invalid!"},
                }
            ],
        }
    ]
    with pytest.raises(MultimodalContentError):
        filter_images_for_capabilities(
            messages,
            MultimodalCapabilities(user_images="unknown"),
        )


def test_openai_responses_conversion_preserves_text_image_and_detail():
    assert to_openai_responses_content(
        [{"type": "text", "text": "inspect"}, image_part("low")]
    ) == [
        {"type": "input_text", "text": "inspect"},
        {"type": "input_image", "image_url": PNG_URL, "detail": "low"},
    ]


def test_anthropic_conversion_decodes_data_url():
    converted = to_anthropic_content(
        [{"type": "text", "text": "inspect"}, image_part()]
    )
    assert converted[0] == {"type": "text", "text": "inspect"}
    assert converted[1] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "iVBORw0KGgo=",
        },
    }


def test_client_prepares_provider_specific_copy_without_mutating_history():
    class TextOnlyProvider:
        def get_multimodal_capabilities(self, _model):
            return MultimodalCapabilities(
                user_images="unsupported",
                tool_result_images="unsupported",
            )

    original = [{"role": "user", "content": [image_part()]}]
    prepared = Completions._prepare_messages_for_provider(
        TextOnlyProvider(), "text-only", original
    )

    assert prepared[0]["content"] == IMAGE_OMITTED_TEXT
    assert isinstance(original[0]["content"], list)


def test_client_keeps_legacy_provider_without_capability_method_compatible():
    class LegacyProvider:
        pass

    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_capture",
            "content": [
                {"type": "text", "text": "capture metadata"},
                image_part(),
            ],
        }
    ]

    prepared = Completions._prepare_messages_for_provider(
        LegacyProvider(), "legacy", messages
    )

    assert prepared == [
        {
            "role": "tool",
            "tool_call_id": "call_capture",
            "content": "capture metadata",
        }
    ]
