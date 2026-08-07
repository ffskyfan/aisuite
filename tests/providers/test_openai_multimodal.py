from aisuite.framework.content import make_image_data_url
from aisuite.providers.openai_provider import OpenaiProvider


def test_responses_input_keeps_images_in_user_and_function_output():
    provider = object.__new__(OpenaiProvider)
    image_url = make_image_data_url("image/png", b"frame")
    content = [
        {"type": "text", "text": "inspect"},
        {
            "type": "image_url",
            "image_url": {"url": image_url, "detail": "high"},
        },
    ]

    items = provider._build_responses_input_items(
        [
            {"role": "user", "content": content},
            {
                "role": "tool",
                "tool_call_id": "call_capture",
                "content": content,
            },
        ]
    )

    expected_content = [
        {"type": "input_text", "text": "inspect"},
        {"type": "input_image", "image_url": image_url, "detail": "high"},
    ]
    assert items[0] == {"role": "user", "content": expected_content}
    assert items[1] == {
        "type": "function_call_output",
        "call_id": "call_capture",
        "output": expected_content,
    }
