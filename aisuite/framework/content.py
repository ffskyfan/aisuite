"""Canonical multimodal message content used across AISuite providers.

AISuite exposes an OpenAI-compatible public message shape while preserving a
provider-neutral implementation boundary.  Provider adapters consume the
helpers in this module instead of stringifying content arrays themselves.
"""

from __future__ import annotations

import base64
import binascii
import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union


CapabilityState = Literal["supported", "unsupported", "unknown"]
MessageContent = Union[str, List[Dict[str, Any]]]

SUPPORTED_IMAGE_MIME_TYPES: Tuple[str, ...] = (
    "image/png",
    "image/jpeg",
    "image/webp",
)
IMAGE_OMITTED_TEXT = "[image omitted: selected model does not support vision]"

_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[-\w.+/]+)(?P<params>(?:;[-\w.+]+=[^;,]+)*);base64,(?P<data>.*)$",
    re.IGNORECASE | re.DOTALL,
)


class MultimodalContentError(ValueError):
    """Raised when canonical multimodal content is malformed or unsupported."""


@dataclass(frozen=True)
class MultimodalCapabilities:
    """Effective multimodal input capabilities for a provider/model pair."""

    user_images: CapabilityState = "unknown"
    tool_result_images: CapabilityState = "unsupported"
    supported_image_mime_types: Tuple[str, ...] = SUPPORTED_IMAGE_MIME_TYPES
    supports_data_urls: bool = True

    def image_state_for_role(self, role: Optional[str]) -> CapabilityState:
        if role == "tool":
            return self.tool_result_images
        if role == "user":
            return self.user_images
        # This design only defines image inputs for user and tool results.
        return "unsupported"


@dataclass(frozen=True)
class ToolResult:
    """Explicit tool return envelope for text or multimodal tool content."""

    content: MessageContent
    is_error: bool = False


def is_text_part(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") == "text"


def is_image_part(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") == "image_url"


def has_image_content(content: Any) -> bool:
    return isinstance(content, list) and any(is_image_part(part) for part in content)


def count_image_parts(content: Any) -> int:
    if not isinstance(content, list):
        return 0
    return sum(1 for part in content if is_image_part(part))


def parse_image_data_url(
    url: str,
    *,
    supported_mime_types: Tuple[str, ...] = SUPPORTED_IMAGE_MIME_TYPES,
    max_image_bytes: Optional[int] = None,
) -> Tuple[str, bytes]:
    """Validate and decode a base64 image data URL."""

    if not isinstance(url, str):
        raise MultimodalContentError("image_url.url must be a string")

    match = _DATA_URL_RE.match(url)
    if not match:
        raise MultimodalContentError("image_url.url must be a base64 data URL")

    mime_type = match.group("mime").lower()
    if mime_type not in supported_mime_types:
        raise MultimodalContentError(
            f"unsupported image MIME type: {mime_type}"
        )

    try:
        data = base64.b64decode(match.group("data"), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MultimodalContentError("image data URL contains invalid base64") from exc

    if max_image_bytes is not None and len(data) > max_image_bytes:
        raise MultimodalContentError(
            f"image exceeds maximum decoded size of {max_image_bytes} bytes"
        )
    return mime_type, data


def make_image_data_url(mime_type: str, data: bytes) -> str:
    if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        raise MultimodalContentError(f"unsupported image MIME type: {mime_type}")
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def get_image_part_url(part: Dict[str, Any]) -> str:
    if not is_image_part(part):
        raise MultimodalContentError("content part is not an image_url part")
    image_url = part.get("image_url")
    if not isinstance(image_url, dict):
        raise MultimodalContentError("image_url part must contain an object")
    url = image_url.get("url")
    if not isinstance(url, str) or not url:
        raise MultimodalContentError("image_url.url must be a non-empty string")
    return url


def get_image_part_detail(part: Dict[str, Any]) -> str:
    image_url = part.get("image_url") if isinstance(part, dict) else None
    detail = image_url.get("detail", "auto") if isinstance(image_url, dict) else "auto"
    if detail not in {"auto", "low", "high"}:
        raise MultimodalContentError("image detail must be auto, low, or high")
    return detail


def validate_message_content(
    content: Any,
    *,
    max_images: Optional[int] = None,
    max_image_bytes: Optional[int] = None,
    supported_mime_types: Tuple[str, ...] = SUPPORTED_IMAGE_MIME_TYPES,
    require_data_urls: bool = False,
) -> MessageContent:
    """Validate canonical content without mutating it."""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise MultimodalContentError("message content must be a string or a list")

    image_count = 0
    for index, part in enumerate(content):
        if not isinstance(part, dict):
            raise MultimodalContentError(f"content part {index} must be an object")
        if is_text_part(part):
            if not isinstance(part.get("text"), str):
                raise MultimodalContentError(
                    f"text content part {index} must contain a string"
                )
            continue
        if is_image_part(part):
            image_count += 1
            url = get_image_part_url(part)
            get_image_part_detail(part)
            if url.startswith("data:"):
                parse_image_data_url(
                    url,
                    supported_mime_types=supported_mime_types,
                    max_image_bytes=max_image_bytes,
                )
            elif require_data_urls:
                raise MultimodalContentError(
                    "this provider path requires base64 image data URLs"
                )
            continue
        raise MultimodalContentError(
            f"unsupported content part type at index {index}: {part.get('type')!r}"
        )

    if max_images is not None and image_count > max_images:
        raise MultimodalContentError(
            f"message contains {image_count} images; maximum is {max_images}"
        )
    return content


def filter_image_content(
    content: Any,
    *,
    placeholder: str = IMAGE_OMITTED_TEXT,
) -> Tuple[MessageContent, int]:
    """Remove image parts while leaving a provider-valid message behind."""

    if isinstance(content, str):
        return content, 0
    if not isinstance(content, list):
        return str(content), 0

    filtered = [copy.deepcopy(part) for part in content if not is_image_part(part)]
    removed = len(content) - len(filtered)
    if removed == 0:
        return copy.deepcopy(content), 0

    text_parts = [
        part.get("text", "")
        for part in filtered
        if is_text_part(part) and isinstance(part.get("text"), str)
    ]
    if not text_parts or not any(text.strip() for text in text_parts):
        return placeholder, removed
    # Filtering runs on a provider-specific copy. Collapse the remaining text
    # blocks to the legacy string shape so adapters without multimodal parsing
    # can still consume the downgraded message.
    return "\n".join(text_parts), removed


def filter_images_for_capabilities(
    messages: List[Any],
    capabilities: MultimodalCapabilities,
) -> Tuple[List[Any], int]:
    """Copy messages and filter image parts for explicitly unsupported roles."""

    filtered_messages: List[Any] = []
    removed_total = 0
    for message in messages:
        if isinstance(message, dict):
            copied = copy.deepcopy(message)
        elif hasattr(message, "model_dump"):
            copied = message.model_dump()
        else:
            filtered_messages.append(message)
            continue

        role = copied.get("role")
        content = copied.get("content")
        if has_image_content(content):
            if capabilities.image_state_for_role(role) == "unsupported":
                copied["content"], removed = filter_image_content(content)
                removed_total += removed
            else:
                validate_message_content(
                    content,
                    supported_mime_types=capabilities.supported_image_mime_types,
                )
        filtered_messages.append(copied)
    return filtered_messages, removed_total


def content_text(content: Any, *, separator: str = "\n") -> str:
    """Extract only canonical text blocks from message content."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    return separator.join(
        part.get("text", "") for part in content if is_text_part(part)
    )


def to_openai_responses_content(content: Any) -> Any:
    """Convert canonical content parts to OpenAI Responses input content."""

    if not isinstance(content, list):
        return content

    converted: List[Dict[str, Any]] = []
    for part in content:
        if is_text_part(part):
            converted.append({"type": "input_text", "text": part.get("text", "")})
        elif is_image_part(part):
            converted_part: Dict[str, Any] = {
                "type": "input_image",
                "image_url": get_image_part_url(part),
            }
            detail = get_image_part_detail(part)
            if detail:
                converted_part["detail"] = detail
            converted.append(converted_part)
        else:
            raise MultimodalContentError(
                f"unsupported canonical content part: {part!r}"
            )
    return converted


def to_anthropic_content(content: Any) -> Any:
    """Convert canonical content parts to Anthropic content blocks."""

    if not isinstance(content, list):
        return content

    converted: List[Dict[str, Any]] = []
    for part in content:
        if is_text_part(part):
            converted.append({"type": "text", "text": part.get("text", "")})
            continue
        if not is_image_part(part):
            raise MultimodalContentError(
                f"unsupported canonical content part: {part!r}"
            )

        url = get_image_part_url(part)
        if url.startswith("data:"):
            mime_type, data = parse_image_data_url(url)
            converted.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64.b64encode(data).decode("ascii"),
                    },
                }
            )
        else:
            converted.append(
                {
                    "type": "image",
                    "source": {"type": "url", "url": url},
                }
            )
    return converted
