import os
from typing import Any, Dict, List, Optional

import openai

from aisuite.framework.replay_payload import (
    ReplayBuildResult,
    ReplayDiagnostic,
    ReplayValidationResult,
    build_replay_payload,
    get_replay_payload,
    unwrap_replay_payload,
)
from aisuite.framework.message import ReasoningContent
from aisuite.framework.content import MultimodalCapabilities
from aisuite.provider import LLMError
from aisuite.providers.deepseek_provider import DeepseekProvider


class KimiProvider(DeepseekProvider):
    """Kimi K3 provider using OpenAI-compatible transport and preserved reasoning replay."""

    PROVIDER_NAME = "kimi"
    REASONING_REPLAY_KIND = "kimi_reasoning_text"

    def get_multimodal_capabilities(
        self, model: str | None = None
    ) -> MultimodalCapabilities:
        # Moonshot model families differ in user-image support. Keep user input
        # optimistic, but do not send non-standard image blocks as tool output.
        return MultimodalCapabilities(
            user_images="unknown",
            tool_result_images="unsupported",
        )

    def __init__(self, **config):
        api_key = (
            config.get("api_key")
            or os.getenv("KIMI_API_KEY")
            or os.getenv("MOONSHOT_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "Kimi API key is missing. Provide api_key or set KIMI_API_KEY/MOONSHOT_API_KEY."
            )

        client_config = dict(config)
        client_config["api_key"] = api_key
        client_config["base_url"] = (
            client_config.get("base_url")
            or os.getenv("KIMI_BASE_URL")
            or os.getenv("MOONSHOT_BASE_URL")
            or "https://api.moonshot.ai/v1"
        )

        self._http_client = client_config.get("http_client")
        self._owns_http_client = self._http_client is None
        if self._http_client is None:
            self._http_client = openai.DefaultAsyncHttpxClient()
            client_config["http_client"] = self._http_client

        self.client = openai.AsyncOpenAI(**client_config)
        self._streaming_tool_calls = {}
        self._streaming_reasoning = ""
        self._stream_content_length = 0
        self._stream_tool_calls_count = 0

    def _build_reasoning_replay_payload(self, reasoning_content: str) -> Dict[str, Any]:
        return build_replay_payload(
            self.PROVIDER_NAME,
            self.REASONING_REPLAY_KIND,
            {"reasoning_content": reasoning_content},
            legacy_fields={"reasoning_content": reasoning_content},
        )

    def _extract_reasoning_input(self, reasoning_content: Any) -> Optional[str]:
        if reasoning_content is None:
            return None

        raw_data: Dict[str, Any] = {}
        fallback = None
        if isinstance(reasoning_content, ReasoningContent):
            raw_data = reasoning_content.raw_data or {}
            fallback = reasoning_content.thinking
        elif hasattr(reasoning_content, "thinking"):
            raw_data = getattr(reasoning_content, "raw_data", None) or {}
            fallback = getattr(reasoning_content, "thinking", None)
        elif isinstance(reasoning_content, dict):
            raw_data = reasoning_content.get("raw_data") or {}
            fallback = (
                reasoning_content.get("reasoning_content")
                or reasoning_content.get("thinking")
                or reasoning_content.get("text")
            )
        elif isinstance(reasoning_content, str):
            return reasoning_content or None
        else:
            return None

        envelope = get_replay_payload(raw_data)
        if envelope and envelope.get("provider") == self.PROVIDER_NAME:
            payload = unwrap_replay_payload(raw_data)
            if isinstance(payload, dict):
                value = payload.get("reasoning_content")
                if isinstance(value, str) and value:
                    return value

        legacy_value = raw_data.get("reasoning_content")
        if isinstance(legacy_value, str) and legacy_value:
            return legacy_value
        return fallback if isinstance(fallback, str) and fallback else None

    def _has_reasoning_input(self, reasoning_content: Any) -> bool:
        value = self._extract_reasoning_input(reasoning_content)
        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def _normalize_message(message: Any) -> Dict[str, Any]:
        if isinstance(message, dict):
            return message
        if hasattr(message, "model_dump"):
            return message.model_dump()
        return {}

    def validate_replay_window(
        self, model: str, messages: list, **kwargs
    ) -> ReplayValidationResult:
        diagnostics: List[ReplayDiagnostic] = []
        normalized_messages = [self._normalize_message(message) for message in messages]

        expected_tool_call_ids = set()
        observed_tool_result_ids = set()
        for message in normalized_messages:
            role = message.get("role")
            if role == "assistant" and message.get("tool_calls"):
                for tool_call in message.get("tool_calls") or []:
                    tool_call_data = self._normalize_message(tool_call)
                    function = tool_call_data.get("function") or {}
                    if not isinstance(function, dict) and hasattr(
                        function, "model_dump"
                    ):
                        function = function.model_dump()
                    tool_call_id = tool_call_data.get("id")
                    if not tool_call_id:
                        diagnostics.append(
                            ReplayDiagnostic(
                                code="missing_tool_call_id",
                                message="Kimi assistant tool call is missing id.",
                                provider=self.PROVIDER_NAME,
                            )
                        )
                    else:
                        expected_tool_call_ids.add(tool_call_id)
                    if not function.get("name"):
                        diagnostics.append(
                            ReplayDiagnostic(
                                code="missing_tool_function_name",
                                message="Kimi assistant tool call is missing function name.",
                                provider=self.PROVIDER_NAME,
                                metadata={"tool_call_id": tool_call_id},
                            )
                        )
                    if function.get("arguments") is None:
                        diagnostics.append(
                            ReplayDiagnostic(
                                code="missing_tool_arguments",
                                message="Kimi assistant tool call is missing arguments.",
                                provider=self.PROVIDER_NAME,
                                metadata={"tool_call_id": tool_call_id},
                            )
                        )

            if role == "tool":
                tool_call_id = message.get("tool_call_id")
                if not tool_call_id:
                    diagnostics.append(
                        ReplayDiagnostic(
                            code="missing_tool_call_id",
                            message="Kimi tool result replay requires tool_call_id.",
                            provider=self.PROVIDER_NAME,
                            metadata={"role": "tool"},
                        )
                    )
                else:
                    observed_tool_result_ids.add(tool_call_id)

            if (
                role == "assistant"
                and message.get("reasoning_content") is not None
                and not self._has_reasoning_input(message.get("reasoning_content"))
            ):
                diagnostics.append(
                    ReplayDiagnostic(
                        code="missing_reasoning_raw_replay",
                        message=(
                            "Kimi preserved reasoning requires provider-native raw replay payload."
                        ),
                        severity="warning",
                        provider=self.PROVIDER_NAME,
                        metadata={"role": "assistant"},
                    )
                )

        for tool_call_id in observed_tool_result_ids - expected_tool_call_ids:
            diagnostics.append(
                ReplayDiagnostic(
                    code="orphan_tool_result",
                    message="Kimi tool result has no matching assistant tool call.",
                    provider=self.PROVIDER_NAME,
                    metadata={"tool_call_id": tool_call_id},
                )
            )

        return ReplayValidationResult(
            ok=not any(item.severity == "error" for item in diagnostics),
            degraded=any(item.severity == "warning" for item in diagnostics),
            diagnostics=tuple(diagnostics),
        )

    def build_replay_view(self, model: str, messages: list, **kwargs):
        validation = self.validate_replay_window(model, messages, **kwargs)
        if not validation.ok:
            error_codes = ", ".join(
                item.code for item in validation.diagnostics if item.severity == "error"
            )
            raise LLMError(f"Kimi replay window validation failed: {error_codes}")
        return ReplayBuildResult(
            request_view=self._prepare_messages(messages),
            replay_mode="canonical_with_reasoning",
            degraded=validation.degraded,
            diagnostics=validation.diagnostics,
        )

    def _prepare_request_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        prepared = dict(kwargs)
        prepared.pop("thinking", None)
        prepared.pop("reasoning", None)
        prepared.pop("extra_body", None)
        prepared.pop("temperature", None)
        prepared.pop("top_p", None)
        prepared.pop("n", None)
        prepared.pop("presence_penalty", None)
        prepared.pop("frequency_penalty", None)
        prepared.pop("verbosity", None)

        if "max_tokens" in prepared and "max_completion_tokens" not in prepared:
            prepared["max_completion_tokens"] = prepared.pop("max_tokens")
        prepared["reasoning_effort"] = "max"
        return prepared

    def _normalize_usage(self, usage_obj):
        normalized = super()._normalize_usage(usage_obj)
        if normalized is None:
            return None

        if isinstance(usage_obj, dict):
            cached_tokens = usage_obj.get("cached_tokens")
        else:
            cached_tokens = getattr(usage_obj, "cached_tokens", None)
        if cached_tokens is not None:
            normalized["cache_read_input_tokens"] = max(0, int(cached_tokens))
        return normalized
