import os
from typing import Any, AsyncGenerator, Dict, Optional, Union

from aisuite.framework.chat_completion_response import ChatCompletionResponse
from aisuite.provider import Provider
from aisuite.providers.openai_provider import OpenaiProvider

VERCEL_OPENAI_BASE_URL = "https://ai-gateway.vercel.sh/v1"
VERCEL_ANTHROPIC_BASE_URL = "https://ai-gateway.vercel.sh"
_VERCEL_ROUTING_CONFIG_KEYS = {
    "default_protocol",
    "protocol",
    "native_protocol",
    "protocols",
    "model_protocols",
    "infer_protocol_from_model",
}
_VERCEL_OPENAI_MODEL_PREFIXES = (
    "gpt-",
    "o1",
    "o3",
    "o4",
    "o5",
    "chatgpt-",
    "codex-",
    "text-embedding-",
    "omni-",
    "whisper-",
)


def _normalize_vercel_protocol(protocol: Optional[str]) -> Optional[str]:
    if protocol is None:
        return None
    normalized = str(protocol).strip().lower()
    if not normalized:
        return None
    return normalized


def _get_vercel_api_key(config: Dict[str, Any]) -> Optional[str]:
    return (
        config.get("api_key")
        or os.getenv("AI_GATEWAY_API_KEY")
        or os.getenv("VERCEL_OIDC_TOKEN")
    )


def _resolve_vercel_protocol_base_url(
    protocol: str, configured_base_url: Optional[str]
) -> str:
    protocol = _normalize_vercel_protocol(protocol) or "openai"
    base_url = (configured_base_url or "").strip().rstrip("/")

    if protocol == "anthropic":
        if not base_url:
            return VERCEL_ANTHROPIC_BASE_URL
        if base_url.endswith("/v1"):
            return base_url[:-3]
        return base_url

    if not base_url:
        return VERCEL_OPENAI_BASE_URL
    if base_url.endswith("/v1"):
        return base_url
    return f"{base_url}/v1"


class VercelOpenaiProvider(OpenaiProvider):
    """Thin Vercel adapter that reuses the OpenAI-compatible provider."""

    def __init__(self, **config):
        vercel_config = config.copy()
        vercel_config["api_key"] = _get_vercel_api_key(vercel_config)
        vercel_config["base_url"] = (
            vercel_config.get("base_url") or VERCEL_OPENAI_BASE_URL
        )

        if not vercel_config["api_key"]:
            raise ValueError(
                "Vercel AI Gateway API key is required. Set AI_GATEWAY_API_KEY "
                "or VERCEL_OIDC_TOKEN, or pass api_key parameter."
            )

        super().__init__(**vercel_config)


class VercelProvider(Provider):
    """Route Vercel AI Gateway requests to protocol-specific providers."""

    SUPPORTED_PROTOCOLS = {"openai", "anthropic"}
    DEFAULT_MODEL_PROTOCOLS = {
        "claude-": "anthropic",
    }

    def __init__(self, **config):
        self._config = config.copy()
        self._protocol_providers: Dict[str, Provider] = {}
        self._default_protocol = self._validate_protocol(
            self._config.get("default_protocol")
            or self._config.get("protocol")
            or "openai"
        )
        self._infer_protocol_from_model = bool(
            self._config.get("infer_protocol_from_model", True)
        )
        self._model_protocols = self._build_model_protocols(
            self._config.get("model_protocols")
        )

    def _validate_protocol(self, protocol: Optional[str]) -> str:
        normalized = _normalize_vercel_protocol(protocol)
        if normalized not in self.SUPPORTED_PROTOCOLS:
            raise ValueError(
                f"Unsupported Vercel protocol '{protocol}'. "
                f"Supported protocols: {sorted(self.SUPPORTED_PROTOCOLS)}"
            )
        return normalized

    def _build_model_protocols(
        self, configured_protocols: Optional[Dict[str, str]]
    ) -> Dict[str, str]:
        if configured_protocols is None:
            configured_protocols = {}

        if not isinstance(configured_protocols, dict):
            raise ValueError(
                "vercel model_protocols must be a dict of model prefix -> protocol"
            )

        merged_protocols = {}
        if self._infer_protocol_from_model:
            merged_protocols.update(self.DEFAULT_MODEL_PROTOCOLS)
        merged_protocols.update(configured_protocols)

        normalized_protocols = {}
        for model_prefix, protocol in merged_protocols.items():
            if not model_prefix:
                continue
            normalized_protocols[str(model_prefix).lower()] = self._validate_protocol(
                protocol
            )
        return normalized_protocols

    def _base_protocol_config(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in self._config.items()
            if key not in _VERCEL_ROUTING_CONFIG_KEYS
        }

    def _build_protocol_config(self, protocol: str) -> Dict[str, Any]:
        protocol_configs = self._config.get("protocols") or {}
        if not isinstance(protocol_configs, dict):
            raise ValueError("vercel protocols must be a dict of protocol -> config")

        shared_config = self._base_protocol_config()
        protocol_override = protocol_configs.get(protocol) or {}
        if not isinstance(protocol_override, dict):
            raise ValueError(f"vercel protocols['{protocol}'] must be a dict")

        protocol_config = shared_config.copy()
        protocol_config.update(protocol_override)

        api_key = _get_vercel_api_key(protocol_config)
        if not api_key:
            raise ValueError(
                "Vercel AI Gateway API key is required. Set AI_GATEWAY_API_KEY "
                "or VERCEL_OIDC_TOKEN, or pass api_key parameter."
            )

        configured_base_url = protocol_config.get("base_url")
        protocol_config["api_key"] = api_key
        protocol_config["base_url"] = _resolve_vercel_protocol_base_url(
            protocol, configured_base_url
        )
        return protocol_config

    def _extract_protocol_from_model(self, model: str) -> tuple[Optional[str], str]:
        for delimiter in ("/", ":"):
            if delimiter not in model:
                continue
            protocol_hint, stripped_model = model.split(delimiter, 1)
            normalized_protocol = _normalize_vercel_protocol(protocol_hint)
            if normalized_protocol in self.SUPPORTED_PROTOCOLS:
                return normalized_protocol, stripped_model
        return None, model

    def _match_protocol_from_model(self, model: str) -> Optional[str]:
        model_lower = model.lower()
        best_match = None
        best_match_length = -1

        for model_prefix, protocol in self._model_protocols.items():
            if (
                model_lower.startswith(model_prefix)
                and len(model_prefix) > best_match_length
            ):
                best_match = protocol
                best_match_length = len(model_prefix)

        return best_match

    def _normalize_model_for_protocol(self, protocol: str, model: str) -> str:
        normalized_model = model.strip()
        if not normalized_model:
            return normalized_model

        model_lower = normalized_model.lower()

        if protocol == "anthropic":
            if model_lower.startswith("anthropic/"):
                return normalized_model
            if "/" not in normalized_model and model_lower.startswith("claude"):
                return f"anthropic/{normalized_model}"
            return normalized_model

        if protocol == "openai":
            if "/" in normalized_model:
                return normalized_model
            if model_lower.startswith(_VERCEL_OPENAI_MODEL_PREFIXES):
                return f"openai/{normalized_model}"
            return normalized_model

        return normalized_model

    def _resolve_protocol(
        self, model: str, request_kwargs: Dict[str, Any]
    ) -> tuple[str, str]:
        explicit_protocol = request_kwargs.pop("protocol", None) or request_kwargs.pop(
            "native_protocol", None
        )
        model_protocol, stripped_model = self._extract_protocol_from_model(model)

        if explicit_protocol is not None:
            protocol = self._validate_protocol(explicit_protocol)
        elif model_protocol is not None:
            protocol = model_protocol
        else:
            matched_protocol = self._match_protocol_from_model(stripped_model)
            protocol = matched_protocol or self._default_protocol

        return protocol, self._normalize_model_for_protocol(protocol, stripped_model)

    def _create_protocol_provider(self, protocol: str) -> Provider:
        protocol_config = self._build_protocol_config(protocol)

        if protocol == "openai":
            return VercelOpenaiProvider(**protocol_config)

        if protocol == "anthropic":
            from aisuite.providers.anthropic_provider import AnthropicProvider

            return AnthropicProvider(**protocol_config)

        raise ValueError(
            f"Unsupported Vercel protocol '{protocol}'. "
            f"Supported protocols: {sorted(self.SUPPORTED_PROTOCOLS)}"
        )

    def _get_protocol_provider(self, protocol: str) -> Provider:
        if protocol not in self._protocol_providers:
            self._protocol_providers[protocol] = self._create_protocol_provider(protocol)
        return self._protocol_providers[protocol]

    async def chat_completions_create(
        self, model, messages, stream: bool = False, **kwargs
    ) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionResponse, None]]:
        request_kwargs = kwargs.copy()
        protocol, resolved_model = self._resolve_protocol(model, request_kwargs)
        provider = self._get_protocol_provider(protocol)
        return await provider.chat_completions_create(
            resolved_model,
            messages,
            stream=stream,
            **request_kwargs,
        )
