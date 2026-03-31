from abc import ABC, abstractmethod
from pathlib import Path
import importlib
import os
import functools

from aisuite.framework.replay_payload import (
    ReplayBuildResult,
    ReplayCaptureResult,
    ProviderReplayCapabilities,
    ReplayValidationResult,
)


class LLMError(Exception):
    """Custom exception for LLM errors."""

    def __init__(self, message):
        super().__init__(message)


class Provider(ABC):
    @abstractmethod
    def chat_completions_create(self, model, messages):
        """Abstract method for chat completion calls, to be implemented by each provider."""
        pass

    def get_replay_capabilities(self, model: str | None = None) -> ProviderReplayCapabilities:
        """Return provider replay capabilities.

        Providers that do not need replay-specific behavior can use the default
        no-op capability set.
        """

        return ProviderReplayCapabilities()

    def capture_response(self, response, model: str | None = None, **kwargs):
        """Capture replay-relevant response metadata.

        Default behavior is a no-op so existing providers do not need to
        implement the replay contract immediately.
        """

        return ReplayCaptureResult()

    def validate_replay_window(self, model: str, messages: list, **kwargs) -> ReplayValidationResult:
        """Validate whether the current replay window is usable."""

        return ReplayValidationResult(ok=True)

    def build_replay_view(self, model: str, messages: list, **kwargs):
        """Build a provider-native replay view from canonical messages."""

        return ReplayBuildResult(request_view=messages, replay_mode="canonical")


class ProviderFactory:
    """Factory to dynamically load provider instances based on naming conventions."""

    PROVIDERS_DIR = Path(__file__).parent / "providers"

    @classmethod
    def create_provider(cls, provider_key, config):
        """Dynamically load and create an instance of a provider based on the naming convention."""
        # Convert provider_key to the expected module and class names
        provider_class_name = f"{provider_key.capitalize()}Provider"
        provider_module_name = f"{provider_key}_provider"

        module_path = f"aisuite.providers.{provider_module_name}"

        # Lazily load the module
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Could not import module {module_path}: {str(e)}. Please ensure the provider is supported by doing ProviderFactory.get_supported_providers()"
            )

        # Instantiate the provider class
        provider_class = getattr(module, provider_class_name)
        return provider_class(**config)

    @classmethod
    @functools.cache
    def get_supported_providers(cls):
        """List all supported provider names based on files present in the providers directory."""
        provider_files = Path(cls.PROVIDERS_DIR).glob("*_provider.py")
        return {file.stem.replace("_provider", "") for file in provider_files}
