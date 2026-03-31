from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Sequence


REPLAY_PAYLOAD_VERSION = 1


@dataclass(frozen=True)
class ProviderReplayCapabilities:
    """Minimal replay capability contract exposed by a provider."""

    needs_exact_turn_replay: bool = False
    needs_provider_call_id_binding: bool = False
    needs_reasoning_raw_replay: bool = False
    supports_canonical_only_history: bool = True
    empty_actionless_stop_is_retryable: bool = False


@dataclass(frozen=True)
class ReplayDiagnostic:
    """Provider replay diagnostic produced during validation/build."""

    code: str
    message: str
    severity: Literal["error", "warning"] = "error"
    provider: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayValidationResult:
    """Replay-window validation result."""

    ok: bool = True
    degraded: bool = False
    diagnostics: Sequence[ReplayDiagnostic] = ()


@dataclass(frozen=True)
class ReplayCaptureResult:
    """Captured replay data produced from a provider response."""

    canonical_message: Any = None
    stop_info: Any = None
    replay_metadata: Dict[str, Any] = field(default_factory=dict)
    protocol_diagnostics: Sequence[ReplayDiagnostic] = ()


@dataclass(frozen=True)
class ReplayBuildResult:
    """Provider-native replay view produced from canonical messages."""

    request_view: Any = None
    replay_mode: str = "canonical"
    degraded: bool = False
    diagnostics: Sequence[ReplayDiagnostic] = ()


def build_replay_payload(
    provider: str,
    kind: str,
    payload: Any,
    *,
    meta: Optional[Dict[str, Any]] = None,
    legacy_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a versioned replay payload envelope.

    `legacy_fields` allows a transitional mirror of commonly accessed fields so
    old readers can keep working while new readers consume the envelope.
    """

    envelope: Dict[str, Any] = {
        "version": REPLAY_PAYLOAD_VERSION,
        "provider": provider,
        "kind": kind,
        "payload": payload,
        "meta": meta or {},
    }

    if legacy_fields:
        for key, value in legacy_fields.items():
            if key not in envelope:
                envelope[key] = value

    return envelope


def is_replay_payload(data: Any) -> bool:
    return isinstance(data, dict) and {
        "version",
        "provider",
        "kind",
        "payload",
    }.issubset(data.keys())


def get_replay_payload(data: Any) -> Optional[Dict[str, Any]]:
    if is_replay_payload(data):
        return data
    return None


def unwrap_replay_payload(data: Any) -> Any:
    envelope = get_replay_payload(data)
    if envelope is not None:
        return envelope.get("payload")
    return data


def get_replay_payload_meta(data: Any) -> Dict[str, Any]:
    envelope = get_replay_payload(data)
    if envelope is None:
        return {}
    meta = envelope.get("meta")
    if isinstance(meta, dict):
        return meta
    return {}
