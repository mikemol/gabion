# gabion:decision_protocol_module
"""Invariant markers for Gabion analysis."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar, Token
import warnings
from typing import Callable, NoReturn, TypeVar, cast

from gabion.exceptions import NeverThrown

_PROOF_MODE_OVERRIDE: ContextVar[bool | None] = ContextVar(
    "gabion_proof_mode_override",
    default=None,
 )


@dataclass(frozen=True)
class ProofModeConfig:
    enabled: bool = False


@dataclass(frozen=True)
class MarkerBehaviorProfile:
    throw_never: bool = True
    throw_todo: bool = True
    throw_deprecated: bool = True
    warn_never: bool = False
    warn_todo: bool = False
    warn_deprecated: bool = False
    warning_cap: int = 1

    def throw_enabled(self, marker_kind: str) -> bool:
        if marker_kind == "never":
            return self.throw_never
        if marker_kind == "todo":
            return self.throw_todo
        if marker_kind == "deprecated":
            return self.throw_deprecated
        return True

    def warn_enabled(self, marker_kind: str) -> bool:
        if marker_kind == "never":
            return self.warn_never
        if marker_kind == "todo":
            return self.warn_todo
        if marker_kind == "deprecated":
            return self.warn_deprecated
        return False


@dataclass
class MarkerRuntimePolicyConfig:
    behavior_profile: MarkerBehaviorProfile = MarkerBehaviorProfile()
    warning_counts: dict[object, int] | None = None

    def warning_count_for(self, key: object) -> int:
        if self.warning_counts is None:
            return 0
        return int(self.warning_counts.get(key, 0))

    def increment_warning_count(self, key: object) -> None:
        if self.warning_counts is None:
            self.warning_counts = {}
        self.warning_counts[key] = self.warning_count_for(key) + 1


_PROOF_MODE_CONFIG: ContextVar[ProofModeConfig] = ContextVar(
    "gabion_proof_mode_config",
    default=ProofModeConfig(),
)

_MARKER_RUNTIME_POLICY_CONFIG: ContextVar[MarkerRuntimePolicyConfig] = ContextVar(
    "gabion_marker_runtime_policy_config",
    default=MarkerRuntimePolicyConfig(),
)

T = TypeVar("T")
FuncT = TypeVar("FuncT", bound=Callable[..., object])


def _normalized_marker_links(raw_links: object) -> tuple[dict[str, str], ...]:
    if type(raw_links) is not list:
        return ()
    links = cast(list[object], raw_links)
    normalized: list[dict[str, str]] = []
    for item in links:
        if type(item) is not dict:
            continue
        payload = cast(dict[object, object], item)
        kind = str(payload.get("kind", "")).strip().lower()
        value = str(payload.get("value", "")).strip()
        if kind and value:
            normalized.append({"kind": kind, "value": value})
    return tuple(normalized)


def _raise_marker(marker_kind: str, reason: str = "", **env: object) -> NoReturn:
    from gabion.analysis.foundation.marker_protocol import (
        MarkerKind,
        marker_identity,
        never_marker_payload,
        normalize_marker_payload,
        normalize_marker_reasoning,
    )

    owner = str(env.get("owner", ""))
    expiry = str(env.get("expiry", ""))
    links = _normalized_marker_links(env.get("links", ()))
    raw_reasoning = env.get("reasoning", reason)
    normalized_reasoning = normalize_marker_reasoning(raw_reasoning)
    extra_env = {key: value for key, value in env.items() if key not in {"owner", "expiry", "links", "reasoning"}}
    if marker_kind == MarkerKind.NEVER.value:
        payload = never_marker_payload(
            reason=reason,
            reasoning=normalized_reasoning,
            env=extra_env,
            owner=owner,
            expiry=expiry,
            links=links,
        )
    else:
        fallback_reasoning = normalized_reasoning
        if not fallback_reasoning.summary:
            fallback_reasoning = normalize_marker_reasoning(f"{marker_kind}() marker reached")
        payload = normalize_marker_payload(
            reason=reason or f"{marker_kind}() marker reached",
            reasoning=fallback_reasoning,
            env=extra_env,
            marker_kind=MarkerKind(marker_kind),
            owner=owner,
            expiry=expiry,
            links=links,
        )
    marker_runtime_policy = _MARKER_RUNTIME_POLICY_CONFIG.get()
    behavior_profile = marker_runtime_policy.behavior_profile
    if behavior_profile.warn_enabled(marker_kind):
        warning_cap = max(0, int(behavior_profile.warning_cap))
        warning_key: object
        if marker_kind == MarkerKind.NEVER.value:
            warning_key = marker_identity(payload)
        else:
            warning_key = (
                payload.marker_kind.value,
                payload.reasoning.summary,
                payload.reasoning.control,
                payload.reasoning.blocking_dependencies,
            )
        if warning_cap <= 0 or marker_runtime_policy.warning_count_for(warning_key) < warning_cap:
            warnings.warn(
                f"{payload.marker_kind.value}() marker reached: {payload.reason}",
                RuntimeWarning,
                stacklevel=3,
            )
            marker_runtime_policy.increment_warning_count(warning_key)
    if behavior_profile.throw_enabled(marker_kind):
        raise NeverThrown(payload.reason, marker_payload=payload)
    return cast(NoReturn, None)


def never(reason: str = "", **env: object) -> NoReturn:
    """Mark a code path as intentionally unreachable.

    The analysis treats this as a sink that should be proven unreachable. The
    optional env payload is metadata only; it is not evaluated at runtime.
    """
    _raise_marker("never", reason, **env)


def todo(reason: str = "", **env: object) -> NoReturn:
    """Mark a code path as intentionally pending implementation."""
    _raise_marker("todo", reason, **env)


def deprecated(reason: str = "", **env: object) -> NoReturn:
    """Mark a code path as a deprecated/blocked semantic surface."""
    _raise_marker("deprecated", reason, **env)


def proof_mode() -> bool:
    override = _PROOF_MODE_OVERRIDE.get()
    if override is not None:
        return bool(override)
    return bool(_PROOF_MODE_CONFIG.get().enabled)


def set_proof_mode_config(config: ProofModeConfig) -> Token[ProofModeConfig]:
    return _PROOF_MODE_CONFIG.set(config)


def reset_proof_mode_config(token: Token[ProofModeConfig]) -> None:
    _PROOF_MODE_CONFIG.reset(token)


@contextmanager
def proof_mode_config_scope(config: ProofModeConfig):
    token = set_proof_mode_config(config)
    try:
        yield
    finally:
        reset_proof_mode_config(token)


@contextmanager
def proof_mode_scope(enabled: bool):
    token = _PROOF_MODE_OVERRIDE.set(bool(enabled))
    try:
        yield
    finally:
        _PROOF_MODE_OVERRIDE.reset(token)


def set_marker_runtime_policy_config(config: MarkerRuntimePolicyConfig) -> Token[MarkerRuntimePolicyConfig]:
    return _MARKER_RUNTIME_POLICY_CONFIG.set(config)


def reset_marker_runtime_policy_config(token: Token[MarkerRuntimePolicyConfig]) -> None:
    _MARKER_RUNTIME_POLICY_CONFIG.reset(token)


@contextmanager
def marker_runtime_policy_scope(config: MarkerRuntimePolicyConfig):
    token = set_marker_runtime_policy_config(config)
    try:
        yield
    finally:
        reset_marker_runtime_policy_config(token)


def require_not_none(
    value: T | None,
    *,
    reason: str = "",
    strict: bool | None = None,
    **env: object,
) -> T | None:
    if value is None:
        if strict is None:
            strict = proof_mode()
        if strict:
            never(reason or "required value is None", **env)
    return value


def decision_protocol(func: FuncT) -> FuncT:
    """Marker decorator for explicit decision-protocol control surfaces."""
    return func


def boundary_normalization(func: FuncT) -> FuncT:
    """Marker decorator for boundary normalization surfaces."""
    return func
