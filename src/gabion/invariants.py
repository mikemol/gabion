# gabion:decision_protocol_module
"""Invariant markers for Gabion analysis."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar, Token
from enum import StrEnum
from typing import Callable, NoReturn, TypeVar, cast

from gabion.exceptions import NeverThrown

_PROOF_MODE_OVERRIDE: ContextVar[bool | None] = ContextVar(
    "gabion_proof_mode_override",
    default=None,
)


@dataclass(frozen=True)
class ProofModeConfig:
    enabled: bool = False


class InvariantRuntimeBehaviorProfile(StrEnum):
    THROW = "throw"
    WARN = "warn"
    RATE = "rate"


@dataclass(frozen=True)
class InvariantRuntimeBehaviorConfig:
    profile: InvariantRuntimeBehaviorProfile = InvariantRuntimeBehaviorProfile.THROW


_PROOF_MODE_CONFIG: ContextVar[ProofModeConfig] = ContextVar(
    "gabion_proof_mode_config",
    default=ProofModeConfig(),
)

_INVARIANT_RUNTIME_BEHAVIOR_CONFIG: ContextVar[InvariantRuntimeBehaviorConfig] = ContextVar(
    "gabion_invariant_runtime_behavior_config",
    default=InvariantRuntimeBehaviorConfig(),
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


def invariant_factory(
    marker_kind: str, reasoning: object = "", **env: object
) -> NoReturn:
    from gabion.analysis.foundation.marker_protocol import (
        MarkerKind,
        never_marker_payload,
        normalize_marker_payload,
        normalize_marker_reasoning,
    )

    owner = str(env.get("owner", ""))
    expiry = str(env.get("expiry", ""))
    links = _normalized_marker_links(env.get("links", ()))
    raw_reasoning = env.get("reasoning", reasoning)
    if "reason" in env:
        reason = str(env["reason"])
    elif isinstance(reasoning, str):
        reason = reasoning
    else:
        reason = ""
    normalized_reasoning = normalize_marker_reasoning(raw_reasoning)
    if not normalized_reasoning.summary and reason:
        normalized_reasoning = normalize_marker_reasoning(reason)
    extra_env = {
        key: value
        for key, value in env.items()
        if key not in {"owner", "expiry", "links", "reasoning", "reason"}
    }
    marker_kind_enum = MarkerKind(marker_kind)
    if marker_kind_enum is MarkerKind.NEVER:
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
            fallback_reasoning = normalize_marker_reasoning(
                f"{marker_kind_enum.value}() marker reached"
            )
        payload = normalize_marker_payload(
            reason=reason or f"{marker_kind_enum.value}() marker reached",
            reasoning=fallback_reasoning,
            env=extra_env,
            marker_kind=marker_kind_enum,
            owner=owner,
            expiry=expiry,
            links=links,
        )
    raise NeverThrown(payload.reason, marker_payload=payload)


def never(reason: str = "", **env: object) -> NoReturn:
    """Mark a code path as intentionally unreachable.

    The analysis treats this as a sink that should be proven unreachable. The
    optional env payload is metadata only; it is not evaluated at runtime.
    """
    invariant_factory("never", reason=reason, **env)


def todo(reason: str = "", **env: object) -> NoReturn:
    """Mark a code path as intentionally pending implementation."""
    invariant_factory("todo", reason=reason, **env)


def deprecated(reason: str = "", **env: object) -> NoReturn:
    """Mark a code path as a deprecated/blocked semantic surface."""
    invariant_factory("deprecated", reason=reason, **env)


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


def invariant_runtime_behavior_config() -> InvariantRuntimeBehaviorConfig:
    return _INVARIANT_RUNTIME_BEHAVIOR_CONFIG.get()


def set_invariant_runtime_behavior_config(
    config: InvariantRuntimeBehaviorConfig,
) -> Token[InvariantRuntimeBehaviorConfig]:
    return _INVARIANT_RUNTIME_BEHAVIOR_CONFIG.set(config)


def reset_invariant_runtime_behavior_config(
    token: Token[InvariantRuntimeBehaviorConfig],
) -> None:
    _INVARIANT_RUNTIME_BEHAVIOR_CONFIG.reset(token)


@contextmanager
def invariant_runtime_behavior_scope(config: InvariantRuntimeBehaviorConfig):
    token = set_invariant_runtime_behavior_config(config)
    try:
        yield
    finally:
        reset_invariant_runtime_behavior_config(token)


@contextmanager
def proof_mode_scope(enabled: bool):
    token = _PROOF_MODE_OVERRIDE.set(bool(enabled))
    try:
        yield
    finally:
        _PROOF_MODE_OVERRIDE.reset(token)


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
