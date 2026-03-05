# gabion:decision_protocol_module
"""Invariant markers for Gabion analysis."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar, Token
from hashlib import sha1
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


_PROOF_MODE_CONFIG: ContextVar[ProofModeConfig] = ContextVar(
    "gabion_proof_mode_config",
    default=ProofModeConfig(),
)

T = TypeVar("T")
FuncT = TypeVar("FuncT", bound=Callable[..., object])


@dataclass(frozen=True)
class InvariantBehavior:
    throws: bool
    emits_warning: bool
    warning_limit: int


INVARIANT_PROFILE_BEHAVIOR: dict[str, dict[str, InvariantBehavior]] = {
    "never": {
        "strict": InvariantBehavior(throws=True, emits_warning=False, warning_limit=0),
        "warn": InvariantBehavior(throws=True, emits_warning=False, warning_limit=0),
        "silent": InvariantBehavior(throws=True, emits_warning=False, warning_limit=0),
    },
    "todo": {
        "strict": InvariantBehavior(throws=True, emits_warning=False, warning_limit=0),
        "warn": InvariantBehavior(throws=False, emits_warning=True, warning_limit=1),
        "silent": InvariantBehavior(throws=False, emits_warning=False, warning_limit=0),
    },
    "deprecated": {
        "strict": InvariantBehavior(throws=True, emits_warning=False, warning_limit=0),
        "warn": InvariantBehavior(throws=False, emits_warning=True, warning_limit=1),
        "silent": InvariantBehavior(throws=False, emits_warning=False, warning_limit=0),
    },
}

_DEFAULT_INVARIANT_PROFILE = "strict"
_INVARIANT_WARNING_COUNTS: ContextVar[dict[str, int]] = ContextVar(
    "gabion_invariant_warning_counts",
    default={},
)


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
    from gabion.analysis.foundation.marker_protocol import MarkerKind, never_marker_payload, normalize_marker_payload

    owner = str(env.get("owner", ""))
    expiry = str(env.get("expiry", ""))
    links = _normalized_marker_links(env.get("links", ()))
    extra_env = {key: value for key, value in env.items() if key not in {"owner", "expiry", "links"}}
    if marker_kind == MarkerKind.NEVER.value:
        payload = never_marker_payload(
            reason=reason,
            env=extra_env,
            owner=owner,
            expiry=expiry,
            links=links,
        )
    else:
        payload = normalize_marker_payload(
            reason=reason or f"{marker_kind}() marker reached",
            env=extra_env,
            marker_kind=MarkerKind(marker_kind),
            owner=owner,
            expiry=expiry,
            links=links,
        )
    raise NeverThrown(payload.reason, marker_payload=payload)


def _invariant_behavior(marker_kind: str, profile_name: str) -> InvariantBehavior:
    marker_profiles = INVARIANT_PROFILE_BEHAVIOR.get(marker_kind)
    if marker_profiles is None:
        return InvariantBehavior(throws=True, emits_warning=False, warning_limit=0)
    return marker_profiles.get(
        profile_name,
        marker_profiles[_DEFAULT_INVARIANT_PROFILE],
    )


def _warning_key(marker_kind: str, reason: str, **env: object) -> str:
    owner = str(env.get("owner", "")).strip()
    expiry = str(env.get("expiry", "")).strip()
    links = _normalized_marker_links(env.get("links", ()))
    stable_env = {
        str(key): value
        for key, value in sorted(env.items(), key=lambda entry: str(entry[0]))
        if key not in {"owner", "expiry", "links", "profile"}
    }
    encoded = repr((marker_kind, str(reason), owner, expiry, links, stable_env)).encode("utf-8")
    return sha1(encoded).hexdigest()


def _emit_warning_once(message: str, key: str, limit: int) -> None:
    if limit <= 0:
        return
    counts = dict(_INVARIANT_WARNING_COUNTS.get())
    if counts.get(key, 0) >= limit:
        return
    counts[key] = counts.get(key, 0) + 1
    _INVARIANT_WARNING_COUNTS.set(counts)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def invariant_factory(marker_kind: str, reasoning: str = "", **env: object) -> None:
    profile_name = str(env.get("profile", _DEFAULT_INVARIANT_PROFILE) or _DEFAULT_INVARIANT_PROFILE).strip().lower()
    behavior = _invariant_behavior(marker_kind, profile_name)
    message = str(reasoning)
    if behavior.emits_warning:
        warning_key = _warning_key(marker_kind, message, **env)
        _emit_warning_once(
            f"{marker_kind}() marker reached: {message or f'{marker_kind}() marker reached'}",
            warning_key,
            behavior.warning_limit,
        )
    if behavior.throws:
        _raise_marker(marker_kind, message, **env)


def reset_invariant_warning_counts() -> None:
    _INVARIANT_WARNING_COUNTS.set({})


def never(reason: str = "", **env: object) -> NoReturn:
    """Mark a code path as intentionally unreachable.

    The analysis treats this as a sink that should be proven unreachable. The
    optional env payload is metadata only; it is not evaluated at runtime.
    """
    invariant_factory("never", reason, **env)
    raise AssertionError("never() marker must throw")


def todo(reason: str = "", **env: object) -> None:
    """Mark a code path as intentionally pending implementation."""
    invariant_factory("todo", reason, **env)


def deprecated(reason: str = "", **env: object) -> None:
    """Mark a code path as a deprecated/blocked semantic surface."""
    invariant_factory("deprecated", reason, **env)


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
