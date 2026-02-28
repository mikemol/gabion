# gabion:decision_protocol_module
"""Invariant markers for Gabion analysis."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar, Token
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
    from gabion.analysis.marker_protocol import MarkerKind, never_marker_payload, normalize_marker_payload

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
