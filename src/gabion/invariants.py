# gabion:decision_protocol_module
"""Invariant markers for Gabion analysis."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Callable, NoReturn, TypeVar

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


def never(reason: str = "", **env: object) -> NoReturn:
    """Mark a code path as intentionally unreachable.

    The analysis treats this as a sink that should be proven unreachable. The
    optional env payload is metadata only; it is not evaluated at runtime.
    """
    _ = env
    message = reason or "never() invariant reached"
    raise NeverThrown(message)


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
