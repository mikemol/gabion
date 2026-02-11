"""Invariant markers for Gabion analysis."""

from __future__ import annotations

import os
from typing import NoReturn, TypeVar

from gabion.exceptions import NeverThrown

_PROOF_ENV = "GABION_PROOF_MODE"
_STRICT_VALUES = {"1", "true", "yes", "on", "strict"}

T = TypeVar("T")


def never(reason: str = "", **env: object) -> NoReturn:
    """Mark a code path as intentionally unreachable.

    The analysis treats this as a sink that should be proven unreachable. The
    optional env payload is metadata only; it is not evaluated at runtime.
    """
    _ = env
    message = reason or "never() invariant reached"
    raise NeverThrown(message)


def proof_mode() -> bool:
    value = os.environ.get(_PROOF_ENV, "")
    return value.strip().lower() in _STRICT_VALUES


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
