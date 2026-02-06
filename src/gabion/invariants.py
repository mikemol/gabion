"""Invariant markers for Gabion analysis."""

from __future__ import annotations

from typing import NoReturn

from gabion.exceptions import NeverThrown


def never(reason: str = "", **env: object) -> NoReturn:
    """Mark a code path as intentionally unreachable.

    The analysis treats this as a sink that should be proven unreachable. The
    optional env payload is metadata only; it is not evaluated at runtime.
    """
    _ = env
    message = reason or "never() invariant reached"
    raise NeverThrown(message)
