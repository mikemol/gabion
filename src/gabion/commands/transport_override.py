from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class TransportOverrideConfig:
    direct_requested: bool | None = None
    override_record_path: str | None = None
    override_record_json: str | None = None


_TRANSPORT_OVERRIDE: ContextVar[TransportOverrideConfig | None] = ContextVar(
    "gabion_transport_override",
    default=None,
)


def transport_override() -> TransportOverrideConfig | None:
    return _TRANSPORT_OVERRIDE.get()


def transport_override_present() -> bool:
    return transport_override() is not None


def set_transport_override(
    override: TransportOverrideConfig | None,
) -> Token[TransportOverrideConfig | None]:
    return _TRANSPORT_OVERRIDE.set(override)


def reset_transport_override(
    token: Token[TransportOverrideConfig | None],
) -> None:
    _TRANSPORT_OVERRIDE.reset(token)


@contextmanager
# gabion:decision_protocol
def transport_override_scope(override: TransportOverrideConfig | None) -> Iterator[None]:
    token = set_transport_override(override)
    try:
        yield
    finally:
        reset_transport_override(token)


__all__ = [
    "TransportOverrideConfig",
    "transport_override",
    "transport_override_present",
    "set_transport_override",
    "reset_transport_override",
    "transport_override_scope",
]
