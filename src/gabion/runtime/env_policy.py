# gabion:decision_protocol_module
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import os
from decimal import Decimal, InvalidOperation
from typing import Sequence
import warnings

from gabion.invariants import never

_TRUTHY_VALUES = {"1", "true", "yes", "on"}
_FALSEY_VALUES = {"0", "false", "no", "off"}

LSP_TIMEOUT_ENV_KEYS: tuple[str, ...] = (
    "GABION_LSP_TIMEOUT_TICKS",
    "GABION_LSP_TIMEOUT_TICK_NS",
    "GABION_LSP_TIMEOUT_MS",
    "GABION_LSP_TIMEOUT_SECONDS",
)

_LEGACY_TIMEOUT_ENV_WARNED = False


@dataclass(frozen=True)
class LspTimeoutConfig:
    ticks: int
    tick_ns: int

    def __post_init__(self) -> None:
        ticks_value = int(self.ticks)
        tick_ns_value = int(self.tick_ns)
        if ticks_value <= 0:
            never("invalid lsp timeout ticks", ticks=self.ticks)
        if tick_ns_value <= 0:
            never("invalid lsp timeout tick_ns", tick_ns=self.tick_ns)
        object.__setattr__(self, "ticks", ticks_value)
        object.__setattr__(self, "tick_ns", tick_ns_value)


_LSP_TIMEOUT_OVERRIDE: ContextVar[LspTimeoutConfig | None] = ContextVar(
    "gabion_lsp_timeout_override",
    default=None,
)


def env_text(name: str, *, default: str = "") -> str:
    return os.getenv(name, default).strip()


def has_any_non_empty_env(keys: Sequence[str]) -> bool:
    return any(env_text(key) for key in keys)


def lsp_timeout_env_present() -> bool:
    if _LSP_TIMEOUT_OVERRIDE.get() is not None:
        return True
    return has_any_non_empty_env(LSP_TIMEOUT_ENV_KEYS)


def lsp_timeout_override() -> LspTimeoutConfig | None:
    return _LSP_TIMEOUT_OVERRIDE.get()


def set_lsp_timeout_override(
    timeout: LspTimeoutConfig | None,
) -> Token[LspTimeoutConfig | None]:
    return _LSP_TIMEOUT_OVERRIDE.set(timeout)


def reset_lsp_timeout_override(
    token: Token[LspTimeoutConfig | None],
) -> None:
    _LSP_TIMEOUT_OVERRIDE.reset(token)


@contextmanager
def lsp_timeout_override_scope(timeout: LspTimeoutConfig | None):
    token = set_lsp_timeout_override(timeout)
    try:
        yield
    finally:
        reset_lsp_timeout_override(token)


def timeout_config_from_cli_flags(
    *,
    ticks: int | None = None,
    tick_ns: int | None = None,
    ms: int | None = None,
    seconds: float | str | None = None,
) -> LspTimeoutConfig:
    if (
        ticks is None
        and tick_ns is None
        and ms is None
        and seconds is None
    ):
        never("missing cli timeout flags")
    if ticks is not None or tick_ns is not None:
        if ticks is None or tick_ns is None:
            never(
                "timeout ticks and tick_ns must be provided together",
                ticks=ticks,
                tick_ns=tick_ns,
            )
        return LspTimeoutConfig(ticks=int(ticks), tick_ns=int(tick_ns))
    if ms is not None:
        return LspTimeoutConfig(ticks=int(ms), tick_ns=1_000_000)
    raw_seconds = str(seconds).strip() if seconds is not None else ""
    try:
        seconds_value = Decimal(raw_seconds)
    except (InvalidOperation, ValueError):
        never("invalid timeout seconds", seconds=seconds)
    if seconds_value <= 0:
        never("invalid timeout seconds", seconds=seconds)
    millis = int(seconds_value * Decimal(1000))
    if millis <= 0:
        never("invalid timeout seconds", seconds=seconds)
    return LspTimeoutConfig(ticks=millis, tick_ns=1_000_000)


def apply_cli_timeout_flags(
    *,
    ticks: int | None = None,
    tick_ns: int | None = None,
    ms: int | None = None,
    seconds: float | str | None = None,
) -> None:
    if (
        ticks is None
        and tick_ns is None
        and ms is None
        and seconds is None
    ):
        set_lsp_timeout_override(None)
        return
    set_lsp_timeout_override(
        timeout_config_from_cli_flags(
            ticks=ticks,
            tick_ns=tick_ns,
            ms=ms,
            seconds=seconds,
        )
    )


def _warn_legacy_timeout_env_usage() -> None:
    global _LEGACY_TIMEOUT_ENV_WARNED
    if _LEGACY_TIMEOUT_ENV_WARNED:
        return
    _LEGACY_TIMEOUT_ENV_WARNED = True
    warnings.warn(
        (
            "Legacy GABION_LSP_TIMEOUT_* env overrides are deprecated. "
            "Use CLI flags (--lsp-timeout-*) instead."
        ),
        DeprecationWarning,
        stacklevel=3,
    )


def env_enabled_default_true(name: str, *, value: str | None = None) -> bool:
    text = value.strip().lower() if isinstance(value, str) else os.getenv(name)
    if text is None:
        return True
    return text.strip().lower() not in _FALSEY_VALUES


def env_enabled_truthy_only(name: str, *, value: str | None = None) -> bool:
    text = value if isinstance(value, str) else os.getenv(name, "")
    return text.strip().lower() in _TRUTHY_VALUES


def env_enabled_flag(name: str, *, value: str | None = None) -> bool:
    text = value if isinstance(value, str) else os.getenv(name, "")
    return text.strip().lower() in _TRUTHY_VALUES


def parse_positive_int_text(raw: str, *, field: str) -> int:
    text = raw.strip()
    try:
        value = int(text)
    except (TypeError, ValueError):
        never(f"invalid {field}", **{field: raw})
    if value <= 0:
        never(f"invalid {field}", **{field: raw})
    return value


def timeout_ticks_from_env(
    *,
    ticks_key: str = "GABION_LSP_TIMEOUT_TICKS",
    tick_ns_key: str = "GABION_LSP_TIMEOUT_TICK_NS",
    ms_key: str = "GABION_LSP_TIMEOUT_MS",
    seconds_key: str = "GABION_LSP_TIMEOUT_SECONDS",
) -> tuple[int, int]:
    override = lsp_timeout_override()
    if override is not None:
        return (int(override.ticks), int(override.tick_ns))
    raw_ticks = env_text(ticks_key)
    raw_tick_ns = env_text(tick_ns_key)
    if raw_ticks:
        _warn_legacy_timeout_env_usage()
        ticks = parse_positive_int_text(raw_ticks, field="ticks")
        if not raw_tick_ns:
            never("missing env timeout tick_ns", ticks=raw_ticks)
        tick_ns = parse_positive_int_text(raw_tick_ns, field="tick_ns")
        return ticks, tick_ns

    raw_ms = env_text(ms_key)
    if raw_ms:
        _warn_legacy_timeout_env_usage()
        ms_value = parse_positive_int_text(raw_ms, field="ms")
        return ms_value, 1_000_000

    raw_seconds = env_text(seconds_key)
    if raw_seconds:
        _warn_legacy_timeout_env_usage()
        try:
            seconds_value = Decimal(raw_seconds)
        except (InvalidOperation, ValueError):
            never("invalid env timeout seconds", seconds=raw_seconds)
        if seconds_value <= 0:
            never("invalid env timeout seconds", seconds=raw_seconds)
        millis = int(seconds_value * Decimal(1000))
        if millis <= 0:
            never("invalid env timeout seconds", seconds=raw_seconds)
        return millis, 1_000_000

    never("missing env timeout configuration")
