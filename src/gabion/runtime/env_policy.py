# gabion:decision_protocol_module
from __future__ import annotations

import os
from decimal import Decimal, InvalidOperation
from typing import Sequence

from gabion.invariants import never

_TRUTHY_VALUES = {"1", "true", "yes", "on"}
_FALSEY_VALUES = {"0", "false", "no", "off"}

LSP_TIMEOUT_ENV_KEYS: tuple[str, ...] = (
    "GABION_LSP_TIMEOUT_TICKS",
    "GABION_LSP_TIMEOUT_TICK_NS",
    "GABION_LSP_TIMEOUT_MS",
    "GABION_LSP_TIMEOUT_SECONDS",
)


def env_text(name: str, *, default: str = "") -> str:
    return os.getenv(name, default).strip()


def has_any_non_empty_env(keys: Sequence[str]) -> bool:
    return any(env_text(key) for key in keys)


def lsp_timeout_env_present() -> bool:
    return has_any_non_empty_env(LSP_TIMEOUT_ENV_KEYS)


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
    raw_ticks = env_text(ticks_key)
    raw_tick_ns = env_text(tick_ns_key)
    if raw_ticks:
        ticks = parse_positive_int_text(raw_ticks, field="ticks")
        if not raw_tick_ns:
            never("missing env timeout tick_ns", ticks=raw_ticks)
        tick_ns = parse_positive_int_text(raw_tick_ns, field="tick_ns")
        return ticks, tick_ns

    raw_ms = env_text(ms_key)
    if raw_ms:
        ms_value = parse_positive_int_text(raw_ms, field="ms")
        return ms_value, 1_000_000

    raw_seconds = env_text(seconds_key)
    if raw_seconds:
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
