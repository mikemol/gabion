# gabion:decision_protocol_module
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import os
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import re
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

_DEFAULT_TIMEOUT_TICK_NS = 1_000_000
_DURATION_TOKEN_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>ns|us|ms|s|m|h)")
_DURATION_UNIT_NS: dict[str, Decimal] = {
    "ns": Decimal("1"),
    "us": Decimal("1000"),
    "ms": Decimal("1000000"),
    "s": Decimal("1000000000"),
    "m": Decimal("60000000000"),
    "h": Decimal("3600000000000"),
}


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
    return _LSP_TIMEOUT_OVERRIDE.get() is not None


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


def parse_duration_to_ns(duration: str, *, field_name: str = "timeout") -> int:
    text = str(duration).strip().lower()
    if not text:
        never(f"invalid {field_name} duration", duration=duration)
    idx = 0
    total_ns = Decimal("0")
    while idx < len(text):
        match = _DURATION_TOKEN_RE.match(text, idx)
        if match is None:
            never(f"invalid {field_name} duration", duration=duration)
        value_text = str(match.group("value"))
        unit = str(match.group("unit"))
        try:
            value = Decimal(value_text)
        except (InvalidOperation, ValueError):
            never(f"invalid {field_name} duration", duration=duration)
        if value <= 0:
            never(f"invalid {field_name} duration", duration=duration)
        unit_factor = _DURATION_UNIT_NS.get(unit)
        if unit_factor is None:
            never(f"invalid {field_name} duration", duration=duration)
        total_ns += value * unit_factor
        idx = match.end()
    if total_ns <= 0:
        never(f"invalid {field_name} duration", duration=duration)
    total_ns_int = int(total_ns.to_integral_value(rounding=ROUND_CEILING))
    if total_ns_int <= 0:
        never(f"invalid {field_name} duration", duration=duration)
    return total_ns_int


def timeout_config_from_duration(duration: str) -> LspTimeoutConfig:
    total_ns = parse_duration_to_ns(duration)
    tick_ns = _DEFAULT_TIMEOUT_TICK_NS
    ticks = (total_ns + tick_ns - 1) // tick_ns
    return LspTimeoutConfig(ticks=ticks, tick_ns=tick_ns)


def duration_text_from_ticks(*, ticks: int, tick_ns: int) -> str:
    ticks_value = int(ticks)
    tick_ns_value = int(tick_ns)
    if ticks_value <= 0:
        never("invalid timeout ticks", ticks=ticks)
    if tick_ns_value <= 0:
        never("invalid timeout tick_ns", tick_ns=tick_ns)
    total_ns = ticks_value * tick_ns_value
    if total_ns <= 0:
        never("invalid timeout duration", ticks=ticks, tick_ns=tick_ns)
    return f"{total_ns}ns"


def apply_cli_timeout_flag(*, timeout: str | None = None) -> None:
    timeout_text = str(timeout).strip() if isinstance(timeout, str) else ""
    if not timeout_text:
        set_lsp_timeout_override(None)
        return
    set_lsp_timeout_override(timeout_config_from_duration(timeout_text))


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
    del ticks_key, tick_ns_key, ms_key, seconds_key
    override = lsp_timeout_override()
    if override is not None:
        return (int(override.ticks), int(override.tick_ns))
    if has_any_non_empty_env(LSP_TIMEOUT_ENV_KEYS):
        never(
            "legacy timeout env overrides removed; use --timeout or runtime override scope",
            keys=LSP_TIMEOUT_ENV_KEYS,
        )
    never("missing timeout override configuration")
