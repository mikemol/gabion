# gabion:decision_protocol_module
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Mapping

from gabion.commands import boundary_order
from gabion.invariants import never
from gabion.order_contract import sort_once


def normalized_command_payload(
    *,
    command: str,
    arguments: list[object],
) -> tuple[list[object], dict[str, object]]:
    command_args = list(arguments)
    if not command_args:
        never("missing command payload arguments", command=command)
    payload_arg = command_args[0]
    if not isinstance(payload_arg, Mapping):
        never(
            "command payload must be a dict",
            command=command,
            payload_type=type(payload_arg).__name__,
        )
    payload = boundary_order.normalize_boundary_mapping_once(
        payload_arg,
        source=f"payload_codec.normalized_command_payload.{command}",
    )
    command_args[0] = payload
    return command_args, payload


def has_analysis_timeout(payload: Mapping[str, object]) -> bool:
    return any(
        payload.get(key) not in (None, "")
        for key in (
            "analysis_timeout_ticks",
            "analysis_timeout_tick_ns",
            "analysis_timeout_ms",
            "analysis_timeout_seconds",
        )
    )


def analysis_timeout_total_ns(
    payload: Mapping[str, object],
    *,
    source: str,
    reject_sub_millisecond_seconds: bool,
) -> int:
    timeout_ticks = payload.get("analysis_timeout_ticks")
    timeout_tick_ns = payload.get("analysis_timeout_tick_ns")
    timeout_ms = payload.get("analysis_timeout_ms")
    timeout_seconds = payload.get("analysis_timeout_seconds")

    if timeout_ticks not in (None, "") or timeout_tick_ns not in (None, ""):
        if timeout_ticks in (None, "") or timeout_tick_ns in (None, ""):
            never(
                "missing analysis timeout tick_ns",
                ticks=timeout_ticks,
                tick_ns=timeout_tick_ns,
            )
        ticks_value = _positive_int(timeout_ticks, field="analysis timeout ticks")
        tick_ns_value = _positive_int(timeout_tick_ns, field="analysis timeout tick_ns")
        return ticks_value * tick_ns_value

    if timeout_ms not in (None, ""):
        ms_value = _positive_int(timeout_ms, field="analysis timeout ms")
        return ms_value * 1_000_000

    if timeout_seconds not in (None, ""):
        try:
            seconds_value = Decimal(str(timeout_seconds))
        except (InvalidOperation, ValueError):
            never("invalid analysis timeout seconds", seconds=timeout_seconds)
        if seconds_value <= 0:
            never("invalid analysis timeout seconds", seconds=timeout_seconds)
        timeout_ns = int(seconds_value * Decimal(1_000_000_000))
        if reject_sub_millisecond_seconds and timeout_ns < 1_000_000:
            never("invalid analysis timeout seconds", seconds=timeout_seconds)
        return timeout_ns

    never(
        "missing analysis timeout",
        # Sort key is lexical payload-key text for stable diagnostics.
        payload_keys=sort_once(
            (str(key) for key in payload.keys()),
            source="src/gabion/commands/payload_codec.py:87",
        ),
    )


def analysis_timeout_total_ticks(
    payload: Mapping[str, object],
    *,
    source: str,
) -> int:
    timeout_ticks = payload.get("analysis_timeout_ticks")
    timeout_ms = payload.get("analysis_timeout_ms")
    timeout_seconds = payload.get("analysis_timeout_seconds")
    if timeout_ticks not in (None, ""):
        return _positive_int(timeout_ticks, field="analysis timeout ticks")
    if timeout_ms not in (None, ""):
        return _positive_int(timeout_ms, field="analysis timeout ms")
    if timeout_seconds not in (None, ""):
        try:
            seconds_value = Decimal(str(timeout_seconds))
        except (InvalidOperation, ValueError):
            never("invalid analysis timeout seconds", seconds=timeout_seconds)
        if seconds_value <= 0:
            never("invalid analysis timeout seconds", seconds=timeout_seconds)
        ticks_value = int(seconds_value * Decimal(1000))
        if ticks_value <= 0:
            never("invalid analysis timeout seconds", seconds=timeout_seconds)
        return ticks_value
    never(
        "missing analysis timeout",
        # Sort key is lexical payload-key text for stable diagnostics.
        payload_keys=sort_once(
            (str(key) for key in payload.keys()),
            source="src/gabion/commands/payload_codec.py:117",
        ),
    )


def _positive_int(value: object, *, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        never(f"invalid {field}", value=value)
    if parsed <= 0:
        never(f"invalid {field}", value=value)
    return parsed


def normalized_command_id_list(payload: Mapping[str, object], *, key: str) -> tuple[str, ...]:
    raw = payload.get(key)
    if raw is None:
        return ()
    if not isinstance(raw, list):
        never("invalid command id list", key=key, value_type=type(raw).__name__)
    normalized = [str(item) for item in raw]
    return tuple(sort_once(normalized, source=f"payload_codec.normalized_command_id_list.{key}"))
