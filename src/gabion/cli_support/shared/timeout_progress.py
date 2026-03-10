from __future__ import annotations

from functools import singledispatch
from typing import Mapping

from gabion.analysis.foundation.timeout_context import deadline_loop_iter
from gabion.invariants import never


def _empty_str_stream(value: object) -> tuple[str, ...]:
    _ = value
    return ()


@singledispatch
def _str_stream(value: object) -> tuple[str, ...]:
    never("unregistered runtime type", value_type=type(value).__name__)


@_str_stream.register
def _sd_reg_1(value: str) -> tuple[str, ...]:
    return (value,)


for _runtime_type in (int, float, bool, bytes, dict, list, tuple, set, frozenset, type(None)):
    _str_stream.register(_runtime_type)(_empty_str_stream)


def _empty_bool_stream(value: object) -> tuple[bool, ...]:
    _ = value
    return ()


@singledispatch
def _bool_stream(value: object) -> tuple[bool, ...]:
    never("unregistered runtime type", value_type=type(value).__name__)


@_bool_stream.register
def _sd_reg_2(value: bool) -> tuple[bool, ...]:
    return (value,)


for _runtime_type in (int, float, str, bytes, dict, list, tuple, set, frozenset, type(None)):
    _bool_stream.register(_runtime_type)(_empty_bool_stream)


def _empty_int_stream(value: object) -> tuple[int, ...]:
    _ = value
    return ()


@singledispatch
def _int_stream(value: object) -> tuple[int, ...]:
    never("unregistered runtime type", value_type=type(value).__name__)


@_int_stream.register
def _sd_reg_3(value: int) -> tuple[int, ...]:
    return (value,)


for _runtime_type in (float, str, bool, bytes, dict, list, tuple, set, frozenset, type(None)):
    _int_stream.register(_runtime_type)(_empty_int_stream)


def _empty_float_stream(value: object) -> tuple[float, ...]:
    _ = value
    return ()


@singledispatch
def _float_stream(value: object) -> tuple[float, ...]:
    never("unregistered runtime type", value_type=type(value).__name__)


@_float_stream.register
def _sd_reg_4(value: float) -> tuple[float, ...]:
    return (value,)


@_float_stream.register
def _sd_reg_5(value: int) -> tuple[float, ...]:
    return (float(value),)


@_float_stream.register
def _sd_reg_6(value: bool) -> tuple[float, ...]:
    return (float(value),)


for _runtime_type in (str, bytes, dict, list, tuple, set, frozenset, type(None)):
    _float_stream.register(_runtime_type)(_empty_float_stream)


def _empty_mapping_stream(value: object) -> tuple[Mapping[str, object], ...]:
    _ = value
    return ()


@singledispatch
def _mapping_stream(value: object) -> tuple[Mapping[str, object], ...]:
    never("unregistered runtime type", value_type=type(value).__name__)


@_mapping_stream.register
def _sd_reg_7(value: dict) -> tuple[Mapping[str, object], ...]:
    return (value,)


for _runtime_type in (str, int, float, bool, bytes, list, tuple, set, frozenset, type(None)):
    _mapping_stream.register(_runtime_type)(_empty_mapping_stream)


def _empty_list_stream(value: object) -> tuple[list[object], ...]:
    _ = value
    return ()


@singledispatch
def _list_stream(value: object) -> tuple[list[object], ...]:
    never("unregistered runtime type", value_type=type(value).__name__)


@_list_stream.register
def _sd_reg_8(value: list) -> tuple[list[object], ...]:
    return (value,)


for _runtime_type in (str, int, float, bool, bytes, dict, tuple, set, frozenset, type(None)):
    _list_stream.register(_runtime_type)(_empty_list_stream)


def render_timeout_progress_markdown(
    *,
    analysis_state: str | None,
    progress: Mapping[str, object],
    deadline_profile: Mapping[str, object] | None = None,
) -> str:
    lines = ["# Timeout Progress", ""]
    if analysis_state:
        lines.append(f"- `analysis_state`: `{analysis_state}`")
    for classification in _str_stream(progress.get("classification")):
        lines.append(f"- `classification`: `{classification}`")
    for retry_recommended in _bool_stream(progress.get("retry_recommended")):
        lines.append(f"- `retry_recommended`: `{retry_recommended}`")
    for resume_supported in _bool_stream(progress.get("resume_supported")):
        lines.append(f"- `resume_supported`: `{resume_supported}`")
    for ticks_consumed in _int_stream(progress.get("ticks_consumed")):
        lines.append(f"- `ticks_consumed`: `{ticks_consumed}`")
    for tick_limit in _int_stream(progress.get("tick_limit")):
        lines.append(f"- `tick_limit`: `{tick_limit}`")
    for ticks_remaining in _int_stream(progress.get("ticks_remaining")):
        lines.append(f"- `ticks_remaining`: `{ticks_remaining}`")
    ticks_per_ns_values = list(_float_stream(progress.get("ticks_per_ns")))
    if not ticks_per_ns_values:
        for profile_payload in _mapping_stream(deadline_profile):
            ticks_per_ns_values.extend(_float_stream(profile_payload.get("ticks_per_ns")))
    if ticks_per_ns_values:
        lines.append(f"- `ticks_per_ns`: `{ticks_per_ns_values[0]:.9f}`")
    for resume in _mapping_stream(progress.get("resume")):
        for token in _mapping_stream(resume.get("resume_token")):
            lines.append("")
            lines.append("## Resume Token")
            lines.append("")
            for key in deadline_loop_iter(
                (
                    "phase",
                    "checkpoint_path",
                    "completed_files",
                    "remaining_files",
                    "total_files",
                    "witness_digest",
                )
            ):
                value = token.get(key)
                if value is None:
                    continue
                lines.append(f"- `{key}`: `{value}`")
    obligations: list[object] = []
    for raw_obligations in _list_stream(progress.get("incremental_obligations")):
        obligations = raw_obligations
    if obligations:
        lines.append("")
        lines.append("## Incremental Obligations")
        lines.append("")
        for raw_entry in deadline_loop_iter(obligations):
            for entry in _mapping_stream(raw_entry):
                status = str(entry.get("status", "UNKNOWN") or "UNKNOWN")
                contract = str(entry.get("contract", "") or "")
                kind = str(entry.get("kind", "") or "")
                detail = str(entry.get("detail", "") or "")
                section_id = str(entry.get("section_id", "") or "")
                section_suffix = f" section={section_id}" if section_id else ""
                lines.append(
                    f"- `{status}` `{contract}` `{kind}`{section_suffix}: {detail}"
                )
    return "\n".join(lines)
