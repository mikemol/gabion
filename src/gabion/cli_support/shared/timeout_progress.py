from __future__ import annotations

from functools import singledispatch
from typing import Mapping

from gabion.analysis.foundation.timeout_context import deadline_loop_iter
from gabion.invariants import never


def _none_optional_str(value: object) -> str | None:
    _ = value
    return None


@singledispatch
def _str_or_none(value: object) -> str | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_str_or_none.register
def _(value: str) -> str | None:
    return value


for _runtime_type in (int, float, bool, bytes, dict, list, tuple, set, frozenset, type(None)):
    _str_or_none.register(_runtime_type)(_none_optional_str)


def _none_optional_bool(value: object) -> bool | None:
    _ = value
    return None


@singledispatch
def _bool_or_none(value: object) -> bool | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_bool_or_none.register
def _(value: bool) -> bool | None:
    return value


for _runtime_type in (int, float, str, bytes, dict, list, tuple, set, frozenset, type(None)):
    _bool_or_none.register(_runtime_type)(_none_optional_bool)


def _none_optional_int(value: object) -> int | None:
    _ = value
    return None


@singledispatch
def _int_or_none(value: object) -> int | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_int_or_none.register
def _(value: int) -> int | None:
    return value


for _runtime_type in (float, str, bool, bytes, dict, list, tuple, set, frozenset, type(None)):
    _int_or_none.register(_runtime_type)(_none_optional_int)


def _none_optional_float(value: object) -> float | None:
    _ = value
    return None


@singledispatch
def _float_or_none(value: object) -> float | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_float_or_none.register
def _(value: float) -> float | None:
    return value


@_float_or_none.register
def _(value: int) -> float | None:
    return float(value)


@_float_or_none.register
def _(value: bool) -> float | None:
    return float(value)


for _runtime_type in (str, bytes, dict, list, tuple, set, frozenset, type(None)):
    _float_or_none.register(_runtime_type)(_none_optional_float)


def _none_optional_mapping(value: object) -> Mapping[str, object] | None:
    _ = value
    return None


@singledispatch
def _mapping_or_none(value: object) -> Mapping[str, object] | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_mapping_or_none.register
def _(value: dict) -> Mapping[str, object] | None:
    return value


for _runtime_type in (str, int, float, bool, bytes, list, tuple, set, frozenset, type(None)):
    _mapping_or_none.register(_runtime_type)(_none_optional_mapping)


def _none_optional_list(value: object) -> list[object] | None:
    _ = value
    return None


@singledispatch
def _list_or_none(value: object) -> list[object] | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_list_or_none.register
def _(value: list) -> list[object] | None:
    return value


for _runtime_type in (str, int, float, bool, bytes, dict, tuple, set, frozenset, type(None)):
    _list_or_none.register(_runtime_type)(_none_optional_list)


def render_timeout_progress_markdown(
    *,
    analysis_state: str | None,
    progress: Mapping[str, object],
    deadline_profile: Mapping[str, object] | None = None,
) -> str:
    lines = ["# Timeout Progress", ""]
    if analysis_state:
        lines.append(f"- `analysis_state`: `{analysis_state}`")
    classification = _str_or_none(progress.get("classification"))
    if classification is not None:
        lines.append(f"- `classification`: `{classification}`")
    retry_recommended = _bool_or_none(progress.get("retry_recommended"))
    if retry_recommended is not None:
        lines.append(f"- `retry_recommended`: `{retry_recommended}`")
    resume_supported = _bool_or_none(progress.get("resume_supported"))
    if resume_supported is not None:
        lines.append(f"- `resume_supported`: `{resume_supported}`")
    ticks_consumed = _int_or_none(progress.get("ticks_consumed"))
    if ticks_consumed is not None:
        lines.append(f"- `ticks_consumed`: `{ticks_consumed}`")
    tick_limit = _int_or_none(progress.get("tick_limit"))
    if tick_limit is not None:
        lines.append(f"- `tick_limit`: `{tick_limit}`")
    ticks_remaining = _int_or_none(progress.get("ticks_remaining"))
    if ticks_remaining is not None:
        lines.append(f"- `ticks_remaining`: `{ticks_remaining}`")
    progress_ticks_per_ns = _float_or_none(progress.get("ticks_per_ns"))
    profile_payload = _mapping_or_none(deadline_profile)
    profile_ticks_per_ns = (
        None if profile_payload is None else _float_or_none(profile_payload.get("ticks_per_ns"))
    )
    resolved_ticks_per_ns = (
        progress_ticks_per_ns if progress_ticks_per_ns is not None else profile_ticks_per_ns
    )
    if resolved_ticks_per_ns is not None:
        lines.append(f"- `ticks_per_ns`: `{float(resolved_ticks_per_ns):.9f}`")
    resume = _mapping_or_none(progress.get("resume"))
    if resume is not None:
        token = _mapping_or_none(resume.get("resume_token"))
        if token is not None:
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
    obligations = _list_or_none(progress.get("incremental_obligations"))
    if obligations:
        lines.append("")
        lines.append("## Incremental Obligations")
        lines.append("")
        for raw_entry in deadline_loop_iter(obligations):
            entry = _mapping_or_none(raw_entry)
            if entry is None:
                continue
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
