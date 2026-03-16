from __future__ import annotations

from typing import Mapping

from gabion.commands.progress_transition import (
    normalize_progress_transition_from_phase_progress, transition_reason_from_phase_progress)
from gabion.order_contract import sort_once
from gabion.runtime.coercion_contract import (
    INT_LIKE_OPTIONAL_POLICY,
    MAPPING_OPTIONAL_POLICY,
    NON_BOOL_FLOAT_OPTIONAL_POLICY,
    NON_BOOL_INT_OPTIONAL_POLICY,
    ROW_FLOAT_OPTIONAL_POLICY,
    STRING_KEY_DICT_OPTIONAL_POLICY,
    STR_OPTIONAL_POLICY,
)
from gabion.schema import CanonicalProgressEventPayloadDTO

LSP_PROGRESS_NOTIFICATION_METHOD = "$/progress"
LSP_PROGRESS_TOKEN_V2 = "gabion.dataflowAudit/progress-v2"
LSP_PROGRESS_TOKEN = LSP_PROGRESS_TOKEN_V2
CANONICAL_PROGRESS_EVENT_SCHEMA_V2 = "gabion/canonical_progress_event_v2"
DEFAULT_TIMELINE_MIN_INTERVAL_SECONDS = 1.0

_PHASE_TIMELINE_COLUMNS: tuple[str, ...] = (
    "ts_utc",
    "event_seq",
    "event_kind",
    "phase",
    "analysis_state",
    "classification",
    "progress_marker",
    "primary",
    "files",
    "stale_for_s",
    "dimensions",
    "progress_path",
    "active_primary",
    "active_depth",
    "transition_reason",
    "root_identity",
    "active_identity",
    "marker_family",
    "marker_step",
    "active_children",
)


def _mapping_optional(value: object) -> dict[str, object] | None:
    return MAPPING_OPTIONAL_POLICY(value)


def _str_optional(value: object) -> str | None:
    return STR_OPTIONAL_POLICY(value)


def _int_non_bool_optional(value: object) -> int | None:
    return NON_BOOL_INT_OPTIONAL_POLICY(value)


def _int_like_optional(value: object) -> int | bool | None:
    return INT_LIKE_OPTIONAL_POLICY(value)


def _float_non_bool_optional(value: object) -> float | None:
    return NON_BOOL_FLOAT_OPTIONAL_POLICY(value)


def _float_row_optional(value: object) -> float | None:
    return ROW_FLOAT_OPTIONAL_POLICY(value)


def _str_key_dict_optional(value: object) -> dict[str, object] | None:
    return STRING_KEY_DICT_OPTIONAL_POLICY(value)


def _text(value: object) -> str:
    return str(value or "")


def _non_negative_optional(value: int | None) -> int | None:
    return max(value, 0) if value is not None else None


def _coerced_non_negative_int_like_optional(value: object) -> int | None:
    raw = _int_like_optional(value)
    if raw is None:
        return None
    return max(int(raw), 0)


def _clamped_done_total(
    *,
    done: int | None,
    total: int | None,
) -> tuple[int | None, int | None]:
    if done is None or total is None or total <= 0:
        return done, total
    return min(done, total), total


def _normalize_transition(
    phase_progress: Mapping[str, object],
):
    return normalize_progress_transition_from_phase_progress(phase_progress)


def _progress_marker_from_transition(
    phase_progress: Mapping[str, object],
    *,
    transition=None,
) -> str:
    if transition is None:
        transition = _normalize_transition(phase_progress)
    if transition is not None and transition.marker.marker_text:
        return transition.marker.marker_text
    return str(phase_progress.get("progress_marker", "") or "")


def _primary_from_transition(
    phase_progress: Mapping[str, object],
    *,
    transition=None,
) -> tuple[str, int | None, int | None]:
    if transition is None:
        transition = _normalize_transition(phase_progress)
    if transition is None:
        return "", None, None
    return transition.primary_unit, transition.primary_done, transition.primary_total


# gabion:boundary_normalization
def _path_from_transition(*, transition) -> str:
    if transition is None:
        return ""
    return " > ".join(transition.active_path)


# gabion:boundary_normalization
def _active_primary_from_transition(*, transition) -> str:
    if transition is None:
        return ""
    active = transition.active_node
    value = f"{active.done}/{active.total}"
    if active.unit:
        value = f"{value} {active.unit}"
    return value


# gabion:boundary_normalization
def _active_depth_from_transition(*, transition) -> str:
    if transition is None:
        return ""
    return str(max(len(transition.active_path) - 1, 0))


# gabion:boundary_normalization
def _root_identity_from_transition(*, transition) -> str:
    if transition is None:
        return ""
    return transition.root.identity


# gabion:boundary_normalization
def _active_identity_from_transition(*, transition) -> str:
    if transition is None:
        return ""
    return transition.active_node.identity


def _marker_family_step_from_text(marker_text: str) -> tuple[str, str]:
    if ":" not in marker_text:
        if marker_text == "complete":
            return "complete", ""
        return marker_text, ""
    marker_family, marker_step = marker_text.split(":", 1)
    return marker_family, marker_step


def _active_marker_family_from_transition(
    *,
    transition,
    marker_text: str,
) -> str:
    _ = transition
    marker_family, _ = _marker_family_step_from_text(marker_text)
    return marker_family


def _active_marker_step_from_transition(
    *,
    transition,
    marker_text: str,
) -> str:
    _ = transition
    _, marker_step = _marker_family_step_from_text(marker_text)
    return marker_step


# gabion:boundary_normalization
def _active_children_from_transition(*, transition) -> str:
    if transition is None:
        return ""
    return str(len(transition.active_node.children))


def _progress_value_from_notification(
    notification: Mapping[str, object],
    *,
    method: str = LSP_PROGRESS_NOTIFICATION_METHOD,
    token: str = LSP_PROGRESS_TOKEN,
) -> Mapping[str, object] | None:
    if _text(notification.get("method", "")) != method:
        return None
    params = _mapping_optional(notification.get("params"))
    if params is None:
        return None
    if _text(params.get("token", "")) != token:
        return None
    value = _mapping_optional(params.get("value"))
    if value is None:
        return None
    return value


# gabion:boundary_normalization
def _canonical_progress_value_from_notification(
    notification: Mapping[str, object],
) -> Mapping[str, object] | None:
    value = _progress_value_from_notification(
        notification,
        token=LSP_PROGRESS_TOKEN_V2,
    )
    if value is None:
        return None
    if _text(value.get("schema", "")) != CANONICAL_PROGRESS_EVENT_SCHEMA_V2:
        return None
    try:
        return CanonicalProgressEventPayloadDTO.model_validate(value).model_dump(
            by_alias=True
        )
    except ValueError:
        return None


def phase_timeline_header_columns() -> list[str]:
    return list(_PHASE_TIMELINE_COLUMNS)


def phase_timeline_header_block() -> str:
    header = phase_timeline_header_columns()
    header_line = "| " + " | ".join(header) + " |"
    separator_line = "| " + " | ".join(["---"] * len(header)) + " |"
    return header_line + "\n" + separator_line


def phase_progress_dimensions_summary(
    phase_progress_v2: Mapping[str, object] | None,
) -> str:
    normalized_phase_progress_v2 = _mapping_optional(phase_progress_v2)
    if normalized_phase_progress_v2 is None:
        return ""
    raw_dimensions = _mapping_optional(normalized_phase_progress_v2.get("dimensions"))
    if raw_dimensions is None:
        return ""
    fragments: list[str] = []
    dimension_names = [
        dim_name
        for raw_name in raw_dimensions
        for dim_name in [_str_optional(raw_name)]
        if dim_name is not None
    ]
    for dim_name in sort_once(
        dimension_names,
        source="phase_progress_dimensions_summary.dim_names",
    ):
        raw_payload = _mapping_optional(raw_dimensions.get(dim_name))
        if raw_payload is None:
            continue
        done = _non_negative_optional(_int_non_bool_optional(raw_payload.get("done")))
        total = _non_negative_optional(_int_non_bool_optional(raw_payload.get("total")))
        if done is None or total is None:
            continue
        if total:
            done = min(done, total)
        fragments.append(f"{dim_name}={done}/{total}")
    return "; ".join(fragments)


def phase_timeline_row_from_phase_progress(phase_progress: Mapping[str, object]) -> str:
    ts_utc = _text(phase_progress.get("ts_utc", ""))
    event_seq = phase_progress.get("event_seq")
    event_kind = _text(phase_progress.get("event_kind", ""))
    transition = _normalize_transition(phase_progress)
    if transition is not None:
        event_kind = transition.event_kind
    phase = _text(phase_progress.get("phase", ""))
    analysis_state = _text(phase_progress.get("analysis_state", ""))
    classification = _text(phase_progress.get("classification", ""))
    progress_marker = _progress_marker_from_transition(
        phase_progress,
        transition=transition,
    )
    phase_progress_v2 = _mapping_optional(phase_progress.get("phase_progress_v2"))
    primary_unit, primary_done, primary_total = _primary_from_transition(
        phase_progress,
        transition=transition,
    )
    if phase_progress_v2 is not None:
        if not primary_unit:
            primary_unit = _text(phase_progress_v2.get("primary_unit", ""))
        raw_primary_done = _non_negative_optional(
            _int_non_bool_optional(phase_progress_v2.get("primary_done"))
        )
        raw_primary_total = _non_negative_optional(
            _int_non_bool_optional(phase_progress_v2.get("primary_total"))
        )
        if primary_done is None and raw_primary_done is not None:
            primary_done = raw_primary_done
        if primary_total is None and raw_primary_total is not None:
            primary_total = raw_primary_total
        primary_done, primary_total = _clamped_done_total(
            done=primary_done,
            total=primary_total,
        )
    if primary_done is None or primary_total is None:
        raw_work_done = _coerced_non_negative_int_like_optional(
            phase_progress.get("work_done")
        )
        raw_work_total = _coerced_non_negative_int_like_optional(
            phase_progress.get("work_total")
        )
        if raw_work_done is not None and raw_work_total is not None:
            primary_done = raw_work_done
            primary_total = raw_work_total
            if primary_total:
                primary_done = min(primary_done, primary_total)
    primary = ""
    if primary_done is not None and primary_total is not None:
        primary = f"{primary_done}/{primary_total}"
        if primary_unit:
            primary = f"{primary} {primary_unit}"
    elif primary_unit:
        primary = primary_unit
    completed_files = phase_progress.get("completed_files")
    remaining_files = phase_progress.get("remaining_files")
    total_files = phase_progress.get("total_files")
    files = ""
    normalized_completed_files = _int_like_optional(completed_files)
    normalized_remaining_files = _int_like_optional(remaining_files)
    normalized_total_files = _int_like_optional(total_files)
    if (
        normalized_completed_files is not None
        and normalized_remaining_files is not None
        and normalized_total_files is not None
    ):
        files = f"{completed_files}/{total_files} rem={remaining_files}"
    raw_stale_for_s = _float_row_optional(phase_progress.get("stale_for_s"))
    stale_for_s = f"{raw_stale_for_s:.1f}" if raw_stale_for_s is not None else ""
    dimensions = phase_progress_dimensions_summary(phase_progress_v2)
    progress_path = _path_from_transition(transition=transition)
    active_primary = _active_primary_from_transition(transition=transition)
    active_depth = _active_depth_from_transition(transition=transition)
    transition_reason = transition_reason_from_phase_progress(phase_progress) or ""
    root_identity = _root_identity_from_transition(transition=transition)
    active_identity = _active_identity_from_transition(transition=transition)
    marker_family = _active_marker_family_from_transition(
        transition=transition,
        marker_text=progress_marker,
    )
    marker_step = _active_marker_step_from_transition(
        transition=transition,
        marker_text=progress_marker,
    )
    active_children = _active_children_from_transition(transition=transition)
    row = [
        ts_utc,
        _int_like_optional(event_seq) if _int_like_optional(event_seq) is not None else "",
        event_kind,
        phase,
        analysis_state,
        classification,
        progress_marker,
        primary,
        files,
        stale_for_s,
        dimensions,
        progress_path,
        active_primary,
        active_depth,
        transition_reason,
        root_identity,
        active_identity,
        marker_family,
        marker_step,
        active_children,
    ]
    return "| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |"


def phase_progress_from_progress_notification(
    notification: Mapping[str, object],
) -> dict[str, object] | None:
    canonical_value = _canonical_progress_value_from_notification(notification)
    if canonical_value is None:
        return None
    adaptation_kind = _text(canonical_value.get("adaptation_kind", "")).strip()
    if adaptation_kind == "valid":
        event_value = _mapping_optional(canonical_value.get("event"))
        if event_value is None:
            return None
        payload = _mapping_optional(event_value.get("payload"))
        if payload is None:
            return None
        return _phase_progress_from_progress_value(payload)
    if adaptation_kind == "rejected":
        rejected_payload = _mapping_optional(
            canonical_value.get("rejected_progress_payload_v2")
        )
        if rejected_payload is None:
            return None
        return _phase_progress_from_progress_value(rejected_payload)
    return None


def _phase_progress_from_progress_value(
    value: Mapping[str, object],
) -> dict[str, object] | None:
    phase = _text(value.get("phase", ""))
    if not phase:
        return None
    work_done = _int_non_bool_optional(value.get("work_done"))
    work_total = _int_non_bool_optional(value.get("work_total"))
    completed_files = _int_non_bool_optional(value.get("completed_files"))
    remaining_files = _int_non_bool_optional(value.get("remaining_files"))
    total_files = _int_non_bool_optional(value.get("total_files"))
    analysis_state = _text(value.get("analysis_state", ""))
    classification = _text(value.get("classification", ""))
    event_kind = _text(value.get("event_kind", ""))
    progress_marker = _text(value.get("progress_marker", ""))
    event_seq = _int_non_bool_optional(value.get("event_seq"))
    stale_for_s = _float_non_bool_optional(value.get("stale_for_s"))
    normalized_phase_progress_v2 = _str_key_dict_optional(value.get("phase_progress_v2"))
    normalized_progress_transition_v2 = _str_key_dict_optional(
        value.get("progress_transition_v2")
    )
    transition_progress_marker = progress_marker
    transition_done: int | None = None
    transition_total: int | None = None
    transition_event_kind = event_kind
    if normalized_progress_transition_v2 is not None:
        transition_phase_progress: dict[str, object] = {
            "phase": phase,
            "analysis_state": analysis_state,
            "event_kind": event_kind,
            "progress_marker": progress_marker,
            "phase_progress_v2": normalized_phase_progress_v2,
            "work_done": work_done,
            "work_total": work_total,
        }
        transition_phase_progress["progress_transition_v2"] = normalized_progress_transition_v2
        normalized_transition = normalize_progress_transition_from_phase_progress(
            transition_phase_progress
        )
        if normalized_transition is not None:
            transition_progress_marker = normalized_transition.marker.marker_text
            transition_done = normalized_transition.primary_done
            transition_total = normalized_transition.primary_total
            transition_event_kind = normalized_transition.event_kind
    if transition_done is not None:
        work_done = transition_done
    if transition_total is not None:
        work_total = transition_total
    phase_timeline_header = value.get("phase_timeline_header")
    phase_timeline_row = value.get("phase_timeline_row")
    done = bool(value.get("done", False))
    normalized: dict[str, object] = {
        "phase": phase,
        "work_done": work_done,
        "work_total": work_total,
        "completed_files": completed_files,
        "remaining_files": remaining_files,
        "total_files": total_files,
        "analysis_state": analysis_state,
        "classification": classification,
        "event_kind": transition_event_kind,
        "event_seq": event_seq,
        "ts_utc": _text(value.get("ts_utc", "")),
        "stale_for_s": stale_for_s,
        "phase_progress_v2": normalized_phase_progress_v2,
        "progress_marker": transition_progress_marker,
        "phase_timeline_header": _str_optional(phase_timeline_header) or "",
        "phase_timeline_row": _str_optional(phase_timeline_row) or "",
        "done": done,
    }
    if normalized_progress_transition_v2 is not None:
        normalized["progress_transition_v2"] = normalized_progress_transition_v2
    return normalized


def phase_progress_signature(phase_progress: Mapping[str, object]) -> tuple[object, ...]:
    transition_reason = transition_reason_from_phase_progress(phase_progress)
    transition_marker = _progress_marker_from_transition(phase_progress)
    transition_unit, transition_done, transition_total = _primary_from_transition(
        phase_progress
    )
    transition = _normalize_transition(phase_progress)
    transition_active_path: tuple[str, ...] = ()
    transition_root_identity: str | None = None
    transition_active_identity: str | None = None
    transition_marker_family, transition_marker_step = _marker_family_step_from_text(
        transition_marker
    )
    if transition is not None:
        transition_active_path = transition.active_path
        transition_root_identity = transition.root.identity
        transition_active_identity = transition.active_node.identity
    return (
        phase_progress.get("phase"),
        phase_progress.get("analysis_state"),
        phase_progress.get("classification"),
        phase_progress.get("event_kind"),
        phase_progress.get("event_seq"),
        phase_progress.get("work_done"),
        phase_progress.get("work_total"),
        phase_progress.get("completed_files"),
        phase_progress.get("remaining_files"),
        phase_progress.get("total_files"),
        phase_progress.get("stale_for_s"),
        transition_marker,
        transition_marker_family,
        transition_marker_step,
        transition_root_identity,
        transition_active_identity,
        transition_active_path,
        transition_unit,
        transition_done,
        transition_total,
        transition_reason,
        phase_progress.get("done"),
    )


def phase_timeline_from_phase_progress(
    phase_progress: Mapping[str, object],
) -> dict[str, str]:
    header_value = _str_optional(phase_progress.get("phase_timeline_header"))
    row_value = _str_optional(phase_progress.get("phase_timeline_row"))
    header = header_value if header_value else phase_timeline_header_block()
    row = row_value if row_value else phase_timeline_row_from_phase_progress(phase_progress)
    return {"header": header, "row": row}


def phase_timeline_from_progress_notification(
    notification: Mapping[str, object],
) -> dict[str, str] | None:
    phase_progress = phase_progress_from_progress_notification(notification)
    if phase_progress is None:
        return None
    return phase_timeline_from_phase_progress(phase_progress)


def is_heartbeat_progress(phase_progress: Mapping[str, object]) -> bool:
    return str(phase_progress.get("event_kind", "") or "") == "heartbeat"


def phase_progress_emit_due(
    *,
    phase_progress: Mapping[str, object],
    timeline_header_emitted: bool,
    last_emitted_phase: str | None,
    last_emitted_monotonic: float | None,
    now_monotonic: float,
    min_interval_seconds: float = DEFAULT_TIMELINE_MIN_INTERVAL_SECONDS,
) -> bool:
    phase = _text(phase_progress.get("phase", ""))
    event_kind = _text(phase_progress.get("event_kind", ""))
    normalized_last_emitted_phase = _str_optional(last_emitted_phase)
    done = phase_progress.get("done") is True
    force_emit = (
        not timeline_header_emitted
        or done
        or event_kind in {"terminal", "checkpoint"}
        or (
            normalized_last_emitted_phase is not None
            and bool(phase)
            and phase != normalized_last_emitted_phase
        )
    )
    if force_emit:
        return True
    if last_emitted_monotonic is None:
        return True
    return (now_monotonic - last_emitted_monotonic) >= max(
        float(min_interval_seconds),
        0.0,
    )
