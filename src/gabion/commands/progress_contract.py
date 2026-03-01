# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from typing import Mapping

from gabion.commands.progress_transition import (
    normalize_progress_transition_from_phase_progress,
    transition_reason_from_phase_progress,
)
from gabion.order_contract import sort_once

LSP_PROGRESS_NOTIFICATION_METHOD = "$/progress"
LSP_PROGRESS_TOKEN = "gabion.dataflowAudit/progress-v1"
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


def _path_from_transition(*, transition) -> str:
    if transition is None:
        return ""
    return " > ".join(transition.active_path)


def _active_primary_from_transition(*, transition) -> str:
    if transition is None:
        return ""
    active = transition.active_node
    value = f"{active.done}/{active.total}"
    if active.unit:
        value = f"{value} {active.unit}"
    return value


def _active_depth_from_transition(*, transition) -> str:
    if transition is None:
        return ""
    return str(max(len(transition.active_path) - 1, 0))


def _root_identity_from_transition(*, transition) -> str:
    if transition is None:
        return ""
    return transition.root.identity


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
    if str(notification.get("method", "") or "") != method:
        return None
    params = notification.get("params")
    if not isinstance(params, Mapping):
        return None
    if str(params.get("token", "") or "") != token:
        return None
    value = params.get("value")
    if not isinstance(value, Mapping):
        return None
    return value


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
    if not isinstance(phase_progress_v2, Mapping):
        return ""
    raw_dimensions = phase_progress_v2.get("dimensions")
    if not isinstance(raw_dimensions, Mapping):
        return ""
    fragments: list[str] = []
    for dim_name in sort_once(
        (name for name in raw_dimensions if isinstance(name, str)),
        source="phase_progress_dimensions_summary.dim_names",
    ):
        raw_payload = raw_dimensions.get(dim_name)
        if not isinstance(raw_payload, Mapping):
            continue
        raw_done = raw_payload.get("done")
        raw_total = raw_payload.get("total")
        if (
            isinstance(raw_done, int)
            and not isinstance(raw_done, bool)
            and isinstance(raw_total, int)
            and not isinstance(raw_total, bool)
        ):
            done = max(int(raw_done), 0)
            total = max(int(raw_total), 0)
            if total:
                done = min(done, total)
            fragments.append(f"{dim_name}={done}/{total}")
    return "; ".join(fragments)


def phase_timeline_row_from_phase_progress(phase_progress: Mapping[str, object]) -> str:
    ts_utc = str(phase_progress.get("ts_utc", "") or "")
    event_seq = phase_progress.get("event_seq")
    event_kind = str(phase_progress.get("event_kind", "") or "")
    transition = _normalize_transition(phase_progress)
    if transition is not None:
        event_kind = transition.event_kind
    phase = str(phase_progress.get("phase", "") or "")
    analysis_state = str(phase_progress.get("analysis_state", "") or "")
    classification = str(phase_progress.get("classification", "") or "")
    progress_marker = _progress_marker_from_transition(
        phase_progress,
        transition=transition,
    )
    phase_progress_v2 = (
        phase_progress.get("phase_progress_v2")
        if isinstance(phase_progress.get("phase_progress_v2"), Mapping)
        else None
    )
    primary_unit, primary_done, primary_total = _primary_from_transition(
        phase_progress,
        transition=transition,
    )
    if isinstance(phase_progress_v2, Mapping):
        if not primary_unit:
            primary_unit = str(phase_progress_v2.get("primary_unit", "") or "")
        raw_primary_done = phase_progress_v2.get("primary_done")
        raw_primary_total = phase_progress_v2.get("primary_total")
        if (
            primary_done is None
            and isinstance(raw_primary_done, int)
            and not isinstance(raw_primary_done, bool)
        ):
            primary_done = max(int(raw_primary_done), 0)
        if (
            primary_total is None
            and isinstance(raw_primary_total, int)
            and not isinstance(raw_primary_total, bool)
        ):
            primary_total = max(int(raw_primary_total), 0)
        if (
            primary_done is not None
            and primary_total is not None
            and primary_total > 0
            and primary_done > primary_total
        ):
            primary_done = primary_total
    if primary_done is None or primary_total is None:
        raw_work_done = phase_progress.get("work_done")
        raw_work_total = phase_progress.get("work_total")
        if isinstance(raw_work_done, int) and isinstance(raw_work_total, int):
            primary_done = max(int(raw_work_done), 0)
            primary_total = max(int(raw_work_total), 0)
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
    if (
        isinstance(completed_files, int)
        and isinstance(remaining_files, int)
        and isinstance(total_files, int)
    ):
        files = f"{completed_files}/{total_files} rem={remaining_files}"
    raw_stale_for_s = phase_progress.get("stale_for_s")
    stale_for_s = (
        f"{float(raw_stale_for_s):.1f}"
        if isinstance(raw_stale_for_s, (int, float))
        else ""
    )
    dimensions = phase_progress_dimensions_summary(
        phase_progress_v2 if isinstance(phase_progress_v2, Mapping) else None
    )
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
        event_seq if isinstance(event_seq, int) else "",
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
    value = _progress_value_from_notification(notification)
    if not isinstance(value, Mapping):
        return None
    phase = str(value.get("phase", "") or "")
    if not phase:
        return None
    raw_work_done = value.get("work_done")
    work_done = (
        int(raw_work_done)
        if isinstance(raw_work_done, int) and not isinstance(raw_work_done, bool)
        else None
    )
    raw_work_total = value.get("work_total")
    work_total = (
        int(raw_work_total)
        if isinstance(raw_work_total, int) and not isinstance(raw_work_total, bool)
        else None
    )
    raw_completed_files = value.get("completed_files")
    completed_files = (
        int(raw_completed_files)
        if isinstance(raw_completed_files, int) and not isinstance(raw_completed_files, bool)
        else None
    )
    raw_remaining_files = value.get("remaining_files")
    remaining_files = (
        int(raw_remaining_files)
        if isinstance(raw_remaining_files, int) and not isinstance(raw_remaining_files, bool)
        else None
    )
    raw_total_files = value.get("total_files")
    total_files = (
        int(raw_total_files)
        if isinstance(raw_total_files, int) and not isinstance(raw_total_files, bool)
        else None
    )
    analysis_state = str(value.get("analysis_state", "") or "")
    classification = str(value.get("classification", "") or "")
    event_kind = str(value.get("event_kind", "") or "")
    progress_marker = str(value.get("progress_marker", "") or "")
    raw_event_seq = value.get("event_seq")
    event_seq = (
        int(raw_event_seq)
        if isinstance(raw_event_seq, int) and not isinstance(raw_event_seq, bool)
        else None
    )
    raw_stale_for_s = value.get("stale_for_s")
    stale_for_s = (
        float(raw_stale_for_s)
        if isinstance(raw_stale_for_s, (int, float)) and not isinstance(raw_stale_for_s, bool)
        else None
    )
    phase_progress_v2 = value.get("phase_progress_v2")
    normalized_phase_progress_v2 = (
        {str(key): phase_progress_v2[key] for key in phase_progress_v2}
        if isinstance(phase_progress_v2, Mapping)
        else None
    )
    progress_transition_v2 = value.get("progress_transition_v2")
    normalized_progress_transition_v2 = (
        {str(key): progress_transition_v2[key] for key in progress_transition_v2}
        if isinstance(progress_transition_v2, Mapping)
        else None
    )
    progress_transition_v1 = value.get("progress_transition_v1")
    normalized_progress_transition_v1 = (
        {str(key): progress_transition_v1[key] for key in progress_transition_v1}
        if isinstance(progress_transition_v1, Mapping)
        else None
    )
    transition_progress_marker = progress_marker
    transition_done: int | None = None
    transition_total: int | None = None
    transition_event_kind = event_kind
    if isinstance(normalized_progress_transition_v1, Mapping) or isinstance(
        normalized_progress_transition_v2, Mapping
    ):
        transition_phase_progress: dict[str, object] = {
            "phase": phase,
            "analysis_state": analysis_state,
            "event_kind": event_kind,
            "progress_marker": progress_marker,
            "phase_progress_v2": normalized_phase_progress_v2,
            "work_done": work_done,
            "work_total": work_total,
        }
        if isinstance(normalized_progress_transition_v2, Mapping):
            transition_phase_progress["progress_transition_v2"] = normalized_progress_transition_v2
        if isinstance(normalized_progress_transition_v1, Mapping):
            transition_phase_progress["progress_transition_v1"] = normalized_progress_transition_v1
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
        "ts_utc": str(value.get("ts_utc", "") or ""),
        "stale_for_s": stale_for_s,
        "phase_progress_v2": normalized_phase_progress_v2,
        "progress_marker": transition_progress_marker,
        "phase_timeline_header": (
            phase_timeline_header if isinstance(phase_timeline_header, str) else ""
        ),
        "phase_timeline_row": (
            phase_timeline_row if isinstance(phase_timeline_row, str) else ""
        ),
        "done": done,
    }
    if isinstance(normalized_progress_transition_v2, Mapping):
        normalized["progress_transition_v2"] = normalized_progress_transition_v2
    if isinstance(normalized_progress_transition_v1, Mapping):
        normalized["progress_transition_v1"] = normalized_progress_transition_v1
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
    header_value = phase_progress.get("phase_timeline_header")
    row_value = phase_progress.get("phase_timeline_row")
    header = (
        str(header_value)
        if isinstance(header_value, str) and header_value
        else phase_timeline_header_block()
    )
    row = (
        str(row_value)
        if isinstance(row_value, str) and row_value
        else phase_timeline_row_from_phase_progress(phase_progress)
    )
    return {"header": header, "row": row}


def phase_timeline_from_progress_notification(
    notification: Mapping[str, object],
) -> dict[str, str] | None:
    phase_progress = phase_progress_from_progress_notification(notification)
    if not isinstance(phase_progress, Mapping):
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
    phase = str(phase_progress.get("phase", "") or "")
    event_kind = str(phase_progress.get("event_kind", "") or "")
    done = phase_progress.get("done") is True
    force_emit = (
        not timeline_header_emitted
        or done
        or event_kind in {"terminal", "checkpoint"}
        or (
            isinstance(last_emitted_phase, str)
            and bool(phase)
            and phase != last_emitted_phase
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
