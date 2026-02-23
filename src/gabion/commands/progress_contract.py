# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from typing import Mapping

from gabion.order_contract import sort_once

LSP_PROGRESS_NOTIFICATION_METHOD = "$/progress"
LSP_PROGRESS_TOKEN = "gabion.dataflowAudit/progress-v1"
DEFAULT_TIMELINE_MIN_INTERVAL_SECONDS = 1.0


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


def resume_checkpoint_from_progress_notification(
    notification: Mapping[str, object],
) -> dict[str, object] | None:
    value = _progress_value_from_notification(notification)
    if not isinstance(value, Mapping):
        return None
    resume_checkpoint = value.get("resume_checkpoint")
    if not isinstance(resume_checkpoint, Mapping):
        return None
    checkpoint_path = str(resume_checkpoint.get("checkpoint_path", "") or "")
    status = str(resume_checkpoint.get("status", "") or "")
    reused_files = int(resume_checkpoint.get("reused_files", 0) or 0)
    total_files = int(resume_checkpoint.get("total_files", 0) or 0)
    return {
        "checkpoint_path": checkpoint_path,
        "status": status,
        "reused_files": reused_files,
        "total_files": total_files,
    }


def checkpoint_intro_timeline_from_progress_notification(
    notification: Mapping[str, object],
) -> dict[str, str] | None:
    value = _progress_value_from_notification(notification)
    if not isinstance(value, Mapping):
        return None
    row = value.get("checkpoint_intro_timeline_row")
    if not isinstance(row, str) or not row:
        return None
    header = value.get("checkpoint_intro_timeline_header")
    return {
        "header": header if isinstance(header, str) else "",
        "row": row,
    }


def phase_timeline_header_columns() -> list[str]:
    return [
        "ts_utc",
        "event_seq",
        "event_kind",
        "phase",
        "analysis_state",
        "classification",
        "progress_marker",
        "primary",
        "files",
        "resume_checkpoint",
        "stale_for_s",
        "dimensions",
    ]


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
    phase = str(phase_progress.get("phase", "") or "")
    analysis_state = str(phase_progress.get("analysis_state", "") or "")
    classification = str(phase_progress.get("classification", "") or "")
    progress_marker = str(phase_progress.get("progress_marker", "") or "")
    phase_progress_v2 = (
        phase_progress.get("phase_progress_v2")
        if isinstance(phase_progress.get("phase_progress_v2"), Mapping)
        else None
    )
    primary_unit = ""
    primary_done: int | None = None
    primary_total: int | None = None
    if isinstance(phase_progress_v2, Mapping):
        primary_unit = str(phase_progress_v2.get("primary_unit", "") or "")
        raw_primary_done = phase_progress_v2.get("primary_done")
        raw_primary_total = phase_progress_v2.get("primary_total")
        if isinstance(raw_primary_done, int) and not isinstance(raw_primary_done, bool):
            primary_done = max(int(raw_primary_done), 0)
        if isinstance(raw_primary_total, int) and not isinstance(raw_primary_total, bool):
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
    resume_checkpoint = ""
    raw_resume = phase_progress.get("resume_checkpoint")
    if isinstance(raw_resume, Mapping):
        checkpoint_path = str(raw_resume.get("checkpoint_path", "") or "")
        status = str(raw_resume.get("status", "") or "")
        raw_reused = raw_resume.get("reused_files")
        raw_resume_total = raw_resume.get("total_files")
        if isinstance(raw_reused, int) and isinstance(raw_resume_total, int):
            resume_checkpoint = (
                f"path={checkpoint_path or '<none>'} status={status or 'unknown'} "
                f"reused_files={raw_reused}/{raw_resume_total}"
            )
        else:
            resume_checkpoint = (
                f"path={checkpoint_path or '<none>'} status={status or 'unknown'} "
                "reused_files=unknown"
            )
    raw_stale_for_s = phase_progress.get("stale_for_s")
    stale_for_s = (
        f"{float(raw_stale_for_s):.1f}"
        if isinstance(raw_stale_for_s, (int, float))
        else ""
    )
    dimensions = phase_progress_dimensions_summary(
        phase_progress_v2 if isinstance(phase_progress_v2, Mapping) else None
    )
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
        resume_checkpoint,
        stale_for_s,
        dimensions,
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
    phase_timeline_header = value.get("phase_timeline_header")
    phase_timeline_row = value.get("phase_timeline_row")
    resume_checkpoint = value.get("resume_checkpoint")
    normalized_resume_checkpoint = (
        {str(key): resume_checkpoint[key] for key in resume_checkpoint}
        if isinstance(resume_checkpoint, Mapping)
        else None
    )
    done = bool(value.get("done", False))
    return {
        "phase": phase,
        "work_done": work_done,
        "work_total": work_total,
        "completed_files": completed_files,
        "remaining_files": remaining_files,
        "total_files": total_files,
        "analysis_state": analysis_state,
        "classification": classification,
        "event_kind": event_kind,
        "event_seq": event_seq,
        "ts_utc": str(value.get("ts_utc", "") or ""),
        "stale_for_s": stale_for_s,
        "phase_progress_v2": normalized_phase_progress_v2,
        "progress_marker": progress_marker,
        "phase_timeline_header": (
            phase_timeline_header if isinstance(phase_timeline_header, str) else ""
        ),
        "phase_timeline_row": (
            phase_timeline_row if isinstance(phase_timeline_row, str) else ""
        ),
        "resume_checkpoint": normalized_resume_checkpoint,
        "done": done,
    }


def phase_progress_signature(phase_progress: Mapping[str, object]) -> tuple[object, ...]:
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
        phase_progress.get("progress_marker"),
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
