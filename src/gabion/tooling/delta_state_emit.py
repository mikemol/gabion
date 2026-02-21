from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Mapping

from gabion import server
from gabion.lsp_client import CommandRequest, run_command_direct
from gabion.order_contract import ordered_or_sorted

_DEFAULT_TIMEOUT_TICKS = "65000000"
_DEFAULT_TIMEOUT_TICK_NS = "1000000"
_LSP_PROGRESS_NOTIFICATION_METHOD = "$/progress"
_LSP_PROGRESS_TOKEN = "gabion.dataflowAudit/progress-v1"
_DEFAULT_RESUME_CHECKPOINT_PATH = Path(
    "artifacts/audit_reports/dataflow_resume_checkpoint_ci.json"
)
_EXPECTED_STATE_PATHS = (
    Path("artifacts/out/test_obsolescence_state.json"),
    Path("artifacts/out/test_annotation_drift.json"),
    Path("artifacts/out/ambiguity_state.json"),
)


def _phase_timeline_header_columns() -> list[str]:
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


def _phase_timeline_header_block() -> str:
    header = _phase_timeline_header_columns()
    header_line = "| " + " | ".join(header) + " |"
    separator_line = "| " + " | ".join(["---"] * len(header)) + " |"
    return header_line + "\n" + separator_line


def _phase_progress_dimensions_summary(
    phase_progress_v2: Mapping[str, object] | None,
) -> str:
    if not isinstance(phase_progress_v2, Mapping):
        return ""
    raw_dimensions = phase_progress_v2.get("dimensions")
    if not isinstance(raw_dimensions, Mapping):
        return ""
    fragments: list[str] = []
    for dim_name in ordered_or_sorted(
        raw_dimensions,
        source="_phase_progress_dimensions_summary.raw_dimensions",
    ):
        raw_payload = raw_dimensions.get(dim_name)
        if not isinstance(dim_name, str) or not isinstance(raw_payload, Mapping):
            continue
        raw_done = raw_payload.get("done")
        raw_total = raw_payload.get("total")
        if isinstance(raw_done, int) and isinstance(raw_total, int):
            done = max(int(raw_done), 0)
            total = max(int(raw_total), 0)
            if total:
                done = min(done, total)
            fragments.append(f"{dim_name}={done}/{total}")
    return "; ".join(fragments)


def _phase_timeline_row_from_phase_progress(
    phase_progress: Mapping[str, object],
) -> str:
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
    primary = ""
    if isinstance(phase_progress_v2, Mapping):
        primary_unit = str(phase_progress_v2.get("primary_unit", "") or "")
        raw_primary_done = phase_progress_v2.get("primary_done")
        raw_primary_total = phase_progress_v2.get("primary_total")
        if isinstance(raw_primary_done, int) and isinstance(raw_primary_total, int):
            done = max(int(raw_primary_done), 0)
            total = max(int(raw_primary_total), 0)
            if total:
                done = min(done, total)
            primary = f"{done}/{total}"
            if primary_unit:
                primary = f"{primary} {primary_unit}"
        elif primary_unit:
            primary = primary_unit
    if not primary:
        work_done = phase_progress.get("work_done")
        work_total = phase_progress.get("work_total")
        if isinstance(work_done, int) and isinstance(work_total, int):
            done = max(int(work_done), 0)
            total = max(int(work_total), 0)
            if total:
                done = min(done, total)
            primary = f"{done}/{total}"
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
    dimensions = _phase_progress_dimensions_summary(
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


def _timeout_ticks() -> int:
    raw = os.getenv("GABION_LSP_TIMEOUT_TICKS", _DEFAULT_TIMEOUT_TICKS)
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return int(_DEFAULT_TIMEOUT_TICKS)
    return parsed if parsed > 0 else int(_DEFAULT_TIMEOUT_TICKS)


def _timeout_tick_ns() -> int:
    raw = os.getenv("GABION_LSP_TIMEOUT_TICK_NS", _DEFAULT_TIMEOUT_TICK_NS)
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return int(_DEFAULT_TIMEOUT_TICK_NS)
    return parsed if parsed > 0 else int(_DEFAULT_TIMEOUT_TICK_NS)


def _build_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "analysis_timeout_ticks": _timeout_ticks(),
        "analysis_timeout_tick_ns": _timeout_tick_ns(),
        "fail_on_violations": False,
        "fail_on_type_ambiguities": False,
        "resume_on_timeout": 1,
        "emit_timeout_progress_report": True,
        "emit_test_obsolescence_state": True,
        "emit_test_annotation_drift": True,
        "emit_ambiguity_state": True,
    }
    if _DEFAULT_RESUME_CHECKPOINT_PATH.exists():
        payload["resume_checkpoint"] = str(_DEFAULT_RESUME_CHECKPOINT_PATH)
    else:
        payload["resume_checkpoint"] = False
    return payload


def _phase_progress_from_notification(
    notification: Mapping[str, object],
) -> dict[str, object] | None:
    if (
        str(notification.get("method", "") or "")
        != _LSP_PROGRESS_NOTIFICATION_METHOD
    ):
        return None
    params = notification.get("params")
    if not isinstance(params, Mapping):
        return None
    if str(params.get("token", "") or "") != _LSP_PROGRESS_TOKEN:
        return None
    value = params.get("value")
    if not isinstance(value, Mapping):
        return None
    phase = str(value.get("phase", "") or "")
    if not phase:
        return None
    work_done = value.get("work_done")
    work_total = value.get("work_total")
    completed_files = value.get("completed_files")
    remaining_files = value.get("remaining_files")
    total_files = value.get("total_files")
    if not isinstance(work_done, int):
        work_done = None
    if not isinstance(work_total, int):
        work_total = None
    if not isinstance(completed_files, int):
        completed_files = None
    if not isinstance(remaining_files, int):
        remaining_files = None
    if not isinstance(total_files, int):
        total_files = None
    raw_event_seq = value.get("event_seq")
    event_seq = (
        int(raw_event_seq)
        if isinstance(raw_event_seq, int) and not isinstance(raw_event_seq, bool)
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
    raw_stale_for_s = value.get("stale_for_s")
    stale_for_s = (
        float(raw_stale_for_s)
        if isinstance(raw_stale_for_s, (int, float)) and not isinstance(raw_stale_for_s, bool)
        else None
    )
    resume_checkpoint = value.get("resume_checkpoint")
    normalized_resume_checkpoint = (
        {str(key): resume_checkpoint[key] for key in resume_checkpoint}
        if isinstance(resume_checkpoint, Mapping)
        else None
    )
    return {
        "phase": phase,
        "analysis_state": str(value.get("analysis_state", "") or ""),
        "classification": str(value.get("classification", "") or ""),
        "event_kind": str(value.get("event_kind", "") or ""),
        "event_seq": event_seq,
        "work_done": work_done,
        "work_total": work_total,
        "completed_files": completed_files,
        "remaining_files": remaining_files,
        "total_files": total_files,
        "progress_marker": str(value.get("progress_marker", "") or ""),
        "ts_utc": str(value.get("ts_utc", "") or ""),
        "stale_for_s": stale_for_s,
        "phase_progress_v2": normalized_phase_progress_v2,
        "phase_timeline_header": (
            phase_timeline_header if isinstance(phase_timeline_header, str) else ""
        ),
        "phase_timeline_row": (
            phase_timeline_row if isinstance(phase_timeline_row, str) else ""
        ),
        "resume_checkpoint": normalized_resume_checkpoint,
        "done": bool(value.get("done", False)),
    }


def _phase_progress_signature(phase_progress: Mapping[str, object]) -> tuple[object, ...]:
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


def _emit_phase_progress_line(
    phase_progress: Mapping[str, object],
    *,
    print_fn: Callable[[str], None] = print,
) -> None:
    phase = str(phase_progress.get("phase", "") or "")
    if not phase:
        return
    header_value = phase_progress.get("phase_timeline_header")
    row_value = phase_progress.get("phase_timeline_row")
    header = (
        str(header_value)
        if isinstance(header_value, str) and header_value
        else _phase_timeline_header_block()
    )
    row = (
        str(row_value)
        if isinstance(row_value, str) and row_value
        else _phase_timeline_row_from_phase_progress(phase_progress)
    )
    print_fn("delta_state_emit timeline:")
    print_fn(header)
    print_fn(row)


def main(
    *,
    run_command_direct_fn: Callable[..., Mapping[str, object]] = run_command_direct,
    print_fn: Callable[[str], None] = print,
    monotonic_fn: Callable[[], float] = time.monotonic,
    expected_state_paths: tuple[Path, ...] = _EXPECTED_STATE_PATHS,
    root_path: Path = Path("."),
) -> int:
    payload = _build_payload()
    start = monotonic_fn()
    print_fn(
        "delta_state_emit: start "
        f"timeout_ticks={payload.get('analysis_timeout_ticks')} "
        f"timeout_tick_ns={payload.get('analysis_timeout_tick_ns')}"
    )
    last_progress_signature: tuple[object, ...] | None = None
    timeline_header_emitted = False

    def _on_notification(notification: dict[str, object]) -> None:
        nonlocal last_progress_signature
        nonlocal timeline_header_emitted
        phase_progress = _phase_progress_from_notification(notification)
        if phase_progress is None:
            return
        signature = _phase_progress_signature(phase_progress)
        if signature == last_progress_signature:
            return
        last_progress_signature = signature
        header_value = phase_progress.get("phase_timeline_header")
        row_value = phase_progress.get("phase_timeline_row")
        header = (
            str(header_value)
            if isinstance(header_value, str) and header_value
            else _phase_timeline_header_block()
        )
        row = (
            str(row_value)
            if isinstance(row_value, str) and row_value
            else _phase_timeline_row_from_phase_progress(phase_progress)
        )
        print_fn("delta_state_emit timeline:")
        if not timeline_header_emitted:
            print_fn(header)
            timeline_header_emitted = True
        print_fn(row)

    result = run_command_direct_fn(
        CommandRequest(
            server.DATAFLOW_COMMAND,
            [payload],
        ),
        root=root_path,
        notification_callback=_on_notification,
    )
    exit_code = int(result.get("exit_code", 0))
    elapsed_seconds = max(0.0, monotonic_fn() - start)
    print_fn(
        "delta_state_emit: complete "
        f"exit={exit_code} elapsed_s={elapsed_seconds:.2f}"
    )
    if exit_code != 0:
        print_fn(f"Delta state emit failed (exit {exit_code}).")
        return exit_code
    missing_outputs = [str(path) for path in expected_state_paths if not path.exists()]
    if missing_outputs:
        print_fn(
            "Delta state emit failed: missing expected state artifacts: "
            + ", ".join(missing_outputs)
        )
        return 1
    print_fn(
        "delta_state_emit: state artifacts ready "
        + ", ".join(str(path) for path in expected_state_paths)
    )
    return 0


