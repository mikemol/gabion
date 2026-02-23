#!/usr/bin/env python3
# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Mapping, Sequence

from gabion.analysis.timeout_context import check_deadline, deadline_loop_iter
from gabion.order_contract import sort_once
from gabion.runtime import env_policy, json_io
from gabion.tooling import tool_specs
from gabion.tooling.deadline_runtime import deadline_scope_from_lsp_env

_STAGE_SEQUENCE: tuple[str, ...] = (
    "run",
    "retry1",
    "retry2",
    "retry3",
    "retry4",
    "retry5",
)
_DELTA_GATE_STEPS: tuple[tool_specs.ToolSpec, ...] = tool_specs.dataflow_stage_gate_specs()
_DELTA_GATE_REGISTRY: dict[str, Callable[[], int]] = {
    spec.id: spec.run for spec in _DELTA_GATE_STEPS
}


@dataclass(frozen=True)
class StageResult:
    stage_id: str
    exit_code: int
    analysis_state: str
    is_timeout_resume: bool
    metrics_line: str
    obligation_rows: tuple[dict[str, object], ...]
    incompleteness_markers: tuple[str, ...]

    @property
    def terminal_status(self) -> str:
        if self.exit_code == 0:
            return "success"
        if self.is_timeout_resume:
            return "timeout_resume"
        return "hard_failure"


@dataclass(frozen=True)
class StagePaths:
    report_path: Path
    timeout_progress_json_path: Path
    timeout_progress_md_path: Path
    deadline_profile_json_path: Path
    deadline_profile_md_path: Path
    obligation_trace_json_path: Path
    resume_checkpoint_path: Path
    baseline_path: Path


@dataclass
class DebugDumpState:
    stage_ids: tuple[str, ...]
    started_wall_seconds: float
    attempts_started: int = 0
    attempts_completed: int = 0
    active_stage_id: str | None = None
    active_stage_started_wall_seconds: float | None = None
    active_stage_strictness: str | None = None
    active_command: tuple[str, ...] = ()
    last_analysis_state: str = "none"
    last_terminal_status: str = "none"


def _load_json_object(path: Path) -> dict[str, object]:
    return json_io.load_json_object_path(path)


def _analysis_state(timeout_progress_path: Path) -> str:
    payload = _load_json_object(timeout_progress_path)
    state = payload.get("analysis_state")
    if isinstance(state, str) and state:
        return state
    progress = payload.get("progress")
    if isinstance(progress, dict):
        classification = progress.get("classification")
        if isinstance(classification, str) and classification:
            return classification
    return "none"


def _metrics_line(deadline_profile_path: Path) -> str:
    payload = _load_json_object(deadline_profile_path)
    if not payload:
        return "ticks=n/a checks=n/a ticks_per_ns=n/a wall_s=n/a"
    ticks = payload.get("ticks_consumed", "n/a")
    checks = payload.get("checks_total", "n/a")
    ticks_per_ns = payload.get("ticks_per_ns", "n/a")
    wall_total_elapsed_ns = payload.get("wall_total_elapsed_ns")
    wall_s = (
        f"{wall_total_elapsed_ns / 1_000_000_000:.3f}"
        if isinstance(wall_total_elapsed_ns, int)
        else "n/a"
    )
    return (
        f"ticks={ticks} checks={checks} ticks_per_ns={ticks_per_ns} "
        f"wall_s={wall_s}"
    )


def _copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


def _resume_checkpoint_metrics_line(resume_checkpoint_path: Path) -> str:
    payload = _load_json_object(resume_checkpoint_path)
    if not payload:
        return "resume_checkpoint=missing"
    completed_paths = payload.get("completed_paths")
    completed_paths_count = len(completed_paths) if isinstance(completed_paths, list) else "n/a"
    analysis_index_resume = payload.get("analysis_index_resume")
    if not isinstance(analysis_index_resume, dict):
        return f"resume_checkpoint=present completed_paths={completed_paths_count} hydrated_paths=n/a"
    hydrated_paths_count = analysis_index_resume.get("hydrated_paths_count", "n/a")
    profiling_v1 = analysis_index_resume.get("profiling_v1")
    parsed_paths = "n/a"
    if isinstance(profiling_v1, dict):
        counters = profiling_v1.get("counters")
        if isinstance(counters, dict):
            parsed_paths = counters.get("analysis_index.paths_parsed", "n/a")
    return (
        "resume_checkpoint=present "
        f"completed_paths={completed_paths_count} "
        f"hydrated_paths={hydrated_paths_count} "
        f"paths_parsed_after_resume={parsed_paths}"
    )


def _obligation_required_action(kind: str) -> str:
    actions = {
        "classification_matches_resume_support": "align timeout classification with resume support semantics",
        "progress_monotonicity": "preserve monotonic semantic progress",
        "substantive_progress_required": "emit substantive progress only when resumable timeout progress exists",
        "checkpoint_present_when_resumable": "persist a resume checkpoint for resumable timeout progress",
        "restart_required_on_witness_mismatch": "restart projection when witness mismatch is detected",
        "no_projection_progress": "resolve at least one projected report section",
        "partial_report_emitted": "emit a partial report during timeout handling",
        "section_projection_state": "reuse or regenerate projected section content according to policy",
    }
    return actions.get(kind, "satisfy contract obligation")


def _normalize_obligation_status(raw_status: str, detail: str) -> str:
    if raw_status == "SATISFIED":
        return "satisfied"
    if raw_status == "VIOLATION":
        return "unsatisfied"
    if raw_status == "OBLIGATION" and detail in {"policy", "stale_input"}:
        return "skipped_by_policy"
    return "unsatisfied"


def _obligation_id(stage_id: str, contract: str, kind: str, section_id: str, phase: str) -> str:
    # dataflow-bundle: contract, kind, phase, section_id, stage_id
    material = "|".join((stage_id, contract, kind, section_id, phase))
    digest = hashlib.sha1(material.encode("utf-8")).hexdigest()
    return f"obl-{digest[:12]}"


def _obligation_rows_from_timeout_payload(
    *, stage_id: str, analysis_state: str, timeout_payload: dict[str, object]
) -> tuple[tuple[dict[str, object], ...], tuple[str, ...]]:
    incremental = timeout_payload.get("incremental_obligations")
    if not isinstance(incremental, list):
        markers = (
            ("missing_incremental_obligations",)
            if analysis_state.startswith("timed_out_")
            else ()
        )
        return (), markers
    rows: list[dict[str, object]] = []
    for raw_entry in deadline_loop_iter(incremental):
        if not isinstance(raw_entry, dict):
            continue
        contract = str(raw_entry.get("contract", "") or "")
        kind = str(raw_entry.get("kind", "") or "")
        if not contract or not kind:
            continue
        section_id = str(raw_entry.get("section_id", "") or "")
        phase = str(raw_entry.get("phase", "") or "")
        detail = str(raw_entry.get("detail", "") or "")
        raw_status = str(raw_entry.get("status", "") or "")
        rows.append(
            {
                "id": _obligation_id(stage_id, contract, kind, section_id, phase),
                "stage_id": stage_id,
                "rule_evaluated": f"{contract}:{kind}",
                "trigger_evidence": detail,
                "required_action": _obligation_required_action(kind),
                "status": _normalize_obligation_status(raw_status, detail),
                "raw_status": raw_status,
                "contract": contract,
                "kind": kind,
                "section_id": section_id,
                "phase": phase,
            }
        )
    rows = sort_once(
        rows,
        source="run_dataflow_stage._collect_incremental_obligations.rows",
        # Lexical obligation-id order stabilizes obligation trace rows.
        key=lambda row: str(row["id"]),
    )
    markers: list[str] = []
    if timeout_payload.get("cleanup_truncated"):
        markers.append("cleanup_truncated")
    return tuple(rows), tuple(markers)


def _obligation_trace_payload(results: Sequence[StageResult]) -> dict[str, object]:
    obligations = [
        row
        for result in deadline_loop_iter(results)
        for row in deadline_loop_iter(result.obligation_rows)
    ]
    obligations = sort_once(
        obligations,
        source="run_dataflow_stage._obligation_trace_payload.obligations",
        # Lexical (id, stage_id) order stabilizes merged cross-stage trace payload.
        key=lambda row: (str(row.get("id", "")), str(row.get("stage_id", ""))),
    )
    markers = {
        marker
        for result in deadline_loop_iter(results)
        for marker in deadline_loop_iter(result.incompleteness_markers)
    }
    if results and results[-1].terminal_status != "success":
        markers.add("terminal_non_success")
    if any(result.is_timeout_resume for result in results):
        markers.add("timeout_or_partial_run")

    summary = {
        "total": len(obligations),
        "satisfied": sum(1 for row in obligations if row.get("status") == "satisfied"),
        "unsatisfied": sum(1 for row in obligations if row.get("status") == "unsatisfied"),
        "skipped_by_policy": sum(
            1 for row in obligations if row.get("status") == "skipped_by_policy"
        ),
    }
    return {
        "trace_version": 1,
        "complete": not markers,
        "incompleteness_markers": sort_once(
            markers,
            source="_obligation_trace_payload.incompleteness_markers",
        ),
        "summary": summary,
        "obligations": obligations,
    }


def _write_obligation_trace(path: Path, results: Sequence[StageResult]) -> dict[str, object]:
    payload = _obligation_trace_payload(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return payload


def _obligation_trace_summary_lines(trace_payload: dict[str, object]) -> list[str]:
    summary = trace_payload.get("summary")
    if not isinstance(summary, dict):
        return []
    markers = trace_payload.get("incompleteness_markers")
    marker_text = (
        ", ".join(str(marker) for marker in markers)
        if isinstance(markers, list) and markers
        else "none"
    )
    return [
        "",
        "## Obligation trace summary",
        (
            "- total="
            f"{summary.get('total', 0)} "
            f"satisfied={summary.get('satisfied', 0)} "
            f"unsatisfied={summary.get('unsatisfied', 0)} "
            f"skipped_by_policy={summary.get('skipped_by_policy', 0)}"
        ),
        f"- complete={trace_payload.get('complete', False)}",
        f"- incompleteness_markers={marker_text}",
    ]


def _append_markdown_summary(path: Path, trace_payload: dict[str, object]) -> None:
    if not path.exists():
        return
    with path.open("a", encoding="utf-8") as handle:
        for line in deadline_loop_iter(_obligation_trace_summary_lines(trace_payload)):
            handle.write(f"{line}\n")


def _stage_snapshot_path(path: Path, stage_id: str) -> Path:
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    return path.with_name(f"{stem}_stage_{stage_id}{suffix}")


def _append_lines(path: Path | None, lines: Sequence[str]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for line in deadline_loop_iter(lines):
            handle.write(f"{line}\n")


def _phase_timeline_markdown_path(report_path: Path) -> Path:
    return report_path.parent / "dataflow_phase_timeline.md"


def _phase_timeline_jsonl_path(report_path: Path) -> Path:
    return report_path.parent / "dataflow_phase_timeline.jsonl"


def _markdown_timeline_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return 0
    table_rows = 0
    for line in lines:
        check_deadline()
        if line.startswith("| "):
            table_rows += 1
    return max(0, table_rows - 2)


def _markdown_timeline_last_row(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return ""
    for line in reversed(lines):
        check_deadline()
        if line.startswith("| ") and not line.startswith("| ---"):
            return line
    return ""


def _phase_timeline_stale_for_s_from_row(row: str) -> str:
    stripped = row.strip()
    if not stripped.startswith("|"):
        return ""
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    # Column order is defined by gabion server telemetry timeline helpers.
    stale_index = 10
    if len(cells) <= stale_index:
        return ""
    value = cells[stale_index]
    return value if value else ""


def _command_preview(command: Sequence[str], *, max_chars: int = 240) -> str:
    text = " ".join(str(token) for token in command)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _timeout_progress_metrics_line(path: Path) -> str:
    payload = _load_json_object(path)
    if not payload:
        return "timeout_progress=missing"
    analysis_state = _analysis_state(path)
    progress = payload.get("progress")
    if not isinstance(progress, dict):
        return f"timeout_progress_state={analysis_state} classification=n/a phase=n/a"
    classification = str(progress.get("classification", "") or "n/a")
    phase = str(progress.get("phase", "") or "n/a")
    work_done = progress.get("work_done")
    work_total = progress.get("work_total")
    completed_files = progress.get("completed_files")
    remaining_files = progress.get("remaining_files")
    total_files = progress.get("total_files")
    return (
        f"timeout_progress_state={analysis_state} "
        f"classification={classification} phase={phase} "
        f"work={work_done}/{work_total} files={completed_files}/{total_files} "
        f"remaining={remaining_files}"
    )


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _reset_run_observability_artifacts(paths: StagePaths) -> None:
    phase_timeline_md = _phase_timeline_markdown_path(paths.report_path)
    phase_timeline_jsonl = _phase_timeline_jsonl_path(paths.report_path)
    checkpoint_intro_timeline = paths.report_path.parent / "dataflow_checkpoint_intro_timeline.md"
    for artifact_path in (
        paths.timeout_progress_json_path,
        paths.timeout_progress_md_path,
        paths.deadline_profile_json_path,
        paths.deadline_profile_md_path,
        paths.obligation_trace_json_path,
        phase_timeline_md,
        phase_timeline_jsonl,
        checkpoint_intro_timeline,
    ):
        check_deadline()
        _unlink_if_exists(artifact_path)


def _debug_dump_stage_start(
    *,
    state: DebugDumpState,
    stage_id: str,
    command: Sequence[str],
    strictness: str | None,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> None:
    state.attempts_started += 1
    state.active_stage_id = stage_id
    state.active_stage_started_wall_seconds = monotonic_fn()
    state.active_stage_strictness = strictness
    state.active_command = tuple(command)


def _debug_dump_stage_end(*, state: DebugDumpState, result: StageResult) -> None:
    state.attempts_completed += 1
    state.last_analysis_state = result.analysis_state
    state.last_terminal_status = result.terminal_status
    state.active_stage_id = None
    state.active_stage_started_wall_seconds = None
    state.active_stage_strictness = None
    state.active_command = ()


def _emit_debug_dump(
    *,
    reason: str,
    state: DebugDumpState,
    paths: StagePaths,
    step_summary_path: Path | None,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> None:
    now_wall = monotonic_fn()
    active_elapsed_s = (
        f"{max(0.0, now_wall - state.active_stage_started_wall_seconds):.1f}"
        if isinstance(state.active_stage_started_wall_seconds, float)
        else "n/a"
    )
    wall_elapsed_s = max(0.0, now_wall - state.started_wall_seconds)
    stage_id = state.active_stage_id or "none"
    stage_strictness = state.active_stage_strictness or "default"
    attempts_total = max(1, len(state.stage_ids))
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines: list[str] = [
        (
            "debug dump: "
            f"reason={reason} ts_utc={timestamp} pid={os.getpid()} "
            f"attempts_started={state.attempts_started}/{attempts_total} "
            f"attempts_completed={state.attempts_completed}/{attempts_total} "
            f"active_stage={stage_id} active_strictness={stage_strictness} "
            f"active_elapsed_s={active_elapsed_s} wall_elapsed_s={wall_elapsed_s:.1f}"
        ),
        f"debug dump: resume={_resume_checkpoint_metrics_line(paths.resume_checkpoint_path)}",
        f"debug dump: timeout={_timeout_progress_metrics_line(paths.timeout_progress_json_path)}",
        f"debug dump: deadline={_metrics_line(paths.deadline_profile_json_path)}",
        (
            "debug dump: "
            f"last_stage_state={state.last_analysis_state} "
            f"last_terminal_status={state.last_terminal_status}"
        ),
    ]
    if state.active_command:
        lines.append(
            f"debug dump: active_command={_command_preview(state.active_command)}"
        )
    phase_timeline_markdown = _phase_timeline_markdown_path(paths.report_path)
    phase_timeline_jsonl = _phase_timeline_jsonl_path(paths.report_path)
    lines.append(
        "debug dump: "
        f"phase_timeline_rows={_markdown_timeline_row_count(phase_timeline_markdown)} "
        f"path={phase_timeline_markdown}"
    )
    lines.append(
        "debug dump: "
        f"phase_timeline_jsonl_present={'yes' if phase_timeline_jsonl.exists() else 'no'} "
        f"path={phase_timeline_jsonl}"
    )
    last_timeline_row = _markdown_timeline_last_row(phase_timeline_markdown)
    if last_timeline_row:
        lines.append(f"debug dump: phase_timeline_last_row={last_timeline_row}")
        stale_for_s = _phase_timeline_stale_for_s_from_row(last_timeline_row)
        if stale_for_s:
            lines.append(f"debug dump: phase_timeline_last_stale_for_s={stale_for_s}")
    for line in deadline_loop_iter(lines):
        print(line, flush=True)
    _append_lines(step_summary_path, [f"- {lines[0]}"])


def _install_signal_debug_dump_handler(
    *,
    emit_dump_fn: Callable[[str], None],
    signal_module: Any = signal,
) -> Callable[[], None]:
    sigusr1 = getattr(signal_module, "SIGUSR1", None)
    signal_fn = getattr(signal_module, "signal", None)
    getsignal_fn = getattr(signal_module, "getsignal", None)
    if sigusr1 is None or not callable(signal_fn):
        return lambda: None
    previous_handler = getsignal_fn(sigusr1) if callable(getsignal_fn) else None

    def _signal_handler(_signum: int, _frame: FrameType | None) -> None:
        emit_dump_fn("SIGUSR1")

    signal_fn(sigusr1, _signal_handler)

    def _restore() -> None:
        if previous_handler is not None:
            signal_fn(sigusr1, previous_handler)

    return _restore


def _env_int(name: str, default: int) -> int:
    text = env_policy.env_text(name)
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        return default


def _check_command(
    *,
    paths: StagePaths,
    resume_on_timeout: int,
    strictness: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "gabion",
        "check",
        "--no-fail-on-violations",
        "--no-fail-on-type-ambiguities",
        "--report",
        str(paths.report_path),
        "--resume-checkpoint",
        str(paths.resume_checkpoint_path),
        "--resume-on-timeout",
        str(max(0, int(resume_on_timeout))),
        "--emit-timeout-progress-report",
        "--baseline",
        str(paths.baseline_path),
    ]
    if strictness:
        command.extend(["--strictness", strictness])
    return command


def run_stage(
    *,
    stage_id: str,
    paths: StagePaths,
    resume_on_timeout: int,
    step_summary_path: Path | None,
    run_command_fn: Callable[[Sequence[str]], int],
    strictness: str | None = None,
    command: Sequence[str] | None = None,
) -> StageResult:
    paths.report_path.parent.mkdir(parents=True, exist_ok=True)
    paths.deadline_profile_json_path.parent.mkdir(parents=True, exist_ok=True)
    resume_metrics_line = _resume_checkpoint_metrics_line(paths.resume_checkpoint_path)
    strictness_note = f" strictness={strictness}" if strictness else ""
    print(f"stage {stage_id.upper()}: {resume_metrics_line}{strictness_note}", flush=True)
    _append_lines(
        step_summary_path,
        [f"- stage {stage_id.upper()}: {resume_metrics_line}{strictness_note}"],
    )
    stage_command = (
        list(command)
        if command is not None
        else _check_command(
            paths=paths,
            resume_on_timeout=resume_on_timeout,
            strictness=strictness,
        )
    )
    exit_code = int(run_command_fn(stage_command))
    analysis_state = _analysis_state(paths.timeout_progress_json_path)
    is_timeout_resume = analysis_state == "timed_out_progress_resume"
    metrics_line = _metrics_line(paths.deadline_profile_json_path)

    _copy_if_exists(paths.report_path, _stage_snapshot_path(paths.report_path, stage_id))
    _copy_if_exists(
        paths.timeout_progress_json_path,
        _stage_snapshot_path(paths.timeout_progress_json_path, stage_id),
    )
    _copy_if_exists(
        paths.timeout_progress_md_path,
        _stage_snapshot_path(paths.timeout_progress_md_path, stage_id),
    )
    _copy_if_exists(
        paths.deadline_profile_json_path,
        _stage_snapshot_path(paths.deadline_profile_json_path, stage_id),
    )
    _copy_if_exists(
        paths.deadline_profile_md_path,
        _stage_snapshot_path(paths.deadline_profile_md_path, stage_id),
    )
    phase_timeline_markdown = _phase_timeline_markdown_path(paths.report_path)
    _copy_if_exists(
        phase_timeline_markdown,
        _stage_snapshot_path(phase_timeline_markdown, stage_id),
    )
    phase_timeline_jsonl = _phase_timeline_jsonl_path(paths.report_path)
    _copy_if_exists(
        phase_timeline_jsonl,
        _stage_snapshot_path(phase_timeline_jsonl, stage_id),
    )

    timeout_payload = _load_json_object(paths.timeout_progress_json_path)
    obligation_rows, incompleteness_markers = _obligation_rows_from_timeout_payload(
        stage_id=stage_id,
        analysis_state=analysis_state,
        timeout_payload=timeout_payload,
    )

    stage_upper = stage_id.upper()
    print(
        f"stage {stage_upper}: exit={exit_code} "
        f"analysis_state={analysis_state} {metrics_line}{strictness_note}",
        flush=True,
    )
    _append_lines(
        step_summary_path,
        [
            (
                f"- stage {stage_upper}: exit=`{exit_code}`, "
                f"state=`{analysis_state}`, {metrics_line}{strictness_note}"
            )
        ],
    )
    return StageResult(
        stage_id=stage_id,
        exit_code=exit_code,
        analysis_state=analysis_state,
        is_timeout_resume=is_timeout_resume,
        metrics_line=metrics_line,
        obligation_rows=obligation_rows,
        incompleteness_markers=incompleteness_markers,
    )


def _parse_stage_strictness_profile(
    raw_profile: str,
) -> dict[str, str]:
    check_deadline()
    profile = raw_profile.strip()
    if not profile:
        return {}
    mapping: dict[str, str] = {}
    if "=" in profile:
        for raw_part in profile.split(","):
            check_deadline()
            part = raw_part.strip()
            if not part or "=" not in part:
                continue
            stage_name, strictness = (token.strip().lower() for token in part.split("=", 1))
            if stage_name in _STAGE_SEQUENCE and strictness in {"low", "high"}:
                mapping[stage_name] = strictness
        return mapping
    values = [token.strip().lower() for token in profile.split(",") if token.strip()]
    for stage_name, strictness in zip(_STAGE_SEQUENCE, values):
        check_deadline()
        if strictness in {"low", "high"}:
            mapping[stage_name] = strictness
    return mapping


def _stage_strictness(stage_id: str, strictness_by_stage: Mapping[str, str]) -> str | None:
    strictness = strictness_by_stage.get(stage_id)
    if strictness in {"low", "high"}:
        return strictness
    return None


def _stage_ids(start_stage: str, max_attempts: int) -> list[str]:
    if max_attempts <= 0:
        return []
    try:
        start_idx = _STAGE_SEQUENCE.index(start_stage)
    except ValueError:
        start_idx = 0
    return list(_STAGE_SEQUENCE[start_idx : start_idx + max_attempts])


def _emit_stage_outputs(
    output_path: Path | None,
    results: Sequence[StageResult],
) -> None:
    if not results:
        return
    terminal = results[-1]
    lines: list[str] = []
    for result in deadline_loop_iter(results):
        prefix = f"stage_{result.stage_id}"
        lines.extend(
            [
                f"{prefix}_exit={result.exit_code}",
                f"{prefix}_analysis_state={result.analysis_state}",
                f"{prefix}_is_timeout_resume={'true' if result.is_timeout_resume else 'false'}",
                f"{prefix}_metrics={result.metrics_line}",
            ]
        )
    lines.extend(
        [
            f"attempts_run={len(results)}",
            f"terminal_stage={terminal.stage_id.upper()}",
            f"terminal_status={terminal.terminal_status}",
            f"exit_code={terminal.exit_code}",
            f"analysis_state={terminal.analysis_state}",
            f"is_timeout_resume={'true' if terminal.is_timeout_resume else 'false'}",
            f"stage_metrics={terminal.metrics_line}",
        ]
    )
    _append_lines(output_path, lines)




def _run_named_delta_gate(step_id: str) -> int:
    runner = _DELTA_GATE_REGISTRY.get(step_id)
    if runner is None:
        print(f"delta gate unknown: {step_id}", flush=True)
        return 2
    try:
        return int(runner())
    except Exception as exc:
        print(f"delta gate crashed: {step_id} ({exc})", flush=True)
        return 2


def _run_delta_gates(run_gate_fn: Callable[[str], int]) -> int:
    for spec in deadline_loop_iter(_DELTA_GATE_STEPS):
        gate_exit = int(run_gate_fn(spec.id))
        if gate_exit != 0:
            print(f"delta gate failed: {spec.id} (exit {gate_exit})", flush=True)
            return gate_exit
    return 0

def run_staged(
    *,
    stage_ids: Sequence[str],
    paths: StagePaths,
    resume_on_timeout: int,
    step_summary_path: Path | None,
    run_command_fn: Callable[[Sequence[str]], int],
    run_gate_fn: Callable[[str], int] | None = None,
    strictness_by_stage: Mapping[str, str] | None = None,
    max_wall_seconds: int | None = None,
    finalize_reserve_seconds: int = 0,
    monotonic_fn: Callable[[], float] = time.monotonic,
    on_stage_start: (
        Callable[[str, Sequence[str], str | None], None] | None
    ) = None,
    on_stage_end: Callable[[StageResult], None] | None = None,
) -> list[StageResult]:
    strictness_profile = dict(strictness_by_stage or {})
    results: list[StageResult] = []
    started_wall_seconds = monotonic_fn()
    for stage_id in deadline_loop_iter(stage_ids):
        if (
            results
            and isinstance(max_wall_seconds, int)
            and max_wall_seconds > 0
        ):
            elapsed = max(0.0, monotonic_fn() - started_wall_seconds)
            remaining = float(max_wall_seconds) - elapsed
            reserve_seconds = max(0, int(finalize_reserve_seconds))
            if remaining <= float(reserve_seconds):
                stage_upper = stage_id.upper()
                message = (
                    f"stage {stage_upper}: skipped due remaining wall budget "
                    f"({remaining:.1f}s <= reserve {reserve_seconds}s)"
                )
                print(message, flush=True)
                _append_lines(step_summary_path, [f"- {message}"])
                break
        stage_strictness = _stage_strictness(stage_id, strictness_profile)
        stage_command = _check_command(
            paths=paths,
            resume_on_timeout=resume_on_timeout,
            strictness=stage_strictness,
        )
        if callable(on_stage_start):
            on_stage_start(stage_id, stage_command, stage_strictness)
        result = run_stage(
            stage_id=stage_id,
            paths=paths,
            resume_on_timeout=resume_on_timeout,
            step_summary_path=step_summary_path,
            run_command_fn=run_command_fn,
            strictness=stage_strictness,
            command=stage_command,
        )
        if callable(on_stage_end):
            on_stage_end(result)
        results.append(result)
        if result.exit_code == 0:
            gate_runner = _run_named_delta_gate if run_gate_fn is None else run_gate_fn
            gate_exit = _run_delta_gates(gate_runner)
            if gate_exit != 0:
                result = StageResult(
                    stage_id=result.stage_id,
                    exit_code=gate_exit,
                    analysis_state="delta_gate_failure",
                    is_timeout_resume=False,
                    metrics_line=result.metrics_line,
                    obligation_rows=result.obligation_rows,
                    incompleteness_markers=result.incompleteness_markers,
                )
                results[-1] = result
                _append_lines(
                    step_summary_path,
                    [
                        (
                            f"- stage {stage_id.upper()}: delta gates failed "
                            f"(exit=`{gate_exit}`)."
                        )
                    ],
                )
            break
        if not result.is_timeout_resume:
            break
    return results


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one dataflow grammar CI invocation with deterministic outputs/artifacts."
        )
    )
    default_debug_dump_interval_seconds = max(
        0, _env_int("GABION_DATAFLOW_DEBUG_DUMP_INTERVAL_SECONDS", 0)
    )
    parser.add_argument(
        "--stage-id",
        default="run",
        choices=_STAGE_SEQUENCE,
        help=(
            "Invocation identifier to start staged retries from."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=len(_STAGE_SEQUENCE),
        help=(
            "Maximum staged attempts to execute from --stage-id."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/audit_reports/dataflow_report.md"),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=Path("artifacts/audit_reports/dataflow_resume_checkpoint_ci.json"),
    )
    parser.add_argument(
        "--resume-on-timeout",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--stage-strictness-profile",
        default="",
        help=(
            "Optional strictness profile. Accepts 'run=low' or positional 'low'."
        ),
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=int,
        default=0,
        help=(
            "Optional wall-clock cap for invocation orchestration. "
            "With single-invocation mode this is advisory only."
        ),
    )
    parser.add_argument(
        "--finalize-reserve-seconds",
        type=int,
        default=0,
        help=(
            "Wall-clock reserve to keep available for finalize/upload steps "
            "before launching another retry stage."
        ),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("baselines/dataflow_baseline.txt"),
    )
    parser.add_argument(
        "--timeout-progress-json",
        type=Path,
        default=Path("artifacts/audit_reports/timeout_progress.json"),
    )
    parser.add_argument(
        "--timeout-progress-md",
        type=Path,
        default=Path("artifacts/audit_reports/timeout_progress.md"),
    )
    parser.add_argument(
        "--deadline-profile-json",
        type=Path,
        default=Path("artifacts/out/deadline_profile.json"),
    )
    parser.add_argument(
        "--deadline-profile-md",
        type=Path,
        default=Path("artifacts/out/deadline_profile.md"),
    )
    parser.add_argument(
        "--obligation-trace-json",
        type=Path,
        default=Path("artifacts/out/obligation_trace.json"),
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        default=None,
        help="Defaults to $GITHUB_OUTPUT when omitted.",
    )
    parser.add_argument(
        "--step-summary",
        type=Path,
        default=None,
        help="Defaults to $GITHUB_STEP_SUMMARY when omitted.",
    )
    parser.add_argument(
        "--debug-dump-interval-seconds",
        type=int,
        default=default_debug_dump_interval_seconds,
        help=(
            "Emit debug state dumps on this interval (seconds). "
            "Also supports on-demand dumps via SIGUSR1."
        ),
    )
    parsed_argv = list(argv) if argv is not None else None
    return parser.parse_args(parsed_argv)


def _run_subprocess(
    command: Sequence[str],
    *,
    heartbeat_interval_seconds: int = 0,
    on_heartbeat: Callable[[], None] | None = None,
    poll_interval_seconds: float = 1.0,
    popen_fn: Callable[[Sequence[str]], subprocess.Popen[Any]] = subprocess.Popen,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    try:
        process = popen_fn(command)
    except OSError:
        return 127
    heartbeat_interval = max(0, int(heartbeat_interval_seconds))
    last_heartbeat = monotonic_fn()
    sleep_interval = max(0.05, float(poll_interval_seconds))
    while True:
        check_deadline()
        return_code = process.poll()
        if return_code is not None:
            return int(return_code)
        if heartbeat_interval > 0 and callable(on_heartbeat):
            now = monotonic_fn()
            if (now - last_heartbeat) >= float(heartbeat_interval):
                on_heartbeat()
                last_heartbeat = now
        sleep_fn(sleep_interval)


def main(
    argv: Sequence[str] | None = None,
    *,
    run_staged_fn: Callable[..., list[StageResult]] = run_staged,
    write_obligation_trace_fn: Callable[[Path, Sequence[StageResult]], dict[str, object]] = _write_obligation_trace,
    append_markdown_summary_fn: Callable[[Path, dict[str, object]], None] = _append_markdown_summary,
    append_lines_fn: Callable[[Path | None, Sequence[str]], None] = _append_lines,
    emit_stage_outputs_fn: Callable[[Path | None, Sequence[StageResult]], None] = _emit_stage_outputs,
    reset_run_observability_artifacts_fn: Callable[[StagePaths], None] = _reset_run_observability_artifacts,
    install_signal_debug_dump_handler_fn: Callable[..., Callable[[], None]] = _install_signal_debug_dump_handler,
    run_subprocess_fn: Callable[..., int] = _run_subprocess,
    deadline_scope_factory: Callable[[], Any] = deadline_scope_from_lsp_env,
) -> int:
    args = _parse_args(argv)
    github_output_path = args.github_output
    if github_output_path is None:
        output_env_text = os.getenv("GITHUB_OUTPUT", "").strip()
        if output_env_text:
            github_output_path = Path(output_env_text)
    step_summary_path = args.step_summary
    if step_summary_path is None:
        summary_env_text = os.getenv("GITHUB_STEP_SUMMARY", "").strip()
        if summary_env_text:
            step_summary_path = Path(summary_env_text)

    stage_ids = _stage_ids(args.stage_id, int(args.max_attempts))
    if not stage_ids:
        print("No stages requested; max-attempts must be > 0.", flush=True)
        return 2
    paths = StagePaths(
        report_path=args.report,
        timeout_progress_json_path=args.timeout_progress_json,
        timeout_progress_md_path=args.timeout_progress_md,
        deadline_profile_json_path=args.deadline_profile_json,
        deadline_profile_md_path=args.deadline_profile_md,
        obligation_trace_json_path=args.obligation_trace_json,
        resume_checkpoint_path=args.resume_checkpoint,
        baseline_path=args.baseline,
    )
    stage_started_wall = time.monotonic()
    debug_state = DebugDumpState(
        stage_ids=tuple(stage_ids),
        started_wall_seconds=stage_started_wall,
    )
    debug_state_lock = threading.Lock()
    debug_interval_seconds = max(0, int(args.debug_dump_interval_seconds))

    def _emit_dump(reason: str) -> None:
        with debug_state_lock:
            snapshot = DebugDumpState(
                stage_ids=debug_state.stage_ids,
                started_wall_seconds=debug_state.started_wall_seconds,
                attempts_started=debug_state.attempts_started,
                attempts_completed=debug_state.attempts_completed,
                active_stage_id=debug_state.active_stage_id,
                active_stage_started_wall_seconds=debug_state.active_stage_started_wall_seconds,
                active_stage_strictness=debug_state.active_stage_strictness,
                active_command=tuple(debug_state.active_command),
                last_analysis_state=debug_state.last_analysis_state,
                last_terminal_status=debug_state.last_terminal_status,
            )
        _emit_debug_dump(
            reason=reason,
            state=snapshot,
            paths=paths,
            step_summary_path=step_summary_path,
        )

    restore_signal_handler = install_signal_debug_dump_handler_fn(emit_dump_fn=_emit_dump)

    def _run_subprocess_with_debug(command: Sequence[str]) -> int:
        return run_subprocess_fn(
            command,
            heartbeat_interval_seconds=debug_interval_seconds,
            on_heartbeat=lambda: _emit_dump("interval"),
        )

    def _on_stage_start(
        stage_id: str, command: Sequence[str], strictness: str | None
    ) -> None:
        with debug_state_lock:
            _debug_dump_stage_start(
                state=debug_state,
                stage_id=stage_id,
                command=command,
                strictness=strictness,
            )

    def _on_stage_end(result: StageResult) -> None:
        with debug_state_lock:
            _debug_dump_stage_end(state=debug_state, result=result)

    with deadline_scope_factory():
        reset_run_observability_artifacts_fn(paths)
        try:
            strictness_by_stage = _parse_stage_strictness_profile(
                args.stage_strictness_profile
            )
            results = run_staged_fn(
                stage_ids=stage_ids,
                paths=paths,
                resume_on_timeout=args.resume_on_timeout,
                step_summary_path=step_summary_path,
                run_command_fn=_run_subprocess_with_debug,
                strictness_by_stage=strictness_by_stage,
                max_wall_seconds=(
                    int(args.max_wall_seconds)
                    if int(args.max_wall_seconds) > 0
                    else None
                ),
                finalize_reserve_seconds=max(0, int(args.finalize_reserve_seconds)),
                on_stage_start=_on_stage_start,
                on_stage_end=_on_stage_end,
            )
            trace_payload = write_obligation_trace_fn(paths.obligation_trace_json_path, results)
            append_markdown_summary_fn(paths.timeout_progress_md_path, trace_payload)
            append_markdown_summary_fn(paths.deadline_profile_md_path, trace_payload)
            append_lines_fn(step_summary_path, _obligation_trace_summary_lines(trace_payload))
            emit_stage_outputs_fn(github_output_path, results)
        finally:
            restore_signal_handler()
    return 0
