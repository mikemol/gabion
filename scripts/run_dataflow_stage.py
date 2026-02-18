#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from gabion.analysis.timeout_context import deadline_loop_iter

try:  # pragma: no cover - import form depends on invocation mode
    from scripts.deadline_runtime import deadline_scope_from_lsp_env
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from deadline_runtime import deadline_scope_from_lsp_env

_STAGE_SEQUENCE: tuple[str, ...] = ("a", "b", "c")
_DELTA_GATE_SCRIPTS: tuple[str, ...] = (
    "scripts/obsolescence_delta_gate.py",
    "scripts/obsolescence_delta_unmapped_gate.py",
    "scripts/annotation_drift_orphaned_gate.py",
    "scripts/ambiguity_delta_gate.py",
)


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


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): payload[key] for key in payload}


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
    rows.sort(key=lambda row: str(row["id"]))
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
    obligations.sort(key=lambda row: (str(row.get("id", "")), str(row.get("stage_id", ""))))
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
        "incompleteness_markers": sorted(markers),
        "summary": summary,
        "obligations": obligations,
    }


def _write_obligation_trace(path: Path, results: Sequence[StageResult]) -> dict[str, object]:
    payload = _obligation_trace_payload(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def _check_command(
    *,
    paths: StagePaths,
    resume_on_timeout: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "gabion",
        "check",
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


def run_stage(
    *,
    stage_id: str,
    paths: StagePaths,
    resume_on_timeout: int,
    step_summary_path: Path | None,
    run_command_fn: Callable[[Sequence[str]], int],
) -> StageResult:
    paths.report_path.parent.mkdir(parents=True, exist_ok=True)
    paths.deadline_profile_json_path.parent.mkdir(parents=True, exist_ok=True)
    resume_metrics_line = _resume_checkpoint_metrics_line(paths.resume_checkpoint_path)
    print(f"stage {stage_id.upper()}: {resume_metrics_line}")
    _append_lines(step_summary_path, [f"- stage {stage_id.upper()}: {resume_metrics_line}"])
    exit_code = int(
        run_command_fn(
            _check_command(
                paths=paths,
                resume_on_timeout=resume_on_timeout,
            )
        )
    )
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

    timeout_payload = _load_json_object(paths.timeout_progress_json_path)
    obligation_rows, incompleteness_markers = _obligation_rows_from_timeout_payload(
        stage_id=stage_id,
        analysis_state=analysis_state,
        timeout_payload=timeout_payload,
    )

    stage_upper = stage_id.upper()
    print(
        f"stage {stage_upper}: exit={exit_code} "
        f"analysis_state={analysis_state} {metrics_line}"
    )
    _append_lines(
        step_summary_path,
        [
            (
                f"- stage {stage_upper}: exit=`{exit_code}`, "
                f"state=`{analysis_state}`, {metrics_line}"
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




def _gate_command(script_path: str) -> list[str]:
    return [sys.executable, script_path]


def _run_delta_gates(run_gate_fn: Callable[[Sequence[str]], int]) -> int:
    for script_path in deadline_loop_iter(_DELTA_GATE_SCRIPTS):
        gate_exit = int(run_gate_fn(_gate_command(script_path)))
        if gate_exit != 0:
            print(f"delta gate failed: {script_path} (exit {gate_exit})")
            return gate_exit
    return 0

def run_staged(
    *,
    stage_ids: Sequence[str],
    paths: StagePaths,
    resume_on_timeout: int,
    step_summary_path: Path | None,
    run_command_fn: Callable[[Sequence[str]], int],
    run_gate_fn: Callable[[Sequence[str]], int] | None = None,
) -> list[StageResult]:
    results: list[StageResult] = []
    for stage_id in deadline_loop_iter(stage_ids):
        result = run_stage(
            stage_id=stage_id,
            paths=paths,
            resume_on_timeout=resume_on_timeout,
            step_summary_path=step_summary_path,
            run_command_fn=run_command_fn,
        )
        results.append(result)
        if result.exit_code == 0:
            gate_runner = _run_subprocess if run_gate_fn is None else run_gate_fn
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one dataflow grammar CI stage with deterministic outputs/artifacts."
    )
    parser.add_argument("--stage-id", default="a", choices=_STAGE_SEQUENCE)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="Number of staged retries to run (uses a->b->c from --stage-id).",
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
        default=1,
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
    return parser.parse_args()


def _run_subprocess(command: Sequence[str]) -> int:
    try:
        completed = subprocess.run(command, check=False)
    except OSError:
        return 127
    return int(completed.returncode)


def main() -> int:
    args = _parse_args()
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
        print("No stages requested; max-attempts must be > 0.")
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
    with deadline_scope_from_lsp_env():
        results = run_staged(
            stage_ids=stage_ids,
            paths=paths,
            resume_on_timeout=args.resume_on_timeout,
            step_summary_path=step_summary_path,
            run_command_fn=_run_subprocess,
        )
        trace_payload = _write_obligation_trace(paths.obligation_trace_json_path, results)
        _append_markdown_summary(paths.timeout_progress_md_path, trace_payload)
        _append_markdown_summary(paths.deadline_profile_md_path, trace_payload)
        _append_lines(step_summary_path, _obligation_trace_summary_lines(trace_payload))
        _emit_stage_outputs(github_output_path, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
