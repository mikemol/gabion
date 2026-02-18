#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
        description="Run one dataflow-audit CI stage with deterministic outputs/artifacts."
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
        _emit_stage_outputs(github_output_path, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
