from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:  # pragma: no cover - import form depends on invocation mode
    from scripts.deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env
from gabion.analysis.timeout_context import check_deadline


OBSOLESCENCE_DELTA_PATH = Path("artifacts/out/test_obsolescence_delta.json")
ANNOTATION_DRIFT_DELTA_PATH = Path("artifacts/out/test_annotation_drift_delta.json")
AMBIGUITY_DELTA_PATH = Path("artifacts/out/ambiguity_delta.json")
DOCFLOW_DELTA_PATH = Path("artifacts/out/docflow_compliance_delta.json")
OBSOLESCENCE_STATE_PATH = Path("artifacts/out/test_obsolescence_state.json")
ANNOTATION_DRIFT_STATE_PATH = Path("artifacts/out/test_annotation_drift.json")
AMBIGUITY_STATE_PATH = Path("artifacts/out/ambiguity_state.json")
DOCFLOW_BASELINE_PATH = Path("baselines/docflow_compliance_baseline.json")
DOCFLOW_CURRENT_PATH = Path("artifacts/out/docflow_compliance.json")
DEFAULT_CHECK_REPORT_PATH = Path("artifacts/audit_reports/dataflow_report.md")
DEFAULT_TIMEOUT_PROGRESS_PATH = Path("artifacts/audit_reports/timeout_progress.md")
DEFAULT_DEADLINE_PROFILE_PATH = Path("artifacts/out/deadline_profile.json")
DEFAULT_RESUME_CHECKPOINT_PATH = Path("artifacts/out/refresh_baselines_resume.json")
FAILURE_ARTIFACT_PATH = Path("artifacts/out/refresh_baselines_failure.json")

ENV_GATE_UNMAPPED = "GABION_GATE_UNMAPPED_DELTA"
ENV_GATE_ORPHANED = "GABION_GATE_ORPHANED_DELTA"
ENV_GATE_AMBIGUITY = "GABION_GATE_AMBIGUITY_DELTA"
_DEFAULT_TIMEOUT_TICKS = 120_000
_DEFAULT_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_TIMEOUT_BUDGET = DeadlineBudget(
    ticks=_DEFAULT_TIMEOUT_TICKS,
    tick_ns=_DEFAULT_TIMEOUT_TICK_NS,
)

_TIMEOUT_ENV_FLAGS = [
    "GABION_LSP_TIMEOUT_TICKS",
    "GABION_LSP_TIMEOUT_TICK_NS",
    "GABION_LSP_TIMEOUT_MS",
    "GABION_LSP_TIMEOUT_SECONDS",
]


@dataclass
class RefreshBaselinesSubprocessFailure(RuntimeError):
    command: list[str]
    timeout_seconds: int | None
    exit_code: int
    expected_artifacts: list[Path]


def _timeout_env_settings() -> dict[str, str | None]:
    return {name: os.getenv(name) for name in _TIMEOUT_ENV_FLAGS}


def _raise_subprocess_failure(
    error: subprocess.CalledProcessError,
    *,
    timeout: int | None,
    expected_artifacts: list[Path] | None = None,
) -> None:
    cmd = [str(part) for part in error.cmd] if isinstance(error.cmd, list) else [str(error.cmd)]
    raise RefreshBaselinesSubprocessFailure(
        command=cmd,
        timeout_seconds=timeout,
        exit_code=int(error.returncode),
        expected_artifacts=expected_artifacts or [],
    ) from error


def _clear_failure_artifact() -> None:
    if FAILURE_ARTIFACT_PATH.exists():
        FAILURE_ARTIFACT_PATH.unlink()


def _write_failure_artifact(
    failure: RefreshBaselinesSubprocessFailure,
) -> Path:
    payload = {
        "attempted_command": failure.command,
        "attempted_flags": [arg for arg in failure.command if arg.startswith("-")],
        "timeout_settings": {
            "cli_timeout_seconds": failure.timeout_seconds,
            "env": _timeout_env_settings(),
        },
        "exit_code": failure.exit_code,
        "expected_artifacts": {
            str(path): path.exists() for path in failure.expected_artifacts
        },
    }
    FAILURE_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FAILURE_ARTIFACT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return FAILURE_ARTIFACT_PATH


@dataclass(frozen=True)
class _RefreshLspTimeoutEnv:
    ticks: int
    tick_ns: int


# dataflow-bundle: timeout_tick_ns, timeout_ticks
def _refresh_lsp_timeout_env(
    timeout_ticks: int | None,
    timeout_tick_ns: int | None,
) -> _RefreshLspTimeoutEnv:
    budget = DeadlineBudget(
        ticks=_DEFAULT_TIMEOUT_TICKS if timeout_ticks is None else timeout_ticks,
        tick_ns=_DEFAULT_TIMEOUT_TICK_NS if timeout_tick_ns is None else timeout_tick_ns,
    )
    return _RefreshLspTimeoutEnv(
        ticks=budget.ticks,
        tick_ns=budget.tick_ns,
    )


def _deadline_scope():
    return deadline_scope_from_lsp_env(
        default_budget=_DEFAULT_TIMEOUT_BUDGET,
    )


def _default_resume_on_timeout() -> int:
    return 1


def _default_resume_checkpoint() -> Path | None:
    if os.getenv("CI", "").strip().lower() in {"1", "true", "yes", "on"}:
        return DEFAULT_RESUME_CHECKPOINT_PATH
    return None


def _format_run_check_failure(
    *,
    cmd: list[str],
    returncode: int,
    report_path: Path,
) -> str:
    command_display = subprocess.list2cmdline(cmd)
    return (
        "refresh_baselines check command failed.\n"
        f"Command: {command_display}\n"
        f"Exit code: {returncode}\n"
        "Expected diagnostics artifacts:\n"
        f"- timeout progress report: {DEFAULT_TIMEOUT_PROGRESS_PATH}\n"
        f"- deadline profile json: {DEFAULT_DEADLINE_PROFILE_PATH}\n"
        f"- check report: {report_path}"
    )


def _refresh_subprocess_env(timeout_env: _RefreshLspTimeoutEnv) -> dict[str, str]:
    env = dict(os.environ)
    env["GABION_DIRECT_RUN"] = "1"
    env["GABION_LSP_TIMEOUT_TICKS"] = str(timeout_env.ticks)
    env["GABION_LSP_TIMEOUT_TICK_NS"] = str(timeout_env.tick_ns)
    return env


def _run_check(
    flag: str,
    timeout: int | None,
    timeout_env: _RefreshLspTimeoutEnv,
    resume_on_timeout: int,
    *,
    resume_checkpoint: Path | None,
    report_path: Path = DEFAULT_CHECK_REPORT_PATH,
    extra: list[str] | None = None,
    run_fn=subprocess.run,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "gabion",
        "check",
        "--no-fail-on-violations",
        "--no-fail-on-type-ambiguities",
        flag,
        "--emit-timeout-progress-report",
        "--resume-on-timeout",
        str(resume_on_timeout),
    ]
    if resume_checkpoint is not None:
        cmd.extend(["--resume-checkpoint", str(resume_checkpoint)])
    if extra:
        cmd.extend(extra)
    try:
        run_fn(
            cmd,
            check=True,
            timeout=timeout,
            env=_refresh_subprocess_env(timeout_env),
        )
    except subprocess.CalledProcessError as error:
        expected_artifacts = [
            DEFAULT_TIMEOUT_PROGRESS_PATH,
            DEFAULT_DEADLINE_PROFILE_PATH,
            report_path,
        ]
        expected_artifacts.extend(
            Path(arg) for arg in (extra or []) if not arg.startswith("-")
        )
        _raise_subprocess_failure(
            error,
            timeout=timeout,
            expected_artifacts=expected_artifacts,
        )


def _run_docflow_delta_emit(
    timeout: int | None,
    timeout_env: _RefreshLspTimeoutEnv,
    *,
    run_fn=subprocess.run,
) -> None:
    cmd = [sys.executable, "scripts/docflow_delta_emit.py"]
    try:
        run_fn(
            cmd,
            check=True,
            timeout=timeout,
            env=_refresh_subprocess_env(timeout_env),
        )
    except subprocess.CalledProcessError as error:
        _raise_subprocess_failure(
            error,
            timeout=timeout,
            expected_artifacts=[DOCFLOW_DELTA_PATH, DOCFLOW_CURRENT_PATH],
        )


def _state_args(path: Path, flag: str) -> list[str]:
    if path.exists():
        return [flag, str(path)]
    return []


def _gate_enabled(env_flag: str) -> bool:
    value = os.getenv(env_flag, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_nested(payload: object, keys: list[str], default: int = 0) -> int:
    current = payload
    for key in keys:
        check_deadline()
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    try:
        return int(current) if current is not None else default
    except (TypeError, ValueError):
        return default


def _risk_entries(
    *,
    obsolescence_payload: dict[str, object] | None = None,
    annotation_payload: dict[str, object] | None = None,
    ambiguity_payload: dict[str, object] | None = None,
    docflow_payload: dict[str, object] | None = None,
) -> tuple[tuple[str, int], ...]:
    values: list[tuple[str, int]] = []
    if obsolescence_payload is not None:
        values.append(
            (
                "obsolescence.opaque",
                _get_nested(obsolescence_payload, ["summary", "opaque_evidence", "delta"]),
            )
        )
        values.append(
            (
                "obsolescence.unmapped",
                _get_nested(obsolescence_payload, ["summary", "counts", "delta", "unmapped"]),
            )
        )
    if annotation_payload is not None:
        values.append(
            ("annotation.orphaned", _get_nested(annotation_payload, ["summary", "delta", "orphaned"]))
        )
    if ambiguity_payload is not None:
        values.append(("ambiguity.total", _get_nested(ambiguity_payload, ["summary", "total", "delta"])))
    if docflow_payload is not None:
        values.append(
            ("docflow.contradicts", _get_nested(docflow_payload, ["summary", "delta", "contradicts"]))
        )
    return tuple(values)


def _ensure_delta(
    flag: str,
    path: Path,
    timeout: int | None,
    timeout_env: _RefreshLspTimeoutEnv,
    *,
    resume_on_timeout: int,
    resume_checkpoint: Path | None,
    extra: list[str] | None = None,
) -> dict[str, object]:
    _run_check(
        flag,
        timeout,
        timeout_env,
        resume_on_timeout,
        resume_checkpoint=resume_checkpoint,
        extra=extra,
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing delta output at {path}")
    return _load_json(path)


def _guard_obsolescence_delta(
    timeout: int | None,
    timeout_env: _RefreshLspTimeoutEnv,
    *,
    resume_on_timeout: int,
    resume_checkpoint: Path | None,
) -> None:
    payload = _ensure_delta(
        "--emit-test-obsolescence-delta",
        OBSOLESCENCE_DELTA_PATH,
        timeout,
        timeout_env,
        resume_on_timeout=resume_on_timeout,
        resume_checkpoint=resume_checkpoint,
        extra=_state_args(OBSOLESCENCE_STATE_PATH, "--test-obsolescence-state"),
    )
    opaque_delta = _get_nested(payload, ["summary", "opaque_evidence", "delta"])
    if opaque_delta > 0:
        raise SystemExit(
            "Refusing to refresh obsolescence baseline: opaque evidence delta > 0."
        )
    if _gate_enabled(ENV_GATE_UNMAPPED):
        unmapped_delta = _get_nested(payload, ["summary", "counts", "delta", "unmapped"])
        if unmapped_delta > 0:
            raise SystemExit(
                "Refusing to refresh obsolescence baseline: unmapped delta > 0."
            )


def _guard_annotation_drift_delta(
    timeout: int | None,
    timeout_env: _RefreshLspTimeoutEnv,
    *,
    resume_on_timeout: int,
    resume_checkpoint: Path | None,
) -> None:
    if not _gate_enabled(ENV_GATE_ORPHANED):
        return
    payload = _ensure_delta(
        "--emit-test-annotation-drift-delta",
        ANNOTATION_DRIFT_DELTA_PATH,
        timeout,
        timeout_env,
        resume_on_timeout=resume_on_timeout,
        resume_checkpoint=resume_checkpoint,
        extra=_state_args(ANNOTATION_DRIFT_STATE_PATH, "--test-annotation-drift-state"),
    )
    orphaned_delta = _get_nested(payload, ["summary", "delta", "orphaned"])
    if orphaned_delta > 0:
        raise SystemExit(
            "Refusing to refresh annotation drift baseline: orphaned delta > 0."
        )


def _guard_ambiguity_delta(
    timeout: int | None,
    timeout_env: _RefreshLspTimeoutEnv,
    *,
    resume_on_timeout: int,
    resume_checkpoint: Path | None,
) -> None:
    if not _gate_enabled(ENV_GATE_AMBIGUITY):
        return
    payload = _ensure_delta(
        "--emit-ambiguity-delta",
        AMBIGUITY_DELTA_PATH,
        timeout,
        timeout_env,
        resume_on_timeout=resume_on_timeout,
        resume_checkpoint=resume_checkpoint,
        extra=_state_args(AMBIGUITY_STATE_PATH, "--ambiguity-state"),
    )
    total_delta = _get_nested(payload, ["summary", "total", "delta"])
    if total_delta > 0:
        raise SystemExit(
            "Refusing to refresh ambiguity baseline: ambiguity delta > 0."
        )


def _guard_docflow_delta(
    timeout: int | None,
    timeout_env: _RefreshLspTimeoutEnv,
) -> None:
    _run_docflow_delta_emit(timeout, timeout_env)
    if not DOCFLOW_DELTA_PATH.exists():
        raise FileNotFoundError(
            f"Missing docflow delta output at {DOCFLOW_DELTA_PATH}"
        )
    payload = _load_json(DOCFLOW_DELTA_PATH)
    if payload.get("baseline_missing"):
        return
    contradicts = _get_nested(payload, ["summary", "delta", "contradicts"])
    if contradicts > 0:
        raise SystemExit(
            "Refusing to refresh docflow baseline: contradictions delta > 0."
        )


def main(
    argv: list[str] | None = None,
    *,
    deadline_scope_factory=_deadline_scope,
    run_check_fn=_run_check,
    guard_obsolescence_delta_fn=_guard_obsolescence_delta,
    guard_annotation_drift_delta_fn=_guard_annotation_drift_delta,
    guard_ambiguity_delta_fn=_guard_ambiguity_delta,
    guard_docflow_delta_fn=_guard_docflow_delta,
) -> int:
    with deadline_scope_factory():
        _clear_failure_artifact()
        parser = argparse.ArgumentParser(
            description="Refresh baseline carriers via gabion check.",
        )
        parser.add_argument(
            "--obsolescence",
            action="store_true",
            help="Refresh baselines/test_obsolescence_baseline.json",
        )
        parser.add_argument(
            "--annotation-drift",
            action="store_true",
            help="Refresh baselines/test_annotation_drift_baseline.json",
        )
        parser.add_argument(
            "--ambiguity",
            action="store_true",
            help="Refresh baselines/ambiguity_baseline.json",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            help="Refresh all baselines (default when no flags provided).",
        )
        parser.add_argument(
            "--docflow",
            action="store_true",
            help="Refresh baselines/docflow_compliance_baseline.json",
        )
        parser.add_argument(
            "--timeout",
            type=int,
            default=None,
            help="Seconds to wait for each gabion check (default: no timeout).",
        )
        parser.add_argument(
            "--lsp-timeout-ticks",
            type=int,
            default=None,
            help=(
                "Override subprocess GABION_LSP_TIMEOUT_TICKS for refresh only "
                f"(default: {_DEFAULT_TIMEOUT_TICKS})."
            ),
        )
        parser.add_argument(
            "--lsp-timeout-tick-ns",
            type=int,
            default=None,
            help=(
                "Override subprocess GABION_LSP_TIMEOUT_TICK_NS for refresh only "
                f"(default: {_DEFAULT_TIMEOUT_TICK_NS})."
            ),
        )
        parser.add_argument(
            "--resume-on-timeout",
            type=int,
            default=_default_resume_on_timeout(),
            help="Resume budget passed to gabion check --resume-on-timeout (default: 1).",
        )
        parser.add_argument(
            "--resume-checkpoint",
            default=_default_resume_checkpoint(),
            help=(
                "Checkpoint path passed to gabion check --resume-checkpoint. "
                "Defaults to artifacts/out/refresh_baselines_resume.json in CI and disabled locally; "
                "set to 'none' to disable explicitly."
            ),
        )
        args = parser.parse_args(argv)
        timeout_env = _refresh_lsp_timeout_env(
            args.lsp_timeout_ticks,
            args.lsp_timeout_tick_ns,
        )
        resume_checkpoint = args.resume_checkpoint
        if isinstance(resume_checkpoint, str) and resume_checkpoint.strip().lower() == "none":
            resume_checkpoint = None
        if isinstance(resume_checkpoint, str):
            resume_checkpoint = Path(resume_checkpoint)

        if not (
            args.obsolescence
            or args.annotation_drift
            or args.ambiguity
            or args.docflow
            or args.all
        ):
            args.all = True

        if args.all or args.obsolescence:
            guard_obsolescence_delta_fn(
                args.timeout,
                timeout_env,
                resume_on_timeout=args.resume_on_timeout,
                resume_checkpoint=resume_checkpoint,
            )
            run_check_fn(
                "--write-test-obsolescence-baseline",
                args.timeout,
                timeout_env,
                args.resume_on_timeout,
                resume_checkpoint=resume_checkpoint,
                extra=_state_args(OBSOLESCENCE_STATE_PATH, "--test-obsolescence-state"),
            )
        if args.all or args.annotation_drift:
            guard_annotation_drift_delta_fn(
                args.timeout,
                timeout_env,
                resume_on_timeout=args.resume_on_timeout,
                resume_checkpoint=resume_checkpoint,
            )
            run_check_fn(
                "--write-test-annotation-drift-baseline",
                args.timeout,
                timeout_env,
                args.resume_on_timeout,
                resume_checkpoint=resume_checkpoint,
                extra=_state_args(
                    ANNOTATION_DRIFT_STATE_PATH, "--test-annotation-drift-state"
                ),
            )
        if args.all or args.ambiguity:
            guard_ambiguity_delta_fn(
                args.timeout,
                timeout_env,
                resume_on_timeout=args.resume_on_timeout,
                resume_checkpoint=resume_checkpoint,
            )
            run_check_fn(
                "--write-ambiguity-baseline",
                args.timeout,
                timeout_env,
                args.resume_on_timeout,
                resume_checkpoint=resume_checkpoint,
                extra=_state_args(AMBIGUITY_STATE_PATH, "--ambiguity-state"),
            )
        if args.all or args.docflow:
            guard_docflow_delta_fn(args.timeout, timeout_env)
            if not DOCFLOW_CURRENT_PATH.exists():
                raise FileNotFoundError(
                    f"Missing docflow compliance output at {DOCFLOW_CURRENT_PATH}"
                )
            DOCFLOW_BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(DOCFLOW_CURRENT_PATH, DOCFLOW_BASELINE_PATH)

        _clear_failure_artifact()
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RefreshBaselinesSubprocessFailure as failure:
        artifact_path = _write_failure_artifact(failure)
        print(f"refresh_baselines failure artifact: {artifact_path}", file=sys.stderr)
        raise SystemExit(1)
