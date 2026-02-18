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

ENV_GATE_UNMAPPED = "GABION_GATE_UNMAPPED_DELTA"
ENV_GATE_ORPHANED = "GABION_GATE_ORPHANED_DELTA"
ENV_GATE_AMBIGUITY = "GABION_GATE_AMBIGUITY_DELTA"
_DEFAULT_TIMEOUT_TICKS = 120_000
_DEFAULT_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_TIMEOUT_BUDGET = DeadlineBudget(
    ticks=_DEFAULT_TIMEOUT_TICKS,
    tick_ns=_DEFAULT_TIMEOUT_TICK_NS,
)
FAILURE_ARTIFACT_PATH = Path("artifacts/out/refresh_baselines_failure.json")
RESUME_CHECKPOINT_DIR = Path("artifacts/out/refresh_baselines_resume")
_TIMEOUT_ENV_KEYS = (
    "GABION_LSP_TIMEOUT_TICKS",
    "GABION_LSP_TIMEOUT_TICK_NS",
    "GABION_LSP_TIMEOUT_MS",
    "GABION_LSP_TIMEOUT_SECONDS",
)


@dataclass(frozen=True)
class _RefreshCommand:
    # dataflow-bundle: cmd, env, timeout, expected_artifacts
    label: str
    cmd: list[str]
    env: dict[str, str]
    timeout: int | None
    expected_artifacts: list[str]


def _refresh_resume_checkpoint(label: str) -> Path:
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)
    return RESUME_CHECKPOINT_DIR / f"{sanitized}.json"


def _timeout_settings_from_env(env: dict[str, str]) -> dict[str, str | None]:
    return {key: env.get(key) for key in _TIMEOUT_ENV_KEYS}


def _format_command(cmd: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([token]) for token in cmd)


def _write_failure_artifact(
    run: _RefreshCommand,
    *,
    failure_type: str,
    exit_code: int | None,
    stderr_tail: str | None = None,
) -> None:
    FAILURE_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "failure_type": failure_type,
        "label": run.label,
        "command": run.cmd,
        "command_text": _format_command(run.cmd),
        "exit_code": exit_code,
        "timeout_seconds": run.timeout,
        "env_timeout_settings": _timeout_settings_from_env(run.env),
        "expected_artifact_paths": run.expected_artifacts,
    }
    if stderr_tail:
        payload["stderr_tail"] = stderr_tail
    FAILURE_ARTIFACT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _clear_failure_artifact() -> None:
    FAILURE_ARTIFACT_PATH.unlink(missing_ok=True)


def _emit_remediation_snippets(run: _RefreshCommand) -> None:
    print(
        "Refresh baseline subprocess failed; remediation commands:",
        file=sys.stderr,
    )
    print(
        f"  GABION_DIRECT_RUN={run.env.get('GABION_DIRECT_RUN', '1')}",
        file=sys.stderr,
    )
    print(
        f"  {' '.join(f'{k}={v}' for k, v in _timeout_settings_from_env(run.env).items() if v)} {_format_command(run.cmd)}",
        file=sys.stderr,
    )
    if run.timeout is not None:
        print(
            f"  python scripts/refresh_baselines.py --timeout {run.timeout} --all",
            file=sys.stderr,
        )


def _run_refresh_command(run: _RefreshCommand) -> None:
    try:
        subprocess.run(
            run.cmd,
            check=True,
            timeout=run.timeout,
            env=run.env,
        )
        _clear_failure_artifact()
    except subprocess.TimeoutExpired as exc:
        stderr_tail = ""
        if isinstance(exc.stderr, bytes):
            stderr_tail = exc.stderr.decode("utf-8", errors="replace")[-2000:]
        elif isinstance(exc.stderr, str):
            stderr_tail = exc.stderr[-2000:]
        _write_failure_artifact(
            run,
            failure_type="TimeoutExpired",
            exit_code=None,
            stderr_tail=stderr_tail or None,
        )
        _emit_remediation_snippets(run)
        raise
    except subprocess.CalledProcessError as exc:
        stderr_tail = ""
        if isinstance(exc.stderr, bytes):
            stderr_tail = exc.stderr.decode("utf-8", errors="replace")[-2000:]
        elif isinstance(exc.stderr, str):
            stderr_tail = exc.stderr[-2000:]
        _write_failure_artifact(
            run,
            failure_type="CalledProcessError",
            exit_code=exc.returncode,
            stderr_tail=stderr_tail or None,
        )
        _emit_remediation_snippets(run)
        raise


def _deadline_scope():
    return deadline_scope_from_lsp_env(
        default_budget=_DEFAULT_TIMEOUT_BUDGET,
    )


def _run_check(flag: str, timeout: int | None, extra: list[str] | None = None) -> None:
    cmd = [
        sys.executable,
        "-m",
        "gabion",
        "check",
        "--no-fail-on-violations",
        "--no-fail-on-type-ambiguities",
        flag,
    ]
    if extra:
        cmd.extend(extra)
    resume_checkpoint = _refresh_resume_checkpoint(flag)
    resume_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    cmd.extend(
        [
            "--emit-timeout-progress-report",
            "--resume-checkpoint",
            str(resume_checkpoint),
            "--resume-on-timeout",
            "1",
        ]
    )
    env = dict(os.environ)
    env["GABION_DIRECT_RUN"] = "1"
    expected_artifacts: list[str] = []
    if flag == "--emit-test-obsolescence-delta":
        expected_artifacts.append(str(OBSOLESCENCE_DELTA_PATH))
    elif flag == "--emit-test-annotation-drift-delta":
        expected_artifacts.append(str(ANNOTATION_DRIFT_DELTA_PATH))
    elif flag == "--emit-ambiguity-delta":
        expected_artifacts.append(str(AMBIGUITY_DELTA_PATH))
    elif flag == "--write-test-obsolescence-baseline":
        expected_artifacts.append("baselines/test_obsolescence_baseline.json")
    elif flag == "--write-test-annotation-drift-baseline":
        expected_artifacts.append("baselines/test_annotation_drift_baseline.json")
    elif flag == "--write-ambiguity-baseline":
        expected_artifacts.append("baselines/ambiguity_baseline.json")
    _run_refresh_command(
        _RefreshCommand(
            label=f"gabion_check:{flag}",
            cmd=cmd,
            env=env,
            timeout=timeout,
            expected_artifacts=expected_artifacts,
        )
    )


def _run_docflow_delta_emit(timeout: int | None) -> None:
    env = dict(os.environ)
    env.setdefault("GABION_DIRECT_RUN", "1")
    _run_refresh_command(
        _RefreshCommand(
            label="docflow_delta_emit",
            cmd=[sys.executable, "scripts/docflow_delta_emit.py"],
            env=env,
            timeout=timeout,
            expected_artifacts=[str(DOCFLOW_DELTA_PATH)],
        )
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


def _ensure_delta(
    flag: str,
    path: Path,
    timeout: int | None,
    *,
    extra: list[str] | None = None,
) -> dict[str, object]:
    _run_check(flag, timeout, extra)
    if not path.exists():
        raise FileNotFoundError(f"Missing delta output at {path}")
    return _load_json(path)


def _guard_obsolescence_delta(timeout: int | None) -> None:
    payload = _ensure_delta(
        "--emit-test-obsolescence-delta",
        OBSOLESCENCE_DELTA_PATH,
        timeout,
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


def _guard_annotation_drift_delta(timeout: int | None) -> None:
    if not _gate_enabled(ENV_GATE_ORPHANED):
        return
    payload = _ensure_delta(
        "--emit-test-annotation-drift-delta",
        ANNOTATION_DRIFT_DELTA_PATH,
        timeout,
        extra=_state_args(ANNOTATION_DRIFT_STATE_PATH, "--test-annotation-drift-state"),
    )
    orphaned_delta = _get_nested(payload, ["summary", "delta", "orphaned"])
    if orphaned_delta > 0:
        raise SystemExit(
            "Refusing to refresh annotation drift baseline: orphaned delta > 0."
        )


def _guard_ambiguity_delta(timeout: int | None) -> None:
    if not _gate_enabled(ENV_GATE_AMBIGUITY):
        return
    payload = _ensure_delta(
        "--emit-ambiguity-delta",
        AMBIGUITY_DELTA_PATH,
        timeout,
        extra=_state_args(AMBIGUITY_STATE_PATH, "--ambiguity-state"),
    )
    total_delta = _get_nested(payload, ["summary", "total", "delta"])
    if total_delta > 0:
        raise SystemExit(
            "Refusing to refresh ambiguity baseline: ambiguity delta > 0."
        )


def _guard_docflow_delta(timeout: int | None) -> None:
    _run_docflow_delta_emit(timeout)
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


def main() -> int:
    with _deadline_scope():
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
        args = parser.parse_args()

        if not (
            args.obsolescence
            or args.annotation_drift
            or args.ambiguity
            or args.docflow
            or args.all
        ):
            args.all = True

        if args.all or args.obsolescence:
            _guard_obsolescence_delta(args.timeout)
            _run_check(
                "--write-test-obsolescence-baseline",
                args.timeout,
                _state_args(OBSOLESCENCE_STATE_PATH, "--test-obsolescence-state"),
            )
        if args.all or args.annotation_drift:
            _guard_annotation_drift_delta(args.timeout)
            _run_check(
                "--write-test-annotation-drift-baseline",
                args.timeout,
                _state_args(
                    ANNOTATION_DRIFT_STATE_PATH, "--test-annotation-drift-state"
                ),
            )
        if args.all or args.ambiguity:
            _guard_ambiguity_delta(args.timeout)
            _run_check(
                "--write-ambiguity-baseline",
                args.timeout,
                _state_args(AMBIGUITY_STATE_PATH, "--ambiguity-state"),
            )
        if args.all or args.docflow:
            _guard_docflow_delta(args.timeout)
            if not DOCFLOW_CURRENT_PATH.exists():
                raise FileNotFoundError(
                    f"Missing docflow compliance output at {DOCFLOW_CURRENT_PATH}"
                )
            DOCFLOW_BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(DOCFLOW_CURRENT_PATH, DOCFLOW_BASELINE_PATH)

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
