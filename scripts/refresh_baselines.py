from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from scripts.deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env
from gabion.analysis.timeout_context import check_deadline
from gabion.tooling import aspf_handoff
from gabion.tooling.governance_rules import load_governance_rules


OBSOLESCENCE_DELTA_PATH = Path("artifacts/out/test_obsolescence_delta.json")
ANNOTATION_DRIFT_DELTA_PATH = Path("artifacts/out/test_annotation_drift_delta.json")
AMBIGUITY_DELTA_PATH = Path("artifacts/out/ambiguity_delta.json")
DOCFLOW_DELTA_PATH = Path("artifacts/out/docflow_compliance_delta.json")
OBSOLESCENCE_STATE_PATH = Path("artifacts/out/test_obsolescence_state.json")
ANNOTATION_DRIFT_STATE_PATH = Path("artifacts/out/test_annotation_drift.json")
AMBIGUITY_STATE_PATH = Path("artifacts/out/ambiguity_state.json")
OBSOLESCENCE_BASELINE_PATH = Path("baselines/test_obsolescence_baseline.json")
ANNOTATION_DRIFT_BASELINE_PATH = Path("baselines/test_annotation_drift_baseline.json")
AMBIGUITY_BASELINE_PATH = Path("baselines/ambiguity_baseline.json")
DOCFLOW_BASELINE_PATH = Path("baselines/docflow_compliance_baseline.json")
DOCFLOW_CURRENT_PATH = Path("artifacts/out/docflow_compliance.json")
DEFAULT_CHECK_REPORT_PATH = Path("artifacts/audit_reports/dataflow_report.md")
DEFAULT_DEADLINE_PROFILE_PATH = Path("artifacts/out/deadline_profile.json")
FAILURE_ARTIFACT_PATH = Path("artifacts/out/refresh_baselines_failure.json")

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
    attempted_flags = [
        arg
        for arg in failure.command
        if arg.startswith("-") and arg != "-m"
    ]
    payload = {
        "attempted_command": failure.command,
        "attempted_flags": attempted_flags,
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
        f"- deadline profile json: {DEFAULT_DEADLINE_PROFILE_PATH}\n"
        f"- check report: {report_path}"
    )


def _refresh_subprocess_env(timeout_env: _RefreshLspTimeoutEnv) -> dict[str, str]:
    _ = timeout_env
    return dict(os.environ)


def _run_check(
    subcommand: list[str],
    timeout: int | None,
    timeout_env: _RefreshLspTimeoutEnv,
    *,
    report_path: Path = DEFAULT_CHECK_REPORT_PATH,
    extra: list[str] | None = None,
    run_fn=subprocess.run,
) -> None:
    timeout_text = f"{int(timeout_env.ticks) * int(timeout_env.tick_ns)}ns"
    cmd = [
        sys.executable,
        "-m",
        "gabion",
        "--timeout",
        timeout_text,
        "check",
        *subcommand,
        "--report",
        str(report_path),
    ]
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
    timeout_text = f"{int(timeout_env.ticks) * int(timeout_env.tick_ns)}ns"
    cmd = [
        sys.executable,
        "-m",
        "gabion",
        "--timeout",
        timeout_text,
        "docflow-delta-emit",
    ]
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


def _gate_enabled(*, gate_id: str) -> bool:
    policy = load_governance_rules().gates[gate_id]
    value = os.getenv(policy.env_flag, "").strip().lower()
    if policy.enabled_mode == "truthy_only":
        return value in {"1", "true", "yes", "on"}
    return value not in {"0", "false", "no", "off"}


def _policy_override_present() -> bool:
    rules = load_governance_rules()
    override_token = os.getenv(rules.override_token_env, "").strip()
    rationale = os.getenv("GABION_POLICY_OVERRIDE_RATIONALE", "").strip()
    return bool(override_token and rationale)


def _requires_block(gate_id: str, delta_value: int) -> bool:
    policy = load_governance_rules().gates[gate_id]
    if delta_value >= policy.severity.blocking_threshold:
        return not _policy_override_present()
    return False


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
    subcommand: list[str],
    path: Path,
    timeout: int | None,
    timeout_env: _RefreshLspTimeoutEnv,
    *,
    extra: list[str] | None = None,
    run_check_fn=_run_check,
) -> dict[str, object]:
    run_check_fn(
        subcommand,
        timeout,
        timeout_env,
        extra=extra,
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing delta output at {path}")
    return _load_json(path)


def _guard_obsolescence_delta(
    timeout: int | None,
    timeout_env: _RefreshLspTimeoutEnv,
    *,
    run_check_fn=_run_check,
) -> None:
    payload = _ensure_delta(
        [
            "obsolescence",
            "delta",
            "--baseline",
            str(OBSOLESCENCE_BASELINE_PATH),
        ],
        OBSOLESCENCE_DELTA_PATH,
        timeout,
        timeout_env,
        extra=_state_args(OBSOLESCENCE_STATE_PATH, "--state-in"),
        run_check_fn=run_check_fn,
    )
    opaque_delta = _get_nested(payload, ["summary", "opaque_evidence", "delta"])
    if _requires_block("obsolescence_opaque", opaque_delta):
        raise SystemExit(
            "Refusing to refresh obsolescence baseline: opaque evidence delta > 0."
        )
    if _gate_enabled(gate_id="obsolescence_unmapped"):
        unmapped_delta = _get_nested(payload, ["summary", "counts", "delta", "unmapped"])
        if _requires_block("obsolescence_unmapped", unmapped_delta):
            raise SystemExit(
                "Refusing to refresh obsolescence baseline: unmapped delta > 0."
            )


def _guard_annotation_drift_delta(
    timeout: int | None,
    timeout_env: _RefreshLspTimeoutEnv,
    *,
    run_check_fn=_run_check,
) -> None:
    if not _gate_enabled(gate_id="annotation_orphaned"):
        return
    payload = _ensure_delta(
        [
            "annotation-drift",
            "delta",
            "--baseline",
            str(ANNOTATION_DRIFT_BASELINE_PATH),
        ],
        ANNOTATION_DRIFT_DELTA_PATH,
        timeout,
        timeout_env,
        extra=_state_args(ANNOTATION_DRIFT_STATE_PATH, "--state-in"),
        run_check_fn=run_check_fn,
    )
    orphaned_delta = _get_nested(payload, ["summary", "delta", "orphaned"])
    if _requires_block("annotation_orphaned", orphaned_delta):
        raise SystemExit(
            "Refusing to refresh annotation drift baseline: orphaned delta > 0."
        )


def _guard_ambiguity_delta(
    timeout: int | None,
    timeout_env: _RefreshLspTimeoutEnv,
    *,
    run_check_fn=_run_check,
) -> None:
    if not _gate_enabled(gate_id="ambiguity"):
        return
    payload = _ensure_delta(
        [
            "ambiguity",
            "delta",
            "--baseline",
            str(AMBIGUITY_BASELINE_PATH),
        ],
        AMBIGUITY_DELTA_PATH,
        timeout,
        timeout_env,
        extra=_state_args(AMBIGUITY_STATE_PATH, "--state-in"),
        run_check_fn=run_check_fn,
    )
    total_delta = _get_nested(payload, ["summary", "total", "delta"])
    if _requires_block("ambiguity", total_delta):
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
    if _requires_block("docflow", contradicts):
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
            "--no-aspf-handoff",
            action="store_true",
            help="Disable ASPF cross-script handoff state/manifest emission.",
        )
        parser.add_argument(
            "--aspf-handoff-manifest",
            default="artifacts/out/aspf_handoff_manifest.json",
            help="Path to ASPF handoff manifest.",
        )
        parser.add_argument(
            "--aspf-handoff-session",
            default="",
            help="Session id used for ASPF handoff entries.",
        )
        parser.add_argument(
            "--aspf-state-root",
            default="artifacts/out/aspf_state",
            help="Root directory for ASPF serialized state objects.",
        )
        args = parser.parse_args(argv)
        timeout_env = _refresh_lsp_timeout_env(
            args.lsp_timeout_ticks,
            args.lsp_timeout_tick_ns,
        )
        handoff_enabled = not bool(args.no_aspf_handoff)
        handoff_session = str(args.aspf_handoff_session).strip() or os.getenv(
            "GABION_ASPF_HANDOFF_SESSION",
            "",
        ).strip()
        if handoff_enabled and not handoff_session:
            handoff_session = (
                f"session-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
            )
        handoff_manifest_path = Path(str(args.aspf_handoff_manifest))
        handoff_state_root = Path(str(args.aspf_state_root))
        raw_run_check_fn = run_check_fn

        def _run_check_with_handoff(
            subcommand: list[str],
            timeout: int | None,
            timeout_env: _RefreshLspTimeoutEnv,
            *,
            report_path: Path = DEFAULT_CHECK_REPORT_PATH,
            extra: list[str] | None = None,
            run_fn=subprocess.run,
        ) -> None:
            merged_extra = list(extra or [])
            if not handoff_enabled:
                raw_run_check_fn(
                    subcommand,
                    timeout,
                    timeout_env,
                    report_path=report_path,
                    extra=merged_extra,
                    run_fn=run_fn,
                )
                return

            step_name = ".".join(token for token in subcommand[:2] if token)
            step_id = f"refresh-baselines.{step_name or 'check'}"
            prepared = aspf_handoff.prepare_step(
                root=Path(".").resolve(),
                session_id=handoff_session,
                step_id=step_id,
                command_profile="refresh-baselines.check",
                manifest_path=handoff_manifest_path,
                state_root=handoff_state_root,
            )
            merged_extra.extend(aspf_handoff.aspf_cli_args(prepared))
            if raw_run_check_fn is not _run_check:
                try:
                    raw_run_check_fn(
                        subcommand,
                        timeout,
                        timeout_env,
                        report_path=report_path,
                        extra=merged_extra,
                        run_fn=run_fn,
                    )
                except Exception:
                    aspf_handoff.record_step(
                        manifest_path=prepared.manifest_path,
                        session_id=prepared.session_id,
                        sequence=prepared.sequence,
                        status="failed",
                        exit_code=1,
                        analysis_state="failed",
                    )
                    raise
                aspf_handoff.record_step(
                    manifest_path=prepared.manifest_path,
                    session_id=prepared.session_id,
                    sequence=prepared.sequence,
                    status="success",
                    exit_code=0,
                    analysis_state="succeeded",
                )
                return

            timeout_text = f"{int(timeout_env.ticks) * int(timeout_env.tick_ns)}ns"
            base_command = [
                sys.executable,
                "-m",
                "gabion",
                "--timeout",
                timeout_text,
                "check",
                *subcommand,
                "--report",
                str(report_path),
            ]
            base_command.extend(merged_extra)

            failure: RefreshBaselinesSubprocessFailure | None = None
            timeout_error: subprocess.TimeoutExpired | None = None

            def _run_command_with_env(command: Sequence[str]) -> int:
                nonlocal failure
                nonlocal timeout_error
                try:
                    completed = run_fn(
                        list(command),
                        check=False,
                        timeout=timeout,
                        env=_refresh_subprocess_env(timeout_env),
                    )
                except subprocess.TimeoutExpired as exc:
                    timeout_error = exc
                    return 124
                exit_code = int(getattr(completed, "returncode", 0))
                if exit_code != 0:
                    expected_artifacts = [
                        DEFAULT_DEADLINE_PROFILE_PATH,
                        report_path,
                    ]
                    expected_artifacts.extend(
                        Path(arg) for arg in command if not str(arg).startswith("-")
                    )
                    failure = RefreshBaselinesSubprocessFailure(
                        command=[str(part) for part in command],
                        timeout_seconds=timeout,
                        exit_code=exit_code,
                        expected_artifacts=expected_artifacts,
                    )
                return exit_code

            command_with_aspf = [*base_command]
            exit_code = _run_command_with_env(command_with_aspf)
            analysis_state = "succeeded" if exit_code == 0 else "failed"
            aspf_handoff.record_step(
                manifest_path=prepared.manifest_path,
                session_id=prepared.session_id,
                sequence=prepared.sequence,
                status="success" if exit_code == 0 else "failed",
                exit_code=exit_code,
                analysis_state=analysis_state,
            )
            if timeout_error is not None:
                raise timeout_error
            if failure is not None:
                raise failure
            if exit_code != 0:
                raise RefreshBaselinesSubprocessFailure(
                    command=command_with_aspf,
                    timeout_seconds=timeout,
                    exit_code=int(exit_code),
                    expected_artifacts=[DEFAULT_DEADLINE_PROFILE_PATH, report_path],
                )

        def _invoke_guard_with_optional_handoff(guard_fn, *args, **kwargs):
            try:
                params = inspect.signature(guard_fn).parameters
            except (TypeError, ValueError):
                params = {}
            if "run_check_fn" in params:
                return guard_fn(*args, **kwargs, run_check_fn=_run_check_with_handoff)
            return guard_fn(*args, **kwargs)

        if not (
            args.obsolescence
            or args.annotation_drift
            or args.ambiguity
            or args.docflow
            or args.all
        ):
            args.all = True

        if args.all or args.obsolescence:
            _invoke_guard_with_optional_handoff(
                guard_obsolescence_delta_fn,
                args.timeout,
                timeout_env,
            )
            _run_check_with_handoff(
                [
                    "obsolescence",
                    "baseline-write",
                    "--baseline",
                    str(OBSOLESCENCE_BASELINE_PATH),
                ],
                args.timeout,
                timeout_env,
                extra=_state_args(OBSOLESCENCE_STATE_PATH, "--state-in"),
            )
        if args.all or args.annotation_drift:
            _invoke_guard_with_optional_handoff(
                guard_annotation_drift_delta_fn,
                args.timeout,
                timeout_env,
            )
            _run_check_with_handoff(
                [
                    "annotation-drift",
                    "baseline-write",
                    "--baseline",
                    str(ANNOTATION_DRIFT_BASELINE_PATH),
                ],
                args.timeout,
                timeout_env,
                extra=_state_args(
                    ANNOTATION_DRIFT_STATE_PATH, "--state-in"
                ),
            )
        if args.all or args.ambiguity:
            _invoke_guard_with_optional_handoff(
                guard_ambiguity_delta_fn,
                args.timeout,
                timeout_env,
            )
            _run_check_with_handoff(
                [
                    "ambiguity",
                    "baseline-write",
                    "--baseline",
                    str(AMBIGUITY_BASELINE_PATH),
                ],
                args.timeout,
                timeout_env,
                extra=_state_args(AMBIGUITY_STATE_PATH, "--state-in"),
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
