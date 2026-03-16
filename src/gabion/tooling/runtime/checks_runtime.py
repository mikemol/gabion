from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
from typing import Protocol


@dataclass(frozen=True)
class ToolingStep:
    label: str
    command: tuple[str, ...] = ()
    timed: bool = False
    env: tuple[tuple[str, str], ...] = ()
    mode: str = "all"


class CiChecksOptions(Protocol):
    python_bin: Path
    pr_base_sha: str
    pr_head_sha: str
    run_extended_checks: bool
    before_sha: str
    after_sha: str
    timeout_ns: str
    aspf_handoff_enabled: bool
    aspf_handoff_manifest: Path
    aspf_handoff_session: str
    aspf_state_root: Path
    step_timing_run_id: str
    step_timing_artifact: Path


@dataclass(frozen=True)
class ChecksCommandOptions:
    root: Path
    python_bin: Path
    run_docflow: bool
    run_dataflow: bool
    run_tests: bool
    run_status_watch: bool
    docflow_mode: str
    aspf_handoff_enabled: bool
    aspf_handoff_manifest: Path
    aspf_handoff_session: str
    aspf_state_root: Path
    status_watch_branch: str
    status_watch_workflow: str
    list_only: bool


def packet_enforce_command(options: CiChecksOptions) -> tuple[str, ...]:
    args = [
        str(options.python_bin),
        "scripts/policy/docflow_packet_enforce.py",
        "--root",
        ".",
        "--packets",
        "artifacts/out/docflow_warning_doc_packets.json",
        "--baseline",
        "docs/baselines/docflow_packet_baseline.json",
        "--out",
        "artifacts/out/docflow_packet_enforcement.json",
        "--debt-out",
        "artifacts/out/docflow_packet_debt_ledger.json",
        "--max-age-days",
        "14",
        "--check",
        "--run-proving-tests",
    ]
    if options.before_sha and options.before_sha != "0000000000000000000000000000000000000000":
        args.extend(["--base-sha", options.before_sha, "--head-sha", options.after_sha])
    return tuple(args)


def delta_bundle_command(options: CiChecksOptions) -> tuple[str, ...]:
    inner = [
        str(options.python_bin),
        "-m",
        "gabion",
        "--carrier",
        "direct",
        "--timeout",
        os.environ.get("GABION_DELTA_BUNDLE_TIMEOUT_NS", options.timeout_ns),
        "check",
        "delta-bundle",
    ]
    if not options.aspf_handoff_enabled:
        return tuple(inner)
    return (
        str(options.python_bin),
        "scripts/misc/aspf_handoff.py",
        "run",
        "--root",
        ".",
        "--session-id",
        options.aspf_handoff_session,
        "--step-id",
        "ci.checks.delta-bundle",
        "--command-profile",
        "ci.checks.delta-bundle",
        "--manifest",
        str(options.aspf_handoff_manifest),
        "--state-root",
        str(options.aspf_state_root),
        "--",
        *inner,
    )


def build_ci_checks_steps(options: CiChecksOptions) -> tuple[ToolingStep, ...]:
    scanner_command = [
        str(options.python_bin),
        "scripts/policy/policy_scanner_suite.py",
        "--root",
        ".",
        "--out-dir",
        "artifacts/out",
    ]
    if options.pr_base_sha and options.pr_head_sha:
        scanner_command.extend(["--base-sha", options.pr_base_sha, "--head-sha", options.pr_head_sha])
    steps = [
        ToolingStep(
            label="checks: policy_check --workflows",
            command=(str(options.python_bin), "-m", "scripts.policy.policy_check", "--workflows"),
            timed=True,
            mode="checks",
        ),
        ToolingStep(
            label="checks: policy_check --ambiguity-contract",
            command=(str(options.python_bin), "-m", "scripts.policy.policy_check", "--ambiguity-contract"),
            timed=True,
            mode="checks",
        ),
        ToolingStep(
            label="checks: policy_check --tier2-residue-contract",
            command=(str(options.python_bin), "-m", "scripts.policy.policy_check", "--tier2-residue-contract"),
            timed=True,
            mode="checks",
        ),
        ToolingStep(
            label="checks: docflow",
            command=(
                str(options.python_bin),
                "-m",
                "gabion",
                "docflow",
                "--root",
                ".",
                "--fail-on-violations",
                "--sppf-gh-ref-mode",
                "required",
            ),
            timed=True,
            mode="checks",
        ),
        ToolingStep(
            label="checks: docflow packetize",
            command=(
                str(options.python_bin),
                "scripts/policy/docflow_packetize.py",
                "--root",
                ".",
                "--compliance",
                "artifacts/out/docflow_compliance.json",
                "--section-reviews",
                "artifacts/out/docflow_section_reviews.json",
                "--out",
                "artifacts/out/docflow_warning_doc_packets.json",
                "--summary-out",
                "artifacts/out/docflow_warning_doc_packet_summary.json",
            ),
            mode="checks",
        ),
        ToolingStep(
            label="checks: docflow packet enforce",
            command=packet_enforce_command(options),
            mode="checks",
        ),
        ToolingStep(
            label="checks: sppf_status_audit",
            command=(str(options.python_bin), "-m", "scripts.sppf.sppf_status_audit", "--root", "."),
            mode="checks",
        ),
        ToolingStep(
            label="checks: extract_test_evidence",
            command=(str(options.python_bin), "-m", "scripts.misc.extract_test_evidence", "--root", ".", "--tests", "tests", "--out", "out/test_evidence.json"),
            mode="checks",
        ),
        ToolingStep(
            label="checks: extract_test_behavior",
            command=(str(options.python_bin), "-m", "scripts.misc.extract_test_behavior", "--root", ".", "--tests", "tests", "--out", "out/test_behavior.json"),
            mode="checks",
        ),
        ToolingStep(
            label="checks: evidence drift diff (strict)",
            command=("git", "diff", "--exit-code", "out/test_evidence.json"),
            mode="checks",
        ),
        ToolingStep(
            label="checks: behavior drift diff (strict)",
            command=("git", "diff", "--exit-code", "out/test_behavior.json"),
            mode="checks",
        ),
        ToolingStep(
            label="checks: policy_check --workflows --output artifacts/out/policy_check_result.json",
            command=(
                str(options.python_bin),
                "scripts/policy/policy_check.py",
                "--workflows",
                "--output",
                "artifacts/out/policy_check_result.json",
            ),
            mode="checks",
        ),
        ToolingStep(
            label="checks: structural_hash_policy_check",
            command=(str(options.python_bin), "scripts/policy/structural_hash_policy_check.py", "--root", "."),
            mode="checks",
        ),
        ToolingStep(
            label="checks: deprecated_nonerasability_policy_check",
            command=(
                str(options.python_bin),
                "scripts/policy/deprecated_nonerasability_policy_check.py",
                "--baseline",
                "out/deprecated_fibers_baseline.json",
                "--current",
                "out/deprecated_fibers_current.json",
                "--output",
                "artifacts/out/deprecated_nonerasability_result.json",
            ),
            mode="checks",
        ),
        ToolingStep(
            label="checks: policy scanner suite",
            command=tuple(scanner_command),
            mode="checks",
        ),
        ToolingStep(
            label="checks: controller drift audit",
            command=(str(options.python_bin), "scripts/governance/governance_controller_audit.py", "--out", "artifacts/out/controller_drift.json"),
            mode="checks",
        ),
        ToolingStep(
            label="checks: override lifecycle record emit",
            command=(str(options.python_bin), "scripts/ci/ci_override_record_emit.py", "--out", "artifacts/out/governance_override_record.json"),
            mode="checks",
        ),
        ToolingStep(
            label="checks: controller drift gate",
            command=(
                str(options.python_bin),
                "scripts/ci/ci_controller_drift_gate.py",
                "--drift-artifact",
                "artifacts/out/controller_drift.json",
                "--override-record",
                "artifacts/out/governance_override_record.json",
                "--history",
                "artifacts/out/controller_drift_gate_history.json",
            ),
            mode="checks",
        ),
        ToolingStep(
            label="checks: lsp parity gate",
            command=(str(options.python_bin), "-m", "gabion", "lsp-parity-gate", "--command", "gabion.check"),
            mode="checks",
        ),
        ToolingStep(
            label="checks: pytest --cov",
            command=(
                str(options.python_bin),
                "-m",
                "pytest",
                "--cov=src/gabion",
                "--cov-branch",
                "--cov-report=term-missing",
                "--cov-report=xml:artifacts/test_runs/coverage.xml",
                "--cov-report=html:artifacts/test_runs/htmlcov",
                "--cov-fail-under=100",
                "--junitxml",
                "artifacts/test_runs/junit.xml",
                "--log-file",
                "artifacts/test_runs/pytest.log",
                "--log-file-level=INFO",
            ),
            timed=True,
            mode="checks",
        ),
        ToolingStep(
            label="checks: check delta-bundle",
            command=delta_bundle_command(options),
            timed=True,
            mode="checks",
        ),
        ToolingStep(
            label="checks: check delta-gates",
            command=(str(options.python_bin), "-m", "gabion", "--carrier", "direct", "--timeout", options.timeout_ns, "check", "delta-gates"),
            timed=True,
            mode="checks",
        ),
        ToolingStep(
            label="checks: governance telemetry emit",
            command=(
                str(options.python_bin),
                "scripts/governance/governance_telemetry_emit.py",
                "--run-id",
                options.step_timing_run_id,
                "--timings",
                str(options.step_timing_artifact),
                "--history",
                "artifacts/out/governance_telemetry_history.json",
                "--json-out",
                "artifacts/out/governance_telemetry.json",
                "--md-out",
                "artifacts/audit_reports/governance_telemetry.md",
            ),
            mode="checks",
        ),
    ]
    if options.run_extended_checks:
        steps.extend(
            (
                ToolingStep(
                    label="checks(ext): order_lifetime_check",
                    command=(str(options.python_bin), "scripts/misc/order_lifetime_check.py", "--root", "."),
                    mode="checks",
                ),
                ToolingStep(
                    label="checks(ext): complexity_audit --fail-on-regression",
                    command=(str(options.python_bin), "scripts/misc/complexity_audit.py", "--root", ".", "--fail-on-regression"),
                    timed=True,
                    mode="checks",
                ),
            )
        )
    return tuple(steps)


def _repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _ensure_aspf_handoff_session(session_id: str) -> str:
    if session_id:
        return session_id
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"session-{timestamp}-{os.getpid()}"


def _aspf_handoff_command(
    *,
    options: ChecksCommandOptions,
    step_id: str,
    command_profile: str,
    inner: tuple[str, ...],
) -> tuple[str, ...]:
    if not options.aspf_handoff_enabled:
        return inner
    return (
        str(options.python_bin),
        "scripts/misc/aspf_handoff.py",
        "run",
        "--root",
        ".",
        "--session-id",
        options.aspf_handoff_session,
        "--step-id",
        step_id,
        "--command-profile",
        command_profile,
        "--manifest",
        str(options.aspf_handoff_manifest),
        "--state-root",
        str(options.aspf_state_root),
        "--",
        *inner,
    )


def build_local_checks_steps(options: ChecksCommandOptions) -> tuple[ToolingStep, ...]:
    steps: list[ToolingStep] = []
    if options.run_dataflow:
        baseline_args: tuple[str, ...] = ()
        if (options.root / "baselines" / "dataflow_baseline.txt").exists():
            baseline_args = ("--baseline", "baselines/dataflow_baseline.txt", "--baseline-mode", "enforce")
        steps.extend(
            (
                ToolingStep(
                    label="checks: lsp parity gate",
                    command=(
                        str(options.python_bin),
                        "-m",
                        "gabion",
                        "lsp-parity-gate",
                        "--command",
                        "gabion.check",
                    ),
                    mode="dataflow",
                ),
                ToolingStep(
                    label="checks: gabion check run",
                    command=_aspf_handoff_command(
                        options=options,
                        step_id="checks.check.run",
                        command_profile="checks.check.run",
                        inner=(
                            str(options.python_bin),
                            "-m",
                            "gabion",
                            "check",
                            "run",
                            *baseline_args,
                        ),
                    ),
                    mode="dataflow",
                ),
                ToolingStep(
                    label="checks: delta advisory telemetry",
                    command=(str(options.python_bin), "-m", "gabion", "delta-advisory-telemetry"),
                    mode="dataflow",
                ),
            )
        )
        if options.run_status_watch:
            steps.append(
                ToolingStep(
                    label="checks: ci-watch",
                    command=(
                        str(options.python_bin),
                        "-m",
                        "gabion",
                        "ci-watch",
                        "--branch",
                        options.status_watch_branch,
                        "--workflow",
                        options.status_watch_workflow,
                    ),
                    mode="dataflow",
                )
            )
    if options.run_docflow:
        steps.extend(
            (
                ToolingStep(
                    label="checks: docflow",
                    command=(
                        str(options.python_bin),
                        "-m",
                        "gabion",
                        "docflow",
                        "--fail-on-violations",
                        "--sppf-gh-ref-mode",
                        options.docflow_mode,
                    ),
                    mode="docflow",
                ),
                ToolingStep(
                    label="checks: docflow packetize",
                    command=(
                        str(options.python_bin),
                        "scripts/policy/docflow_packetize.py",
                        "--root",
                        ".",
                        "--compliance",
                        "artifacts/out/docflow_compliance.json",
                        "--section-reviews",
                        "artifacts/out/docflow_section_reviews.json",
                        "--out",
                        "artifacts/out/docflow_warning_doc_packets.json",
                        "--summary-out",
                        "artifacts/out/docflow_warning_doc_packet_summary.json",
                    ),
                    mode="docflow",
                ),
                ToolingStep(
                    label="checks: docflow packet enforce",
                    command=(
                        str(options.python_bin),
                        "scripts/policy/docflow_packet_enforce.py",
                        "--root",
                        ".",
                        "--packets",
                        "artifacts/out/docflow_warning_doc_packets.json",
                        "--baseline",
                        "docs/baselines/docflow_packet_baseline.json",
                        "--out",
                        "artifacts/out/docflow_packet_enforcement.json",
                        "--debt-out",
                        "artifacts/out/docflow_packet_debt_ledger.json",
                        "--max-age-days",
                        "14",
                        "--check",
                        "--run-proving-tests",
                    ),
                    mode="docflow",
                ),
                ToolingStep(
                    label="checks: sppf_status_audit",
                    command=(str(options.python_bin), "-m", "scripts.sppf.sppf_status_audit", "--root", "."),
                    mode="docflow",
                ),
            )
        )
    if options.run_tests:
        steps.append(
            ToolingStep(
                label="checks: pytest",
                command=(
                    str(options.python_bin),
                    "-m",
                    "pytest",
                    "--junitxml",
                    "artifacts/test_runs/junit.xml",
                    "--log-file",
                    "artifacts/test_runs/pytest.log",
                    "--log-file-level=INFO",
                ),
                mode="tests",
            )
        )
    return tuple(steps)


def render_local_checks_listing(options: ChecksCommandOptions) -> tuple[str, ...]:
    lines = ["Checks to run:"]
    if options.run_dataflow:
        lines.append("- lsp parity gate (gabion lsp-parity-gate --command gabion.check)")
        lines.append("- dataflow (gabion check run)")
        if options.run_status_watch:
            lines.append(
                f"- status watch (gabion ci-watch --branch {options.status_watch_branch} --workflow {options.status_watch_workflow})"
            )
        if options.aspf_handoff_enabled:
            lines.append("- aspf handoff (state + cumulative imports)")
    if options.run_docflow:
        lines.append(
            f"- docflow (gabion docflow --fail-on-violations --sppf-gh-ref-mode {options.docflow_mode})"
        )
        lines.append(
            "- docflow packet loop (docflow_packetize + docflow_packet_enforce --check --run-proving-tests)"
        )
    if options.run_tests:
        lines.append("- tests (pytest)")
    return tuple(lines)


def _run_step(step: ToolingStep, *, root: Path) -> None:
    print()
    print(f"[checks] {step.label}")
    env = dict(os.environ)
    env.update({key: value for key, value in step.env})
    subprocess.run(step.command, check=True, cwd=root, env=env)


def run_local_checks(options: ChecksCommandOptions) -> int:
    os.chdir(options.root)
    (options.root / "artifacts" / "test_runs").mkdir(parents=True, exist_ok=True)
    if options.docflow_mode == "advisory" and options.run_docflow:
        print(
            "WARNING: running docflow in advisory GH-reference mode (local debugging only).",
            file=sys.stderr,
        )
    if options.list_only:
        for line in render_local_checks_listing(options):
            print(line, file=sys.stderr)
        return 0
    for step in build_local_checks_steps(options):
        _run_step(step, root=options.root)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docflow", dest="run_docflow", action="store_true", default=True)
    parser.add_argument("--no-docflow", dest="run_docflow", action="store_false")
    parser.add_argument("--docflow-only", dest="mode", action="store_const", const="docflow")
    parser.add_argument("--dataflow-only", dest="mode", action="store_const", const="dataflow")
    parser.add_argument("--tests-only", dest="mode", action="store_const", const="tests")
    parser.add_argument("--status-watch", dest="run_status_watch", action="store_true", default=False)
    parser.add_argument("--no-status-watch", dest="run_status_watch", action="store_false")
    parser.add_argument("--status-watch-branch", default=os.environ.get("GABION_STATUS_WATCH_BRANCH", "stage"))
    parser.add_argument("--status-watch-workflow", default=os.environ.get("GABION_STATUS_WATCH_WORKFLOW", "ci"))
    parser.add_argument("--docflow-advisory", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--no-aspf-handoff", action="store_true")
    parser.add_argument("--aspf-handoff-manifest", default=os.environ.get("GABION_ASPF_HANDOFF_MANIFEST", "artifacts/out/aspf_handoff_manifest.json"))
    parser.add_argument("--aspf-handoff-session", default=os.environ.get("GABION_ASPF_HANDOFF_SESSION", ""))
    parser.add_argument("--aspf-state-root", default=os.environ.get("GABION_ASPF_STATE_ROOT", "artifacts/out/aspf_state"))
    return parser


def _options_from_namespace(namespace: argparse.Namespace) -> ChecksCommandOptions:
    root = _repo_root()
    python_bin = Path(sys.executable)
    run_docflow = bool(namespace.run_docflow)
    run_dataflow = True
    run_tests = True
    mode = str(namespace.mode or "all")
    if mode == "docflow":
        run_docflow, run_dataflow, run_tests = True, False, False
    elif mode == "dataflow":
        run_docflow, run_dataflow, run_tests = False, True, False
    elif mode == "tests":
        run_docflow, run_dataflow, run_tests = False, False, True
    if namespace.mode == "docflow" and namespace.run_docflow is False:
        run_docflow = False
    aspf_handoff_enabled = not bool(namespace.no_aspf_handoff)
    aspf_handoff_session = (
        _ensure_aspf_handoff_session(str(namespace.aspf_handoff_session or ""))
        if aspf_handoff_enabled
        else str(namespace.aspf_handoff_session or "")
    )
    return ChecksCommandOptions(
        root=root,
        python_bin=python_bin,
        run_docflow=run_docflow,
        run_dataflow=run_dataflow,
        run_tests=run_tests,
        run_status_watch=bool(namespace.run_status_watch),
        docflow_mode="advisory" if bool(namespace.docflow_advisory) else "required",
        aspf_handoff_enabled=aspf_handoff_enabled,
        aspf_handoff_manifest=root / str(namespace.aspf_handoff_manifest),
        aspf_handoff_session=aspf_handoff_session,
        aspf_state_root=root / str(namespace.aspf_state_root),
        status_watch_branch=str(namespace.status_watch_branch),
        status_watch_workflow=str(namespace.status_watch_workflow),
        list_only=bool(namespace.list),
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    namespace = parser.parse_args(argv)
    options = _options_from_namespace(namespace)
    return run_local_checks(options)


__all__ = [
    "ChecksCommandOptions",
    "ToolingStep",
    "build_ci_checks_steps",
    "build_local_checks_steps",
    "delta_bundle_command",
    "main",
    "packet_enforce_command",
    "render_local_checks_listing",
    "run_local_checks",
]
