"""CLI entry point for `gabion repo audit-snapshot`."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence


def _run(cmd: list[str], *, stdout_path: Path | None = None) -> int:
    if stdout_path is None:
        completed = subprocess.run(cmd, check=False)
        return int(completed.returncode)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(cmd, check=False, stdout=handle)
    return int(completed.returncode)


def _ensure_handoff_session(session_id: str) -> str:
    if session_id:
        return session_id
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"session-{stamp}-{os.getpid()}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture an audit snapshot bundle.")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--no-aspf-handoff", action="store_true")
    parser.add_argument(
        "--aspf-handoff-manifest",
        default=os.environ.get(
            "GABION_ASPF_HANDOFF_MANIFEST",
            "artifacts/out/aspf_handoff_manifest.json",
        ),
    )
    parser.add_argument(
        "--aspf-handoff-session",
        default=os.environ.get("GABION_ASPF_HANDOFF_SESSION", ""),
    )
    parser.add_argument(
        "--aspf-state-root",
        default=os.environ.get("GABION_ASPF_STATE_ROOT", "artifacts/out/aspf_state"),
    )
    args = parser.parse_args(list(argv or []))

    root = Path(args.root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifacts_dir = Path("artifacts/audit_snapshots") / timestamp
    latest_marker = Path("artifacts/audit_snapshots/LATEST.txt")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataflow_report = artifacts_dir / "dataflow_report.md"
    dataflow_dot = artifacts_dir / "dataflow_graph.dot"
    dataflow_plan = artifacts_dir / "synthesis_plan.json"
    dataflow_protocols = artifacts_dir / "protocol_stubs.py"
    dataflow_refactor = artifacts_dir / "refactor_plan.json"
    fingerprint_synth = artifacts_dir / "fingerprint_synth.json"
    fingerprint_provenance = artifacts_dir / "fingerprint_provenance.json"
    fingerprint_deadness = artifacts_dir / "fingerprint_deadness.json"
    fingerprint_coherence = artifacts_dir / "fingerprint_coherence.json"
    fingerprint_rewrite_plans = artifacts_dir / "fingerprint_rewrite_plans.json"
    fingerprint_exception_obligations = artifacts_dir / "fingerprint_exception_obligations.json"
    fingerprint_handledness = artifacts_dir / "fingerprint_handledness.json"
    decision_snapshot = artifacts_dir / "decision_snapshot.json"
    decision_tier_candidates = artifacts_dir / "decision_tier_candidates.toml"
    consolidation_report = artifacts_dir / "consolidation_report.md"
    consolidation_suggestions = artifacts_dir / "consolidation_suggestions.json"
    docflow_report = artifacts_dir / "docflow_audit.txt"
    lint_report = artifacts_dir / "lint.txt"
    lint_jsonl = artifacts_dir / "lint.jsonl"
    lint_sarif = artifacts_dir / "lint.sarif"
    handoff_result = artifacts_dir / "aspf_handoff_run_check_raw.json"

    python_cmd = [sys.executable, "-m", "gabion"]
    check_raw_command = [
        *python_cmd,
        "check",
        "raw",
        "--",
        str(root),
        "--root",
        str(root),
        "--report",
        str(dataflow_report),
        "--dot",
        str(dataflow_dot),
        "--type-audit-report",
        "--synthesis-plan",
        str(dataflow_plan),
        "--synthesis-protocols",
        str(dataflow_protocols),
        "--synthesis-report",
        "--refactor-plan",
        "--refactor-plan-json",
        str(dataflow_refactor),
        "--fingerprint-synth-json",
        str(fingerprint_synth),
        "--fingerprint-provenance-json",
        str(fingerprint_provenance),
        "--fingerprint-deadness-json",
        str(fingerprint_deadness),
        "--fingerprint-coherence-json",
        str(fingerprint_coherence),
        "--fingerprint-rewrite-plans-json",
        str(fingerprint_rewrite_plans),
        "--fingerprint-exception-obligations-json",
        str(fingerprint_exception_obligations),
        "--fingerprint-handledness-json",
        str(fingerprint_handledness),
        "--emit-decision-snapshot",
        str(decision_snapshot),
        "--lint",
        "--lint-jsonl",
        str(lint_jsonl),
        "--lint-sarif",
        str(lint_sarif),
    ]
    if args.no_aspf_handoff:
        check_raw_rc = _run(check_raw_command, stdout_path=lint_report)
    else:
        session_id = _ensure_handoff_session(args.aspf_handoff_session)
        command_text = f"{shlex.join(check_raw_command)} > {shlex.quote(str(lint_report))}"
        handoff_command = [
            *python_cmd,
            "aspf",
            "handoff",
            "run",
            "--root",
            ".",
            "--session-id",
            session_id,
            "--step-id",
            "audit-snapshot.check.raw",
            "--command-profile",
            "audit-snapshot.check.raw",
            "--manifest",
            args.aspf_handoff_manifest,
            "--state-root",
            args.aspf_state_root,
            "--",
            "bash",
            "-lc",
            command_text,
        ]
        check_raw_rc = _run(handoff_command, stdout_path=handoff_result)
    if check_raw_rc != 0:
        return check_raw_rc

    if _run([*python_cmd, "docflow", "--root", str(root)], stdout_path=docflow_report) != 0:
        return 1
    if _run(
        [*python_cmd, "decision-tiers", "--root", str(root), "--lint", str(lint_report)],
        stdout_path=decision_tier_candidates,
    ) != 0:
        return 1
    if _run(
        [
            *python_cmd,
            "consolidation",
            "--root",
            str(root),
            "--decision",
            str(decision_snapshot),
            "--lint",
            str(lint_report),
            "--output",
            str(consolidation_report),
            "--json-output",
            str(consolidation_suggestions),
        ]
    ) != 0:
        return 1

    latest_marker.parent.mkdir(parents=True, exist_ok=True)
    latest_marker.write_text(f"{timestamp}\n", encoding="utf-8")
    for path in (
        dataflow_report,
        dataflow_dot,
        dataflow_plan,
        dataflow_protocols,
        dataflow_refactor,
        fingerprint_synth,
        fingerprint_provenance,
        fingerprint_deadness,
        fingerprint_coherence,
        fingerprint_rewrite_plans,
        fingerprint_exception_obligations,
        fingerprint_handledness,
        decision_snapshot,
        decision_tier_candidates,
        consolidation_report,
        consolidation_suggestions,
        docflow_report,
        lint_report,
        lint_jsonl,
        lint_sarif,
        latest_marker,
    ):
        print(f"- {path}")
    return 0
