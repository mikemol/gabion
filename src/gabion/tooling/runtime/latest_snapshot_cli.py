"""CLI entry point for `gabion repo latest-snapshot`."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def _snapshot_paths(root: Path) -> dict[str, Path]:
    marker = root / "artifacts" / "audit_snapshots" / "LATEST.txt"
    if not marker.is_file():
        raise SystemExit(f"Latest snapshot marker not found: {marker}")
    stamp = marker.read_text(encoding="utf-8").strip()
    if not stamp:
        raise SystemExit(f"Latest snapshot marker is empty: {marker}")
    snapshot_dir = root / "artifacts" / "audit_snapshots" / stamp
    if not snapshot_dir.is_dir():
        raise SystemExit(f"Snapshot directory not found: {snapshot_dir}")
    return {
        "dir": snapshot_dir,
        "report": snapshot_dir / "dataflow_report.md",
        "dot": snapshot_dir / "dataflow_graph.dot",
        "plan": snapshot_dir / "synthesis_plan.json",
        "protocols": snapshot_dir / "protocol_stubs.py",
        "refactor": snapshot_dir / "refactor_plan.json",
        "fingerprint_synth": snapshot_dir / "fingerprint_synth.json",
        "fingerprint_provenance": snapshot_dir / "fingerprint_provenance.json",
        "fingerprint_deadness": snapshot_dir / "fingerprint_deadness.json",
        "fingerprint_coherence": snapshot_dir / "fingerprint_coherence.json",
        "fingerprint_rewrite_plans": snapshot_dir / "fingerprint_rewrite_plans.json",
        "fingerprint_exception_obligations": snapshot_dir / "fingerprint_exception_obligations.json",
        "fingerprint_handledness": snapshot_dir / "fingerprint_handledness.json",
        "decision": snapshot_dir / "decision_snapshot.json",
        "decision_tier_candidates": snapshot_dir / "decision_tier_candidates.toml",
        "consolidation_report": snapshot_dir / "consolidation_report.md",
        "consolidation_suggestions": snapshot_dir / "consolidation_suggestions.json",
        "docflow": snapshot_dir / "docflow_audit.txt",
        "lint": snapshot_dir / "lint.txt",
        "lint_jsonl": snapshot_dir / "lint.jsonl",
        "lint_sarif": snapshot_dir / "lint.sarif",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the latest audit snapshot paths.")
    parser.add_argument("--root", default=".", help="Repo root (default: .).")
    parser.add_argument(
        "selector",
        nargs="?",
        choices=(
            "report",
            "dot",
            "plan",
            "protocols",
            "refactor",
            "fingerprint-synth",
            "fingerprint-provenance",
            "fingerprint-deadness",
            "fingerprint-coherence",
            "fingerprint-rewrite-plans",
            "fingerprint-exception-obligations",
            "fingerprint-handledness",
            "decision",
            "decision-tier-candidates",
            "consolidation-report",
            "consolidation-suggestions",
            "docflow",
            "lint",
            "lint-jsonl",
            "lint-sarif",
            "dir",
        ),
    )
    args = parser.parse_args(list(argv or []))
    paths = _snapshot_paths(Path(args.root).resolve())
    if args.selector is not None:
        key = args.selector.replace("-", "_")
        print(paths[key])
        return 0
    print("Latest snapshot:")
    for key in (
        "report",
        "dot",
        "plan",
        "protocols",
        "refactor",
        "fingerprint_synth",
        "fingerprint_provenance",
        "fingerprint_deadness",
        "fingerprint_coherence",
        "fingerprint_rewrite_plans",
        "fingerprint_exception_obligations",
        "fingerprint_handledness",
        "decision",
        "decision_tier_candidates",
        "consolidation_report",
        "consolidation_suggestions",
        "docflow",
        "lint",
        "lint_jsonl",
        "lint_sarif",
    ):
        print(f"- {paths[key]}")
    return 0
