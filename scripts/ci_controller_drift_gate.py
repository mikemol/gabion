#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gabion.tooling.governance_rules import load_governance_rules
from gabion.tooling.override_record import validate_override_record_file

SEVERITY = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_history_streak(history: Path) -> int:
    if not history.exists():
        return 0
    try:
        history_payload = _load_json(history)
    except (json.JSONDecodeError, OSError):
        return 0
    streak_raw = history_payload.get("clean_streak_length", 0)
    try:
        return max(0, int(streak_raw))
    except (TypeError, ValueError):
        return 0


def run(*, drift_artifact: Path, override_record: Path, out: Path, history: Path) -> int:
    rules = load_governance_rules().controller_drift
    drift = _load_json(drift_artifact)
    findings = drift.get("findings", [])
    max_rank = 0
    for finding in findings if isinstance(findings, list) else []:
        sev = str((finding or {}).get("severity", "")).lower()
        max_rank = max(max_rank, SEVERITY.get(sev, 0))
    enforce_rank = SEVERITY.get(rules.enforce_at_or_above.lower(), 99)
    override_validation = validate_override_record_file(override_record)

    prior_streak = _read_history_streak(history)

    status = "clean"
    if max_rank >= enforce_rank:
        status = "override" if override_validation.valid else "fail"

    clean_streak_length = prior_streak + 1 if status == "clean" else 0
    stabilization_achieved = clean_streak_length >= rules.consecutive_passes_required
    if status == "clean" and stabilization_achieved:
        status = "stabilized"

    payload = {
        "status": status,
        "max_severity_rank": max_rank,
        "enforce_at_or_above": rules.enforce_at_or_above,
        "required_remediation": rules.remediation_by_severity,
        "override_record": override_validation.record,
        "override_diagnostics": override_validation.telemetry(source="controller_drift_gate"),
        "clean_streak_length": clean_streak_length,
        "required_clean_streak_length": rules.consecutive_passes_required,
        "stabilization_achieved": stabilization_achieved,
    }
    history.parent.mkdir(parents=True, exist_ok=True)
    history.write_text(
        json.dumps({"clean_streak_length": clean_streak_length}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 2 if status == "fail" else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drift-artifact", type=Path, required=True)
    parser.add_argument("--override-record", type=Path, default=Path("artifacts/out/governance_override_record.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/out/controller_drift_gate.json"))
    parser.add_argument("--history", type=Path, default=Path("artifacts/out/controller_drift_gate_history.json"))
    args = parser.parse_args()
    return run(drift_artifact=args.drift_artifact, override_record=args.override_record, out=args.out, history=args.history)


if __name__ == "__main__":
    raise SystemExit(main())
