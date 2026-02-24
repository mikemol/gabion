#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from gabion.tooling.governance_rules import load_governance_rules

SEVERITY = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def run(*, drift_artifact: Path, override_record: Path, out: Path) -> int:
    rules = load_governance_rules().controller_drift
    drift = _load_json(drift_artifact)
    findings = drift.get("findings", [])
    max_rank = 0
    for finding in findings if isinstance(findings, list) else []:
        sev = str((finding or {}).get("severity", "")).lower()
        max_rank = max(max_rank, SEVERITY.get(sev, 0))
    enforce_rank = SEVERITY.get(rules.enforce_at_or_above.lower(), 99)
    override_used = override_record.exists()
    override_valid = False
    override_payload: dict[str, object] | None = None
    if override_used:
        override_payload = _load_json(override_record)
        required = {"actor", "rationale", "scope", "start", "expiry", "rollback_condition", "evidence_links"}
        override_valid = required.issubset(set(override_payload.keys()))
        if override_valid:
            override_valid = _parse_time(str(override_payload["expiry"])) > datetime.now(timezone.utc)

    status = "pass"
    if max_rank >= enforce_rank:
        status = "override" if override_valid else "fail"

    payload = {
        "status": status,
        "max_severity_rank": max_rank,
        "enforce_at_or_above": rules.enforce_at_or_above,
        "required_remediation": rules.remediation_by_severity,
        "override_record": override_payload,
        "consecutive_passes_required": rules.consecutive_passes_required,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 2 if status == "fail" else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drift-artifact", type=Path, required=True)
    parser.add_argument("--override-record", type=Path, default=Path("artifacts/out/governance_override_record.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/out/controller_drift_gate.json"))
    args = parser.parse_args()
    return run(drift_artifact=args.drift_artifact, override_record=args.override_record, out=args.out)


if __name__ == "__main__":
    raise SystemExit(main())
