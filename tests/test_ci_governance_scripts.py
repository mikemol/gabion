from __future__ import annotations

import json
from pathlib import Path

from scripts import ci_controller_drift_gate as drift_gate
from scripts import governance_controller_audit
from scripts import policy_check


def test_policy_check_requires_ci_script_entrypoints() -> None:
    errors: list[str] = []
    policy_check._check_ci_script_entrypoints(
        {
            "jobs": {
                "checks": {
                    "steps": [
                        {"run": "python scripts/ci_seed_dataflow_checkpoint.py"},
                        {"run": "python scripts/ci_finalize_dataflow_outcome.py --terminal-exit 0"},
                    ]
                }
            }
        },
        Path(".github/workflows/ci.yml"),
        errors,
    )
    assert any("scripts/ci_controller_drift_gate.py" in error for error in errors)


def test_controller_drift_gate_override_expiry_behavior(tmp_path: Path) -> None:
    drift = tmp_path / "drift.json"
    out = tmp_path / "gate.json"
    drift.write_text(json.dumps({"findings": [{"severity": "high"}]}), encoding="utf-8")

    missing = tmp_path / "missing.json"
    assert drift_gate.run(drift_artifact=drift, override_record=missing, out=out) == 2
    payload_missing = json.loads(out.read_text(encoding="utf-8"))
    assert payload_missing["override_diagnostics"]["override_reason"] == "missing"

    expired = tmp_path / "expired.json"
    expired.write_text(
        json.dumps(
            {
                "actor": "ci",
                "rationale": "temporary",
                "scope": "controller_drift",
                "start": "2024-01-01T00:00:00Z",
                "expiry": "2024-01-02T00:00:00Z",
                "rollback_condition": "fix merged",
                "evidence_links": ["artifact://x"],
            }
        ),
        encoding="utf-8",
    )
    assert drift_gate.run(drift_artifact=drift, override_record=expired, out=out) == 2

    valid = tmp_path / "valid.json"
    valid.write_text(
        json.dumps(
            {
                "actor": "ci",
                "rationale": "temporary",
                "scope": "controller_drift",
                "start": "2024-01-01T00:00:00Z",
                "expiry": "2999-01-02T00:00:00Z",
                "rollback_condition": "fix merged",
                "evidence_links": ["artifact://x"],
            }
        ),
        encoding="utf-8",
    )
    assert drift_gate.run(drift_artifact=drift, override_record=valid, out=out) == 0
    payload_valid = json.loads(out.read_text(encoding="utf-8"))
    assert payload_valid["override_diagnostics"]["override_valid"] is True
    assert payload_valid["override_diagnostics"]["override_source"] == "controller_drift_gate"


def test_controller_audit_requires_clause_anchors_for_enforcement_surfaces() -> None:
    assert governance_controller_audit._enforcement_clause_findings() == []
