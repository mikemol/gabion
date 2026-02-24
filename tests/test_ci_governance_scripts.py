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


def test_policy_check_normative_enforcement_map_validates_current_repo() -> None:
    policy_check.check_normative_enforcement_map()


def test_policy_check_normative_enforcement_map_fails_missing_module(tmp_path: Path) -> None:
    broken = tmp_path / "normative_enforcement_map.yaml"
    broken.write_text(
        """version: 1
clauses:
  NCI-LSP-FIRST:
    status: enforced
    enforcing_modules: [missing/file.py]
    ci_anchors: []
    expected_artifacts: []
  NCI-ACTIONS-PINNED:
    status: document-only
    enforcing_modules: []
    ci_anchors: []
    expected_artifacts: []
  NCI-ACTIONS-ALLOWLIST:
    status: document-only
    enforcing_modules: []
    ci_anchors: []
    expected_artifacts: []
  NCI-DATAFLOW-BUNDLE-TIERS:
    status: document-only
    enforcing_modules: []
    ci_anchors: []
    expected_artifacts: []
  NCI-SHIFT-AMBIGUITY-LEFT:
    status: document-only
    enforcing_modules: []
    ci_anchors: []
    expected_artifacts: []
  NCI-COMMAND-MATURITY-PARITY:
    status: document-only
    enforcing_modules: []
    ci_anchors: []
    expected_artifacts: []
  NCI-CONTROLLER-DRIFT-LIFECYCLE:
    status: document-only
    enforcing_modules: []
    ci_anchors: []
    expected_artifacts: []
""",
        encoding="utf-8",
    )
    original = policy_check.NORMATIVE_ENFORCEMENT_MAP
    try:
        policy_check.NORMATIVE_ENFORCEMENT_MAP = broken
        try:
            policy_check.check_normative_enforcement_map()
            assert False, "expected check_normative_enforcement_map to fail"
        except SystemExit as exc:
            assert exc.code == 2
    finally:
        policy_check.NORMATIVE_ENFORCEMENT_MAP = original
