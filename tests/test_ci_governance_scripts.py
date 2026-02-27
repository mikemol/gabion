from __future__ import annotations

import json
from pathlib import Path

from scripts import ci_controller_drift_gate as drift_gate
from scripts import governance_controller_audit
from scripts import policy_check


# gabion:evidence E:function_site::test_ci_governance_scripts.py::tests.test_ci_governance_scripts.test_policy_check_requires_ci_script_entrypoints
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


# gabion:evidence E:function_site::test_ci_governance_scripts.py::tests.test_ci_governance_scripts.test_controller_drift_gate_override_expiry_behavior
def test_controller_drift_gate_override_expiry_behavior(tmp_path: Path) -> None:
    drift = tmp_path / "drift.json"
    out = tmp_path / "gate.json"
    history = tmp_path / "history.json"
    drift.write_text(json.dumps({"findings": [{"severity": "high"}]}), encoding="utf-8")

    missing = tmp_path / "missing.json"
    assert drift_gate.run(drift_artifact=drift, override_record=missing, out=out, history=history) == 2
    payload_missing = json.loads(out.read_text(encoding="utf-8"))
    assert payload_missing["override_diagnostics"]["override_reason"] == "missing"
    assert payload_missing["clean_streak_length"] == 0

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
    assert drift_gate.run(drift_artifact=drift, override_record=expired, out=out, history=history) == 2

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
    assert drift_gate.run(drift_artifact=drift, override_record=valid, out=out, history=history) == 0
    payload_valid = json.loads(out.read_text(encoding="utf-8"))
    assert payload_valid["override_diagnostics"]["override_valid"] is True
    assert payload_valid["override_diagnostics"]["override_source"] == "controller_drift_gate"


# gabion:evidence E:function_site::test_ci_governance_scripts.py::tests.test_ci_governance_scripts.test_controller_drift_gate_streak_progression_and_reset
def test_controller_drift_gate_streak_progression_and_reset(tmp_path: Path) -> None:
    clean_drift = tmp_path / "clean_drift.json"
    failing_drift = tmp_path / "failing_drift.json"
    override = tmp_path / "override.json"
    history = tmp_path / "history.json"
    out = tmp_path / "gate.json"

    clean_drift.write_text(json.dumps({"findings": [{"severity": "low"}]}), encoding="utf-8")
    failing_drift.write_text(json.dumps({"findings": [{"severity": "high"}]}), encoding="utf-8")
    override.write_text("{}", encoding="utf-8")

    required = drift_gate.load_governance_rules().controller_drift.consecutive_passes_required

    assert drift_gate.run(drift_artifact=clean_drift, override_record=override, out=out, history=history) == 0
    first = json.loads(out.read_text(encoding="utf-8"))
    assert first["status"] == "clean"
    assert first["clean_streak_length"] == 1
    assert first["required_clean_streak_length"] == required
    assert first["stabilization_achieved"] is (required <= 1)

    for expected_streak in range(2, required):
        assert drift_gate.run(drift_artifact=clean_drift, override_record=override, out=out, history=history) == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["status"] == "clean"
        assert payload["clean_streak_length"] == expected_streak
        assert payload["stabilization_achieved"] is False

    assert drift_gate.run(drift_artifact=clean_drift, override_record=override, out=out, history=history) == 0
    stabilized = json.loads(out.read_text(encoding="utf-8"))
    assert stabilized["status"] == "stabilized"
    assert stabilized["clean_streak_length"] == required
    assert stabilized["stabilization_achieved"] is True

    assert drift_gate.run(drift_artifact=failing_drift, override_record=override, out=out, history=history) == 2
    reset = json.loads(out.read_text(encoding="utf-8"))
    assert reset["status"] == "fail"
    assert reset["clean_streak_length"] == 0
    assert reset["stabilization_achieved"] is False


# gabion:evidence E:function_site::test_ci_governance_scripts.py::tests.test_ci_governance_scripts.test_controller_audit_requires_clause_anchors_for_enforcement_surfaces
def test_controller_audit_requires_clause_anchors_for_enforcement_surfaces() -> None:
    assert governance_controller_audit._enforcement_clause_findings() == []


# gabion:evidence E:function_site::test_ci_governance_scripts.py::tests.test_ci_governance_scripts.test_policy_check_normative_enforcement_map_validates_current_repo
def test_policy_check_normative_enforcement_map_validates_current_repo() -> None:
    policy_check.check_normative_enforcement_map()


# gabion:evidence E:function_site::test_ci_governance_scripts.py::tests.test_ci_governance_scripts.test_policy_check_required_clause_set_matches_index
def test_policy_check_required_clause_set_matches_index() -> None:
    index_path = Path("docs/normative_clause_index.md")
    canonical = {
        line.split("`")[1]
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("### `NCI-")
    }
    assert policy_check._REQUIRED_NORMATIVE_CLAUSES == canonical


# gabion:evidence E:function_site::test_ci_governance_scripts.py::tests.test_ci_governance_scripts.test_policy_check_normative_enforcement_map_fails_missing_module
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
  NCI-BASELINE-RATCHET:
    status: document-only
    enforcing_modules: []
    ci_anchors: []
    expected_artifacts: []
  NCI-DEADLINE-TIMEOUT-PROPAGATION:
    status: document-only
    enforcing_modules: []
    ci_anchors: []
    expected_artifacts: []
  NCI-CONTROLLER-ADAPTATION-LAW:
    status: document-only
    enforcing_modules: []
    ci_anchors: []
    expected_artifacts: []
  NCI-OVERRIDE-LIFECYCLE:
    status: document-only
    enforcing_modules: []
    ci_anchors: []
    expected_artifacts: []
  NCI-CONTROLLER-DRIFT-LIFECYCLE:
    status: document-only
    enforcing_modules: []
    ci_anchors: []
    expected_artifacts: []
  NCI-COMMAND-MATURITY-PARITY:
    status: document-only
    enforcing_modules: []
    ci_anchors: []
    expected_artifacts: []
  NCI-DUAL-SENSOR-CORRECTION-LOOP:
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


# gabion:evidence E:function_site::test_ci_governance_scripts.py::tests.test_ci_governance_scripts.test_policy_check_normative_enforcement_map_fails_missing_clause
def test_policy_check_normative_enforcement_map_fails_missing_clause(tmp_path: Path) -> None:
    broken = tmp_path / "normative_enforcement_map.yaml"
    broken.write_text(
        """version: 1
clauses:
  NCI-LSP-FIRST:
    status: document-only
    enforcing_modules: []
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
  NCI-BASELINE-RATCHET:
    status: document-only
    enforcing_modules: []
    ci_anchors: []
    expected_artifacts: []
  NCI-DEADLINE-TIMEOUT-PROPAGATION:
    status: document-only
    enforcing_modules: []
    ci_anchors: []
    expected_artifacts: []
  NCI-CONTROLLER-ADAPTATION-LAW:
    status: document-only
    enforcing_modules: []
    ci_anchors: []
    expected_artifacts: []
  NCI-OVERRIDE-LIFECYCLE:
    status: document-only
    enforcing_modules: []
    ci_anchors: []
    expected_artifacts: []
  NCI-CONTROLLER-DRIFT-LIFECYCLE:
    status: document-only
    enforcing_modules: []
    ci_anchors: []
    expected_artifacts: []
  NCI-DUAL-SENSOR-CORRECTION-LOOP:
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


# gabion:evidence E:function_site::test_ci_governance_scripts.py::tests.test_ci_governance_scripts.test_controller_audit_detects_contradictory_anchors_across_declared_normative_docs
def test_controller_audit_detects_contradictory_anchors_across_declared_normative_docs(
    tmp_path: Path,
) -> None:
    (tmp_path / "docs").mkdir()
    policy = tmp_path / "POLICY_SEED.md"
    policy.write_text(
        "\n".join(
            (
                "- `controller-normative-doc: POLICY_SEED.md`",
                "- `controller-normative-doc: docs/governance_control_loops.md`",
                "- `controller-anchor: CD-999 | doc: POLICY_SEED.md#change_protocol | sensor: contradictory_anchors_across_normative_docs | check: scripts/governance_controller_audit.py | severity: high`",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "governance_control_loops.md").write_text(
        "- `controller-anchor: CD-999 | doc: POLICY_SEED.md#change_protocol | sensor: contradictory_anchors_across_normative_docs | check: scripts/governance_controller_audit.py | severity: medium`\n",
        encoding="utf-8",
    )

    original_root = governance_controller_audit.REPO_ROOT
    try:
        governance_controller_audit.REPO_ROOT = tmp_path
        signatures, missing = governance_controller_audit._collect_normative_anchor_signatures(
            governance_controller_audit._normative_docs(policy.read_text(encoding="utf-8"))
        )
    finally:
        governance_controller_audit.REPO_ROOT = original_root

    assert missing == []
    assert len(signatures["CD-999"]) == 2


# gabion:evidence E:function_site::test_ci_governance_scripts.py::tests.test_ci_governance_scripts.test_controller_audit_reports_missing_declared_normative_docs
def test_controller_audit_reports_missing_declared_normative_docs(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    policy = tmp_path / "POLICY_SEED.md"
    policy.write_text(
        "\n".join(
            (
                "- `controller-normative-doc: POLICY_SEED.md`",
                "- `controller-normative-doc: docs/governance_control_loops.md`",
                "- `controller-anchor: CD-001 | doc: POLICY_SEED.md#change_protocol | sensor: policy_clauses_without_enforcing_check | check: scripts/governance_controller_audit.py | severity: high`",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    original_root = governance_controller_audit.REPO_ROOT
    try:
        governance_controller_audit.REPO_ROOT = tmp_path
        out = tmp_path / "out.json"
        rc = governance_controller_audit.run(policy_path=policy, out_path=out, fail_on_severity="high")
        payload = json.loads(out.read_text(encoding="utf-8"))
    finally:
        governance_controller_audit.REPO_ROOT = original_root

    assert rc == 2
    assert any(item["sensor"] == "missing_normative_docs_in_repo" for item in payload["findings"])


# gabion:evidence E:function_site::test_ci_governance_scripts.py::tests.test_ci_governance_scripts.test_policy_check_payload_branching_allows_boundary_decode

def test_policy_check_payload_branching_allows_boundary_decode(tmp_path: Path) -> None:
    module = tmp_path / "ok.py"
    module.write_text(
        """
from collections.abc import Mapping

def _decode_payload(value: object) -> object:
    match value:
        case Mapping() as payload:
            return payload
        case list() as payload:
            return payload
        case _:
            return None
""".strip()
        + "\n",
        encoding="utf-8",
    )
    assert policy_check._raw_payload_branching_violations(module) == []


# gabion:evidence E:function_site::test_ci_governance_scripts.py::tests.test_ci_governance_scripts.test_policy_check_payload_branching_flags_non_decode_functions

def test_policy_check_payload_branching_flags_non_decode_functions(tmp_path: Path) -> None:
    module = tmp_path / "bad.py"
    module.write_text(
        """
from collections.abc import Mapping

def semantic(value: object) -> object:
    match value:
        case Mapping() as payload:
            return payload
        case _:
            return None
""".strip()
        + "\n",
        encoding="utf-8",
    )
    violations = policy_check._raw_payload_branching_violations(module)
    assert violations
