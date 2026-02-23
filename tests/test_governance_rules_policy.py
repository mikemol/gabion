from __future__ import annotations

from gabion.tooling import governance_rules


# gabion:evidence E:function_site::tests/test_governance_rules_policy.py::test_governance_rules_define_required_gates
def test_governance_rules_define_required_gates() -> None:
    rules = governance_rules.load_governance_rules()
    assert {"obsolescence_opaque", "obsolescence_unmapped", "annotation_orphaned", "ambiguity", "docflow"}.issubset(rules.gates)


# gabion:evidence E:function_site::tests/test_governance_rules_policy.py::test_governance_severity_thresholds_are_monotonic
def test_governance_severity_thresholds_are_monotonic() -> None:
    rules = governance_rules.load_governance_rules()
    for policy in rules.gates.values():
        assert policy.severity.warning_threshold <= policy.severity.blocking_threshold
        assert policy.correction.transitions
        assert "baseline_write_requires_explicit_flag" in policy.correction.bounded_steps
