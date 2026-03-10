from __future__ import annotations

from pathlib import Path

from gabion.analysis import aspf_rule_engine
from gabion.policy_dsl import PolicyDomain, evaluate_policy
from gabion.policy_dsl.registry import build_registry
from gabion.tooling import ambiguity_contract_policy_check as ambiguity_policy
from gabion.tooling import delta_gate
from gabion.tooling.policy_scanner_suite import PolicySuiteResult


def test_registry_rule_ids_are_unique_and_stable() -> None:
    first = [rule.rule_id for rule in build_registry().program.rules]
    second = [rule.rule_id for rule in build_registry().program.rules]
    assert first == second
    assert len(first) == len(set(first))


def test_governance_gate_dsl_matches_blocking_threshold() -> None:
    decision = evaluate_policy(
        domain=PolicyDomain.GOVERNANCE_GATE,
        data={
            "gate_id": "obsolescence_opaque",
            "summary": {"opaque_evidence": {"delta": 1}},
        },
    )
    assert decision.rule_id == "gate.obsolescence_opaque.blocking"


def test_docflow_baseline_missing_uses_skip_rule() -> None:
    decision = evaluate_policy(
        domain=PolicyDomain.GOVERNANCE_GATE,
        data={"gate_id": "docflow", "baseline_missing": True},
    )
    assert decision.rule_id == "gate.docflow.baseline_missing"


def test_delta_gate_value_helpers_remain_compatible() -> None:
    payload = {"summary": {"opaque_evidence": {"delta": "2"}}}
    assert delta_gate.obsolescence_opaque_delta_value(payload) == 2


def test_scanner_payload_uses_dsl_decision_shape() -> None:
    result = PolicySuiteResult(
        root=Path('.').resolve(),
        inventory_hash="h1",
        rule_set_hash="r1",
        violations_by_rule={"branchless": [{}], "defensive_fallback": [], "no_monkeypatch": []},
        cached=False,
    )
    payload = result.to_payload()
    decision = payload.get("decision")
    assert isinstance(decision, dict)
    assert decision.get("rule_id") == "scanner.branchless.blocking"
    assert decision.get("outcome") == "block"


def test_ambiguity_contract_run_uses_dsl_blocking_rule(monkeypatch: object, tmp_path: Path) -> None:
    violation = ambiguity_policy.Violation(
        rule_id="ACP-003",
        path="src/gabion/example.py",
        line=1,
        column=1,
        qualname="f",
        message="x",
    )
    monkeypatch.setattr(ambiguity_policy, "collect_violations", lambda root: [violation])
    baseline = tmp_path / "baseline.json"
    exit_code = ambiguity_policy.run(root=tmp_path, baseline=baseline, baseline_write=False)
    assert exit_code == 1


def test_aspf_rule_engine_classifies_two_cell() -> None:
    decision = aspf_rule_engine.classify_aspf_opportunity(
        {"witness": {"two_cell": True, "cofibration": False}}
    )
    assert decision.rule_id == "aspf.opportunity.two_cell"


def test_registry_python_source_does_not_inline_scanner_rule_ids() -> None:
    source = Path("src/gabion/policy_dsl/registry.py").read_text(encoding="utf-8")
    assert "scanner.branchless.blocking" not in source
    assert "ambiguity.new_violations" not in source


def test_ambiguity_ast_event_rules_are_dsl_backed() -> None:
    decision = evaluate_policy(
        domain=PolicyDomain.AMBIGUITY_CONTRACT_AST,
        data={"event": "runtime_isinstance_call"},
    )
    assert decision.rule_id == "ACP-003"


def test_governance_adapter_emits_baseline_missing_rule() -> None:
    source = Path("src/gabion/tooling/governance_rules.py").read_text(encoding="utf-8")
    assert ".baseline_missing" in source


def test_delta_gate_no_python_docflow_baseline_branch() -> None:
    source = Path("src/gabion/tooling/delta_gate.py").read_text(encoding="utf-8")
    assert "baseline_missing = bool(payload.get" not in source
