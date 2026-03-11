from __future__ import annotations

from pathlib import Path

from gabion.analysis import aspf_rule_engine
from gabion.policy_dsl import PolicyDomain, evaluate_policy
from gabion.policy_dsl.registry import build_registry
from gabion.tooling.delta import delta_gate
from gabion.tooling.governance import ambiguity_contract_policy_check as ambiguity_policy
from gabion.tooling.runtime.policy_scanner_suite import PolicySuiteResult


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
        root=Path(".").resolve(),
        inventory_hash="h1",
        rule_set_hash="r1",
        violations_by_rule={"branchless": [{}], "defensive_fallback": [], "no_monkeypatch": []},
        policy_results={},
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
    monkeypatch.setattr(ambiguity_policy, "collect_violations", lambda batch: [violation])
    baseline = tmp_path / "baseline.json"
    exit_code = ambiguity_policy.run(root=tmp_path, baseline=baseline, baseline_write=False)
    assert exit_code == 1


def test_aspf_rule_engine_classifies_two_cell() -> None:
    decision = aspf_rule_engine.classify_aspf_opportunity(
        {"witness": {"two_cell": True, "cofibration": False}}
    )
    assert decision.rule_id == "aspf.opportunity.two_cell"


def test_aspf_rule_engine_classifies_fungible_observation_kind() -> None:
    decision = aspf_rule_engine.classify_aspf_opportunity(
        {
            "kind": "fungible_execution_path_substitution",
            "witness_requirement": "two_cell_witness",
            "actionability": "actionable",
        }
    )
    assert decision.rule_id == "aspf.opportunity.fungible_execution_path_substitution"


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
    fallthrough_decision = evaluate_policy(
        domain=PolicyDomain.AMBIGUITY_CONTRACT_AST,
        data={"event": "match_fallthrough_without_never"},
    )
    assert fallthrough_decision.rule_id == "ACP-005"
    probe_recovery_decision = evaluate_policy(
        domain=PolicyDomain.AMBIGUITY_CONTRACT_AST,
        data={"event": "probe_state_recovery"},
    )
    assert probe_recovery_decision.rule_id == "ACP-006"
    assert probe_recovery_decision.details == {
        "guidance": {
            "why": (
                "stores unresolved shape or type ambiguity in local carrier state "
                "and resolves it later, recreating downstream control ambiguity"
            ),
            "prefer": (
                "move dispatch to the boundary, or use a reducer over "
                "already-accepted variants, or return an explicit tagged result"
            ),
            "avoid": [
                (
                    "do not add matched_* locals, placeholder strings or ints, "
                    "or post-probe if-not-matched recovery branches"
                ),
                "do not silence the site by inserting never() downstream of the structural probe",
            ],
        }
    }
    nullable_contract_decision = evaluate_policy(
        domain=PolicyDomain.AMBIGUITY_CONTRACT_AST,
        data={"event": "nullable_contract_control"},
    )
    assert nullable_contract_decision.rule_id == "ACP-007"
    assert nullable_contract_decision.details == {
        "guidance": {
            "why": (
                "core logic is classifying a nullable carrier imperatively "
                "instead of receiving a strict contract from ingress"
            ),
            "prefer": (
                "normalize at ingress into a non-null DTO, tagged result, or "
                "explicit decision protocol so downstream code is only called on "
                "strict inputs"
            ),
            "avoid": [
                "do not replace None with a custom sentinel or alternate magic value",
                "do not add more is None or is not None branches deeper in deterministic core",
            ],
        }
    }

# gabion:evidence E:call_footprint::tests/test_policy_dsl.py::test_grade_monotonicity_summary_rules_are_dsl_backed::policy_rules.yaml::grade_monotonicity.new_violations::policy_rules.yaml::grade_monotonicity.ok
# gabion:behavior primary=desired
def test_grade_monotonicity_summary_rules_are_dsl_backed() -> None:
    blocking = evaluate_policy(
        domain=PolicyDomain.GRADE_MONOTONICITY,
        data={"new_violations": 1},
    )
    assert blocking.rule_id == "grade_monotonicity.new_violations"
    clean = evaluate_policy(
        domain=PolicyDomain.GRADE_MONOTONICITY,
        data={"new_violations": 0},
    )
    assert clean.rule_id == "grade_monotonicity.ok"


def test_governance_adapter_emits_baseline_missing_rule() -> None:
    source = Path("src/gabion/tooling/governance/governance_rules.py").read_text(
        encoding="utf-8"
    )
    assert ".baseline_missing" in source


def test_delta_gate_no_python_docflow_baseline_branch() -> None:
    source = Path("src/gabion/tooling/delta/delta_gate.py").read_text(encoding="utf-8")
    assert "baseline_missing = bool(payload.get" not in source


def test_migrated_modules_use_dsl_evaluator() -> None:
    ambiguity_source = Path(
        "src/gabion/tooling/governance/ambiguity_contract_policy_check.py"
    ).read_text(encoding="utf-8")
    scanner_source = Path(
        "src/gabion/tooling/runtime/policy_scanner_suite.py"
    ).read_text(encoding="utf-8")
    delta_source = Path("src/gabion/tooling/delta/delta_gate.py").read_text(
        encoding="utf-8"
    )
    assert "evaluate_policy(" in ambiguity_source
    assert "evaluate_policy(" in scanner_source
    assert "compile_rules(" not in delta_source
    assert "first_match(" not in delta_source


def test_aspf_opportunity_taxonomy_no_python_predicate_registry() -> None:
    source = Path("src/gabion/analysis/foundation/aspf_visitors_impl.py").read_text(
        encoding="utf-8"
    )
    assert "OpportunityTaxonomyRegistry" not in source
    assert "OpportunityAlgebraicPredicate" not in source
    assert "classify_aspf_opportunity(" in source


def test_projection_rule_blocks_on_unerased_obligation() -> None:
    decision = evaluate_policy(
        domain=PolicyDomain.PROJECTION_FIBER,
        data={
            "witness_rows": [
                {
                    "witness_kind": "unmapped_witness",
                    "mapping_complete": False,
                    "boundary_crossed": True,
                }
            ]
        },
    )
    assert decision.rule_id == "projection_fiber.convergence.blocking"


def test_projection_rule_passes_after_erasure() -> None:
    decision = evaluate_policy(
        domain=PolicyDomain.PROJECTION_FIBER,
        data={
            "witness_rows": [
                {
                    "witness_kind": "unmapped_witness",
                    "mapping_complete": True,
                    "boundary_crossed": True,
                }
            ]
        },
    )
    assert decision.rule_id == "projection_fiber.convergence.ok"


def test_projection_fiber_transforms_are_loaded_in_registry() -> None:
    transforms = tuple(build_registry().program.transforms)
    transform_ids = tuple(item.transform_id for item in transforms)
    assert "projection.unmapped_intro" in transform_ids


def test_aspf_lattice_algebra_has_no_projection_transform_runtime() -> None:
    source = Path("src/gabion/analysis/aspf/aspf_lattice_algebra.py").read_text(
        encoding="utf-8"
    )
    assert "projection_fiber_rules.yaml" not in source
    assert "_projection_transform_specs" not in source
    assert "_run_projection_fixpoint" not in source
    assert "RecombinationFrontier" not in source
    assert "compute_recombination_frontier" not in source
    assert "empty_recombination_frontier" not in source


def test_policy_substrate_exports_no_legacy_recombination_surface() -> None:
    adapter_source = Path(
        "src/gabion/tooling/policy_substrate/dataflow_fibration.py"
    ).read_text(encoding="utf-8")
    package_source = Path("src/gabion/tooling/policy_substrate/__init__.py").read_text(
        encoding="utf-8"
    )
    banned = (
        "RecombinationFrontier",
        "compute_recombination_frontier",
        "empty_recombination_frontier",
    )
    for symbol in banned:
        assert symbol not in adapter_source
        assert symbol not in package_source


def test_opportunity_rule_ids_deterministic_order() -> None:
    first = tuple(
        rule.rule_id
        for rule in build_registry().program.by_domain(PolicyDomain.ASPF_OPPORTUNITY)
    )
    second = tuple(
        rule.rule_id
        for rule in build_registry().program.by_domain(PolicyDomain.ASPF_OPPORTUNITY)
    )
    assert first == second
    assert "aspf.opportunity.fungible_execution_path_substitution" in first
    assert "aspf.opportunity.materialize_load_fusion" in first


def test_policy_check_lattice_convergence_uses_semantic_collector_and_dsl() -> None:
    source = Path("scripts/policy/policy_check.py").read_text(encoding="utf-8")
    assert "iter_semantic_lattice_convergence" in source
    assert "materialize_semantic_lattice_convergence" in source
    assert "PolicyDomain.PROJECTION_FIBER" in source
    assert "collect_lattice_convergence_probe" not in source
    assert "legacy frontier implementation token remains" not in source
