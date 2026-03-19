from __future__ import annotations

from gabion.analysis.foundation.marker_protocol import MarkerLifecycleState
from gabion.tooling.policy_substrate.connectivity_synergy_registry import (
    connectivity_synergy_workstream_registries,
)
from gabion.tooling.policy_substrate.invariant_graph import (
    declared_workstream_registries,
)
from gabion.tooling.policy_substrate.policy_rule_frontmatter_migration_registry import (
    prf_workstream_registry,
)
from gabion.tooling.policy_substrate.public_surface_normalization_registry import (
    public_surface_normalization_workstream_registry,
)
from gabion.tooling.policy_substrate.projection_semantic_fragment_phase5_registry import (
    phase5_workstream_registry,
)
from gabion.tooling.policy_substrate.surface_contract_convergence_registry import (
    surface_contract_convergence_workstream_registry,
)
from gabion.tooling.policy_substrate.runtime_context_injection_registry import (
    runtime_context_injection_workstream_registry,
)
from gabion.tooling.policy_substrate.boundary_ingress_convergence_registry import (
    boundary_ingress_convergence_workstream_registry,
)
from gabion.tooling.policy_substrate.dataflow_grammar_readiness_registry import (
    dataflow_grammar_readiness_workstream_registry,
)
from gabion.tooling.policy_substrate.unit_test_readiness_registry import (
    unit_test_readiness_workstream_registry,
)
from gabion.tooling.policy_substrate.structural_anti_pattern_convergence_registry import (
    structural_anti_pattern_convergence_workstream_registry,
)
from gabion.tooling.policy_substrate.wrapper_retirement_drain_registry import (
    wrapper_retirement_drain_workstream_registry,
)


# gabion:behavior primary=desired
def test_prf_workstream_registry_exposes_queue_sequence_and_active_playbook_touchpoint() -> None:
    registry = prf_workstream_registry()
    touchpoints = {item.touchpoint_id: item for item in registry.touchpoints}
    subqueues = {item.subqueue_id: item for item in registry.subqueues}

    assert registry.root.root_id == "PRF"
    assert registry.tags == ("registry_convergence",)
    assert registry.root.status_hint == "landed"
    assert registry.root.marker_payload.lifecycle_state is MarkerLifecycleState.LANDED
    assert registry.root.subqueue_ids == (
        "PRF-001",
        "PRF-002",
        "PRF-003",
        "PRF-004",
        "PRF-005",
        "PRF-006",
        "PRF-007",
        "PRF-008",
        "PRF-009",
    )
    assert tuple(item.subqueue_id for item in registry.subqueues) == (
        "PRF-001",
        "PRF-002",
        "PRF-003",
        "PRF-004",
        "PRF-005",
        "PRF-006",
        "PRF-007",
        "PRF-008",
        "PRF-009",
    )
    assert subqueues["PRF-005"].status_hint == "landed"
    assert subqueues["PRF-005"].marker_payload.lifecycle_state is MarkerLifecycleState.LANDED
    assert subqueues["PRF-005"].touchpoint_ids == ()
    assert subqueues["PRF-006"].status_hint == "landed"
    assert subqueues["PRF-006"].touchpoint_ids == ("PRF-TP-006",)
    assert subqueues["PRF-007"].status_hint == "landed"
    assert subqueues["PRF-007"].touchpoint_ids == ("PRF-TP-007",)
    assert subqueues["PRF-008"].status_hint == "landed"
    assert subqueues["PRF-008"].touchpoint_ids == ("PRF-TP-008",)
    assert subqueues["PRF-009"].status_hint == "landed"
    assert subqueues["PRF-009"].touchpoint_ids == ("PRF-TP-009",)
    assert all(
        item.touchpoint_ids == ()
        for item in registry.subqueues
        if item.subqueue_id in {"PRF-001", "PRF-002", "PRF-003", "PRF-004", "PRF-005"}
    )
    assert set(touchpoints) == {
        "PRF-TP-006",
        "PRF-TP-007",
        "PRF-TP-008",
        "PRF-TP-009",
    }
    assert all(item.status_hint == "landed" for item in touchpoints.values())
    assert all(
        item.marker_payload.lifecycle_state is MarkerLifecycleState.LANDED
        for item in touchpoints.values()
    )
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["PRF-TP-006"].declared_touchsites
    } >= {
        (
            "docs/governance_control_loops.md",
            "governance_control_loops#registry",
        ),
        (
            "docs/governance_loop_matrix.md",
            "governance_loop_matrix#generated_matrix",
        ),
        ("docs/governance_rules.yaml", "governance_rules.gates"),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["PRF-TP-007"].declared_touchsites
    } >= {
        (
            "docs/policy_rules/ambiguity_contract.md",
            "ambiguity_contract_policy_rules",
        ),
        (
            "docs/policy_rules/grade_monotonicity.md",
            "grade_monotonicity_policy_rules",
        ),
        (
            "src/gabion/tooling/policy_substrate/policy_rule_playbook_docs.py",
            "render_policy_rule_playbook_docs",
        ),
        (
            "scripts/policy/render_policy_rule_playbooks.py",
            "render_policy_rule_playbooks",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["PRF-TP-008"].declared_touchsites
    } >= {
        ("AGENTS.md", "AGENTS.md#agent_obligations"),
        ("CONTRIBUTING.md", "CONTRIBUTING.md#contributing_contract"),
        ("docs/clause_obligation_decks.yaml", "clause_obligation_decks"),
        (
            "src/gabion/tooling/policy_substrate/clause_obligation_decks.py",
            "render_clause_obligation_decks",
        ),
        (
            "scripts/policy/render_clause_obligation_decks.py",
            "render_clause_obligation_decks",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["PRF-TP-009"].declared_touchsites
    } >= {
        ("docs/enforceable_rules_cheat_sheet.md", "enforceable_rules_cheat_sheet"),
        ("docs/enforceable_rules_catalog.yaml", "enforceable_rules_catalog"),
        ("docs/governance_control_loops.yaml", "governance_control_loops"),
        (
            "src/gabion/tooling/policy_substrate/enforceable_rules_cheat_sheet.py",
            "render_enforceable_rules_cheat_sheet",
        ),
        (
            "scripts/policy/render_enforceable_rules_cheat_sheet.py",
            "render_enforceable_rules_cheat_sheet",
        ),
    }


# gabion:behavior primary=desired
def test_phase5_workstream_registry_exposes_touchpoint_scan_contract() -> None:
    registry = phase5_workstream_registry()
    touchpoints = {item.touchpoint_id: item for item in registry.touchpoints}

    assert registry.root.root_id == "PSF-007"
    assert registry.tags == ("identity_rendering",)
    assert registry.root.subqueue_ids == (
        "PSF-007-SQ-001",
        "PSF-007-SQ-002",
        "PSF-007-SQ-003",
        "PSF-007-SQ-004",
        "PSF-007-SQ-005",
    )
    assert all(item.scan_touchsites for item in registry.touchpoints)
    assert touchpoints["PSF-007-TP-001"].surviving_boundary_names == (
        "semantic_fragment.reflect_projection_fiber_witness",
        "semantic_fragment.canonical_value_materialization",
    )
    assert (
        touchpoints["PSF-007-TP-001"].declared_counterfactual_actions[0].action_id
        == "PSF-007-TP-001-ACT-001"
    )
    assert (
        touchpoints["PSF-007-TP-001"].declared_counterfactual_actions[0].predicted_readiness_class
        == "policy_blocked"
    )


# gabion:behavior primary=desired
def test_surface_contract_convergence_workstream_registry_exposes_queue_and_touchsites() -> None:
    registry = surface_contract_convergence_workstream_registry()
    touchpoints = {item.touchpoint_id: item for item in registry.touchpoints}
    subqueues = {item.subqueue_id: item for item in registry.subqueues}

    assert registry.root.root_id == "SCC"
    assert registry.tags == ("contract_convergence",)
    assert registry.root.status_hint == "landed"
    assert registry.root.marker_payload.lifecycle_state is MarkerLifecycleState.LANDED
    assert registry.root.subqueue_ids == (
        "SCC-SQ-001",
        "SCC-SQ-002",
        "SCC-SQ-003",
        "SCC-SQ-004",
    )
    assert tuple(item.subqueue_id for item in registry.subqueues) == (
        "SCC-SQ-001",
        "SCC-SQ-002",
        "SCC-SQ-003",
        "SCC-SQ-004",
    )
    assert subqueues["SCC-SQ-001"].touchpoint_ids == (
        "SCC-TP-001",
        "SCC-TP-002",
        "SCC-TP-008",
    )
    assert subqueues["SCC-SQ-002"].touchpoint_ids == ("SCC-TP-003", "SCC-TP-004")
    assert subqueues["SCC-SQ-003"].touchpoint_ids == ("SCC-TP-005", "SCC-TP-006")
    assert subqueues["SCC-SQ-004"].touchpoint_ids == ("SCC-TP-007",)
    assert all(item.status_hint == "landed" for item in registry.subqueues)
    assert all(
        item.marker_payload.lifecycle_state is MarkerLifecycleState.LANDED
        for item in registry.subqueues
    )
    assert set(touchpoints) == {
        "SCC-TP-001",
        "SCC-TP-002",
        "SCC-TP-003",
        "SCC-TP-004",
        "SCC-TP-005",
        "SCC-TP-006",
        "SCC-TP-007",
        "SCC-TP-008",
    }
    assert touchpoints["SCC-TP-001"].status_hint == "landed"
    assert touchpoints["SCC-TP-002"].status_hint == "landed"
    assert touchpoints["SCC-TP-003"].status_hint == "landed"
    assert touchpoints["SCC-TP-004"].status_hint == "landed"
    assert touchpoints["SCC-TP-005"].status_hint == "landed"
    assert touchpoints["SCC-TP-006"].status_hint == "landed"
    assert touchpoints["SCC-TP-007"].status_hint == "landed"
    assert touchpoints["SCC-TP-008"].status_hint == "landed"
    assert all(
        item.marker_payload.lifecycle_state is MarkerLifecycleState.LANDED
        for item in registry.touchpoints
    )
    assert all(
        touchpoints[touchpoint_id].status_hint == "queued"
        for touchpoint_id in touchpoints
        if touchpoint_id
        not in {
            "SCC-TP-001",
            "SCC-TP-002",
            "SCC-TP-003",
            "SCC-TP-004",
            "SCC-TP-005",
            "SCC-TP-006",
            "SCC-TP-007",
            "SCC-TP-008",
        }
    )
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["SCC-TP-001"].declared_touchsites
    } >= {
        ("src/gabion/runtime/coercion_contract.py", "coercion_contract"),
        ("src/gabion/runtime_shape_dispatch.py", "runtime_shape_dispatch"),
        ("src/gabion/commands/progress_contract.py", "progress_contract"),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["SCC-TP-004"].declared_touchsites
    } >= {
        ("src/gabion/server_core/output_primitives.py", "output_primitives"),
        ("src/gabion/server_core/progress_primitives.py", "progress_primitives"),
        ("src/gabion/server_core/timeout_primitives.py", "timeout_primitives"),
        ("src/gabion/server_core/ingress_contracts.py", "ingress_contracts"),
        (
            "src/gabion/tooling/policy_rules/orchestrator_primitive_barrel_rule.py",
            "orchestrator_primitive_barrel_rule",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["SCC-TP-005"].declared_touchsites
    } >= {
        (
            "src/gabion/analysis/dataflow/engine/dataflow_indexed_file_scan.py",
            "dataflow_indexed_file_scan",
        ),
        (
            "src/gabion/analysis/dataflow/engine/dataflow_indexed_file_scan_alias_inventory.py",
            "dataflow_indexed_file_scan_alias_inventory",
        ),
        (
            "docs/audits/dataflow_runtime_debt_ledger.md",
            "dataflow_runtime_debt_ledger",
        ),
        (
            "docs/audits/dataflow_runtime_retirement_ledger.md",
            "dataflow_runtime_retirement_ledger",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["SCC-TP-006"].declared_touchsites
    } >= {
        (
            "src/gabion/analysis/dataflow/engine/dataflow_indexed_file_scan.py",
            "dataflow_indexed_file_scan",
        ),
        (
            "src/gabion/analysis/dataflow/engine/dataflow_indexed_file_scan_alias_inventory.py",
            "dataflow_indexed_file_scan_alias_inventory",
        ),
        (
            "src/gabion/analysis/dataflow/engine/dataflow_indexed_file_scan_alias_contract.py",
            "dataflow_indexed_file_scan_alias_contract",
        ),
        (
            "src/gabion/analysis/dataflow/engine/dataflow_indexed_file_scan_alias_adapter_compatibility.py",
            "dataflow_indexed_file_scan_alias_adapter_compatibility",
        ),
        (
            "src/gabion/analysis/dataflow/engine/dataflow_indexed_file_scan_alias_adapter_decision.py",
            "dataflow_indexed_file_scan_alias_adapter_decision",
        ),
        (
            "src/gabion/analysis/dataflow/engine/dataflow_indexed_file_scan_alias_adapter_runtime.py",
            "dataflow_indexed_file_scan_alias_adapter_runtime",
        ),
        (
            "src/gabion/analysis/dataflow/engine/dataflow_indexed_file_scan_alias_adapter_analysis.py",
            "dataflow_indexed_file_scan_alias_adapter_analysis",
        ),
        (
            "src/gabion/analysis/dataflow/engine/dataflow_indexed_file_scan_alias_adapter_projection.py",
            "dataflow_indexed_file_scan_alias_adapter_projection",
        ),
        (
            "docs/audits/dataflow_runtime_debt_ledger.md",
            "dataflow_runtime_debt_ledger",
        ),
        (
            "docs/audits/dataflow_runtime_retirement_ledger.md",
            "dataflow_runtime_retirement_ledger",
        ),
        (
            "docs/audits/dataflow_legacy_monolith_test_replacement_matrix.md",
            "dataflow_legacy_monolith_test_replacement_matrix",
        ),
        (
            "docs/compatibility_layer_debt_register.md",
            "compatibility_layer_debt_register",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["SCC-TP-007"].declared_touchsites
    } >= {
        ("src/gabion_governance/governance_audit_impl.py", "governance_audit_impl"),
        ("AGENTS.md", "AGENTS.md#agent_obligations"),
        ("README.md", "README.md#repo_contract"),
        ("CONTRIBUTING.md", "CONTRIBUTING.md#contributing_contract"),
        ("POLICY_SEED.md", "POLICY_SEED.md#policy_seed"),
        ("glossary.md", "glossary.md#contract"),
        (
            "docs/normative_clause_index.md",
            "docs/normative_clause_index.md#normative_clause_index",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["SCC-TP-008"].declared_touchsites
    } >= {
        ("src/gabion/server_core/command_orchestrator.py", "command_orchestrator"),
        ("src/gabion/config.py", "config"),
    }


# gabion:behavior primary=desired
def test_runtime_context_injection_workstream_registry_exposes_queue_and_touchsites() -> None:
    registry = runtime_context_injection_workstream_registry()
    touchpoints = {item.touchpoint_id: item for item in registry.touchpoints}
    subqueues = {item.subqueue_id: item for item in registry.subqueues}

    assert registry.root.root_id == "RCI"
    assert registry.tags == ("runtime_context_injection",)
    assert registry.root.status_hint == "landed"
    assert registry.root.marker_payload.lifecycle_state is MarkerLifecycleState.LANDED
    assert registry.root.subqueue_ids == (
        "RCI-SQ-001",
        "RCI-SQ-002",
        "RCI-SQ-003",
        "RCI-SQ-004",
    )
    assert tuple(item.subqueue_id for item in registry.subqueues) == (
        "RCI-SQ-001",
        "RCI-SQ-002",
        "RCI-SQ-003",
        "RCI-SQ-004",
    )
    assert subqueues["RCI-SQ-001"].touchpoint_ids == ("RCI-TP-001",)
    assert subqueues["RCI-SQ-002"].touchpoint_ids == ("RCI-TP-002", "RCI-TP-003")
    assert subqueues["RCI-SQ-003"].touchpoint_ids == ("RCI-TP-004", "RCI-TP-005")
    assert subqueues["RCI-SQ-004"].touchpoint_ids == (
        "RCI-TP-006",
        "RCI-TP-007",
        "RCI-TP-008",
    )
    assert all(item.status_hint == "landed" for item in registry.subqueues)
    assert all(
        item.marker_payload.lifecycle_state is MarkerLifecycleState.LANDED
        for item in registry.subqueues
    )
    assert set(touchpoints) == {
        "RCI-TP-001",
        "RCI-TP-002",
        "RCI-TP-003",
        "RCI-TP-004",
        "RCI-TP-005",
        "RCI-TP-006",
        "RCI-TP-007",
        "RCI-TP-008",
    }
    assert all(item.status_hint == "landed" for item in registry.touchpoints)
    assert all(
        item.marker_payload.lifecycle_state is MarkerLifecycleState.LANDED
        for item in registry.touchpoints
    )
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["RCI-TP-001"].declared_touchsites
    } == {
        ("src/gabion/tooling/policy_substrate/invariant_graph.py", "invariant_graph"),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["RCI-TP-002"].declared_touchsites
    } == {
        (
            "tests/gabion/tooling/runtime_policy/invariant_graph_test_support.py",
            "invariant_graph_test_support",
        ),
        ("tests/gabion/tooling/runtime_policy/test_invariant_graph.py", "test_invariant_graph"),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["RCI-TP-003"].declared_touchsites
    } == {
        (
            "tests/gabion/tooling/runtime_policy/invariant_graph_test_support.py",
            "invariant_graph_test_support",
        ),
        ("tests/gabion/tooling/runtime_policy/test_invariant_graph.py", "test_invariant_graph"),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["RCI-TP-004"].declared_touchsites
    } == {
        ("scripts/policy/policy_check.py", "policy_check"),
        (
            "tests/gabion/tooling/runtime_policy/test_policy_check_output.py",
            "test_policy_check_output",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["RCI-TP-005"].declared_touchsites
    } == {
        ("src/gabion/tooling/runtime/invariant_graph.py", "runtime_invariant_graph"),
        (
            "tests/gabion/tooling/runtime_policy/test_runtime_invariant_graph_perf.py",
            "test_runtime_invariant_graph_perf",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["RCI-TP-006"].declared_touchsites
    } == {
        ("tests/gabion/tooling/runtime_policy/test_invariant_graph.py", "test_invariant_graph"),
        (
            "tests/gabion/tooling/runtime_policy/test_invariant_graph_live_repo.py",
            "test_invariant_graph_live_repo",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["RCI-TP-007"].declared_touchsites
    } == {
        ("tests/gabion/tooling/runtime_policy/test_invariant_graph.py", "test_invariant_graph"),
        (
            "tests/gabion/tooling/runtime_policy/test_invariant_graph_live_repo.py",
            "test_invariant_graph_live_repo",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["RCI-TP-008"].declared_touchsites
    } == {
        (
            "tests/gabion/tooling/runtime_policy/test_connectivity_synergy_registry_live_repo.py",
            "test_connectivity_synergy_registry_live_repo",
        ),
        (
            "tests/gabion/tooling/runtime_policy/test_identity_grammar_completion_artifact_live_repo.py",
            "test_identity_grammar_completion_artifact_live_repo",
        ),
        (
            "tests/gabion/tooling/runtime_policy/test_invariant_graph_live_repo.py",
            "test_invariant_graph_live_repo",
        ),
        (
            "tests/gabion/tooling/runtime_policy/test_kernel_vm_alignment_artifact_live_repo.py",
            "test_kernel_vm_alignment_artifact_live_repo",
        ),
    }


# gabion:behavior primary=desired
def test_boundary_ingress_convergence_workstream_registry_exposes_queue_and_touchsites() -> None:
    registry = boundary_ingress_convergence_workstream_registry()
    touchpoints = {item.touchpoint_id: item for item in registry.touchpoints}
    subqueues = {item.subqueue_id: item for item in registry.subqueues}

    assert registry.root.root_id == "BIC"
    assert registry.tags == ("boundary_ingress_convergence",)
    assert registry.root.status_hint == "landed"
    assert registry.root.marker_payload.lifecycle_state is MarkerLifecycleState.LANDED
    assert registry.root.subqueue_ids == (
        "BIC-SQ-001",
        "BIC-SQ-002",
        "BIC-SQ-003",
    )
    assert tuple(item.subqueue_id for item in registry.subqueues) == (
        "BIC-SQ-001",
        "BIC-SQ-002",
        "BIC-SQ-003",
    )
    assert subqueues["BIC-SQ-001"].touchpoint_ids == ("BIC-TP-001", "BIC-TP-005")
    assert subqueues["BIC-SQ-002"].touchpoint_ids == (
        "BIC-TP-002",
        "BIC-TP-003",
        "BIC-TP-006",
    )
    assert subqueues["BIC-SQ-003"].touchpoint_ids == ("BIC-TP-004",)
    assert all(item.status_hint == "landed" for item in registry.subqueues)
    assert all(
        item.marker_payload.lifecycle_state is MarkerLifecycleState.LANDED
        for item in registry.subqueues
    )
    assert set(touchpoints) == {
        "BIC-TP-001",
        "BIC-TP-005",
        "BIC-TP-002",
        "BIC-TP-003",
        "BIC-TP-006",
        "BIC-TP-004",
    }
    assert all(item.status_hint == "landed" for item in registry.touchpoints)
    assert all(
        item.marker_payload.lifecycle_state is MarkerLifecycleState.LANDED
        for item in registry.touchpoints
    )
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["BIC-TP-001"].declared_touchsites
    } == {
        (
            "src/gabion/cli_support/shared/dataflow_transport_ingress.py",
            "dataflow_transport_ingress",
        ),
        ("src/gabion/cli.py", "cli"),
        (
            "src/gabion/tooling/runtime/dataflow_invocation_runner.py",
            "dataflow_invocation_runner",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["BIC-TP-005"].declared_touchsites
    } == {
        (
            "src/gabion/tooling/runtime/dataflow_invocation_runner.py",
            "dataflow_invocation_runner",
        ),
        (
            "src/gabion/cli_support/shared/dataflow_transport_ingress.py",
            "dataflow_transport_ingress",
        ),
        (
            "tests/gabion/analysis/dataflow_s1/test_dataflow_invocation_runner.py",
            "test_dataflow_invocation_runner",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["BIC-TP-002"].declared_touchsites
    } == {
        ("src/gabion/server_core/coercion_contract.py", "coercion_contract"),
        ("src/gabion/server_core/command_orchestrator.py", "command_orchestrator"),
        (
            "src/gabion/server_core/command_orchestrator_progress.py",
            "command_orchestrator_progress",
        ),
        ("tests/gabion/runtime/test_coercion_contract.py", "test_coercion_contract"),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["BIC-TP-003"].declared_touchsites
    } == {
        (
            "src/gabion/server_core/command_orchestrator_primitives.py",
            "command_orchestrator_primitives",
        ),
        ("src/gabion/server_core/server_payload_dispatch.py", "server_payload_dispatch"),
        ("tests/gabion/runtime/test_coercion_contract.py", "test_coercion_contract"),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["BIC-TP-006"].declared_touchsites
    } == {
        ("src/gabion/server_core/coercion_contract.py", "coercion_contract"),
        ("src/gabion/cli.py", "cli"),
        ("tests/gabion/runtime/test_coercion_contract.py", "test_coercion_contract"),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["BIC-TP-004"].declared_touchsites
    } == {
        ("tests/gabion/cli/cli_commands_cases.py", "cli_commands_cases"),
        ("tests/gabion/cli/cli_live_repo_cases.py", "cli_live_repo_cases"),
        ("tests/gabion/cli/test_cli.py", "test_cli"),
        ("tests/gabion/cli/test_cli_live_repo.py", "test_cli_live_repo"),
    }


# gabion:behavior primary=desired
def test_unit_test_readiness_workstream_registry_exposes_selector_clusters() -> None:
    registry = unit_test_readiness_workstream_registry()
    subqueues = {item.subqueue_id: item for item in registry.subqueues}
    touchpoints = {item.touchpoint_id: item for item in registry.touchpoints}

    assert registry.root.root_id == "UTR"
    assert registry.tags == ("unit_test_readiness",)
    assert registry.root.status_hint == "in_progress"
    assert registry.root.subqueue_ids == (
        "UTR-SQ-001",
        "UTR-SQ-002",
        "UTR-SQ-003",
        "UTR-SQ-004",
    )
    assert tuple(item.subqueue_id for item in registry.subqueues) == (
        "UTR-SQ-001",
        "UTR-SQ-002",
        "UTR-SQ-003",
        "UTR-SQ-004",
    )
    assert subqueues["UTR-SQ-001"].touchpoint_ids == ("UTR-TP-001", "UTR-TP-002")
    assert subqueues["UTR-SQ-002"].touchpoint_ids == ("UTR-TP-003", "UTR-TP-004")
    assert subqueues["UTR-SQ-003"].touchpoint_ids == (
        "UTR-TP-005",
        "UTR-TP-006",
        "UTR-TP-007",
    )
    assert subqueues["UTR-SQ-004"].touchpoint_ids == ("UTR-TP-008",)
    assert all(item.status_hint == "in_progress" for item in registry.subqueues)
    assert set(touchpoints) == {
        "UTR-TP-001",
        "UTR-TP-002",
        "UTR-TP-003",
        "UTR-TP-004",
        "UTR-TP-005",
        "UTR-TP-006",
        "UTR-TP-007",
        "UTR-TP-008",
    }
    assert touchpoints["UTR-TP-001"].status_hint == "landed"
    assert all(
        item.status_hint == "queued"
        for touchpoint_id, item in touchpoints.items()
        if touchpoint_id != "UTR-TP-001"
    )
    assert touchpoints["UTR-TP-001"].test_path_prefixes == (
        "tests/gabion/analysis/evidence/",
        "tests/gabion/analysis/type/",
        "tests/gabion/config/",
        "tests/gabion/analysis/indexed_scan/",
        "tests/gabion/analysis/structure/",
        "tests/gabion/analysis/forest/",
        "tests/gabion/analysis/call_cluster/",
    )
    assert touchpoints["UTR-TP-005"].test_path_prefixes == (
        "tests/gabion/tooling/ci/test_ci_governance_scripts.py",
    )
    assert touchpoints["UTR-TP-006"].test_path_prefixes == (
        "tests/gabion/tooling/policy/test_render_generated_artifact_manifest.py",
    )
    assert touchpoints["UTR-TP-008"].test_path_prefixes == (
        "tests/gabion/server_core/test_command_orchestrator.py",
        "tests/gabion/server/test_server.py",
        "tests/gabion/runtime/test_runtime_kernel_contracts.py",
    )


# gabion:behavior primary=desired
def test_dataflow_grammar_readiness_workstream_registry_exposes_local_signal_clusters() -> None:
    registry = dataflow_grammar_readiness_workstream_registry()
    subqueues = {item.subqueue_id: item for item in registry.subqueues}
    touchpoints = {item.touchpoint_id: item for item in registry.touchpoints}

    assert registry.root.root_id == "DGR"
    assert registry.tags == ("dataflow_grammar_readiness",)
    assert registry.root.status_hint == "in_progress"
    assert registry.root.subqueue_ids == ("DGR-SQ-001", "DGR-SQ-002")
    assert tuple(item.subqueue_id for item in registry.subqueues) == (
        "DGR-SQ-001",
        "DGR-SQ-002",
    )
    assert subqueues["DGR-SQ-001"].touchpoint_ids == ("DGR-TP-001", "DGR-TP-002")
    assert subqueues["DGR-SQ-002"].touchpoint_ids == ("DGR-TP-003", "DGR-TP-004")
    assert all(item.status_hint == "in_progress" for item in registry.subqueues)
    assert set(touchpoints) == {
        "DGR-TP-001",
        "DGR-TP-002",
        "DGR-TP-003",
        "DGR-TP-004",
    }
    assert (
        touchpoints["DGR-TP-001"].dataflow_signal_selector is not None
        and touchpoints["DGR-TP-001"].dataflow_signal_selector.terminal_statuses
        == ("hard_failure",)
    )
    assert (
        touchpoints["DGR-TP-002"].dataflow_signal_selector is not None
        and touchpoints["DGR-TP-002"].dataflow_signal_selector.terminal_statuses
        == ("timeout_resume",)
    )
    assert (
        touchpoints["DGR-TP-002"].dataflow_signal_selector is not None
        and touchpoints["DGR-TP-002"].dataflow_signal_selector.incompleteness_markers
        == ("terminal_non_success", "timeout_or_partial_run")
    )
    assert (
        touchpoints["DGR-TP-003"].dataflow_signal_selector is not None
        and touchpoints["DGR-TP-003"].dataflow_signal_selector.obligation_statuses
        == ("unsatisfied",)
    )
    assert (
        touchpoints["DGR-TP-004"].dataflow_signal_selector is not None
        and touchpoints["DGR-TP-004"].dataflow_signal_selector.obligation_statuses
        == ("skipped_by_policy",)
    )
    assert all(not item.test_path_prefixes for item in registry.touchpoints)


# gabion:behavior primary=desired
def test_structural_anti_pattern_convergence_workstream_registry_exposes_contract_root() -> None:
    registry = structural_anti_pattern_convergence_workstream_registry()
    subqueues = {item.subqueue_id: item for item in registry.subqueues}
    touchpoints = {item.touchpoint_id: item for item in registry.touchpoints}

    assert registry.root.root_id == "SAC"
    assert registry.tags == ("structural_convergence",)
    assert registry.root.status_hint == "in_progress"
    assert registry.root.subqueue_ids == (
        "SAC-SQ-001",
        "SAC-SQ-002",
        "SAC-SQ-003",
        "SAC-SQ-004",
    )
    assert tuple(item.subqueue_id for item in registry.subqueues) == (
        "SAC-SQ-001",
        "SAC-SQ-002",
        "SAC-SQ-003",
        "SAC-SQ-004",
    )
    assert subqueues["SAC-SQ-001"].touchpoint_ids == ("SAC-TP-001",)
    assert subqueues["SAC-SQ-002"].touchpoint_ids == ("SAC-TP-002",)
    assert subqueues["SAC-SQ-003"].touchpoint_ids == ("SAC-TP-003",)
    assert subqueues["SAC-SQ-004"].touchpoint_ids == ("SAC-TP-004", "SAC-TP-005")
    assert all(item.status_hint == "in_progress" for item in registry.subqueues)
    assert set(touchpoints) == {
        "SAC-TP-001",
        "SAC-TP-002",
        "SAC-TP-003",
        "SAC-TP-004",
        "SAC-TP-005",
    }
    assert all(item.status_hint == "queued" for item in touchpoints.values())
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["SAC-TP-005"].declared_touchsites
    } == {
        (
            "src/gabion/tooling/policy_substrate/structural_anti_pattern_contract.py",
            "collect_findings",
        ),
        (
            "scripts/policy/structural_anti_pattern_contract.py",
            "main",
        ),
        (
            "scripts/policy/policy_check.py",
            "check_structural_anti_pattern_contract",
        ),
        (
            "tests/gabion/tooling/runtime_policy/test_structural_anti_pattern_contract.py",
            "test_collect_findings",
        ),
    }


# gabion:behavior primary=desired
def test_wrapper_retirement_drain_workstream_registry_exposes_migration_program() -> None:
    registry = wrapper_retirement_drain_workstream_registry()
    subqueues = {item.subqueue_id: item for item in registry.subqueues}
    touchpoints = {item.touchpoint_id: item for item in registry.touchpoints}

    assert registry.root.root_id == "WRD"
    assert registry.tags == ("wrapper_retirement",)
    assert registry.root.status_hint == "landed"
    assert registry.root.marker_payload.lifecycle_state is MarkerLifecycleState.LANDED
    assert registry.root.subqueue_ids == (
        "WRD-SQ-001",
        "WRD-SQ-002",
        "WRD-SQ-003",
        "WRD-SQ-004",
        "WRD-SQ-005",
    )
    assert tuple(item.subqueue_id for item in registry.subqueues) == (
        "WRD-SQ-001",
        "WRD-SQ-002",
        "WRD-SQ-003",
        "WRD-SQ-004",
        "WRD-SQ-005",
    )
    assert subqueues["WRD-SQ-001"].status_hint == "landed"
    assert subqueues["WRD-SQ-001"].touchpoint_ids == (
        "WRD-TP-001",
        "WRD-TP-002",
        "WRD-TP-003",
    )
    assert subqueues["WRD-SQ-002"].status_hint == "landed"
    assert subqueues["WRD-SQ-002"].touchpoint_ids == ("WRD-TP-004", "WRD-TP-005")
    assert subqueues["WRD-SQ-003"].status_hint == "landed"
    assert subqueues["WRD-SQ-003"].touchpoint_ids == ("WRD-TP-006",)
    assert subqueues["WRD-SQ-004"].status_hint == "landed"
    assert subqueues["WRD-SQ-004"].touchpoint_ids == ("WRD-TP-007",)
    assert subqueues["WRD-SQ-005"].status_hint == "landed"
    assert subqueues["WRD-SQ-005"].touchpoint_ids == ("WRD-TP-008",)
    assert all(
        item.marker_payload.lifecycle_state is MarkerLifecycleState.LANDED
        for item in registry.subqueues
    )
    assert set(touchpoints) == {
        "WRD-TP-001",
        "WRD-TP-002",
        "WRD-TP-003",
        "WRD-TP-004",
        "WRD-TP-005",
        "WRD-TP-006",
        "WRD-TP-007",
        "WRD-TP-008",
    }
    assert touchpoints["WRD-TP-001"].status_hint == "landed"
    assert touchpoints["WRD-TP-002"].status_hint == "landed"
    assert touchpoints["WRD-TP-003"].status_hint == "landed"
    assert touchpoints["WRD-TP-004"].status_hint == "landed"
    assert touchpoints["WRD-TP-005"].status_hint == "landed"
    assert touchpoints["WRD-TP-006"].status_hint == "landed"
    assert touchpoints["WRD-TP-007"].status_hint == "landed"
    assert touchpoints["WRD-TP-008"].status_hint == "landed"
    assert all(
        item.marker_payload.lifecycle_state is MarkerLifecycleState.LANDED
        for item in registry.touchpoints
    )
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["WRD-TP-001"].declared_touchsites
    } >= {
        ("src/gabion/tooling/runtime/run_dataflow_stage.py", "main"),
        (
            "src/gabion/cli_support/tooling_commands.py",
            "register_tooling_passthrough_commands.<locals>.run_dataflow_stage",
        ),
        ("docs/user_workflows.md", "user_workflows"),
        (".github/workflows/ci.yml", "ci_workflow_dataflow_grammar_invocation"),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["WRD-TP-008"].declared_touchsites
    } >= {
        ("docs/user_workflows.md", "user_workflows"),
        ("README.md", "repo_contract"),
        ("CONTRIBUTING.md", "contributing_contract"),
        (".github/workflows/ci.yml", "ci_workflow_wrapper_invocations"),
        ("docs/generated_artifact_manifest.md", "generated_artifact_manifest"),
        ("docs/generated_artifact_manifest.yaml", "generated_artifact_manifest"),
        ("docs/governance_control_loops.yaml", "governance_control_loops"),
        ("docs/governance_loop_matrix.md", "governance_loop_matrix"),
        (".github/workflows/pr-dataflow-grammar.yml", "pr_dataflow_grammar_workflow_wrapper_invocations"),
        (".github/workflows/release-tag.yml", "release_tag_workflow_wrapper_invocations"),
        (".github/workflows/auto-test-tag.yml", "auto_test_tag_workflow_wrapper_invocations"),
        (".github/workflows/release-testpypi.yml", "release_testpypi_workflow_wrapper_invocations"),
        (".github/workflows/release-pypi.yml", "release_pypi_workflow_wrapper_invocations"),
        ("Makefile", "make_targets_wrapper_guidance"),
        ("scripts/policy/policy_check.py", "main"),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["WRD-TP-007"].declared_touchsites
    } >= {
        ("scripts/release/release_tag.py", "main"),
        ("scripts/release/release_read_project_version.py", "main"),
        ("scripts/release/release_set_test_version.py", "main"),
        ("scripts/release/release_verify_test_tag.py", "main"),
        ("scripts/release/release_verify_pypi_tag.py", "main"),
        ("scripts/misc/extract_test_evidence.py", "main"),
        ("scripts/misc/extract_test_behavior.py", "main"),
        ("scripts/misc/refresh_baselines.py", "main"),
        ("scripts/audit_snapshot.sh", "audit_snapshot_wrapper"),
        ("scripts/latest_snapshot.sh", "latest_snapshot_wrapper"),
    }


# gabion:behavior primary=desired
def test_public_surface_normalization_workstream_registry_exposes_drain_program() -> None:
    registry = public_surface_normalization_workstream_registry()
    subqueues = {item.subqueue_id: item for item in registry.subqueues}
    touchpoints = {item.touchpoint_id: item for item in registry.touchpoints}

    assert registry.root.root_id == "PSN"
    assert registry.tags == ("public_surface_normalization",)
    assert registry.root.status_hint == "in_progress"
    assert registry.root.marker_payload.lifecycle_state is MarkerLifecycleState.ACTIVE
    assert registry.root.subqueue_ids == (
        "PSN-SQ-001",
        "PSN-SQ-002",
        "PSN-SQ-003",
        "PSN-SQ-004",
    )
    assert tuple(item.subqueue_id for item in registry.subqueues) == (
        "PSN-SQ-001",
        "PSN-SQ-002",
        "PSN-SQ-003",
        "PSN-SQ-004",
    )
    assert subqueues["PSN-SQ-001"].status_hint == "in_progress"
    assert subqueues["PSN-SQ-001"].touchpoint_ids == ("PSN-TP-001", "PSN-TP-002")
    assert subqueues["PSN-SQ-002"].status_hint == "in_progress"
    assert subqueues["PSN-SQ-002"].touchpoint_ids == ("PSN-TP-003",)
    assert subqueues["PSN-SQ-003"].status_hint == "in_progress"
    assert subqueues["PSN-SQ-003"].touchpoint_ids == (
        "PSN-TP-004",
        "PSN-TP-005",
        "PSN-TP-006",
    )
    assert subqueues["PSN-SQ-004"].status_hint == "in_progress"
    assert subqueues["PSN-SQ-004"].touchpoint_ids == ("PSN-TP-007", "PSN-TP-008")
    assert all(
        item.marker_payload.lifecycle_state is MarkerLifecycleState.ACTIVE
        for item in registry.subqueues
    )
    assert set(touchpoints) == {
        "PSN-TP-001",
        "PSN-TP-002",
        "PSN-TP-003",
        "PSN-TP-004",
        "PSN-TP-005",
        "PSN-TP-006",
        "PSN-TP-007",
        "PSN-TP-008",
    }
    assert {
        touchpoints["PSN-TP-001"].status_hint,
        touchpoints["PSN-TP-002"].status_hint,
        touchpoints["PSN-TP-003"].status_hint,
        touchpoints["PSN-TP-004"].status_hint,
        touchpoints["PSN-TP-005"].status_hint,
        touchpoints["PSN-TP-006"].status_hint,
        touchpoints["PSN-TP-007"].status_hint,
        touchpoints["PSN-TP-008"].status_hint,
    } == {"queued", "in_progress", "landed"}
    assert touchpoints["PSN-TP-001"].status_hint == "landed"
    assert touchpoints["PSN-TP-002"].status_hint == "in_progress"
    assert touchpoints["PSN-TP-003"].status_hint == "in_progress"
    assert touchpoints["PSN-TP-004"].status_hint == "in_progress"
    assert touchpoints["PSN-TP-005"].status_hint == "landed"
    assert touchpoints["PSN-TP-006"].status_hint == "in_progress"
    assert touchpoints["PSN-TP-007"].status_hint == "queued"
    assert touchpoints["PSN-TP-008"].status_hint == "queued"
    assert (
        touchpoints["PSN-TP-001"].marker_payload.lifecycle_state
        is MarkerLifecycleState.LANDED
    )
    assert all(
        touchpoints[touchpoint_id].marker_payload.lifecycle_state
        is MarkerLifecycleState.ACTIVE
        for touchpoint_id in (
            "PSN-TP-002",
            "PSN-TP-003",
            "PSN-TP-004",
            "PSN-TP-006",
            "PSN-TP-007",
            "PSN-TP-008",
        )
    )
    assert (
        touchpoints["PSN-TP-005"].marker_payload.lifecycle_state
        is MarkerLifecycleState.LANDED
    )
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["PSN-TP-001"].declared_touchsites
    } >= {
        ("src/gabion/cli.py", "_TOOLING_ARGV_RUNNERS"),
        ("scripts/policy/policy_check.py", "main"),
        ("scripts/policy/docflow_packetize.py", "main"),
        ("scripts/policy/docflow_packet_enforce.py", "main"),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["PSN-TP-003"].declared_touchsites
    } >= {
        ("src/gabion/tooling/governance/governance_audit.py", "BOUNDARY_ADAPTER_METADATA"),
        ("src/gabion_governance/governance_entrypoint.py", "main"),
        ("src/gabion_governance/governance_audit_impl.py", "run_docflow_cli"),
        ("src/gabion_governance/governance_audit_impl.py", "run_status_consistency_cli"),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["PSN-TP-005"].declared_touchsites
    } >= {
        (
            "src/gabion/tooling/policy_substrate/structured_artifact_ingress.py",
            "build_ingress_merge_parity_artifact",
        ),
        ("src/gabion/policy_dsl/registry.py", "build_registry_for_root"),
        (
            "src/gabion/tooling/policy_substrate/grade_monotonicity_semantic.py",
            "collect_grade_monotonicity",
        ),
        (
            "src/gabion/analysis/dataflow/engine/dataflow_evidence_helpers.py",
            "module_name",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["PSN-TP-006"].declared_touchsites
    } >= {
        ("src/gabion/analysis/dataflow/engine/dataflow_evidence_helpers.py", "__all__"),
        (
            "src/gabion/analysis/dataflow/engine/dataflow_analysis_index.py",
            "_accumulate_symbol_table_for_tree",
        ),
        (
            "src/gabion/analysis/dataflow/engine/dataflow_projection_materialization.py",
            "_collect_call_ambiguities_indexed",
        ),
        (
            "src/gabion/analysis/dataflow/engine/dataflow_deadline_runtime.py",
            "_COLLECT_DEADLINE_LOCAL_INFO_DEPS",
        ),
        (
            "src/gabion/analysis/dataflow/engine/dataflow_bundle_iteration.py",
            "iter_dataclass_call_bundle_effects",
        ),
        (
            "src/gabion/analysis/dataflow/engine/dataflow_lambda_runtime_support.py",
            "_collect_closure_lambda_factories",
        ),
        (
            "src/gabion/analysis/dataflow/engine/dataflow_obligations.py",
            "_append_origin_obligations",
        ),
        (
            "src/gabion/analysis/dataflow/io/dataflow_refactor_planning.py",
            "build_refactor_plan",
        ),
        (
            "src/gabion/analysis/dataflow/io/dataflow_synthesis.py",
            "_build_synthesis_plan",
        ),
    }
    assert {
        link.value
        for link in touchpoints["PSN-TP-001"].marker_payload.links
        if link.kind == "object_id"
    } >= {"PSN", "PSN-SQ-001", "PSN-TP-001", "WRD-TP-002", "WRD-TP-004"}
    assert {
        (item.rel_path, item.qualname)
        for item in touchpoints["PSN-TP-007"].declared_touchsites
    } >= {
        ("scripts/policy/private_symbol_import_guard.py", "main"),
        ("scripts/policy/policy_check.py", "main"),
    }


# gabion:behavior primary=desired
def test_declared_workstream_registries_include_structural_anti_pattern_convergence_root() -> None:
    assert "SAC" in {
        registry.root.root_id for registry in declared_workstream_registries()
    }


# gabion:behavior primary=desired
def test_declared_workstream_registries_include_dataflow_grammar_readiness_root() -> None:
    assert "DGR" in {
        registry.root.root_id for registry in declared_workstream_registries()
    }


# gabion:behavior primary=desired
def test_declared_workstream_registries_include_wrapper_retirement_drain_root() -> None:
    assert "WRD" in {
        registry.root.root_id for registry in declared_workstream_registries()
    }


# gabion:behavior primary=desired
def test_declared_workstream_registries_include_public_surface_normalization_root() -> None:
    assert "PSN" in {
        registry.root.root_id for registry in declared_workstream_registries()
    }


# gabion:behavior primary=desired
def test_connectivity_synergy_workstream_registries_expose_expected_roots_and_touchsites() -> None:
    registries = connectivity_synergy_workstream_registries()
    by_root = {registry.root.root_id: registry for registry in registries}

    assert tuple(registry.root.root_id for registry in registries) == (
        "CSA-IDR",
        "CSA-IGM",
        "CSA-IVL",
        "CSA-RGC",
    )
    assert by_root["CSA-IDR"].tags == ("identity_rendering",)
    assert by_root["CSA-IGM"].tags == ("ingress_merge",)
    assert by_root["CSA-IVL"].tags == ("impact_velocity",)
    assert by_root["CSA-RGC"].tags == ("registry_convergence",)

    igm_touchpoints = {
        item.touchpoint_id: item for item in by_root["CSA-IGM"].touchpoints
    }
    idr_touchpoints = {
        item.touchpoint_id: item for item in by_root["CSA-IDR"].touchpoints
    }
    rgc_touchpoints = {
        item.touchpoint_id: item for item in by_root["CSA-RGC"].touchpoints
    }
    ivl_touchpoints = {
        item.touchpoint_id: item for item in by_root["CSA-IVL"].touchpoints
    }

    assert by_root["CSA-IDR"].root.subqueue_ids == (
        "CSA-IDR-SQ-001",
        "CSA-IDR-SQ-002",
        "CSA-IDR-SQ-003",
        "CSA-IDR-SQ-004",
    )
    assert by_root["CSA-IGM"].root.subqueue_ids == (
        "CSA-IGM-SQ-001",
        "CSA-IGM-SQ-002",
        "CSA-IGM-SQ-003",
        "CSA-IGM-SQ-004",
    )
    assert by_root["CSA-RGC"].root.subqueue_ids == (
        "CSA-RGC-SQ-001",
        "CSA-RGC-SQ-002",
        "CSA-RGC-SQ-003",
        "CSA-RGC-SQ-004",
        "CSA-RGC-SQ-005",
        "CSA-RGC-SQ-006",
        "CSA-RGC-SQ-007",
    )
    assert (
        next(
            item
            for item in by_root["CSA-IGM"].subqueues
            if item.subqueue_id == "CSA-IGM-SQ-002"
        ).touchpoint_ids
        == ("CSA-IGM-TP-002", "CSA-IGM-TP-005", "CSA-IGM-TP-006")
    )
    assert (
        next(
            item
            for item in by_root["CSA-IGM"].subqueues
            if item.subqueue_id == "CSA-IGM-SQ-004"
        ).touchpoint_ids
        == ("CSA-IGM-TP-004", "CSA-IGM-TP-007")
    )
    assert (
        next(
            item
            for item in by_root["CSA-RGC"].subqueues
            if item.subqueue_id == "CSA-RGC-SQ-002"
        ).touchpoint_ids
        == ("CSA-RGC-TP-002", "CSA-RGC-TP-009")
    )
    assert (
        next(
            item
            for item in by_root["CSA-RGC"].subqueues
            if item.subqueue_id == "CSA-RGC-SQ-003"
        ).touchpoint_ids
        == ("CSA-RGC-TP-003", "CSA-RGC-TP-010")
    )
    assert (
        next(
            item
            for item in by_root["CSA-RGC"].subqueues
            if item.subqueue_id == "CSA-RGC-SQ-006"
        ).touchpoint_ids
        == (
            "CSA-RGC-TP-006",
            "CSA-RGC-TP-007",
            "CSA-RGC-TP-011",
            "CSA-RGC-TP-012",
            "CSA-RGC-TP-015",
        )
    )
    assert (
        next(
            item
            for item in by_root["CSA-RGC"].subqueues
            if item.subqueue_id == "CSA-RGC-SQ-001"
        ).touchpoint_ids
        == ("CSA-RGC-TP-001", "CSA-RGC-TP-014")
    )
    assert igm_touchpoints["CSA-IGM-TP-005"].status_hint == "landed"
    assert igm_touchpoints["CSA-IGM-TP-007"].status_hint == "landed"
    assert rgc_touchpoints["CSA-RGC-TP-009"].status_hint == "landed"
    assert rgc_touchpoints["CSA-RGC-TP-010"].status_hint == "landed"
    assert rgc_touchpoints["CSA-RGC-TP-011"].status_hint == "landed"
    assert rgc_touchpoints["CSA-RGC-TP-014"].status_hint == "landed"
    assert rgc_touchpoints["CSA-RGC-TP-015"].status_hint == "landed"

    assert {
        (item.rel_path, item.qualname)
        for item in idr_touchpoints["CSA-IDR-TP-004"].declared_touchsites
    } >= {
        (
            "scripts/policy/hotspot_neighborhood_queue.py",
            "_file_family_counts",
        ),
        (
            "scripts/policy/hotspot_neighborhood_queue.py",
            "_file_ref",
        ),
        (
            "scripts/policy/hotspot_neighborhood_queue.py",
            "_scope_ref",
        ),
        (
            "src/gabion/tooling/policy_substrate/planning_chart_identity.py",
            "build_planning_chart_identity_grammar",
        ),
        (
            "src/gabion/tooling/policy_substrate/identity_zone/grammar.py",
            "HierarchicalIdentityGrammar.add_two_cell",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in igm_touchpoints["CSA-IGM-TP-004"].declared_touchsites
    } >= {
        (
            "src/gabion/tooling/policy_substrate/invariant_graph.py",
            "_join_test_coverage",
        ),
        (
            "src/gabion/tooling/runtime/perf_artifact.py",
            "build_cprofile_perf_artifact_payload",
        ),
        (
            "src/gabion/tooling/runtime/git_state_artifact.py",
            "build_git_state_artifact_payload",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in igm_touchpoints["CSA-IGM-TP-007"].declared_touchsites
    } >= {
        (
            "src/gabion/tooling/runtime/git_state_artifact.py",
            "collect_git_state_command_outputs",
        ),
        (
            "src/gabion/tooling/runtime/git_state_artifact.py",
            "assemble_git_state_artifact_payload",
        ),
        (
            "tests/gabion/tooling/runtime_policy/test_git_state_artifact.py",
            "test_assemble_git_state_artifact_payload_supports_injected_command_outputs",
        ),
    }
    assert {
        item.rel_path for item in rgc_touchpoints["CSA-RGC-TP-005"].declared_touchsites
    } >= {
        "src/gabion/tooling/policy_substrate/invariant_graph.py",
        "scripts/sppf/sppf_status_audit.py",
    }
    assert {
        (item.rel_path, item.qualname)
        for item in rgc_touchpoints["CSA-RGC-TP-008"].declared_touchsites
    } >= {
        (
            "docs/ttl_kernel_semantics.md",
            "ttl_kernel_semantics",
        ),
        (
            "in/lg_kernel_ontology_cut_elim-1.ttl",
            "lg:AugmentedRule",
        ),
        (
            "src/gabion/analysis/projection/semantic_fragment.py",
            "reflect_projection_fiber_witness",
        ),
        (
            "src/gabion/tooling/runtime/kernel_vm_alignment_artifact.py",
            "build_kernel_vm_alignment_artifact_payload",
        ),
        (
            "scripts/policy/policy_check.py",
            "collect_aspf_lattice_convergence_result",
        ),
        (
            "src/gabion/tooling/policy_substrate/invariant_graph.py",
            "_join_kernel_vm_alignment_artifact",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in igm_touchpoints["CSA-IGM-TP-005"].declared_touchsites
    } >= {
        ("src/gabion/frontmatter.py", "parse_lenient_yaml_frontmatter"),
        ("scripts/governance/docflow_promote_sections.py", "main"),
        ("scripts/audit/audit_in_step_structure.py", "_parse_frontmatter"),
        ("src/gabion/analysis/semantics/impact_index.py", "_parse_frontmatter"),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in igm_touchpoints["CSA-IGM-TP-006"].declared_touchsites
    } >= {
        ("src/gabion/frontmatter.py", "parse_lenient_yaml_frontmatter"),
        (
            "src/gabion/tooling/governance/normative_symdiff.py",
            "collect_scope_inventory",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in rgc_touchpoints["CSA-RGC-TP-009"].declared_touchsites
    } >= {
        ("src/gabion/tooling/governance/normative_symdiff.py", "_capture_policy_check"),
        ("src/gabion/tooling/governance/normative_symdiff.py", "_collect_default_probes"),
        ("src/gabion/tooling/policy_rules/branchless_rule.py", "_load_baseline"),
        ("src/gabion/tooling/policy_rules/defensive_fallback_rule.py", "_load_baseline"),
        ("src/gabion/tooling/policy_rules/no_monkeypatch_rule.py", "main"),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in rgc_touchpoints["CSA-RGC-TP-010"].declared_touchsites
    } >= {
        ("src/gabion/runtime/deadline_policy.py", "deadline_scope_from_ticks"),
        ("src/gabion/tooling/runtime/deadline_runtime.py", "deadline_scope_from_ticks"),
        ("scripts/policy/policy_check.py", "main"),
        ("scripts/sppf/sppf_status_audit.py", "main"),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in rgc_touchpoints["CSA-RGC-TP-011"].declared_touchsites
    } >= {
        ("src/gabion/tooling/runtime/checks_runtime.py", "build_ci_checks_steps"),
        ("src/gabion/cli_support/tooling_commands.py", "tooling_commands#checks"),
        ("scripts/checks.sh", "checks_sh"),
        ("scripts/policy/policy_check.py", "main"),
        ("docs/governance_control_loops.yaml", "governance_control_loops.gates"),
        ("scripts/install_hooks.sh", "install_hooks"),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in rgc_touchpoints["CSA-RGC-TP-012"].declared_touchsites
    } >= {
        ("src/gabion/analysis/foundation/marker_protocol.py", "normalize_marker_payload"),
        ("src/gabion/invariants.py", "landed_todo_decorator"),
        (
            "src/gabion/tooling/policy_substrate/invariant_marker_scan.py",
            "_lifecycle_state",
        ),
        (
            "src/gabion/tooling/policy_substrate/workstream_registry.py",
            "validate_workstream_closure_consistency",
        ),
        ("scripts/policy/policy_check.py", "check_workstream_closure_consistency"),
        (
            "tests/gabion/tooling/runtime_policy/test_workstream_closure_consistency.py",
            "test_workstream_closure_consistency",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in rgc_touchpoints["CSA-RGC-TP-014"].declared_touchsites
    } >= {
        (
            "src/gabion_governance/governance_doc_registry.py",
            "load_governance_docflow_registry",
        ),
        (
            "src/gabion_governance/governance_audit_impl.py",
            "_docflow_invariant_rows",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in rgc_touchpoints["CSA-RGC-TP-015"].declared_touchsites
    } >= {
        (
            "src/gabion/tooling/runtime/policy_scanner_suite.py",
            "_policy_scanner_rule_manifest",
        ),
        (
            "src/gabion/tooling/runtime/policy_scanner_suite.py",
            "scan_policy_suite",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in ivl_touchpoints["CSA-IVL-TP-005"].declared_touchsites
    } >= {
        (
            "src/gabion/tooling/runtime/invariant_graph.py",
            "_resolve_perf_dsl_overlay",
        ),
        (
            "src/gabion/tooling/runtime/perf_artifact.py",
            "build_cprofile_perf_artifact_payload",
        ),
    }
