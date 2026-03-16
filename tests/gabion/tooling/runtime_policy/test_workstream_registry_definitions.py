from __future__ import annotations

from gabion.tooling.policy_substrate.connectivity_synergy_registry import (
    connectivity_synergy_workstream_registries,
)
from gabion.tooling.policy_substrate.policy_rule_frontmatter_migration_registry import (
    prf_workstream_registry,
)
from gabion.tooling.policy_substrate.projection_semantic_fragment_phase5_registry import (
    phase5_workstream_registry,
)
from gabion.tooling.policy_substrate.surface_contract_convergence_registry import (
    surface_contract_convergence_workstream_registry,
)


def test_prf_workstream_registry_exposes_queue_sequence_and_active_playbook_touchpoint() -> None:
    registry = prf_workstream_registry()
    touchpoints = {item.touchpoint_id: item for item in registry.touchpoints}
    subqueues = {item.subqueue_id: item for item in registry.subqueues}

    assert registry.root.root_id == "PRF"
    assert registry.tags == ("registry_convergence",)
    assert registry.root.status_hint == "landed"
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


def test_surface_contract_convergence_workstream_registry_exposes_queue_and_touchsites() -> None:
    registry = surface_contract_convergence_workstream_registry()
    touchpoints = {item.touchpoint_id: item for item in registry.touchpoints}
    subqueues = {item.subqueue_id: item for item in registry.subqueues}

    assert registry.root.root_id == "SCC"
    assert registry.tags == ("contract_convergence",)
    assert registry.root.status_hint == "in_progress"
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
    assert subqueues["SCC-SQ-001"].touchpoint_ids == ("SCC-TP-001", "SCC-TP-002")
    assert subqueues["SCC-SQ-002"].touchpoint_ids == ("SCC-TP-003", "SCC-TP-004")
    assert subqueues["SCC-SQ-003"].touchpoint_ids == ("SCC-TP-005", "SCC-TP-006")
    assert subqueues["SCC-SQ-004"].touchpoint_ids == ("SCC-TP-007",)
    assert all(item.status_hint == "in_progress" for item in registry.subqueues)
    assert set(touchpoints) == {
        "SCC-TP-001",
        "SCC-TP-002",
        "SCC-TP-003",
        "SCC-TP-004",
        "SCC-TP-005",
        "SCC-TP-006",
        "SCC-TP-007",
    }
    assert touchpoints["SCC-TP-001"].status_hint == "landed"
    assert touchpoints["SCC-TP-002"].status_hint == "landed"
    assert touchpoints["SCC-TP-003"].status_hint == "landed"
    assert touchpoints["SCC-TP-004"].status_hint == "landed"
    assert touchpoints["SCC-TP-005"].status_hint == "landed"
    assert touchpoints["SCC-TP-006"].status_hint == "landed"
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
        == ("CSA-IGM-TP-002", "CSA-IGM-TP-005")
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
        == ("CSA-RGC-TP-006", "CSA-RGC-TP-007", "CSA-RGC-TP-011")
    )
    assert igm_touchpoints["CSA-IGM-TP-005"].status_hint == "landed"
    assert rgc_touchpoints["CSA-RGC-TP-009"].status_hint == "landed"
    assert rgc_touchpoints["CSA-RGC-TP-010"].status_hint == "landed"
    assert rgc_touchpoints["CSA-RGC-TP-011"].status_hint == "landed"

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
        ("src/gabion/tooling/governance/normative_symdiff.py", "_parse_frontmatter"),
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
