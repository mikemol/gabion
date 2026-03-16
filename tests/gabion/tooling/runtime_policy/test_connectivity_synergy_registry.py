from __future__ import annotations

from gabion.tooling.policy_substrate import invariant_graph
from gabion.tooling.policy_substrate.connectivity_synergy_registry import (
    connectivity_synergy_workstream_registries,
)
from tests.path_helpers import REPO_ROOT


def _work_item_node(graph: invariant_graph.InvariantGraph, object_id: str):
    return next(
        node
        for node in invariant_graph.trace_nodes(graph, object_id)
        if node.node_kind == "synthetic_work_item"
    )


def test_connectivity_synergy_registry_defines_expected_roots_and_subqueues() -> None:
    registries = connectivity_synergy_workstream_registries()
    by_root = {registry.root.root_id: registry for registry in registries}
    igm_subqueues = {
        subqueue.subqueue_id: subqueue for subqueue in by_root["CSA-IGM"].subqueues
    }
    rgc_subqueues = {
        subqueue.subqueue_id: subqueue for subqueue in by_root["CSA-RGC"].subqueues
    }
    ivl_subqueues = {
        subqueue.subqueue_id: subqueue for subqueue in by_root["CSA-IVL"].subqueues
    }
    idr_touchpoints = {
        touchpoint.touchpoint_id: touchpoint
        for touchpoint in by_root["CSA-IDR"].touchpoints
    }
    igm_touchpoints = {
        touchpoint.touchpoint_id: touchpoint
        for touchpoint in by_root["CSA-IGM"].touchpoints
    }
    ivl_touchpoints = {
        touchpoint.touchpoint_id: touchpoint
        for touchpoint in by_root["CSA-IVL"].touchpoints
    }
    rgc_touchpoints = {
        touchpoint.touchpoint_id: touchpoint
        for touchpoint in by_root["CSA-RGC"].touchpoints
    }
    perf_touchpoints = {
        touchpoint.touchpoint_id: touchpoint
        for touchpoint in by_root["CSA-IVL"].touchpoints
    }

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
    assert by_root["CSA-IVL"].root.subqueue_ids == (
        "CSA-IVL-SQ-001",
        "CSA-IVL-SQ-002",
        "CSA-IVL-SQ-003",
        "CSA-IVL-SQ-004",
        "CSA-IVL-SQ-005",
    )
    assert set(
        by_root["CSA-IDR"].subqueues[3].marker_payload.reasoning.blocking_dependencies
    ) == {"CSA-IDR-SQ-001", "CSA-IDR-TP-004"}
    assert set(
        idr_touchpoints["CSA-IDR-TP-004"].marker_payload.reasoning.blocking_dependencies
    ) == {"CSA-IDR-SQ-001", "CSA-IDR-SQ-004"}
    assert set(
        by_root["CSA-IDR"].subqueues[1].marker_payload.reasoning.blocking_dependencies
    ) == {"CSA-IDR-TP-002", "PSF-007"}
    assert set(
        igm_subqueues["CSA-IGM-SQ-002"].marker_payload.reasoning.blocking_dependencies
    ) == {"CSA-IGM-SQ-001", "CSA-IGM-TP-002"}
    assert igm_subqueues["CSA-IGM-SQ-002"].touchpoint_ids == (
        "CSA-IGM-TP-002",
        "CSA-IGM-TP-005",
    )
    assert set(
        igm_subqueues["CSA-IGM-SQ-004"].marker_payload.reasoning.blocking_dependencies
    ) == {"CSA-IGM-SQ-001", "CSA-IGM-TP-004"}
    assert set(
        igm_subqueues["CSA-IGM-SQ-003"].marker_payload.reasoning.blocking_dependencies
    ) == {"CSA-IGM-SQ-002", "CSA-IGM-TP-003"}
    assert set(
        rgc_subqueues["CSA-RGC-SQ-001"].marker_payload.reasoning.blocking_dependencies
    ) == {"CSA-IGM-SQ-002", "CSA-RGC-TP-001"}
    assert set(
        rgc_subqueues["CSA-RGC-SQ-003"].marker_payload.reasoning.blocking_dependencies
    ) == {"CSA-RGC-SQ-002", "CSA-RGC-TP-003"}
    assert rgc_subqueues["CSA-RGC-SQ-002"].touchpoint_ids == (
        "CSA-RGC-TP-002",
        "CSA-RGC-TP-009",
    )
    assert rgc_subqueues["CSA-RGC-SQ-003"].touchpoint_ids == (
        "CSA-RGC-TP-003",
        "CSA-RGC-TP-010",
    )
    assert set(
        rgc_subqueues["CSA-RGC-SQ-004"].marker_payload.reasoning.blocking_dependencies
    ) == {"CSA-IGM-SQ-002", "CSA-RGC-TP-004"}
    assert set(
        rgc_subqueues["CSA-RGC-SQ-005"].marker_payload.reasoning.blocking_dependencies
    ) == {"CSA-RGC-SQ-004", "CSA-RGC-TP-005"}
    assert set(
        rgc_subqueues["CSA-RGC-SQ-006"].marker_payload.reasoning.blocking_dependencies
    ) == {
        "CSA-IGM-SQ-004",
        "CSA-RGC-SQ-004",
        "CSA-RGC-SQ-005",
        "CSA-RGC-TP-006",
        "CSA-RGC-TP-007",
    }
    assert rgc_subqueues["CSA-RGC-SQ-006"].touchpoint_ids == (
        "CSA-RGC-TP-006",
        "CSA-RGC-TP-007",
        "CSA-RGC-TP-011",
    )
    assert set(
        rgc_subqueues["CSA-RGC-SQ-007"].marker_payload.reasoning.blocking_dependencies
    ) == {"CSA-RGC-SQ-004", "CSA-RGC-TP-008"}
    assert set(
        rgc_touchpoints["CSA-RGC-TP-007"].marker_payload.reasoning.blocking_dependencies
    ) == {
        "CSA-IGM-SQ-004",
        "CSA-IVL-SQ-004",
        "CSA-RGC-SQ-005",
        "CSA-RGC-SQ-006",
    }
    assert set(
        rgc_touchpoints["CSA-RGC-TP-008"].marker_payload.reasoning.blocking_dependencies
    ) == {"CSA-IVL-SQ-001", "CSA-RGC-SQ-004", "CSA-RGC-SQ-007"}
    assert set(
        ivl_subqueues["CSA-IVL-SQ-002"].marker_payload.reasoning.blocking_dependencies
    ) == {"CSA-IVL-SQ-001", "CSA-IVL-TP-002"}
    assert set(
        ivl_subqueues["CSA-IVL-SQ-003"].marker_payload.reasoning.blocking_dependencies
    ) == {"CSA-IVL-SQ-001", "CSA-IVL-SQ-002", "CSA-IVL-TP-003"}
    assert set(
        ivl_subqueues["CSA-IVL-SQ-004"].marker_payload.reasoning.blocking_dependencies
    ) == {"CSA-IVL-SQ-002", "CSA-IVL-TP-004"}
    assert set(
        ivl_subqueues["CSA-IVL-SQ-005"].marker_payload.reasoning.blocking_dependencies
    ) == {"CSA-IGM-SQ-001", "CSA-IVL-SQ-001", "CSA-IVL-TP-005", "CSA-RGC-SQ-004"}
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
        item.rel_path for item in igm_touchpoints["CSA-IGM-TP-001"].declared_touchsites
    } >= {
        "src/gabion/server.py",
        "src/gabion/tooling/policy_substrate/aspf_union_view.py",
        "src/gabion/tooling/policy_substrate/overlap_eval.py",
        "src/gabion/tooling/runtime/cross_origin_witness_artifact.py",
        "src/gabion/tooling/policy_substrate/structured_artifact_ingress.py",
        "scripts/policy/policy_check.py",
        "src/gabion/tooling/policy_substrate/invariant_graph.py",
    }
    assert {
        (item.rel_path, item.qualname)
        for item in igm_touchpoints["CSA-IGM-TP-001"].declared_touchsites
    } >= {
        (
            "src/gabion/server.py",
            "_analysis_manifest_digest_from_witness",
        ),
        (
            "src/gabion/server.py",
            "_analysis_input_witness",
        ),
        (
            "src/gabion/tooling/policy_substrate/aspf_union_view.py",
            "build_aspf_union_view",
        ),
        (
            "src/gabion/tooling/policy_substrate/overlap_eval.py",
            "evaluate_condition_overlaps",
        ),
        (
            "src/gabion/tooling/runtime/cross_origin_witness_artifact.py",
            "build_cross_origin_witness_contract_artifact_payload",
        ),
        (
            "src/gabion/tooling/policy_substrate/structured_artifact_ingress.py",
            "load_cross_origin_witness_contract_artifact",
        ),
        (
            "scripts/policy/policy_check.py",
            "_write_cross_origin_witness_contract_artifact",
        ),
        (
            "src/gabion/tooling/policy_substrate/invariant_graph.py",
            "_join_cross_origin_witness_contract_artifact",
        ),
    }
    assert {
        item.rel_path for item in igm_touchpoints["CSA-IGM-TP-002"].declared_touchsites
    } >= {
        "src/gabion/frontmatter_ingress.py",
        "src/gabion/frontmatter.py",
        "src/gabion_governance/governance_audit_impl.py",
    }
    assert {
        (item.rel_path, item.qualname)
        for item in igm_touchpoints["CSA-IGM-TP-002"].declared_touchsites
    } >= {
        (
            "src/gabion/frontmatter_ingress.py",
            "parse_frontmatter_document",
        ),
        (
            "src/gabion/frontmatter.py",
            "parse_strict_yaml_frontmatter",
        ),
        (
            "src/gabion_governance/governance_audit_impl.py",
            "_parse_frontmatter_with_mode",
        ),
    }
    assert igm_touchpoints["CSA-IGM-TP-005"].status_hint == "landed"
    assert {
        (item.rel_path, item.qualname)
        for item in igm_touchpoints["CSA-IGM-TP-005"].declared_touchsites
    } >= {
        (
            "src/gabion/frontmatter.py",
            "parse_lenient_yaml_frontmatter",
        ),
        (
            "scripts/governance/docflow_promote_sections.py",
            "main",
        ),
        (
            "scripts/audit/audit_in_step_structure.py",
            "_parse_frontmatter",
        ),
        (
            "src/gabion/analysis/semantics/impact_index.py",
            "_parse_frontmatter",
        ),
        (
            "src/gabion/tooling/governance/normative_symdiff.py",
            "_parse_frontmatter",
        ),
    }
    assert {
        item.rel_path for item in igm_touchpoints["CSA-IGM-TP-003"].declared_touchsites
    } >= {
        "src/gabion/tooling/policy_substrate/structured_artifact_ingress.py",
        "scripts/policy/policy_check.py",
        "src/gabion/tooling/policy_substrate/invariant_graph.py",
        "tests/gabion/ingest/test_adapter_contract.py",
        "tests/test_policy_dsl.py",
    }
    assert {
        (item.rel_path, item.qualname)
        for item in igm_touchpoints["CSA-IGM-TP-003"].declared_touchsites
    } >= {
        (
            "src/gabion/tooling/policy_substrate/structured_artifact_ingress.py",
            "build_ingress_merge_parity_artifact",
        ),
        (
            "src/gabion/tooling/policy_substrate/structured_artifact_ingress.py",
            "load_ingress_merge_parity_artifact",
        ),
        (
            "scripts/policy/policy_check.py",
            "_write_ingress_merge_parity_artifact",
        ),
        (
            "src/gabion/tooling/policy_substrate/invariant_graph.py",
            "_join_ingress_merge_parity_artifact",
        ),
        (
            "tests/gabion/ingest/test_adapter_contract.py",
            "test_adapter_parity_on_overlapping_decision_surfaces",
        ),
        (
            "tests/test_policy_dsl.py",
            "test_registry_rejects_duplicate_rule_ids_across_yaml_and_markdown",
        ),
    }
    assert {
        item.rel_path for item in igm_touchpoints["CSA-IGM-TP-004"].declared_touchsites
    } >= {
        "src/gabion/tooling/policy_substrate/invariant_graph.py",
        "src/gabion/tooling/runtime/perf_artifact.py",
        "src/gabion/tooling/runtime/git_state_artifact.py",
        "scripts/ci/ci_observability_guard.py",
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
            "src/gabion/tooling/policy_substrate/invariant_graph.py",
            "_join_test_failures",
        ),
        (
            "src/gabion/tooling/runtime/perf_artifact.py",
            "build_cprofile_perf_artifact_payload",
        ),
        (
            "src/gabion/tooling/runtime/git_state_artifact.py",
            "build_git_state_artifact_payload",
        ),
        (
            "src/gabion/tooling/policy_substrate/structured_artifact_ingress.py",
            "load_git_state_artifact",
        ),
        (
            "scripts/ci/ci_observability_guard.py",
            "_parse_deadline_profile",
        ),
    }
    assert {
        item.rel_path for item in ivl_touchpoints["CSA-IVL-TP-001"].declared_touchsites
    } >= {
        "scripts/policy/policy_check.py",
        "src/gabion/tooling/policy_substrate/lattice_convergence_semantic.py",
        "src/gabion/analysis/aspf/aspf_lattice_algebra.py",
        "src/gabion/tooling/policy_substrate/invariant_graph.py",
    }
    assert {
        item.rel_path for item in perf_touchpoints["CSA-IVL-TP-005"].declared_touchsites
    } >= {
        "src/gabion/tooling/runtime/invariant_graph.py",
        "src/gabion/tooling/runtime/perf_artifact.py",
        "scripts/policy/policy_check.py",
    }
    assert {
        (item.rel_path, item.qualname)
        for item in perf_touchpoints["CSA-IVL-TP-005"].declared_touchsites
    } >= {
        (
            "src/gabion/tooling/runtime/invariant_graph.py",
            "_resolve_perf_dsl_overlay",
        ),
    }
    assert {
        item.rel_path for item in rgc_touchpoints["CSA-RGC-TP-004"].declared_touchsites
    } >= {
        "src/gabion/analysis/semantics/impact_index.py",
        "src/gabion/tooling/runtime/invariant_graph.py",
    }
    assert {
        (item.rel_path, item.qualname)
        for item in rgc_touchpoints["CSA-RGC-TP-004"].declared_touchsites
    } >= {
        (
            "src/gabion/analysis/semantics/impact_index.py",
            "build_impact_index",
        ),
        (
            "src/gabion/tooling/runtime/invariant_graph.py",
            "_resolve_perf_dsl_overlay",
        ),
    }
    assert {
        item.rel_path for item in rgc_touchpoints["CSA-RGC-TP-005"].declared_touchsites
    } >= {
        "src/gabion/tooling/policy_substrate/invariant_graph.py",
        "scripts/sppf/sppf_status_audit.py",
    }
    assert rgc_touchpoints["CSA-RGC-TP-009"].status_hint == "landed"
    assert rgc_touchpoints["CSA-RGC-TP-010"].status_hint == "landed"
    assert rgc_touchpoints["CSA-RGC-TP-011"].status_hint == "landed"
    assert {
        (item.rel_path, item.qualname)
        for item in rgc_touchpoints["CSA-RGC-TP-009"].declared_touchsites
    } >= {
        (
            "src/gabion/tooling/governance/normative_symdiff.py",
            "_capture_policy_check",
        ),
        (
            "src/gabion/tooling/governance/normative_symdiff.py",
            "_collect_default_probes",
        ),
        (
            "src/gabion/tooling/policy_rules/branchless_rule.py",
            "_load_baseline",
        ),
        (
            "src/gabion/tooling/policy_rules/defensive_fallback_rule.py",
            "_load_baseline",
        ),
        (
            "src/gabion/tooling/policy_rules/no_monkeypatch_rule.py",
            "main",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in rgc_touchpoints["CSA-RGC-TP-010"].declared_touchsites
    } >= {
        (
            "src/gabion/runtime/deadline_policy.py",
            "deadline_scope_from_ticks",
        ),
        (
            "src/gabion/tooling/runtime/deadline_runtime.py",
            "deadline_scope_from_ticks",
        ),
        (
            "scripts/policy/policy_check.py",
            "main",
        ),
        (
            "scripts/sppf/sppf_status_audit.py",
            "main",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in rgc_touchpoints["CSA-RGC-TP-011"].declared_touchsites
    } >= {
        (
            "src/gabion/tooling/runtime/checks_runtime.py",
            "build_ci_checks_steps",
        ),
        (
            "src/gabion/cli_support/tooling_commands.py",
            "tooling_commands#checks",
        ),
        (
            "scripts/checks.sh",
            "checks_sh",
        ),
        (
            "scripts/policy/policy_check.py",
            "main",
        ),
        (
            "docs/governance_control_loops.yaml",
            "governance_control_loops.gates",
        ),
        (
            "scripts/install_hooks.sh",
            "install_hooks",
        ),
    }
    assert {
        (item.rel_path, item.qualname)
        for item in rgc_touchpoints["CSA-RGC-TP-005"].declared_touchsites
    } >= {
        (
            "src/gabion/tooling/policy_substrate/invariant_graph.py",
            "_join_governance_convergence_sources",
        ),
        (
            "src/gabion/tooling/policy_substrate/invariant_graph.py",
            "_join_sppf_dependency_graph",
        ),
        (
            "src/gabion/tooling/policy_substrate/invariant_graph.py",
            "_join_inbox_governance_docs",
        ),
        (
            "scripts/sppf/sppf_status_audit.py",
            "run_audit",
        ),
    }
    assert {
        item.rel_path for item in rgc_touchpoints["CSA-RGC-TP-006"].declared_touchsites
    } >= {
        "scripts/policy/docflow_packet_enforce.py",
        "scripts/governance/governance_controller_audit.py",
        "src/gabion/tooling/runtime/ci_local_repro.py",
        "src/gabion_governance/governance_audit_impl.py",
        "src/gabion/tooling/runtime/ci_watch.py",
        "scripts/policy/policy_scanner_suite.py",
        "scripts/policy/symbol_activity_audit.py",
        "scripts/policy/policy_check.py",
        "src/gabion/plan.py",
        "src/gabion/tooling/policy_substrate/invariant_graph.py",
        "src/gabion/tooling/policy_substrate/structured_artifact_ingress.py",
    }
    assert {
        (item.rel_path, item.qualname)
        for item in rgc_touchpoints["CSA-RGC-TP-006"].declared_touchsites
    } >= {
        (
            "scripts/policy/docflow_packet_enforce.py",
            "main",
        ),
        (
            "scripts/governance/governance_controller_audit.py",
            "main",
        ),
        (
            "src/gabion/tooling/runtime/ci_local_repro.py",
            "main",
        ),
        (
            "src/gabion/tooling/runtime/ci_watch.py",
            "run_watch",
        ),
        (
            "scripts/policy/policy_scanner_suite.py",
            "main",
        ),
        (
            "scripts/policy/symbol_activity_audit.py",
            "main",
        ),
        (
            "scripts/policy/policy_check.py",
            "_write_invariant_graph_artifact",
        ),
        (
            "scripts/policy/policy_check.py",
            "_write_git_state_artifact",
        ),
        (
            "src/gabion/plan.py",
            "write_execution_plan_artifact",
        ),
        (
            "src/gabion/tooling/policy_substrate/invariant_graph.py",
            "_join_git_state_artifact",
        ),
        (
            "src/gabion_governance/governance_audit_impl.py",
            "_emit_docflow_compliance",
        ),
        (
            "src/gabion/tooling/policy_substrate/structured_artifact_ingress.py",
            "load_docflow_compliance_artifact",
        ),
        (
            "src/gabion/tooling/policy_substrate/invariant_graph.py",
            "_join_docflow_compliance_artifact",
        ),
        (
            "scripts/policy/policy_check.py",
            "_write_local_ci_repro_contract_artifact",
        ),
        (
            "src/gabion/tooling/policy_substrate/structured_artifact_ingress.py",
            "load_local_ci_repro_contract_artifact",
        ),
        (
            "src/gabion/tooling/policy_substrate/invariant_graph.py",
            "_join_local_ci_repro_contract_artifact",
        ),
    }
    assert {
        item.rel_path for item in rgc_touchpoints["CSA-RGC-TP-007"].declared_touchsites
    } >= {
        "src/gabion_governance/governance_audit_impl.py",
        "src/gabion/tooling/sppf/sync_core.py",
        "src/gabion/analysis/semantics/obligation_registry.py",
        "src/gabion/execution_plan.py",
        "src/gabion/tooling/policy_substrate/invariant_graph.py",
    }
    assert {
        (item.rel_path, item.qualname)
        for item in rgc_touchpoints["CSA-RGC-TP-007"].declared_touchsites
    } >= {
        (
            "src/gabion_governance/governance_audit_impl.py",
            "_sppf_sync_check",
        ),
        (
            "src/gabion_governance/governance_audit_impl.py",
            "_evaluate_docflow_obligations",
        ),
        (
            "src/gabion/tooling/sppf/sync_core.py",
            "_collect_commits",
        ),
        (
            "src/gabion/tooling/sppf/sync_core.py",
            "_issue_ids_from_commits",
        ),
        (
            "src/gabion/tooling/sppf/sync_core.py",
            "_build_issue_link_facet",
        ),
        (
            "src/gabion/tooling/sppf/sync_core.py",
            "_fetch_issue",
        ),
        (
            "src/gabion/tooling/sppf/sync_core.py",
            "_validate_issue_lifecycle",
        ),
        (
            "src/gabion/tooling/sppf/sync_core.py",
            "_run_validate_mode",
        ),
        (
            "src/gabion/analysis/semantics/obligation_registry.py",
            "evaluate_obligations",
        ),
        (
            "src/gabion/execution_plan.py",
            "ExecutionPlan.with_issue_link",
        ),
        (
            "src/gabion/tooling/policy_substrate/invariant_graph.py",
            "_join_docflow_provenance_artifact",
        ),
    }
    assert {
        item.rel_path for item in rgc_touchpoints["CSA-RGC-TP-008"].declared_touchsites
    } >= {
        "docs/ttl_kernel_semantics.md",
        "in/lg_kernel_ontology_cut_elim-1.ttl",
        "src/gabion/analysis/aspf/aspf_lattice_algebra.py",
        "src/gabion/analysis/projection/semantic_fragment.py",
        "src/gabion/analysis/projection/projection_semantic_lowering.py",
        "src/gabion/analysis/projection/semantic_fragment_compile.py",
        "src/gabion/tooling/runtime/kernel_vm_alignment_artifact.py",
        "src/gabion/tooling/policy_substrate/lattice_convergence_semantic.py",
        "src/gabion/tooling/policy_substrate/invariant_graph.py",
        "scripts/policy/policy_check.py",
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
            "in/lg_kernel_ontology_cut_elim-1.ttl",
            "lg:RulePolarity",
        ),
        (
            "in/lg_kernel_ontology_cut_elim-1.ttl",
            "lg:ClosedRuleCell",
        ),
        (
            "src/gabion/analysis/aspf/aspf_lattice_algebra.py",
            "NaturalityWitness",
        ),
        (
            "src/gabion/analysis/aspf/aspf_lattice_algebra.py",
            "FrontierWitness",
        ),
        (
            "src/gabion/analysis/projection/semantic_fragment.py",
            "SemanticOpKind",
        ),
        (
            "src/gabion/analysis/projection/semantic_fragment.py",
            "CanonicalWitnessedSemanticRow",
        ),
        (
            "src/gabion/analysis/projection/semantic_fragment.py",
            "reflect_projection_fiber_witness",
        ),
        (
            "src/gabion/analysis/projection/projection_semantic_lowering.py",
            "ProjectionSemanticLoweringPlan",
        ),
        (
            "src/gabion/analysis/projection/semantic_fragment_compile.py",
            "CompiledShaclPlan",
        ),
        (
            "src/gabion/analysis/projection/semantic_fragment_compile.py",
            "CompiledSparqlPlan",
        ),
        (
            "src/gabion/tooling/policy_substrate/lattice_convergence_semantic.py",
            "materialize_semantic_lattice_convergence",
        ),
        (
            "scripts/policy/policy_check.py",
            "collect_aspf_lattice_convergence_result",
        ),
        (
            "src/gabion/tooling/runtime/kernel_vm_alignment_artifact.py",
            "build_kernel_vm_alignment_artifact_payload",
        ),
        (
            "src/gabion/tooling/policy_substrate/invariant_graph.py",
            "_join_kernel_vm_alignment_artifact",
        ),
    }


def test_connectivity_synergy_graph_exposes_cross_root_dependencies_and_mixed_root_lane() -> None:
    graph = invariant_graph.build_invariant_graph(REPO_ROOT)
    node_by_id = graph.node_by_id()
    edges_from = graph.edges_from()

    assert {
        "CSA-IDR",
        "CSA-IGM",
        "CSA-IVL",
        "CSA-RGC",
        "PRF",
        "PSF-007",
    }.issubset(set(graph.workstream_root_ids))

    csa_rgc_sq3 = _work_item_node(graph, "CSA-RGC-SQ-003")
    assert any(
        node_by_id[edge.target_id].matches_raw_id("CSA-RGC-SQ-002")
        for edge in edges_from.get(csa_rgc_sq3.node_id, ())
        if edge.edge_kind == "depends_on"
    )

    csa_rgc_sq4 = _work_item_node(graph, "CSA-RGC-SQ-004")
    assert any(
        node_by_id[edge.target_id].matches_raw_id("CSA-IGM-SQ-002")
        for edge in edges_from.get(csa_rgc_sq4.node_id, ())
        if edge.edge_kind == "depends_on"
    )

    csa_rgc_sq5 = _work_item_node(graph, "CSA-RGC-SQ-005")
    assert any(
        node_by_id[edge.target_id].matches_raw_id("CSA-RGC-SQ-004")
        for edge in edges_from.get(csa_rgc_sq5.node_id, ())
        if edge.edge_kind == "depends_on"
    )

    csa_rgc_sq6 = _work_item_node(graph, "CSA-RGC-SQ-006")
    assert any(
        node_by_id[edge.target_id].matches_raw_id("CSA-IGM-SQ-004")
        for edge in edges_from.get(csa_rgc_sq6.node_id, ())
        if edge.edge_kind == "depends_on"
    )
    assert any(
        node_by_id[edge.target_id].matches_raw_id("CSA-RGC-SQ-004")
        for edge in edges_from.get(csa_rgc_sq6.node_id, ())
        if edge.edge_kind == "depends_on"
    )
    assert any(
        node_by_id[edge.target_id].matches_raw_id("CSA-RGC-SQ-005")
        for edge in edges_from.get(csa_rgc_sq6.node_id, ())
        if edge.edge_kind == "depends_on"
    )
    assert any(
        node_by_id[edge.target_id].matches_raw_id("CSA-RGC-TP-007")
        for edge in edges_from.get(csa_rgc_sq6.node_id, ())
        if edge.edge_kind == "depends_on"
    )

    csa_rgc_tp7 = _work_item_node(graph, "CSA-RGC-TP-007")
    assert any(
        node_by_id[edge.target_id].matches_raw_id("CSA-IGM-SQ-004")
        for edge in edges_from.get(csa_rgc_tp7.node_id, ())
        if edge.edge_kind == "depends_on"
    )
    assert any(
        node_by_id[edge.target_id].matches_raw_id("CSA-IVL-SQ-004")
        for edge in edges_from.get(csa_rgc_tp7.node_id, ())
        if edge.edge_kind == "depends_on"
    )
    assert any(
        node_by_id[edge.target_id].matches_raw_id("CSA-RGC-SQ-005")
        for edge in edges_from.get(csa_rgc_tp7.node_id, ())
        if edge.edge_kind == "depends_on"
    )
    assert any(
        node_by_id[edge.target_id].matches_raw_id("CSA-RGC-SQ-006")
        for edge in edges_from.get(csa_rgc_tp7.node_id, ())
        if edge.edge_kind == "depends_on"
    )
    csa_rgc_sq7 = _work_item_node(graph, "CSA-RGC-SQ-007")
    assert any(
        node_by_id[edge.target_id].matches_raw_id("CSA-RGC-SQ-004")
        for edge in edges_from.get(csa_rgc_sq7.node_id, ())
        if edge.edge_kind == "depends_on"
    )
    assert any(
        node_by_id[edge.target_id].matches_raw_id("CSA-RGC-TP-008")
        for edge in edges_from.get(csa_rgc_sq7.node_id, ())
        if edge.edge_kind == "depends_on"
    )

    csa_rgc_tp8 = _work_item_node(graph, "CSA-RGC-TP-008")
    assert any(
        node_by_id[edge.target_id].matches_raw_id("CSA-IVL-SQ-001")
        for edge in edges_from.get(csa_rgc_tp8.node_id, ())
        if edge.edge_kind == "depends_on"
    )
    assert any(
        node_by_id[edge.target_id].matches_raw_id("CSA-RGC-SQ-004")
        for edge in edges_from.get(csa_rgc_tp8.node_id, ())
        if edge.edge_kind == "depends_on"
    )
    assert any(
        node_by_id[edge.target_id].matches_raw_id("CSA-RGC-SQ-007")
        for edge in edges_from.get(csa_rgc_tp8.node_id, ())
        if edge.edge_kind == "depends_on"
    )

    csa_idr_sq2 = _work_item_node(graph, "CSA-IDR-SQ-002")
    assert any(
        node_by_id[edge.target_id].matches_raw_id("PSF-007")
        for edge in edges_from.get(csa_idr_sq2.node_id, ())
        if edge.edge_kind == "depends_on"
    )

    csa_ivl_sq3 = _work_item_node(graph, "CSA-IVL-SQ-003")
    assert any(
        node_by_id[edge.target_id].matches_raw_id("CSA-IVL-SQ-002")
        for edge in edges_from.get(csa_ivl_sq3.node_id, ())
        if edge.edge_kind == "depends_on"
    )

    csa_ivl_sq4 = _work_item_node(graph, "CSA-IVL-SQ-004")
    assert any(
        node_by_id[edge.target_id].matches_raw_id("CSA-IVL-SQ-002")
        for edge in edges_from.get(csa_ivl_sq4.node_id, ())
        if edge.edge_kind == "depends_on"
    )

    csa_ivl_sq5 = _work_item_node(graph, "CSA-IVL-SQ-005")
    assert any(
        node_by_id[edge.target_id].matches_raw_id("CSA-IVL-SQ-001")
        for edge in edges_from.get(csa_ivl_sq5.node_id, ())
        if edge.edge_kind == "depends_on"
    )
    assert any(
        node_by_id[edge.target_id].matches_raw_id("CSA-RGC-SQ-004")
        for edge in edges_from.get(csa_ivl_sq5.node_id, ())
        if edge.edge_kind == "depends_on"
    )
    csa_igm_sq4 = _work_item_node(graph, "CSA-IGM-SQ-004")
    assert any(
        node_by_id[edge.target_id].matches_raw_id("CSA-IGM-SQ-001")
        for edge in edges_from.get(csa_igm_sq4.node_id, ())
        if edge.edge_kind == "depends_on"
    )
    assert _work_item_node(graph, "CSA-IGM-SQ-004").doc_ids == (
        "connectivity_synergy_audit",
    )
    assert _work_item_node(graph, "CSA-IGM-TP-004").doc_ids == (
        "connectivity_synergy_audit",
    )
    assert _work_item_node(graph, "CSA-IVL").doc_ids == ("connectivity_synergy_audit",)
    assert _work_item_node(graph, "CSA-RGC-SQ-004").doc_ids == (
        "connectivity_synergy_audit",
    )
    assert _work_item_node(graph, "CSA-RGC-SQ-006").doc_ids == (
        "connectivity_synergy_audit",
    )
    assert set(_work_item_node(graph, "CSA-RGC-SQ-007").doc_ids) == {
        "connectivity_synergy_audit",
        "ttl_kernel_semantics",
    }
    assert _work_item_node(graph, "CSA-IVL-TP-001").doc_ids == (
        "connectivity_synergy_audit",
    )
    assert _work_item_node(graph, "CSA-IVL-TP-005").doc_ids == (
        "connectivity_synergy_audit",
    )
    assert _work_item_node(graph, "CSA-RGC-TP-004").doc_ids == (
        "connectivity_synergy_audit",
    )
    assert _work_item_node(graph, "CSA-RGC-TP-006").doc_ids == (
        "connectivity_synergy_audit",
    )
    assert set(_work_item_node(graph, "CSA-RGC-TP-008").doc_ids) == {
        "connectivity_synergy_audit",
        "ttl_kernel_semantics",
    }
    assert set(_work_item_node(graph, "CSA-RGC-TP-007").doc_ids) == {
        "connectivity_synergy_audit",
        "influence_index",
        "sppf_checklist",
    }
    assert set(_work_item_node(graph, "CSA-RGC-SQ-005").doc_ids) == {
        "connectivity_synergy_audit",
        "influence_index",
        "sppf_checklist",
    }
    assert set(_work_item_node(graph, "CSA-RGC-TP-005").doc_ids) == {
        "connectivity_synergy_audit",
        "influence_index",
        "sppf_checklist",
    }

    workstreams = invariant_graph.build_invariant_workstreams(graph, root=REPO_ROOT)
    recommended_code_followup = workstreams.recommended_repo_code_followup()
    assert recommended_code_followup is not None
    assert recommended_code_followup.followup_family in {
        "coverage_gap",
        "structural_cut",
    }
    assert recommended_code_followup.selection_scope_kind in {
        "singleton",
        "mixed_root_followup_family",
    }
    if recommended_code_followup.selection_scope_kind == "mixed_root_followup_family":
        scope_roots = set(
            recommended_code_followup.selection_scope_id.split(":", 1)[1].split(",")
        )
        assert recommended_code_followup.selection_scope_id.startswith("coverage_gap:")
        assert {
            item.owner_root_object_id
            for item in recommended_code_followup.cofrontier_followup_cohort
        } == scope_roots
        assert scope_roots.issuperset({"CSA-IDR", "CSA-IGM", "CSA-RGC", "PSF-007"})
    else:
        assert recommended_code_followup.owner_root_object_id in {
            "CSA-IDR",
            "CSA-IGM",
            "CSA-IVL",
            "CSA-RGC",
            "PSF-007",
        }

    recommended_code_lane = workstreams.recommended_repo_code_followup_lane()
    assert recommended_code_lane is not None
    assert set(recommended_code_lane.root_object_ids).issubset(
        {"CSA-IDR", "CSA-IGM", "CSA-IVL", "CSA-RGC", "PSF-007"}
    )
    assert recommended_code_lane.root_object_ids
