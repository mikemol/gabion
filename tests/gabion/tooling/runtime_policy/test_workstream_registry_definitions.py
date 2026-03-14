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


def test_prf_workstream_registry_exposes_landed_root_and_subqueues() -> None:
    registry = prf_workstream_registry()

    assert registry.root.root_id == "PRF"
    assert registry.tags == ("registry_convergence",)
    assert registry.root.subqueue_ids == ("PRF-001", "PRF-002", "PRF-003", "PRF-004")
    assert tuple(item.subqueue_id for item in registry.subqueues) == (
        "PRF-001",
        "PRF-002",
        "PRF-003",
        "PRF-004",
    )
    assert all(item.touchpoint_ids == () for item in registry.subqueues)


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
    rgc_touchpoints = {
        item.touchpoint_id: item for item in by_root["CSA-RGC"].touchpoints
    }
    ivl_touchpoints = {
        item.touchpoint_id: item for item in by_root["CSA-IVL"].touchpoints
    }

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
