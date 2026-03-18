from __future__ import annotations

import pytest

from gabion.tooling.policy_substrate import invariant_graph
from tests.path_helpers import REPO_ROOT


pytestmark = pytest.mark.live_repo_signal


def _work_item_node(graph: invariant_graph.InvariantGraph, object_id: str):
    return next(
        node
        for node in invariant_graph.trace_nodes(graph, object_id)
        if node.node_kind == "synthetic_work_item"
    )


# gabion:behavior primary=desired
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
        "RCI",
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
        "diagnostic_blocked",
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
        assert recommended_code_followup.selection_scope_id.startswith(
            recommended_code_followup.followup_family + ":"
        )
        assert {
            item.owner_root_object_id
            for item in recommended_code_followup.cofrontier_followup_cohort
        } == scope_roots
        assert scope_roots.issubset(
            {"BIC", "CSA-IDR", "CSA-IGM", "CSA-RGC", "PSF-007", "RCI", "UTR"}
        )
    else:
        assert recommended_code_followup.owner_root_object_id in {
            "BIC",
            "CSA-IDR",
            "CSA-IGM",
            "CSA-IVL",
            "CSA-RGC",
            "PSF-007",
            "RCI",
            "UTR",
        }

    recommended_code_lane = workstreams.recommended_repo_code_followup_lane()
    assert recommended_code_lane is not None
    assert set(recommended_code_lane.root_object_ids).issubset(
        {
            "BIC",
            "CSA-IDR",
            "CSA-IGM",
            "CSA-IVL",
            "CSA-RGC",
            "PRF",
            "PSF-007",
            "RCI",
            "SCC",
            "UTR",
        }
    )
    assert recommended_code_lane.root_object_ids
