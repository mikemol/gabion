from __future__ import annotations

from pathlib import Path

import pytest

from tests.path_helpers import REPO_ROOT

from gabion.invariants import deprecated_decorator
from gabion.tooling.policy_substrate import invariant_graph
from gabion.tooling.runtime import invariant_graph as invariant_graph_runtime


def test_build_invariant_graph_materializes_planning_chart_overlay_live_repo() -> None:
    graph = invariant_graph.build_invariant_graph(REPO_ROOT)
    payload = graph.as_payload()

    node_kind_counts = payload["counts"]["node_kind_counts"]
    assert node_kind_counts["planning_chart_report"] == 1
    assert node_kind_counts["planning_phase"] == 3
    assert node_kind_counts["planning_chart_item"] > 0

    planning_chart_summary = payload["planning_chart_summary"]
    assert planning_chart_summary is not None
    phases = {
        phase["phase_kind"]: phase for phase in planning_chart_summary["phases"]
    }
    assert set(phases) == {"scan", "predict", "complete"}
    assert any(
        item["source_kind"] == "kernel_vm_alignment"
        for item in phases["scan"]["items"]
    )
    assert any(
        item["source_kind"] == "declared_counterfactual"
        for item in phases["predict"]["items"]
    )
    assert any(
        item["source_kind"] == "recommended_cut"
        for item in phases["complete"]["items"]
    )


def test_build_invariant_workstreams_includes_planning_chart_summary_live_repo() -> None:
    graph = invariant_graph.build_invariant_graph(REPO_ROOT)
    workstreams = invariant_graph.build_invariant_workstreams(graph, root=REPO_ROOT)
    payload = workstreams.as_payload()

    assert payload["planning_chart_summary"] == graph.as_payload()["planning_chart_summary"]


@pytest.mark.skip(
    reason=(
        "Deprecated live-repo snapshot test; replaced by injected workstream "
        "decomposition coverage under CSA-IDR-SQ-003."
    )
)
@deprecated_decorator(
    reason=(
        "Deprecated live-repo snapshot test; use injected invariant-workstream "
        "units instead of exact repo-state assertions."
    ),
    reasoning={
        "summary": (
            "The live PSF snapshot assertion set is intentionally retired in favor of "
            "dependency-injected workstream tests."
        ),
        "control": "connectivity_synergy.identity_rendering.deprecated_live_snapshot_test",
        "blocking_dependencies": ["CSA-IDR-SQ-003", "CSA-IDR-TP-003"],
    },
    owner="gabion.tooling.runtime_policy",
    expiry="CSA-IDR closure",
    links=[
        {"kind": "object_id", "value": "CSA-IDR-SQ-003"},
        {"kind": "object_id", "value": "CSA-IDR-TP-003"},
    ],
)
def test_build_psf_phase5_projection_matches_current_live_repo_state() -> None:
    graph = invariant_graph.build_invariant_graph(REPO_ROOT)
    projection = invariant_graph.build_psf_phase5_projection(graph)

    assert projection["queue_id"] == "PSF-007"
    assert graph.workstream_root_ids == (
        "CSA-IDR",
        "CSA-IGM",
        "CSA-IVL",
        "CSA-RGC",
        "PRF",
        "PSF-007",
        "RCI",
        "SCC",
    )


def test_runtime_invariant_graph_cli_blockers_reports_psf007_chains(
    tmp_path: Path,
    capsys,
) -> None:
    artifact = tmp_path / "artifacts/out/invariant_graph.json"
    workstreams_artifact = tmp_path / "artifacts/out/invariant_workstreams.json"
    ledger_artifact = tmp_path / "artifacts/out/invariant_ledger_projections.json"

    assert (
        invariant_graph_runtime.main(
            [
                "--root",
                str(REPO_ROOT),
                "--artifact",
                str(artifact),
                "--workstreams-artifact",
                str(workstreams_artifact),
                "--ledger-artifact",
                str(ledger_artifact),
                "build",
            ]
        )
        == 0
    )
    assert (
        invariant_graph_runtime.main(
            [
                "--root",
                str(REPO_ROOT),
                "--artifact",
                str(artifact),
                "blockers",
                "--object-id",
                "PSF-007",
            ]
        )
        == 0
    )
    blocker_output = capsys.readouterr().out
    assert "collapsible_helper_seam:" in blocker_output
    assert "surviving_carrier_seam:" in blocker_output

    assert (
        invariant_graph_runtime.main(
            [
                "--root",
                str(REPO_ROOT),
                "--artifact",
                str(artifact),
                "summary",
            ]
        )
        == 0
    )
    summary_output = capsys.readouterr().out
    assert "dominant_followup_class:" in summary_output
    assert "next_human_followup_family:" in summary_output
    assert "diagnostic_summary: unmatched_policy_signals=" in summary_output
    assert ":: unresolved_dependencies=" in summary_output
    assert ":: workspace_preservation=" in summary_output
    assert ":: workspace_orphans=" in summary_output
    assert "recommended_repo_followup:" in summary_output
    assert "recommended_repo_followup_cohort:" in summary_output
    assert "recommended_repo_code_followup:" in summary_output
    assert "recommended_repo_human_followup:" in summary_output
    assert "recommended_repo_followup_lane:" in summary_output
    assert "recommended_repo_code_followup_lane:" in summary_output
    assert "recommended_repo_human_followup_lane:" in summary_output
    assert "recommended_repo_followup_frontier_tradeoff:" in summary_output
    assert "recommended_repo_followup_frontier_explanation:" in summary_output
    assert "recommended_repo_followup_decision_protocol:" in summary_output
    assert "recommended_repo_followup_frontier_triad:" in summary_output
    assert "recommended_repo_followup_same_class_tradeoff:" in summary_output
    assert "recommended_repo_followup_cross_class_tradeoff:" in summary_output
    assert "repo_followup_lanes:" in summary_output
    assert "repo_diagnostic_lanes:" in summary_output

    assert (
        invariant_graph_runtime.main(
            [
                "--root",
                str(REPO_ROOT),
                "--artifact",
                str(artifact),
                "--workstreams-artifact",
                str(workstreams_artifact),
                "--ledger-artifact",
                str(ledger_artifact),
                "workstream",
                "--object-id",
                "PSF-007",
            ]
        )
        == 0
    )
    workstream_output = capsys.readouterr().out
    assert "object_id: PSF-007" in workstream_output
    assert "touchsites:" in workstream_output
    assert "health_summary:" in workstream_output
    assert "touchsite_blockers:" in workstream_output
    assert "health_cuts:" in workstream_output
    assert "dominant_blocker_class:" in workstream_output
    assert "recommended_remediation_family:" in workstream_output
    assert "recommended_cut:" in workstream_output
    assert "recommended_cut_frontier_explanation:" in workstream_output
    assert "recommended_cut_decision_protocol:" in workstream_output
    assert "recommended_cut_frontier_stability:" in workstream_output
    assert "recommended_ready_cut:" in workstream_output
    assert "recommended_coverage_gap_cut:" in workstream_output
    assert "recommended_policy_blocked_cut:" in workstream_output
    assert "recommended_diagnostic_blocked_cut:" in workstream_output
    assert "remediation_lanes:" in workstream_output
    assert "best=<none>" not in workstream_output
    assert "ranked_touchpoint_cuts:" in workstream_output
    assert "ranked_subqueue_cuts:" in workstream_output
    assert "ledger_alignment_summary: target_docs=2 ::" in workstream_output
    assert "dominant_doc_alignment_status: append_pending_existing_object" in workstream_output
    assert "recommended_doc_alignment_action:" in workstream_output
    assert "next_human_followup_family: documentation_alignment" in workstream_output
    assert "recommended_doc_followup_target_doc_id:" in workstream_output
    assert "recommended_followup:" in workstream_output
    assert "misaligned_target_doc_ids:" in workstream_output
    assert "documentation_followup_lanes:" in workstream_output
    assert "ranked_followups:" in workstream_output
    assert "documentation_alignment" in workstream_output

    assert (
        invariant_graph_runtime.main(
            [
                "--ledger-artifact",
                str(ledger_artifact),
                "ledger",
                "--object-id",
                "PSF-007",
            ]
        )
        == 0
    )
    ledger_output = capsys.readouterr().out
    assert (
        "target_doc_ids: projection_semantic_fragment_ledger, "
        "projection_semantic_fragment_rfc" in ledger_output
    )
    assert "recommended_ledger_action: record_progress_state" in ledger_output
    assert "current_snapshot: touchsites=" in ledger_output
    assert ":: surviving=" in ledger_output
    assert "alignment_summary: target_docs=2 ::" in ledger_output
    assert "recommended_doc_alignment_action:" in ledger_output
    assert "target_doc_alignments:" in ledger_output

    assert (
        invariant_graph_runtime.main(
            [
                "--root",
                str(REPO_ROOT),
                "--artifact",
                str(artifact),
                "blast-radius",
                "--id",
                "PSF-007",
            ]
        )
        == 0
    )
    blast_radius_output = capsys.readouterr().out
    assert "covering_tests:" in blast_radius_output
