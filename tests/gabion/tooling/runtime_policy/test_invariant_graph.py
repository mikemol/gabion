from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pytest

from tests.path_helpers import REPO_ROOT

from gabion.analysis.aspf.aspf_lattice_algebra import ReplayableStream
from gabion.invariants import deprecated_decorator
from gabion.tooling.policy_substrate import invariant_graph
from gabion.tooling.policy_substrate.projection_semantic_fragment_phase5_registry import (
    phase5_workstream_registry,
)
from gabion.tooling.policy_substrate.policy_queue_identity import PolicyQueueIdentitySpace
from gabion.tooling.policy_substrate.structured_artifact_ingress import (
    StructuredArtifactIdentitySpace,
    build_ingress_merge_parity_artifact,
    write_ingress_merge_parity_artifact,
)
from gabion.tooling.policy_substrate.workstream_registry import (
    RegisteredCounterfactualActionDefinition,
)
from gabion.tooling.runtime import invariant_graph as invariant_graph_runtime
from tests.gabion.tooling.runtime_policy.invariant_graph_test_support import (
    connectivity_synergy_with_psf_stub_workstream_registries,
    synthetic_connectivity_workstream_registries,
    write_minimal_invariant_repo,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


_NO_DECLARED_REGISTRIES = ()
_CONNECTIVITY_SYNERGY_WITH_PSF_STUB_DECLARED_REGISTRIES = (
    connectivity_synergy_with_psf_stub_workstream_registries()
)
_SYNTHETIC_CONNECTIVITY_DECLARED_REGISTRIES = (
    synthetic_connectivity_workstream_registries()
)


def _sample_repo(tmp_path: Path) -> Path:
    _write(tmp_path / "src" / "gabion" / "__init__.py", "")
    _write(
        tmp_path / "src" / "gabion" / "sample.py",
        "\n".join(
            [
                "from gabion.invariants import never, todo_decorator",
                "",
                "@todo_decorator(",
                "    reason='decorate sample',",
                "    owner='policy',",
                "    expiry='2099-01-01',",
                "    reasoning={",
                "        'summary': 'sample decorated marker',",
                "        'control': 'sample.decorated',",
                "        'blocking_dependencies': ['MISSING-OBJECT'],",
                "    },",
                "    links=[",
                "        {'kind': 'doc_id', 'value': 'sample_doc'},",
                "        {'kind': 'object_id', 'value': 'OBJ-TODO'},",
                "    ],",
                ")",
                "def decorated() -> None:",
                "    never(",
                "        'callsite marker',",
                "        owner='policy',",
                "        expiry='2099-01-01',",
                "        reasoning={",
                "            'summary': 'sample callsite marker',",
                "            'control': 'sample.callsite',",
                "            'blocking_dependencies': ['ANOTHER-MISSING-OBJECT'],",
                "        },",
                "        links=[{'kind': 'doc_id', 'value': 'sample_doc'}],",
                "    )",
                "    return None",
            ]
        ),
    )
    _write(
        tmp_path / "docs" / "sample.md",
        "\n".join(
            [
                "---",
                "doc_id: sample_doc",
                "doc_targets:",
                "  - gabion.sample.decorated",
                "---",
                "",
                "# Sample Doc",
            ]
        ),
    )
    return tmp_path


def _sample_decorated_line(root: Path) -> int:
    for line_number, line in enumerate(
        (root / "src" / "gabion" / "sample.py").read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if line.startswith("def decorated"):
            return line_number
    raise AssertionError("sample decorated line not found")


def _stream_from_items(items):
    return ReplayableStream(factory=lambda items=tuple(items): iter(items))


def _synthetic_workstreams_payload(workstreams: list[dict[str, object]]) -> dict[str, object]:
    return {
        "format_version": 1,
        "generated_at_utc": "2026-03-13T00:00:00+00:00",
        "root": str(REPO_ROOT),
        "workstreams": workstreams,
        "counts": {"workstream_count": len(workstreams)},
    }


def test_build_invariant_graph_scans_decorated_symbols_and_marker_callsites(
    tmp_path: Path,
) -> None:
    root = _sample_repo(tmp_path)

    graph = invariant_graph.build_invariant_graph(
        root,
        declared_registries=_NO_DECLARED_REGISTRIES,
    )
    payload = graph.as_payload()

    node_kind_counts = payload["counts"]["node_kind_counts"]
    assert node_kind_counts["decorated_symbol"] == 1
    assert node_kind_counts["marker_callsite"] == 1
    traced = invariant_graph.trace_nodes(graph, "OBJ-TODO")
    assert len(traced) == 2
    assert any(node.node_kind == "decorated_symbol" for node in traced)
    assert any(
        item.code == "unresolved_blocking_dependency"
        and item.raw_dependency == "MISSING-OBJECT"
        for item in graph.diagnostics
    )
    assert any(
        item.code == "unresolved_blocking_dependency"
        and item.raw_dependency == "ANOTHER-MISSING-OBJECT"
        for item in graph.diagnostics
    )


def test_invariant_graph_write_and_load_round_trip(
    tmp_path: Path,
) -> None:
    root = _sample_repo(tmp_path)
    artifact = tmp_path / "artifacts/out/invariant_graph.json"

    graph = invariant_graph.build_invariant_graph(
        root,
        declared_registries=_NO_DECLARED_REGISTRIES,
    )
    invariant_graph.write_invariant_graph(artifact, graph)
    reloaded = invariant_graph.load_invariant_graph(artifact)

    assert artifact.exists()
    assert len(reloaded.nodes) == len(graph.nodes)
    assert len(reloaded.edges) == len(graph.edges)
    assert len(reloaded.diagnostics) == len(graph.diagnostics)
    assert reloaded.planning_chart_summary == graph.planning_chart_summary


def test_phase5_touchsite_scan_uses_active_build_root_and_fails_closed_when_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="phase5 touchsite scan requires source under active build root",
    ) as exc_info:
        invariant_graph.build_invariant_graph(
            tmp_path,
            declared_registries=(phase5_workstream_registry(),),
        )

    assert str(tmp_path.resolve()) in str(exc_info.value)
    assert "src/gabion/analysis/projection/semantic_fragment.py" in str(exc_info.value)


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
    workstreams = invariant_graph.build_invariant_workstreams(graph, root=REPO_ROOT)
    ledgers = invariant_graph.build_invariant_ledger_projections(
        workstreams,
        root=REPO_ROOT,
    )

    assert not hasattr(invariant_graph, "_PHASE5_SURVIVING_TOUCHSITE_BOUNDARY_NAMES")
    assert projection["queue_id"] == "PSF-007"
    assert projection["remaining_touchsite_count"] == 72
    assert projection["collapsible_touchsite_count"] == 47
    assert projection["surviving_touchsite_count"] == 25
    assert len(projection["subqueues"]) == 5
    assert len(projection["touchpoints"]) == 6
    assert graph.workstream_root_ids == (
        "CSA-IDR",
        "CSA-IGM",
        "CSA-RGC",
        "PRF",
        "PSF-007",
    )
    workstreams_payload = workstreams.as_payload()
    assert workstreams_payload["diagnostic_summary"] == {
        "diagnostic_count": 7,
        "unmatched_policy_signal_count": 7,
        "unresolved_blocking_dependency_count": 0,
        "workspace_preservation_count": 0,
        "orphaned_workspace_change_count": 0,
        "buckets": [
            {
                "code": "unmatched_policy_signal",
                "severity": "warning",
                "count": 7,
            }
        ],
    }
    recommended_followup = workstreams_payload["repo_next_actions"]["recommended_followup"]
    assert recommended_followup == {
        "followup_family": "governance_orphan_resolution",
        "action_kind": "diagnostic_resolution",
        "priority_rank": 0,
        "object_id": None,
        "owner_object_id": None,
        "owner_root_object_id": None,
        "diagnostic_code": "unmatched_policy_signal",
        "target_doc_id": None,
        "policy_ids": ["GMP-001"],
        "title": "seed ownership for grade:GMP-001 from src/gabion/analysis/dataflow/io",
        "blocker_class": "policy_orphan",
        "readiness_class": None,
        "alignment_status": None,
        "recommended_action": "seed_owned_workstream_from_source_family",
        "owner_seed_path": "src/gabion/analysis/dataflow/io",
        "owner_seed_object_id": "WS-SEED:gabion.analysis.dataflow.io",
        "owner_resolution_kind": "seed_new_owner",
        "owner_resolution_score": 100,
        "owner_resolution_options": [
            {
                "resolution_kind": "seed_new_owner",
                "owner_status": "source_family_seed_owner",
                "object_id": "WS-SEED:gabion.analysis.dataflow.io",
                "score": 100,
                "rationale": "source_family_seed",
                "score_components": [
                    {
                        "kind": "seed_new_owner_base",
                        "score": 100,
                        "rationale": "source_family_seed",
                    }
                ],
                "selection_rank": 1,
                "opportunity_cost_score": 0,
                "opportunity_cost_reason": "frontier",
                "opportunity_cost_components": [],
            }
        ],
        "runner_up_owner_object_id": None,
        "runner_up_owner_resolution_kind": None,
        "runner_up_owner_resolution_score": None,
        "owner_choice_margin_score": 100,
        "owner_choice_margin_reason": "uncontested_best_option",
        "owner_choice_margin_components": [
            {
                "kind": "seed_new_owner_base",
                "score": 100,
                "rationale": "source_family_seed",
            }
        ],
        "owner_option_tradeoff_score": 100,
        "owner_option_tradeoff_reason": "uncontested_best_option",
        "owner_option_tradeoff_components": [
            {
                "kind": "seed_new_owner_base",
                "score": 100,
                "rationale": "source_family_seed",
            }
        ],
        "utility_score": 1190,
        "utility_reason": (
            "governance_orphan:seed_new_owner+owner_option_tradeoff:100"
            "+governance_priority:GMP-001:10"
        ),
        "utility_components": [
            {
                "kind": "governance_orphan_base",
                "score": 900,
                "rationale": "governance_orphan",
            },
            {
                "kind": "owner_resolution_bonus",
                "score": 100,
                "rationale": "seed_new_owner",
            },
            {
                "kind": "owner_option_tradeoff_bonus",
                "score": 100,
                "rationale": "uncontested_best_option",
            },
            {
                "kind": "governance_priority_bonus",
                "score": 90,
                "rationale": "governance_priority:GMP-001:10",
            },
        ],
        "selection_certainty_kind": "frontier_unique",
        "cofrontier_followup_count": 1,
        "cofrontier_followup_cohort": [
            {
                "followup_family": "governance_orphan_resolution",
                "followup_class": "governance",
                "action_kind": "diagnostic_resolution",
                "object_id": None,
                "owner_root_object_id": None,
                "diagnostic_code": "unmatched_policy_signal",
                "target_doc_id": None,
                "policy_ids": ["GMP-001"],
                "title": (
                    "seed ownership for grade:GMP-001 from "
                    "src/gabion/analysis/dataflow/io"
                ),
                "utility_score": 1190,
                "selection_rank": 1,
                "selection_reason": "frontier_tiebreak_winner",
            }
        ],
        "selection_scope_kind": "singleton",
        "selection_scope_id": None,
        "runner_up_followup_family": "governance_orphan_resolution",
        "runner_up_followup_class": "governance",
        "runner_up_followup_object_id": None,
        "runner_up_followup_utility_score": 1180,
        "frontier_choice_margin_score": 10,
        "frontier_choice_margin_reason": (
            "governance_orphan:seed_new_owner+owner_option_tradeoff:100"
            "+governance_priority:GMP-001:10"
            "->governance_orphan:seed_new_owner+owner_option_tradeoff:100"
            "+governance_priority:GMP-002:20"
        ),
        "frontier_choice_margin_components": [
            {
                "kind": "governance_orphan_base",
                "score": 900,
                "rationale": "governance_orphan",
            },
            {
                "kind": "owner_resolution_bonus",
                "score": 100,
                "rationale": "seed_new_owner",
            },
            {
                "kind": "owner_option_tradeoff_bonus",
                "score": 100,
                "rationale": "uncontested_best_option",
            },
            {
                "kind": "governance_priority_bonus",
                "score": 90,
                "rationale": "governance_priority:GMP-001:10",
            },
            {
                "kind": "runner_up_offset:governance_orphan_base",
                "score": -900,
                "rationale": "governance_orphan",
            },
            {
                "kind": "runner_up_offset:owner_resolution_bonus",
                "score": -100,
                "rationale": "seed_new_owner",
            },
            {
                "kind": "runner_up_offset:owner_option_tradeoff_bonus",
                "score": -100,
                "rationale": "uncontested_best_option",
            },
            {
                "kind": "runner_up_offset:governance_priority_bonus",
                "score": -80,
                "rationale": "governance_priority:GMP-002:20",
            },
        ],
        "selection_rank": 1,
        "opportunity_cost_score": 0,
        "opportunity_cost_reason": "frontier",
        "opportunity_cost_components": [],
        "queue_id": (
            "planner_queue|followup_family=governance_orphan_resolution|"
            "followup_class=governance|selection_scope_kind=singleton|"
            "selection_scope_id=|root_object_ids="
        ),
        "count": 1,
    }
    assert workstreams_payload["repo_next_actions"]["dominant_followup_class"] == (
        "governance"
    )
    assert workstreams_payload["repo_next_actions"]["next_human_followup_family"] == (
        "governance_orphan_resolution"
    )
    recommended_code_followup = workstreams_payload["repo_next_actions"][
        "recommended_code_followup"
    ]
    assert recommended_code_followup["followup_family"] == "coverage_gap"
    assert recommended_code_followup["action_kind"] == "touchpoint_cut"
    assert recommended_code_followup["object_id"] == "PSF-007-TP-001"
    assert recommended_code_followup["owner_object_id"] == "PSF-007"
    assert recommended_code_followup["owner_root_object_id"] == "PSF-007"
    assert recommended_code_followup["readiness_class"] == "coverage_gap"
    assert recommended_code_followup["utility_score"] == 480
    assert recommended_code_followup["utility_reason"] == "code:coverage_gap"
    assert recommended_code_followup["queue_id"].startswith(
        "planner_queue|followup_family=coverage_gap|"
    )
    assert recommended_code_followup["selection_certainty_kind"] == "ranked_plateau"
    assert recommended_code_followup["selection_scope_kind"] == "mixed_root_followup_family"
    assert recommended_code_followup["selection_scope_id"] == (
        "coverage_gap:CSA-IDR,CSA-IGM,CSA-RGC,PSF-007"
    )
    assert recommended_code_followup["count"] == 4
    assert workstreams_payload["repo_next_actions"]["recommended_human_followup"] == (
        recommended_followup
    )
    assert workstreams_payload["repo_next_actions"]["recommended_followup_lane"] == {
        "queue_id": (
            "planner_queue|followup_family=governance_orphan_resolution|"
            "followup_class=governance|selection_scope_kind=singleton|"
            "selection_scope_id=|root_object_ids="
        ),
        "followup_family": "governance_orphan_resolution",
        "followup_class": "governance",
        "selection_scope_kind": "singleton",
        "selection_scope_id": None,
        "action_count": 7,
        "root_object_ids": [],
        "member_object_ids": [],
        "strongest_owner_resolution_kind": "seed_new_owner",
        "strongest_owner_resolution_score": 100,
        "strongest_utility_score": 1190,
        "strongest_utility_reason": (
            "governance_orphan:seed_new_owner+owner_option_tradeoff:100"
            "+governance_priority:GMP-001:10"
        ),
        "lane_utility_score": 1250,
        "lane_utility_reason": (
            "governance_orphan:seed_new_owner+owner_option_tradeoff:100"
            "+governance_priority:GMP-001:10"
            "+lane_breadth:7+lane:governance_orphan_resolution"
        ),
        "lane_utility_components": [
            {
                "kind": "best_followup_utility",
                "score": 1190,
                "rationale": (
                    "governance_orphan:seed_new_owner+owner_option_tradeoff:100"
                    "+governance_priority:GMP-001:10"
                ),
            },
            {
                "kind": "lane_breadth_bonus",
                "score": 35,
                "rationale": "lane_breadth:7",
            },
            {
                "kind": "lane_class_bonus",
                "score": 25,
                "rationale": "lane_class:governance",
            },
        ],
        "selection_rank": 1,
        "opportunity_cost_score": 0,
        "opportunity_cost_reason": "frontier",
        "opportunity_cost_components": [],
        "best_followup": recommended_followup,
    }
    assert (
        workstreams_payload["repo_next_actions"]["recommended_queue"]
        == workstreams_payload["repo_next_actions"]["recommended_followup_lane"]
    )
    recommended_code_followup_lane = workstreams_payload["repo_next_actions"][
        "recommended_code_followup_lane"
    ]
    assert recommended_code_followup_lane["followup_family"] == "coverage_gap"
    assert recommended_code_followup_lane["followup_class"] == "code"
    assert recommended_code_followup_lane["queue_id"].startswith(
        "planner_queue|followup_family=coverage_gap|"
    )
    assert recommended_code_followup_lane["root_object_ids"] == [
        "CSA-IDR",
        "CSA-IGM",
        "CSA-RGC",
        "PSF-007",
    ]
    assert recommended_code_followup_lane["selection_scope_kind"] == (
        "mixed_root_followup_family"
    )
    assert recommended_code_followup_lane["selection_rank"] == 2
    assert recommended_code_followup_lane["best_followup"] == recommended_code_followup
    assert (
        workstreams_payload["repo_next_actions"]["recommended_human_followup_lane"]
        == workstreams_payload["repo_next_actions"]["recommended_followup_lane"]
    )
    frontier_tradeoff = workstreams_payload["repo_next_actions"][
        "recommended_followup_frontier_tradeoff"
    ]
    assert frontier_tradeoff["frontier_followup_family"] == (
        "governance_orphan_resolution"
    )
    assert frontier_tradeoff["runner_up_followup_family"] == "coverage_gap"
    assert frontier_tradeoff["runner_up_followup_class"] == "code"
    assert frontier_tradeoff["margin_score"] == 740
    frontier_explanation = workstreams_payload["repo_next_actions"][
        "recommended_followup_frontier_explanation"
    ]
    assert frontier_explanation["frontier_policy_ids"] == ["GMP-001"]
    assert frontier_explanation["same_class_runner_up_policy_ids"] == ["GMP-002"]
    assert frontier_explanation["cross_class_runner_up_object_id"] == "PSF-007-TP-001"
    assert frontier_explanation["cross_class_runner_up_utility_reason"] == (
        "code:coverage_gap"
    )
    decision_protocol = workstreams_payload["repo_next_actions"][
        "recommended_followup_decision_protocol"
    ]
    assert decision_protocol["frontier_followup_family"] == (
        "governance_orphan_resolution"
    )
    assert decision_protocol["frontier_followup_class"] == "governance"
    assert decision_protocol["frontier_action_kind"] == "diagnostic_resolution"
    assert decision_protocol["frontier_diagnostic_code"] == "unmatched_policy_signal"
    assert decision_protocol["frontier_policy_ids"] == ["GMP-001"]
    assert decision_protocol["decision_mode"] == "frontier_watch_same_class"
    assert decision_protocol["same_class_pressure"] == "high"
    assert decision_protocol["cross_class_pressure"] == "low"
    assert workstreams_payload["repo_next_actions"][
        "recommended_followup_frontier_triad"
    ] == {
        "frontier_followup_family": "governance_orphan_resolution",
        "frontier_followup_class": "governance",
        "frontier_action_kind": "diagnostic_resolution",
        "frontier_object_id": None,
        "frontier_diagnostic_code": "unmatched_policy_signal",
        "frontier_target_doc_id": None,
        "frontier_policy_ids": ["GMP-001"],
        "frontier_utility_score": 1190,
        "frontier_utility_reason": (
            "governance_orphan:seed_new_owner+owner_option_tradeoff:100"
            "+governance_priority:GMP-001:10"
        ),
        "same_class_tradeoff": workstreams_payload["repo_next_actions"][
            "recommended_followup_same_class_tradeoff"
        ],
        "cross_class_tradeoff": workstreams_payload["repo_next_actions"][
            "recommended_followup_cross_class_tradeoff"
        ],
    }
    assert workstreams_payload["repo_next_actions"][
        "recommended_followup_same_class_tradeoff"
    ] == {
        "frontier_followup_family": "governance_orphan_resolution",
        "frontier_followup_class": "governance",
        "frontier_action_kind": "diagnostic_resolution",
        "frontier_object_id": None,
        "frontier_diagnostic_code": "unmatched_policy_signal",
        "frontier_target_doc_id": None,
        "frontier_policy_ids": ["GMP-001"],
        "frontier_utility_score": 1190,
        "frontier_utility_reason": (
            "governance_orphan:seed_new_owner+owner_option_tradeoff:100"
            "+governance_priority:GMP-001:10"
        ),
        "runner_up_followup_family": "governance_orphan_resolution",
        "runner_up_followup_class": "governance",
        "runner_up_action_kind": "diagnostic_resolution",
        "runner_up_object_id": None,
        "runner_up_diagnostic_code": "unmatched_policy_signal",
        "runner_up_target_doc_id": None,
        "runner_up_policy_ids": ["GMP-002"],
        "runner_up_utility_score": 1180,
        "runner_up_utility_reason": (
            "governance_orphan:seed_new_owner+owner_option_tradeoff:100"
            "+governance_priority:GMP-002:20"
        ),
        "margin_score": 10,
        "margin_reason": (
            "governance_orphan:seed_new_owner+owner_option_tradeoff:100"
            "+governance_priority:GMP-001:10"
            "->governance_orphan:seed_new_owner+owner_option_tradeoff:100"
            "+governance_priority:GMP-002:20"
        ),
        "margin_components": [
            {
                "kind": "governance_orphan_base",
                "score": 900,
                "rationale": "governance_orphan",
            },
            {
                "kind": "owner_resolution_bonus",
                "score": 100,
                "rationale": "seed_new_owner",
            },
            {
                "kind": "owner_option_tradeoff_bonus",
                "score": 100,
                "rationale": "uncontested_best_option",
            },
            {
                "kind": "governance_priority_bonus",
                "score": 90,
                "rationale": "governance_priority:GMP-001:10",
            },
            {
                "kind": "runner_up_offset:governance_orphan_base",
                "score": -900,
                "rationale": "governance_orphan",
            },
            {
                "kind": "runner_up_offset:owner_resolution_bonus",
                "score": -100,
                "rationale": "seed_new_owner",
            },
            {
                "kind": "runner_up_offset:owner_option_tradeoff_bonus",
                "score": -100,
                "rationale": "uncontested_best_option",
            },
            {
                "kind": "runner_up_offset:governance_priority_bonus",
                "score": -80,
                "rationale": "governance_priority:GMP-002:20",
            },
        ],
    }
    assert workstreams_payload["repo_next_actions"][
        "recommended_followup_cross_class_tradeoff"
    ]["runner_up_object_id"] == "PSF-007-TP-001"
    assert workstreams_payload["repo_next_actions"][
        "recommended_followup_cross_class_tradeoff"
    ]["runner_up_utility_reason"] == "code:coverage_gap"
    assert workstreams_payload["repo_next_actions"][
        "recommended_followup_cross_class_tradeoff"
    ]["margin_score"] == 710
    ranked_repo_followups = workstreams_payload["repo_next_actions"]["ranked_followups"]
    assert ranked_repo_followups[0] == recommended_followup
    assert ranked_repo_followups[1]["followup_family"] == "governance_orphan_resolution"
    assert ranked_repo_followups[1]["diagnostic_code"] == "unmatched_policy_signal"
    assert ranked_repo_followups[1]["title"].startswith("seed ownership for grade:GMP-")
    assert any(
        item == workstreams_payload["repo_next_actions"]["recommended_code_followup"]
        for item in ranked_repo_followups
    )
    assert any(
        item["followup_family"] == "documentation_alignment"
        and item["target_doc_id"] == "projection_semantic_fragment_ledger"
        and item["recommended_action"] == "append_existing_ledger_entry"
        for item in ranked_repo_followups
    )
    repo_followup_lanes = workstreams_payload["repo_next_actions"]["followup_lanes"]
    assert repo_followup_lanes[0]["followup_family"] == "governance_orphan_resolution"
    assert repo_followup_lanes[0]["selection_rank"] == 1
    assert repo_followup_lanes[0]["best_followup"] == recommended_followup
    assert repo_followup_lanes[1]["followup_family"] == "coverage_gap"
    assert repo_followup_lanes[1]["followup_class"] == "code"
    assert any(
        lane["followup_family"] == "documentation_alignment"
        and lane["followup_class"] == "documentation"
        for lane in repo_followup_lanes
    )
    repo_diagnostic_lanes = workstreams_payload["repo_next_actions"]["diagnostic_lanes"]
    assert len(repo_diagnostic_lanes) == 7
    assert repo_diagnostic_lanes[0]["diagnostic_code"] == "unmatched_policy_signal"
    assert repo_diagnostic_lanes[0]["severity"] == "warning"
    assert repo_diagnostic_lanes[0]["title"] == "grade:GMP-001"
    assert repo_diagnostic_lanes[0]["recommended_action"] == (
        "seed_owned_workstream_from_source_family"
    )
    assert repo_diagnostic_lanes[0]["count"] == 1
    assert repo_diagnostic_lanes[0]["policy_ids"] == ["GMP-001"]
    assert repo_diagnostic_lanes[0]["rel_path"].startswith("src/gabion/")
    assert repo_diagnostic_lanes[0]["qualname"].startswith("gabion.")
    assert repo_diagnostic_lanes[0]["line"] > 0
    assert repo_diagnostic_lanes[0]["column"] > 0
    assert repo_diagnostic_lanes[0]["candidate_owner_status"] == "source_family_seed_owner"
    assert repo_diagnostic_lanes[0]["candidate_owner_object_id"] is None
    assert repo_diagnostic_lanes[0]["candidate_owner_object_ids"] == []
    assert repo_diagnostic_lanes[0]["candidate_owner_seed_path"] == (
        "src/gabion/analysis/dataflow/io"
    )
    assert repo_diagnostic_lanes[0]["candidate_owner_seed_object_id"] == (
        "WS-SEED:gabion.analysis.dataflow.io"
    )
    assert repo_diagnostic_lanes[0]["candidate_owner_options"] == [
        {
            "resolution_kind": "seed_new_owner",
            "owner_status": "source_family_seed_owner",
            "object_id": "WS-SEED:gabion.analysis.dataflow.io",
            "score": 100,
            "rationale": "source_family_seed",
            "score_components": [
                {
                    "kind": "seed_new_owner_base",
                    "score": 100,
                    "rationale": "source_family_seed",
                }
            ],
            "selection_rank": 1,
            "opportunity_cost_score": 0,
            "opportunity_cost_reason": "frontier",
            "opportunity_cost_components": [],
        }
    ]
    assert repo_diagnostic_lanes[0]["runner_up_candidate_owner_option"] is None
    assert repo_diagnostic_lanes[0]["candidate_owner_choice_margin_score"] == 100
    assert (
        repo_diagnostic_lanes[0]["candidate_owner_choice_margin_reason"]
        == "uncontested_best_option"
    )
    assert repo_diagnostic_lanes[0]["candidate_owner_choice_margin_components"] == [
        {
            "kind": "seed_new_owner_base",
            "score": 100,
            "rationale": "source_family_seed",
        }
    ]
    assert len(repo_diagnostic_lanes[0]["node_ids"]) == 1
    assert repo_diagnostic_lanes[-1]["title"] == "grade:GMP-007"
    assert all(
        lane["recommended_action"] == "seed_owned_workstream_from_source_family"
        for lane in repo_diagnostic_lanes
    )
    projected_ids = [
        str(item.get("object_id", ""))
        for item in workstreams_payload.get("workstreams", [])
        if isinstance(item, dict)
    ]
    assert projected_ids == ["CSA-IDR", "CSA-IGM", "CSA-RGC", "PRF", "PSF-007"]
    prf = next(
        item
        for item in workstreams_payload["workstreams"]
        if isinstance(item, dict) and item.get("object_id") == "PRF"
    )
    assert prf["status"] == "in_progress"
    assert prf["touchsite_count"] == 3
    assert set(prf["doc_ids"]) == {
        "policy_rule_frontmatter_migration_ledger",
        "enforceable_rules_cheat_sheet",
    }
    psf = next(
        item
        for item in workstreams_payload["workstreams"]
        if isinstance(item, dict) and item.get("object_id") == "PSF-007"
    )
    assert psf["touchsite_count"] == 72
    assert psf["collapsible_touchsite_count"] == 47
    assert psf["surviving_touchsite_count"] == 25
    assert "projection_semantic_fragment_ledger" in psf["doc_ids"]
    assert "projection_semantic_fragment_rfc" in psf["doc_ids"]
    assert psf["health_summary"]["covered_touchsite_count"] == 2
    assert psf["health_summary"]["uncovered_touchsite_count"] == 70
    assert psf["health_summary"]["governed_touchsite_count"] == 0
    assert psf["health_summary"]["diagnosed_touchsite_count"] == 0
    assert psf["health_summary"]["ready_touchsite_count"] == 2
    assert psf["health_summary"]["coverage_gap_touchsite_count"] == 70
    assert psf["health_summary"]["policy_blocked_touchsite_count"] == 0
    assert psf["health_summary"]["diagnostic_blocked_touchsite_count"] == 0
    assert psf["health_summary"]["ready_touchpoint_cut_count"] == 0
    assert psf["health_summary"]["coverage_gap_touchpoint_cut_count"] == 5
    assert psf["health_summary"]["ready_subqueue_cut_count"] == 0
    assert psf["health_summary"]["coverage_gap_subqueue_cut_count"] == 5
    assert psf["next_actions"]["recommended_cut"]["object_id"] == "PSF-007-TP-001"
    assert psf["next_actions"]["recommended_cut"]["cut_kind"] == "touchpoint_cut"
    assert psf["next_actions"]["recommended_cut"]["touchsite_count"] == 4
    assert psf["next_actions"]["recommended_cut_frontier_explanation"][
        "frontier_object_id"
    ] == "PSF-007-TP-001"
    assert psf["next_actions"]["recommended_cut_frontier_explanation"][
        "same_kind_tradeoff"
    ]["runner_up_object_id"] == "PSF-007-TP-006"
    assert psf["next_actions"]["recommended_cut_frontier_explanation"][
        "cross_kind_tradeoff"
    ]["runner_up_object_id"] == "PSF-007-SQ-001"
    assert (
        psf["next_actions"]["recommended_cut_decision_protocol"]["frontier_object_id"]
        == "PSF-007-TP-001"
    )
    assert (
        psf["next_actions"]["recommended_cut_frontier_stability"]["frontier_object_id"]
        == "PSF-007-TP-001"
    )
    assert psf["next_actions"]["recommended_ready_cut"] is None
    assert psf["next_actions"]["recommended_coverage_gap_cut"]["object_id"] == (
        "PSF-007-TP-001"
    )
    assert psf["next_actions"]["dominant_blocker_class"] == "coverage_gap"
    assert psf["next_actions"]["recommended_remediation_family"] == "coverage_gap"
    assert psf["next_actions"]["recommended_policy_blocked_cut"] is None
    assert psf["next_actions"]["recommended_diagnostic_blocked_cut"] is None
    assert psf["doc_alignment_summary"]["target_doc_count"] == 2
    assert psf["next_actions"]["dominant_doc_alignment_status"] == (
        "append_pending_existing_object"
    )
    assert psf["next_actions"]["recommended_doc_alignment_action"] == (
        "append_existing_ledger_entry"
    )
    assert psf["next_actions"]["next_human_followup_family"] == (
        "documentation_alignment"
    )
    assert psf["next_actions"]["recommended_doc_followup_target_doc_id"] == (
        "projection_semantic_fragment_ledger"
    )
    assert psf["next_actions"]["recommended_followup"] == {
        "followup_family": "coverage_gap",
        "action_kind": "touchpoint_cut",
        "priority_rank": 2,
        "object_id": "PSF-007-TP-001",
        "owner_object_id": "PSF-007-SQ-001",
        "target_doc_id": None,
        "title": "semantic_fragment.py canonicalization surfaces",
        "blocker_class": "coverage_gap",
        "readiness_class": "coverage_gap",
        "alignment_status": None,
        "recommended_action": None,
        "touchsite_count": 4,
        "collapsible_touchsite_count": 0,
        "surviving_touchsite_count": 4,
    }
    assert psf["next_actions"]["misaligned_target_doc_ids"] == [
        "projection_semantic_fragment_ledger",
        "projection_semantic_fragment_rfc",
    ]
    assert psf["next_actions"]["documentation_followup_lanes"] == [
        {
            "followup_family": "documentation_alignment",
            "alignment_status": "append_pending_existing_object",
            "target_doc_count": 2,
            "misaligned_target_doc_count": 2,
            "target_doc_ids": [
                "projection_semantic_fragment_ledger",
                "projection_semantic_fragment_rfc",
            ],
            "misaligned_target_doc_ids": [
                "projection_semantic_fragment_ledger",
                "projection_semantic_fragment_rfc",
            ],
            "recommended_action": "append_existing_ledger_entry",
            "best_target_doc_id": "projection_semantic_fragment_ledger",
        }
    ]
    assert psf["next_actions"]["remediation_lanes"][0]["remediation_family"] == (
        "structural_cut"
    )
    assert psf["next_actions"]["remediation_lanes"][0]["best_cut"] is None
    assert psf["next_actions"]["remediation_lanes"][1]["remediation_family"] == (
        "coverage_gap"
    )
    assert psf["next_actions"]["remediation_lanes"][1]["best_cut"]["object_id"] == (
        "PSF-007-TP-001"
    )
    assert psf["next_actions"]["ranked_followups"] == [
        {
            "followup_family": "coverage_gap",
            "action_kind": "touchpoint_cut",
            "priority_rank": 2,
            "object_id": "PSF-007-TP-001",
            "owner_object_id": "PSF-007-SQ-001",
            "target_doc_id": None,
            "title": "semantic_fragment.py canonicalization surfaces",
            "blocker_class": "coverage_gap",
            "readiness_class": "coverage_gap",
            "alignment_status": None,
            "recommended_action": None,
            "touchsite_count": 4,
            "collapsible_touchsite_count": 0,
            "surviving_touchsite_count": 4,
        },
        {
            "followup_family": "documentation_alignment",
            "action_kind": "doc_alignment",
            "priority_rank": 24,
            "object_id": None,
            "owner_object_id": "PSF-007",
            "target_doc_id": "projection_semantic_fragment_ledger",
            "title": "projection_semantic_fragment_ledger",
            "blocker_class": None,
            "readiness_class": None,
            "alignment_status": "append_pending_existing_object",
            "recommended_action": "append_existing_ledger_entry",
            "touchsite_count": 72,
            "collapsible_touchsite_count": 47,
            "surviving_touchsite_count": 25,
        },
    ]
    assert psf["next_actions"]["ranked_touchpoint_cuts"][0]["object_id"] == "PSF-007-TP-001"
    assert psf["next_actions"]["ranked_touchpoint_cuts"][0]["readiness_class"] == (
        "coverage_gap"
    )
    assert psf["next_actions"]["ranked_subqueue_cuts"][0]["object_id"] == "PSF-007-SQ-001"
    assert psf["next_actions"]["ranked_subqueue_cuts"][0]["readiness_class"] == (
        "coverage_gap"
    )
    ledger_payload = ledgers.as_payload()
    psf_ledger = next(
        item
        for item in ledger_payload["ledgers"]
        if isinstance(item, dict) and item.get("object_id") == "PSF-007"
    )
    assert "projection_semantic_fragment_ledger" in psf_ledger["target_doc_ids"]
    assert psf_ledger["recommended_ledger_action"] == "record_progress_state"
    assert psf_ledger["current_snapshot"]["recommended_cut_object_id"] == "PSF-007-TP-001"
    assert psf_ledger["alignment_summary"]["target_doc_count"] == 2
    assert psf_ledger["alignment_summary"]["recommended_doc_alignment_action"] in {
        "append_existing_ledger_entry",
        "append_new_ledger_entry",
        "none",
    }
    assert len(psf_ledger["target_doc_alignments"]) == 2


def test_injected_repo_followup_plateau_reports_mixed_root_scope(
    tmp_path: Path,
) -> None:
    root = write_minimal_invariant_repo(tmp_path)

    graph = invariant_graph.build_invariant_graph(
        root,
        declared_registries=_SYNTHETIC_CONNECTIVITY_DECLARED_REGISTRIES,
    )
    workstreams = invariant_graph.build_invariant_workstreams(graph, root=root)

    assert graph.workstream_root_ids == (
        "CSA-IDR",
        "CSA-IGM",
        "CSA-IVL",
        "CSA-RGC",
        "SCC",
    )
    recommended_code_followup = workstreams.recommended_repo_code_followup()
    assert recommended_code_followup is not None
    assert recommended_code_followup.followup_family == "coverage_gap"
    assert recommended_code_followup.selection_certainty_kind == "frontier_plateau"
    assert recommended_code_followup.selection_scope_kind == "mixed_root_followup_family"
    assert recommended_code_followup.selection_scope_id == (
        "coverage_gap:CSA-IDR,CSA-IGM,CSA-IVL,CSA-RGC,SCC"
    )
    assert recommended_code_followup.cofrontier_followup_count == 5


def test_injected_repo_followup_plateau_preserves_root_provenance(
    tmp_path: Path,
) -> None:
    root = write_minimal_invariant_repo(tmp_path)

    graph = invariant_graph.build_invariant_graph(
        root,
        declared_registries=_SYNTHETIC_CONNECTIVITY_DECLARED_REGISTRIES,
    )
    workstreams = invariant_graph.build_invariant_workstreams(graph, root=root)

    recommended_code_followup = workstreams.recommended_repo_code_followup()
    assert recommended_code_followup is not None
    assert {
        item.owner_root_object_id
        for item in recommended_code_followup.cofrontier_followup_cohort
    } == {"CSA-IDR", "CSA-IGM", "CSA-IVL", "CSA-RGC", "SCC"}
    recommended_code_lane = workstreams.recommended_repo_code_followup_lane()
    assert recommended_code_lane is not None
    assert recommended_code_lane.root_object_ids == (
        "CSA-IDR",
        "CSA-IGM",
        "CSA-IVL",
        "CSA-RGC",
        "SCC",
    )
    assert recommended_code_lane.best_followup.owner_root_object_id in {
        "CSA-IDR",
        "CSA-IGM",
        "CSA-IVL",
        "CSA-RGC",
        "SCC",
    }


def test_planner_queue_overlay_uses_envelops_without_reusing_contains(
    tmp_path: Path,
) -> None:
    root = write_minimal_invariant_repo(tmp_path)

    graph = invariant_graph.build_invariant_graph(
        root,
        declared_registries=_CONNECTIVITY_SYNERGY_WITH_PSF_STUB_DECLARED_REGISTRIES,
    )

    queue_nodes = [
        node for node in graph.nodes if node.node_kind == "planner_queue"
    ]
    assert queue_nodes
    coverage_queue = next(
        node for node in queue_nodes if node.title == "coverage_gap queue"
    )
    graph_edges = {(edge.edge_kind, edge.source_id, edge.target_id) for edge in graph.edges}
    coverage_envelop_targets = {
        target_id
        for edge_kind, source_id, target_id in graph_edges
        if edge_kind == "envelops" and source_id == coverage_queue.node_id
    }
    assert "object_id:CSA-IDR" in coverage_envelop_targets
    assert any(
        target_id.startswith("object_id:") for target_id in coverage_envelop_targets
    )
    assert not any(
        edge_kind == "contains" and source_id == coverage_queue.node_id
        for edge_kind, source_id, _target_id in graph_edges
    )
    envelop_counts_by_target: dict[str, int] = {}
    for edge_kind, source_id, target_id in graph_edges:
        if edge_kind != "envelops" or not source_id.startswith("planner_queue:"):
            continue
        if not target_id.startswith("object_id:"):
            continue
        envelop_counts_by_target[target_id] = (
            envelop_counts_by_target.get(target_id, 0) + 1
        )
    assert any(count >= 2 for count in envelop_counts_by_target.values())


def test_workstream_projection_surfaces_policy_and_diagnostic_remediation_families() -> None:
    space = PolicyQueueIdentitySpace()
    workstream_id = space.workstream_id("WS-SYNTH")
    subqueue_policy = space.subqueue_id("WS-SYNTH-SQ-POLICY")
    subqueue_diag = space.subqueue_id("WS-SYNTH-SQ-DIAG")
    touchpoint_policy = space.touchpoint_id("WS-SYNTH-TP-POLICY")
    touchpoint_diag = space.touchpoint_id("WS-SYNTH-TP-DIAG")
    policy_touchsite = invariant_graph.InvariantTouchsiteProjection(
        object_id=space.touchsite_id("WS-SYNTH-TS-POLICY"),
        touchpoint_id=touchpoint_policy,
        subqueue_id=subqueue_policy,
        title="policy touchsite",
        status="in_progress",
        rel_path="src/gabion/sample.py",
        qualname="policy_touchsite",
        boundary_name="policy_touchsite",
        line=10,
        column=1,
        node_kind="marker_callsite",
        site_identity=space.site_ref_id("site.policy"),
        structural_identity=space.structural_ref_id("struct.policy"),
        seam_class="surviving_carrier_seam",
        touchpoint_marker_identity="tp.policy",
        touchpoint_structural_identity=space.structural_ref_id("tp.struct.policy"),
        subqueue_marker_identity="sq.policy",
        subqueue_structural_identity=space.structural_ref_id("sq.struct.policy"),
        policy_signal_count=1,
        coverage_count=1,
        diagnostic_count=0,
        object_ids=(),
    )
    diagnostic_touchsite = invariant_graph.InvariantTouchsiteProjection(
        object_id=space.touchsite_id("WS-SYNTH-TS-DIAG"),
        touchpoint_id=touchpoint_diag,
        subqueue_id=subqueue_diag,
        title="diagnostic touchsite",
        status="in_progress",
        rel_path="src/gabion/sample.py",
        qualname="diagnostic_touchsite",
        boundary_name="diagnostic_touchsite",
        line=20,
        column=1,
        node_kind="marker_callsite",
        site_identity=space.site_ref_id("site.diagnostic"),
        structural_identity=space.structural_ref_id("struct.diagnostic"),
        seam_class="surviving_carrier_seam",
        touchpoint_marker_identity="tp.diagnostic",
        touchpoint_structural_identity=space.structural_ref_id("tp.struct.diagnostic"),
        subqueue_marker_identity="sq.diagnostic",
        subqueue_structural_identity=space.structural_ref_id("sq.struct.diagnostic"),
        policy_signal_count=0,
        coverage_count=1,
        diagnostic_count=1,
        object_ids=(),
    )
    projection = invariant_graph.InvariantWorkstreamProjection(
        object_id=workstream_id,
        title="synthetic workstream",
        status="in_progress",
        site_identity=space.site_ref_id("ws.site"),
        structural_identity=space.structural_ref_id("ws.struct"),
        marker_identity="ws.marker",
        reasoning_summary="synthetic",
        reasoning_control="synthetic.control",
        blocking_dependencies=(),
        object_ids=(),
        doc_ids=(),
        policy_ids=(),
        touchsite_count=2,
        collapsible_touchsite_count=0,
        surviving_touchsite_count=2,
        policy_signal_count=1,
        coverage_count=2,
        diagnostic_count=1,
        subqueues=_stream_from_items(
            (
                invariant_graph.InvariantSubqueueProjection(
                    object_id=subqueue_policy,
                    title="policy subqueue",
                    status="in_progress",
                    site_identity=space.site_ref_id("sq.site.policy"),
                    structural_identity=space.structural_ref_id("sq.struct.policy.root"),
                    marker_identity="sq.policy.marker",
                    reasoning_summary="policy",
                    reasoning_control="policy",
                    blocking_dependencies=(),
                    object_ids=(),
                    touchpoint_ids=(touchpoint_policy,),
                    touchsite_count=1,
                    collapsible_touchsite_count=0,
                    surviving_touchsite_count=1,
                    policy_signal_count=1,
                    coverage_count=1,
                    diagnostic_count=0,
                ),
                invariant_graph.InvariantSubqueueProjection(
                    object_id=subqueue_diag,
                    title="diagnostic subqueue",
                    status="in_progress",
                    site_identity=space.site_ref_id("sq.site.diagnostic"),
                    structural_identity=space.structural_ref_id("sq.struct.diagnostic.root"),
                    marker_identity="sq.diagnostic.marker",
                    reasoning_summary="diagnostic",
                    reasoning_control="diagnostic",
                    blocking_dependencies=(),
                    object_ids=(),
                    touchpoint_ids=(touchpoint_diag,),
                    touchsite_count=1,
                    collapsible_touchsite_count=0,
                    surviving_touchsite_count=1,
                    policy_signal_count=0,
                    coverage_count=1,
                    diagnostic_count=1,
                ),
            )
        ),
        touchpoints=_stream_from_items(
            (
                invariant_graph.InvariantTouchpointProjection(
                    object_id=touchpoint_policy,
                    subqueue_id=subqueue_policy,
                    title="policy touchpoint",
                    status="in_progress",
                    rel_path="src/gabion/sample.py",
                    site_identity=space.site_ref_id("tp.site.policy"),
                    structural_identity=space.structural_ref_id("tp.struct.policy.root"),
                    marker_identity="tp.policy.marker",
                    reasoning_summary="policy",
                    reasoning_control="policy",
                    blocking_dependencies=(),
                    object_ids=(),
                    touchsite_count=1,
                    collapsible_touchsite_count=0,
                    surviving_touchsite_count=1,
                    policy_signal_count=1,
                    coverage_count=1,
                    diagnostic_count=0,
                    touchsites=_stream_from_items((policy_touchsite,)),
                ),
                invariant_graph.InvariantTouchpointProjection(
                    object_id=touchpoint_diag,
                    subqueue_id=subqueue_diag,
                    title="diagnostic touchpoint",
                    status="in_progress",
                    rel_path="src/gabion/sample.py",
                    site_identity=space.site_ref_id("tp.site.diagnostic"),
                    structural_identity=space.structural_ref_id("tp.struct.diagnostic.root"),
                    marker_identity="tp.diagnostic.marker",
                    reasoning_summary="diagnostic",
                    reasoning_control="diagnostic",
                    blocking_dependencies=(),
                    object_ids=(),
                    touchsite_count=1,
                    collapsible_touchsite_count=0,
                    surviving_touchsite_count=1,
                    policy_signal_count=0,
                    coverage_count=1,
                    diagnostic_count=1,
                    touchsites=_stream_from_items((diagnostic_touchsite,)),
                ),
            )
        ),
    )

    health_summary = projection.health_summary()
    assert health_summary.ready_touchsite_count == 0
    assert health_summary.coverage_gap_touchsite_count == 0
    assert health_summary.policy_blocked_touchsite_count == 1
    assert health_summary.diagnostic_blocked_touchsite_count == 1
    assert projection.dominant_blocker_class() == "diagnostic_blocked"
    assert projection.recommended_remediation_family() == "diagnostic_blocked"
    assert projection.recommended_policy_blocked_cut() is not None
    assert projection.recommended_policy_blocked_cut().object_id.wire() == (
        "WS-SYNTH-TP-POLICY"
    )
    assert projection.recommended_diagnostic_blocked_cut() is not None
    assert projection.recommended_diagnostic_blocked_cut().object_id.wire() == (
        "WS-SYNTH-TP-DIAG"
    )
    payload = projection.as_payload()
    assert payload["next_actions"]["dominant_blocker_class"] == "diagnostic_blocked"
    assert payload["next_actions"]["recommended_remediation_family"] == (
        "diagnostic_blocked"
    )
    assert payload["next_actions"]["recommended_policy_blocked_cut"]["object_id"] == (
        "WS-SYNTH-TP-POLICY"
    )
    assert payload["next_actions"]["recommended_diagnostic_blocked_cut"]["object_id"] == (
        "WS-SYNTH-TP-DIAG"
    )
    assert payload["next_actions"]["remediation_lanes"][0]["remediation_family"] == (
        "diagnostic_blocked"
    )
    assert payload["next_actions"]["remediation_lanes"][0]["best_cut"]["object_id"] == (
        "WS-SYNTH-TP-DIAG"
    )
    assert payload["next_actions"]["remediation_lanes"][1]["remediation_family"] == (
        "policy_blocked"
    )
    assert payload["next_actions"]["remediation_lanes"][1]["best_cut"]["object_id"] == (
        "WS-SYNTH-TP-POLICY"
    )


def test_workstream_projection_uses_ranking_signal_score_to_break_diagnostic_ties() -> None:
    space = PolicyQueueIdentitySpace()
    workstream_id = space.workstream_id("WS-RANK")
    subqueue_id = space.subqueue_id("WS-RANK-SQ-001")
    touchpoint_low = space.touchpoint_id("WS-RANK-TP-LOW")
    touchpoint_high = space.touchpoint_id("WS-RANK-TP-HIGH")
    low_touchsite = invariant_graph.InvariantTouchsiteProjection(
        object_id=space.touchsite_id("WS-RANK-TS-LOW"),
        touchpoint_id=touchpoint_low,
        subqueue_id=subqueue_id,
        title="low pressure touchsite",
        status="in_progress",
        rel_path="src/gabion/sample.py",
        qualname="low_pressure",
        boundary_name="low_pressure",
        line=10,
        column=1,
        node_kind="marker_callsite",
        site_identity=space.site_ref_id("site.low"),
        structural_identity=space.structural_ref_id("struct.low"),
        seam_class="surviving_carrier_seam",
        touchpoint_marker_identity="tp.low",
        touchpoint_structural_identity=space.structural_ref_id("tp.struct.low"),
        subqueue_marker_identity="sq.rank",
        subqueue_structural_identity=space.structural_ref_id("sq.struct.rank"),
        policy_signal_count=0,
        coverage_count=1,
        diagnostic_count=1,
        object_ids=(),
    )
    high_touchsite = invariant_graph.InvariantTouchsiteProjection(
        object_id=space.touchsite_id("WS-RANK-TS-HIGH"),
        touchpoint_id=touchpoint_high,
        subqueue_id=subqueue_id,
        title="high pressure touchsite",
        status="in_progress",
        rel_path="src/gabion/sample.py",
        qualname="high_pressure",
        boundary_name="high_pressure",
        line=20,
        column=1,
        node_kind="marker_callsite",
        site_identity=space.site_ref_id("site.high"),
        structural_identity=space.structural_ref_id("struct.high"),
        seam_class="surviving_carrier_seam",
        touchpoint_marker_identity="tp.high",
        touchpoint_structural_identity=space.structural_ref_id("tp.struct.high"),
        subqueue_marker_identity="sq.rank",
        subqueue_structural_identity=space.structural_ref_id("sq.struct.rank"),
        policy_signal_count=0,
        coverage_count=1,
        diagnostic_count=1,
        object_ids=(),
    )

    projection = invariant_graph.InvariantWorkstreamProjection(
        object_id=workstream_id,
        title="ranking score workstream",
        status="in_progress",
        site_identity=space.site_ref_id("ws.rank.site"),
        structural_identity=space.structural_ref_id("ws.rank.struct"),
        marker_identity="ws.rank.marker",
        reasoning_summary="ranking",
        reasoning_control="ranking",
        blocking_dependencies=(),
        object_ids=(),
        doc_ids=(),
        policy_ids=(),
        touchsite_count=2,
        collapsible_touchsite_count=0,
        surviving_touchsite_count=2,
        policy_signal_count=0,
        coverage_count=2,
        diagnostic_count=2,
        ranking_signal_count=2,
        ranking_signal_score=7,
        subqueues=_stream_from_items(
            (
                invariant_graph.InvariantSubqueueProjection(
                    object_id=subqueue_id,
                    title="rank subqueue",
                    status="in_progress",
                    site_identity=space.site_ref_id("sq.rank.site"),
                    structural_identity=space.structural_ref_id("sq.rank.struct"),
                    marker_identity="sq.rank.marker",
                    reasoning_summary="rank",
                    reasoning_control="rank",
                    blocking_dependencies=(),
                    object_ids=(),
                    touchpoint_ids=(touchpoint_low, touchpoint_high),
                    touchsite_count=2,
                    collapsible_touchsite_count=0,
                    surviving_touchsite_count=2,
                    policy_signal_count=0,
                    coverage_count=2,
                    diagnostic_count=2,
                    ranking_signal_count=2,
                    ranking_signal_score=7,
                ),
            )
        ),
        touchpoints=_stream_from_items(
            (
                invariant_graph.InvariantTouchpointProjection(
                    object_id=touchpoint_low,
                    subqueue_id=subqueue_id,
                    title="low diagnostic pressure",
                    status="in_progress",
                    rel_path="src/gabion/sample.py",
                    site_identity=space.site_ref_id("tp.low.site"),
                    structural_identity=space.structural_ref_id("tp.low.struct"),
                    marker_identity="tp.low.marker",
                    reasoning_summary="low",
                    reasoning_control="low",
                    blocking_dependencies=(),
                    object_ids=(),
                    touchsite_count=1,
                    collapsible_touchsite_count=0,
                    surviving_touchsite_count=1,
                    policy_signal_count=0,
                    coverage_count=1,
                    diagnostic_count=1,
                    ranking_signal_count=0,
                    ranking_signal_score=0,
                    touchsites=_stream_from_items((low_touchsite,)),
                ),
                invariant_graph.InvariantTouchpointProjection(
                    object_id=touchpoint_high,
                    subqueue_id=subqueue_id,
                    title="high diagnostic pressure",
                    status="in_progress",
                    rel_path="src/gabion/sample.py",
                    site_identity=space.site_ref_id("tp.high.site"),
                    structural_identity=space.structural_ref_id("tp.high.struct"),
                    marker_identity="tp.high.marker",
                    reasoning_summary="high",
                    reasoning_control="high",
                    blocking_dependencies=(),
                    object_ids=(),
                    touchsite_count=1,
                    collapsible_touchsite_count=0,
                    surviving_touchsite_count=1,
                    policy_signal_count=0,
                    coverage_count=1,
                    diagnostic_count=1,
                    ranking_signal_count=2,
                    ranking_signal_score=7,
                    touchsites=_stream_from_items((high_touchsite,)),
                ),
            )
        ),
    )

    ranked_cuts = projection.ranked_touchpoint_cuts()
    assert ranked_cuts[0].object_id.wire() == "WS-RANK-TP-HIGH"
    assert ranked_cuts[0].ranking_signal_score == 7
    assert projection.recommended_diagnostic_blocked_cut() is not None
    assert projection.recommended_diagnostic_blocked_cut().object_id.wire() == (
        "WS-RANK-TP-HIGH"
    )


def test_workstream_projection_prefers_actionable_coverage_gap_over_counterfactual_blocked_cut() -> None:
    space = PolicyQueueIdentitySpace()
    workstream_id = space.workstream_id("WS-CF")
    subqueue_id = space.subqueue_id("WS-CF-SQ-001")
    touchpoint_counterfactual = space.touchpoint_id("WS-CF-TP-CF")
    touchpoint_coverage = space.touchpoint_id("WS-CF-TP-COVER")
    counterfactual_touchsite = invariant_graph.InvariantTouchsiteProjection(
        object_id=space.touchsite_id("WS-CF-TS-CF"),
        touchpoint_id=touchpoint_counterfactual,
        subqueue_id=subqueue_id,
        title="counterfactual touchsite",
        status="in_progress",
        rel_path="src/gabion/sample.py",
        qualname="counterfactual_touchsite",
        boundary_name="counterfactual_touchsite",
        line=10,
        column=1,
        node_kind="marker_callsite",
        site_identity=space.site_ref_id("site.cf"),
        structural_identity=space.structural_ref_id("struct.cf"),
        seam_class="surviving_carrier_seam",
        touchpoint_marker_identity="tp.cf",
        touchpoint_structural_identity=space.structural_ref_id("tp.struct.cf"),
        subqueue_marker_identity="sq.cf",
        subqueue_structural_identity=space.structural_ref_id("sq.struct.cf"),
        policy_signal_count=0,
        coverage_count=1,
        diagnostic_count=0,
        object_ids=(),
    )
    coverage_touchsite = invariant_graph.InvariantTouchsiteProjection(
        object_id=space.touchsite_id("WS-CF-TS-COVER"),
        touchpoint_id=touchpoint_coverage,
        subqueue_id=subqueue_id,
        title="coverage touchsite",
        status="in_progress",
        rel_path="src/gabion/sample.py",
        qualname="coverage_touchsite",
        boundary_name="coverage_touchsite",
        line=20,
        column=1,
        node_kind="marker_callsite",
        site_identity=space.site_ref_id("site.cover"),
        structural_identity=space.structural_ref_id("struct.cover"),
        seam_class="surviving_carrier_seam",
        touchpoint_marker_identity="tp.cover",
        touchpoint_structural_identity=space.structural_ref_id("tp.struct.cover"),
        subqueue_marker_identity="sq.cf",
        subqueue_structural_identity=space.structural_ref_id("sq.struct.cf"),
        policy_signal_count=0,
        coverage_count=0,
        diagnostic_count=0,
        object_ids=(),
    )
    projection = invariant_graph.InvariantWorkstreamProjection(
        object_id=workstream_id,
        title="counterfactual workstream",
        status="in_progress",
        site_identity=space.site_ref_id("ws.cf.site"),
        structural_identity=space.structural_ref_id("ws.cf.struct"),
        marker_identity="ws.cf.marker",
        reasoning_summary="counterfactual",
        reasoning_control="counterfactual",
        blocking_dependencies=(),
        object_ids=(),
        doc_ids=(),
        policy_ids=(),
        touchsite_count=2,
        collapsible_touchsite_count=0,
        surviving_touchsite_count=2,
        policy_signal_count=0,
        coverage_count=1,
        diagnostic_count=0,
        subqueues=_stream_from_items(
            (
                invariant_graph.InvariantSubqueueProjection(
                    object_id=subqueue_id,
                    title="counterfactual subqueue",
                    status="in_progress",
                    site_identity=space.site_ref_id("sq.cf.site"),
                    structural_identity=space.structural_ref_id("sq.cf.struct"),
                    marker_identity="sq.cf.marker",
                    reasoning_summary="counterfactual",
                    reasoning_control="counterfactual",
                    blocking_dependencies=(),
                    object_ids=(),
                    touchpoint_ids=(touchpoint_counterfactual, touchpoint_coverage),
                    touchsite_count=2,
                    collapsible_touchsite_count=0,
                    surviving_touchsite_count=2,
                    policy_signal_count=0,
                    coverage_count=1,
                    diagnostic_count=0,
                    counterfactual_action_count=1,
                    viable_counterfactual_action_count=0,
                    blocked_counterfactual_action_count=1,
                ),
            )
        ),
        touchpoints=_stream_from_items(
            (
                invariant_graph.InvariantTouchpointProjection(
                    object_id=touchpoint_counterfactual,
                    subqueue_id=subqueue_id,
                    title="counterfactual touchpoint",
                    status="in_progress",
                    rel_path="src/gabion/sample.py",
                    site_identity=space.site_ref_id("tp.cf.site"),
                    structural_identity=space.structural_ref_id("tp.cf.struct"),
                    marker_identity="tp.cf.marker",
                    reasoning_summary="counterfactual",
                    reasoning_control="counterfactual",
                    blocking_dependencies=(),
                    object_ids=(),
                    touchsite_count=1,
                    collapsible_touchsite_count=0,
                    surviving_touchsite_count=1,
                    policy_signal_count=0,
                    coverage_count=1,
                    diagnostic_count=0,
                    touchsites=_stream_from_items((counterfactual_touchsite,)),
                    ranking_signal_count=1,
                    ranking_signal_score=8,
                    counterfactual_action_count=1,
                    viable_counterfactual_action_count=0,
                    blocked_counterfactual_action_count=1,
                ),
                invariant_graph.InvariantTouchpointProjection(
                    object_id=touchpoint_coverage,
                    subqueue_id=subqueue_id,
                    title="coverage touchpoint",
                    status="in_progress",
                    rel_path="src/gabion/sample.py",
                    site_identity=space.site_ref_id("tp.cover.site"),
                    structural_identity=space.structural_ref_id("tp.cover.struct"),
                    marker_identity="tp.cover.marker",
                    reasoning_summary="coverage",
                    reasoning_control="coverage",
                    blocking_dependencies=(),
                    object_ids=(),
                    touchsite_count=1,
                    collapsible_touchsite_count=0,
                    surviving_touchsite_count=1,
                    policy_signal_count=0,
                    coverage_count=0,
                    diagnostic_count=0,
                    touchsites=_stream_from_items((coverage_touchsite,)),
                ),
            )
        ),
        counterfactual_action_count=1,
        viable_counterfactual_action_count=0,
        blocked_counterfactual_action_count=1,
    )

    ranked_cuts = projection.ranked_touchpoint_cuts()
    by_id = {item.object_id.wire(): item for item in ranked_cuts}

    assert by_id["WS-CF-TP-CF"].readiness_class == "counterfactual_blocked"
    assert projection.recommended_counterfactual_blocked_cut() is not None
    assert projection.recommended_counterfactual_blocked_cut().object_id.wire() == (
        "WS-CF-TP-CF"
    )
    assert projection.recommended_cut() is not None
    assert projection.recommended_cut().object_id.wire() == "WS-CF-TP-COVER"
    assert projection.recommended_remediation_family() == "coverage_gap"


def test_build_invariant_graph_joins_declared_counterfactual_actions_via_registry_injection(
    tmp_path: Path,
) -> None:
    root = write_minimal_invariant_repo(tmp_path)
    base_registry = synthetic_connectivity_workstream_registries()[0]
    base_touchpoint = base_registry.touchpoints[0]
    injected_registry = replace(
        base_registry,
        touchpoints=(
            replace(
                base_touchpoint,
                declared_counterfactual_actions=(
                    RegisteredCounterfactualActionDefinition(
                        action_id="CSA-IDR-TP-T01-ACT-001",
                        title="Retire Synthetic IDR touchpoint",
                        action_kind="retire_boundary",
                        target_boundary_name="Synthetic IDR touchpoint",
                        predicted_readiness_class="policy_blocked",
                        predicted_touchsite_delta=-1,
                        predicted_surviving_touchsite_delta=-1,
                        predicted_policy_signal_delta=1,
                        score=9,
                        rationale="Synthetic touchpoint is blocked by a declared policy counterfactual.",
                        object_ids=("CSA-IDR-TP-T01",),
                    ),
                ),
            ),
        ),
    )

    graph = invariant_graph.build_invariant_graph(
        root,
        declared_registries=(injected_registry,),
    )
    payload = graph.as_payload()

    assert payload["counts"]["node_kind_counts"]["counterfactual_action"] == 1
    traced = invariant_graph.trace_nodes(graph, "CSA-IDR-TP-T01-ACT-001")
    assert any(node.node_kind == "counterfactual_action" for node in traced)

    workstreams = invariant_graph.build_invariant_workstreams(graph, root=root)
    workstream_payload = workstreams.as_payload()["workstreams"][0]

    assert workstream_payload["next_actions"]["recommended_cut"]["object_id"] == (
        "CSA-IDR-TP-T01"
    )
    assert workstream_payload["touchpoints"][0]["counterfactual_action_count"] == 1
    assert workstream_payload["touchpoints"][0]["blocked_counterfactual_action_count"] == 1
    assert workstream_payload["touchpoints"][0]["ranking_signal_count"] == 1
    assert workstream_payload["next_actions"]["ranked_touchpoint_cuts"][0][
        "counterfactual_action_count"
    ] == 1


def test_repo_diagnostic_lane_attributes_candidate_owner_from_exact_path() -> None:
    space = PolicyQueueIdentitySpace()
    workstream_id = space.workstream_id("WS-OWNER")
    subqueue_id = space.subqueue_id("WS-OWNER-SQ-001")
    touchpoint_id = space.touchpoint_id("WS-OWNER-TP-001")
    touchsite = invariant_graph.InvariantTouchsiteProjection(
        object_id=space.touchsite_id("WS-OWNER-TS-001"),
        touchpoint_id=touchpoint_id,
        subqueue_id=subqueue_id,
        title="owner touchsite",
        status="in_progress",
        rel_path="src/gabion/sample_owner.py",
        qualname="owner_touchsite",
        boundary_name="owner_touchsite",
        line=12,
        column=1,
        node_kind="marker_callsite",
        site_identity=space.site_ref_id("site.owner"),
        structural_identity=space.structural_ref_id("struct.owner"),
        seam_class="surviving_carrier_seam",
        touchpoint_marker_identity="tp.owner",
        touchpoint_structural_identity=space.structural_ref_id("tp.struct.owner"),
        subqueue_marker_identity="sq.owner",
        subqueue_structural_identity=space.structural_ref_id("sq.struct.owner"),
        policy_signal_count=0,
        coverage_count=0,
        diagnostic_count=0,
        object_ids=(),
    )
    workstream = invariant_graph.InvariantWorkstreamProjection(
        object_id=workstream_id,
        title="owner workstream",
        status="in_progress",
        site_identity=space.site_ref_id("ws.owner.site"),
        structural_identity=space.structural_ref_id("ws.owner.struct"),
        marker_identity="ws.owner.marker",
        reasoning_summary="owner",
        reasoning_control="owner.control",
        blocking_dependencies=(),
        object_ids=(),
        doc_ids=(),
        policy_ids=(),
        touchsite_count=1,
        collapsible_touchsite_count=0,
        surviving_touchsite_count=1,
        policy_signal_count=0,
        coverage_count=0,
        diagnostic_count=0,
        subqueues=_stream_from_items(
            (
                invariant_graph.InvariantSubqueueProjection(
                    object_id=subqueue_id,
                    title="owner subqueue",
                    status="in_progress",
                    site_identity=space.site_ref_id("sq.owner.site"),
                    structural_identity=space.structural_ref_id("sq.owner.struct"),
                    marker_identity="sq.owner.marker",
                    reasoning_summary="owner",
                    reasoning_control="owner",
                    blocking_dependencies=(),
                    object_ids=(),
                    touchpoint_ids=(touchpoint_id,),
                    touchsite_count=1,
                    collapsible_touchsite_count=0,
                    surviving_touchsite_count=1,
                    policy_signal_count=0,
                    coverage_count=0,
                    diagnostic_count=0,
                ),
            )
        ),
        touchpoints=_stream_from_items(
            (
                invariant_graph.InvariantTouchpointProjection(
                    object_id=touchpoint_id,
                    subqueue_id=subqueue_id,
                    title="owner touchpoint",
                    status="in_progress",
                    rel_path="src/gabion/sample_owner.py",
                    site_identity=space.site_ref_id("tp.owner.site"),
                    structural_identity=space.structural_ref_id("tp.owner.struct"),
                    marker_identity="tp.owner.marker",
                    reasoning_summary="owner",
                    reasoning_control="owner",
                    blocking_dependencies=(),
                    object_ids=(),
                    touchsite_count=1,
                    collapsible_touchsite_count=0,
                    surviving_touchsite_count=1,
                    policy_signal_count=0,
                    coverage_count=0,
                    diagnostic_count=0,
                    touchsites=_stream_from_items((touchsite,)),
                ),
            )
        ),
    )
    node = invariant_graph.InvariantGraphNode(
        node_id="policy_signal:owner",
        node_kind="policy_signal",
        title="grade:GMP-SYNTH",
        marker_name="never",
        marker_kind="policy_signal",
        marker_id="policy-signal-owner",
        site_identity="site.synthetic.owner",
        structural_identity="struct.synthetic.owner",
        object_ids=(),
        doc_ids=(),
        policy_ids=("GMP-SYNTH",),
        invariant_ids=(),
        reasoning_summary="owner signal",
        reasoning_control="owner.signal",
        blocking_dependencies=(),
        rel_path="src/gabion/sample_owner.py",
        qualname="gabion.sample_owner.emit_signal",
        line=12,
        column=3,
        ast_node_kind="Call",
        seam_class="policy_signal",
        source_marker_node_id="policy_signal:owner",
        status_hint="warning",
    )
    diagnostics = (
        invariant_graph.InvariantGraphDiagnostic(
            diagnostic_id="diag-owner",
            severity="warning",
            code="unmatched_policy_signal",
            node_id=node.node_id,
            raw_dependency="",
            message="grade:GMP-SYNTH did not resolve to an owned workstream",
        ),
    )
    projection = invariant_graph.InvariantWorkstreamsProjection(
        root=str(REPO_ROOT),
        generated_at_utc="2026-03-13T00:00:00+00:00",
        workstreams=_stream_from_items((workstream,)),
        diagnostics=diagnostics,
        node_lookup={node.node_id: node},
    )

    lane = projection.repo_diagnostic_lanes()[0]
    assert lane.candidate_owner_status == "exact_path_owner"
    assert lane.candidate_owner_object_id == "WS-OWNER"
    assert lane.candidate_owner_object_ids == ("WS-OWNER",)
    assert lane.candidate_owner_seed_path == "src/gabion"
    assert lane.candidate_owner_seed_object_id == "WS-SEED:gabion"
    assert lane.candidate_owner_options == (
        invariant_graph.InvariantOwnerCandidateOption(
            resolution_kind="attach_existing_owner",
            owner_status="exact_path_owner",
            object_id="WS-OWNER",
            score=300,
            rationale="exact_path_match",
            score_components=(
                invariant_graph.InvariantScoreComponent(
                    kind="attach_existing_owner_base",
                    score=200,
                    rationale="attach_existing_owner",
                ),
                invariant_graph.InvariantScoreComponent(
                    kind="exact_path_bonus",
                    score=100,
                    rationale="exact_path_match",
                ),
            ),
            selection_rank=1,
            opportunity_cost_score=0,
            opportunity_cost_reason="frontier",
            opportunity_cost_components=(),
        ),
        invariant_graph.InvariantOwnerCandidateOption(
            resolution_kind="seed_new_owner",
            owner_status="source_family_seed_owner",
            object_id="WS-SEED:gabion",
            score=100,
            rationale="source_family_seed",
            score_components=(
                invariant_graph.InvariantScoreComponent(
                    kind="seed_new_owner_base",
                    score=100,
                    rationale="source_family_seed",
                ),
            ),
            selection_rank=2,
            opportunity_cost_score=200,
            opportunity_cost_reason="exact_path_match->source_family_seed",
            opportunity_cost_components=(
                invariant_graph.InvariantScoreComponent(
                    kind="attach_existing_owner_base",
                    score=200,
                    rationale="attach_existing_owner",
                ),
                invariant_graph.InvariantScoreComponent(
                    kind="exact_path_bonus",
                    score=100,
                    rationale="exact_path_match",
                ),
                invariant_graph.InvariantScoreComponent(
                    kind="runner_up_offset:seed_new_owner_base",
                    score=-100,
                    rationale="source_family_seed",
                ),
            ),
        ),
    )
    assert lane.recommended_action == "attach_policy_signals_to_candidate_owner"
    assert lane.runner_up_candidate_owner_option == invariant_graph.InvariantOwnerCandidateOption(
        resolution_kind="seed_new_owner",
        owner_status="source_family_seed_owner",
        object_id="WS-SEED:gabion",
        score=100,
        rationale="source_family_seed",
        score_components=(
            invariant_graph.InvariantScoreComponent(
                kind="seed_new_owner_base",
                score=100,
                rationale="source_family_seed",
            ),
        ),
        selection_rank=2,
        opportunity_cost_score=200,
        opportunity_cost_reason="exact_path_match->source_family_seed",
        opportunity_cost_components=(
            invariant_graph.InvariantScoreComponent(
                kind="attach_existing_owner_base",
                score=200,
                rationale="attach_existing_owner",
            ),
            invariant_graph.InvariantScoreComponent(
                kind="exact_path_bonus",
                score=100,
                rationale="exact_path_match",
            ),
            invariant_graph.InvariantScoreComponent(
                kind="runner_up_offset:seed_new_owner_base",
                score=-100,
                rationale="source_family_seed",
            ),
        ),
    )
    assert lane.candidate_owner_choice_margin_score == 200
    assert (
        lane.candidate_owner_choice_margin_reason
        == "exact_path_match->source_family_seed"
    )
    assert lane.candidate_owner_choice_margin_components == (
        invariant_graph.InvariantScoreComponent(
            kind="attach_existing_owner_base",
            score=200,
            rationale="attach_existing_owner",
        ),
        invariant_graph.InvariantScoreComponent(
            kind="exact_path_bonus",
            score=100,
            rationale="exact_path_match",
        ),
        invariant_graph.InvariantScoreComponent(
            kind="runner_up_offset:seed_new_owner_base",
            score=-100,
            rationale="source_family_seed",
        ),
    )
    followup = projection.recommended_repo_followup()
    assert followup is not None
    assert followup.owner_object_id == "WS-OWNER"
    assert followup.owner_seed_path == "src/gabion"
    assert followup.owner_seed_object_id == "WS-SEED:gabion"
    assert followup.owner_option_tradeoff_score == 200
    assert followup.owner_option_tradeoff_reason == "exact_path_match->source_family_seed"
    assert followup.owner_option_tradeoff_components == (
        invariant_graph.InvariantScoreComponent(
            kind="attach_existing_owner_base",
            score=200,
            rationale="attach_existing_owner",
        ),
        invariant_graph.InvariantScoreComponent(
            kind="exact_path_bonus",
            score=100,
            rationale="exact_path_match",
        ),
        invariant_graph.InvariantScoreComponent(
            kind="runner_up_offset:seed_new_owner_base",
            score=-100,
            rationale="source_family_seed",
        ),
    )
    assert followup.owner_choice_margin_components == (
        invariant_graph.InvariantScoreComponent(
            kind="attach_existing_owner_base",
            score=200,
            rationale="attach_existing_owner",
        ),
        invariant_graph.InvariantScoreComponent(
            kind="exact_path_bonus",
            score=100,
            rationale="exact_path_match",
        ),
        invariant_graph.InvariantScoreComponent(
            kind="runner_up_offset:seed_new_owner_base",
            score=-100,
            rationale="source_family_seed",
        ),
    )


def test_repo_diagnostic_lane_ranks_structural_proximity_owner_over_seed() -> None:
    space = PolicyQueueIdentitySpace()
    workstream_id = space.workstream_id("WS-PROX")
    subqueue_id = space.subqueue_id("WS-PROX-SQ-001")
    touchpoint_id = space.touchpoint_id("WS-PROX-TP-001")
    touchsite = invariant_graph.InvariantTouchsiteProjection(
        object_id=space.touchsite_id("WS-PROX-TS-001"),
        touchpoint_id=touchpoint_id,
        subqueue_id=subqueue_id,
        title="proximity touchsite",
        status="in_progress",
        rel_path="src/gabion/analysis/dataflow/engine/report_engine.py",
        qualname="proximity_touchsite",
        boundary_name="proximity_touchsite",
        line=12,
        column=1,
        node_kind="marker_callsite",
        site_identity=space.site_ref_id("site.prox"),
        structural_identity=space.structural_ref_id("struct.prox"),
        seam_class="surviving_carrier_seam",
        touchpoint_marker_identity="tp.prox",
        touchpoint_structural_identity=space.structural_ref_id("tp.struct.prox"),
        subqueue_marker_identity="sq.prox",
        subqueue_structural_identity=space.structural_ref_id("sq.struct.prox"),
        policy_signal_count=0,
        coverage_count=0,
        diagnostic_count=0,
        object_ids=(),
    )
    workstream = invariant_graph.InvariantWorkstreamProjection(
        object_id=workstream_id,
        title="proximity workstream",
        status="in_progress",
        site_identity=space.site_ref_id("ws.prox.site"),
        structural_identity=space.structural_ref_id("ws.prox.struct"),
        marker_identity="ws.prox.marker",
        reasoning_summary="prox",
        reasoning_control="prox.control",
        blocking_dependencies=(),
        object_ids=(),
        doc_ids=(),
        policy_ids=(),
        touchsite_count=1,
        collapsible_touchsite_count=0,
        surviving_touchsite_count=1,
        policy_signal_count=0,
        coverage_count=0,
        diagnostic_count=0,
        subqueues=_stream_from_items(
            (
                invariant_graph.InvariantSubqueueProjection(
                    object_id=subqueue_id,
                    title="proximity subqueue",
                    status="in_progress",
                    site_identity=space.site_ref_id("sq.prox.site"),
                    structural_identity=space.structural_ref_id("sq.prox.struct"),
                    marker_identity="sq.prox.marker",
                    reasoning_summary="prox",
                    reasoning_control="prox",
                    blocking_dependencies=(),
                    object_ids=(),
                    touchpoint_ids=(touchpoint_id,),
                    touchsite_count=1,
                    collapsible_touchsite_count=0,
                    surviving_touchsite_count=1,
                    policy_signal_count=0,
                    coverage_count=0,
                    diagnostic_count=0,
                ),
            )
        ),
        touchpoints=_stream_from_items(
            (
                invariant_graph.InvariantTouchpointProjection(
                    object_id=touchpoint_id,
                    subqueue_id=subqueue_id,
                    title="proximity touchpoint",
                    status="in_progress",
                    rel_path="src/gabion/analysis/dataflow/engine/report_engine.py",
                    site_identity=space.site_ref_id("tp.prox.site"),
                    structural_identity=space.structural_ref_id("tp.prox.struct"),
                    marker_identity="tp.prox.marker",
                    reasoning_summary="prox",
                    reasoning_control="prox",
                    blocking_dependencies=(),
                    object_ids=(),
                    touchsite_count=1,
                    collapsible_touchsite_count=0,
                    surviving_touchsite_count=1,
                    policy_signal_count=0,
                    coverage_count=0,
                    diagnostic_count=0,
                    touchsites=_stream_from_items((touchsite,)),
                ),
            )
        ),
    )
    node = invariant_graph.InvariantGraphNode(
        node_id="policy_signal:prox",
        node_kind="policy_signal",
        title="grade:GMP-PROX",
        marker_name="never",
        marker_kind="policy_signal",
        marker_id="policy-signal-prox",
        site_identity="site.synthetic.prox",
        structural_identity="struct.synthetic.prox",
        object_ids=(),
        doc_ids=(),
        policy_ids=("GMP-PROX",),
        invariant_ids=(),
        reasoning_summary="prox signal",
        reasoning_control="prox.signal",
        blocking_dependencies=(),
        rel_path="src/gabion/analysis/dataflow/io/dataflow_reporting.py",
        qualname="gabion.analysis.dataflow.io.dataflow_reporting.emit_signal",
        line=12,
        column=3,
        ast_node_kind="Call",
        seam_class="policy_signal",
        source_marker_node_id="policy_signal:prox",
        status_hint="warning",
    )
    diagnostics = (
        invariant_graph.InvariantGraphDiagnostic(
            diagnostic_id="diag-prox",
            severity="warning",
            code="unmatched_policy_signal",
            node_id=node.node_id,
            raw_dependency="",
            message="grade:GMP-PROX did not resolve to an owned workstream",
        ),
    )
    projection = invariant_graph.InvariantWorkstreamsProjection(
        root=str(REPO_ROOT),
        generated_at_utc="2026-03-13T00:00:00+00:00",
        workstreams=_stream_from_items((workstream,)),
        diagnostics=diagnostics,
        node_lookup={node.node_id: node},
    )

    lane = projection.repo_diagnostic_lanes()[0]
    assert lane.candidate_owner_status == "structural_proximity_owner"
    assert lane.candidate_owner_object_id == "WS-PROX"
    assert lane.candidate_owner_object_ids == ("WS-PROX",)
    assert lane.candidate_owner_seed_path == "src/gabion/analysis/dataflow/io"
    assert lane.candidate_owner_seed_object_id == "WS-SEED:gabion.analysis.dataflow.io"
    assert lane.candidate_owner_options == (
        invariant_graph.InvariantOwnerCandidateOption(
            resolution_kind="attach_existing_owner",
            owner_status="structural_proximity_owner",
            object_id="WS-PROX",
            score=160,
            rationale="shared_source_family_prefix:4",
            score_components=(
                invariant_graph.InvariantScoreComponent(
                    kind="attach_existing_owner_base",
                    score=120,
                    rationale="attach_existing_owner",
                ),
                invariant_graph.InvariantScoreComponent(
                    kind="structural_proximity_bonus",
                    score=40,
                    rationale="shared_source_family_prefix:4",
                ),
            ),
            selection_rank=1,
            opportunity_cost_score=0,
            opportunity_cost_reason="frontier",
            opportunity_cost_components=(),
        ),
        invariant_graph.InvariantOwnerCandidateOption(
            resolution_kind="seed_new_owner",
            owner_status="source_family_seed_owner",
            object_id="WS-SEED:gabion.analysis.dataflow.io",
            score=100,
            rationale="source_family_seed",
            score_components=(
                invariant_graph.InvariantScoreComponent(
                    kind="seed_new_owner_base",
                    score=100,
                    rationale="source_family_seed",
                ),
            ),
            selection_rank=2,
            opportunity_cost_score=60,
            opportunity_cost_reason="shared_source_family_prefix:4->source_family_seed",
            opportunity_cost_components=(
                invariant_graph.InvariantScoreComponent(
                    kind="attach_existing_owner_base",
                    score=120,
                    rationale="attach_existing_owner",
                ),
                invariant_graph.InvariantScoreComponent(
                    kind="structural_proximity_bonus",
                    score=40,
                    rationale="shared_source_family_prefix:4",
                ),
                invariant_graph.InvariantScoreComponent(
                    kind="runner_up_offset:seed_new_owner_base",
                    score=-100,
                    rationale="source_family_seed",
                ),
            ),
        ),
    )
    assert lane.recommended_action == "choose_candidate_owner_from_ranked_options"
    assert lane.runner_up_candidate_owner_option == invariant_graph.InvariantOwnerCandidateOption(
        resolution_kind="seed_new_owner",
        owner_status="source_family_seed_owner",
        object_id="WS-SEED:gabion.analysis.dataflow.io",
        score=100,
        rationale="source_family_seed",
        score_components=(
            invariant_graph.InvariantScoreComponent(
                kind="seed_new_owner_base",
                score=100,
                rationale="source_family_seed",
            ),
        ),
        selection_rank=2,
        opportunity_cost_score=60,
        opportunity_cost_reason="shared_source_family_prefix:4->source_family_seed",
        opportunity_cost_components=(
            invariant_graph.InvariantScoreComponent(
                kind="attach_existing_owner_base",
                score=120,
                rationale="attach_existing_owner",
            ),
            invariant_graph.InvariantScoreComponent(
                kind="structural_proximity_bonus",
                score=40,
                rationale="shared_source_family_prefix:4",
            ),
            invariant_graph.InvariantScoreComponent(
                kind="runner_up_offset:seed_new_owner_base",
                score=-100,
                rationale="source_family_seed",
            ),
        ),
    )
    assert lane.candidate_owner_choice_margin_score == 60
    assert (
        lane.candidate_owner_choice_margin_reason
        == "shared_source_family_prefix:4->source_family_seed"
    )
    assert lane.candidate_owner_choice_margin_components == (
        invariant_graph.InvariantScoreComponent(
            kind="attach_existing_owner_base",
            score=120,
            rationale="attach_existing_owner",
        ),
        invariant_graph.InvariantScoreComponent(
            kind="structural_proximity_bonus",
            score=40,
            rationale="shared_source_family_prefix:4",
        ),
        invariant_graph.InvariantScoreComponent(
            kind="runner_up_offset:seed_new_owner_base",
            score=-100,
            rationale="source_family_seed",
        ),
    )
    followup = projection.recommended_repo_followup()
    assert followup is not None
    assert followup.owner_object_id == "WS-PROX"
    assert followup.owner_seed_path == "src/gabion/analysis/dataflow/io"
    assert (
        followup.owner_seed_object_id == "WS-SEED:gabion.analysis.dataflow.io"
    )
    assert followup.recommended_action == "choose_candidate_owner_from_ranked_options"
    assert followup.owner_option_tradeoff_score == 60
    assert (
        followup.owner_option_tradeoff_reason
        == "shared_source_family_prefix:4->source_family_seed"
    )
    assert followup.owner_option_tradeoff_components == (
        invariant_graph.InvariantScoreComponent(
            kind="attach_existing_owner_base",
            score=120,
            rationale="attach_existing_owner",
        ),
        invariant_graph.InvariantScoreComponent(
            kind="structural_proximity_bonus",
            score=40,
            rationale="shared_source_family_prefix:4",
        ),
        invariant_graph.InvariantScoreComponent(
            kind="runner_up_offset:seed_new_owner_base",
            score=-100,
            rationale="source_family_seed",
        ),
    )
    assert followup.owner_choice_margin_components == (
        invariant_graph.InvariantScoreComponent(
            kind="attach_existing_owner_base",
            score=120,
            rationale="attach_existing_owner",
        ),
        invariant_graph.InvariantScoreComponent(
            kind="structural_proximity_bonus",
            score=40,
            rationale="shared_source_family_prefix:4",
        ),
        invariant_graph.InvariantScoreComponent(
            kind="runner_up_offset:seed_new_owner_base",
            score=-100,
            rationale="source_family_seed",
        ),
    )
    assert followup.title == "resolve grade:GMP-PROX ownership via WS-PROX"


def test_ranked_repo_followups_prefers_stronger_owner_resolution_score() -> None:
    space = PolicyQueueIdentitySpace()
    exact_workstream_id = space.workstream_id("WS-OWNER")
    exact_subqueue_id = space.subqueue_id("WS-OWNER-SQ-001")
    exact_touchpoint_id = space.touchpoint_id("WS-OWNER-TP-001")
    exact_touchsite = invariant_graph.InvariantTouchsiteProjection(
        object_id=space.touchsite_id("WS-OWNER-TS-001"),
        touchpoint_id=exact_touchpoint_id,
        subqueue_id=exact_subqueue_id,
        title="owner touchsite",
        status="in_progress",
        rel_path="src/gabion/sample_owner.py",
        qualname="owner_touchsite",
        boundary_name="owner_touchsite",
        line=12,
        column=1,
        node_kind="marker_callsite",
        site_identity=space.site_ref_id("site.owner"),
        structural_identity=space.structural_ref_id("struct.owner"),
        seam_class="surviving_carrier_seam",
        touchpoint_marker_identity="tp.owner",
        touchpoint_structural_identity=space.structural_ref_id("tp.struct.owner"),
        subqueue_marker_identity="sq.owner",
        subqueue_structural_identity=space.structural_ref_id("sq.struct.owner"),
        policy_signal_count=0,
        coverage_count=0,
        diagnostic_count=0,
        object_ids=(),
    )
    exact_workstream = invariant_graph.InvariantWorkstreamProjection(
        object_id=exact_workstream_id,
        title="owner workstream",
        status="in_progress",
        site_identity=space.site_ref_id("ws.owner.site"),
        structural_identity=space.structural_ref_id("ws.owner.struct"),
        marker_identity="ws.owner.marker",
        reasoning_summary="owner",
        reasoning_control="owner.control",
        blocking_dependencies=(),
        object_ids=(),
        doc_ids=(),
        policy_ids=(),
        touchsite_count=1,
        collapsible_touchsite_count=0,
        surviving_touchsite_count=1,
        policy_signal_count=0,
        coverage_count=0,
        diagnostic_count=0,
        subqueues=_stream_from_items(
            (
                invariant_graph.InvariantSubqueueProjection(
                    object_id=exact_subqueue_id,
                    title="owner subqueue",
                    status="in_progress",
                    site_identity=space.site_ref_id("sq.owner.site"),
                    structural_identity=space.structural_ref_id("sq.owner.struct"),
                    marker_identity="sq.owner.marker",
                    reasoning_summary="owner",
                    reasoning_control="owner",
                    blocking_dependencies=(),
                    object_ids=(),
                    touchpoint_ids=(exact_touchpoint_id,),
                    touchsite_count=1,
                    collapsible_touchsite_count=0,
                    surviving_touchsite_count=1,
                    policy_signal_count=0,
                    coverage_count=0,
                    diagnostic_count=0,
                ),
            )
        ),
        touchpoints=_stream_from_items(
            (
                invariant_graph.InvariantTouchpointProjection(
                    object_id=exact_touchpoint_id,
                    subqueue_id=exact_subqueue_id,
                    title="owner touchpoint",
                    status="in_progress",
                    rel_path="src/gabion/sample_owner.py",
                    site_identity=space.site_ref_id("tp.owner.site"),
                    structural_identity=space.structural_ref_id("tp.owner.struct"),
                    marker_identity="tp.owner.marker",
                    reasoning_summary="owner",
                    reasoning_control="owner",
                    blocking_dependencies=(),
                    object_ids=(),
                    touchsite_count=1,
                    collapsible_touchsite_count=0,
                    surviving_touchsite_count=1,
                    policy_signal_count=0,
                    coverage_count=0,
                    diagnostic_count=0,
                    touchsites=_stream_from_items((exact_touchsite,)),
                ),
            )
        ),
    )
    exact_node = invariant_graph.InvariantGraphNode(
        node_id="policy_signal:owner-rank",
        node_kind="policy_signal",
        title="grade:GMP-RANK-EXACT",
        marker_name="never",
        marker_kind="policy_signal",
        marker_id="policy-signal-owner-rank",
        site_identity="site.synthetic.owner.rank",
        structural_identity="struct.synthetic.owner.rank",
        object_ids=(),
        doc_ids=(),
        policy_ids=("GMP-RANK-EXACT",),
        invariant_ids=(),
        reasoning_summary="owner signal",
        reasoning_control="owner.signal",
        blocking_dependencies=(),
        rel_path="src/gabion/sample_owner.py",
        qualname="gabion.sample_owner.emit_signal",
        line=12,
        column=3,
        ast_node_kind="Call",
        seam_class="policy_signal",
        source_marker_node_id="policy_signal:owner-rank",
        status_hint="warning",
    )
    seed_only_node = invariant_graph.InvariantGraphNode(
        node_id="policy_signal:seed-rank",
        node_kind="policy_signal",
        title="grade:GMP-RANK-SEED",
        marker_name="never",
        marker_kind="policy_signal",
        marker_id="policy-signal-seed-rank",
        site_identity="site.synthetic.seed.rank",
        structural_identity="struct.synthetic.seed.rank",
        object_ids=(),
        doc_ids=(),
        policy_ids=("GMP-RANK-SEED",),
        invariant_ids=(),
        reasoning_summary="seed signal",
        reasoning_control="seed.signal",
        blocking_dependencies=(),
        rel_path="src/gabion/analysis/dataflow/io/dataflow_reporting.py",
        qualname="gabion.analysis.dataflow.io.dataflow_reporting.emit_seed_signal",
        line=15,
        column=3,
        ast_node_kind="Call",
        seam_class="policy_signal",
        source_marker_node_id="policy_signal:seed-rank",
        status_hint="warning",
    )
    diagnostics = (
        invariant_graph.InvariantGraphDiagnostic(
            diagnostic_id="diag-owner-rank",
            severity="warning",
            code="unmatched_policy_signal",
            node_id=exact_node.node_id,
            raw_dependency="",
            message="grade:GMP-RANK-EXACT did not resolve to an owned workstream",
        ),
        invariant_graph.InvariantGraphDiagnostic(
            diagnostic_id="diag-seed-rank",
            severity="warning",
            code="unmatched_policy_signal",
            node_id=seed_only_node.node_id,
            raw_dependency="",
            message="grade:GMP-RANK-SEED did not resolve to an owned workstream",
        ),
    )
    projection = invariant_graph.InvariantWorkstreamsProjection(
        root=str(REPO_ROOT),
        generated_at_utc="2026-03-13T00:00:00+00:00",
        workstreams=_stream_from_items((exact_workstream,)),
        diagnostics=diagnostics,
        node_lookup={
            exact_node.node_id: exact_node,
            seed_only_node.node_id: seed_only_node,
        },
    )

    ranked = projection.ranked_repo_followups()
    assert ranked[0].diagnostic_code == "unmatched_policy_signal"
    assert ranked[0].owner_object_id == "WS-OWNER"
    assert ranked[0].owner_resolution_score == 300
    assert ranked[0].owner_resolution_options == (
        invariant_graph.InvariantOwnerCandidateOption(
            resolution_kind="attach_existing_owner",
            owner_status="exact_path_owner",
            object_id="WS-OWNER",
            score=300,
            rationale="exact_path_match",
            score_components=(
                invariant_graph.InvariantScoreComponent(
                    kind="attach_existing_owner_base",
                    score=200,
                    rationale="attach_existing_owner",
                ),
                invariant_graph.InvariantScoreComponent(
                    kind="exact_path_bonus",
                    score=100,
                    rationale="exact_path_match",
                ),
            ),
            selection_rank=1,
            opportunity_cost_score=0,
            opportunity_cost_reason="frontier",
            opportunity_cost_components=(),
        ),
        invariant_graph.InvariantOwnerCandidateOption(
            resolution_kind="seed_new_owner",
            owner_status="source_family_seed_owner",
            object_id="WS-SEED:gabion",
            score=100,
            rationale="source_family_seed",
            score_components=(
                invariant_graph.InvariantScoreComponent(
                    kind="seed_new_owner_base",
                    score=100,
                    rationale="source_family_seed",
                ),
            ),
            selection_rank=2,
            opportunity_cost_score=200,
            opportunity_cost_reason="exact_path_match->source_family_seed",
            opportunity_cost_components=(
                invariant_graph.InvariantScoreComponent(
                    kind="attach_existing_owner_base",
                    score=200,
                    rationale="attach_existing_owner",
                ),
                invariant_graph.InvariantScoreComponent(
                    kind="exact_path_bonus",
                    score=100,
                    rationale="exact_path_match",
                ),
                invariant_graph.InvariantScoreComponent(
                    kind="runner_up_offset:seed_new_owner_base",
                    score=-100,
                    rationale="source_family_seed",
                ),
            ),
        ),
    )
    assert ranked[0].runner_up_owner_object_id == "WS-SEED:gabion"
    assert ranked[0].runner_up_owner_resolution_kind == "seed_new_owner"
    assert ranked[0].runner_up_owner_resolution_score == 100
    assert ranked[0].owner_choice_margin_score == 200
    assert ranked[0].owner_choice_margin_reason == "exact_path_match->source_family_seed"
    assert ranked[0].owner_choice_margin_components == (
        invariant_graph.InvariantScoreComponent(
            kind="attach_existing_owner_base",
            score=200,
            rationale="attach_existing_owner",
        ),
        invariant_graph.InvariantScoreComponent(
            kind="exact_path_bonus",
            score=100,
            rationale="exact_path_match",
        ),
        invariant_graph.InvariantScoreComponent(
            kind="runner_up_offset:seed_new_owner_base",
            score=-100,
            rationale="source_family_seed",
        ),
    )
    assert ranked[0].owner_option_tradeoff_score == 200
    assert (
        ranked[0].owner_option_tradeoff_reason
        == "exact_path_match->source_family_seed"
    )
    assert ranked[0].owner_option_tradeoff_components == (
        invariant_graph.InvariantScoreComponent(
            kind="attach_existing_owner_base",
            score=200,
            rationale="attach_existing_owner",
        ),
        invariant_graph.InvariantScoreComponent(
            kind="exact_path_bonus",
            score=100,
            rationale="exact_path_match",
        ),
        invariant_graph.InvariantScoreComponent(
            kind="runner_up_offset:seed_new_owner_base",
            score=-100,
            rationale="source_family_seed",
        ),
    )
    assert ranked[0].utility_score == 1400
    assert (
        ranked[0].utility_reason
        == "governance_orphan:attach_existing_owner+owner_option_tradeoff:200"
    )
    assert ranked[0].utility_components == (
        invariant_graph.InvariantScoreComponent(
            kind="governance_orphan_base",
            score=900,
            rationale="governance_orphan",
        ),
        invariant_graph.InvariantScoreComponent(
            kind="owner_resolution_bonus",
            score=300,
            rationale="attach_existing_owner",
        ),
        invariant_graph.InvariantScoreComponent(
            kind="owner_option_tradeoff_bonus",
            score=200,
            rationale="exact_path_match->source_family_seed",
        ),
    )
    assert ranked[1].diagnostic_code == "unmatched_policy_signal"
    assert ranked[1].owner_object_id is None
    assert ranked[1].owner_resolution_score == 100
    assert ranked[1].owner_resolution_options == (
        invariant_graph.InvariantOwnerCandidateOption(
            resolution_kind="seed_new_owner",
            owner_status="source_family_seed_owner",
            object_id="WS-SEED:gabion.analysis.dataflow.io",
            score=100,
            rationale="source_family_seed",
            score_components=(
                invariant_graph.InvariantScoreComponent(
                    kind="seed_new_owner_base",
                    score=100,
                    rationale="source_family_seed",
                ),
            ),
            selection_rank=1,
            opportunity_cost_score=0,
            opportunity_cost_reason="frontier",
            opportunity_cost_components=(),
        ),
    )
    assert ranked[1].runner_up_owner_object_id is None
    assert ranked[1].runner_up_owner_resolution_kind is None
    assert ranked[1].runner_up_owner_resolution_score is None
    assert ranked[1].owner_choice_margin_score == 100
    assert ranked[1].owner_choice_margin_reason == "uncontested_best_option"
    assert ranked[1].owner_choice_margin_components == (
        invariant_graph.InvariantScoreComponent(
            kind="seed_new_owner_base",
            score=100,
            rationale="source_family_seed",
        ),
    )
    assert ranked[1].owner_option_tradeoff_score == 100
    assert ranked[1].owner_option_tradeoff_reason == "uncontested_best_option"
    assert ranked[1].owner_option_tradeoff_components == (
        invariant_graph.InvariantScoreComponent(
            kind="seed_new_owner_base",
            score=100,
            rationale="source_family_seed",
        ),
    )
    assert ranked[1].utility_score == 1100
    assert (
        ranked[1].utility_reason
        == "governance_orphan:seed_new_owner+owner_option_tradeoff:100"
    )
    assert ranked[1].utility_components == (
        invariant_graph.InvariantScoreComponent(
            kind="governance_orphan_base",
            score=900,
            rationale="governance_orphan",
        ),
        invariant_graph.InvariantScoreComponent(
            kind="owner_resolution_bonus",
            score=100,
            rationale="seed_new_owner",
        ),
        invariant_graph.InvariantScoreComponent(
            kind="owner_option_tradeoff_bonus",
            score=100,
            rationale="uncontested_best_option",
        ),
    )
    assert projection.recommended_repo_followup() == ranked[0]


def test_runtime_invariant_graph_cli_build_summary_trace_and_blockers(
    tmp_path: Path,
    capsys,
) -> None:
    root = _sample_repo(tmp_path)
    artifact = tmp_path / "artifacts/out/invariant_graph.json"
    workstreams_artifact = tmp_path / "artifacts/out/invariant_workstreams.json"
    ledger_artifact = tmp_path / "artifacts/out/invariant_ledger_projections.json"

    assert (
        invariant_graph_runtime.main(
            [
                "--root",
                str(root),
                "--artifact",
                str(artifact),
                "--workstreams-artifact",
                str(workstreams_artifact),
                "--ledger-artifact",
                str(ledger_artifact),
                "build",
            ],
            declared_registries=_NO_DECLARED_REGISTRIES,
        )
        == 0
    )
    assert artifact.exists()
    assert workstreams_artifact.exists()
    assert ledger_artifact.exists()

    assert (
        invariant_graph_runtime.main(
            ["--root", str(root), "--artifact", str(artifact), "summary"]
        )
        == 0
    )
    summary_output = capsys.readouterr().out
    assert "nodes:" in summary_output
    assert "edges:" in summary_output

    assert (
        invariant_graph_runtime.main(
            ["--root", str(root), "--artifact", str(artifact), "trace", "--id", "OBJ-TODO"]
        )
        == 0
    )
    trace_output = capsys.readouterr().out
    assert "marker_id:" in trace_output
    assert "ownership_chain:" in trace_output

    assert (
        invariant_graph_runtime.main(
            [
                "--artifact",
                str(artifact),
                "blockers",
                "--object-id",
                "PSF-007",
            ]
        )
        == 1
    )
    blocker_output = capsys.readouterr().out
    assert "no blocker chains" in blocker_output


def test_runtime_invariant_graph_cli_blast_radius_flags_impacted_tests(
    tmp_path: Path,
    capsys,
) -> None:
    root = _sample_repo(tmp_path)
    (root / "out").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "audit_reports").mkdir(parents=True, exist_ok=True)
    (root / "out" / "test_evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tests": [
                    {
                        "test_id": "tests/test_sample.py::test_decorated",
                        "file": "tests/test_sample.py",
                        "line": 10,
                        "status": "pass",
                        "evidence": [
                            {
                                "key": {
                                    "site": {
                                        "path": "src/gabion/sample.py",
                                        "qual": "decorated",
                                    }
                                }
                            }
                        ],
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    impact_artifact = root / "artifacts" / "audit_reports" / "impact_selection.json"
    impact_artifact.write_text(
        json.dumps(
            {
                "selection": {
                    "impacted_tests": ["tests/test_sample.py::test_decorated"],
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact = tmp_path / "artifacts/out/invariant_graph.json"
    workstreams_artifact = tmp_path / "artifacts/out/invariant_workstreams.json"
    ledger_artifact = tmp_path / "artifacts/out/invariant_ledger_projections.json"

    assert (
        invariant_graph_runtime.main(
            [
                "--root",
                str(root),
                "--artifact",
                str(artifact),
                "--workstreams-artifact",
                str(workstreams_artifact),
                "--ledger-artifact",
                str(ledger_artifact),
                "build",
            ],
            declared_registries=_NO_DECLARED_REGISTRIES,
        )
        == 0
    )
    assert (
        invariant_graph_runtime.main(
            [
                "--root",
                str(root),
                "--artifact",
                str(artifact),
                "blast-radius",
                "--id",
                "OBJ-TODO",
                "--impact-artifact",
                str(impact_artifact),
            ]
        )
        == 0
    )
    blast_radius_output = capsys.readouterr().out
    assert "tests/test_sample.py::test_decorated [impacted]" in blast_radius_output


def test_runtime_invariant_graph_cli_perf_heat_maps_profile_artifacts(
    tmp_path: Path,
    capsys,
) -> None:
    root = _sample_repo(tmp_path)
    (root / "artifacts" / "audit_reports").mkdir(parents=True, exist_ok=True)
    artifact = tmp_path / "artifacts/out/invariant_graph.json"
    workstreams_artifact = tmp_path / "artifacts/out/invariant_workstreams.json"
    ledger_artifact = tmp_path / "artifacts/out/invariant_ledger_projections.json"
    perf_artifact = root / "artifacts" / "audit_reports" / "perf_profile.json"

    assert (
        invariant_graph_runtime.main(
            [
                "--root",
                str(root),
                "--artifact",
                str(artifact),
                "--workstreams-artifact",
                str(workstreams_artifact),
                "--ledger-artifact",
                str(ledger_artifact),
                "build",
            ],
            declared_registries=_NO_DECLARED_REGISTRIES,
        )
        == 0
    )
    graph = invariant_graph.load_invariant_graph(artifact)
    decorated_node = next(
        node
        for node in graph.nodes
        if node.rel_path == "src/gabion/sample.py" and node.qualname == "decorated"
    )
    perf_artifact.write_text(
        json.dumps(
            {
                "format_version": 1,
                "profiles": [
                    {
                        "profiler": "py-spy",
                        "metric_kind": "wall_samples",
                        "unit": "samples",
                        "samples": [
                            {
                                "artifact_node": {
                                    "wire": "::".join(
                                        (
                                            decorated_node.site_identity,
                                            decorated_node.structural_identity,
                                        )
                                    ),
                                    "site_identity": decorated_node.site_identity,
                                    "structural_identity": (
                                        decorated_node.structural_identity
                                    ),
                                    "rel_path": decorated_node.rel_path,
                                    "qualname": decorated_node.qualname,
                                    "line": decorated_node.line,
                                    "column": decorated_node.column,
                                },
                                "inclusive_value": 31,
                            },
                            {
                                "rel_path": "src/gabion/missing.py",
                                "qualname": "missing",
                                "line": 1,
                                "inclusive_value": 99,
                            },
                        ],
                    },
                    {
                        "profiler": "cProfile",
                        "metric_kind": "cpu_time",
                        "unit": "seconds",
                        "samples": [
                            {
                                "artifact_node": {
                                    "wire": "::".join(
                                        (
                                            decorated_node.site_identity,
                                            decorated_node.structural_identity,
                                        )
                                    ),
                                    "site_identity": decorated_node.site_identity,
                                    "structural_identity": (
                                        decorated_node.structural_identity
                                    ),
                                    "rel_path": decorated_node.rel_path,
                                    "qualname": decorated_node.qualname,
                                    "line": decorated_node.line,
                                    "column": decorated_node.column,
                                },
                                "inclusive_value": 0.42,
                            }
                        ],
                    },
                    {
                        "profiler": "pyinstrument",
                        "metric_kind": "wall_time",
                        "unit": "seconds",
                        "samples": [
                            {
                                "artifact_node": {
                                    "wire": "::".join(
                                        (
                                            decorated_node.site_identity,
                                            decorated_node.structural_identity,
                                        )
                                    ),
                                    "site_identity": decorated_node.site_identity,
                                    "structural_identity": (
                                        decorated_node.structural_identity
                                    ),
                                    "rel_path": decorated_node.rel_path,
                                    "qualname": decorated_node.qualname,
                                    "line": decorated_node.line,
                                    "column": decorated_node.column,
                                },
                                "inclusive_value": 0.99,
                            }
                        ],
                    },
                    {
                        "profiler": "memray",
                        "metric_kind": "allocated_bytes",
                        "unit": "bytes",
                        "samples": [
                            {
                                "rel_path": "src/gabion/sample.py",
                                "qualname": "decorated",
                                "line": decorated_node.line,
                                "inclusive_value": 2048,
                            }
                        ],
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    assert (
        invariant_graph_runtime.main(
            [
                "--root",
                str(root),
                "--artifact",
                str(artifact),
                "perf-heat",
                "--id",
                "OBJ-TODO",
                "--perf-artifact",
                str(perf_artifact),
            ]
        )
        == 0
    )
    perf_heat_output = capsys.readouterr().out
    assert "perf_query_overlay:" in perf_heat_output
    assert "doc_ids: sample_doc" in perf_heat_output
    assert "doc_paths: docs/sample.md" in perf_heat_output
    assert "target_symbols: gabion.sample.decorated" in perf_heat_output
    assert "source: dsl_overlay" in perf_heat_output
    assert "matched_profile_observations: 4" in perf_heat_output
    assert "wall_samples:samples total=31" in perf_heat_output
    assert "cpu_time:seconds total=0.42" in perf_heat_output
    assert "wall_time:seconds total=0.99" in perf_heat_output
    assert "allocated_bytes:bytes total=2048" in perf_heat_output
    assert "py-spy :: 31" in perf_heat_output
    assert "cProfile :: 0.42" in perf_heat_output
    assert "pyinstrument :: 0.99" in perf_heat_output
    assert "memray :: 2048" in perf_heat_output
    assert f"src/gabion/sample.py:{decorated_node.line}::decorated" in perf_heat_output
    assert "perf_infimum_buckets:" in perf_heat_output
    assert "- <none>" in perf_heat_output


def test_perf_dsl_overlay_resolves_doc_targets_to_invariant_candidates(
    tmp_path: Path,
) -> None:
    root = _sample_repo(tmp_path)
    graph = invariant_graph.build_invariant_graph(
        root,
        declared_registries=_NO_DECLARED_REGISTRIES,
    )
    traced = invariant_graph.trace_nodes(graph, "OBJ-TODO")
    descendant_ids = tuple(
        invariant_graph_runtime._sorted(
            list(
                {
                    descendant_id
                    for node in traced
                    for descendant_id in invariant_graph_runtime._descendant_ids(
                        graph,
                        node.node_id,
                    )
                }
            )
        )
    )

    overlay = invariant_graph_runtime._resolve_perf_dsl_overlay(
        root=root,
        graph=graph,
        scope_node_ids=descendant_ids,
    )

    assert overlay.doc_ids == ("sample_doc",)
    assert overlay.doc_paths == ("docs/sample.md",)
    assert overlay.target_symbols == ("gabion.sample.decorated",)
    candidate_nodes = {graph.node_by_id()[node_id] for node_id in overlay.candidate_node_ids}
    assert {node.qualname for node in candidate_nodes} == {"decorated"}


def test_perf_dsl_overlay_reuses_shared_doc_selector_for_inferred_targets(
    tmp_path: Path,
) -> None:
    root = _sample_repo(tmp_path)
    _write(
        root / "docs" / "sample.md",
        "\n".join(
            [
                "---",
                "doc_id: sample_doc",
                "---",
                "",
                "Touches `gabion.sample.decorated`.",
            ]
        ),
    )
    graph = invariant_graph.build_invariant_graph(
        root,
        declared_registries=_NO_DECLARED_REGISTRIES,
    )
    traced = invariant_graph.trace_nodes(graph, "OBJ-TODO")
    descendant_ids = tuple(
        invariant_graph_runtime._sorted(
            list(
                {
                    descendant_id
                    for node in traced
                    for descendant_id in invariant_graph_runtime._descendant_ids(
                        graph,
                        node.node_id,
                    )
                }
            )
        )
    )

    overlay = invariant_graph_runtime._resolve_perf_dsl_overlay(
        root=root,
        graph=graph,
        scope_node_ids=descendant_ids,
    )

    assert overlay.doc_ids == ("sample_doc",)
    assert overlay.doc_paths == ("docs/sample.md",)
    assert overlay.target_symbols == ("gabion.sample.decorated",)
    candidate_nodes = {graph.node_by_id()[node_id] for node_id in overlay.candidate_node_ids}
    assert {node.qualname for node in candidate_nodes} == {"decorated"}


def test_perf_dsl_overlay_infers_script_symbols_from_shared_selector(tmp_path: Path) -> None:
    _write(
        tmp_path / "scripts" / "policy" / "policy_check.py",
        "def main():\n    return None\n",
    )
    _write(
        tmp_path / "docs" / "perf.md",
        "\n".join(
            [
                "---",
                "doc_id: perf_doc",
                "---",
                "",
                "Touches `scripts.policy.policy_check.main`.",
            ]
        ),
    )
    graph = invariant_graph.InvariantGraph(
        root=str(tmp_path),
        workstream_root_ids=(),
        nodes=(
            invariant_graph.InvariantGraphNode(
                node_id="script-main",
                node_kind="touchsite",
                title="policy_check main",
                marker_name="todo",
                marker_kind="touchsite",
                marker_id="marker:script-main",
                site_identity="site:script-main",
                structural_identity="struct:script-main",
                object_ids=("CSA-IVL-TS-016",),
                doc_ids=("perf_doc",),
                policy_ids=(),
                invariant_ids=(),
                reasoning_summary="script main",
                reasoning_control="script.main",
                blocking_dependencies=(),
                rel_path="scripts/policy/policy_check.py",
                qualname="main",
                line=1,
                column=1,
                ast_node_kind="FunctionDef",
                seam_class="boundary",
                source_marker_node_id="marker-node:script-main",
                status_hint="open",
            ),
        ),
        edges=(),
        diagnostics=(),
    )

    overlay = invariant_graph_runtime._resolve_perf_dsl_overlay(
        root=tmp_path,
        graph=graph,
        scope_node_ids=("script-main",),
    )

    assert overlay.doc_ids == ("perf_doc",)
    assert overlay.doc_paths == ("docs/perf.md",)
    assert overlay.target_symbols == ("scripts.policy.policy_check.main",)
    assert overlay.candidate_node_ids == ("script-main",)


def test_perf_infimum_buckets_use_meet_over_containment_topology() -> None:
    graph = invariant_graph.InvariantGraph(
        root=str(REPO_ROOT),
        workstream_root_ids=("ROOT",),
        nodes=(
            invariant_graph.InvariantGraphNode(
                node_id="root",
                node_kind="synthetic_work_item",
                title="Root Workstream",
                marker_name="todo",
                marker_kind="todo",
                marker_id="root",
                site_identity="site.root",
                structural_identity="struct.root",
                object_ids=("ROOT",),
                doc_ids=(),
                policy_ids=(),
                invariant_ids=(),
                reasoning_summary="root",
                reasoning_control="root",
                blocking_dependencies=(),
                rel_path="",
                qualname="",
                line=0,
                column=0,
                ast_node_kind="",
                seam_class="",
                source_marker_node_id="",
                status_hint="",
            ),
            invariant_graph.InvariantGraphNode(
                node_id="touchpoint",
                node_kind="synthetic_work_item",
                title="Shared Touchpoint",
                marker_name="todo",
                marker_kind="todo",
                marker_id="touchpoint",
                site_identity="site.touchpoint",
                structural_identity="struct.touchpoint",
                object_ids=("ROOT-TP",),
                doc_ids=(),
                policy_ids=(),
                invariant_ids=(),
                reasoning_summary="touchpoint",
                reasoning_control="touchpoint",
                blocking_dependencies=(),
                rel_path="",
                qualname="",
                line=0,
                column=0,
                ast_node_kind="",
                seam_class="",
                source_marker_node_id="",
                status_hint="",
            ),
            invariant_graph.InvariantGraphNode(
                node_id="leaf-a",
                node_kind="synthetic_touchsite",
                title="Leaf A",
                marker_name="todo",
                marker_kind="todo",
                marker_id="leaf-a",
                site_identity="site.leaf_a",
                structural_identity="struct.leaf_a",
                object_ids=("ROOT-TP-TS-A",),
                doc_ids=(),
                policy_ids=(),
                invariant_ids=(),
                reasoning_summary="leaf-a",
                reasoning_control="leaf-a",
                blocking_dependencies=(),
                rel_path="src/gabion/a.py",
                qualname="a",
                line=10,
                column=1,
                ast_node_kind="function_def",
                seam_class="surviving_carrier_seam",
                source_marker_node_id="",
                status_hint="",
            ),
            invariant_graph.InvariantGraphNode(
                node_id="leaf-b",
                node_kind="synthetic_touchsite",
                title="Leaf B",
                marker_name="todo",
                marker_kind="todo",
                marker_id="leaf-b",
                site_identity="site.leaf_b",
                structural_identity="struct.leaf_b",
                object_ids=("ROOT-TP-TS-B",),
                doc_ids=(),
                policy_ids=(),
                invariant_ids=(),
                reasoning_summary="leaf-b",
                reasoning_control="leaf-b",
                blocking_dependencies=(),
                rel_path="src/gabion/b.py",
                qualname="b",
                line=20,
                column=1,
                ast_node_kind="function_def",
                seam_class="surviving_carrier_seam",
                source_marker_node_id="",
                status_hint="",
            ),
        ),
        edges=(
            invariant_graph.InvariantGraphEdge(
                edge_id="root-touchpoint",
                edge_kind="contains",
                source_id="root",
                target_id="touchpoint",
            ),
            invariant_graph.InvariantGraphEdge(
                edge_id="touchpoint-a",
                edge_kind="contains",
                source_id="touchpoint",
                target_id="leaf-a",
            ),
            invariant_graph.InvariantGraphEdge(
                edge_id="touchpoint-b",
                edge_kind="contains",
                source_id="touchpoint",
                target_id="leaf-b",
            ),
        ),
        diagnostics=(),
    )
    matched = (
        invariant_graph_runtime._MatchedProfileObservation(
            node_id="leaf-a",
            profiler="cProfile",
            metric_kind="cpu_time",
            unit="seconds",
            rel_path="src/gabion/a.py",
            qualname="a",
            line=10,
            title="Leaf A",
            inclusive_value=3.0,
        ),
        invariant_graph_runtime._MatchedProfileObservation(
            node_id="leaf-b",
            profiler="cProfile",
            metric_kind="cpu_time",
            unit="seconds",
            rel_path="src/gabion/b.py",
            qualname="b",
            line=20,
            title="Leaf B",
            inclusive_value=5.0,
        ),
    )

    buckets = invariant_graph_runtime._perf_infimum_buckets(
        graph=graph,
        descendant_ids=("root", "touchpoint", "leaf-a", "leaf-b"),
        matched=matched,
    )

    assert buckets
    assert buckets[0].node_id == "touchpoint"
    assert buckets[0].is_global_infimum is True
    assert buckets[0].is_virtual_intersection is True
    assert buckets[0].matched_leaf_node_count == 2
    assert buckets[0].total_inclusive_value == 8.0


def test_build_invariant_graph_joins_policy_signals_and_test_coverage(
    tmp_path: Path,
) -> None:
    root = _sample_repo(tmp_path)
    decorated_line = _sample_decorated_line(root)
    (root / "artifacts" / "out").mkdir(parents=True, exist_ok=True)
    (root / "out").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "out" / "ambiguity_contract_policy_check.json").write_text(
        json.dumps(
            {
                "ast": {"violations": []},
                "grade": {
                    "violations": [
                        {
                            "rule_id": "GMP-001",
                            "message": "sample grade issue",
                            "path": "src/gabion/sample.py",
                            "line": decorated_line,
                            "qualname": "decorated",
                            "details": {},
                        }
                    ]
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "out" / "test_evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tests": [
                    {
                        "test_id": "tests/test_sample.py::test_decorated",
                        "file": "tests/test_sample.py",
                        "line": 10,
                        "status": "pass",
                        "evidence": [
                            {
                                "key": {
                                    "site": {
                                        "path": "src/gabion/sample.py",
                                        "qual": "decorated",
                                    }
                                }
                            }
                        ],
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    graph = invariant_graph.build_invariant_graph(
        root,
        declared_registries=_NO_DECLARED_REGISTRIES,
    )
    node_kind_counts = graph.as_payload()["counts"]["node_kind_counts"]
    assert node_kind_counts["policy_signal"] == 1
    assert node_kind_counts["test_case"] == 1
    edges = {(edge.edge_kind, edge.source_id, edge.target_id) for edge in graph.edges}
    matched_nodes = invariant_graph.trace_nodes(graph, "OBJ-TODO")
    assert any(node.node_kind == "decorated_symbol" for node in matched_nodes)
    decorated_node = next(node for node in matched_nodes if node.node_kind == "decorated_symbol")
    assert any(
        edge_kind == "covered_by" and source_id == decorated_node.node_id
        for edge_kind, source_id, _target_id in edges
    )
    assert any(
        edge_kind == "blocks"
        and target_id == decorated_node.node_id
        for edge_kind, _source_id, target_id in edges
    )


def test_build_invariant_graph_joins_pytest_failures_and_couples_them_to_work(
    tmp_path: Path,
) -> None:
    root = _sample_repo(tmp_path)
    decorated_line = _sample_decorated_line(root)
    (root / "artifacts" / "test_runs").mkdir(parents=True, exist_ok=True)
    (root / "out").mkdir(parents=True, exist_ok=True)
    (root / "out" / "test_evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tests": [
                    {
                        "test_id": "tests/test_sample.py::test_decorated",
                        "file": "tests/test_sample.py",
                        "line": 10,
                        "status": "pass",
                        "evidence": [
                            {
                                "key": {
                                    "site": {
                                        "path": "src/gabion/sample.py",
                                        "qual": "decorated",
                                    }
                                }
                            }
                        ],
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "artifacts" / "test_runs" / "junit.xml").write_text(
        "\n".join(
            [
                "<?xml version=\"1.0\" encoding=\"utf-8\"?>",
                "<testsuites>",
                "  <testsuite name=\"pytest\" tests=\"1\" failures=\"1\" errors=\"0\">",
                "    <testcase classname=\"tests.test_sample\" name=\"test_decorated\" file=\"tests/test_sample.py\" line=\"10\">",
                "      <failure message=\"assert 1 == 2\">",
                "Traceback (most recent call last):",
                f"  File \"src/gabion/sample.py\", line {decorated_line}, in decorated",
                "    assert 1 == 2",
                "AssertionError: assert 1 == 2",
                "      </failure>",
                "    </testcase>",
                "  </testsuite>",
                "</testsuites>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    graph = invariant_graph.build_invariant_graph(
        root,
        declared_registries=_NO_DECLARED_REGISTRIES,
    )
    node_kind_counts = graph.as_payload()["counts"]["node_kind_counts"]
    assert node_kind_counts["test_case"] == 1
    assert node_kind_counts["test_failure"] == 1

    traced = invariant_graph.trace_nodes(graph, "OBJ-TODO")
    decorated_node = next(node for node in traced if node.node_kind == "decorated_symbol")
    test_case_node = next(node for node in graph.nodes if node.node_kind == "test_case")
    test_failure_node = next(node for node in graph.nodes if node.node_kind == "test_failure")
    edges = {(edge.edge_kind, edge.source_id, edge.target_id) for edge in graph.edges}
    assert ("covered_by", decorated_node.node_id, test_case_node.node_id) in edges
    assert ("fails_with", test_case_node.node_id, test_failure_node.node_id) in edges
    assert ("fails_on", test_failure_node.node_id, decorated_node.node_id) in edges

    projection_graph = invariant_graph.InvariantGraph(
        root=str(root),
        workstream_root_ids=("WS-TEST",),
        nodes=graph.nodes
        + (
            invariant_graph.InvariantGraphNode(
                node_id="object_id:WS-TEST",
                node_kind="synthetic_work_item",
                title="Test Workstream",
                marker_name="todo",
                marker_kind="todo",
                marker_id="ws.test",
                site_identity="site.ws.test",
                structural_identity="struct.ws.test",
                object_ids=("WS-TEST",),
                doc_ids=(),
                policy_ids=(),
                invariant_ids=(),
                reasoning_summary="test workstream",
                reasoning_control="tests.runtime_policy.ws",
                blocking_dependencies=(),
                rel_path="",
                qualname="",
                line=0,
                column=0,
                ast_node_kind="",
                seam_class="",
                source_marker_node_id="",
                status_hint="open",
            ),
            invariant_graph.InvariantGraphNode(
                node_id="object_id:WS-TEST-SQ-001",
                node_kind="synthetic_work_item",
                title="Test Subqueue",
                marker_name="todo",
                marker_kind="todo",
                marker_id="ws.test.sq",
                site_identity="site.ws.test.sq",
                structural_identity="struct.ws.test.sq",
                object_ids=("WS-TEST-SQ-001",),
                doc_ids=(),
                policy_ids=(),
                invariant_ids=(),
                reasoning_summary="test subqueue",
                reasoning_control="tests.runtime_policy.ws.sq",
                blocking_dependencies=(),
                rel_path="",
                qualname="",
                line=0,
                column=0,
                ast_node_kind="",
                seam_class="",
                source_marker_node_id="",
                status_hint="open",
            ),
            invariant_graph.InvariantGraphNode(
                node_id="object_id:WS-TEST-TP-001",
                node_kind="synthetic_work_item",
                title="Test Touchpoint",
                marker_name="todo",
                marker_kind="todo",
                marker_id="ws.test.tp",
                site_identity="site.ws.test.tp",
                structural_identity="struct.ws.test.tp",
                object_ids=("WS-TEST-TP-001",),
                doc_ids=(),
                policy_ids=(),
                invariant_ids=(),
                reasoning_summary="test touchpoint",
                reasoning_control="tests.runtime_policy.ws.tp",
                blocking_dependencies=(),
                rel_path=decorated_node.rel_path,
                qualname=decorated_node.qualname,
                line=decorated_node.line,
                column=decorated_node.column,
                ast_node_kind="",
                seam_class="",
                source_marker_node_id="",
                status_hint="open",
            ),
            invariant_graph.InvariantGraphNode(
                node_id="object_id:WS-TEST-TS-001",
                node_kind="synthetic_touchsite",
                title="decorated touchsite",
                marker_name="todo",
                marker_kind="todo",
                marker_id="ws.test.ts",
                site_identity="site.ws.test.ts",
                structural_identity="struct.ws.test.ts",
                object_ids=("WS-TEST-TS-001",),
                doc_ids=(),
                policy_ids=(),
                invariant_ids=(),
                reasoning_summary="decorated touchsite",
                reasoning_control="tests.runtime_policy.ws.ts",
                blocking_dependencies=(),
                rel_path=decorated_node.rel_path,
                qualname=decorated_node.qualname,
                line=decorated_node.line,
                column=decorated_node.column,
                ast_node_kind=decorated_node.ast_node_kind,
                seam_class="surviving_carrier_seam",
                source_marker_node_id="",
                status_hint="open",
            ),
        ),
        edges=graph.edges
        + (
            invariant_graph.InvariantGraphEdge(
                edge_id="contains:ws-root-sq",
                edge_kind="contains",
                source_id="object_id:WS-TEST",
                target_id="object_id:WS-TEST-SQ-001",
            ),
            invariant_graph.InvariantGraphEdge(
                edge_id="contains:ws-sq-tp",
                edge_kind="contains",
                source_id="object_id:WS-TEST-SQ-001",
                target_id="object_id:WS-TEST-TP-001",
            ),
            invariant_graph.InvariantGraphEdge(
                edge_id="contains:ws-tp-ts",
                edge_kind="contains",
                source_id="object_id:WS-TEST-TP-001",
                target_id="object_id:WS-TEST-TS-001",
            ),
            invariant_graph.InvariantGraphEdge(
                edge_id="covered_by:ws-ts-test-case",
                edge_kind="covered_by",
                source_id="object_id:WS-TEST-TS-001",
                target_id=test_case_node.node_id,
            ),
            invariant_graph.InvariantGraphEdge(
                edge_id="fails_on:test-failure-ws-ts",
                edge_kind="fails_on",
                source_id=test_failure_node.node_id,
                target_id="object_id:WS-TEST-TS-001",
            ),
        ),
        diagnostics=graph.diagnostics,
    )
    projection = invariant_graph.build_invariant_workstreams(projection_graph, root=root)
    workstream = next(projection.iter_workstreams())
    touchpoint = next(workstream.iter_touchpoints())
    touchsite = next(touchpoint.iter_touchsites())
    assert touchsite.coverage_count == 1
    assert touchsite.failing_test_case_count == 1
    assert touchsite.test_failure_count == 1
    assert touchpoint.failing_test_case_count == 1
    assert touchpoint.test_failure_count == 1
    assert workstream.failing_test_case_count == 1
    assert workstream.test_failure_count == 1
    payload = workstream.as_payload()
    assert payload["failing_test_case_count"] == 1
    assert payload["test_failure_count"] == 1
    assert payload["touchpoints"][0]["failing_test_case_count"] == 1
    assert payload["touchpoints"][0]["test_failure_count"] == 1
    assert payload["touchpoints"][0]["touchsites"][0]["failing_test_case_count"] == 1
    assert payload["touchpoints"][0]["touchsites"][0]["test_failure_count"] == 1


def test_build_invariant_graph_joins_sppf_and_inbox_governance_sources(
    tmp_path: Path,
) -> None:
    root = tmp_path
    _write(root / "src" / "gabion" / "__init__.py", "")
    _write(
        root / "docs" / "sppf_checklist.md",
        "\n".join(
            [
                "---",
                "doc_id: sppf_checklist",
                "---",
                "",
                "# SPPF Checklist",
                "- [~] Example convergence lane. (GH-60) sppf{doc=partial; impl=partial; doc_ref=in-15@2}",
            ]
        ),
    )
    _write(
        root / "docs" / "influence_index.md",
        "\n".join(
            [
                "---",
                "doc_id: influence_index",
                "---",
                "",
                "# Influence Index",
                "- in/in-15.md — **partial** (tracked in checklist and still converging.)",
            ]
        ),
    )
    _write(
        root / "in" / "in-15.md",
        "\n".join(
            [
                "---",
                "doc_id: in_15",
                "doc_role: inbox",
                "doc_authority: informative",
                "---",
                "",
                "## Goals",
                "1. Close GH-60 through a typed witness carrier.",
                "",
                "## Next Steps",
                "1. Land the shared governance graph substrate.",
            ]
        ),
    )
    _write_json(
        root / "artifacts" / "sppf_dependency_graph.json",
        {
            "format_version": 1,
            "generated_at": "2026-03-13T00:00:00+00:00",
            "source": "docs/sppf_checklist.md",
            "docs": {
                "in-15@2": {
                    "doc_id": "in-15",
                    "id": "in-15@2",
                    "issues": ["GH-60"],
                    "revision": 2,
                }
            },
            "issues": {
                "GH-60": {
                    "id": "GH-60",
                    "checklist_state": "~",
                    "doc_refs": ["in-15@2"],
                    "doc_status": "partial",
                    "impl_status": "partial",
                    "line": "- [~] Example convergence lane. (GH-60)",
                    "line_no": 6,
                }
            },
            "edges": [{"from": "in-15@2", "kind": "doc_ref", "to": "GH-60"}],
            "docs_without_issue": [],
            "issues_without_doc_ref": [],
        },
    )

    graph = invariant_graph.build_invariant_graph(
        root,
        declared_registries=_NO_DECLARED_REGISTRIES,
    )
    node_kind_counts = graph.as_payload()["counts"]["node_kind_counts"]
    assert node_kind_counts["sppf_doc_ref"] == 1
    assert node_kind_counts["sppf_issue"] == 1
    assert node_kind_counts["governance_doc"] >= 3
    assert node_kind_counts["governance_section"] == 2
    assert node_kind_counts["governance_action_item"] == 2

    inbox_doc = next(
        node
        for node in graph.nodes
        if node.node_kind == "governance_doc" and node.rel_path == "in/in-15.md"
    )
    sppf_doc = next(node for node in graph.nodes if node.node_kind == "sppf_doc_ref")
    sppf_issue = next(node for node in graph.nodes if node.node_kind == "sppf_issue")
    action_item = next(
        node
        for node in graph.nodes
        if node.node_kind == "governance_action_item"
        and "Close GH-60 through a typed witness carrier." in node.title
    )

    assert inbox_doc.status_hint == "partial"
    assert {"in-15", "in_15"}.issubset(set(inbox_doc.doc_ids))
    assert sppf_doc.doc_ids == ("in-15",)

    edges = {(edge.edge_kind, edge.source_id, edge.target_id) for edge in graph.edges}
    assert ("tracks", inbox_doc.node_id, sppf_doc.node_id) in edges
    assert ("doc_ref", sppf_doc.node_id, sppf_issue.node_id) in edges
    assert ("tracks", action_item.node_id, sppf_issue.node_id) in edges
    assert any(
        edge_kind == "contains" and target_id == action_item.node_id
        for edge_kind, _source_id, target_id in edges
    )


def test_build_invariant_graph_joins_control_loop_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path
    _write(root / "src" / "gabion" / "__init__.py", "")
    _write(
        root / "in" / "in-54.md",
        "\n".join(
            [
                "---",
                "doc_id: in_54",
                "doc_role: inbox",
                "doc_authority: informative",
                "---",
                "",
                "## Goals",
                "1. Close packet debt through one graph substrate.",
            ]
        ),
    )
    _write_json(
        root / "artifacts" / "out" / "docflow_packet_enforcement.json",
        {
            "scope": {"kind": "docflow_packet_enforcement"},
            "summary": {
                "active_packets": 1,
                "active_rows": 1,
                "new_rows": 1,
                "drifted_rows": 0,
                "ready": 0,
                "blocked": 1,
                "drifted": 0,
            },
            "new_rows": [
                {
                    "row_id": "docflow:row-1",
                    "path": "in/in-54.md",
                    "classification": "metadata_only",
                }
            ],
            "drifted_rows": [],
            "changed_paths": ["in/in-54.md"],
            "out_of_scope_touches": ["AGENTS.md"],
            "unresolved_touched_packets": [],
            "packet_status": [
                {
                    "path": "in/in-54.md",
                    "classification": "metadata_only",
                    "status": "blocked",
                    "row_count": 1,
                    "row_ids": ["docflow:row-1"],
                    "proving_tests": [],
                }
            ],
            "proving_tests": {"status": "skipped", "returncode": 0, "targets": []},
        },
    )
    _write_json(
        root / "artifacts" / "out" / "docflow_compliance.json",
        {
            "version": 2,
            "summary": {
                "compliant": 0,
                "contradicts": 1,
                "excess": 0,
                "proposed": 0,
            },
            "rows": [
                {
                    "row_kind": "docflow_compliance",
                    "invariant": "docflow:missing_explicit_reference",
                    "invariant_kind": "never",
                    "status": "contradicts",
                    "path": "in/in-54.md",
                    "source_row_kind": "doc_requires_ref",
                    "detail": "missing explicit reference",
                }
            ],
            "obligations": {
                "summary": {
                    "total": 1,
                    "triggered": 1,
                    "met": 0,
                    "unmet_fail": 1,
                    "unmet_warn": 0,
                },
                "context": {
                    "changed_paths": ["in/in-54.md"],
                    "sppf_relevant_paths_changed": True,
                    "gh_reference_validated": False,
                    "baseline_write_emitted": False,
                    "delta_guard_checked": False,
                    "doc_status_changed": True,
                    "checklist_influence_consistent": False,
                    "rev_range": "origin/stage..HEAD",
                    "commits": [
                        {
                            "sha": "b" * 40,
                            "subject": "Add PM view boundary renderer",
                        }
                    ],
                    "issue_ids": [],
                    "checklist_impact": [],
                },
                "entries": [
                    {
                        "obligation_id": "sppf_gh_reference_validation",
                        "triggered": True,
                        "status": "unmet",
                        "enforcement": "fail",
                        "description": "SPPF-relevant path changes require GH-reference validation.",
                    }
                ],
            },
        },
    )
    _write_json(
        root / "artifacts" / "out" / "controller_drift.json",
        {
            "summary": {
                "high_severity_findings": 1,
                "highest_severity": "high",
                "sensors": ["checks_without_normative_anchor"],
                "severity_counts": {"critical": 0, "high": 1, "medium": 0, "low": 0},
                "total_findings": 1,
            },
            "findings": [
                {
                    "sensor": "checks_without_normative_anchor",
                    "severity": "high",
                    "anchor": "CD-999",
                    "detail": "Workflow references `AGENTS.md` without a controlling anchor.",
                }
            ],
        },
    )
    _write_json(
        root / "artifacts" / "out" / "git_state.json",
        {
            "schema_version": 1,
            "artifact_kind": "git_state",
            "head_sha": "a" * 40,
            "branch": "main",
            "upstream": "origin/main",
            "is_detached": False,
            "summary": {
                "committed_count": 0,
                "staged_count": 1,
                "unstaged_count": 0,
                "untracked_count": 0,
            },
            "entries": [
                {
                    "state_class": "staged",
                    "change_code": "M",
                    "path": "in/in-54.md",
                    "previous_path": "",
                }
            ],
        },
    )
    _write_json(
        root / "artifacts" / "out" / "local_repro_closure_ledger.json",
        {
            "schema_version": 1,
            "generated_by": "tests",
            "workstream": "full_local_repro_closure",
            "entries": [
                {
                    "cu_id": "CU-R1",
                    "summary": "Close the local repro loop.",
                    "validation": {
                        "policy_workflows": "pass",
                        "policy_ambiguity_contract": "pass",
                    },
                }
            ],
        },
    )
    _write_json(
        root / "artifacts" / "out" / "local_ci_repro_contract.json",
        {
            "schema_version": 2,
            "artifact_kind": "local_ci_repro_contract",
            "generated_by": "tests",
            "summary": "Local CI reproduction topology.",
            "surfaces": [
                {
                    "surface_id": "workflow:ci.yml:checks",
                    "surface_kind": "workflow_job",
                    "title": "CI checks workflow job",
                    "summary": "Strict gates.",
                    "source_ref": ".github/workflows/ci.yml",
                    "mode": "checks",
                    "status": "pass",
                    "required_capabilities": [
                        {
                            "capability_id": "policy_workflows_output",
                            "summary": "Materialize the workflow policy artifact.",
                            "status": "pass",
                            "source_alternative_token_groups": [["policy_check.py", "--workflows"]],
                            "command_alternative_token_groups": [],
                            "matched_source_alternative_index": 0,
                            "matched_command_alternative_index": None,
                        }
                    ],
                    "missing_capability_ids": [],
                    "required_token_groups": [["policy_check.py", "--workflows"]],
                    "missing_token_groups": [],
                    "commands": ["python scripts/policy/policy_check.py --workflows"],
                    "artifacts": [
                        "artifacts/out/controller_drift.json",
                        "artifacts/out/docflow_compliance.json",
                    ],
                },
                {
                    "surface_id": "local_script:scripts/ci_local_repro.sh:checks",
                    "surface_kind": "local_repro_lane",
                    "title": "Local CI reproduction checks lane",
                    "summary": "Local parity lane.",
                    "source_ref": "scripts/ci_local_repro.sh",
                    "mode": "checks-only",
                    "status": "fail",
                    "required_capabilities": [
                        {
                            "capability_id": "policy_workflows_output",
                            "summary": "Materialize the workflow policy artifact.",
                            "status": "fail",
                            "source_alternative_token_groups": [["checks_policy_workflows_output"]],
                            "command_alternative_token_groups": [],
                            "matched_source_alternative_index": None,
                            "matched_command_alternative_index": None,
                        }
                    ],
                    "missing_capability_ids": ["policy_workflows_output"],
                    "required_token_groups": [["checks_policy_workflows_output"]],
                    "missing_token_groups": [["checks_policy_workflows_output"]],
                    "commands": ["scripts/ci_local_repro.sh --checks-only"],
                    "artifacts": ["artifacts/out/docflow_compliance.json"],
                },
            ],
            "relations": [
                {
                    "relation_id": "ci-repro:local-checks->workflow-checks",
                    "relation_kind": "reproduces",
                    "source_surface_id": "local_script:scripts/ci_local_repro.sh:checks",
                    "target_surface_id": "workflow:ci.yml:checks",
                    "source_missing_capability_ids": ["policy_workflows_output"],
                    "target_missing_capability_ids": [],
                    "status": "fail",
                    "summary": "Local checks should reproduce workflow checks.",
                }
            ],
        },
    )
    _write_json(
        root / "artifacts" / "out" / "kernel_vm_alignment.json",
        {
            "artifact_kind": "kernel_vm_alignment",
            "schema_version": 1,
            "generated_by": "tests",
            "fragment_id": "ttl_kernel_vm.fragment.augmented_rule_polarity_query_ast",
            "summary": {
                "binding_count": 1,
                "pass_count": 0,
                "partial_count": 1,
                "fail_count": 0,
                "residue_count": 1,
            },
            "bindings": [
                {
                    "binding_id": "kernel_vm.augmented_rule_core",
                    "fragment_id": "ttl_kernel_vm.fragment.augmented_rule_polarity_query_ast",
                    "title": "AugmentedRule core object over semantic-row reflection",
                    "status": "partial",
                    "summary": "Synthetic kernel VM binding",
                    "kernel_terms": ["lg:AugmentedRule"],
                    "runtime_surface_symbols": ["CanonicalWitnessedSemanticRow"],
                    "realizer_symbols": [],
                    "runtime_object_symbols": ["AugmentedRule"],
                    "missing_capability_ids": ["runtime_object_image"],
                    "residue_ids": [
                        "kernel_vm.augmented_rule_core:missing_runtime_object_image"
                    ],
                    "evidence_paths": [
                        "in/lg_kernel_ontology_cut_elim-1.ttl",
                        "src/gabion/analysis/projection/semantic_fragment.py",
                    ],
                    "capabilities": [
                        {
                            "capability_id": "runtime_object_image",
                            "requirement_kind": "runtime_object_image",
                            "status": "fail",
                            "match_mode": "all",
                            "description": "explicit runtime object image for AugmentedRule",
                            "residue_kind": "missing_runtime_object_image",
                            "severity": "warning",
                            "score": 6,
                            "expected_refs": [
                                {
                                    "rel_path": "src/gabion/analysis/projection/semantic_fragment.py",
                                    "evidence_kind": "python_symbol",
                                    "symbol": "AugmentedRule",
                                    "present": False,
                                }
                            ],
                            "matched_refs": [],
                            "missing_refs": [
                                {
                                    "rel_path": "src/gabion/analysis/projection/semantic_fragment.py",
                                    "evidence_kind": "python_symbol",
                                    "symbol": "AugmentedRule",
                                    "present": False,
                                }
                            ],
                        }
                    ],
                }
            ],
            "residues": [
                {
                    "residue_id": "kernel_vm.augmented_rule_core:missing_runtime_object_image",
                    "binding_id": "kernel_vm.augmented_rule_core",
                    "fragment_id": "ttl_kernel_vm.fragment.augmented_rule_polarity_query_ast",
                    "residue_kind": "missing_runtime_object_image",
                    "severity": "warning",
                    "score": 6,
                    "title": "AugmentedRule core object over semantic-row reflection",
                    "message": "Synthetic kernel VM residue",
                    "missing_capability_ids": ["runtime_object_image"],
                    "kernel_terms": ["lg:AugmentedRule"],
                    "runtime_surface_symbols": ["CanonicalWitnessedSemanticRow"],
                    "realizer_symbols": [],
                    "runtime_object_symbols": ["AugmentedRule"],
                    "evidence_paths": [
                        "in/lg_kernel_ontology_cut_elim-1.ttl",
                        "src/gabion/analysis/projection/semantic_fragment.py",
                    ],
                }
            ],
        },
    )
    _write_json(
        root / "artifacts" / "out" / "identity_grammar_completion.json",
        {
            "artifact_kind": "identity_grammar_completion",
            "schema_version": 1,
            "generated_by": "tests",
            "summary": {
                "surface_count": 2,
                "pass_count": 0,
                "fail_count": 2,
                "residue_count": 2,
                "highest_severity": "high",
            },
            "surfaces": [
                {
                    "surface_id": "identity_grammar.hotspot.raw_string_grouping",
                    "title": "Hotspot queue still groups by raw path strings",
                    "status": "fail",
                    "summary": "Synthetic identity grammar surface",
                    "evidence_paths": ["scripts/policy/hotspot_neighborhood_queue.py"],
                    "residue_ids": [
                        "identity_grammar.hotspot.raw_string_grouping:raw_string_grouping_in_core_queue_logic"
                    ],
                },
                {
                    "surface_id": "identity_grammar.coherence.two_cell",
                    "title": "Coherence witness carrier exists but is not emitted",
                    "status": "fail",
                    "summary": "Synthetic coherence surface",
                    "evidence_paths": [
                        "src/gabion/tooling/policy_substrate/identity_zone/grammar.py"
                    ],
                    "residue_ids": [
                        "identity_grammar.coherence.two_cell:coherence_witness_emission_missing"
                    ],
                },
            ],
            "residues": [
                {
                    "residue_id": "identity_grammar.hotspot.raw_string_grouping:raw_string_grouping_in_core_queue_logic",
                    "surface_id": "identity_grammar.hotspot.raw_string_grouping",
                    "residue_kind": "raw_string_grouping_in_core_queue_logic",
                    "severity": "high",
                    "score": 9,
                    "title": "Hotspot queue still groups by raw path strings",
                    "message": "Synthetic hotspot residue",
                    "evidence_paths": ["scripts/policy/hotspot_neighborhood_queue.py"],
                },
                {
                    "residue_id": "identity_grammar.coherence.two_cell:coherence_witness_emission_missing",
                    "surface_id": "identity_grammar.coherence.two_cell",
                    "residue_kind": "coherence_witness_emission_missing",
                    "severity": "medium",
                    "score": 6,
                    "title": "Coherence witness carrier exists but is not emitted",
                    "message": "Synthetic coherence residue",
                    "evidence_paths": [
                        "src/gabion/tooling/policy_substrate/identity_zone/grammar.py"
                    ],
                },
            ],
        },
    )

    graph = invariant_graph.build_invariant_graph(
        root,
        declared_registries=_CONNECTIVITY_SYNERGY_WITH_PSF_STUB_DECLARED_REGISTRIES,
    )
    node_kind_counts = graph.as_payload()["counts"]["node_kind_counts"]
    assert node_kind_counts["docflow_packet_enforcement"] == 1
    assert node_kind_counts["docflow_packet"] == 1
    assert node_kind_counts["docflow_packet_row"] == 1
    assert node_kind_counts["docflow_compliance_report"] == 1
    assert node_kind_counts["docflow_provenance_report"] == 1
    assert node_kind_counts["docflow_commit_range"] == 1
    assert node_kind_counts["docflow_commit"] == 1
    assert node_kind_counts["docflow_compliance_row"] == 1
    assert node_kind_counts["docflow_obligation"] == 1
    assert node_kind_counts["controller_drift_report"] == 1
    assert node_kind_counts["controller_drift_finding"] == 1
    assert node_kind_counts["local_repro_closure_ledger"] == 1
    assert node_kind_counts["local_repro_entry"] == 1
    assert node_kind_counts["local_ci_repro_contract"] == 1
    assert node_kind_counts["local_ci_repro_surface"] == 2
    assert node_kind_counts["local_ci_repro_capability"] == 2
    assert node_kind_counts["local_ci_repro_relation"] == 1
    assert node_kind_counts["kernel_vm_alignment_report"] == 1
    assert node_kind_counts["kernel_vm_alignment_binding"] == 1
    assert node_kind_counts["kernel_vm_alignment_residue"] == 1
    assert node_kind_counts["identity_grammar_completion_report"] == 1
    assert node_kind_counts["identity_grammar_completion_surface"] == 2
    assert node_kind_counts["identity_grammar_completion_residue"] == 2
    assert node_kind_counts["git_state_entry"] == 1

    inbox_doc = next(
        node
        for node in graph.nodes
        if node.node_kind == "governance_doc" and node.rel_path == "in/in-54.md"
    )
    packet_node = next(node for node in graph.nodes if node.node_kind == "docflow_packet")
    packet_row_node = next(
        node for node in graph.nodes if node.node_kind == "docflow_packet_row"
    )
    compliance_row_node = next(
        node for node in graph.nodes if node.node_kind == "docflow_compliance_row"
    )
    compliance_report_node = next(
        node for node in graph.nodes if node.node_kind == "docflow_compliance_report"
    )
    obligation_node = next(
        node for node in graph.nodes if node.node_kind == "docflow_obligation"
    )
    provenance_report_node = next(
        node for node in graph.nodes if node.node_kind == "docflow_provenance_report"
    )
    rev_range_node = next(
        node for node in graph.nodes if node.node_kind == "docflow_commit_range"
    )
    commit_node = next(node for node in graph.nodes if node.node_kind == "docflow_commit")
    controller_drift_finding = next(
        node
        for node in graph.nodes
        if node.node_kind == "controller_drift_finding"
    )
    git_state_entry = next(
        node for node in graph.nodes if node.node_kind == "git_state_entry"
    )
    agents_doc = next(
        node for node in graph.nodes if node.node_kind == "governance_doc" and node.rel_path == "AGENTS.md"
    )
    local_repro_entry = next(
        node for node in graph.nodes if node.node_kind == "local_repro_entry"
    )
    local_ci_repro_surface = next(
        node
        for node in graph.nodes
        if node.node_kind == "local_ci_repro_surface"
        and "local_script:scripts/ci_local_repro.sh:checks" in node.object_ids
    )
    local_ci_repro_capability = next(
        node
        for node in graph.nodes
        if node.node_kind == "local_ci_repro_capability"
        and "policy_workflows_output" in node.object_ids
        and "local_script:scripts/ci_local_repro.sh:checks" in node.object_ids
    )
    local_ci_repro_relation = next(
        node for node in graph.nodes if node.node_kind == "local_ci_repro_relation"
    )
    kernel_vm_binding = next(
        node for node in graph.nodes if node.node_kind == "kernel_vm_alignment_binding"
    )
    kernel_vm_residue = next(
        node for node in graph.nodes if node.node_kind == "kernel_vm_alignment_residue"
    )
    identity_grammar_surface = next(
        node
        for node in graph.nodes
        if node.node_kind == "identity_grammar_completion_surface"
        and "identity_grammar.hotspot.raw_string_grouping" in node.object_ids
    )
    identity_grammar_residue = next(
        node
        for node in graph.nodes
        if node.node_kind == "identity_grammar_completion_residue"
        and "raw_string_grouping_in_core_queue_logic" in node.object_ids
    )
    edges = {(edge.edge_kind, edge.source_id, edge.target_id) for edge in graph.edges}

    assert {"in-54", "in_54"}.issubset(set(inbox_doc.doc_ids))
    assert {"in-54", "in_54"}.issubset(set(packet_node.doc_ids))
    assert {"in-54", "in_54"}.issubset(set(compliance_row_node.doc_ids))
    assert "docflow:row-1" in packet_row_node.object_ids
    assert "docflow:missing_explicit_reference" in compliance_row_node.object_ids
    assert "sppf_gh_reference_validation" in obligation_node.object_ids
    assert "origin/stage..HEAD" in provenance_report_node.object_ids
    assert "origin/stage..HEAD" in rev_range_node.object_ids
    assert "b" * 40 in commit_node.object_ids
    assert any(
        item.startswith("docflow_packet_enforcement:")
        for item in packet_row_node.object_ids
    )
    assert "CD-999" in controller_drift_finding.object_ids
    assert any(
        item.startswith("controller_drift:")
        for item in controller_drift_finding.object_ids
    )
    assert "CU-R1" in local_repro_entry.object_ids
    assert any(
        item.startswith("local_repro_closure_ledger:")
        for item in local_repro_entry.object_ids
    )
    assert "local_script:scripts/ci_local_repro.sh:checks" in local_ci_repro_surface.object_ids
    assert "policy_workflows_output" in local_ci_repro_capability.object_ids
    assert "ci-repro:local-checks->workflow-checks" in local_ci_repro_relation.object_ids
    assert "kernel_vm.augmented_rule_core" in kernel_vm_binding.object_ids
    assert "missing_runtime_object_image" in kernel_vm_residue.object_ids
    assert "identity_grammar.hotspot.raw_string_grouping" in identity_grammar_surface.object_ids
    assert "raw_string_grouping_in_core_queue_logic" in identity_grammar_residue.object_ids
    assert ("tracks", packet_node.node_id, inbox_doc.node_id) in edges
    assert ("tracks", packet_row_node.node_id, inbox_doc.node_id) in edges
    assert ("tracks", compliance_row_node.node_id, inbox_doc.node_id) in edges
    assert ("tracks", local_ci_repro_surface.node_id, compliance_report_node.node_id) in edges
    assert ("contains", local_ci_repro_surface.node_id, local_ci_repro_capability.node_id) in edges
    assert ("tracks", local_ci_repro_relation.node_id, local_ci_repro_surface.node_id) in edges
    assert ("contains", kernel_vm_binding.node_id, kernel_vm_residue.node_id) in edges
    assert ("contains", identity_grammar_surface.node_id, identity_grammar_residue.node_id) in edges
    assert ("touches", compliance_row_node.node_id, git_state_entry.node_id) in edges
    assert ("touches", obligation_node.node_id, git_state_entry.node_id) in edges
    assert ("touches", provenance_report_node.node_id, git_state_entry.node_id) in edges
    assert ("tracks", controller_drift_finding.node_id, agents_doc.node_id) in edges
    assert invariant_graph.trace_nodes(graph, "CU-R1")
    assert invariant_graph.trace_nodes(graph, "docflow:row-1")
    assert invariant_graph.trace_nodes(graph, "origin/stage..HEAD")
    assert invariant_graph.trace_nodes(graph, "b" * 40)
    assert invariant_graph.trace_nodes(graph, "sppf_gh_reference_validation")
    assert invariant_graph.trace_nodes(graph, "ci-repro:local-checks->workflow-checks")
    assert invariant_graph.trace_nodes(graph, "local_script:scripts/ci_local_repro.sh:checks")
    assert invariant_graph.trace_nodes(graph, "policy_workflows_output")
    assert invariant_graph.trace_nodes(graph, "kernel_vm.augmented_rule_core")
    assert invariant_graph.trace_nodes(graph, "missing_runtime_object_image")
    assert invariant_graph.trace_nodes(
        graph, "raw_string_grouping_in_core_queue_logic"
    )

    workstreams = invariant_graph.build_invariant_workstreams(graph, root=root)
    payload = workstreams.as_payload()
    idr = next(item for item in payload["workstreams"] if item["object_id"] == "CSA-IDR")
    rgc = next(item for item in payload["workstreams"] if item["object_id"] == "CSA-RGC")
    assert (
        idr["next_actions"]["recommended_diagnostic_blocked_cut"]["object_id"]
        == "CSA-IDR-TP-004"
    )
    tp7 = next(
        item
        for item in rgc["next_actions"]["ranked_touchpoint_cuts"]
        if item["object_id"] == "CSA-RGC-TP-007"
    )
    tp8 = next(
        item
        for item in rgc["next_actions"]["ranked_touchpoint_cuts"]
        if item["object_id"] == "CSA-RGC-TP-008"
    )

    assert payload["diagnostic_summary"]["diagnostic_count"] >= 3
    assert rgc["next_actions"]["recommended_diagnostic_blocked_cut"] is not None
    assert (
        rgc["next_actions"]["recommended_diagnostic_blocked_cut"]["object_id"]
        == "CSA-RGC-TP-008"
    )
    assert tp7["diagnostic_count"] >= 1
    assert tp7["readiness_class"] == "diagnostic_blocked"
    assert tp8["diagnostic_count"] >= 1
    assert tp8["ranking_signal_count"] >= 1
    assert tp8["ranking_signal_score"] >= 6
    assert tp8["readiness_class"] == "diagnostic_blocked"


def test_build_invariant_graph_joins_docflow_issue_lifecycle_nodes(
    tmp_path: Path,
) -> None:
    root = tmp_path
    _write(root / "src" / "gabion" / "__init__.py", "")
    _write(
        root / "docs" / "sppf_checklist.md",
        "\n".join(
            [
                "# SPPF Checklist",
                "",
                "- [~] Project manager view linkage remains pending. (GH-214)",
            ]
        ),
    )
    _write_json(
        root / "artifacts" / "sppf_dependency_graph.json",
        {
            "format_version": 1,
            "generated_at": "2026-03-13T00:00:00+00:00",
            "source": "docs/sppf_checklist.md",
            "docs": {
                "pm-view@1": {
                    "doc_id": "pm-view",
                    "id": "pm-view@1",
                    "issues": ["GH-214"],
                    "revision": 1,
                }
            },
            "issues": {
                "GH-214": {
                    "id": "GH-214",
                    "checklist_state": "~",
                    "doc_refs": ["pm-view@1"],
                    "doc_status": "partial",
                    "impl_status": "partial",
                    "line": "- [~] Project manager view linkage remains pending. (GH-214)",
                    "line_no": 3,
                }
            },
            "edges": [{"from": "pm-view@1", "kind": "doc_ref", "to": "GH-214"}],
            "docs_without_issue": [],
            "issues_without_doc_ref": [],
        },
    )
    _write_json(
        root / "artifacts" / "out" / "docflow_compliance.json",
        {
            "version": 2,
            "summary": {
                "compliant": 1,
                "contradicts": 0,
                "excess": 0,
                "proposed": 0,
            },
            "rows": [],
            "obligations": {
                "summary": {
                    "total": 1,
                    "triggered": 1,
                    "met": 1,
                    "unmet_fail": 0,
                    "unmet_warn": 0,
                },
                "context": {
                    "changed_paths": [
                        "src/gabion/tooling/policy_substrate/project_manager_view.py"
                    ],
                    "sppf_relevant_paths_changed": True,
                    "gh_reference_validated": True,
                    "baseline_write_emitted": False,
                    "delta_guard_checked": False,
                    "doc_status_changed": True,
                    "checklist_influence_consistent": True,
                    "rev_range": "origin/stage..HEAD",
                    "commits": [
                        {
                            "sha": "c" * 40,
                            "subject": "Track PM view linkage through strict docflow",
                        }
                    ],
                    "issue_ids": ["214"],
                    "checklist_impact": [{"issue_id": "214", "commit_count": 1}],
                    "issue_lifecycle_fetch_status": "ok",
                    "issue_lifecycles": [
                        {
                            "issue_id": "214",
                            "state": "open",
                            "labels": ["done-on-stage", "status/pending-release"],
                            "url": "https://example.invalid/214",
                        }
                    ],
                    "issue_lifecycle_errors": [],
                },
                "entries": [
                    {
                        "obligation_id": "sppf_gh_reference_validation",
                        "triggered": True,
                        "status": "met",
                        "enforcement": "fail",
                        "description": "SPPF-relevant path changes require GH-reference validation.",
                    }
                ],
            },
        },
    )

    graph = invariant_graph.build_invariant_graph(
        root,
        declared_registries=_NO_DECLARED_REGISTRIES,
    )
    node_kind_counts = graph.as_payload()["counts"]["node_kind_counts"]
    assert node_kind_counts["docflow_issue_reference"] == 1
    assert node_kind_counts["docflow_issue_lifecycle"] == 1

    issue_reference_node = next(
        node for node in graph.nodes if node.node_kind == "docflow_issue_reference"
    )
    lifecycle_node = next(
        node for node in graph.nodes if node.node_kind == "docflow_issue_lifecycle"
    )
    sppf_issue_node = next(node for node in graph.nodes if node.node_kind == "sppf_issue")
    edges = {(edge.edge_kind, edge.source_id, edge.target_id) for edge in graph.edges}

    assert "GH-214" in issue_reference_node.object_ids
    assert "done-on-stage" in lifecycle_node.object_ids
    assert lifecycle_node.status_hint == "open"
    assert ("contains", issue_reference_node.node_id, lifecycle_node.node_id) in edges
    assert any(
        edge_kind == "tracks"
        and source_id == lifecycle_node.node_id
        and target_id == sppf_issue_node.node_id
        for edge_kind, source_id, target_id in edges
    )
    assert invariant_graph.trace_nodes(graph, "214")
    assert invariant_graph.trace_nodes(graph, "done-on-stage")


def test_docflow_issue_lifecycle_rules_emit_ranking_pressure_for_csa_rgc_tp_007(
    tmp_path: Path,
) -> None:
    root = tmp_path
    _write(root / "src" / "gabion" / "__init__.py", "")
    _write(
        root / "docs" / "sppf_checklist.md",
        "\n".join(
            [
                "# SPPF Checklist",
                "",
                "- [~] Project manager view linkage remains pending. (GH-214)",
            ]
        ),
    )
    _write_json(
        root / "artifacts" / "sppf_dependency_graph.json",
        {
            "format_version": 1,
            "generated_at": "2026-03-13T00:00:00+00:00",
            "source": "docs/sppf_checklist.md",
            "docs": {
                "pm-view@1": {
                    "doc_id": "pm-view",
                    "id": "pm-view@1",
                    "issues": ["GH-214"],
                    "revision": 1,
                }
            },
            "issues": {
                "GH-214": {
                    "id": "GH-214",
                    "checklist_state": "~",
                    "doc_refs": ["pm-view@1"],
                    "doc_status": "partial",
                    "impl_status": "partial",
                    "line": "- [~] Project manager view linkage remains pending. (GH-214)",
                    "line_no": 3,
                }
            },
            "edges": [{"from": "pm-view@1", "kind": "doc_ref", "to": "GH-214"}],
            "docs_without_issue": [],
            "issues_without_doc_ref": [],
        },
    )
    _write_json(
        root / "artifacts" / "out" / "docflow_compliance.json",
        {
            "version": 2,
            "summary": {
                "compliant": 1,
                "contradicts": 0,
                "excess": 0,
                "proposed": 0,
            },
            "rows": [],
            "obligations": {
                "summary": {
                    "total": 1,
                    "triggered": 1,
                    "met": 1,
                    "unmet_fail": 0,
                    "unmet_warn": 0,
                },
                "context": {
                    "changed_paths": [
                        "src/gabion/tooling/policy_substrate/project_manager_view.py"
                    ],
                    "sppf_relevant_paths_changed": True,
                    "gh_reference_validated": True,
                    "baseline_write_emitted": False,
                    "delta_guard_checked": False,
                    "doc_status_changed": True,
                    "checklist_influence_consistent": True,
                    "rev_range": "origin/stage..HEAD",
                    "commits": [
                        {
                            "sha": "d" * 40,
                            "subject": "Track PM view lifecycle pressure",
                        }
                    ],
                    "issue_ids": ["214"],
                    "checklist_impact": [{"issue_id": "214", "commit_count": 1}],
                    "issue_lifecycle_fetch_status": "ok",
                    "issue_lifecycles": [
                        {
                            "issue_id": "214",
                            "state": "closed",
                            "labels": ["done-on-stage"],
                            "url": "https://example.invalid/214",
                        }
                    ],
                    "issue_lifecycle_errors": [],
                },
                "entries": [
                    {
                        "obligation_id": "sppf_gh_reference_validation",
                        "triggered": True,
                        "status": "met",
                        "enforcement": "fail",
                        "description": "SPPF-relevant path changes require GH-reference validation.",
                    }
                ],
            },
        },
    )

    graph = invariant_graph.build_invariant_graph(
        root,
        declared_registries=_CONNECTIVITY_SYNERGY_WITH_PSF_STUB_DECLARED_REGISTRIES,
    )
    payload = graph.as_payload()
    assert payload["counts"]["ranking_signal_count"] >= 2
    assert payload["counts"]["ranking_signal_score_total"] >= 7
    ranking_codes = {item["code"] for item in payload["ranking_signals"]}
    assert {
        "docflow_issue_lifecycle_state_mismatch",
        "docflow_issue_lifecycle_missing_required_labels",
    } <= ranking_codes

    workstreams = invariant_graph.build_invariant_workstreams(graph, root=root)
    workstream_payload = workstreams.as_payload()
    rgc = next(item for item in workstream_payload["workstreams"] if item["object_id"] == "CSA-RGC")
    tp7 = next(
        item
        for item in rgc["next_actions"]["ranked_touchpoint_cuts"]
        if item["object_id"] == "CSA-RGC-TP-007"
    )

    assert tp7["ranking_signal_count"] == 2
    assert tp7["ranking_signal_score"] == 7
    assert rgc["next_actions"]["recommended_diagnostic_blocked_cut"] is not None
    assert (
        rgc["next_actions"]["recommended_diagnostic_blocked_cut"]["object_id"]
        == "CSA-RGC-TP-007"
    )


def test_build_invariant_graph_joins_ingress_merge_parity_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture_root = REPO_ROOT / "tests" / "fixtures" / "ingest_adapter"
    for name in (
        "python_raw.json",
        "synthetic_raw.json",
        "python_expected.json",
        "synthetic_expected.json",
    ):
        _write(
            tmp_path / "tests" / "fixtures" / "ingest_adapter" / name,
            (fixture_root / name).read_text(encoding="utf-8"),
        )
    _write(
        tmp_path / "docs" / "policy_rules.yaml",
        "\n".join(
            [
                "rules:",
                "  - rule_id: sample.rule",
                "    domain: ambiguity_contract",
                "    severity: blocking",
                "    predicate:",
                "      op: always",
                "    outcome:",
                "      kind: block",
                "      message: sample",
                "    evidence_contract: none",
            ]
        ),
    )
    write_ingress_merge_parity_artifact(
        root=tmp_path,
        rel_path="artifacts/out/ingress_merge_parity.json",
        artifact=build_ingress_merge_parity_artifact(
            root=tmp_path,
            identities=StructuredArtifactIdentitySpace(),
        ),
    )

    graph = invariant_graph.build_invariant_graph(
        tmp_path,
        declared_registries=_CONNECTIVITY_SYNERGY_WITH_PSF_STUB_DECLARED_REGISTRIES,
    )
    node_kind_counts = graph.as_payload()["counts"]["node_kind_counts"]
    assert node_kind_counts["ingress_merge_parity_report"] == 1
    assert node_kind_counts["ingress_merge_parity_case"] == 4

    report_node = next(
        node for node in graph.nodes if node.node_kind == "ingress_merge_parity_report"
    )
    case_node = next(
        node
        for node in graph.nodes
        if node.node_kind == "ingress_merge_parity_case"
        and "adapter_decision_surface_parity" in node.object_ids
    )
    touchpoint_node = next(
        node
        for node in invariant_graph.trace_nodes(graph, "CSA-IGM-TP-003")
        if node.node_kind == "synthetic_work_item"
    )
    edges = {(edge.edge_kind, edge.source_id, edge.target_id) for edge in graph.edges}

    assert any(item.startswith("ingress_merge_parity:") for item in report_node.object_ids)
    assert ("contains", touchpoint_node.node_id, report_node.node_id) in edges
    assert ("contains", report_node.node_id, case_node.node_id) in edges
    assert invariant_graph.trace_nodes(graph, "adapter_decision_surface_parity")
    assert invariant_graph.trace_nodes(graph, "frontmatter_adapter_projection_parity")


def test_build_invariant_graph_joins_git_state_artifact(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "artifacts" / "out" / "git_state.json",
        {
            "schema_version": 1,
            "artifact_kind": "git_state",
            "head_sha": "b" * 40,
            "branch": "main",
            "upstream": "origin/main",
            "is_detached": False,
            "summary": {
                "committed_count": 1,
                "staged_count": 1,
                "unstaged_count": 1,
                "untracked_count": 1,
            },
            "entries": [
                {
                    "state_class": "committed",
                    "change_code": "A",
                    "path": "tracked.txt",
                    "previous_path": "",
                },
                {
                    "state_class": "staged",
                    "change_code": "M",
                    "path": "src/example.py",
                    "previous_path": "",
                },
                {
                    "state_class": "unstaged",
                    "change_code": "M",
                    "path": "README.md",
                    "previous_path": "",
                },
                {
                    "state_class": "untracked",
                    "change_code": "??",
                    "path": "notes/todo.md",
                    "previous_path": "",
                },
            ],
        },
    )

    graph = invariant_graph.build_invariant_graph(
        tmp_path,
        declared_registries=_NO_DECLARED_REGISTRIES,
    )
    node_kind_counts = graph.as_payload()["counts"]["node_kind_counts"]
    assert node_kind_counts["git_state_report"] == 1
    assert node_kind_counts["git_head_commit"] == 1
    assert node_kind_counts["git_state_entry"] == 4

    report_node = next(
        node for node in graph.nodes if node.node_kind == "git_state_report"
    )
    head_node = next(
        node for node in graph.nodes if node.node_kind == "git_head_commit"
    )
    entry_node = next(
        node
        for node in graph.nodes
        if node.node_kind == "git_state_entry" and "src/example.py" in node.object_ids
    )
    edges = {(edge.edge_kind, edge.source_id, edge.target_id) for edge in graph.edges}

    assert "b" * 40 in report_node.object_ids
    assert ("contains", report_node.node_id, head_node.node_id) in edges
    assert ("contains", report_node.node_id, entry_node.node_id) in edges
    assert invariant_graph.trace_nodes(graph, "src/example.py")


def test_build_invariant_graph_prefers_live_git_state_over_stale_artifact(
    tmp_path: Path,
) -> None:
    root = write_minimal_invariant_repo(tmp_path)
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Gabion Tests")
    _git(root, "config", "user.email", "tests@example.com")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "initial synthetic repo")
    live_head_sha = _git(root, "rev-parse", "HEAD")

    _write_json(
        root / "artifacts" / "out" / "git_state.json",
        {
            "schema_version": 1,
            "artifact_kind": "git_state",
            "head_sha": "f" * 40,
            "branch": "main",
            "upstream": "origin/main",
            "is_detached": False,
            "summary": {
                "committed_count": 0,
                "staged_count": 0,
                "unstaged_count": 1,
                "untracked_count": 0,
            },
            "entries": [
                {
                    "state_class": "unstaged",
                    "change_code": "M",
                    "path": "src/gabion/synthetic_boundaries.py",
                    "previous_path": "",
                    "current_line_spans": [
                        {
                            "start_line": 1,
                            "line_count": 1,
                        }
                    ],
                },
            ],
        },
    )

    graph = invariant_graph.build_invariant_graph(
        root,
        declared_registries=_SYNTHETIC_CONNECTIVITY_DECLARED_REGISTRIES,
    )

    workspace_diagnostics = [
        item
        for item in graph.diagnostics
        if item.code
        in {
            "workspace_preservation_needed",
            "orphaned_workspace_change",
        }
    ]
    assert workspace_diagnostics == []

    report_node = next(
        node for node in graph.nodes if node.node_kind == "git_state_report"
    )
    assert live_head_sha in report_node.object_ids
    assert "f" * 40 not in report_node.object_ids

    workstreams = invariant_graph.build_invariant_workstreams(graph, root=root)
    payload = workstreams.as_payload()
    assert payload["diagnostic_summary"]["workspace_preservation_count"] == 0
    assert payload["diagnostic_summary"]["orphaned_workspace_change_count"] == 0


def test_git_state_dirty_graph_participant_exerts_workspace_preservation_pressure(
    tmp_path: Path,
) -> None:
    root = write_minimal_invariant_repo(tmp_path)
    _write_json(
        root / "artifacts" / "out" / "git_state.json",
        {
            "schema_version": 1,
            "artifact_kind": "git_state",
            "head_sha": "c" * 40,
            "branch": "main",
            "upstream": "origin/main",
            "is_detached": False,
            "summary": {
                "committed_count": 0,
                "staged_count": 0,
                "unstaged_count": 1,
                "untracked_count": 1,
            },
            "entries": [
                {
                    "state_class": "unstaged",
                    "change_code": "M",
                    "path": "src/gabion/synthetic_boundaries.py",
                    "previous_path": "",
                    "current_line_spans": [
                        {
                            "start_line": 1,
                            "line_count": 1,
                        }
                    ],
                },
                {
                    "state_class": "untracked",
                    "change_code": "??",
                    "path": "artifacts/out/generated_only.json",
                    "previous_path": "",
                },
            ],
        },
    )

    graph = invariant_graph.build_invariant_graph(
        root,
        declared_registries=_SYNTHETIC_CONNECTIVITY_DECLARED_REGISTRIES,
    )

    workspace_diagnostics = [
        item for item in graph.diagnostics if item.code == "workspace_preservation_needed"
    ]
    assert len(workspace_diagnostics) == 1
    assert workspace_diagnostics[0].raw_dependency == "unstaged"

    git_entry_node = next(
        node
        for node in graph.nodes
        if node.node_kind == "git_state_entry"
        and "src/gabion/synthetic_boundaries.py" in node.object_ids
    )
    graph_edges = {(edge.edge_kind, edge.source_id, edge.target_id) for edge in graph.edges}
    assert any(
        edge_kind == "touches" and source_id == git_entry_node.node_id
        for edge_kind, source_id, _target_id in graph_edges
    )

    workstreams = invariant_graph.build_invariant_workstreams(graph, root=root)
    payload = workstreams.as_payload()
    assert payload["diagnostic_summary"]["workspace_preservation_count"] == 1
    recommended_followup = payload["repo_next_actions"]["recommended_followup"]
    assert recommended_followup is not None
    assert recommended_followup["followup_family"] == "workspace_preservation"
    assert recommended_followup["action_kind"] == "state_preservation"
    assert recommended_followup["diagnostic_code"] == "workspace_preservation_needed"
    assert recommended_followup["recommended_action"] == (
        "stage_validate_commit_graph_participating_change"
    )
    assert recommended_followup["blocker_class"] == "workspace_unstaged"
    assert recommended_followup["queue_id"].startswith(
        "planner_queue|followup_family=workspace_preservation|"
    )
    workspace_lane = next(
        lane
        for lane in payload["repo_next_actions"]["diagnostic_lanes"]
        if lane["diagnostic_code"] == "workspace_preservation_needed"
    )
    assert workspace_lane["title"] == "unstaged:src/gabion/synthetic_boundaries.py"
    assert workspace_lane["candidate_owner_status"] == "ambiguous_exact_path_owner"
    assert workspace_lane["candidate_owner_object_ids"] == [
        "CSA-IDR",
        "CSA-IGM",
        "CSA-IVL",
        "CSA-RGC",
        "SCC",
    ]
    workspace_commit_unit = payload["repo_next_actions"][
        "recommended_workspace_commit_unit"
    ]
    assert workspace_commit_unit is not None
    assert workspace_commit_unit["followup_family"] == "workspace_preservation"
    assert workspace_commit_unit["diagnostic_code"] == "workspace_preservation_needed"
    assert workspace_commit_unit["owner_scope_kind"] == "ambiguous_owner_set"
    assert workspace_commit_unit["queue_id"].startswith(
        "planner_queue|followup_family=workspace_preservation|"
    )
    assert workspace_commit_unit["root_object_ids"] == [
        "CSA-IDR",
        "CSA-IGM",
        "CSA-IVL",
        "CSA-RGC",
        "SCC",
    ]
    assert workspace_commit_unit["rel_paths"] == [
        "src/gabion/synthetic_boundaries.py"
    ]


def test_git_state_dirty_nonoverlap_change_emits_orphaned_workspace_pressure(
    tmp_path: Path,
) -> None:
    root = write_minimal_invariant_repo(tmp_path)
    _write_json(
        root / "artifacts" / "out" / "git_state.json",
        {
            "schema_version": 1,
            "artifact_kind": "git_state",
            "head_sha": "d" * 40,
            "branch": "main",
            "upstream": "origin/main",
            "is_detached": False,
            "summary": {
                "committed_count": 0,
                "staged_count": 0,
                "unstaged_count": 1,
                "untracked_count": 0,
            },
            "entries": [
                {
                    "state_class": "unstaged",
                    "change_code": "M",
                    "path": "src/gabion/synthetic_boundaries.py",
                    "previous_path": "",
                    "current_line_spans": [
                        {
                            "start_line": 12,
                            "line_count": 1,
                        }
                    ],
                },
            ],
        },
    )

    graph = invariant_graph.build_invariant_graph(
        root,
        declared_registries=_SYNTHETIC_CONNECTIVITY_DECLARED_REGISTRIES,
    )

    orphaned_workspace_diagnostics = [
        item for item in graph.diagnostics if item.code == "orphaned_workspace_change"
    ]
    assert len(orphaned_workspace_diagnostics) == 1
    assert orphaned_workspace_diagnostics[0].raw_dependency == "unstaged"

    git_entry_node = next(
        node
        for node in graph.nodes
        if node.node_kind == "git_state_entry"
        and "src/gabion/synthetic_boundaries.py" in node.object_ids
    )
    graph_edges = {(edge.edge_kind, edge.source_id, edge.target_id) for edge in graph.edges}
    assert not any(
        edge_kind == "touches" and source_id == git_entry_node.node_id
        for edge_kind, source_id, _target_id in graph_edges
    )
    assert any(
        edge_kind == "shares_path_with" and source_id == git_entry_node.node_id
        for edge_kind, source_id, _target_id in graph_edges
    )

    workstreams = invariant_graph.build_invariant_workstreams(graph, root=root)
    payload = workstreams.as_payload()
    assert payload["diagnostic_summary"]["orphaned_workspace_change_count"] == 1
    recommended_followup = payload["repo_next_actions"]["recommended_followup"]
    assert recommended_followup is not None
    assert recommended_followup["followup_family"] == "workspace_orphan_resolution"
    assert recommended_followup["action_kind"] == "state_preservation"
    assert recommended_followup["diagnostic_code"] == "orphaned_workspace_change"
    assert recommended_followup["recommended_action"] == (
        "attribute_stage_validate_commit_orphaned_change"
    )
    assert recommended_followup["blocker_class"] == "workspace_orphan_unstaged"
    assert recommended_followup["queue_id"].startswith(
        "planner_queue|followup_family=workspace_orphan_resolution|"
    )
    workspace_lane = next(
        lane
        for lane in payload["repo_next_actions"]["diagnostic_lanes"]
        if lane["diagnostic_code"] == "orphaned_workspace_change"
    )
    assert workspace_lane["title"] == "unstaged:src/gabion/synthetic_boundaries.py"
    assert workspace_lane["candidate_owner_status"] == "ambiguous_exact_path_owner"
    assert workspace_lane["candidate_owner_object_ids"] == [
        "CSA-IDR",
        "CSA-IGM",
        "CSA-IVL",
        "CSA-RGC",
        "SCC",
    ]
    workspace_commit_unit = payload["repo_next_actions"][
        "recommended_workspace_commit_unit"
    ]
    assert workspace_commit_unit is not None
    assert workspace_commit_unit["followup_family"] == "workspace_orphan_resolution"
    assert workspace_commit_unit["diagnostic_code"] == "orphaned_workspace_change"
    assert workspace_commit_unit["owner_scope_kind"] == "ambiguous_owner_set"
    assert workspace_commit_unit["queue_id"].startswith(
        "planner_queue|followup_family=workspace_orphan_resolution|"
    )
    assert workspace_commit_unit["root_object_ids"] == [
        "CSA-IDR",
        "CSA-IGM",
        "CSA-IVL",
        "CSA-RGC",
        "SCC",
    ]
    assert workspace_commit_unit["rel_paths"] == [
        "src/gabion/synthetic_boundaries.py"
    ]

    artifact_path = root / "artifacts" / "out" / "invariant_workstreams.json"
    invariant_graph.write_invariant_workstreams(artifact_path, workstreams)
    written_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert written_payload["diagnostic_summary"]["workspace_preservation_count"] == 0
    assert written_payload["diagnostic_summary"]["orphaned_workspace_change_count"] == 1
    assert any(
        lane["diagnostic_code"] == "orphaned_workspace_change"
        for lane in written_payload["repo_next_actions"]["diagnostic_lanes"]
    )
    assert (
        written_payload["repo_next_actions"]["recommended_workspace_commit_unit"][
            "diagnostic_code"
        ]
        == "orphaned_workspace_change"
    )


def test_build_invariant_graph_joins_cross_origin_witness_contract_artifact(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "artifacts" / "out" / "cross_origin_witness_contract.json",
        {
            "schema_version": 1,
            "artifact_kind": "cross_origin_witness_contract",
            "producer": "tests",
            "cases": [
                {
                    "case_key": "analysis_union_path_remap",
                    "case_kind": "cross_origin_path_remap",
                    "title": "analysis witness to union-view path remap",
                    "status": "pass",
                    "summary": "rows=1 mismatches=0",
                    "left_label": "analysis_input_witness",
                    "right_label": "aspf_union_view",
                    "evidence_paths": [
                        "src/gabion/server.py",
                        "src/gabion/tooling/policy_substrate/aspf_union_view.py",
                    ],
                    "row_keys": [
                        "path_remap:src/gabion/sample_alpha.py",
                    ],
                    "field_checks": [
                        {
                            "field_name": "manifest_digest_present",
                            "matches": True,
                            "left_value": "true",
                            "right_value": "true",
                        }
                    ],
                }
            ],
            "witness_rows": [
                {
                    "row_key": "path_remap:src/gabion/sample_alpha.py",
                    "row_kind": "path_remap",
                    "left_origin_kind": "analysis_input_witness.file",
                    "left_origin_key": "src/gabion/sample_alpha.py",
                    "right_origin_kind": "aspf_union_view.module",
                    "right_origin_key": "src/gabion/sample_alpha.py",
                    "remap_key": "src/gabion/sample_alpha.py",
                    "summary": "analysis witness file remapped to union module",
                }
            ],
        },
    )

    graph = invariant_graph.build_invariant_graph(
        tmp_path,
        declared_registries=_NO_DECLARED_REGISTRIES,
    )
    node_kind_counts = graph.as_payload()["counts"]["node_kind_counts"]
    assert node_kind_counts["cross_origin_witness_report"] == 1
    assert node_kind_counts["cross_origin_witness_case"] == 1
    assert node_kind_counts["cross_origin_witness_row"] == 1

    report_node = next(
        node for node in graph.nodes if node.node_kind == "cross_origin_witness_report"
    )
    case_node = next(
        node for node in graph.nodes if node.node_kind == "cross_origin_witness_case"
    )
    row_node = next(
        node
        for node in graph.nodes
        if node.node_kind == "cross_origin_witness_row"
        and "src/gabion/sample_alpha.py" in node.object_ids
    )
    edges = {(edge.edge_kind, edge.source_id, edge.target_id) for edge in graph.edges}

    assert any(
        item.startswith("cross_origin_witness_contract:")
        for item in report_node.object_ids
    )
    assert ("contains", report_node.node_id, case_node.node_id) in edges
    assert ("contains", case_node.node_id, row_node.node_id) in edges
    assert invariant_graph.trace_nodes(graph, "analysis_union_path_remap")
    assert invariant_graph.trace_nodes(graph, "src/gabion/sample_alpha.py")


def test_build_invariant_graph_splits_orphan_policy_signals_by_source_seed(
    tmp_path: Path,
) -> None:
    root = _sample_repo(tmp_path)
    (root / "artifacts" / "out").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "out" / "ambiguity_contract_policy_check.json").write_text(
        json.dumps(
            {
                "ast": {"violations": []},
                "grade": {
                    "violations": [
                        {
                            "rule_id": "GMP-001",
                            "message": "alpha grade issue",
                            "path": "src/gabion/analysis/alpha/emitters.py",
                            "line": 10,
                            "qualname": "gabion.analysis.alpha.emitters.emit_alpha",
                            "details": {},
                        },
                        {
                            "rule_id": "GMP-001",
                            "message": "beta grade issue",
                            "path": "src/gabion/analysis/beta/renderers.py",
                            "line": 20,
                            "qualname": "gabion.analysis.beta.renderers.emit_beta",
                            "details": {},
                        },
                    ]
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    graph = invariant_graph.build_invariant_graph(
        root,
        declared_registries=_NO_DECLARED_REGISTRIES,
    )
    policy_signal_nodes = sorted(
        [
            node
            for node in graph.nodes
            if node.node_kind == "policy_signal"
        ],
        key=lambda node: (node.rel_path, node.qualname),
    )

    assert len(policy_signal_nodes) == 2
    assert [node.rel_path for node in policy_signal_nodes] == [
        "src/gabion/analysis/alpha/emitters.py",
        "src/gabion/analysis/beta/renderers.py",
    ]


def test_build_invariant_graph_fails_closed_on_declared_workstream_dependency(
    tmp_path: Path,
) -> None:
    root = tmp_path
    _write(root / "src" / "gabion" / "__init__.py", "")
    _write(
        root / "src" / "gabion" / "broken.py",
        "\n".join(
            [
                "from gabion.invariants import todo_decorator",
                "",
                "@todo_decorator(",
                "    reason='broken dependency',",
                "    owner='policy',",
                "    expiry='2099-01-01',",
                "    reasoning={",
                "        'summary': 'declared workstream dependency should fail closed',",
                "        'control': 'broken.dependency',",
                "        'blocking_dependencies': ['PRF-999'],",
                "    },",
                ")",
                "def broken() -> None:",
                "    return None",
            ]
        ),
    )

    import pytest

    with pytest.raises(ValueError, match="declared workstream blocking dependency"):
        invariant_graph.build_invariant_graph(
            root,
            declared_registries=_NO_DECLARED_REGISTRIES,
        )


def test_compare_invariant_workstreams_classifies_reduced_and_relocated() -> None:
    before_payload = _synthetic_workstreams_payload(
        [
            {
                "object_id": "WS-REDUCE",
                "title": "reduce stream",
                "doc_ids": ["ledger.reduce"],
                "status": "in_progress",
                "touchsite_count": 2,
                "surviving_touchsite_count": 1,
                "touchpoints": [
                    {
                        "touchsites": [
                            {"object_id": "TS-1"},
                            {"object_id": "TS-2"},
                        ]
                    }
                ],
                "health_summary": {
                    "ready_touchsite_count": 0,
                    "coverage_gap_touchsite_count": 2,
                    "policy_blocked_touchsite_count": 0,
                    "diagnostic_blocked_touchsite_count": 0,
                },
                "next_actions": {
                    "dominant_blocker_class": "coverage_gap",
                    "recommended_cut": {"object_id": "TP-REDUCE-1"},
                },
            },
            {
                "object_id": "WS-RELOCATE",
                "title": "relocate stream",
                "doc_ids": ["ledger.relocate"],
                "status": "in_progress",
                "touchsite_count": 1,
                "surviving_touchsite_count": 1,
                "touchpoints": [
                    {
                        "touchsites": [
                            {"object_id": "TS-OLD"},
                        ]
                    }
                ],
                "health_summary": {
                    "ready_touchsite_count": 0,
                    "coverage_gap_touchsite_count": 1,
                    "policy_blocked_touchsite_count": 0,
                    "diagnostic_blocked_touchsite_count": 0,
                },
                "next_actions": {
                    "dominant_blocker_class": "coverage_gap",
                    "recommended_cut": {"object_id": "TP-OLD"},
                },
            },
        ]
    )
    after_payload = _synthetic_workstreams_payload(
        [
            {
                "object_id": "WS-REDUCE",
                "title": "reduce stream",
                "doc_ids": ["ledger.reduce"],
                "status": "in_progress",
                "touchsite_count": 1,
                "surviving_touchsite_count": 1,
                "touchpoints": [
                    {
                        "touchsites": [
                            {"object_id": "TS-1"},
                        ]
                    }
                ],
                "health_summary": {
                    "ready_touchsite_count": 1,
                    "coverage_gap_touchsite_count": 0,
                    "policy_blocked_touchsite_count": 0,
                    "diagnostic_blocked_touchsite_count": 0,
                },
                "next_actions": {
                    "dominant_blocker_class": "ready_structural",
                    "recommended_cut": {"object_id": "TP-REDUCE-2"},
                },
            },
            {
                "object_id": "WS-RELOCATE",
                "title": "relocate stream",
                "doc_ids": ["ledger.relocate"],
                "status": "in_progress",
                "touchsite_count": 1,
                "surviving_touchsite_count": 1,
                "touchpoints": [
                    {
                        "touchsites": [
                            {"object_id": "TS-NEW"},
                        ]
                    }
                ],
                "health_summary": {
                    "ready_touchsite_count": 0,
                    "coverage_gap_touchsite_count": 0,
                    "policy_blocked_touchsite_count": 1,
                    "diagnostic_blocked_touchsite_count": 0,
                },
                "next_actions": {
                    "dominant_blocker_class": "policy_blocked",
                    "recommended_cut": {"object_id": "TP-NEW"},
                },
            },
        ]
    )

    drifts = invariant_graph.compare_invariant_workstreams(before_payload, after_payload)
    by_object_id = {item.object_id: item for item in drifts}

    assert by_object_id["WS-REDUCE"].classification == "reduced"
    assert by_object_id["WS-REDUCE"].touchsite_delta == -1
    assert by_object_id["WS-REDUCE"].blocker_deltas["coverage_gap_touchsite_count"] == -2
    assert by_object_id["WS-REDUCE"].blocker_deltas["ready_touchsite_count"] == 1
    assert by_object_id["WS-REDUCE"].removed_touchsite_ids == ("TS-2",)

    assert by_object_id["WS-RELOCATE"].classification == "relocated"
    assert by_object_id["WS-RELOCATE"].touchsite_delta == 0
    assert by_object_id["WS-RELOCATE"].before_dominant_blocker_class == "coverage_gap"
    assert by_object_id["WS-RELOCATE"].after_dominant_blocker_class == "policy_blocked"
    assert by_object_id["WS-RELOCATE"].added_touchsite_ids == ("TS-NEW",)
    assert by_object_id["WS-RELOCATE"].removed_touchsite_ids == ("TS-OLD",)


def test_compare_invariant_ledger_projections_synthesizes_doc_targets_and_actions() -> None:
    before_payload = _synthetic_workstreams_payload(
        [
            {
                "object_id": "WS-REDUCE",
                "title": "reduce stream",
                "doc_ids": ["ledger.reduce"],
                "status": "in_progress",
                "touchsite_count": 2,
                "surviving_touchsite_count": 1,
                "touchpoints": [{"touchsites": [{"object_id": "TS-1"}, {"object_id": "TS-2"}]}],
                "health_summary": {
                    "ready_touchsite_count": 0,
                    "coverage_gap_touchsite_count": 2,
                    "policy_blocked_touchsite_count": 0,
                    "diagnostic_blocked_touchsite_count": 0,
                },
                "next_actions": {
                    "dominant_blocker_class": "coverage_gap",
                    "recommended_cut": {"object_id": "TP-BEFORE"},
                },
            }
        ]
    )
    after_payload = _synthetic_workstreams_payload(
        [
            {
                "object_id": "WS-REDUCE",
                "title": "reduce stream",
                "doc_ids": ["ledger.reduce"],
                "status": "in_progress",
                "touchsite_count": 1,
                "surviving_touchsite_count": 1,
                "touchpoints": [{"touchsites": [{"object_id": "TS-1"}]}],
                "health_summary": {
                    "ready_touchsite_count": 1,
                    "coverage_gap_touchsite_count": 0,
                    "policy_blocked_touchsite_count": 0,
                    "diagnostic_blocked_touchsite_count": 0,
                },
                "next_actions": {
                    "dominant_blocker_class": "ready_structural",
                    "recommended_cut": {"object_id": "TP-AFTER"},
                },
            }
        ]
    )

    deltas = invariant_graph.compare_invariant_ledger_projections(
        before_payload,
        after_payload,
    )
    assert len(deltas) == 1
    delta = deltas[0]
    assert delta.object_id == "WS-REDUCE"
    assert delta.title == "reduce stream"
    assert delta.target_doc_ids == ("ledger.reduce",)
    assert delta.classification == "reduced"
    assert delta.recommended_ledger_action == "append_reduction_delta"
    assert "recommended cut TP-BEFORE->TP-AFTER" in delta.summary
    assert delta.as_payload()["append_entry"]["recommended_ledger_action"] == (
        "append_reduction_delta"
    )


def test_build_invariant_ledger_delta_projections_groups_append_targets() -> None:
    before_payload = _synthetic_workstreams_payload(
        [
            {
                "object_id": "WS-REDUCE",
                "title": "reduce stream",
                "doc_ids": ["ledger.reduce"],
                "status": "in_progress",
                "touchsite_count": 2,
                "surviving_touchsite_count": 1,
                "touchpoints": [{"touchsites": [{"object_id": "TS-1"}, {"object_id": "TS-2"}]}],
                "health_summary": {
                    "ready_touchsite_count": 0,
                    "coverage_gap_touchsite_count": 2,
                    "policy_blocked_touchsite_count": 0,
                    "diagnostic_blocked_touchsite_count": 0,
                },
                "next_actions": {
                    "dominant_blocker_class": "coverage_gap",
                    "recommended_cut": {"object_id": "TP-BEFORE"},
                },
            }
        ]
    )
    after_payload = _synthetic_workstreams_payload(
        [
            {
                "object_id": "WS-REDUCE",
                "title": "reduce stream",
                "doc_ids": ["ledger.reduce"],
                "status": "in_progress",
                "touchsite_count": 1,
                "surviving_touchsite_count": 1,
                "touchpoints": [{"touchsites": [{"object_id": "TS-1"}]}],
                "health_summary": {
                    "ready_touchsite_count": 1,
                    "coverage_gap_touchsite_count": 0,
                    "policy_blocked_touchsite_count": 0,
                    "diagnostic_blocked_touchsite_count": 0,
                },
                "next_actions": {
                    "dominant_blocker_class": "ready_structural",
                    "recommended_cut": {"object_id": "TP-AFTER"},
                },
            }
        ]
    )

    projections = invariant_graph.build_invariant_ledger_delta_projections(
        root="/tmp/root",
        before_workstreams_artifact="/tmp/before.json",
        after_workstreams_artifact="/tmp/after.json",
        before_payload=before_payload,
        after_payload=after_payload,
    )

    payload = projections.as_payload()
    assert payload["counts"]["delta_count"] == 1
    assert payload["counts"]["target_doc_count"] == 1
    assert payload["deltas"][0]["title"] == "reduce stream"
    grouped = projections.grouped_by_target_doc_id()
    assert grouped[0][0] == "ledger.reduce"
    assert grouped[0][1][0].object_id == "WS-REDUCE"


def test_build_invariant_ledger_alignments_detects_append_pending_existing_object(
    tmp_path: Path,
) -> None:
    before_payload = _synthetic_workstreams_payload(
        [
            {
                "object_id": "WS-REDUCE",
                "title": "reduce stream",
                "doc_ids": ["ledger.reduce"],
                "status": "in_progress",
                "touchsite_count": 2,
                "surviving_touchsite_count": 1,
                "touchpoints": [{"touchsites": [{"object_id": "TS-1"}, {"object_id": "TS-2"}]}],
                "health_summary": {
                    "ready_touchsite_count": 0,
                    "coverage_gap_touchsite_count": 2,
                    "policy_blocked_touchsite_count": 0,
                    "diagnostic_blocked_touchsite_count": 0,
                },
                "next_actions": {
                    "dominant_blocker_class": "coverage_gap",
                    "recommended_cut": {"object_id": "TP-BEFORE"},
                },
            }
        ]
    )
    after_payload = _synthetic_workstreams_payload(
        [
            {
                "object_id": "WS-REDUCE",
                "title": "reduce stream",
                "doc_ids": ["ledger.reduce"],
                "status": "in_progress",
                "touchsite_count": 1,
                "surviving_touchsite_count": 1,
                "touchpoints": [{"touchsites": [{"object_id": "TS-1"}]}],
                "health_summary": {
                    "ready_touchsite_count": 1,
                    "coverage_gap_touchsite_count": 0,
                    "policy_blocked_touchsite_count": 0,
                    "diagnostic_blocked_touchsite_count": 0,
                },
                "next_actions": {
                    "dominant_blocker_class": "ready_structural",
                    "recommended_cut": {"object_id": "TP-AFTER"},
                },
            }
        ]
    )
    projections = invariant_graph.build_invariant_ledger_delta_projections(
        root=str(tmp_path),
        before_workstreams_artifact=str(tmp_path / "before.json"),
        after_workstreams_artifact=str(tmp_path / "after.json"),
        before_payload=before_payload,
        after_payload=after_payload,
    )
    doc_path = tmp_path / "docs" / "ledger_reduce.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(
        "---\n"
        "doc_revision: 1\n"
        "doc_id: ledger.reduce\n"
        "---\n\n"
        "# Ledger\n\n"
        "Existing object reference: `WS-REDUCE`.\n",
        encoding="utf-8",
    )

    alignments = invariant_graph.build_invariant_ledger_alignments(
        root=tmp_path,
        ledger_deltas=projections,
    )
    payload = alignments.as_payload()
    assert payload["counts"]["alignment_count"] == 1
    assert payload["counts"]["status_counts"]["append_pending_existing_object"] == 1
    assert payload["alignments"][0]["target_doc_path"] == "docs/ledger_reduce.md"
    assert payload["alignments"][0]["object_reference_present"] is True
    assert payload["alignments"][0]["summary_present"] is False


def test_runtime_invariant_graph_cli_compare_reports_workstream_drift(
    tmp_path: Path,
    capsys,
) -> None:
    doc_path = tmp_path / "docs" / "ledger_reduce.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(
        "---\n"
        "doc_revision: 1\n"
        "doc_id: ledger.reduce\n"
        "---\n\n"
        "# Ledger\n\n"
        "Existing object reference: `WS-REDUCE`.\n",
        encoding="utf-8",
    )
    before_artifact = tmp_path / "before_workstreams.json"
    after_artifact = tmp_path / "after_workstreams.json"
    ledger_deltas_artifact = tmp_path / "ledger_deltas.json"
    ledger_deltas_markdown_artifact = tmp_path / "ledger_deltas.md"
    ledger_alignments_artifact = tmp_path / "ledger_alignments.json"
    ledger_alignments_markdown_artifact = tmp_path / "ledger_alignments.md"
    _write_json(
        before_artifact,
        _synthetic_workstreams_payload(
            [
                {
                    "object_id": "WS-REDUCE",
                    "title": "reduce stream",
                    "doc_ids": ["ledger.reduce"],
                    "status": "in_progress",
                    "touchsite_count": 2,
                    "surviving_touchsite_count": 1,
                    "touchpoints": [
                        {"touchsites": [{"object_id": "TS-1"}, {"object_id": "TS-2"}]}
                    ],
                    "health_summary": {
                        "ready_touchsite_count": 0,
                        "coverage_gap_touchsite_count": 2,
                        "policy_blocked_touchsite_count": 0,
                        "diagnostic_blocked_touchsite_count": 0,
                    },
                    "next_actions": {
                        "dominant_blocker_class": "coverage_gap",
                        "recommended_cut": {"object_id": "TP-BEFORE"},
                    },
                }
            ]
        ),
    )
    _write_json(
        after_artifact,
        _synthetic_workstreams_payload(
            [
                {
                    "object_id": "WS-REDUCE",
                    "title": "reduce stream",
                    "doc_ids": ["ledger.reduce"],
                    "status": "in_progress",
                    "touchsite_count": 1,
                    "surviving_touchsite_count": 1,
                    "touchpoints": [{"touchsites": [{"object_id": "TS-1"}]}],
                    "health_summary": {
                        "ready_touchsite_count": 1,
                        "coverage_gap_touchsite_count": 0,
                        "policy_blocked_touchsite_count": 0,
                        "diagnostic_blocked_touchsite_count": 0,
                    },
                    "next_actions": {
                        "dominant_blocker_class": "ready_structural",
                        "recommended_cut": {"object_id": "TP-AFTER"},
                    },
                }
            ]
        ),
    )

    assert (
        invariant_graph_runtime.main(
            [
                "compare",
                "--root",
                str(tmp_path),
                "--before-workstreams-artifact",
                str(before_artifact),
                "--after-workstreams-artifact",
                str(after_artifact),
                "--ledger-deltas-artifact",
                str(ledger_deltas_artifact),
                "--ledger-deltas-markdown-artifact",
                str(ledger_deltas_markdown_artifact),
                "--ledger-alignments-artifact",
                str(ledger_alignments_artifact),
                "--ledger-alignments-markdown-artifact",
                str(ledger_alignments_markdown_artifact),
            ]
        )
        == 0
    )
    compare_output = capsys.readouterr().out
    assert "classification_counts:" in compare_output
    assert "- reduced: 1" in compare_output
    assert (
        "- WS-REDUCE :: reduced :: touchsites=2->1 (-1) :: surviving=1->1 (+0) :: dominant=coverage_gap->ready_structural :: recommended=TP-BEFORE->TP-AFTER"
        in compare_output
    )
    assert (
        "  blocker_deltas: ready=+1 :: coverage_gap=-2 :: policy=+0 :: diagnostic=+0"
        in compare_output
    )
    assert "ledger_deltas:" in compare_output
    assert (
        "- WS-REDUCE :: reduced :: action=append_reduction_delta :: docs=ledger.reduce"
        in compare_output
    )
    assert "summary: reduce stream reduced: touchsites 2->1" in compare_output
    assert f"ledger_delta_artifact: {ledger_deltas_artifact}" in compare_output
    assert (
        f"ledger_delta_markdown_artifact: {ledger_deltas_markdown_artifact}"
        in compare_output
    )
    assert "ledger_alignment_counts:" in compare_output
    assert "- append_pending_existing_object: 1" in compare_output
    assert f"ledger_alignment_artifact: {ledger_alignments_artifact}" in compare_output
    assert (
        f"ledger_alignment_markdown_artifact: {ledger_alignments_markdown_artifact}"
        in compare_output
    )
    ledger_deltas_payload = json.loads(
        ledger_deltas_artifact.read_text(encoding="utf-8")
    )
    assert ledger_deltas_payload["counts"]["delta_count"] == 1
    assert ledger_deltas_payload["deltas"][0]["append_entry"]["object_id"] == "WS-REDUCE"
    ledger_alignments_payload = json.loads(
        ledger_alignments_artifact.read_text(encoding="utf-8")
    )
    assert (
        ledger_alignments_payload["counts"]["status_counts"][
            "append_pending_existing_object"
        ]
        == 1
    )
    assert (
        ledger_alignments_payload["alignments"][0]["target_doc_path"]
        == "docs/ledger_reduce.md"
    )
    markdown_output = ledger_deltas_markdown_artifact.read_text(encoding="utf-8")
    assert "# Invariant Ledger Deltas" in markdown_output
    assert "## ledger.reduce" in markdown_output
    assert "### WS-REDUCE :: reduced" in markdown_output
    assert "reduce stream reduced: touchsites 2->1" in markdown_output
    alignment_markdown_output = ledger_alignments_markdown_artifact.read_text(
        encoding="utf-8"
    )
    assert "# Invariant Ledger Alignments" in alignment_markdown_output
    assert "## ledger.reduce" in alignment_markdown_output
    assert "### WS-REDUCE :: append_pending_existing_object" in alignment_markdown_output


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
    assert "target_doc_ids: projection_semantic_fragment_ledger, projection_semantic_fragment_rfc" in ledger_output
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
