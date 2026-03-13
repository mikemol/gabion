from __future__ import annotations

import json
from pathlib import Path

from tests.path_helpers import REPO_ROOT

from gabion.analysis.aspf.aspf_lattice_algebra import ReplayableStream
from gabion.tooling.policy_substrate import invariant_graph
from gabion.tooling.policy_substrate.policy_queue_identity import PolicyQueueIdentitySpace
from gabion.tooling.runtime import invariant_graph as invariant_graph_runtime


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _disable_phase5_enricher(monkeypatch) -> None:
    monkeypatch.setattr(invariant_graph, "iter_phase5_queues", lambda: ())
    monkeypatch.setattr(invariant_graph, "iter_phase5_subqueues", lambda: ())
    monkeypatch.setattr(invariant_graph, "iter_phase5_touchpoints", lambda: ())
    monkeypatch.setattr(invariant_graph, "iter_prf_queues", lambda: ())
    monkeypatch.setattr(invariant_graph, "iter_prf_subqueues", lambda: ())


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
                "    links=[{'kind': 'object_id', 'value': 'OBJ-TODO'}],",
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
    return tmp_path


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
    monkeypatch,
) -> None:
    _disable_phase5_enricher(monkeypatch)
    root = _sample_repo(tmp_path)

    graph = invariant_graph.build_invariant_graph(root)
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
    monkeypatch,
) -> None:
    _disable_phase5_enricher(monkeypatch)
    root = _sample_repo(tmp_path)
    artifact = tmp_path / "artifacts/out/invariant_graph.json"

    graph = invariant_graph.build_invariant_graph(root)
    invariant_graph.write_invariant_graph(artifact, graph)
    reloaded = invariant_graph.load_invariant_graph(artifact)

    assert artifact.exists()
    assert len(reloaded.nodes) == len(graph.nodes)
    assert len(reloaded.edges) == len(graph.edges)
    assert len(reloaded.diagnostics) == len(graph.diagnostics)


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
    assert projection["remaining_touchsite_count"] == 73
    assert projection["collapsible_touchsite_count"] == 47
    assert projection["surviving_touchsite_count"] == 26
    assert len(projection["subqueues"]) == 5
    assert len(projection["touchpoints"]) == 6
    workstreams_payload = workstreams.as_payload()
    assert workstreams_payload["diagnostic_summary"] == {
        "diagnostic_count": 7,
        "unmatched_policy_signal_count": 7,
        "unresolved_blocking_dependency_count": 0,
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
        "utility_score": 1100,
        "utility_reason": "governance_orphan:seed_new_owner+owner_option_tradeoff:100",
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
        ],
        "selection_certainty_kind": "frontier_plateau",
        "cofrontier_followup_count": 7,
        "cofrontier_followup_cohort": [
            {
                "followup_family": "governance_orphan_resolution",
                "followup_class": "governance",
                "action_kind": "diagnostic_resolution",
                "object_id": None,
                "diagnostic_code": "unmatched_policy_signal",
                "target_doc_id": None,
                "policy_ids": [f"GMP-{value:03d}"],
                "title": (
                    "seed ownership for grade:GMP-{index} from "
                    "src/gabion/analysis/dataflow/io"
                ).format(index=f"{value:03d}"),
                "utility_score": 1100,
                "selection_rank": value,
                "selection_reason": (
                    "frontier_tiebreak_winner"
                    if value == 1
                    else f"policy_ids:GMP-{value:03d}"
                ),
            }
            for value in range(1, 8)
        ],
        "selection_scope_kind": "shared_owner_resolution_surface",
        "selection_scope_id": "seed_new_owner:WS-SEED:gabion.analysis.dataflow.io",
        "runner_up_followup_family": "governance_orphan_resolution",
        "runner_up_followup_class": "governance",
        "runner_up_followup_object_id": None,
        "runner_up_followup_utility_score": 1100,
        "frontier_choice_margin_score": 0,
        "frontier_choice_margin_reason": "cofrontier",
        "frontier_choice_margin_components": [],
        "selection_rank": 1,
        "opportunity_cost_score": 0,
        "opportunity_cost_reason": "frontier",
        "opportunity_cost_components": [],
        "count": 1,
    }
    assert workstreams_payload["repo_next_actions"]["dominant_followup_class"] == (
        "governance"
    )
    assert workstreams_payload["repo_next_actions"]["next_human_followup_family"] == (
        "governance_orphan_resolution"
    )
    assert workstreams_payload["repo_next_actions"]["recommended_code_followup"] == {
        "followup_family": "structural_cut",
        "action_kind": "touchpoint_cut",
        "priority_rank": 100,
        "object_id": "PSF-007-TP-005",
        "owner_object_id": "PSF-007",
        "diagnostic_code": None,
        "target_doc_id": None,
        "policy_ids": [],
        "title": "projection_exec_plan.py planning surfaces",
        "blocker_class": "ready_structural",
        "readiness_class": "ready_structural",
        "alignment_status": None,
        "recommended_action": None,
        "owner_seed_path": None,
        "owner_seed_object_id": None,
        "owner_resolution_kind": None,
        "owner_resolution_score": None,
        "owner_resolution_options": [],
        "runner_up_owner_object_id": None,
        "runner_up_owner_resolution_kind": None,
        "runner_up_owner_resolution_score": None,
        "owner_choice_margin_score": None,
        "owner_choice_margin_reason": None,
        "owner_choice_margin_components": [],
        "owner_option_tradeoff_score": None,
        "owner_option_tradeoff_reason": None,
        "owner_option_tradeoff_components": [],
        "utility_score": 700,
        "utility_reason": "code:ready_structural",
        "utility_components": [
            {
                "kind": "code_touchpoint_base",
                "score": 450,
                "rationale": "code:touchpoint_cut",
            },
            {
                "kind": "readiness_bonus",
                "score": 250,
                "rationale": "readiness:ready_structural",
            },
        ],
        "selection_certainty_kind": "ranked_unique",
        "cofrontier_followup_count": 1,
        "cofrontier_followup_cohort": [
            {
                "followup_family": "structural_cut",
                "followup_class": "code",
                "action_kind": "touchpoint_cut",
                "object_id": "PSF-007-TP-005",
                "diagnostic_code": None,
                "target_doc_id": None,
                "policy_ids": [],
                "title": "projection_exec_plan.py planning surfaces",
                "utility_score": 700,
                "selection_rank": 1,
                "selection_reason": "frontier_tiebreak_winner",
            }
        ],
        "selection_scope_kind": "singleton",
        "selection_scope_id": None,
        "runner_up_followup_family": "coverage_gap",
        "runner_up_followup_class": "code",
        "runner_up_followup_object_id": "PSF-007-TP-001",
        "runner_up_followup_utility_score": 480,
        "frontier_choice_margin_score": 220,
        "frontier_choice_margin_reason": "code:ready_structural->code:coverage_gap",
        "frontier_choice_margin_components": [
            {
                "kind": "code_touchpoint_base",
                "score": 450,
                "rationale": "code:touchpoint_cut",
            },
            {
                "kind": "readiness_bonus",
                "score": 250,
                "rationale": "readiness:ready_structural",
            },
            {
                "kind": "runner_up_offset:code_touchpoint_base",
                "score": -450,
                "rationale": "code:touchpoint_cut",
            },
            {
                "kind": "runner_up_offset:readiness_bonus",
                "score": -30,
                "rationale": "readiness:coverage_gap",
            },
        ],
        "selection_rank": 8,
        "opportunity_cost_score": 400,
        "opportunity_cost_reason": "governance_orphan:seed_new_owner+owner_option_tradeoff:100->code:ready_structural",
        "opportunity_cost_components": [
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
                "kind": "runner_up_offset:code_touchpoint_base",
                "score": -450,
                "rationale": "code:touchpoint_cut",
            },
            {
                "kind": "runner_up_offset:readiness_bonus",
                "score": -250,
                "rationale": "readiness:ready_structural",
            },
        ],
        "count": 1,
    }
    assert workstreams_payload["repo_next_actions"]["recommended_human_followup"] == (
        recommended_followup
    )
    assert workstreams_payload["repo_next_actions"]["recommended_followup_lane"] == {
        "followup_family": "governance_orphan_resolution",
        "followup_class": "governance",
        "action_count": 7,
        "strongest_owner_resolution_kind": "seed_new_owner",
        "strongest_owner_resolution_score": 100,
        "strongest_utility_score": 1100,
        "strongest_utility_reason": "governance_orphan:seed_new_owner+owner_option_tradeoff:100",
        "lane_utility_score": 1160,
        "lane_utility_reason": (
            "governance_orphan:seed_new_owner+owner_option_tradeoff:100+lane_breadth:7+lane:governance_orphan_resolution"
        ),
        "lane_utility_components": [
            {
                "kind": "best_followup_utility",
                "score": 1100,
                "rationale": "governance_orphan:seed_new_owner+owner_option_tradeoff:100",
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
    assert workstreams_payload["repo_next_actions"]["recommended_code_followup_lane"] == {
        "followup_family": "structural_cut",
        "followup_class": "code",
        "action_count": 1,
        "strongest_owner_resolution_kind": None,
        "strongest_owner_resolution_score": None,
        "strongest_utility_score": 700,
        "strongest_utility_reason": "code:ready_structural",
        "lane_utility_score": 715,
        "lane_utility_reason": "code:ready_structural+lane_breadth:1+lane:structural_cut",
        "lane_utility_components": [
            {
                "kind": "best_followup_utility",
                "score": 700,
                "rationale": "code:ready_structural",
            },
            {
                "kind": "lane_breadth_bonus",
                "score": 5,
                "rationale": "lane_breadth:1",
            },
            {
                "kind": "lane_class_bonus",
                "score": 10,
                "rationale": "lane_class:code",
            },
        ],
        "selection_rank": 2,
        "opportunity_cost_score": 445,
        "opportunity_cost_reason": "deferred_by:governance_orphan_resolution",
        "opportunity_cost_components": [
            {
                "kind": "best_followup_utility_gap",
                "score": 400,
                "rationale": "governance_orphan:seed_new_owner+owner_option_tradeoff:100->code:ready_structural",
                },
            {
                "kind": "lane_breadth_bonus_gap",
                "score": 30,
                "rationale": "lane_breadth:7->lane_breadth:1",
            },
            {
                "kind": "lane_class_bonus_gap",
                "score": 15,
                "rationale": "lane_class:governance->lane_class:code",
            },
        ],
        "best_followup": workstreams_payload["repo_next_actions"]["recommended_code_followup"],
    }
    assert (
        workstreams_payload["repo_next_actions"]["recommended_human_followup_lane"]
        == workstreams_payload["repo_next_actions"]["recommended_followup_lane"]
    )
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
    assert repo_followup_lanes[0] == {
        "followup_family": "governance_orphan_resolution",
        "followup_class": "governance",
        "action_count": 7,
        "strongest_owner_resolution_kind": "seed_new_owner",
        "strongest_owner_resolution_score": 100,
        "strongest_utility_score": 1100,
        "strongest_utility_reason": "governance_orphan:seed_new_owner+owner_option_tradeoff:100",
        "lane_utility_score": 1160,
        "lane_utility_reason": (
            "governance_orphan:seed_new_owner+owner_option_tradeoff:100+lane_breadth:7+lane:governance_orphan_resolution"
        ),
        "lane_utility_components": [
            {
                "kind": "best_followup_utility",
                "score": 1100,
                "rationale": "governance_orphan:seed_new_owner+owner_option_tradeoff:100",
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
    assert repo_followup_lanes[1]["followup_family"] == "structural_cut"
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
    assert projected_ids == ["PRF", "PSF-007"]
    prf = next(
        item
        for item in workstreams_payload["workstreams"]
        if isinstance(item, dict) and item.get("object_id") == "PRF"
    )
    assert prf["status"] == "landed"
    assert prf["touchsite_count"] == 0
    assert prf["doc_ids"] == ["policy_rule_frontmatter_migration_ledger"]
    psf = next(
        item
        for item in workstreams_payload["workstreams"]
        if isinstance(item, dict) and item.get("object_id") == "PSF-007"
    )
    assert psf["touchsite_count"] == 73
    assert psf["collapsible_touchsite_count"] == 47
    assert psf["surviving_touchsite_count"] == 26
    assert "projection_semantic_fragment_ledger" in psf["doc_ids"]
    assert "projection_semantic_fragment_rfc" in psf["doc_ids"]
    assert psf["health_summary"]["covered_touchsite_count"] == 3
    assert psf["health_summary"]["uncovered_touchsite_count"] == 70
    assert psf["health_summary"]["governed_touchsite_count"] == 0
    assert psf["health_summary"]["diagnosed_touchsite_count"] == 0
    assert psf["health_summary"]["ready_touchsite_count"] == 3
    assert psf["health_summary"]["coverage_gap_touchsite_count"] == 70
    assert psf["health_summary"]["policy_blocked_touchsite_count"] == 0
    assert psf["health_summary"]["diagnostic_blocked_touchsite_count"] == 0
    assert psf["health_summary"]["ready_touchpoint_cut_count"] == 1
    assert psf["health_summary"]["coverage_gap_touchpoint_cut_count"] == 5
    assert psf["health_summary"]["ready_subqueue_cut_count"] == 0
    assert psf["health_summary"]["coverage_gap_subqueue_cut_count"] == 5
    assert psf["next_actions"]["recommended_cut"]["object_id"] == "PSF-007-TP-005"
    assert psf["next_actions"]["recommended_cut"]["cut_kind"] == "touchpoint_cut"
    assert psf["next_actions"]["recommended_cut"]["touchsite_count"] == 1
    assert psf["next_actions"]["recommended_ready_cut"]["object_id"] == "PSF-007-TP-005"
    assert psf["next_actions"]["recommended_ready_cut"]["readiness_class"] == (
        "ready_structural"
    )
    assert psf["next_actions"]["recommended_coverage_gap_cut"]["object_id"] == (
        "PSF-007-TP-001"
    )
    assert psf["next_actions"]["recommended_coverage_gap_cut"]["readiness_class"] == (
        "coverage_gap"
    )
    assert psf["next_actions"]["dominant_blocker_class"] == "coverage_gap"
    assert psf["next_actions"]["recommended_remediation_family"] == "structural_cut"
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
        "followup_family": "structural_cut",
        "action_kind": "touchpoint_cut",
        "priority_rank": 0,
        "object_id": "PSF-007-TP-005",
        "owner_object_id": "PSF-007-SQ-005",
        "target_doc_id": None,
        "title": "projection_exec_plan.py planning surfaces",
        "blocker_class": "ready_structural",
        "readiness_class": "ready_structural",
        "alignment_status": None,
        "recommended_action": None,
        "touchsite_count": 1,
        "collapsible_touchsite_count": 0,
        "surviving_touchsite_count": 1,
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
    assert psf["next_actions"]["remediation_lanes"][0]["best_cut"]["object_id"] == (
        "PSF-007-TP-005"
    )
    assert psf["next_actions"]["remediation_lanes"][1]["remediation_family"] == (
        "coverage_gap"
    )
    assert psf["next_actions"]["remediation_lanes"][1]["best_cut"]["object_id"] == (
        "PSF-007-TP-001"
    )
    assert psf["next_actions"]["ranked_followups"] == [
        {
            "followup_family": "structural_cut",
            "action_kind": "touchpoint_cut",
            "priority_rank": 0,
            "object_id": "PSF-007-TP-005",
            "owner_object_id": "PSF-007-SQ-005",
            "target_doc_id": None,
            "title": "projection_exec_plan.py planning surfaces",
            "blocker_class": "ready_structural",
            "readiness_class": "ready_structural",
            "alignment_status": None,
            "recommended_action": None,
            "touchsite_count": 1,
            "collapsible_touchsite_count": 0,
            "surviving_touchsite_count": 1,
        },
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
            "touchsite_count": 73,
            "collapsible_touchsite_count": 47,
            "surviving_touchsite_count": 26,
        },
    ]
    assert psf["next_actions"]["ranked_touchpoint_cuts"][0]["object_id"] == "PSF-007-TP-005"
    assert psf["next_actions"]["ranked_touchpoint_cuts"][0]["readiness_class"] == (
        "ready_structural"
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
    assert psf_ledger["current_snapshot"]["recommended_cut_object_id"] == "PSF-007-TP-005"
    assert psf_ledger["alignment_summary"]["target_doc_count"] == 2
    assert psf_ledger["alignment_summary"]["recommended_doc_alignment_action"] in {
        "append_existing_ledger_entry",
        "append_new_ledger_entry",
        "none",
    }
    assert len(psf_ledger["target_doc_alignments"]) == 2


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
    monkeypatch,
    capsys,
) -> None:
    _disable_phase5_enricher(monkeypatch)
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
            ]
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
    monkeypatch,
    capsys,
) -> None:
    _disable_phase5_enricher(monkeypatch)
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
            ]
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


def test_build_invariant_graph_joins_policy_signals_and_test_coverage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _disable_phase5_enricher(monkeypatch)
    root = _sample_repo(tmp_path)
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
                            "line": 14,
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

    graph = invariant_graph.build_invariant_graph(root)
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


def test_build_invariant_graph_fails_closed_on_declared_workstream_dependency(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _disable_phase5_enricher(monkeypatch)
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
        invariant_graph.build_invariant_graph(root)


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
    assert "dominant_followup_class: governance" in summary_output
    assert "next_human_followup_family: governance_orphan_resolution" in summary_output
    assert "diagnostic_summary: unmatched_policy_signals=7 :: unresolved_dependencies=0" in summary_output
    assert (
        "recommended_repo_followup: governance_orphan_resolution :: diagnostic=unmatched_policy_signal :: owner=<none> :: seed=src/gabion/analysis/dataflow/io :: seed_object=WS-SEED:gabion.analysis.dataflow.io :: owner_kind=seed_new_owner :: owner_score=100 :: owner_options=seed_new_owner:WS-SEED:gabion.analysis.dataflow.io:100:seed_new_owner_base:100:source_family_seed:rank=1:opp=0:frontier:none :: runner_up_owner=<none> :: runner_up_kind=none :: runner_up_score=none :: owner_choice_margin=100:uncontested_best_option :: owner_choice_margin_components=seed_new_owner_base:100:source_family_seed :: owner_option_tradeoff=100:uncontested_best_option :: owner_option_tradeoff_components=seed_new_owner_base:100:source_family_seed :: count=1 :: action=seed_owned_workstream_from_source_family :: utility=1100:governance_orphan:seed_new_owner+owner_option_tradeoff:100 :: utility_components=governance_orphan_base:900:governance_orphan | owner_resolution_bonus:100:seed_new_owner | owner_option_tradeoff_bonus:100:uncontested_best_option :: certainty=frontier_plateau:7 :: scope=shared_owner_resolution_surface:seed_new_owner:WS-SEED:gabion.analysis.dataflow.io :: runner_up_followup=governance_orphan_resolution:governance:<none>:1100 :: frontier_choice_margin=0:cofrontier :: frontier_choice_margin_components=none"
        in summary_output
    )
    assert "recommended_repo_followup_cohort: 7 ::" in summary_output
    assert (
        "governance_orphan_resolution:diagnostic_resolution:unmatched_policy_signal:seed ownership for grade:GMP-001 from src/gabion/analysis/dataflow/io@1100:rank=1:frontier_tiebreak_winner"
        in summary_output
    )
    assert (
        "governance_orphan_resolution:diagnostic_resolution:unmatched_policy_signal:seed ownership for grade:GMP-007 from src/gabion/analysis/dataflow/io@1100:rank=7:policy_ids:GMP-007"
        in summary_output
    )
    assert (
        "recommended_repo_code_followup: structural_cut :: owner=PSF-007 :: touchpoint_cut :: PSF-007-TP-005 :: count=1 :: blocker=ready_structural :: utility=700:code:ready_structural :: utility_components=code_touchpoint_base:450:code:touchpoint_cut | readiness_bonus:250:readiness:ready_structural :: certainty=ranked_unique:1 :: scope=singleton:<none>"
        in summary_output
    )
    assert (
        "recommended_repo_human_followup: governance_orphan_resolution :: diagnostic=unmatched_policy_signal :: owner=<none> :: seed=src/gabion/analysis/dataflow/io :: seed_object=WS-SEED:gabion.analysis.dataflow.io :: owner_kind=seed_new_owner :: owner_score=100 :: owner_options=seed_new_owner:WS-SEED:gabion.analysis.dataflow.io:100:seed_new_owner_base:100:source_family_seed:rank=1:opp=0:frontier:none :: runner_up_owner=<none> :: runner_up_kind=none :: runner_up_score=none :: owner_choice_margin=100:uncontested_best_option :: owner_choice_margin_components=seed_new_owner_base:100:source_family_seed :: owner_option_tradeoff=100:uncontested_best_option :: owner_option_tradeoff_components=seed_new_owner_base:100:source_family_seed :: count=1 :: action=seed_owned_workstream_from_source_family :: utility=1100:governance_orphan:seed_new_owner+owner_option_tradeoff:100 :: utility_components=governance_orphan_base:900:governance_orphan | owner_resolution_bonus:100:seed_new_owner | owner_option_tradeoff_bonus:100:uncontested_best_option :: certainty=frontier_plateau:7 :: scope=shared_owner_resolution_surface:seed_new_owner:WS-SEED:gabion.analysis.dataflow.io :: runner_up_followup=governance_orphan_resolution:governance:<none>:1100 :: frontier_choice_margin=0:cofrontier :: frontier_choice_margin_components=none"
        in summary_output
    )
    assert (
        "recommended_repo_followup_lane: governance_orphan_resolution :: class=governance :: rank=1 :: utility=1160:governance_orphan:seed_new_owner+owner_option_tradeoff:100+lane_breadth:7+lane:governance_orphan_resolution :: utility_components=best_followup_utility:1100:governance_orphan:seed_new_owner+owner_option_tradeoff:100 | lane_breadth_bonus:35:lane_breadth:7 | lane_class_bonus:25:lane_class:governance :: opportunity=0:frontier :: opportunity_components=none"
        in summary_output
    )
    assert (
        "recommended_repo_code_followup_lane: structural_cut :: class=code :: rank=2 :: utility=715:code:ready_structural+lane_breadth:1+lane:structural_cut :: utility_components=best_followup_utility:700:code:ready_structural | lane_breadth_bonus:5:lane_breadth:1 | lane_class_bonus:10:lane_class:code :: opportunity=445:deferred_by:governance_orphan_resolution :: opportunity_components=best_followup_utility_gap:400:governance_orphan:seed_new_owner+owner_option_tradeoff:100->code:ready_structural | lane_breadth_bonus_gap:30:lane_breadth:7->lane_breadth:1 | lane_class_bonus_gap:15:lane_class:governance->lane_class:code"
        in summary_output
    )
    assert (
        "recommended_repo_human_followup_lane: governance_orphan_resolution :: class=governance :: rank=1 :: utility=1160:governance_orphan:seed_new_owner+owner_option_tradeoff:100+lane_breadth:7+lane:governance_orphan_resolution :: utility_components=best_followup_utility:1100:governance_orphan:seed_new_owner+owner_option_tradeoff:100 | lane_breadth_bonus:35:lane_breadth:7 | lane_class_bonus:25:lane_class:governance :: opportunity=0:frontier :: opportunity_components=none"
        in summary_output
    )
    assert "repo_followup_lanes:" in summary_output
    assert (
        "- governance_orphan_resolution :: class=governance :: actions=7 :: best=diagnostic_resolution::unmatched_policy_signal :: owner_strength=seed_new_owner:100 :: utility=1100:governance_orphan:seed_new_owner+owner_option_tradeoff:100 :: lane_utility=1160:governance_orphan:seed_new_owner+owner_option_tradeoff:100+lane_breadth:7+lane:governance_orphan_resolution :: lane_components=best_followup_utility:1100:governance_orphan:seed_new_owner+owner_option_tradeoff:100 | lane_breadth_bonus:35:lane_breadth:7 | lane_class_bonus:25:lane_class:governance :: rank=1 :: opportunity=0:frontier :: opportunity_components=none"
        in summary_output
    )
    assert "repo_diagnostic_lanes:" in summary_output
    assert (
        "- grade:GMP-001 :: code=unmatched_policy_signal :: severity=warning :: count=1 :: source=src/gabion/analysis/dataflow/io/dataflow_reporting.py::gabion.analysis.dataflow.io.dataflow_reporting._append_report_tail_sections :: policy_ids=GMP-001 :: owner_status=source_family_seed_owner :: owner=<none> :: seed=src/gabion/analysis/dataflow/io :: seed_object=WS-SEED:gabion.analysis.dataflow.io :: best_option=seed_new_owner:WS-SEED:gabion.analysis.dataflow.io:100 :: best_option_components=seed_new_owner_base:100:source_family_seed :: runner_up_option=<none> :: runner_up_components=none :: choice_margin=100:uncontested_best_option :: action=seed_owned_workstream_from_source_family"
        in summary_output
    )

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
    assert "touchsites: 73" in workstream_output
    assert (
        "health_summary: covered=3 :: uncovered=70 :: governed=0 :: diagnosed=0"
        in workstream_output
    )
    assert (
        "touchsite_blockers: ready=3 :: coverage_gap=70 :: policy=0 :: diagnostic=0"
        in workstream_output
    )
    assert (
        "health_cuts: touchpoints(ready=1, coverage_gap=5, policy=0, diagnostic=0) :: subqueues(ready=0, coverage_gap=5, policy=0, diagnostic=0)"
        in workstream_output
    )
    assert "dominant_blocker_class: coverage_gap" in workstream_output
    assert "recommended_remediation_family: structural_cut" in workstream_output
    assert "recommended_cut: touchpoint_cut :: PSF-007-TP-005 :: touchsites=1 :: surviving=1" in workstream_output
    assert (
        "recommended_ready_cut: touchpoint_cut :: PSF-007-TP-005 :: touchsites=1 :: uncovered=0"
        in workstream_output
    )
    assert (
        "recommended_coverage_gap_cut: touchpoint_cut :: PSF-007-TP-001 :: touchsites=4 :: uncovered=4"
        in workstream_output
    )
    assert "recommended_policy_blocked_cut: <none>" in workstream_output
    assert "recommended_diagnostic_blocked_cut: <none>" in workstream_output
    assert "remediation_lanes:" in workstream_output
    assert (
        "- structural_cut :: blocker=ready_structural :: touchsites=3 :: touchpoints=1 :: subqueues=0 :: best=touchpoint_cut::PSF-007-TP-005"
        in workstream_output
    )
    assert (
        "- coverage_gap :: blocker=coverage_gap :: touchsites=70 :: touchpoints=5 :: subqueues=5 :: best=touchpoint_cut::PSF-007-TP-001"
        in workstream_output
    )
    assert "ranked_touchpoint_cuts:" in workstream_output
    assert (
        "- PSF-007-TP-005 :: readiness=ready_structural :: touchsites=1 :: collapsible=0 :: surviving=1 :: uncovered=0"
        in workstream_output
    )
    assert "ranked_subqueue_cuts:" in workstream_output
    assert (
        "- PSF-007-SQ-001 :: readiness=coverage_gap :: touchsites=4 :: collapsible=0 :: surviving=4 :: uncovered=4"
        in workstream_output
    )
    assert "ledger_alignment_summary: target_docs=2 ::" in workstream_output
    assert "dominant_doc_alignment_status: append_pending_existing_object" in workstream_output
    assert "recommended_doc_alignment_action:" in workstream_output
    assert "next_human_followup_family: documentation_alignment" in workstream_output
    assert (
        "recommended_doc_followup_target_doc_id: projection_semantic_fragment_ledger"
        in workstream_output
    )
    assert (
        "recommended_followup: structural_cut :: touchpoint_cut :: PSF-007-TP-005 :: touchsites=1 :: blocker=ready_structural"
        in workstream_output
    )
    assert "misaligned_target_doc_ids:" in workstream_output
    assert "documentation_followup_lanes:" in workstream_output
    assert (
        "- documentation_alignment :: alignment=append_pending_existing_object :: target_docs=2 :: misaligned=2 :: best=projection_semantic_fragment_ledger :: action=append_existing_ledger_entry"
        in workstream_output
    )
    assert "ranked_followups:" in workstream_output
    assert (
        "- structural_cut :: touchpoint_cut :: PSF-007-TP-005 :: readiness=ready_structural :: touchsites=1 :: surviving=1"
        in workstream_output
    )
    assert (
        "- coverage_gap :: touchpoint_cut :: PSF-007-TP-001 :: readiness=coverage_gap :: touchsites=4 :: surviving=4"
        in workstream_output
    )
    assert (
        "- documentation_alignment :: target_doc=projection_semantic_fragment_ledger :: alignment=append_pending_existing_object :: action=append_existing_ledger_entry"
        in workstream_output
    )

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
    assert "current_snapshot: touchsites=73 :: surviving=26" in ledger_output
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
