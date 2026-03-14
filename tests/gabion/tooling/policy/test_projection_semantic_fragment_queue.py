from __future__ import annotations

import json
from pathlib import Path

from gabion.exceptions import NeverThrown
from gabion.analysis.projection.projection_registry import (
    iter_projection_fiber_semantic_specs,
)
import pytest
from gabion.tooling.policy_substrate import invariant_graph
from scripts.policy import projection_semantic_fragment_queue
from tests.path_helpers import REPO_ROOT


def _policy_check_payload() -> dict[str, object]:
    return {
        "rule_id": "policy_check",
        "status": "pass",
        "violations": [],
        "projection_fiber_semantics": {
            "decision": {
                "rule_id": "projection_fiber.convergence.ok",
                "outcome": "pass",
                "severity": "info",
            },
            "report": {
                "semantic_rows": [
                    {
                        "structural_identity": "row-1",
                        "obligation_state": "discharged",
                        "payload": {
                            "path": "src/gabion/example.py",
                            "qualname": "example.frontier",
                            "structural_path": "example.frontier::branch[0]",
                            "complete": True,
                        },
                    }
                ],
                "compiled_projection_semantic_bundles": [
                    {
                        "spec_name": "projection_fiber_frontier",
                        "bindings": [
                            {
                                "quotient_face": "projection_fiber.frontier",
                                "source_structural_identity": "row-1",
                            }
                        ],
                    }
                ],
            },
        },
    }


# gabion:evidence E:function_site::test_projection_semantic_fragment_queue.py::tests.gabion.tooling.policy.test_projection_semantic_fragment_queue.test_analyze_emits_landed_and_active_queue_rows
# gabion:behavior primary=desired
def test_analyze_emits_landed_and_active_queue_rows() -> None:
    expected_phase5 = next(
        item.as_payload()
        for item in invariant_graph.build_invariant_workstreams(
            invariant_graph.build_invariant_graph(REPO_ROOT),
            root=REPO_ROOT,
        ).iter_workstreams()
        if item.object_id.wire() == "PSF-007"
    )
    queue = projection_semantic_fragment_queue.analyze(
        payload=_policy_check_payload(),
        source_artifact="artifacts/out/policy_check_result.json",
    ).as_payload()

    current_state = queue["current_state"]
    assert current_state["decision"]["rule_id"] == "projection_fiber.convergence.ok"
    assert current_state["semantic_row_count"] == 1
    assert current_state["compiled_projection_semantic_bundle_count"] == 1
    assert current_state["semantic_preview_count"] == 1
    assert current_state["compiled_projection_semantic_spec_names"] == [
        "projection_fiber_frontier"
    ]
    items = {item["queue_id"]: item for item in queue["items"]}
    assert items["PSF-001"]["status"] == "landed"
    assert items["PSF-002"]["status"] == "landed"
    assert items["PSF-003"]["status"] == "landed"
    assert items["PSF-004"]["status"] == "in_progress"
    assert items["PSF-005"]["status"] == "queued"
    assert items["PSF-006"]["status"] == "landed"
    assert items["PSF-007"]["status"] == "in_progress"
    assert queue["next_queue_ids"] == ["PSF-004", "PSF-005", "PSF-007"]
    phase5_structure = queue["phase5_structure"]
    assert phase5_structure["queue_id"] == "PSF-007"
    assert phase5_structure["remaining_touchsite_count"] == expected_phase5[
        "touchsite_count"
    ]
    current_frontier = phase5_structure["current_frontier"]
    assert current_frontier is not None
    assert current_frontier["action_kind"] in {"touchpoint_cut", "subqueue_cut"}
    assert current_frontier["object_id"]
    assert current_frontier["owner_object_id"]
    assert current_frontier["decision_mode"]
    assert current_frontier["same_kind_pressure"]
    assert current_frontier["cross_kind_pressure"]
    assert phase5_structure["collapsible_touchsite_count"] > 0
    assert phase5_structure["surviving_touchsite_count"] > 0
    subqueue_ids = [item["subqueue_id"] for item in phase5_structure["subqueues"]]
    assert subqueue_ids == [
        "PSF-007-SQ-001",
        "PSF-007-SQ-002",
        "PSF-007-SQ-003",
        "PSF-007-SQ-004",
        "PSF-007-SQ-005",
    ]
    touchpoint_ids = [item["touchpoint_id"] for item in phase5_structure["touchpoints"]]
    assert touchpoint_ids == [
        "PSF-007-TP-001",
        "PSF-007-TP-002",
        "PSF-007-TP-003",
        "PSF-007-TP-004",
        "PSF-007-TP-005",
        "PSF-007-TP-006",
    ]
    assert sum(item["touchsite_count"] for item in phase5_structure["touchpoints"]) == (
        expected_phase5["touchsite_count"]
    )
    runtime_touchpoint = next(
        item
        for item in phase5_structure["touchpoints"]
        if item["touchpoint_id"] == "PSF-007-TP-006"
    )
    runtime_touchsite = next(
        item
        for item in runtime_touchpoint["touchsites"]
        if item["boundary_name"] == "projection_exec.apply_execution_ops"
    )
    planning_chain = items["PSF-007"]["planning_chain"]
    assert planning_chain is not None
    assert current_frontier["object_id"] in planning_chain["observed_state"]
    assert current_frontier["object_id"] in planning_chain["next_slice"]
    assert current_frontier["decision_mode"] in planning_chain["stabilization_goal"]
    assert str(runtime_touchsite["touchsite_id"]).startswith("PSF-007-TS:")
    assert runtime_touchsite["site_identity"]
    assert runtime_touchsite["structural_identity"]
    assert runtime_touchsite["touchpoint_structural_identity"] == runtime_touchpoint["structural_identity"]


# gabion:evidence E:function_site::test_projection_semantic_fragment_queue.py::tests.gabion.tooling.policy.test_projection_semantic_fragment_queue.test_markdown_summary_lists_queue_and_semantic_preview_context
# gabion:behavior primary=desired
def test_markdown_summary_lists_queue_and_semantic_preview_context() -> None:
    queue = projection_semantic_fragment_queue.analyze(
        payload=_policy_check_payload(),
        source_artifact="artifacts/out/policy_check_result.json",
    )
    queue_payload = queue.as_payload()
    current_frontier = queue_payload["phase5_structure"]["current_frontier"]
    assert current_frontier is not None

    markdown = projection_semantic_fragment_queue._markdown_summary(queue)
    assert "# Projection Semantic Fragment Queue" in markdown
    assert "## current_state" in markdown
    assert "- rule_id: `projection_fiber.convergence.ok`" in markdown
    assert "### compiled_projection_semantic_spec_names" in markdown
    assert "- `projection_fiber_frontier`" in markdown
    assert "## items" in markdown
    assert "- status: `in_progress`" in markdown
    assert "Friendly-surface convergence via typed ProjectionSpec lowering" in markdown
    assert "Move policy and authoring consumers toward direct canonical-carrier judgment" in markdown
    assert "Cut over legacy adapters and retire semantic_carrier_adapter boundaries" in markdown
    assert "## phase5_structure" in markdown
    assert "current_frontier" in markdown
    assert current_frontier["object_id"] in markdown
    assert "planning_chain" in markdown
    assert "- subqueue_id: `PSF-007-SQ-005`" in markdown
    assert "- touchpoint_id: `PSF-007-TP-006`" in markdown
    assert "- rel_path: `src/gabion/analysis/projection/projection_exec.py`" in markdown
    assert "- boundary_name: `projection_exec.apply_execution_ops`" in markdown
    assert "- seam_class: `surviving_carrier_seam`" in markdown
    assert "### semantic_previews" in markdown
    assert "src/gabion/example.py" in markdown


# gabion:evidence E:function_site::test_projection_semantic_fragment_queue.py::tests.gabion.tooling.policy.test_projection_semantic_fragment_queue.test_analyze_accepts_policy_check_result_payloads_directly
# gabion:behavior primary=desired
def test_analyze_accepts_policy_check_result_payloads_directly() -> None:
    queue = projection_semantic_fragment_queue.analyze(
        payload=_policy_check_payload(),
        source_artifact="artifacts/out/policy_check_result.json",
    ).as_payload()

    current_state = queue["current_state"]
    assert current_state["decision"]["rule_id"] == "projection_fiber.convergence.ok"
    assert current_state["semantic_row_count"] == 1
    assert current_state["compiled_projection_semantic_bundle_count"] == 1
    assert current_state["semantic_previews"][0]["path"] == "src/gabion/example.py"


# gabion:evidence E:function_site::test_projection_semantic_fragment_queue.py::tests.gabion.tooling.policy.test_projection_semantic_fragment_queue.test_run_writes_json_and_markdown_outputs
# gabion:behavior primary=desired
def test_run_writes_json_and_markdown_outputs(tmp_path: Path) -> None:
    policy_check = tmp_path / "artifacts/out/policy_check_result.json"
    out = tmp_path / "artifacts/out/projection_semantic_fragment_queue.json"
    markdown_out = tmp_path / "artifacts/out/projection_semantic_fragment_queue.md"
    policy_check.parent.mkdir(parents=True, exist_ok=True)
    policy_check.write_text(json.dumps(_policy_check_payload(), indent=2) + "\n", encoding="utf-8")

    rc = projection_semantic_fragment_queue.run(
        source_artifact_path=policy_check,
        out_path=out,
        markdown_out=markdown_out,
    )

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["current_state"]["semantic_row_count"] == 1
    assert payload["items"][3]["queue_id"] == "PSF-004"
    assert markdown_out.exists()


def test_phase5_structure_prefers_colocated_invariant_workstreams_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy_check = tmp_path / "artifacts/out/policy_check_result.json"
    invariant_workstreams_artifact = tmp_path / "artifacts/out/invariant_workstreams.json"
    policy_check.parent.mkdir(parents=True, exist_ok=True)
    policy_check.write_text(json.dumps(_policy_check_payload(), indent=2) + "\n", encoding="utf-8")
    invariant_workstreams_artifact.write_text(
        json.dumps(
            {
                "format_version": 1,
                "generated_at_utc": "2026-03-13T00:00:00+00:00",
                "root": str(REPO_ROOT),
                "workstreams": [
                    {
                        "object_id": "PSF-007",
                        "title": "Synthetic Phase 5",
                        "touchsite_count": 4,
                        "collapsible_touchsite_count": 1,
                        "surviving_touchsite_count": 3,
                        "subqueues": [],
                        "touchpoints": [],
                        "next_actions": {
                            "recommended_followup": {
                                "followup_family": "structural_cut",
                                "action_kind": "touchpoint_cut",
                                "object_id": "PSF-007-TP-999",
                                "owner_object_id": "PSF-007-SQ-999",
                                "title": "Synthetic frontier",
                                "blocker_class": "ready_structural",
                                "touchsite_count": 4,
                                "surviving_touchsite_count": 3,
                            },
                            "recommended_cut_decision_protocol": {
                                "decision_mode": "frontier_hold",
                                "decision_reason": "synthetic:test_frontier",
                                "same_kind_pressure": "low",
                                "cross_kind_pressure": "low",
                            },
                        },
                    }
                ],
                "counts": {"workstream_count": 1},
                "repo_next_actions": {},
                "diagnostic_summary": {},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        projection_semantic_fragment_queue,
        "build_invariant_graph",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected rebuild")),
    )
    monkeypatch.setattr(
        projection_semantic_fragment_queue,
        "build_invariant_workstreams",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected rebuild")),
    )

    projection_semantic_fragment_queue._phase5_structure.cache_clear()
    phase5 = projection_semantic_fragment_queue._phase5_structure(str(policy_check))
    projection_semantic_fragment_queue._phase5_structure.cache_clear()

    assert phase5.title == "Synthetic Phase 5"
    assert phase5.remaining_touchsite_count == 4
    assert phase5.current_frontier is not None
    assert phase5.current_frontier.object_id == "PSF-007-TP-999"


# gabion:evidence E:function_site::test_projection_semantic_fragment_queue.py::tests.gabion.tooling.policy.test_projection_semantic_fragment_queue.test_analyze_accepts_phase5_workstreams_projection_directly_without_loading
# gabion:behavior primary=desired
def test_analyze_accepts_phase5_workstreams_projection_directly_without_loading(
    monkeypatch,
) -> None:
    phase5_workstreams_projection = {
        "format_version": 1,
        "generated_at_utc": "2026-03-13T00:00:00+00:00",
        "root": str(REPO_ROOT),
        "workstreams": [
            {
                "object_id": "PSF-007",
                "title": "Synthetic Phase 5",
                "touchsite_count": 4,
                "collapsible_touchsite_count": 1,
                "surviving_touchsite_count": 3,
                "subqueues": [],
                "touchpoints": [],
                "next_actions": {
                    "recommended_followup": {
                        "followup_family": "structural_cut",
                        "action_kind": "touchpoint_cut",
                        "object_id": "PSF-007-TP-999",
                        "owner_object_id": "PSF-007-SQ-999",
                        "title": "Synthetic frontier",
                        "blocker_class": "ready_structural",
                        "touchsite_count": 4,
                        "surviving_touchsite_count": 3,
                    },
                    "recommended_cut_decision_protocol": {
                        "decision_mode": "frontier_hold",
                        "decision_reason": "synthetic:test_frontier",
                        "same_kind_pressure": "low",
                        "cross_kind_pressure": "low",
                    },
                },
            }
        ],
        "counts": {"workstream_count": 1},
        "repo_next_actions": {},
        "diagnostic_summary": {},
    }
    monkeypatch.setattr(
        projection_semantic_fragment_queue,
        "_load_phase5_workstreams_projection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected load")),
    )
    monkeypatch.setattr(
        projection_semantic_fragment_queue,
        "build_invariant_graph",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected rebuild")),
    )
    monkeypatch.setattr(
        projection_semantic_fragment_queue,
        "build_invariant_workstreams",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected rebuild")),
    )

    queue = projection_semantic_fragment_queue.analyze(
        payload=_policy_check_payload(),
        source_artifact="artifacts/out/policy_check_result.json",
        phase5_workstreams_projection=phase5_workstreams_projection,
    ).as_payload()

    phase5_structure = queue["phase5_structure"]
    assert phase5_structure["title"] == "Synthetic Phase 5"
    assert phase5_structure["remaining_touchsite_count"] == 4
    assert phase5_structure["current_frontier"] is not None
    assert phase5_structure["current_frontier"]["object_id"] == "PSF-007-TP-999"


def test_analyze_marks_semantic_op_expansion_landed_when_witness_synthesis_is_present() -> None:
    payload = _policy_check_payload()
    report = payload["projection_fiber_semantics"]["report"]
    compiled_bundles = list(report["compiled_projection_semantic_bundles"])
    compiled_bundles.append(
        {
            "spec_name": "projection_fiber_witness_synthesis",
            "bindings": [],
            "compiled_shacl_plans": [],
            "compiled_sparql_plans": [],
        }
    )
    report["compiled_projection_semantic_bundles"] = compiled_bundles

    queue = projection_semantic_fragment_queue.analyze(
        payload=payload,
        source_artifact="artifacts/out/policy_check_result.json",
    ).as_payload()

    items = {item["queue_id"]: item for item in queue["items"]}
    assert items["PSF-005"]["status"] == "landed"
    assert queue["next_queue_ids"] == ["PSF-004", "PSF-007"]


def test_analyze_marks_friendly_surface_convergence_landed_when_all_declared_specs_compile() -> None:
    payload = _policy_check_payload()
    report = payload["projection_fiber_semantics"]["report"]
    report["compiled_projection_semantic_bundles"] = [
        {
            "spec_name": str(spec.name),
            "bindings": [
                {
                    "quotient_face": "projection_fiber.frontier",
                    "source_structural_identity": "row-1",
                }
            ]
            if str(spec.name) == "projection_fiber_frontier"
            else [],
            "compiled_shacl_plans": [],
            "compiled_sparql_plans": [],
        }
        for spec in iter_projection_fiber_semantic_specs()
    ]

    queue = projection_semantic_fragment_queue.analyze(
        payload=payload,
        source_artifact="artifacts/out/policy_check_result.json",
    ).as_payload()

    items = {item["queue_id"]: item for item in queue["items"]}
    assert items["PSF-004"]["status"] == "landed"
    assert items["PSF-005"]["status"] == "landed"
    assert queue["next_queue_ids"] == ["PSF-007"]


def test_analyze_marks_phase5_landed_when_adapter_markers_are_retired(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        projection_semantic_fragment_queue,
        "_legacy_projection_exec_ingress_retired",
        lambda: True,
    )
    live_structure = projection_semantic_fragment_queue._phase5_structure(
        "artifacts/out/policy_check_result.json"
    )
    monkeypatch.setattr(
        projection_semantic_fragment_queue,
        "_phase5_structure",
        lambda source_artifact: projection_semantic_fragment_queue.ProjectionSemanticFragmentPhase5Structure(
            queue_id=live_structure.queue_id,
            title=live_structure.title,
            remaining_touchsite_count=0,
            collapsible_touchsite_count=0,
            surviving_touchsite_count=0,
            subqueues=live_structure.subqueues,
            touchpoints=live_structure.touchpoints,
            current_frontier=None,
        ),
    )
    queue = projection_semantic_fragment_queue.analyze(
        payload=_policy_check_payload(),
        source_artifact="artifacts/out/policy_check_result.json",
    ).as_payload()

    items = {item["queue_id"]: item for item in queue["items"]}
    assert items["PSF-007"]["status"] == "landed"
    assert items["PSF-007"]["planning_chain"] is None


def test_analyze_requires_phase5_frontier_when_phase5_is_in_progress(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        projection_semantic_fragment_queue,
        "_legacy_projection_exec_ingress_retired",
        lambda: True,
    )
    live_structure = projection_semantic_fragment_queue._phase5_structure(
        "artifacts/out/policy_check_result.json"
    )
    monkeypatch.setattr(
        projection_semantic_fragment_queue,
        "_phase5_structure",
        lambda source_artifact: projection_semantic_fragment_queue.ProjectionSemanticFragmentPhase5Structure(
            queue_id=live_structure.queue_id,
            title=live_structure.title,
            remaining_touchsite_count=max(1, live_structure.remaining_touchsite_count),
            collapsible_touchsite_count=live_structure.collapsible_touchsite_count,
            surviving_touchsite_count=live_structure.surviving_touchsite_count,
            subqueues=live_structure.subqueues,
            touchpoints=live_structure.touchpoints,
            current_frontier=None,
        ),
    )

    with pytest.raises(NeverThrown):
        projection_semantic_fragment_queue.analyze(
            payload=_policy_check_payload(),
            source_artifact="artifacts/out/policy_check_result.json",
        )
