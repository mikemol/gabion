from __future__ import annotations

from pathlib import Path

from gabion.analysis import aspf_action_plan


# gabion:evidence E:function_site::aspf_action_plan.py::gabion.analysis.aspf_action_plan.build_action_plan_payload
def test_build_action_plan_payload_ranks_multi_source_paths(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "src" / "gabion" / "tooling").mkdir(parents=True)
    (root / "scripts").mkdir(parents=True)

    opportunities_payload = {
        "opportunities": [
            {
                "kind": "reusable_boundary_artifact",
                "opportunity_id": "opp:1",
                "confidence": 0.67,
                "affected_surfaces": ["groups_by_path"],
                "witness_ids": ["w:1"],
                "reason": "shared representative payload can be reused",
            }
        ]
    }
    semantic_surface_payloads = {
        "groups_by_path": {
            "src/gabion/tooling/aspf_handoff.py": {"fn": ["bundle"]}
        },
        "delta_payload": {
            "call_stack": {
                "site_table": [
                    {
                        "kind": "FileSite",
                        "key": [
                            str(
                                root
                                / "src"
                                / "gabion"
                                / "tooling"
                                / "run_dataflow_stage.py"
                            ),
                            10,
                            1,
                            10,
                        ],
                    },
                    {
                        "nested": {
                            "kind": "FileSite",
                            "key": "scripts/ci_watch.py",
                        }
                    },
                ]
            }
        },
    }
    delta_ledger_payload = {
        "records": [
            {
                "mutation_value": {
                    "metadata": {
                        "path": str(root / "scripts" / "ci_local_repro.sh"),
                        "checkpoint_path": "<memory>",
                    }
                }
            }
        ]
    }
    trace_payload = {
        "one_cells": [
            {
                "metadata": {
                    "path": "src/gabion/tooling/aspf_handoff.py",
                }
            }
        ]
    }

    payload = aspf_action_plan.build_action_plan_payload(
        trace_id="aspf-trace:test",
        command_profile="check.run",
        opportunities_payload=opportunities_payload,
        semantic_surface_payloads=semantic_surface_payloads,
        root=root,
        delta_ledger_payload=delta_ledger_payload,
        trace_payload=trace_payload,
    )

    actions = payload["actions"]
    assert isinstance(actions, list)
    assert len(actions) == 1
    action = actions[0]
    targets = action["targets"]
    assert isinstance(targets, dict)
    assert targets["paths"]
    assert targets["paths"][0] == "src/gabion/tooling/aspf_handoff.py"
    path_scores = targets["path_scores"]
    assert path_scores["src/gabion/tooling/aspf_handoff.py"] >= 10
    assert "scripts/ci_local_repro.sh" in path_scores

    evidence_refs = action["evidence_refs"]
    assert isinstance(evidence_refs, dict)
    path_sources = evidence_refs["path_sources"]
    assert "semantic_surfaces.groups_by_path" in path_sources[
        "src/gabion/tooling/aspf_handoff.py"
    ]


# gabion:evidence E:function_site::aspf_action_plan.py::gabion.analysis.aspf_action_plan.evaluate_action_plan_quality
def test_evaluate_action_plan_quality_warns_for_missing_action_mappings() -> None:
    quality = aspf_action_plan.evaluate_action_plan_quality(
        action_plan_payload={
            "actions": [
                {
                    "action_id": "action:1",
                    "targets": {"paths": []},
                    "validation_commands": [],
                }
            ]
        },
        opportunities_payload={
            "opportunities": [
                {"kind": "reusable_boundary_artifact", "opportunity_id": "opp:1"},
                {"kind": "materialize_load_fusion", "opportunity_id": "opp:2"},
            ]
        },
    )

    assert quality["status"] == "warning"
    summary = quality["summary"]
    assert summary["opportunity_count"] == 2
    assert summary["action_count"] == 1
    issues = quality["issues"]
    issue_ids = {str(issue["issue_id"]) for issue in issues}
    assert "action_count_below_opportunity_count" in issue_ids
    assert "missing_target_paths:action:1" in issue_ids
    assert "missing_validation_commands:action:1" in issue_ids


# gabion:evidence E:function_site::aspf_action_plan.py::gabion.analysis.aspf_action_plan.evaluate_action_plan_quality
def test_evaluate_action_plan_quality_ok_when_actions_are_actionable() -> None:
    quality = aspf_action_plan.evaluate_action_plan_quality(
        action_plan_payload={
            "actions": [
                {
                    "action_id": "action:1",
                    "targets": {"paths": ["src/gabion/tooling/aspf_handoff.py"]},
                    "validation_commands": ["mise exec -- python -m pytest tests/test_aspf_handoff.py"],
                }
            ]
        },
        opportunities_payload={
            "opportunities": [
                {"kind": "reusable_boundary_artifact", "opportunity_id": "opp:1"}
            ]
        },
    )

    assert quality["status"] == "ok"
    assert quality["summary"]["issue_count"] == 0
    assert quality["issues"] == []
