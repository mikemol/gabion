from __future__ import annotations

import json
from pathlib import Path

from gabion.tooling.policy_substrate import project_manager_view


def _sample_payload() -> dict[str, object]:
    return {
        "format_version": 1,
        "generated_at_utc": "2026-03-14T05:33:19+00:00",
        "root": "/repo",
        "planning_chart_summary": {
            "item_count": 3,
            "selected_completion_item_ids": ["repo.followup:PSF-007-TP-001"],
            "phases": [
                {
                    "phase_kind": "scan",
                    "item_count": 1,
                    "status_counts": {"in_progress": 1},
                    "blocker_counts": {},
                    "selected_item_ids": [],
                    "items": [],
                },
                {
                    "phase_kind": "predict",
                    "item_count": 1,
                    "status_counts": {"policy_blocked": 1},
                    "blocker_counts": {"policy_blocked": 1},
                    "selected_item_ids": [],
                    "items": [],
                },
                {
                    "phase_kind": "complete",
                    "item_count": 1,
                    "status_counts": {"ready_structural": 1},
                    "blocker_counts": {"ready_structural": 1},
                    "selected_item_ids": ["repo.followup:PSF-007-TP-001"],
                    "items": [],
                },
            ],
        },
        "workstreams": [
            {
                "object_id": "CSA-IVL",
                "title": "Invariant velocity and latency",
                "status": "in_progress",
                "touchsite_count": 10,
                "surviving_touchsite_count": 6,
                "policy_signal_count": 1,
                "diagnostic_count": 2,
                "failing_test_case_count": 0,
                "test_failure_count": 1,
                "coverage_count": 4,
                "doc_alignment_summary": {
                    "missing_target_doc_count": 1,
                    "ambiguous_target_doc_count": 0,
                    "unassigned_target_doc_count": 1,
                    "append_pending_existing_target_doc_count": 0,
                    "append_pending_new_target_doc_count": 0,
                },
                "recommended_followup": {
                    "followup_family": "planner_velocity",
                    "action_kind": "touchpoint_cut",
                    "object_id": "CSA-IVL-TP-002",
                    "owner_root_object_id": "CSA-IVL",
                    "title": "Trim planner artifact fanout",
                    "blocker_class": "ready_structural",
                    "recommended_action": "cut_followup",
                },
            },
            {
                "object_id": "PSF-007",
                "title": "Projection semantic fragment continuation",
                "status": "in_progress",
                "touchsite_count": 20,
                "surviving_touchsite_count": 12,
                "policy_signal_count": 0,
                "diagnostic_count": 0,
                "failing_test_case_count": 0,
                "test_failure_count": 0,
                "coverage_count": 8,
                "doc_alignment_summary": {
                    "missing_target_doc_count": 0,
                    "ambiguous_target_doc_count": 0,
                    "unassigned_target_doc_count": 0,
                    "append_pending_existing_target_doc_count": 0,
                    "append_pending_new_target_doc_count": 0,
                },
                "next_actions": {
                    "recommended_followup": {
                        "followup_family": "structural_cut",
                        "action_kind": "touchpoint_cut",
                        "object_id": "PSF-007-TP-001",
                        "owner_root_object_id": "PSF-007",
                        "title": "Projection structural cut",
                        "blocker_class": "ready_structural",
                        "recommended_action": "cut_followup",
                    }
                },
            },
            {
                "object_id": "PRF",
                "title": "Policy registry foundation",
                "status": "landed",
                "touchsite_count": 0,
                "surviving_touchsite_count": 0,
                "policy_signal_count": 0,
                "diagnostic_count": 0,
                "failing_test_case_count": 0,
                "test_failure_count": 0,
                "coverage_count": 0,
                "doc_alignment_summary": {
                    "missing_target_doc_count": 0,
                    "ambiguous_target_doc_count": 0,
                    "unassigned_target_doc_count": 0,
                    "append_pending_existing_target_doc_count": 0,
                    "append_pending_new_target_doc_count": 0,
                },
            },
        ],
        "repo_next_actions": {
            "dominant_followup_class": "code",
            "next_human_followup_family": "documentation_alignment",
            "recommended_followup": {
                "followup_family": "structural_cut",
                "action_kind": "touchpoint_cut",
                "object_id": "PSF-007-TP-001",
                "owner_root_object_id": "PSF-007",
                "title": "Projection structural cut",
                "blocker_class": "ready_structural",
                "recommended_action": "cut_followup",
            },
            "recommended_code_followup": {
                "followup_family": "structural_cut",
                "action_kind": "touchpoint_cut",
                "object_id": "PSF-007-TP-001",
                "owner_root_object_id": "PSF-007",
                "title": "Projection structural cut",
                "blocker_class": "ready_structural",
                "recommended_action": "cut_followup",
            },
        },
    }


def test_analyze_summarizes_portfolio_and_sorts_by_pressure() -> None:
    view = project_manager_view.analyze(
        payload=_sample_payload(),
        source_artifact="artifacts/out/invariant_workstreams.json",
        visual_limit=2,
    )

    assert view.portfolio_summary.workstream_count == 3
    assert view.portfolio_summary.status_counts == {
        "in_progress": 2,
        "landed": 1,
    }
    assert view.repo_next_action is not None
    assert view.repo_next_action.object_id == "PSF-007-TP-001"
    assert view.workstreams[0].object_id == "CSA-IVL"
    assert view.workstreams[0].pressure_score > view.workstreams[1].pressure_score
    assert view.workstreams[1].recommended_followup is not None
    assert view.workstreams[1].recommended_followup.object_id == "PSF-007-TP-001"


def test_render_markdown_includes_summary_table_and_mermaid() -> None:
    view = project_manager_view.analyze(
        payload=_sample_payload(),
        source_artifact="artifacts/out/invariant_workstreams.json",
        visual_limit=2,
    )

    markdown = project_manager_view.render_markdown(view)

    assert "# Project Manager View" in markdown
    assert "## Portfolio Summary" in markdown
    assert "| workstream_id | status | pressure |" in markdown
    assert "CSA-IVL" in markdown
    assert "PSF-007-TP-001" in markdown
    assert "```mermaid" in markdown
    assert "flowchart TB" in markdown


def test_render_readme_section_includes_actions_and_workstream_table() -> None:
    view = project_manager_view.analyze(
        payload=_sample_payload(),
        source_artifact="artifacts/out/invariant_workstreams.json",
        visual_limit=2,
    )

    section = project_manager_view.render_readme_section(view)

    assert "## Live Project Manager View" in section
    assert "**Repo next action:**" in section
    assert "`PSF-007-TP-001`" in section
    assert "| ID | Workstream | Status | Pressure | Recommended followup |" in section
    assert "Projection semantic fragment continuation" in section
    assert "project_manager_view.json" in section


def test_run_writes_json_markdown_mermaid_and_updates_readme(tmp_path: Path) -> None:
    source_artifact = tmp_path / "artifacts/out/invariant_workstreams.json"
    json_out = tmp_path / "artifacts/out/project_manager_view.json"
    markdown_out = tmp_path / "artifacts/out/project_manager_view.md"
    mermaid_out = tmp_path / "artifacts/out/project_manager_view.mmd"
    readme_path = tmp_path / "README.md"
    source_artifact.parent.mkdir(parents=True, exist_ok=True)
    source_artifact.write_text(
        json.dumps(_sample_payload(), indent=2) + "\n",
        encoding="utf-8",
    )
    readme_path.write_text(
        "\n".join(
            (
                "# Temp README",
                "",
                project_manager_view._README_SECTION_BEGIN,
                "stale body",
                project_manager_view._README_SECTION_END,
                "",
                "after",
                "",
            )
        ),
        encoding="utf-8",
    )

    rc = project_manager_view.run(
        source_artifact_path=source_artifact,
        out_path=json_out,
        markdown_out=markdown_out,
        mermaid_out=mermaid_out,
        readme_path=readme_path,
        visual_limit=2,
    )

    assert rc == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["portfolio_summary"]["workstream_count"] == 3
    assert payload["repo_next_action"]["object_id"] == "PSF-007-TP-001"
    assert payload["workstreams"][0]["object_id"] == "CSA-IVL"
    assert "flowchart TB" in payload["visualization_mermaid"]
    assert markdown_out.exists()
    assert "```mermaid" in markdown_out.read_text(encoding="utf-8")
    assert mermaid_out.exists()
    assert "flowchart TB" in mermaid_out.read_text(encoding="utf-8")
    readme_text = readme_path.read_text(encoding="utf-8")
    assert "## Live Project Manager View" in readme_text
    assert "Projection semantic fragment continuation" in readme_text
    assert "stale body" not in readme_text


def test_run_raises_when_readme_markers_are_missing(tmp_path: Path) -> None:
    source_artifact = tmp_path / "artifacts/out/invariant_workstreams.json"
    json_out = tmp_path / "artifacts/out/project_manager_view.json"
    markdown_out = tmp_path / "artifacts/out/project_manager_view.md"
    mermaid_out = tmp_path / "artifacts/out/project_manager_view.mmd"
    readme_path = tmp_path / "README.md"
    source_artifact.parent.mkdir(parents=True, exist_ok=True)
    source_artifact.write_text(
        json.dumps(_sample_payload(), indent=2) + "\n",
        encoding="utf-8",
    )
    readme_path.write_text("# README without markers\n", encoding="utf-8")

    try:
        project_manager_view.run(
            source_artifact_path=source_artifact,
            out_path=json_out,
            markdown_out=markdown_out,
            mermaid_out=mermaid_out,
            readme_path=readme_path,
            visual_limit=2,
        )
    except ValueError as exc:
        assert "missing project manager view markers" in str(exc)
    else:
        raise AssertionError("expected missing README markers to raise ValueError")
