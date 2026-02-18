from __future__ import annotations

from scripts import docflow_delta_emit, refresh_baselines, sppf_sync
from gabion.execution_plan import (
    BaselineFacet,
    DocflowFacet,
    ExecutionPlan,
)


def test_execution_plan_decorations_are_deterministic() -> None:
    plan = ExecutionPlan()
    plan.decorate("zeta", {"value": 1})
    plan.decorate("alpha", {"value": 2})
    plan.decorate("beta", {"value": 3})

    assert [name for name, _ in plan.decorations()] == ["alpha", "beta", "zeta"]


def test_docflow_facet_propagates_changed_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        docflow_delta_emit,
        "_changed_paths_from_git",
        lambda: ("docs/sppf_checklist.md", "src/gabion/cli.py"),
    )

    plan = docflow_delta_emit._build_execution_plan()

    assert isinstance(plan.docflow, DocflowFacet)
    assert plan.docflow.changed_paths == ("docs/sppf_checklist.md", "src/gabion/cli.py")


def test_issue_link_facet_tracks_checklist_impact() -> None:
    commits = [
        sppf_sync.CommitInfo(sha="a", subject="GH-12", body="Refs #99"),
        sppf_sync.CommitInfo(sha="b", subject="fix", body="GH-12 and GH-33"),
    ]

    facet = sppf_sync._build_issue_link_facet(commits)

    assert facet.issue_ids == ("12", "33", "99")
    assert dict(facet.checklist_impact) == {"12": 2, "33": 1, "99": 1}


def test_baseline_refresh_reads_delta_risk_facet() -> None:
    payload = {
        "summary": {
            "opaque_evidence": {"delta": 1},
            "counts": {"delta": {"unmapped": 4}},
        }
    }
    plan = ExecutionPlan().with_baseline(
        BaselineFacet(risks=refresh_baselines._risk_entries(obsolescence_payload=payload))
    )

    assert plan.baseline.risk("obsolescence.opaque") == 1
    assert plan.baseline.risk("obsolescence.unmapped") == 4
