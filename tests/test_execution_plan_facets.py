from __future__ import annotations

from gabion.tooling import docflow_delta_emit
from scripts import refresh_baselines, sppf_sync
from gabion.execution_plan import (
    BaselineFacet,
    DeadlineFacet,
    DocflowFacet,
    ExecutionPlan,
    IssueLinkFacet,
)


# gabion:evidence E:call_footprint::tests/test_execution_plan_facets.py::test_execution_plan_decorations_are_deterministic::execution_plan.py::gabion.execution_plan.ExecutionPlan
def test_execution_plan_decorations_are_deterministic() -> None:
    plan = ExecutionPlan()
    plan.decorate("zeta", {"value": 1})
    plan.decorate("alpha", {"value": 2})
    plan.decorate("beta", {"value": 3})

    assert [name for name, _ in plan.decorations()] == ["alpha", "beta", "zeta"]


# gabion:evidence E:call_footprint::tests/test_execution_plan_facets.py::test_docflow_facet_propagates_changed_paths::docflow_delta_emit.py::gabion.tooling.docflow_delta_emit._build_execution_plan
def test_docflow_facet_propagates_changed_paths() -> None:
    plan = docflow_delta_emit._build_execution_plan(
        changed_paths_fn=lambda: ("docs/sppf_checklist.md", "src/gabion/cli.py"),
    )

    assert isinstance(plan.docflow, DocflowFacet)
    assert plan.docflow.changed_paths == ("docs/sppf_checklist.md", "src/gabion/cli.py")


# gabion:evidence E:call_footprint::tests/test_execution_plan_facets.py::test_issue_link_facet_tracks_checklist_impact::sppf_sync.py::scripts.sppf_sync._build_issue_link_facet
def test_issue_link_facet_tracks_checklist_impact() -> None:
    commits = [
        sppf_sync.CommitInfo(sha="a", subject="GH-12", body="Refs #99"),
        sppf_sync.CommitInfo(sha="b", subject="fix", body="GH-12 and GH-33"),
    ]

    facet = sppf_sync._build_issue_link_facet(commits)

    assert facet.issue_ids == ("12", "33", "99")
    assert dict(facet.checklist_impact) == {"12": 2, "33": 1, "99": 1}


# gabion:evidence E:call_footprint::tests/test_execution_plan_facets.py::test_baseline_refresh_reads_delta_risk_facet::refresh_baselines.py::scripts.refresh_baselines._risk_entries
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


# gabion:evidence E:call_footprint::tests/test_execution_plan_facets.py::test_execution_plan_mutators_and_default_risk::execution_plan.py::gabion.execution_plan.ExecutionPlan
def test_execution_plan_mutators_and_default_risk() -> None:
    plan = ExecutionPlan()
    issue = IssueLinkFacet(issue_ids=("42",), checklist_impact=(("42", 1),))
    deadline = DeadlineFacet(timeout_seconds=15)

    assert plan.with_issue_link(issue) is plan
    assert plan.with_deadline(deadline) is plan
    assert plan.issue_link is issue
    assert plan.deadline is deadline
    assert plan.baseline.risk("missing") == 0
