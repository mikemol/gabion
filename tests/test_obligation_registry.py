from __future__ import annotations

from gabion.analysis.obligation_registry import evaluate_obligations, summarize_obligations


# gabion:evidence E:call_footprint::tests/test_obligation_registry.py::test_evaluate_obligations_marks_unmet_rules::obligation_registry.py::gabion.analysis.obligation_registry.evaluate_obligations
def test_evaluate_obligations_marks_unmet_rules() -> None:
    context = {
        "sppf_relevant_paths_changed": True,
        "gh_reference_validated": False,
        "baseline_write_emitted": True,
        "delta_guard_checked": False,
        "doc_status_changed": True,
        "checklist_influence_consistent": False,
    }

    entries = evaluate_obligations(operation="docflow_plan", context=context)

    by_id = {str(entry["obligation_id"]): entry for entry in entries}
    assert by_id["sppf_gh_reference_validation"]["status"] == "unmet"
    assert by_id["sppf_gh_reference_validation"]["enforcement"] == "fail"
    assert by_id["baseline_delta_guard"]["status"] == "unmet"
    assert by_id["baseline_delta_guard"]["enforcement"] == "warn"
    assert by_id["doc_status_consistency"]["status"] == "unmet"


# gabion:evidence E:call_footprint::tests/test_obligation_registry.py::test_summarize_obligations_counts_unmet_by_enforcement::obligation_registry.py::gabion.analysis.obligation_registry.evaluate_obligations::obligation_registry.py::gabion.analysis.obligation_registry.summarize_obligations
def test_summarize_obligations_counts_unmet_by_enforcement() -> None:
    entries = evaluate_obligations(
        operation="docflow_plan",
        context={
            "sppf_relevant_paths_changed": True,
            "gh_reference_validated": True,
            "baseline_write_emitted": True,
            "delta_guard_checked": False,
            "doc_status_changed": False,
            "checklist_influence_consistent": True,
        },
    )

    summary = summarize_obligations(entries)

    assert summary == {
        "total": 3,
        "triggered": 2,
        "met": 1,
        "unmet_fail": 0,
        "unmet_warn": 1,
    }


# gabion:evidence E:call_footprint::tests/test_obligation_registry.py::test_obligation_registry_filters_operation_and_counts_unmet_fail::obligation_registry.py::gabion.analysis.obligation_registry.evaluate_obligations::obligation_registry.py::gabion.analysis.obligation_registry.summarize_obligations
def test_obligation_registry_filters_operation_and_counts_unmet_fail() -> None:
    assert evaluate_obligations(operation="other", context={}) == []

    summary = summarize_obligations(
        [
            {
                "obligation_id": "x",
                "operation": "docflow_plan",
                "context": "test",
                "description": "test",
                "enforcement": "fail",
                "triggered": True,
                "status": "unmet",
            }
        ]
    )
    assert summary == {
        "total": 1,
        "triggered": 1,
        "met": 0,
        "unmet_fail": 1,
        "unmet_warn": 0,
    }
