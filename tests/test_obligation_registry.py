from __future__ import annotations

from gabion.analysis.obligation_registry import evaluate_obligations, summarize_obligations


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
