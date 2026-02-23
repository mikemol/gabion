from __future__ import annotations

from gabion.commands import progress_contract


# gabion:evidence E:call_footprint::tests/test_progress_contract_edges.py::test_phase_progress_dimensions_summary_keeps_zero_total_without_clamp::progress_contract.py::gabion.commands.progress_contract.phase_progress_dimensions_summary
def test_phase_progress_dimensions_summary_keeps_zero_total_without_clamp() -> None:
    summary = progress_contract.phase_progress_dimensions_summary(
        {
            "dimensions": {
                "zero": {"done": 5, "total": 0},
                "skip_bool": {"done": True, "total": 1},
            }
        }
    )
    assert summary == "zero=5/0"


# gabion:evidence E:call_footprint::tests/test_progress_contract_edges.py::test_phase_timeline_row_primary_fallback_and_empty_primary_paths::progress_contract.py::gabion.commands.progress_contract.phase_timeline_row_from_phase_progress
def test_phase_timeline_row_primary_fallback_and_empty_primary_paths() -> None:
    row_with_zero_total = progress_contract.phase_timeline_row_from_phase_progress(
        {
            "phase": "collection",
            "work_done": 7,
            "work_total": 0,
        }
    )
    assert "| 7/0 |" in row_with_zero_total

    row_empty_primary = progress_contract.phase_timeline_row_from_phase_progress(
        {"phase": "collection"}
    )
    fields = [field.strip() for field in row_empty_primary.strip("|").split("|")]
    assert len(fields) == len(progress_contract.phase_timeline_header_columns())
    # The primary column remains empty when no primary values and no primary unit exist.
    assert fields[7] == ""
