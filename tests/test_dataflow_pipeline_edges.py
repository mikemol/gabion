from __future__ import annotations

import pytest

from gabion.analysis import dataflow_pipeline
from gabion.exceptions import NeverThrown


def _bind() -> None:
    dataflow_pipeline._bind_audit_symbols()


# gabion:evidence E:call_footprint::tests/test_dataflow_pipeline_edges.py::test_normalized_dimension_payload_filters_invalid_entries::dataflow_pipeline.py::gabion.analysis.dataflow_pipeline._normalized_dimension_payload
def test_normalized_dimension_payload_filters_invalid_entries() -> None:
    _bind()
    payload = {
        "valid": {"done": 4, "total": 7},
        "invalid_done_type": {"done": "4", "total": 7},
        "invalid_done_bool": {"done": True, "total": 7},
        "invalid_payload": [],
        ("tuple-key",): {"done": 1, "total": 1},
        "clamped": {"done": -5, "total": -2},
    }
    normalized = dataflow_pipeline._normalized_dimension_payload(payload)
    assert normalized == {
        "valid": {"done": 4, "total": 7},
        "clamped": {"done": 0, "total": 0},
    }


# gabion:evidence E:call_footprint::tests/test_dataflow_pipeline_edges.py::test_apply_forest_progress_delta_non_mapping_is_noop::dataflow_pipeline.py::gabion.analysis.dataflow_pipeline._apply_forest_progress_delta
def test_apply_forest_progress_delta_non_mapping_is_noop() -> None:
    _bind()
    base_dimensions = {"x": {"done": 1, "total": 2}}
    assert dataflow_pipeline._apply_forest_progress_delta(
        0,
        forest_mutable_progress_done=1,
        forest_mutable_progress_total=2,
        forest_progress_marker="start",
        forest_dimensions=base_dimensions,
    ) == (1, 2, "start", base_dimensions, False)


# gabion:evidence E:call_footprint::tests/test_dataflow_pipeline_edges.py::test_apply_forest_progress_delta_handles_invalid_scalars_and_dimension_shape::dataflow_pipeline.py::gabion.analysis.dataflow_pipeline._apply_forest_progress_delta
def test_apply_forest_progress_delta_handles_invalid_scalars_and_dimension_shape() -> None:
    _bind()
    base_dimensions = {"x": {"done": 1, "total": 2}}
    done, total, marker, dimensions, changed = dataflow_pipeline._apply_forest_progress_delta(
        {
            "primary_done": True,
            "primary_total": "bad",
            "marker": "",
            "dimensions": [],
        },
        forest_mutable_progress_done=1,
        forest_mutable_progress_total=2,
        forest_progress_marker="start",
        forest_dimensions=base_dimensions,
    )
    assert changed is True
    assert done == 1
    assert total == 2
    assert marker == "start"
    assert dimensions == base_dimensions


# gabion:evidence E:decision_surface/direct::dataflow_pipeline.py::gabion.analysis.dataflow_pipeline.analyze_paths::on_collection_progress
def test_analyze_paths_rejects_invalid_collection_progress_callback() -> None:
    _bind()
    with pytest.raises(NeverThrown):
        dataflow_pipeline.analyze_paths(
            paths=[],
            forest=dataflow_pipeline.Forest(),
            recursive=False,
            type_audit=False,
            type_audit_report=False,
            type_audit_max=1,
            include_constant_smells=False,
            include_unused_arg_smells=False,
            on_collection_progress="not-callable",  # type: ignore[arg-type]
        )


# gabion:evidence E:decision_surface/direct::dataflow_pipeline.py::gabion.analysis.dataflow_pipeline.analyze_paths::on_phase_progress
def test_analyze_paths_rejects_invalid_phase_progress_callback() -> None:
    _bind()
    with pytest.raises(NeverThrown):
        dataflow_pipeline.analyze_paths(
            paths=[],
            forest=dataflow_pipeline.Forest(),
            recursive=False,
            type_audit=False,
            type_audit_report=False,
            type_audit_max=1,
            include_constant_smells=False,
            include_unused_arg_smells=False,
            on_phase_progress="not-callable",  # type: ignore[arg-type]
        )
