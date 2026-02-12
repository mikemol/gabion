from __future__ import annotations

from pathlib import Path

from gabion.analysis import dataflow_audit
from gabion.exceptions import NeverThrown
import pytest


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._format_span_fields
def test_format_span_fields_handles_invalid_values() -> None:
    assert dataflow_audit._format_span_fields("bad", 0, 0, 0) == ""
    assert dataflow_audit._format_span_fields(-1, 0, 0, 0) == ""
    assert dataflow_audit._format_span_fields(0, 0, 1, 2) == "1:1-2:3"


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_never_invariants::entries,include_proven_unreachable,max_entries E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec.apply_spec::params_override E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_never_invariants._format_evidence::status
def test_summarize_never_invariants_filters_and_formats() -> None:
    entries = [
        "not-a-mapping",
        {
            "status": "VIOLATION",
            "site": {"path": "a.py", "function": "f"},
            "never_id": "n1",
            "span": [0, 0, 0, 1],
            "witness_ref": "w",
            "environment_ref": {"env": 1},
        },
        {
            "status": "OBLIGATION",
            "site": {"path": "a.py", "function": "g"},
            "never_id": "n2",
            "span": [0, 0, 0, 1],
        },
        {
            "status": "PROVEN_UNREACHABLE",
            "site": {"path": "a.py", "function": "h"},
            "never_id": "n3",
            "span": [0, 0, 0, 1],
            "witness_ref": "dead",
        },
    ]
    lines = dataflow_audit._summarize_never_invariants(
        entries,
        max_entries=10,
        include_proven_unreachable=False,
    )
    assert any("VIOLATION" in line for line in lines)
    assert "PROVEN_UNREACHABLE:" not in lines


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._copy_forest_signature_metadata
def test_copy_forest_signature_metadata_marks_missing_signature() -> None:
    payload: dict[str, object] = {}
    snapshot: dict[str, object] = {}
    dataflow_audit._copy_forest_signature_metadata(payload, snapshot)
    assert payload["forest_signature_partial"] is True
    assert payload["forest_signature_basis"] == "missing"


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._copy_forest_signature_metadata
def test_copy_forest_signature_metadata_copies_fields() -> None:
    payload: dict[str, object] = {}
    snapshot = {
        "forest_signature_partial": False,
        "forest_signature_basis": "bundles_only",
    }
    dataflow_audit._copy_forest_signature_metadata(payload, snapshot, prefix="x_")
    assert payload["x_forest_signature_partial"] is True
    assert payload["x_forest_signature_basis"] == "bundles_only"


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot::forest,project_root
def test_render_decision_snapshot_requires_forest(tmp_path: Path) -> None:
    with pytest.raises(NeverThrown):
        dataflow_audit.render_decision_snapshot(
            decision_surfaces=[],
            value_decision_surfaces=[],
            project_root=tmp_path,
            forest=None,
            forest_spec=None,
        )
