from __future__ import annotations

from pathlib import Path

import pytest

from gabion.analysis.baseline_io import (
    attach_spec_metadata,
    load_json,
    parse_spec_metadata,
    parse_version,
    write_json,
)
from gabion.analysis.delta_tools import (
    coerce_int,
    count_delta,
    format_delta,
    format_transition,
)
from gabion.analysis.projection_registry import TEST_ANNOTATION_DRIFT_DELTA_SPEC
from gabion.analysis.report_doc import ReportDoc


# gabion:evidence E:function_site::baseline_io.py::gabion.analysis.baseline_io.load_json E:function_site::baseline_io.py::gabion.analysis.baseline_io.write_json E:function_site::baseline_io.py::gabion.analysis.baseline_io.parse_version E:function_site::baseline_io.py::gabion.analysis.baseline_io.parse_spec_metadata E:function_site::baseline_io.py::gabion.analysis.baseline_io.attach_spec_metadata
def test_baseline_io_helpers_roundtrip(tmp_path: Path) -> None:
    payload = attach_spec_metadata(
        {"version": 1, "value": 2},
        spec=TEST_ANNOTATION_DRIFT_DELTA_SPEC,
    )
    path = tmp_path / "baseline.json"
    write_json(path, payload)
    loaded = load_json(path)
    assert parse_version(loaded, expected=1, error_context="unit-test") == 1
    spec_id, spec_payload = parse_spec_metadata(loaded)
    assert spec_id
    assert isinstance(spec_payload, dict)


# gabion:evidence E:function_site::baseline_io.py::gabion.analysis.baseline_io.parse_version
def test_baseline_io_parse_version_rejects_bad_version() -> None:
    with pytest.raises(ValueError):
        parse_version({"version": "bad"}, expected=1, error_context="unit-test")
    assert parse_version(
        {"schema_version": 2},
        expected=(1, 2),
        field="schema_version",
        error_context="unit-test",
    ) == 2


# gabion:evidence E:function_site::baseline_io.py::gabion.analysis.baseline_io.load_json
def test_baseline_io_load_json_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_json(path)


# gabion:evidence E:function_site::delta_tools.py::gabion.analysis.delta_tools.coerce_int E:function_site::delta_tools.py::gabion.analysis.delta_tools.format_delta E:function_site::delta_tools.py::gabion.analysis.delta_tools.format_transition E:function_site::delta_tools.py::gabion.analysis.delta_tools.count_delta
def test_delta_tools_helpers() -> None:
    assert coerce_int("nope", 7) == 7
    assert format_delta(3) == "+3"
    assert format_delta(-2) == "-2"
    assert format_transition(1, 4, None) == "1 -> 4 (+3)"
    delta = count_delta({"a": 1, "b": "bad"}, {"a": 2, "c": "5"})
    assert delta["baseline"]["a"] == 1
    assert delta["baseline"]["b"] == 0
    assert delta["current"]["c"] == 5
    assert delta["delta"]["a"] == 1


# gabion:evidence E:function_site::report_doc.py::gabion.analysis.report_doc.ReportDoc.emit
def test_report_doc_emit_renders_markdown() -> None:
    doc = ReportDoc("unit_report")
    doc.header(2, "Overview")
    doc.section("Summary")
    doc.bullets(["one", "two"])
    doc.codeblock({"k": 1})
    doc.table(["name", "count"], [["alpha", 1], ["beta", 2]])
    rendered = doc.emit()
    assert "doc_id: unit_report" in rendered
    assert "## Overview" in rendered
    assert "Summary:" in rendered
    assert "- one" in rendered
    assert '"k": 1' in rendered
    assert "| name | count |" in rendered
    assert "| alpha | 1 |" in rendered


# gabion:evidence E:function_site::report_doc.py::gabion.analysis.report_doc.ReportDoc.header E:function_site::report_doc.py::gabion.analysis.report_doc.ReportDoc.table
def test_report_doc_guards_invalid_table_or_header() -> None:
    doc = ReportDoc("unit_report")
    with pytest.raises(RuntimeError):
        doc.header(7, "bad")
    with pytest.raises(RuntimeError):
        doc.table([], [])
    with pytest.raises(RuntimeError):
        doc.table(["one"], [["a", "b"]])
