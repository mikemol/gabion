from __future__ import annotations

import pytest

from gabion.analysis.indexed_scan.scanners import report_sections
from gabion.exceptions import NeverThrown


# gabion:evidence E:function_site::indexed_scan/report_sections.py::gabion.analysis.indexed_scan.report_sections.parse_report_section_marker E:function_site::indexed_scan/report_sections.py::gabion.analysis.indexed_scan.report_sections.extract_report_sections
def test_parse_and_extract_report_sections() -> None:
    assert (
        report_sections.parse_report_section_marker("<!-- report-section:alpha -->")
        == "alpha"
    )
    assert report_sections.parse_report_section_marker("<!-- report-section: -->") is None
    assert (
        report_sections.parse_report_section_marker("<!-- report-section:alpha")
        is None
    )
    assert report_sections.parse_report_section_marker("not-a-marker") is None

    markdown = "\n".join(
        [
            "<!-- report-section:alpha -->",
            "line-a",
            "<!-- report-section:beta -->",
            "line-b1",
            "line-b2",
        ]
    )
    extracted = report_sections.extract_report_sections(markdown)
    assert extracted == {
        "alpha": ["line-a"],
        "beta": ["line-b1", "line-b2"],
    }


# gabion:evidence E:function_site::indexed_scan/report_sections.py::gabion.analysis.indexed_scan.report_sections.spec_row_span
def test_spec_row_span_validation() -> None:
    assert report_sections.spec_row_span(
        {
            "span_line": 1,
            "span_col": 2,
            "span_end_line": 3,
            "span_end_col": 4,
        }
    ) == (1, 2, 3, 4)

    with pytest.raises(NeverThrown):
        report_sections.spec_row_span(
            {
                "span_line": -1,
                "span_col": 2,
                "span_end_line": 3,
                "span_end_col": 4,
            }
        )
    with pytest.raises(NeverThrown):
        report_sections.spec_row_span(
            {
                "span_col": 2,
                "span_end_line": 3,
                "span_end_col": 4,
            }
        )
    with pytest.raises(NeverThrown):
        report_sections.spec_row_span(
            {
                "span_line": "x",
                "span_col": 2,
                "span_end_line": 3,
                "span_end_col": 4,
            }
        )
