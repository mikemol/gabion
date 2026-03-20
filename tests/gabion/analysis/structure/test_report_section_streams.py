from __future__ import annotations

from gabion.analysis.dataflow.io.dataflow_snapshot_io import (
    extract_report_sections,
    iter_report_sections,
)


# gabion:evidence E:function_site::dataflow/io/dataflow_snapshot_io.py::gabion.analysis.dataflow.io.dataflow_snapshot_io.iter_report_sections E:function_site::dataflow/io/dataflow_snapshot_io.py::gabion.analysis.dataflow.io.dataflow_snapshot_io.extract_report_sections
# gabion:behavior primary=desired
def test_report_section_streams_are_replayable_and_preserve_lines() -> None:
    markdown = "\n".join(
        [
            "<!-- report-section:alpha -->",
            "line-a",
            "<!-- report-section:beta -->",
            "line-b1",
            "line-b2",
        ]
    )

    def collect_sections() -> dict[str, list[str]]:
        return {
            section.section_id: list(section.lines())
            for section in iter_report_sections(markdown)
        }

    assert collect_sections() == {
        "alpha": ["line-a"],
        "beta": ["line-b1", "line-b2"],
    }
    assert collect_sections() == extract_report_sections(markdown)
