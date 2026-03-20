# gabion:ambiguity_boundary_module
# gabion:boundary_normalization_module
# gabion:grade_boundary kind=semantic_carrier_adapter name=dataflow_report_sections
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from gabion.analysis.dataflow.io.dataflow_report_section_contracts import (
    ReportSectionState,
)
from gabion.analysis.foundation.timeout_context import check_deadline

_REPORT_SECTION_MARKER_PREFIX = "<!-- report-section:"
_REPORT_SECTION_MARKER_SUFFIX = "-->"


@dataclass(frozen=True)
class ReportSectionMarkerParseResult:
    matched: bool = False
    section_id: str = ""


def report_section_marker(section_id: str) -> str:
    return f"{_REPORT_SECTION_MARKER_PREFIX}{section_id}{_REPORT_SECTION_MARKER_SUFFIX}"


def report_section_marker_parse_result(line: str) -> ReportSectionMarkerParseResult:
    text = line.strip()
    if not text.startswith(_REPORT_SECTION_MARKER_PREFIX):
        return ReportSectionMarkerParseResult()
    if not text.endswith(_REPORT_SECTION_MARKER_SUFFIX):
        return ReportSectionMarkerParseResult()
    section_id = text[
        len(_REPORT_SECTION_MARKER_PREFIX) : -len(_REPORT_SECTION_MARKER_SUFFIX)
    ].strip()
    if not section_id:
        return ReportSectionMarkerParseResult()
    return ReportSectionMarkerParseResult(matched=True, section_id=section_id)


def iter_report_sections(markdown: str) -> Iterator[ReportSectionState]:
    markdown_lines = markdown.splitlines()
    active_section_id = ""
    active_start_index = 0
    has_active_section = False
    for line_index, raw_line in enumerate(markdown_lines):
        check_deadline()
        marker_result = report_section_marker_parse_result(raw_line)
        if not marker_result.matched:
            continue
        if has_active_section:
            yield ReportSectionState(
                section_id=active_section_id,
                _line_iterator_factory=(
                    lambda lines=markdown_lines,
                    start_index=active_start_index,
                    end_index=line_index: iter(lines[start_index:end_index])
                ),
            )
        active_section_id = marker_result.section_id
        active_start_index = line_index + 1
        has_active_section = True
    if has_active_section:
        yield ReportSectionState(
            section_id=active_section_id,
            _line_iterator_factory=(
                lambda lines=markdown_lines,
                start_index=active_start_index: iter(lines[start_index:])
            ),
        )


def extract_report_sections(markdown: str) -> dict[str, list[str]]:
    return {
        section.section_id: list(section._line_iterator_factory())
        for section in iter_report_sections(markdown)
    }


__all__ = [
    "extract_report_sections",
    "iter_report_sections",
    "ReportSectionMarkerParseResult",
    "report_section_marker",
    "report_section_marker_parse_result",
]
