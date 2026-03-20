# gabion:ambiguity_boundary_module
# gabion:boundary_normalization_module
# gabion:grade_boundary kind=semantic_carrier_adapter name=dataflow_report_sections
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from gabion.analysis.dataflow.io.dataflow_report_section_contracts import (
    PendingReportSectionState,
    ReportSectionState,
    ReportSectionsState,
)
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.foundation.replayable_stream import (
    ReplayableStream,
    empty_stream,
    map_stream,
    stream_from_iterable,
    stream_from_iterator,
    stream_from_single,
)

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


def stream_from_resolved_report_sections(
    resolved_sections: Iterator[tuple[str, Iterator[str] | Iterable[str]]],
) -> ReplayableStream[ReportSectionState]:
    replayable_entries = stream_from_iterator(
        (
            (
                section_id,
                stream_from_iterator(iter(section_lines)),
            )
            for section_id, section_lines in resolved_sections
        )
    )
    return map_stream(
        replayable_entries,
        lambda entry: ReportSectionState(section_id=entry[0], lines=entry[1]),
    )


def stream_from_single_report_section(
    *,
    section_id: str,
    lines: Iterable[str],
) -> ReplayableStream[ReportSectionState]:
    return stream_from_single(
        ReportSectionState(
            section_id=section_id,
            lines=stream_from_iterable(lines),
        )
    )


def stream_from_pending_report_sections(
    entries: Iterator[PendingReportSectionState],
) -> ReplayableStream[PendingReportSectionState]:
    return stream_from_iterator(entries)


def resolved_section_mapping(
    resolved_sections: ReplayableStream[ReportSectionState],
) -> dict[str, list[str]]:
    return {
        section.section_id: list(section.lines)
        for section in resolved_sections
    }


def resolved_mapping(
    sections_state: ReportSectionsState,
) -> dict[str, list[str]]:
    return resolved_section_mapping(sections_state.resolved_sections)


def pending_reason_mapping(
    sections_state: ReportSectionsState,
) -> dict[str, str]:
    return {
        section.section_id: section.reason
        for section in sections_state.pending_sections
    }


def pending_section_count(
    sections_state: ReportSectionsState,
) -> int:
    return sum(1 for _ in sections_state.pending_sections)


def report_section_ids(
    sections_state: ReportSectionsState,
) -> tuple[str, ...]:
    return tuple(section.section_id for section in sections_state.resolved_sections)


def report_sections_resolved_count(
    sections_state: ReportSectionsState,
) -> int:
    return sum(1 for _ in sections_state.resolved_sections)


def report_sections_state(
    *,
    resolved_sections: ReplayableStream[ReportSectionState],
    pending_sections: ReplayableStream[PendingReportSectionState] | None = None,
) -> ReportSectionsState:
    return ReportSectionsState(
        resolved_sections=resolved_sections,
        pending_sections=pending_sections if pending_sections is not None else empty_stream(),
    )


def _projection_row_section_id(row: Mapping[str, object]) -> str:
    return str(row.get("section_id", "") or "")


def _projection_row_phase(row: Mapping[str, object]) -> str:
    return str(row.get("phase", "") or "")


def _projection_row_deps(row: Mapping[str, object]) -> tuple[str, ...]:
    deps = row.get("deps")
    if not isinstance(deps, Sequence) or isinstance(deps, str | bytes):
        return ()
    return tuple(str(dep) for dep in deps if dep is not None)


def projection_pending_sections(
    *,
    projection_rows: Sequence[Mapping[str, object]],
    resolved_sections: ReplayableStream[ReportSectionState],
    journal_reason: str | None = None,
) -> ReplayableStream[PendingReportSectionState]:
    def iter_pending_sections() -> Iterator[PendingReportSectionState]:
        resolved_section_ids = {
            section.section_id for section in resolved_sections if section.section_id
        }
        for row in projection_rows:
            check_deadline()
            section_id = _projection_row_section_id(row)
            if not section_id or section_id in resolved_section_ids:
                continue
            deps = _projection_row_deps(row)
            reason = (
                journal_reason
                if journal_reason in {"stale_input", "policy"}
                else (
                    "missing_dep"
                    if not set(deps).issubset(resolved_section_ids)
                    else "policy"
                )
            )
            yield PendingReportSectionState(
                section_id=section_id,
                phase=_projection_row_phase(row),
                deps=deps,
                reason=reason,
            )

    return stream_from_iterator(iter_pending_sections())


def overlay_report_sections_with_journal_reason(
    *,
    projection_rows: Sequence[Mapping[str, object]],
    sections_state: ReportSectionsState,
    journal_reason: str | None,
) -> ReportSectionsState:
    if journal_reason not in {"stale_input", "policy"}:
        return sections_state

    overlay_pending = projection_pending_sections(
        projection_rows=projection_rows,
        resolved_sections=sections_state.resolved_sections,
        journal_reason=journal_reason,
    )

    def iter_pending_sections() -> Iterator[PendingReportSectionState]:
        overlay_sections = tuple(overlay_pending)
        overlay_ids = {section.section_id for section in overlay_sections}
        yield from overlay_sections
        for section in sections_state.pending_sections:
            if section.section_id not in overlay_ids:
                yield section

    return report_sections_state(
        resolved_sections=sections_state.resolved_sections,
        pending_sections=stream_from_iterator(iter_pending_sections()),
    )


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
                lines=stream_from_iterable(markdown_lines[active_start_index:line_index]),
            )
        active_section_id = marker_result.section_id
        active_start_index = line_index + 1
        has_active_section = True
    if has_active_section:
        yield ReportSectionState(
            section_id=active_section_id,
            lines=stream_from_iterable(markdown_lines[active_start_index:]),
        )


def extract_report_sections(markdown: str) -> dict[str, list[str]]:
    return {
        section.section_id: list(section.lines)
        for section in iter_report_sections(markdown)
    }


__all__ = [
    "extract_report_sections",
    "iter_report_sections",
    "pending_reason_mapping",
    "pending_section_count",
    "projection_pending_sections",
    "ReportSectionMarkerParseResult",
    "overlay_report_sections_with_journal_reason",
    "report_section_ids",
    "report_section_marker",
    "report_section_marker_parse_result",
    "report_sections_resolved_count",
    "report_sections_state",
    "resolved_mapping",
    "resolved_section_mapping",
    "stream_from_pending_report_sections",
    "stream_from_resolved_report_sections",
    "stream_from_single_report_section",
]
