# gabion:ambiguity_boundary_module
# gabion:boundary_normalization_module
# gabion:grade_boundary kind=semantic_carrier_adapter name=dataflow_report_sections
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import chain, tee

from gabion.analysis.dataflow.io.dataflow_report_section_contracts import (
    PendingReportSectionState,
    ReportSectionState,
    ReportSectionsState,
)
from gabion.analysis.foundation.timeout_context import check_deadline

_REPORT_SECTION_MARKER_PREFIX = "<!-- report-section:"
_REPORT_SECTION_MARKER_SUFFIX = "-->"


@dataclass(frozen=True)
class ReportSectionMarkerParseResult:
    matched: bool = False
    section_id: str = ""


def tee_iterator_factory[T](items: Iterator[T]) -> Callable[[], Iterator[T]]:
    source = items

    def iter_items() -> Iterator[T]:
        nonlocal source
        source, clone = tee(source)
        return clone

    return iter_items


def report_section_lines(section: ReportSectionState) -> Iterator[str]:
    return section._line_iterator_factory()


def resolved_sections(
    sections_state: ReportSectionsState,
) -> Iterator[ReportSectionState]:
    return sections_state._resolved_section_iterator_factory()


def pending_sections(
    sections_state: ReportSectionsState,
) -> Iterator[PendingReportSectionState]:
    return sections_state._pending_section_iterator_factory()


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


def resolved_report_section_states(
    resolved_sections: Iterator[tuple[str, Iterator[str] | Iterable[str]]],
) -> Callable[[], Iterator[ReportSectionState]]:
    section_iterator_factory = tee_iterator_factory(resolved_sections)

    def iter_sections() -> Iterator[ReportSectionState]:
        for section_id, section_lines in section_iterator_factory():
            yield ReportSectionState(
                section_id=section_id,
                _line_iterator_factory=tee_iterator_factory(iter(section_lines)),
            )

    return iter_sections


def single_report_section_state(
    *,
    section_id: str,
    lines: Iterable[str],
) -> Callable[[], Iterator[ReportSectionState]]:
    line_iterator_factory = tee_iterator_factory(iter(lines))

    def iter_sections() -> Iterator[ReportSectionState]:
        yield ReportSectionState(
            section_id=section_id,
            _line_iterator_factory=line_iterator_factory,
        )

    return iter_sections


def chain_report_section_states(
    *section_streams: Callable[[], Iterator[ReportSectionState]],
) -> Callable[[], Iterator[ReportSectionState]]:
    return lambda: chain.from_iterable(
        section_stream() for section_stream in section_streams
    )


def empty_report_section_states() -> Iterator[ReportSectionState]:
    return iter(())


def pending_report_section_states(
    entries: Iterator[PendingReportSectionState],
) -> Callable[[], Iterator[PendingReportSectionState]]:
    return tee_iterator_factory(entries)


def resolved_section_mapping(
    resolved_sections: Callable[[], Iterator[ReportSectionState]],
) -> dict[str, list[str]]:
    return {
        section.section_id: list(report_section_lines(section))
        for section in resolved_sections()
    }


def resolved_mapping(
    sections_state: ReportSectionsState,
) -> dict[str, list[str]]:
    return resolved_section_mapping(lambda: resolved_sections(sections_state))


def pending_reason_mapping(
    sections_state: ReportSectionsState,
) -> dict[str, str]:
    return {
        section.section_id: section.reason
        for section in pending_sections(sections_state)
    }


def report_section_ids(
    sections_state: ReportSectionsState,
) -> tuple[str, ...]:
    return tuple(section.section_id for section in resolved_sections(sections_state))


def report_sections_resolved_count(
    sections_state: ReportSectionsState,
) -> int:
    return sum(1 for _ in resolved_sections(sections_state))


def report_sections_state(
    *,
    resolved_sections: Callable[[], Iterator[ReportSectionState]],
    pending_sections: Callable[[], Iterator[PendingReportSectionState]] | None = None,
) -> ReportSectionsState:
    return ReportSectionsState(
        _resolved_section_iterator_factory=resolved_sections,
        _pending_section_iterator_factory=(
            pending_sections if pending_sections is not None else (lambda: iter(()))
        ),
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
        section.section_id: list(report_section_lines(section))
        for section in iter_report_sections(markdown)
    }


__all__ = [
    "chain_report_section_states",
    "empty_report_section_states",
    "extract_report_sections",
    "iter_report_sections",
    "pending_reason_mapping",
    "pending_report_section_states",
    "pending_sections",
    "ReportSectionMarkerParseResult",
    "report_section_ids",
    "report_section_marker",
    "report_section_marker_parse_result",
    "report_section_lines",
    "report_sections_resolved_count",
    "report_sections_state",
    "resolved_mapping",
    "resolved_report_section_states",
    "resolved_section_mapping",
    "resolved_sections",
    "single_report_section_state",
    "tee_iterator_factory",
]
