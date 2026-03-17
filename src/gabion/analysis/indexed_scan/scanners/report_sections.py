from __future__ import annotations

from collections.abc import Callable, Mapping

from gabion.analysis.foundation.json_types import JSONValue
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import never

_REPORT_SECTION_MARKER_PREFIX = "<!-- report-section:"
_REPORT_SECTION_MARKER_SUFFIX = "-->"


def parse_report_section_marker(line: str) -> object:
    text = line.strip()
    if not text.startswith(_REPORT_SECTION_MARKER_PREFIX):
        return None
    if not text.endswith(_REPORT_SECTION_MARKER_SUFFIX):
        return None
    section_id = text[
        len(_REPORT_SECTION_MARKER_PREFIX) : -len(_REPORT_SECTION_MARKER_SUFFIX)
    ].strip()
    if not section_id:
        return None
    return section_id


def extract_report_sections(
    markdown: str,
    *,
    check_deadline_fn: Callable[[], None] = check_deadline,
) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    active_section_id = ""
    for raw_line in markdown.splitlines():
        check_deadline_fn()
        section_id = parse_report_section_marker(raw_line)
        match section_id:
            case str() as active_marker:
                active_section_id = active_marker
                sections.setdefault(active_marker, [])
            case None:
                if active_section_id:
                    sections[active_section_id].append(raw_line)
    return sections


def _coerce_span_field(name: str, value: JSONValue) -> int:
    if value is None:
        never(
            f"projection spec missing {name}",
            field=name,
        )
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        never(
            f"projection spec {name} must be an int",
            field=name,
            value=value,
        )
    if coerced < 0:
        never(
            "projection spec span fields must be non-negative",
            field=name,
            value=coerced,
        )
    return coerced


def spec_row_span(
    row: Mapping[str, JSONValue],
) -> tuple[int, int, int, int]:
    return (
        _coerce_span_field("span_line", row.get("span_line")),
        _coerce_span_field("span_col", row.get("span_col")),
        _coerce_span_field("span_end_line", row.get("span_end_line")),
        _coerce_span_field("span_end_col", row.get("span_end_col")),
    )


__all__ = [
    "extract_report_sections",
    "parse_report_section_marker",
    "spec_row_span",
]
