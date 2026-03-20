from __future__ import annotations

from collections.abc import Mapping

from gabion.analysis.dataflow.io.dataflow_report_sections import (
    extract_report_sections,
    report_section_marker_parse_result,
)
from gabion.analysis.foundation.json_types import JSONValue
from gabion.invariants import never


def parse_report_section_marker(line: str) -> object:
    marker_result = report_section_marker_parse_result(line)
    if marker_result.matched:
        return marker_result.section_id
    return None


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
