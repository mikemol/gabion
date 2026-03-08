from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, cast

from gabion.analysis.foundation.json_types import JSONObject, JSONValue


@dataclass(frozen=True)
class CallAmbiguitySummaryDeps:
    check_deadline_fn: Callable[[], None]
    apply_spec_fn: Callable[..., list[dict[str, JSONValue]]]
    ambiguity_summary_spec: object
    spec_metadata_lines_from_payload_fn: Callable[..., list[str]]
    spec_metadata_payload_fn: Callable[..., JSONObject]
    sort_once_fn: Callable[..., list[object]]
    format_span_fields_fn: Callable[..., str]


def summarize_call_ambiguities(
    entries: list[JSONObject],
    *,
    max_entries: int = 20,
    deps: CallAmbiguitySummaryDeps,
) -> list[str]:
    deps.check_deadline_fn()
    if not entries:
        return []
    relation: list[dict[str, JSONValue]] = []
    for entry in entries:
        deps.check_deadline_fn()
        match entry:
            case dict() as entry_payload:
                pass
            case _:
                continue
        kind = str(entry_payload.get("kind", "") or "unknown")
        site_payload = entry_payload.get("site", {})
        match site_payload:
            case dict() as site_mapping:
                pass
            case _:
                site_mapping = {}
        path = str(site_mapping.get("path", "") or "")
        function = str(site_mapping.get("function", "") or "")
        span = site_mapping.get("span")
        line = col = end_line = end_col = -1
        match span:
            case list() as span_values if len(span_values) == 4:
                try:
                    line = int(span_values[0])
                    col = int(span_values[1])
                    end_line = int(span_values[2])
                    end_col = int(span_values[3])
                except (TypeError, ValueError):
                    line = col = end_line = end_col = -1
        candidate_count = entry_payload.get("candidate_count")
        try:
            candidate_count = int(candidate_count) if candidate_count is not None else 0
        except (TypeError, ValueError):
            candidate_count = 0
        relation.append(
            {
                "kind": kind,
                "site_path": path,
                "site_function": function,
                "span_line": line,
                "span_col": col,
                "span_end_line": end_line,
                "span_end_col": end_col,
                "candidate_count": candidate_count,
            }
        )
    projected = deps.apply_spec_fn(deps.ambiguity_summary_spec, relation)
    counts: dict[str, int] = {}
    for row in relation:
        deps.check_deadline_fn()
        kind = str(row.get("kind", "") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    lines: list[str] = []
    lines.extend(
        deps.spec_metadata_lines_from_payload_fn(
            deps.spec_metadata_payload_fn(deps.ambiguity_summary_spec)
        )
    )
    lines.append("Counts by witness kind:")
    for kind in deps.sort_once_fn(
        counts,
        source="gabion.analysis.dataflow_indexed_file_scan._summarize_call_ambiguities.site_1",
    ):
        deps.check_deadline_fn()
        lines.append(f"- {kind}: {counts[kind]}")
    lines.append("Top ambiguous sites:")
    for row in projected[:max_entries]:
        deps.check_deadline_fn()
        path = row.get("site_path") or "?"
        function = row.get("site_function") or "?"
        span = deps.format_span_fields_fn(
            row.get("span_line", -1),
            row.get("span_col", -1),
            row.get("span_end_line", -1),
            row.get("span_end_col", -1),
        )
        count = row.get("candidate_count", 0)
        suffix = f"@{span}" if span else ""
        lines.append(f"- {path}:{function}{suffix} candidates={count}")
    if len(projected) > max_entries:
        lines.append(f"... {len(projected) - max_entries} more")
    return lines
