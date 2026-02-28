# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, cast

from gabion.analysis.baseline_io import (
    attach_spec_metadata,
    load_json,
    parse_spec_metadata,
    parse_version,
    write_json,
)
from gabion.analysis.delta_tools import coerce_int, count_delta, format_delta
from gabion.analysis.projection_registry import (
    TAINT_BASELINE_SPEC,
    TAINT_DELTA_SPEC,
    spec_metadata_lines_from_payload,
)
from gabion.analysis.report_doc import ReportDoc
from gabion.analysis.resume_codec import mapping_or_none, sequence_or_none
from gabion.analysis.taint_projection import build_taint_summary
from gabion.analysis.timeout_context import check_deadline
from gabion.json_types import JSONObject, JSONValue
from gabion.order_contract import sort_once

BASELINE_VERSION = 1
DELTA_VERSION = 1
BASELINE_RELATIVE_PATH = Path("baselines/taint_baseline.json")


@dataclass(frozen=True)
class TaintBaseline:
    summary: JSONObject
    records: list[JSONObject]
    generated_by_spec_id: str
    generated_by_spec: dict[str, JSONValue]

    @property
    def by_status(self) -> Mapping[str, int]:
        payload = mapping_or_none(self.summary.get("by_status")) or {}
        return {str(key): coerce_int(payload[key], 0) for key in payload}

    @property
    def by_kind(self) -> Mapping[str, int]:
        payload = mapping_or_none(self.summary.get("by_kind")) or {}
        return {str(key): coerce_int(payload[key], 0) for key in payload}


def resolve_baseline_path(root: Path) -> Path:
    return root / BASELINE_RELATIVE_PATH


def build_baseline_payload(records: Iterable[Mapping[str, object]]) -> JSONObject:
    normalized_records = _normalize_records(records)
    summary = build_taint_summary(normalized_records)
    payload: JSONObject = {
        "version": BASELINE_VERSION,
        "summary": summary,
        "taint_records": normalized_records,
    }
    return attach_spec_metadata(payload, spec=TAINT_BASELINE_SPEC)


def parse_baseline_payload(payload: Mapping[str, JSONValue]) -> TaintBaseline:
    parse_version(payload, expected=BASELINE_VERSION, error_context="taint baseline")
    summary_payload = mapping_or_none(payload.get("summary")) or {}
    summary: JSONObject = {str(key): summary_payload[key] for key in summary_payload}
    records = _normalize_records(sequence_or_none(payload.get("taint_records"), allow_str=False) or ())
    if not summary:
        summary = build_taint_summary(records)
    spec_id, spec = parse_spec_metadata(payload)
    return TaintBaseline(
        summary=summary,
        records=records,
        generated_by_spec_id=spec_id,
        generated_by_spec=spec,
    )


def load_baseline(path: str) -> TaintBaseline:
    return parse_baseline_payload(load_json(path))


def write_baseline(path: str, payload: Mapping[str, JSONValue]) -> None:
    write_json(path, payload)


def build_delta_payload(
    baseline: TaintBaseline,
    current: TaintBaseline,
    *,
    baseline_path: str = "",
) -> JSONObject:
    status_delta = count_delta(baseline.by_status, current.by_status)
    kind_delta = count_delta(baseline.by_kind, current.by_kind)
    baseline_total = coerce_int(baseline.summary.get("total"), 0)
    current_total = coerce_int(current.summary.get("total"), 0)
    baseline_strict = coerce_int(baseline.summary.get("strict_unresolved"), 0)
    current_strict = coerce_int(current.summary.get("strict_unresolved"), 0)
    payload: JSONObject = {
        "version": DELTA_VERSION,
        "summary": {
            "total": {
                "baseline": baseline_total,
                "current": current_total,
                "delta": current_total - baseline_total,
            },
            "strict_unresolved": {
                "baseline": baseline_strict,
                "current": current_strict,
                "delta": current_strict - baseline_strict,
            },
            "by_status": status_delta,
            "by_kind": kind_delta,
        },
    }
    if baseline_path:
        payload["baseline"] = {"path": baseline_path}
    return attach_spec_metadata(payload, spec=TAINT_DELTA_SPEC)


def render_markdown(payload: Mapping[str, JSONValue]) -> str:
    check_deadline()
    summary_payload = mapping_or_none(payload.get("summary")) or {}
    total_payload = mapping_or_none(summary_payload.get("total")) or {}
    strict_payload = mapping_or_none(summary_payload.get("strict_unresolved")) or {}
    by_status_payload = mapping_or_none(summary_payload.get("by_status")) or {}
    by_kind_payload = mapping_or_none(summary_payload.get("by_kind")) or {}
    doc = ReportDoc("out_taint_delta")
    doc.lines(spec_metadata_lines_from_payload(payload))
    doc.section("Summary")
    rows: list[str] = []
    rows.append(
        _render_delta_row(
            label="total",
            payload=total_payload,
        )
    )
    rows.append(
        _render_delta_row(
            label="strict_unresolved",
            payload=strict_payload,
        )
    )
    rows.extend(_render_bucket_rows(name="by_status", payload=by_status_payload))
    rows.extend(_render_bucket_rows(name="by_kind", payload=by_kind_payload))
    doc.codeblock("\n".join(rows))
    return doc.emit()


def _render_delta_row(*, label: str, payload: Mapping[str, object]) -> str:
    before = coerce_int(payload.get("baseline"), 0)
    after = coerce_int(payload.get("current"), 0)
    delta = coerce_int(payload.get("delta"), after - before)
    return f"- {label}: {before} -> {after} ({format_delta(delta)})"


def _render_bucket_rows(*, name: str, payload: Mapping[str, object]) -> list[str]:
    baseline = mapping_or_none(payload.get("baseline")) or {}
    current = mapping_or_none(payload.get("current")) or {}
    delta = mapping_or_none(payload.get("delta")) or {}
    keys = sort_once(
        {*baseline.keys(), *current.keys(), *delta.keys()},
        source=f"taint_delta._render_bucket_rows.{name}.keys",
    )
    rows: list[str] = []
    for key in keys:
        check_deadline()
        before = coerce_int(baseline.get(key), 0)
        after = coerce_int(current.get(key), 0)
        change = coerce_int(delta.get(key), after - before)
        rows.append(f"- {name}[{key}]: {before} -> {after} ({format_delta(change)})")
    return rows


def _normalize_records(records: Iterable[Mapping[str, object]]) -> list[JSONObject]:
    normalized: list[JSONObject] = []
    for row in records:
        check_deadline()
        normalized.append({str(key): cast(JSONValue, row[key]) for key in row})
    return sort_once(
        normalized,
        source="taint_delta._normalize_records.normalized",
        key=lambda row: (
            str(row.get("record_id", "")),
            str(row.get("taint_kind", "")),
            str(row.get("status", "")),
        ),
    )
