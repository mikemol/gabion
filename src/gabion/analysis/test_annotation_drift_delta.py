# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from gabion.analysis.baseline_io import (
    attach_spec_metadata,
    load_json,
    parse_spec_metadata,
    parse_version,
    write_json,
)
from gabion.analysis.delta_tools import coerce_int, count_delta, format_delta
from gabion.analysis.projection_registry import (
    TEST_ANNOTATION_DRIFT_BASELINE_SPEC,
    TEST_ANNOTATION_DRIFT_DELTA_SPEC,
    spec_metadata_lines_from_payload,
)
from gabion.analysis.report_doc import ReportDoc
from gabion.analysis.resume_codec import mapping_or_none
from gabion.analysis.timeout_context import check_deadline
from gabion.json_types import JSONValue
from gabion.order_contract import sort_once

BASELINE_VERSION = 1
DELTA_VERSION = 1
BASELINE_RELATIVE_PATH = Path("baselines/test_annotation_drift_baseline.json")


@dataclass(frozen=True)
class AnnotationDriftBaseline:
    summary: dict[str, int]
    generated_by_spec_id: str
    generated_by_spec: dict[str, JSONValue]


@dataclass(frozen=True)
class _AnnotationDriftDeltaSummary:
    baseline: dict[str, int]
    current: dict[str, int]
    delta: dict[str, int]


def resolve_baseline_path(root: Path) -> Path:
    return root / BASELINE_RELATIVE_PATH


def build_baseline_payload(
    summary: Mapping[str, object],
) -> dict[str, JSONValue]:
    payload: dict[str, JSONValue] = {
        "version": BASELINE_VERSION,
        "summary": _normalize_summary(summary),
    }
    return attach_spec_metadata(payload, spec=TEST_ANNOTATION_DRIFT_BASELINE_SPEC)


def parse_baseline_payload(payload: Mapping[str, JSONValue]) -> AnnotationDriftBaseline:
    parse_version(
        payload,
        expected=BASELINE_VERSION,
        error_context="annotation drift baseline",
    )
    summary = _normalize_summary(payload.get("summary", {}))
    spec_id, spec = parse_spec_metadata(payload)
    return AnnotationDriftBaseline(
        summary=summary,
        generated_by_spec_id=spec_id,
        generated_by_spec=spec,
    )


def load_baseline(path: str) -> AnnotationDriftBaseline:
    return parse_baseline_payload(load_json(path))


def write_baseline(path: str, payload: Mapping[str, JSONValue]) -> None:
    write_json(path, payload)


def build_delta_payload(
    baseline: AnnotationDriftBaseline,
    current: AnnotationDriftBaseline,
    *,
    baseline_path: str = "",
) -> dict[str, JSONValue]:
    # dataflow-bundle: baseline, current
    payload: dict[str, JSONValue] = {
        "version": DELTA_VERSION,
        "summary": count_delta(baseline.summary, current.summary),
    }
    if baseline_path:
        payload["baseline"] = {"path": baseline_path}
    return attach_spec_metadata(payload, spec=TEST_ANNOTATION_DRIFT_DELTA_SPEC)


def render_markdown(payload: Mapping[str, JSONValue]) -> str:
    check_deadline()
    summary = _parse_delta_summary(payload)
    keys = sort_once(
        {*summary.baseline.keys(), *summary.current.keys(), *summary.delta.keys()},
        source="test_annotation_drift_delta.render_markdown.keys",
    )
    doc = ReportDoc("out_test_annotation_drift_delta")
    doc.lines(spec_metadata_lines_from_payload(payload))
    doc.section("Summary")
    rows = [
        f"- {key}: {summary.baseline.get(key, 0)} -> {summary.current.get(key, 0)} "
        f"({format_delta(summary.delta.get(key, summary.current.get(key, 0) - summary.baseline.get(key, 0)))})"
        for key in keys
    ]
    doc.codeblock("\n".join(rows))
    return doc.emit()


def _parse_delta_summary(
    payload: Mapping[str, JSONValue],
) -> _AnnotationDriftDeltaSummary:
    summary_payload = mapping_or_none(payload.get("summary")) or {}
    baseline_payload = mapping_or_none(summary_payload.get("baseline")) or {}
    current_payload = mapping_or_none(summary_payload.get("current")) or {}
    delta_payload = mapping_or_none(summary_payload.get("delta")) or {}
    return _AnnotationDriftDeltaSummary(
        baseline=_normalize_summary(baseline_payload),
        current=_normalize_summary(current_payload),
        delta=_normalize_summary(delta_payload),
    )


def _normalize_summary(summary: Mapping[str, object]) -> dict[str, int]:
    check_deadline()
    normalized: dict[str, int] = {}
    for key, raw in summary.items():
        normalized[str(key)] = coerce_int(raw, 0)
    return normalized
