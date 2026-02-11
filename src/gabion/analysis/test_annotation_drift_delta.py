from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from gabion.analysis.projection_registry import (
    TEST_ANNOTATION_DRIFT_BASELINE_SPEC,
    TEST_ANNOTATION_DRIFT_DELTA_SPEC,
    spec_metadata_lines,
    spec_metadata_payload,
)
from gabion.analysis.report_markdown import render_report_markdown
from gabion.json_types import JSONValue
from gabion.analysis.timeout_context import check_deadline

BASELINE_VERSION = 1
DELTA_VERSION = 1
BASELINE_RELATIVE_PATH = Path("baselines/test_annotation_drift_baseline.json")


@dataclass(frozen=True)
class AnnotationDriftBaseline:
    summary: dict[str, int]
    generated_by_spec_id: str
    generated_by_spec: dict[str, JSONValue]


def resolve_baseline_path(root: Path) -> Path:
    return root / BASELINE_RELATIVE_PATH


def build_baseline_payload(
    summary: Mapping[str, object],
) -> dict[str, JSONValue]:
    payload: dict[str, JSONValue] = {
        "version": BASELINE_VERSION,
        "summary": _normalize_summary(summary),
    }
    payload.update(spec_metadata_payload(TEST_ANNOTATION_DRIFT_BASELINE_SPEC))
    return payload


def parse_baseline_payload(payload: Mapping[str, JSONValue]) -> AnnotationDriftBaseline:
    version = payload.get("version", BASELINE_VERSION)
    try:
        version_value = int(version) if version is not None else BASELINE_VERSION
    except (TypeError, ValueError):
        version_value = -1
    if version_value != BASELINE_VERSION:
        raise ValueError(
            "Unsupported annotation drift baseline "
            f"version={version!r}; expected {BASELINE_VERSION}"
        )
    summary = _normalize_summary(payload.get("summary", {}))
    spec_id = str(payload.get("generated_by_spec_id", "") or "")
    spec_payload = payload.get("generated_by_spec", {})
    spec: dict[str, JSONValue] = {}
    if isinstance(spec_payload, Mapping):
        spec = {str(key): spec_payload[key] for key in spec_payload}
    return AnnotationDriftBaseline(
        summary=summary,
        generated_by_spec_id=spec_id,
        generated_by_spec=spec,
    )


def load_baseline(path: str) -> AnnotationDriftBaseline:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Annotation drift baseline must be a JSON object.")
    return parse_baseline_payload(payload)


def write_baseline(path: str, payload: Mapping[str, JSONValue]) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_delta_payload(
    baseline: AnnotationDriftBaseline,
    current: AnnotationDriftBaseline,
    *,
    baseline_path: str | None = None,
) -> dict[str, JSONValue]:
    # dataflow-bundle: baseline, current
    summary_keys = sorted(set(baseline.summary) | set(current.summary))
    baseline_counts = {key: baseline.summary.get(key, 0) for key in summary_keys}
    current_counts = {key: current.summary.get(key, 0) for key in summary_keys}
    delta_counts = {
        key: current_counts.get(key, 0) - baseline_counts.get(key, 0)
        for key in summary_keys
    }
    payload: dict[str, JSONValue] = {
        "version": DELTA_VERSION,
        "summary": {
            "baseline": baseline_counts,
            "current": current_counts,
            "delta": delta_counts,
        },
    }
    if baseline_path:
        payload["baseline"] = {"path": baseline_path}
    payload.update(spec_metadata_payload(TEST_ANNOTATION_DRIFT_DELTA_SPEC))
    return payload


def render_markdown(payload: Mapping[str, JSONValue]) -> str:
    check_deadline()
    summary = payload.get("summary", {})
    baseline = summary.get("baseline", {}) if isinstance(summary, Mapping) else {}
    current = summary.get("current", {}) if isinstance(summary, Mapping) else {}
    delta = summary.get("delta", {}) if isinstance(summary, Mapping) else {}
    keys = sorted({*baseline.keys(), *current.keys(), *delta.keys()})
    lines: list[str] = []
    lines.extend(spec_metadata_lines(TEST_ANNOTATION_DRIFT_DELTA_SPEC))
    lines.append("Summary:")
    lines.append("```")
    for key in keys:
        before = baseline.get(key, 0)
        after = current.get(key, 0)
        change = delta.get(key, after - before)
        lines.append(f"- {key}: {before} -> {after} ({_format_delta_value(change)})")
    lines.append("```")
    return render_report_markdown("out_test_annotation_drift_delta", lines)


def _format_delta_value(delta: object) -> str:
    try:
        value = int(delta)
    except (TypeError, ValueError):
        value = 0
    sign = "+" if value > 0 else ""
    return f"{sign}{value}"


def _normalize_summary(summary: Mapping[str, object]) -> dict[str, int]:
    check_deadline()
    normalized: dict[str, int] = {}
    for key, raw in summary.items():
        name = str(key)
        try:
            value = int(raw) if raw is not None else 0
        except (TypeError, ValueError):
            value = 0
        normalized[name] = value
    return normalized
