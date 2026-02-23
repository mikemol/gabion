# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from gabion.analysis.baseline_io import (
    attach_spec_metadata,
    load_json,
    parse_spec_metadata,
    parse_version,
    write_json,
)
from gabion.analysis.delta_tools import coerce_int, count_delta, format_delta
from gabion.analysis.projection_registry import (
    AMBIGUITY_BASELINE_SPEC,
    AMBIGUITY_DELTA_SPEC,
    spec_metadata_lines_from_payload,
)
from gabion.analysis.report_doc import ReportDoc
from gabion.analysis.timeout_context import check_deadline
from gabion.json_types import JSONValue
from gabion.order_contract import sort_once

BASELINE_VERSION = 1
DELTA_VERSION = 1
BASELINE_RELATIVE_PATH = Path("baselines/ambiguity_baseline.json")


@dataclass(frozen=True)
class AmbiguityBaseline:
    total: int
    by_kind: dict[str, int]
    generated_by_spec_id: str
    generated_by_spec: dict[str, JSONValue]


def resolve_baseline_path(root: Path) -> Path:
    return root / BASELINE_RELATIVE_PATH


def build_baseline_payload(
    ambiguity_witnesses: Iterable[Mapping[str, object]],
) -> dict[str, JSONValue]:
    counts = _count_by_kind(ambiguity_witnesses)
    payload: dict[str, JSONValue] = {
        "version": BASELINE_VERSION,
        "summary": {
            "total": sum(counts.values()),
            "by_kind": counts,
        },
    }
    return attach_spec_metadata(payload, spec=AMBIGUITY_BASELINE_SPEC)


def parse_baseline_payload(
    payload: Mapping[str, JSONValue],
) -> AmbiguityBaseline:
    check_deadline(allow_frame_fallback=True)
    parse_version(
        payload, expected=BASELINE_VERSION, error_context="ambiguity baseline"
    )
    summary = payload.get("summary", {})
    total = 0
    by_kind: dict[str, int] = {}
    if isinstance(summary, Mapping):
        total = coerce_int(summary.get("total"), 0)
        by_kind_payload = summary.get("by_kind", {})
        if isinstance(by_kind_payload, Mapping):
            for key, raw in by_kind_payload.items():
                check_deadline()
                by_kind[str(key)] = coerce_int(raw, 0)
    spec_id, spec = parse_spec_metadata(payload)
    return AmbiguityBaseline(
        total=total,
        by_kind=by_kind,
        generated_by_spec_id=spec_id,
        generated_by_spec=spec,
    )


def load_baseline(path: str) -> AmbiguityBaseline:
    return parse_baseline_payload(load_json(path))


def write_baseline(path: str, payload: Mapping[str, JSONValue]) -> None:
    write_json(path, payload)


def build_delta_payload(
    baseline: AmbiguityBaseline,
    current: AmbiguityBaseline,
    *,
    baseline_path: str | None = None,
) -> dict[str, JSONValue]:
    # dataflow-bundle: baseline, current
    by_kind = count_delta(baseline.by_kind, current.by_kind)
    payload: dict[str, JSONValue] = {
        "version": DELTA_VERSION,
        "summary": {
            "total": {
                "baseline": baseline.total,
                "current": current.total,
                "delta": current.total - baseline.total,
            },
            "by_kind": by_kind,
        },
    }
    if baseline_path:
        payload["baseline"] = {"path": baseline_path}
    return attach_spec_metadata(payload, spec=AMBIGUITY_DELTA_SPEC)


def render_markdown(
    payload: Mapping[str, JSONValue],
) -> str:
    check_deadline(allow_frame_fallback=True)
    summary = payload.get("summary", {})
    total = summary.get("total", {}) if isinstance(summary, Mapping) else {}
    by_kind = summary.get("by_kind", {}) if isinstance(summary, Mapping) else {}
    doc = ReportDoc("out_ambiguity_delta")
    doc.lines(spec_metadata_lines_from_payload(payload))
    doc.section("Summary")
    baseline_total = coerce_int(
        total.get("baseline") if isinstance(total, Mapping) else None
    )
    current_total = coerce_int(
        total.get("current") if isinstance(total, Mapping) else None
    )
    delta_total = coerce_int(
        total.get("delta") if isinstance(total, Mapping) else None,
        current_total - baseline_total,
    )
    rows = [
        f"- total: {baseline_total} -> {current_total} ({format_delta(delta_total)})"
    ]
    if isinstance(by_kind, Mapping):
        baseline = by_kind.get("baseline", {})
        current = by_kind.get("current", {})
        delta = by_kind.get("delta", {})
        kinds = sort_once(
            {*baseline.keys(), *current.keys(), *delta.keys()},
            source="render_markdown.by_kind.kinds",
        )
        for kind in kinds:
            check_deadline()
            before = baseline.get(kind, 0)
            after = current.get(kind, 0)
            change = delta.get(kind, after - before)
            rows.append(f"- {kind}: {before} -> {after} ({format_delta(change)})")
    doc.codeblock("\n".join(rows))
    return doc.emit()


def _count_by_kind(
    entries: Iterable[Mapping[str, object]],
) -> dict[str, int]:
    check_deadline(allow_frame_fallback=True)
    counts: dict[str, int] = {}
    for entry in entries:
        check_deadline()
        kind = str(entry.get("kind", "") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return counts
