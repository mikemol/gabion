from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from gabion.analysis.projection_registry import (
    AMBIGUITY_BASELINE_SPEC,
    AMBIGUITY_DELTA_SPEC,
    spec_metadata_lines,
    spec_metadata_payload,
)
from gabion.analysis.report_markdown import render_report_markdown
from gabion.analysis.timeout_context import check_deadline
from gabion.json_types import JSONValue
from gabion.order_contract import ordered_or_sorted

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
    payload.update(spec_metadata_payload(AMBIGUITY_BASELINE_SPEC))
    return payload


def parse_baseline_payload(
    payload: Mapping[str, JSONValue],
) -> AmbiguityBaseline:
    check_deadline(allow_frame_fallback=True)
    version = payload.get("version", BASELINE_VERSION)
    try:
        version_value = int(version) if version is not None else BASELINE_VERSION
    except (TypeError, ValueError):
        version_value = -1
    if version_value != BASELINE_VERSION:
        raise ValueError(
            f"Unsupported ambiguity baseline version={version!r}; expected {BASELINE_VERSION}"
        )
    summary = payload.get("summary", {})
    total = 0
    by_kind: dict[str, int] = {}
    if isinstance(summary, Mapping):
        total = _coerce_int(summary.get("total"), 0)
        by_kind_payload = summary.get("by_kind", {})
        if isinstance(by_kind_payload, Mapping):
            for key, raw in by_kind_payload.items():
                check_deadline()
                by_kind[str(key)] = _coerce_int(raw, 0)
    spec_id = str(payload.get("generated_by_spec_id", "") or "")
    spec_payload = payload.get("generated_by_spec", {})
    spec: dict[str, JSONValue] = {}
    if isinstance(spec_payload, Mapping):
        spec = {str(key): spec_payload[key] for key in spec_payload}
    return AmbiguityBaseline(
        total=total,
        by_kind=by_kind,
        generated_by_spec_id=spec_id,
        generated_by_spec=spec,
    )


def load_baseline(path: str) -> AmbiguityBaseline:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Ambiguity baseline must be a JSON object.")
    return parse_baseline_payload(payload)


def write_baseline(path: str, payload: Mapping[str, JSONValue]) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_delta_payload(
    baseline: AmbiguityBaseline,
    current: AmbiguityBaseline,
    *,
    baseline_path: str | None = None,
) -> dict[str, JSONValue]:
    # dataflow-bundle: baseline, current
    kinds = ordered_or_sorted(
        set(baseline.by_kind) | set(current.by_kind),
        source="build_delta_payload.kinds",
    )
    baseline_counts = {kind: baseline.by_kind.get(kind, 0) for kind in kinds}
    current_counts = {kind: current.by_kind.get(kind, 0) for kind in kinds}
    delta_counts = {
        kind: current_counts.get(kind, 0) - baseline_counts.get(kind, 0)
        for kind in kinds
    }
    payload: dict[str, JSONValue] = {
        "version": DELTA_VERSION,
        "summary": {
            "total": {
                "baseline": baseline.total,
                "current": current.total,
                "delta": current.total - baseline.total,
            },
            "by_kind": {
                "baseline": baseline_counts,
                "current": current_counts,
                "delta": delta_counts,
            },
        },
    }
    if baseline_path:
        payload["baseline"] = {"path": baseline_path}
    payload.update(spec_metadata_payload(AMBIGUITY_DELTA_SPEC))
    return payload


def render_markdown(
    payload: Mapping[str, JSONValue],
) -> str:
    check_deadline(allow_frame_fallback=True)
    summary = payload.get("summary", {})
    total = summary.get("total", {}) if isinstance(summary, Mapping) else {}
    by_kind = summary.get("by_kind", {}) if isinstance(summary, Mapping) else {}
    lines: list[str] = []
    lines.extend(spec_metadata_lines(AMBIGUITY_DELTA_SPEC))
    lines.append("Summary:")
    lines.append("```")
    baseline_total = _coerce_int(
        total.get("baseline") if isinstance(total, Mapping) else None, 0
    )
    current_total = _coerce_int(
        total.get("current") if isinstance(total, Mapping) else None, 0
    )
    delta_total = _coerce_int(
        total.get("delta") if isinstance(total, Mapping) else None,
        current_total - baseline_total,
    )
    lines.append(
        f"- total: {baseline_total} -> {current_total} ({_format_delta_value(delta_total)})"
    )
    if isinstance(by_kind, Mapping):
        baseline = by_kind.get("baseline", {})
        current = by_kind.get("current", {})
        delta = by_kind.get("delta", {})
        kinds = ordered_or_sorted(
            {*baseline.keys(), *current.keys(), *delta.keys()},
            source="render_markdown.by_kind.kinds",
        )
        for kind in kinds:
            check_deadline()
            before = baseline.get(kind, 0)
            after = current.get(kind, 0)
            change = delta.get(kind, after - before)
            lines.append(
                f"- {kind}: {before} -> {after} ({_format_delta_value(change)})"
            )
    lines.append("```")
    return render_report_markdown("out_ambiguity_delta", lines)


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


def _coerce_int(value: object, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _format_delta_value(delta: object) -> str:
    value = _coerce_int(delta, 0)
    sign = "+" if value > 0 else ""
    return f"{sign}{value}"
