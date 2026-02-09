from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from gabion.analysis import ambiguity_delta
from gabion.analysis.projection_registry import (
    AMBIGUITY_STATE_SPEC,
    spec_metadata_payload,
)
from gabion.analysis.timeout_context import check_deadline
from gabion.json_types import JSONValue

STATE_VERSION = 1


@dataclass(frozen=True)
class AmbiguityState:
    witnesses: list[dict[str, JSONValue]]
    baseline: ambiguity_delta.AmbiguityBaseline
    generated_by_spec_id: str
    generated_by_spec: dict[str, JSONValue]


def build_state_payload(
    ambiguity_witnesses: Iterable[Mapping[str, object]],
) -> dict[str, JSONValue]:
    # dataflow-bundle: ambiguity_witnesses
    ordered = _normalize_witnesses(ambiguity_witnesses)
    baseline_payload = ambiguity_delta.build_baseline_payload(ordered)
    payload: dict[str, JSONValue] = {
        "version": STATE_VERSION,
        "ambiguity_witnesses": ordered,
        "summary": baseline_payload.get("summary", {}),
    }
    payload.update(spec_metadata_payload(AMBIGUITY_STATE_SPEC))
    return payload


def parse_state_payload(
    payload: Mapping[str, JSONValue],
) -> AmbiguityState:
    version = payload.get("version", STATE_VERSION)
    try:
        version_value = int(version) if version is not None else STATE_VERSION
    except (TypeError, ValueError):
        version_value = -1
    if version_value != STATE_VERSION:
        raise ValueError(
            "Unsupported ambiguity state "
            f"version={version!r}; expected {STATE_VERSION}"
        )
    witnesses_payload = payload.get("ambiguity_witnesses", [])
    witnesses = _normalize_witnesses(witnesses_payload)
    baseline_payload = ambiguity_delta.build_baseline_payload(witnesses)
    baseline = ambiguity_delta.parse_baseline_payload(baseline_payload)
    spec_id = str(payload.get("generated_by_spec_id", "") or "")
    spec_payload = payload.get("generated_by_spec", {})
    spec: dict[str, JSONValue] = {}
    if isinstance(spec_payload, Mapping):
        spec = {str(key): spec_payload[key] for key in spec_payload}
    return AmbiguityState(
        witnesses=witnesses,
        baseline=baseline,
        generated_by_spec_id=spec_id,
        generated_by_spec=spec,
    )


def load_state(path: str) -> AmbiguityState:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Ambiguity state must be a JSON object.")
    return parse_state_payload(payload)


def _normalize_witnesses(
    payload: Iterable[Mapping[str, object]] | object,
) -> list[dict[str, JSONValue]]:
    if not isinstance(payload, Iterable) or isinstance(payload, (str, bytes, dict)):
        return []
    check_deadline(allow_frame_fallback=True)
    witnesses: list[dict[str, JSONValue]] = []
    for entry in payload:
        if not isinstance(entry, Mapping):
            continue
        witnesses.append({str(k): entry[k] for k in entry})
    witnesses.sort(key=_witness_sort_key)
    return witnesses


def _witness_sort_key(entry: Mapping[str, object]) -> tuple[object, ...]:
    kind = str(entry.get("kind", "") or "")
    site = entry.get("site", {})
    if isinstance(site, Mapping):
        path = str(site.get("path", "") or "")
        func = str(site.get("function", "") or "")
        span = site.get("span")
    else:
        path = ""
        func = ""
        span = None
    span_values: tuple[object, ...]
    if isinstance(span, list) and len(span) == 4:
        span_values = tuple(span)
    else:
        span_values = ()
    candidate_count = entry.get("candidate_count", 0)
    return (kind, path, func, span_values, candidate_count)
