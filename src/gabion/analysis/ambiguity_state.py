# gabion:decision_protocol_module
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from gabion.analysis import ambiguity_delta
from gabion.analysis.baseline_io import (
    attach_spec_metadata,
    load_json,
    parse_spec_metadata,
    parse_version,
)
from gabion.analysis.projection_registry import (
    AMBIGUITY_STATE_SPEC,
)
from gabion.analysis.resume_codec import mapping_or_empty, mapping_or_none, sequence_or_none
from gabion.analysis.timeout_context import check_deadline
from gabion.json_types import JSONValue
from gabion.order_contract import sort_once

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
    return attach_spec_metadata(payload, spec=AMBIGUITY_STATE_SPEC)


def parse_state_payload(
    payload: Mapping[str, JSONValue],
) -> AmbiguityState:
    parse_version(payload, expected=STATE_VERSION, error_context="ambiguity state")
    witnesses_payload = payload.get("ambiguity_witnesses", [])
    witnesses = _normalize_witnesses(witnesses_payload)
    baseline_payload = ambiguity_delta.build_baseline_payload(witnesses)
    baseline = ambiguity_delta.parse_baseline_payload(baseline_payload)
    spec_id, spec = parse_spec_metadata(payload)
    return AmbiguityState(
        witnesses=witnesses,
        baseline=baseline,
        generated_by_spec_id=spec_id,
        generated_by_spec=spec,
    )


def load_state(path: str) -> AmbiguityState:
    return parse_state_payload(load_json(path))


def _normalize_witnesses(
    payload: object,
) -> list[dict[str, JSONValue]]:
    entries = sequence_or_none(payload)
    if entries is None:
        return list()
    check_deadline(allow_frame_fallback=True)
    witnesses: list[dict[str, JSONValue]] = []
    for entry in entries:
        check_deadline()
        parsed_entry = mapping_or_none(entry)
        if parsed_entry is not None:
            witnesses.append({str(key): parsed_entry[key] for key in parsed_entry})
    return sort_once(
        witnesses,
        source="_normalize_witnesses.witnesses",
        key=_witness_sort_key,
    )


def _witness_sort_key(entry: Mapping[str, object]) -> tuple[object, ...]:
    kind = str(entry.get("kind", "") or "")
    site = mapping_or_empty(entry.get("site", {}))
    path = str(site.get("path", "") or "")
    func = str(site.get("function", "") or "")
    span = sequence_or_none(site.get("span"))
    span_values: tuple[object, ...] = tuple()
    if span is not None and len(span) == 4:
        span_values = tuple(span)
    candidate_count = entry.get("candidate_count", 0)
    return (kind, path, func, span_values, candidate_count)
