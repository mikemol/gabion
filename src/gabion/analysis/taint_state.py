# gabion:decision_protocol_module
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from gabion.analysis.baseline_io import (
    attach_spec_metadata,
    load_json,
    parse_spec_metadata,
    parse_version,
)
from gabion.analysis.projection_registry import (
    TAINT_STATE_SPEC,
)
from gabion.analysis.resume_codec import mapping_or_none, sequence_or_none
from gabion.analysis.taint_projection import (
    TaintBoundaryLocus,
    TaintProfile,
    boundary_payloads,
    build_taint_summary,
    normalize_taint_profile,
    parse_taint_boundary_registry,
    project_taint_ledgers,
)
from gabion.json_types import JSONObject, JSONValue
from gabion.order_contract import sort_once

STATE_VERSION = 1


@dataclass(frozen=True)
class TaintState:
    profile: TaintProfile
    records: list[JSONObject]
    witnesses: list[JSONObject]
    boundaries: tuple[TaintBoundaryLocus, ...]
    summary: JSONObject
    generated_by_spec_id: str
    generated_by_spec: dict[str, JSONValue]


def build_state_payload(
    *,
    marker_rows: list[Mapping[str, object]],
    boundary_registry: object = None,
    profile: object = TaintProfile.OBSERVE,
) -> JSONObject:
    normalized_profile = normalize_taint_profile(profile)
    boundaries = parse_taint_boundary_registry(boundary_registry or {})
    records, witnesses = project_taint_ledgers(
        marker_rows=marker_rows,
        boundary_registry=boundaries,
        profile=normalized_profile,
    )
    summary = build_taint_summary(records)
    payload: JSONObject = {
        "version": STATE_VERSION,
        "profile": normalized_profile.value,
        "taint_records": records,
        "taint_witnesses": witnesses,
        "boundary_registry": boundary_payloads(boundaries),
        "summary": summary,
    }
    return attach_spec_metadata(payload, spec=TAINT_STATE_SPEC)


def parse_state_payload(payload: Mapping[str, JSONValue]) -> TaintState:
    parse_version(payload, expected=STATE_VERSION, error_context="taint state")
    profile = normalize_taint_profile(payload.get("profile"))
    records = _normalized_rows(payload.get("taint_records"))
    witnesses = _normalized_rows(payload.get("taint_witnesses"))
    boundaries = parse_taint_boundary_registry(payload.get("boundary_registry", []))
    summary_payload = mapping_or_none(payload.get("summary")) or {}
    summary: JSONObject = {str(key): summary_payload[key] for key in summary_payload}
    if not summary:
        summary = build_taint_summary(records)
    spec_id, spec = parse_spec_metadata(payload)
    return TaintState(
        profile=profile,
        records=records,
        witnesses=witnesses,
        boundaries=boundaries,
        summary=summary,
        generated_by_spec_id=spec_id,
        generated_by_spec=spec,
    )


def load_state(path: str) -> TaintState:
    return parse_state_payload(load_json(path))


def _normalized_rows(payload: object) -> list[JSONObject]:
    rows_payload = sequence_or_none(payload, allow_str=False) or ()
    rows: list[JSONObject] = []
    for raw in rows_payload:
        row = mapping_or_none(raw)
        if row is not None:
            rows.append({str(key): row[key] for key in row})
    return sort_once(
        rows,
        source="taint_state._normalized_rows.rows",
        key=lambda row: (
            str(row.get("record_id", row.get("witness_id", ""))),
            str(row.get("taint_kind", "")),
            str(row.get("status", "")),
        ),
    )
