from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, TypedDict

from gabion.analysis.aspf.aspf_lattice_algebra import (
    FrontierWitness,
    canonical_structural_identity,
)
from gabion.analysis.foundation.artifact_ordering import canonical_mapping_keys
from gabion.json_types import JSONObject, JSONValue
from gabion.invariants import grade_boundary
from gabion.runtime_shape_dispatch import json_list_optional, json_mapping_optional

if TYPE_CHECKING:
    from gabion.analysis.aspf.aspf_lattice_algebra import UnmappedWitness


class SemanticOpKind(str):
    REFLECT = "reflect"
    QUOTIENT_FACE = "quotient_face"
    SYNTHESIZE_WITNESS = "synthesize_witness"
    WEDGE = "wedge"
    REINDEX = "reindex"
    EXISTENTIAL_IMAGE = "existential_image"
    SUPPORT_REFLECT = "support_reflect"
    NEGATE = "negate"


class SemanticTransformTrace(TypedDict):
    op: str
    details: JSONObject


class ProjectionFiberRequestContext(TypedDict):
    path: str
    qualname: str
    structural_path: str
    line: int
    column: int
    node_kind: str
    required_symbols: tuple[str, ...]


class CanonicalWitnessedSemanticRow(TypedDict):
    row_id: str
    structural_identity: str
    site_identity: str
    surface: str
    carrier_kind: str
    payload: JSONObject
    input_witnesses: list[JSONObject]
    synthesized_witnesses: list[JSONObject]
    obligations: list[JSONObject]
    boundary_trace: list[JSONObject]
    transform_trace: list[SemanticTransformTrace]
    obligation_state: str

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment.reflect_projection_fiber_witness",
)
def reflect_projection_fiber_witness(
    *,
    context: ProjectionFiberRequestContext,
    witness: FrontierWitness,
) -> CanonicalWitnessedSemanticRow:
    structural_identity = canonical_structural_identity(
        rel_path=context["path"],
        qualname=context["qualname"],
        structural_path=context["structural_path"],
        node_kind=context["node_kind"],
        surface="projection_fiber",
    )
    synthesized_witnesses = _synthesized_witness_payloads(witness)
    boundary_trace = [
        {
            "kind": "boundary_crossing",
            "crossing_id": item.crossing_id,
            "branch_site_id": item.branch_site_id,
            "branch_site_identity": item.branch_site_identity,
            "boundary_kind": item.boundary_kind,
        }
        for item in witness.boundary_crossings
    ]
    intro_ids = {item.obligation_id for item in witness.obligations}
    erased_ids = {item.obligation_id for item in witness.erasures}
    unresolved_obligations = intro_ids - erased_ids
    obligation_state = (
        "unresolved"
        if witness.violation is not None
        or not witness.complete
        or unresolved_obligations
        else "erased"
        if intro_ids
        else "discharged"
    )
    row: CanonicalWitnessedSemanticRow = {
        "row_id": structural_identity,
        "structural_identity": structural_identity,
        "site_identity": witness.branch_site_identity,
        "surface": "projection_fiber",
        "carrier_kind": "frontier_witness",
        "payload": {
            "path": context["path"],
            "qualname": context["qualname"],
            "line": context["line"],
            "column": context["column"],
            "node_kind": context["node_kind"],
            "structural_path": context["structural_path"],
            "required_symbols": [item for item in context["required_symbols"]],
            "unresolved_symbols": [item for item in witness.unresolved_symbols],
            "complete": witness.complete,
            "has_violation": witness.violation is not None,
            "bundle_event_count": witness.bundle_event_count,
            "bundle_edge_count": witness.bundle_edge_count,
            "execution_event_count": witness.execution_event_count,
            "execution_edge_count": witness.execution_edge_count,
            "data_anchor_site_identity": witness.data_anchor_site_identity,
            "exec_frontier_site_identity": witness.exec_frontier_site_identity,
        },
        "input_witnesses": _input_witness_payloads(witness),
        "synthesized_witnesses": synthesized_witnesses,
        "obligations": _obligation_payloads(witness),
        "boundary_trace": boundary_trace,
        "transform_trace": [
            {
                "op": SemanticOpKind.REFLECT,
                "details": _normalize_object(
                    {
                        "surface": "projection_fiber",
                        "carrier_kind": "frontier_witness",
                        "structural_path": context["structural_path"],
                    }
                ),
            },
            {
                "op": SemanticOpKind.SYNTHESIZE_WITNESS,
                "details": _normalize_object(
                    {
                        "synthesized_witness_kind": "projection_fiber",
                        "boundary_crossing_present": bool(boundary_trace),
                    }
                ),
            },
        ],
        "obligation_state": obligation_state,
    }
    return close_canonical_semantic_row(row)

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment.close_canonical_semantic_row",
)
def close_canonical_semantic_row(
    row: CanonicalWitnessedSemanticRow,
) -> CanonicalWitnessedSemanticRow:
    return {
        "row_id": row["row_id"],
        "structural_identity": row["structural_identity"],
        "site_identity": row["site_identity"],
        "surface": row["surface"],
        "carrier_kind": row["carrier_kind"],
        "payload": _normalize_object(row["payload"]),
        "input_witnesses": _stable_unique_objects(row["input_witnesses"]),
        "synthesized_witnesses": _stable_unique_objects(row["synthesized_witnesses"]),
        "obligations": _stable_unique_objects(row["obligations"]),
        "boundary_trace": _stable_unique_objects(row["boundary_trace"]),
        "transform_trace": [
            {
                "op": item["op"],
                "details": _normalize_object(item["details"]),
            }
            for item in row["transform_trace"]
        ],
        "obligation_state": row["obligation_state"],
    }

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment.input_witness_payloads",
)
def _input_witness_payloads(witness: FrontierWitness) -> list[JSONObject]:
    return [
        {
            "kind": "branch_frontier",
            "branch_site_id": witness.branch_site_id,
            "branch_site_identity": witness.branch_site_identity,
            "branch_line": witness.branch_line,
            "branch_column": witness.branch_column,
            "required_symbols": [item for item in witness.required_symbols],
        },
        {
            "kind": "data_anchor",
            "site_id": witness.data_anchor_site_id,
            "site_identity": witness.data_anchor_site_identity,
            "line": witness.data_anchor_line,
            "column": witness.data_anchor_column,
            "ordinal": witness.data_anchor_ordinal,
        },
        {
            "kind": "exec_frontier",
            "site_id": witness.exec_frontier_site_id,
            "site_identity": witness.exec_frontier_site_identity,
            "line": witness.exec_frontier_line,
            "column": witness.exec_frontier_column,
            "ordinal": witness.exec_frontier_ordinal,
        },
    ]

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment.synthesized_witness_payloads",
)
def _synthesized_witness_payloads(witness: FrontierWitness) -> list[JSONObject]:
    return [
        _join_payload(
            kind="join",
            left=witness.data_exec_join.left_ids,
            right=witness.data_exec_join.right_ids,
            result=witness.data_exec_join.result_ids,
            deterministic=witness.data_exec_join.deterministic,
        ),
        _join_payload(
            kind="meet",
            left=witness.data_exec_meet.left_ids,
            right=witness.data_exec_meet.right_ids,
            result=witness.data_exec_meet.result_ids,
            deterministic=witness.data_exec_meet.deterministic,
        ),
        _naturality_payload(
            kind="eta_data_to_exec",
            direction=witness.eta_data_to_exec.direction,
            mapped_source=witness.eta_data_to_exec.mapped_source_site_ids,
            mapped_target=witness.eta_data_to_exec.mapped_target_site_ids,
            unmapped=witness.eta_data_to_exec.unmapped,
            complete=witness.eta_data_to_exec.complete,
        ),
        _naturality_payload(
            kind="eta_exec_to_data",
            direction=witness.eta_exec_to_data.direction,
            mapped_source=witness.eta_exec_to_data.mapped_source_site_ids,
            mapped_target=witness.eta_exec_to_data.mapped_target_site_ids,
            unmapped=witness.eta_exec_to_data.unmapped,
            complete=witness.eta_exec_to_data.complete,
        ),
    ]

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment.join_payload",
)
def _join_payload(
    *,
    kind: str,
    left: Iterable[str],
    right: Iterable[str],
    result: Iterable[str],
    deterministic: bool,
) -> JSONObject:
    return {
        "kind": kind,
        "left_ids": [item for item in left],
        "right_ids": [item for item in right],
        "result_ids": [item for item in result],
        "deterministic": deterministic,
    }

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment.naturality_payload",
)
def _naturality_payload(
    *,
    kind: str,
    direction: str,
    mapped_source: Iterable[str],
    mapped_target: Iterable[str],
    unmapped: Iterable[UnmappedWitness],
    complete: bool,
) -> JSONObject:
    return {
        "kind": kind,
        "direction": direction,
        "mapped_source_site_ids": [item for item in mapped_source],
        "mapped_target_site_ids": [item for item in mapped_target],
        "unmapped": [
            {
                "source_kind": item.source_kind,
                "source_site_id": item.source_site_id,
                "source_site_identity": item.source_site_identity,
                "reason": item.reason,
            }
            for item in unmapped
        ],
        "complete": complete,
    }

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment.obligation_payloads",
)
def _obligation_payloads(witness: FrontierWitness) -> list[JSONObject]:
    violation_payloads = (
        [
            {
                "kind": "violation",
                "violation_id": witness.violation.violation_id,
                "boundary_crossing_id": witness.violation.boundary_crossing_id,
                "unresolved_obligation_ids": [
                    item for item in witness.violation.unresolved_obligation_ids
                ],
                "reason": witness.violation.reason,
            }
        ]
        if witness.violation is not None
        else []
    )
    return [
        {
            "kind": "completeness",
            "complete": witness.complete,
        },
        *[
            {
                "kind": "obligation_intro",
                "obligation_id": item.obligation_id,
                "source_kind": item.source_kind,
                "source_site_id": item.source_site_id,
                "source_site_identity": item.source_site_identity,
                "reason": item.reason,
                "introduced_by": item.introduced_by,
            }
            for item in witness.obligations
        ],
        *[
            {
                "kind": "obligation_erase",
                "obligation_id": item.obligation_id,
                "erased_by": item.erased_by,
                "reason": item.reason,
            }
            for item in witness.erasures
        ],
        *violation_payloads,
    ]

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment.stable_unique_objects",
)
def _stable_unique_objects(values: list[JSONObject]) -> list[JSONObject]:
    keyed: dict[str, JSONObject] = {}
    for value in values:
        normalized = _normalize_object(value)
        keyed[_stable_json_key(normalized)] = normalized
    return [keyed[key] for key in canonical_mapping_keys(keyed)]

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment.normalize_object",
)
def _normalize_object(value: JSONObject) -> JSONObject:
    return {key: _normalize_value(value[key]) for key in canonical_mapping_keys(value)}

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment.normalize_value",
)
def _normalize_value(value: object) -> JSONValue:
    mapping_value = json_mapping_optional(value)
    if mapping_value is not None:
        return _normalize_mapping(mapping_value)
    list_value = json_list_optional(value)
    if list_value is not None:
        return _normalize_list(list_value)
    return value

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment.normalize_mapping",
)
def _normalize_mapping(value: dict[str, JSONValue]) -> JSONValue:
    return {key: _normalize_value(value[key]) for key in canonical_mapping_keys(value)}

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment.normalize_list",
)
def _normalize_list(value: list[JSONValue]) -> JSONValue:
    return [_normalize_value(item) for item in value]

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="semantic_fragment.stable_json_key",
)
def _stable_json_key(value: object) -> str:
    mapping_value = json_mapping_optional(value)
    if mapping_value is not None:
        parts = (
            f"{key}:{_stable_json_key(mapping_value[key])}"
            for key in canonical_mapping_keys(mapping_value)
        )
        return "{" + "|".join(parts) + "}"
    list_value = json_list_optional(value)
    if list_value is not None:
        return "[" + "|".join(_stable_json_key(item) for item in list_value) + "]"
    return f"scalar:{value}"


__all__ = [
    "CanonicalWitnessedSemanticRow",
    "ProjectionFiberRequestContext",
    "SemanticOpKind",
    "SemanticTransformTrace",
    "close_canonical_semantic_row",
    "reflect_projection_fiber_witness",
]
