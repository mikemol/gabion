from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from typing import TYPE_CHECKING, TypedDict

from gabion.analysis.aspf.aspf_lattice_algebra import (
    FrontierWitness,
    canonical_structural_identity,
)
from gabion.analysis.foundation.artifact_ordering import canonical_mapping_keys
from gabion.json_types import JSONObject
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


@dataclass(frozen=True)
class _CanonicalValueMaterialization:
    normalized_value: object
    stable_key: str

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
    synthesized_witnesses = [
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
                "details": {
                    key: _canonical_value_materialization(value).normalized_value
                    for key, value in {
                        "surface": "projection_fiber",
                        "carrier_kind": "frontier_witness",
                        "structural_path": context["structural_path"],
                    }.items()
                },
            },
            {
                "op": SemanticOpKind.SYNTHESIZE_WITNESS,
                "details": {
                    key: _canonical_value_materialization(value).normalized_value
                    for key, value in {
                        "synthesized_witness_kind": "projection_fiber",
                        "boundary_crossing_present": bool(boundary_trace),
                    }.items()
                },
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
    def stable_objects(values: list[JSONObject]) -> list[JSONObject]:
        keyed: dict[str, JSONObject] = {}
        for value in values:
            normalized = {
                key: _canonical_value_materialization(value[key]).normalized_value
                for key in canonical_mapping_keys(value)
            }
            keyed[_canonical_value_materialization(normalized).stable_key] = normalized
        return [keyed[key] for key in canonical_mapping_keys(keyed)]

    return {
        "row_id": row["row_id"],
        "structural_identity": row["structural_identity"],
        "site_identity": row["site_identity"],
        "surface": row["surface"],
        "carrier_kind": row["carrier_kind"],
        "payload": {
            key: _canonical_value_materialization(row["payload"][key]).normalized_value
            for key in canonical_mapping_keys(row["payload"])
        },
        "input_witnesses": stable_objects(row["input_witnesses"]),
        "synthesized_witnesses": stable_objects(row["synthesized_witnesses"]),
        "obligations": stable_objects(row["obligations"]),
        "boundary_trace": stable_objects(row["boundary_trace"]),
        "transform_trace": [
            {
                "op": item["op"],
                "details": {
                    key: _canonical_value_materialization(item["details"][key]).normalized_value
                    for key in canonical_mapping_keys(item["details"])
                },
            }
            for item in row["transform_trace"]
        ],
        "obligation_state": row["obligation_state"],
    }

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
    name="semantic_fragment.canonical_value_materialization",
)
def _canonical_value_materialization(value: object) -> _CanonicalValueMaterialization:
    mapping_value = json_mapping_optional(value)
    if mapping_value is not None:
        entries = tuple(
            (
                key,
                _canonical_value_materialization(mapping_value[key]),
            )
            for key in canonical_mapping_keys(mapping_value)
        )
        return _CanonicalValueMaterialization(
            normalized_value={
                key: materialization.normalized_value
                for key, materialization in entries
            },
            stable_key="{"
            + "|".join(
                f"{key}:{materialization.stable_key}"
                for key, materialization in entries
            )
            + "}",
        )
    list_value = json_list_optional(value)
    if list_value is not None:
        items = tuple(
            _canonical_value_materialization(item) for item in list_value
        )
        return _CanonicalValueMaterialization(
            normalized_value=[item.normalized_value for item in items],
            stable_key="[" + "|".join(item.stable_key for item in items) + "]",
        )
    return _CanonicalValueMaterialization(
        normalized_value=value,
        stable_key=f"scalar:{value}",
    )


def _normalize_value(value: object) -> object:
    return _canonical_value_materialization(value).normalized_value


def _stable_json_key(value: object) -> str:
    return _canonical_value_materialization(value).stable_key


__all__ = [
    "CanonicalWitnessedSemanticRow",
    "ProjectionFiberRequestContext",
    "SemanticOpKind",
    "SemanticTransformTrace",
    "close_canonical_semantic_row",
    "reflect_projection_fiber_witness",
]
