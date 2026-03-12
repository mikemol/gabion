# gabion:grade_boundary kind=semantic_carrier_adapter name=projection_exec_ingress
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Final

from gabion.analysis.projection.projection_normalize import (
    _extract_predicates,
    _normalize_fields,
    _normalize_group_fields,
    _normalize_limit,
    _normalize_predicates,
    _normalize_sort_by,
    _normalize_value,
)
from gabion.analysis.projection.projection_spec import ProjectionOp, ProjectionSpec
from gabion.json_types import JSONValue

BOUNDARY_ADAPTER_METADATA: Final[dict[str, object]] = {
    "actor": "codex",
    "rationale": (
        "Keep semantic-op erasure and presentation normalization at execution "
        "ingress while projection_exec remains a legacy row executor."
    ),
    "scope": "projection_exec.ingress_projection_spec_normalization",
    "start": "2026-03-12",
    "expiry": "projection_exec semantic_carrier_adapter retirement",
    "rollback_condition": (
        "Legacy row execution no longer accepts ProjectionSpec input directly."
    ),
    "evidence_links": [
        "src/gabion/analysis/projection/projection_exec.py",
        "docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc",
        "docs/audits/projection_semantic_fragment_ledger.md#projection_semantic_fragment_ledger",
    ],
}


@dataclass(frozen=True)
class ExecutionProjectionOp:
    source_index: int
    op_name: str
    params: dict[str, JSONValue]


def execution_ops_from_spec(spec: ProjectionSpec) -> tuple[ExecutionProjectionOp, ...]:
    execution_ops: list[ExecutionProjectionOp] = []
    for index, op in enumerate(spec.pipeline):
        execution_op = _execution_projection_op_from_op(index=index, op=op)
        if execution_op.op_name:
            execution_ops.append(execution_op)
    return tuple(execution_ops)


def _execution_projection_op_from_op(
    *,
    index: int,
    op: ProjectionOp,
) -> ExecutionProjectionOp:
    op_name = str(op.op).strip()
    if not op_name:
        return ExecutionProjectionOp(source_index=index, op_name="", params={})
    params = _copy_json_mapping(op.params)
    if op_name == "select":
        predicates = _normalize_predicates(_extract_predicates(params))
        if not predicates:
            return ExecutionProjectionOp(source_index=index, op_name="", params={})
        return ExecutionProjectionOp(
            source_index=index,
            op_name=op_name,
            params={"predicates": predicates},
        )
    if op_name == "project":
        fields = _normalize_fields(_mapping_value(params, "fields"))
        if not fields:
            return ExecutionProjectionOp(source_index=index, op_name="", params={})
        return ExecutionProjectionOp(
            source_index=index,
            op_name=op_name,
            params={"fields": list(fields)},
        )
    if op_name == "count_by":
        fields = _normalize_group_fields(
            _mapping_value(params, "fields")
            if "fields" in params
            else _mapping_value(params, "field")
        )
        if not fields:
            return ExecutionProjectionOp(source_index=index, op_name="", params={})
        return ExecutionProjectionOp(
            source_index=index,
            op_name=op_name,
            params={"fields": list(fields)},
        )
    if op_name == "traverse":
        field = _normalized_nonempty_string(_mapping_value(params, "field"))
        if not field:
            return ExecutionProjectionOp(source_index=index, op_name="", params={})
        normalized_params: dict[str, JSONValue] = {"field": field}
        for key in ("merge", "keep", "prefix", "as", "index"):
            if key in params:
                normalized_params[key] = _normalize_value(params[key])
        return ExecutionProjectionOp(
            source_index=index,
            op_name=op_name,
            params=normalized_params,
        )
    if op_name == "sort":
        by = _normalize_sort_by(_mapping_value(params, "by"))
        if not by:
            return ExecutionProjectionOp(source_index=index, op_name="", params={})
        return ExecutionProjectionOp(
            source_index=index,
            op_name=op_name,
            params={"by": by},
        )
    if op_name == "limit":
        count = _normalize_limit(_mapping_value(params, "count"))
        if count is None:
            return ExecutionProjectionOp(source_index=index, op_name="", params={})
        return ExecutionProjectionOp(
            source_index=index,
            op_name=op_name,
            params={"count": count},
        )
    return ExecutionProjectionOp(source_index=index, op_name="", params={})


def _normalized_nonempty_string(value: JSONValue) -> str:
    match value:
        case str() as text_value:
            return text_value.strip()
    return ""


def _copy_json_mapping(params: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    return {str(key): value for key, value in params.items()}


def _mapping_value(params: Mapping[str, JSONValue], key: str) -> JSONValue:
    if key in params:
        return params[key]
    return []
