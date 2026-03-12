# gabion:grade_boundary kind=semantic_carrier_adapter name=projection_exec_ingress
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Final

from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.projection.projection_exec import apply_execution_ops
from gabion.analysis.projection.projection_exec_protocol import (
    CountByExecutionOp,
    ExecutionProjectionOp,
    LimitExecutionOp,
    ProjectExecutionOp,
    SelectExecutionOp,
    SortKey,
    SortExecutionOp,
    TraverseExecutionOp,
)
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
from gabion.runtime_shape_dispatch import str_optional

PredicateRegistry = Mapping[
    str,
    Callable[[Mapping[str, JSONValue], Mapping[str, JSONValue]], bool],
]
_EMPTY_PREDICATE_REGISTRY: Final[PredicateRegistry] = {}
_EMPTY_PARAMS_OVERRIDE: Final[Mapping[str, JSONValue]] = {}

BOUNDARY_ADAPTER_METADATA: Final[dict[str, object]] = {
    "actor": "codex",
    "rationale": (
        "Keep ProjectionSpec normalization, semantic-op erasure, and "
        "presentation shaping at execution ingress while projection_exec "
        "executes typed legacy runtime steps only."
    ),
    "scope": "projection_exec.ingress_projection_spec_normalization",
    "start": "2026-03-12",
    "expiry": "projection_exec semantic_carrier_adapter retirement",
    "rollback_condition": (
        "ProjectionSpec callers no longer require a row-runtime compatibility "
        "adapter."
    ),
    "evidence_links": [
        "src/gabion/analysis/projection/projection_exec.py",
        "docs/projection_semantic_fragment_rfc.md#projection_semantic_fragment_rfc",
        "docs/audits/projection_semantic_fragment_ledger.md#projection_semantic_fragment_ledger",
    ],
}


def apply_spec(
    spec: ProjectionSpec,
    rows: Iterable[Mapping[str, JSONValue]],
    *,
    op_registry: PredicateRegistry = _EMPTY_PREDICATE_REGISTRY,
    params_override: Mapping[str, JSONValue] = _EMPTY_PARAMS_OVERRIDE,
) -> list[dict[str, JSONValue]]:
    check_deadline()
    runtime_params = _copy_json_mapping(spec.params)
    if params_override:
        runtime_params.update(
            {
                str(key): value
                for key, value in params_override.items()
            }
        )
    return apply_execution_ops(
        execution_ops_from_spec(spec),
        rows,
        op_registry=op_registry,
        runtime_params=runtime_params,
    )


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
        return ExecutionProjectionOp(source_index=index, op_name="")
    params = _copy_json_mapping(op.params)
    if op_name == "select":
        predicates = _normalize_predicates(_extract_predicates(params))
        if not predicates:
            return ExecutionProjectionOp(source_index=index, op_name="")
        return SelectExecutionOp(
            source_index=index,
            op_name=op_name,
            predicates=tuple(predicates),
        )
    if op_name == "project":
        fields = _normalize_fields(_mapping_value(params, "fields"))
        if not fields:
            return ExecutionProjectionOp(source_index=index, op_name="")
        return ProjectExecutionOp(
            source_index=index,
            op_name=op_name,
            fields=tuple(fields),
        )
    if op_name == "count_by":
        fields = _normalize_group_fields(
            _mapping_value(params, "fields")
            if "fields" in params
            else _mapping_value(params, "field")
        )
        if not fields:
            return ExecutionProjectionOp(source_index=index, op_name="")
        return CountByExecutionOp(
            source_index=index,
            op_name=op_name,
            fields=tuple(fields),
        )
    if op_name == "traverse":
        traverse_params = _traverse_params_from_mapping(params)
        if not traverse_params.field:
            return ExecutionProjectionOp(source_index=index, op_name="")
        return TraverseExecutionOp(
            source_index=index,
            op_name=op_name,
            field=traverse_params.field,
            merge=traverse_params.merge,
            keep=traverse_params.keep,
            prefix=traverse_params.prefix,
            as_field=traverse_params.as_field,
            index_field=traverse_params.index_field,
        )
    if op_name == "sort":
        sort_params = _sort_params_from_mapping(params)
        if not sort_params.keys:
            return ExecutionProjectionOp(source_index=index, op_name="")
        return SortExecutionOp(
            source_index=index,
            op_name=op_name,
            keys=sort_params.keys,
        )
    if op_name == "limit":
        count = _normalize_limit(_mapping_value(params, "count"))
        if count is None:
            return ExecutionProjectionOp(source_index=index, op_name="")
        return LimitExecutionOp(
            source_index=index,
            op_name=op_name,
            count=int(count),
        )
    return ExecutionProjectionOp(source_index=index, op_name="")


def _copy_json_mapping(params: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    return {str(key): value for key, value in params.items()}


def _mapping_value(params: Mapping[str, JSONValue], key: str) -> JSONValue:
    if key in params:
        return params[key]
    return []


def _traverse_params_from_mapping(params: Mapping[str, JSONValue]) -> TraverseExecutionOp:
    field = _normalized_nonempty_string(_mapping_value(params, "field"))
    if not field:
        return TraverseExecutionOp(source_index=-1, op_name="traverse", field="")
    merge = True
    merge_raw = _mapping_value(params, "merge")
    match merge_raw:
        case bool() as merge_bool:
            merge = merge_bool
    keep = False
    keep_raw = _mapping_value(params, "keep")
    match keep_raw:
        case bool() as keep_bool:
            keep = keep_bool
    prefix = str_optional(_normalize_value(_mapping_value(params, "prefix"))) or ""
    as_value = _normalize_value(_mapping_value(params, "as"))
    as_field_text = str_optional(as_value)
    as_field = as_field_text if as_field_text and as_field_text.strip() else field
    index_value = _normalize_value(_mapping_value(params, "index"))
    index_text = str_optional(index_value)
    index_field = index_text if index_text and index_text.strip() else ""
    return TraverseExecutionOp(
        source_index=-1,
        op_name="traverse",
        field=field,
        merge=merge,
        keep=keep,
        prefix=prefix,
        as_field=as_field,
        index_field=index_field,
    )


def _sort_params_from_mapping(params: Mapping[str, JSONValue]) -> SortExecutionOp:
    normalized_entries = _normalize_sort_by(_mapping_value(params, "by"))
    keys = tuple(
        SortKey(
            field=str(entry["field"]),
            order=str(entry["order"]),
        )
        for entry in normalized_entries
    )
    return SortExecutionOp(source_index=-1, op_name="sort", keys=keys)


def _normalized_nonempty_string(value: JSONValue) -> str:
    match value:
        case str() as text_value:
            return text_value.strip()
    return ""
