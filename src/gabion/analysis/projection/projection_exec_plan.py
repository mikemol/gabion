from __future__ import annotations

from collections.abc import Mapping
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
from gabion.invariants import grade_boundary
from gabion.runtime_shape_dispatch import str_optional

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="projection_exec_plan.execution_ops_from_spec",
)
def execution_ops_from_spec(spec: ProjectionSpec) -> tuple[ExecutionProjectionOp, ...]:
    execution_ops: list[ExecutionProjectionOp] = []
    for index, op in enumerate(spec.pipeline):
        execution_op = _execution_projection_op_from_op(index=index, op=op)
        if execution_op.op_name:
            execution_ops.append(execution_op)
    return tuple(execution_ops)


@grade_boundary(
    kind="semantic_carrier_adapter",
    name="projection_exec_plan.execution_projection_op_from_op",
)
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


@grade_boundary(
    kind="semantic_carrier_adapter",
    name="projection_exec_plan.copy_json_mapping",
)
def _copy_json_mapping(params: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    return {str(key): value for key, value in params.items()}


def _mapping_value(params: Mapping[str, JSONValue], key: str) -> JSONValue:
    if key in params:
        return params[key]
    return []


@grade_boundary(
    kind="semantic_carrier_adapter",
    name="projection_exec_plan.traverse_params_from_mapping",
)
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


@grade_boundary(
    kind="semantic_carrier_adapter",
    name="projection_exec_plan.sort_params_from_mapping",
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


@grade_boundary(
    kind="semantic_carrier_adapter",
    name="projection_exec_plan.normalized_nonempty_string",
)
def _normalized_nonempty_string(value: JSONValue) -> str:
    match value:
        case str() as text_value:
            return text_value.strip()
    return ""
