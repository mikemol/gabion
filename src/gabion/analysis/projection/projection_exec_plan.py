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
    _normalize_predicates,
    _normalize_sort_by,
    _normalize_value,
)
from gabion.analysis.projection.projection_spec import ProjectionSpec
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
        op_name = str(op.op).strip()
        if not op_name:
            continue
        params = {str(key): value for key, value in op.params.items()}
        execution_op: ExecutionProjectionOp
        if op_name == "select":
            predicates = _normalize_predicates(_extract_predicates(params))
            if not predicates:
                continue
            execution_op = SelectExecutionOp(
                source_index=index,
                op_name=op_name,
                predicates=tuple(predicates),
            )
        elif op_name == "project":
            fields = _normalize_fields(_mapping_value(params, "fields"))
            if not fields:
                continue
            execution_op = ProjectExecutionOp(
                source_index=index,
                op_name=op_name,
                fields=tuple(fields),
            )
        elif op_name == "count_by":
            fields = _normalize_group_fields(
                _mapping_value(params, "fields")
                if "fields" in params
                else _mapping_value(params, "field")
            )
            if not fields:
                continue
            execution_op = CountByExecutionOp(
                source_index=index,
                op_name=op_name,
                fields=tuple(fields),
            )
        elif op_name == "traverse":
            field_value = _normalize_value(_mapping_value(params, "field"))
            field_text = str_optional(field_value)
            field = field_text.strip() if field_text else ""
            if not field:
                continue
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
            execution_op = TraverseExecutionOp(
                source_index=index,
                op_name=op_name,
                field=field,
                merge=merge,
                keep=keep,
                prefix=prefix,
                as_field=as_field,
                index_field=index_field,
            )
        elif op_name == "sort":
            normalized_entries = _normalize_sort_by(_mapping_value(params, "by"))
            keys = tuple(
                SortKey(
                    field=str(entry["field"]),
                    order=str(entry["order"]),
                )
                for entry in normalized_entries
            )
            if not keys:
                continue
            execution_op = SortExecutionOp(
                source_index=index,
                op_name=op_name,
                keys=keys,
            )
        elif op_name == "limit":
            try:
                count = int(_mapping_value(params, "count"))
            except (TypeError, ValueError):
                continue
            if count < 0:
                continue
            execution_op = LimitExecutionOp(
                source_index=index,
                op_name=op_name,
                count=count,
            )
        else:
            continue
        if execution_op.op_name:
            execution_ops.append(execution_op)
    return tuple(execution_ops)


def _mapping_value(params: Mapping[str, JSONValue], key: str) -> JSONValue:
    if key in params:
        return params[key]
    return []
