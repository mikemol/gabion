# gabion:grade_boundary kind=semantic_carrier_adapter name=projection_exec
from __future__ import annotations

from functools import singledispatch
import json
from collections.abc import Callable, Iterable, Mapping
from typing import cast

from gabion.analysis.projection.projection_exec_ingress import (
    ExecutionProjectionOp,
    execution_ops_from_spec,
)
from gabion.analysis.projection.projection_exec_protocol import (
    CountByExecutionOp,
    LimitExecutionOp,
    ProjectExecutionOp,
    SelectExecutionOp,
    SortExecutionOp,
    TraverseExecutionOp,
)
from gabion.analysis.projection.projection_spec import ProjectionSpec
from gabion.json_types import JSONValue
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.order_contract import OrderPolicy, sort_once
from gabion.runtime_shape_dispatch import (
    json_list_optional,
    json_mapping_optional,
)

Relation = list[dict[str, JSONValue]]


def apply_spec(
    spec: ProjectionSpec,
    rows: Iterable[Mapping[str, JSONValue]],
    *,
    op_registry = None,
    params_override = None,
) -> Relation:
    check_deadline()
    op_registry = op_registry or {}
    params = _copy_json_mapping(spec.params)
    if params_override:
        params.update(params_override)

    current: Relation = [
        dict(cast(Mapping[str, JSONValue], row))
        for row in rows
    ]

    for execution_op in execution_ops_from_spec(spec):
        check_deadline()
        current = _apply_normalized_execution_op(
            current,
            execution_op=execution_op,
            op_registry=op_registry,
            runtime_params=params,
        )

    return current


def _apply_normalized_execution_op(
    rows: Relation,
    *,
    execution_op: ExecutionProjectionOp,
    op_registry: Mapping[str, Callable[[Mapping[str, JSONValue], Mapping[str, JSONValue]], bool]],
    runtime_params: Mapping[str, JSONValue],
) -> Relation:
    return _apply_execution_op(
        execution_op,
        rows,
        op_registry=op_registry,
        runtime_params=runtime_params,
    )


@singledispatch
def _apply_execution_op(
    execution_op: ExecutionProjectionOp,
    rows: Relation,
    *,
    op_registry: Mapping[str, Callable[[Mapping[str, JSONValue], Mapping[str, JSONValue]], bool]],
    runtime_params: Mapping[str, JSONValue],
) -> Relation:
    _ = execution_op, op_registry, runtime_params
    return rows


@_apply_execution_op.register
def _apply_select_execution_op(
    execution_op: SelectExecutionOp,
    rows: Relation,
    *,
    op_registry: Mapping[str, Callable[[Mapping[str, JSONValue], Mapping[str, JSONValue]], bool]],
    runtime_params: Mapping[str, JSONValue],
) -> Relation:
    return _apply_select(
        rows,
        execution_op,
        op_registry=op_registry,
        runtime_params=runtime_params,
    )


@_apply_execution_op.register
def _apply_project_execution_op(
    execution_op: ProjectExecutionOp,
    rows: Relation,
    *,
    op_registry: Mapping[str, Callable[[Mapping[str, JSONValue], Mapping[str, JSONValue]], bool]],
    runtime_params: Mapping[str, JSONValue],
) -> Relation:
    _ = op_registry, runtime_params
    if execution_op.fields:
        return _apply_project(rows, execution_op)
    return rows


@_apply_execution_op.register
def _apply_count_by_execution_op(
    execution_op: CountByExecutionOp,
    rows: Relation,
    *,
    op_registry: Mapping[str, Callable[[Mapping[str, JSONValue], Mapping[str, JSONValue]], bool]],
    runtime_params: Mapping[str, JSONValue],
) -> Relation:
    _ = op_registry, runtime_params
    if execution_op.fields:
        return _apply_count_by(rows, execution_op)
    return rows


@_apply_execution_op.register
def _apply_traverse_execution_op(
    execution_op: TraverseExecutionOp,
    rows: Relation,
    *,
    op_registry: Mapping[str, Callable[[Mapping[str, JSONValue], Mapping[str, JSONValue]], bool]],
    runtime_params: Mapping[str, JSONValue],
) -> Relation:
    _ = op_registry, runtime_params
    if execution_op.field:
        return _apply_traverse(rows, execution_op)
    return rows


@_apply_execution_op.register
def _apply_sort_execution_op(
    execution_op: SortExecutionOp,
    rows: Relation,
    *,
    op_registry: Mapping[str, Callable[[Mapping[str, JSONValue], Mapping[str, JSONValue]], bool]],
    runtime_params: Mapping[str, JSONValue],
) -> Relation:
    _ = op_registry, runtime_params
    current = rows
    for key in reversed(execution_op.keys):
        check_deadline()
        current = sort_once(
            current,
            source=f"apply_spec.sort[{key.field}]",
            policy=OrderPolicy.SORT,
            key=lambda row, name=key.field: _sort_value(row.get(name)),
            reverse=key.order == "desc",
        )
    return current


@_apply_execution_op.register
def _apply_limit_execution_op(
    execution_op: LimitExecutionOp,
    rows: Relation,
    *,
    op_registry: Mapping[str, Callable[[Mapping[str, JSONValue], Mapping[str, JSONValue]], bool]],
    runtime_params: Mapping[str, JSONValue],
) -> Relation:
    _ = op_registry, runtime_params
    if execution_op.count >= 0:
        return rows[:execution_op.count]
    return rows


def _copy_json_mapping(params: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    return {str(key): value for key, value in params.items()}


def _sort_value(value: JSONValue) -> tuple[int, object]:
    if value is None:
        return (1, "")
    match value:
        case int() | float() | str():
            return (0, value)
    return (0, str(value))


def _hashable(value: JSONValue) -> object:
    try:
        hash(value)
    except TypeError:
        return json.dumps(value, sort_keys=False, separators=(",", ":"))
    return value


def _apply_select(
    rows: Relation,
    select_params: SelectExecutionOp,
    *,
    op_registry: Mapping[str, Callable[[Mapping[str, JSONValue], Mapping[str, JSONValue]], bool]],
    runtime_params: Mapping[str, JSONValue],
) -> Relation:
    selected = rows
    for predicate_name in select_params.predicates:
        check_deadline()
        predicate = op_registry.get(predicate_name)
        if predicate is not None:
            selected = [row for row in selected if predicate(row, runtime_params)]
    return selected


def _apply_project(rows: Relation, params: ProjectExecutionOp) -> Relation:
    projected: Relation = []
    for row in rows:
        check_deadline()
        projected.append({field: row.get(field) for field in params.fields})
    return projected


def _apply_count_by(rows: Relation, params: CountByExecutionOp) -> Relation:
    counts: dict[tuple[object, ...], dict[str, JSONValue]] = {}
    for row in rows:
        check_deadline()
        key_parts: list[object] = []
        for field in params.fields:
            check_deadline()
            key_parts.append(_hashable(row.get(field)))
        key = tuple(key_parts)
        record = counts.get(key)
        if record is None:
            record = {field: row.get(field) for field in params.fields}
            record["count"] = 0
            counts[key] = record
        record["count"] = int(record.get("count", 0)) + 1
    ordered_group_keys = sort_once(
        counts,
        source="apply_spec.count_by.group_keys",
        policy=OrderPolicy.SORT,
        key=lambda key: tuple(_sort_value(cast(JSONValue, part)) for part in key),
    )
    return [counts[key] for key in ordered_group_keys]


def _apply_traverse(rows: Relation, params: TraverseExecutionOp) -> Relation:
    traversed: Relation = []
    for row in rows:
        check_deadline()
        seq = row.get(params.field)
        items = json_list_optional(seq)
        if items is not None:
            base = dict(row)
            if not params.keep:
                base.pop(params.field, None)
            for idx, element in enumerate(items):
                check_deadline()
                out = dict(base)
                if params.index_field:
                    out[params.index_field] = idx
                element_map = json_mapping_optional(element)
                if element_map is not None and params.merge:
                    for key, value in element_map.items():
                        check_deadline()
                        merged_key = f"{params.prefix}{key}" if params.prefix else key
                        out[str(merged_key)] = value
                else:
                    out[params.as_field] = element
                traversed.append(out)
    return traversed
