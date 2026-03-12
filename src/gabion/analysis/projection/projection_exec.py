from __future__ import annotations

from functools import singledispatch
from collections.abc import Callable, Iterable, Mapping
from typing import Final, cast

from gabion.analysis.projection.projection_exec_protocol import (
    CountByExecutionOp,
    ExecutionProjectionOp,
    LimitExecutionOp,
    ProjectExecutionOp,
    SelectExecutionOp,
    SortExecutionOp,
    TraverseExecutionOp,
)
from gabion.json_types import JSONValue
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import grade_boundary
from gabion.order_contract import OrderPolicy, sort_once
from gabion.runtime.stable_encode import stable_compact_text
from gabion.runtime_shape_dispatch import (
    json_list_optional,
    json_mapping_optional,
)

Relation = list[dict[str, JSONValue]]
PredicateRegistry = Mapping[
    str,
    Callable[[Mapping[str, JSONValue], Mapping[str, JSONValue]], bool],
]
_EMPTY_PREDICATE_REGISTRY: Final[PredicateRegistry] = {}
_EMPTY_RUNTIME_PARAMS: Final[Mapping[str, JSONValue]] = {}

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="projection_exec.apply_execution_ops",
)
def apply_execution_ops(
    execution_ops: Iterable[ExecutionProjectionOp],
    rows: Iterable[Mapping[str, JSONValue]],
    *,
    op_registry: PredicateRegistry = _EMPTY_PREDICATE_REGISTRY,
    runtime_params: Mapping[str, JSONValue] = _EMPTY_RUNTIME_PARAMS,
) -> Relation:
    check_deadline()
    params = {str(key): value for key, value in runtime_params.items()}

    current: Relation = [
        dict(cast(Mapping[str, JSONValue], row))
        for row in rows
    ]

    for execution_op in execution_ops:
        check_deadline()
        current = _apply_execution_op(
            execution_op,
            current,
            op_registry=op_registry,
            runtime_params=params,
        )

    return current

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="projection_exec.apply_execution_op",
)
@singledispatch
def _apply_execution_op(
    execution_op: ExecutionProjectionOp,
    rows: Relation,
    *,
    op_registry: PredicateRegistry,
    runtime_params: Mapping[str, JSONValue],
) -> Relation:
    _ = execution_op, op_registry, runtime_params
    return rows

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="projection_exec.apply_select_execution_op",
)
@_apply_execution_op.register
def _apply_select_execution_op(
    execution_op: SelectExecutionOp,
    rows: Relation,
    *,
    op_registry: PredicateRegistry,
    runtime_params: Mapping[str, JSONValue],
) -> Relation:
    selected = rows
    for predicate_name in execution_op.predicates:
        check_deadline()
        predicate = op_registry.get(predicate_name)
        if predicate is not None:
            selected = [row for row in selected if predicate(row, runtime_params)]
    return selected

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="projection_exec.apply_project_execution_op",
)
@_apply_execution_op.register
def _apply_project_execution_op(
    execution_op: ProjectExecutionOp,
    rows: Relation,
    *,
    op_registry: PredicateRegistry,
    runtime_params: Mapping[str, JSONValue],
) -> Relation:
    _ = op_registry, runtime_params
    if not execution_op.fields:
        return rows
    projected: Relation = []
    for row in rows:
        check_deadline()
        projected.append({field: row.get(field) for field in execution_op.fields})
    return projected

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="projection_exec.apply_count_by_execution_op",
)
@_apply_execution_op.register
def _apply_count_by_execution_op(
    execution_op: CountByExecutionOp,
    rows: Relation,
    *,
    op_registry: PredicateRegistry,
    runtime_params: Mapping[str, JSONValue],
) -> Relation:
    _ = op_registry, runtime_params
    if not execution_op.fields:
        return rows
    counts: dict[tuple[str, ...], dict[str, JSONValue]] = {}
    for row in rows:
        check_deadline()
        key_parts: list[str] = []
        for field in execution_op.fields:
            check_deadline()
            key_parts.append(_canonical_group_reference(row.get(field)))
        key = tuple(key_parts)
        record = counts.get(key)
        if record is None:
            record = {field: row.get(field) for field in execution_op.fields}
            record["count"] = 0
            counts[key] = record
        record["count"] = int(record.get("count", 0)) + 1
    ordered_group_keys = sort_once(
        counts,
        source="apply_execution_ops.count_by.group_keys",
        policy=OrderPolicy.SORT,
        key=lambda key: tuple(
            _sort_value(counts[key].get(field)) for field in execution_op.fields
        ),
    )
    return [counts[key] for key in ordered_group_keys]

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="projection_exec.apply_traverse_execution_op",
)
@_apply_execution_op.register
def _apply_traverse_execution_op(
    execution_op: TraverseExecutionOp,
    rows: Relation,
    *,
    op_registry: PredicateRegistry,
    runtime_params: Mapping[str, JSONValue],
) -> Relation:
    _ = op_registry, runtime_params
    if not execution_op.field:
        return rows
    traversed: Relation = []
    for row in rows:
        check_deadline()
        seq = row.get(execution_op.field)
        items = json_list_optional(seq)
        if items is not None:
            base = dict(row)
            if not execution_op.keep:
                base.pop(execution_op.field, None)
            for idx, element in enumerate(items):
                check_deadline()
                out = dict(base)
                if execution_op.index_field:
                    out[execution_op.index_field] = idx
                element_map = json_mapping_optional(element)
                if element_map is not None and execution_op.merge:
                    for key, value in element_map.items():
                        check_deadline()
                        merged_key = (
                            f"{execution_op.prefix}{key}"
                            if execution_op.prefix
                            else key
                        )
                        out[str(merged_key)] = value
                else:
                    out[execution_op.as_field] = element
                traversed.append(out)
    return traversed

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="projection_exec.apply_sort_execution_op",
)
@_apply_execution_op.register
def _apply_sort_execution_op(
    execution_op: SortExecutionOp,
    rows: Relation,
    *,
    op_registry: PredicateRegistry,
    runtime_params: Mapping[str, JSONValue],
) -> Relation:
    _ = op_registry, runtime_params
    current = rows
    for key in reversed(execution_op.keys):
        check_deadline()
        current = sort_once(
            current,
            source=f"apply_execution_ops.sort[{key.field}]",
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
    op_registry: PredicateRegistry,
    runtime_params: Mapping[str, JSONValue],
) -> Relation:
    _ = op_registry, runtime_params
    return rows[:execution_op.count]

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="projection_exec.sort_value",
)
def _sort_value(value: JSONValue) -> tuple[int, object]:
    if value is None:
        return (4, "")
    match value:
        case bool():
            return (0, int(value))
        case int() | float():
            return (1, value)
        case str():
            return (2, value)
    return (3, stable_compact_text(value))

@grade_boundary(
    kind="semantic_carrier_adapter",
    name="projection_exec.canonical_group_reference",
)
def _canonical_group_reference(value: JSONValue) -> str:
    return stable_compact_text(value)
