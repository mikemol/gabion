from __future__ import annotations

from dataclasses import dataclass
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
from gabion.analysis.projection.projection_spec import ProjectionSpec
from gabion.invariants import decision_protocol, never


@dataclass(frozen=True)
class _LimitPlanningDecision:
    decision_kind: str
    count: int


@dataclass(frozen=True)
class _ExecutionPlanningDecision:
    source_index: int
    op_name: str


@dataclass(frozen=True)
class _SkipExecutionPlanningDecision(_ExecutionPlanningDecision):
    pass


@dataclass(frozen=True)
class _EmitExecutionPlanningDecision(_ExecutionPlanningDecision):
    execution_op: ExecutionProjectionOp


@decision_protocol
def execution_ops_from_spec(spec: ProjectionSpec) -> tuple[ExecutionProjectionOp, ...]:
    execution_ops: list[ExecutionProjectionOp] = []
    for source_index, op in enumerate(spec.pipeline):
        planning_decision = _plan_execution_op(source_index=source_index, op_name=op.op.strip(), params=op.params)
        match planning_decision:
            case _SkipExecutionPlanningDecision():
                continue
            case _EmitExecutionPlanningDecision(execution_op=execution_op):
                execution_ops.append(execution_op)
            case _ as unreachable_decision:
                never(
                    reasoning={
                        "summary": (
                            "Execution planning decisions must collapse to one "
                            "typed skip-or-emit carrier before runtime planning "
                            "continues."
                        ),
                        "control": (
                            "projection_exec_plan.execution_ops_from_spec."
                            "planning_decision_exhaustive"
                        ),
                        "blocking_dependencies": ("PSF-007",),
                    },
                    planning_decision=unreachable_decision,
                )
                continue  # pragma: no cover - never() raises
    return tuple(execution_ops)


def _plan_execution_op(
    *,
    source_index: int,
    op_name: str,
    params: dict[str, object],
) -> _ExecutionPlanningDecision:
    if not op_name:
        return _SkipExecutionPlanningDecision(source_index=source_index, op_name="")
    match op_name:
        case "select":
            predicates = tuple(_normalize_predicates(_extract_predicates(params)))
            if not predicates:
                return _SkipExecutionPlanningDecision(
                    source_index=source_index,
                    op_name=op_name,
                )
            return _EmitExecutionPlanningDecision(
                source_index=source_index,
                op_name=op_name,
                execution_op=SelectExecutionOp(
                    source_index=source_index,
                    op_name=op_name,
                    predicates=predicates,
                ),
            )
        case "project":
            fields = tuple(
                _normalize_fields(params["fields"] if "fields" in params else [])
            )
            if not fields:
                return _SkipExecutionPlanningDecision(
                    source_index=source_index,
                    op_name=op_name,
                )
            return _EmitExecutionPlanningDecision(
                source_index=source_index,
                op_name=op_name,
                execution_op=ProjectExecutionOp(
                    source_index=source_index,
                    op_name=op_name,
                    fields=fields,
                ),
            )
        case "count_by":
            fields = tuple(
                _normalize_group_fields(
                    (params["fields"] if "fields" in params else [])
                    if "fields" in params
                    else (params["field"] if "field" in params else [])
                )
            )
            if not fields:
                return _SkipExecutionPlanningDecision(
                    source_index=source_index,
                    op_name=op_name,
                )
            return _EmitExecutionPlanningDecision(
                source_index=source_index,
                op_name=op_name,
                execution_op=CountByExecutionOp(
                    source_index=source_index,
                    op_name=op_name,
                    fields=fields,
                ),
            )
        case "traverse":
            return _plan_traverse_execution_op(
                source_index=source_index,
                op_name=op_name,
                params=params,
            )
        case "sort":
            normalized_entries = _normalize_sort_by(
                params["by"] if "by" in params else []
            )
            if not normalized_entries:
                return _SkipExecutionPlanningDecision(
                    source_index=source_index,
                    op_name=op_name,
                )
            return _EmitExecutionPlanningDecision(
                source_index=source_index,
                op_name=op_name,
                execution_op=SortExecutionOp(
                    source_index=source_index,
                    op_name=op_name,
                    keys=tuple(
                        SortKey(
                            field=str(entry["field"]),
                            order=str(entry["order"]),
                        )
                        for entry in normalized_entries
                    ),
                ),
            )
        case "limit":
            normalized_limit_count = _normalize_limit(
                params["count"] if "count" in params else []
            )
            limit_decision = (
                _LimitPlanningDecision(decision_kind="skip", count=0)
                if normalized_limit_count is None
                else _LimitPlanningDecision(
                    decision_kind="emit",
                    count=normalized_limit_count,
                )
            )
            if limit_decision.decision_kind == "skip":
                return _SkipExecutionPlanningDecision(
                    source_index=source_index,
                    op_name=op_name,
                )
            return _EmitExecutionPlanningDecision(
                source_index=source_index,
                op_name=op_name,
                execution_op=LimitExecutionOp(
                    source_index=source_index,
                    op_name=op_name,
                    count=limit_decision.count,
                ),
            )
        case str():
            return _SkipExecutionPlanningDecision(
                source_index=source_index,
                op_name=op_name,
            )
        case _ as unreachable_op_name:
            never(
                reasoning={
                    "summary": (
                        "Projection op names must be normalized to strings "
                        "before execution planning continues."
                    ),
                    "control": (
                        "projection_exec_plan.execution_ops_from_spec."
                        "op_name_exhaustive"
                    ),
                    "blocking_dependencies": ("PSF-007",),
                },
                source_index=source_index,
                op_name=unreachable_op_name,
            )
            return _SkipExecutionPlanningDecision(
                source_index=source_index,
                op_name="",
            )  # pragma: no cover - never() raises


def _plan_traverse_execution_op(
    *,
    source_index: int,
    op_name: str,
    params: dict[str, object],
) -> _ExecutionPlanningDecision:
    field_value = _normalize_value(params["field"] if "field" in params else [])
    match field_value:
        case str() as raw_field if raw_field.strip():
            field = raw_field.strip()
        case str():
            return _SkipExecutionPlanningDecision(
                source_index=source_index,
                op_name=op_name,
            )
        case list() | dict() | int() | float() | bool() | None:
            return _SkipExecutionPlanningDecision(
                source_index=source_index,
                op_name=op_name,
            )
        case _ as unreachable_field:
            never(
                reasoning={
                    "summary": (
                        "Projection traversal field normalization must "
                        "discharge all normalized JSON variants before "
                        "runtime planning continues."
                    ),
                    "control": (
                        "projection_exec_plan.execution_ops_from_spec."
                        "traverse_field_exhaustive"
                    ),
                    "blocking_dependencies": ("PSF-007",),
                },
                source_index=source_index,
                op_name=op_name,
                field_value=unreachable_field,
            )
            return _SkipExecutionPlanningDecision(
                source_index=source_index,
                op_name=op_name,
            )  # pragma: no cover - never() raises

    merge_value = params["merge"] if "merge" in params else []
    match merge_value:
        case bool() as merge:
            normalized_merge = merge
        case list() | dict() | int() | float() | str() | None:
            normalized_merge = True
        case _ as unreachable_merge:
            never(
                reasoning={
                    "summary": (
                        "Projection traversal merge normalization must "
                        "discharge all normalized JSON variants before "
                        "runtime planning continues."
                    ),
                    "control": (
                        "projection_exec_plan.execution_ops_from_spec."
                        "traverse_merge_exhaustive"
                    ),
                    "blocking_dependencies": ("PSF-007",),
                },
                source_index=source_index,
                op_name=op_name,
                merge_value=unreachable_merge,
            )
            return _SkipExecutionPlanningDecision(
                source_index=source_index,
                op_name=op_name,
            )  # pragma: no cover - never() raises

    keep_value = params["keep"] if "keep" in params else []
    match keep_value:
        case bool() as keep:
            normalized_keep = keep
        case list() | dict() | int() | float() | str() | None:
            normalized_keep = False
        case _ as unreachable_keep:
            never(
                reasoning={
                    "summary": (
                        "Projection traversal keep normalization must "
                        "discharge all normalized JSON variants before "
                        "runtime planning continues."
                    ),
                    "control": (
                        "projection_exec_plan.execution_ops_from_spec."
                        "traverse_keep_exhaustive"
                    ),
                    "blocking_dependencies": ("PSF-007",),
                },
                source_index=source_index,
                op_name=op_name,
                keep_value=unreachable_keep,
            )
            return _SkipExecutionPlanningDecision(
                source_index=source_index,
                op_name=op_name,
            )  # pragma: no cover - never() raises

    prefix_value = _normalize_value(params["prefix"] if "prefix" in params else [])
    match prefix_value:
        case str() as raw_prefix if raw_prefix.strip():
            prefix = raw_prefix.strip()
        case str() | list() | dict() | int() | float() | bool() | None:
            prefix = ""
        case _ as unreachable_prefix:
            never(
                reasoning={
                    "summary": (
                        "Projection traversal prefix normalization must "
                        "discharge all normalized JSON variants before "
                        "runtime planning continues."
                    ),
                    "control": (
                        "projection_exec_plan.execution_ops_from_spec."
                        "traverse_prefix_exhaustive"
                    ),
                    "blocking_dependencies": ("PSF-007",),
                },
                source_index=source_index,
                op_name=op_name,
                prefix_value=unreachable_prefix,
            )
            return _SkipExecutionPlanningDecision(
                source_index=source_index,
                op_name=op_name,
            )  # pragma: no cover - never() raises

    as_value = _normalize_value(params["as"] if "as" in params else [])
    match as_value:
        case str() as raw_as if raw_as.strip():
            as_field = raw_as.strip()
        case str() | list() | dict() | int() | float() | bool() | None:
            as_field = field
        case _ as unreachable_as:
            never(
                reasoning={
                    "summary": (
                        "Projection traversal alias normalization must "
                        "discharge all normalized JSON variants before "
                        "runtime planning continues."
                    ),
                    "control": (
                        "projection_exec_plan.execution_ops_from_spec."
                        "traverse_as_field_exhaustive"
                    ),
                    "blocking_dependencies": ("PSF-007",),
                },
                source_index=source_index,
                op_name=op_name,
                as_value=unreachable_as,
            )
            return _SkipExecutionPlanningDecision(
                source_index=source_index,
                op_name=op_name,
            )  # pragma: no cover - never() raises

    index_value = _normalize_value(params["index"] if "index" in params else [])
    match index_value:
        case str() as raw_index if raw_index.strip():
            index_field = raw_index.strip()
        case str() | list() | dict() | int() | float() | bool() | None:
            index_field = ""
        case _ as unreachable_index:
            never(
                reasoning={
                    "summary": (
                        "Projection traversal index normalization must "
                        "discharge all normalized JSON variants before "
                        "runtime planning continues."
                    ),
                    "control": (
                        "projection_exec_plan.execution_ops_from_spec."
                        "traverse_index_field_exhaustive"
                    ),
                    "blocking_dependencies": ("PSF-007",),
                },
                source_index=source_index,
                op_name=op_name,
                index_value=unreachable_index,
            )
            return _SkipExecutionPlanningDecision(
                source_index=source_index,
                op_name=op_name,
            )  # pragma: no cover - never() raises

    return _EmitExecutionPlanningDecision(
        source_index=source_index,
        op_name=op_name,
        execution_op=TraverseExecutionOp(
            source_index=source_index,
            op_name=op_name,
            field=field,
            merge=normalized_merge,
            keep=normalized_keep,
            prefix=prefix,
            as_field=as_field,
            index_field=index_field,
        ),
    )
