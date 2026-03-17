# gabion:ambiguity_boundary_module
# gabion:grade_boundary kind=semantic_carrier_adapter name=projection_exec_plan_ingress
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
from gabion.analysis.projection.projection_spec import ProjectionOp, ProjectionSpec
from gabion.invariants import never


@dataclass(frozen=True)
class _NormalizedExecutionOp:
    source_index: int
    op_name: str


@dataclass(frozen=True)
class _NormalizedSkipExecutionOp(_NormalizedExecutionOp):
    pass


@dataclass(frozen=True)
class _NormalizedSelectExecutionOp(_NormalizedExecutionOp):
    predicates: tuple[str, ...]


@dataclass(frozen=True)
class _NormalizedProjectExecutionOp(_NormalizedExecutionOp):
    fields: tuple[str, ...]


@dataclass(frozen=True)
class _NormalizedCountByExecutionOp(_NormalizedExecutionOp):
    fields: tuple[str, ...]


@dataclass(frozen=True)
class _NormalizedSortExecutionOp(_NormalizedExecutionOp):
    keys: tuple[SortKey, ...]


@dataclass(frozen=True)
class _NormalizedLimitExecutionOp(_NormalizedExecutionOp):
    count: int


@dataclass(frozen=True)
class _NormalizedTraverseExecutionOp(_NormalizedExecutionOp):
    field: str
    merge: bool
    keep: bool
    prefix: str
    as_field: str
    index_field: str


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


def execution_ops_from_spec(spec: ProjectionSpec) -> tuple[ExecutionProjectionOp, ...]:
    execution_ops: list[ExecutionProjectionOp] = []
    for normalized_op in _normalize_execution_ops(spec):
        planning_decision = _plan_execution_op(normalized_op=normalized_op)
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


def _normalize_execution_ops(
    spec: ProjectionSpec,
) -> tuple[_NormalizedExecutionOp, ...]:
    return tuple(
        _normalize_execution_op(source_index=source_index, op=op)
        for source_index, op in enumerate(spec.pipeline)
    )


def _normalize_execution_op(
    *,
    source_index: int,
    op: ProjectionOp,
) -> _NormalizedExecutionOp:
    op_name = op.op.strip()
    params = op.params
    if not op_name:
        return _NormalizedSkipExecutionOp(source_index=source_index, op_name="")
    match op_name:
        case "select":
            predicates = tuple(_normalize_predicates(_extract_predicates(params)))
            if not predicates:
                return _NormalizedSkipExecutionOp(
                    source_index=source_index,
                    op_name=op_name,
                )
            return _NormalizedSelectExecutionOp(
                source_index=source_index,
                op_name=op_name,
                predicates=predicates,
            )
        case "project":
            fields = tuple(_normalize_fields(params["fields"] if "fields" in params else []))
            if not fields:
                return _NormalizedSkipExecutionOp(
                    source_index=source_index,
                    op_name=op_name,
                )
            return _NormalizedProjectExecutionOp(
                source_index=source_index,
                op_name=op_name,
                fields=fields,
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
                return _NormalizedSkipExecutionOp(
                    source_index=source_index,
                    op_name=op_name,
                )
            return _NormalizedCountByExecutionOp(
                source_index=source_index,
                op_name=op_name,
                fields=fields,
            )
        case "traverse":
            return _normalize_traverse_execution_op(
                source_index=source_index,
                op_name=op_name,
                params=params,
            )
        case "sort":
            normalized_entries = _normalize_sort_by(
                params["by"] if "by" in params else []
            )
            if not normalized_entries:
                return _NormalizedSkipExecutionOp(
                    source_index=source_index,
                    op_name=op_name,
                )
            return _NormalizedSortExecutionOp(
                source_index=source_index,
                op_name=op_name,
                keys=tuple(
                    SortKey(
                        field=str(entry["field"]),
                        order=str(entry["order"]),
                    )
                    for entry in normalized_entries
                ),
            )
        case "limit":
            normalized_limit_count = _normalize_limit(
                params["count"] if "count" in params else []
            )
            if normalized_limit_count is None:
                return _NormalizedSkipExecutionOp(
                    source_index=source_index,
                    op_name=op_name,
                )
            return _NormalizedLimitExecutionOp(
                source_index=source_index,
                op_name=op_name,
                count=normalized_limit_count,
            )
        case str():
            return _NormalizedSkipExecutionOp(
                source_index=source_index,
                op_name=op_name,
            )
        case _ as unreachable_op_name:
            never(
                reasoning={
                    "summary": (
                        "Projection op names must be normalized to strings "
                        "before execution normalization continues."
                    ),
                    "control": (
                        "projection_exec_plan.normalize_execution_op."
                        "op_name_exhaustive"
                    ),
                    "blocking_dependencies": ("PSF-007",),
                },
                source_index=source_index,
                op_name=unreachable_op_name,
            )
            return _NormalizedSkipExecutionOp(
                source_index=source_index,
                op_name="",
            )  # pragma: no cover - never() raises


def _normalize_traverse_execution_op(
    *,
    source_index: int,
    op_name: str,
    params: dict[str, object],
) -> _NormalizedExecutionOp:
    field_value = _normalize_value(params["field"] if "field" in params else [])
    match field_value:
        case str() as raw_field if raw_field.strip():
            field = raw_field.strip()
        case str():
            return _NormalizedSkipExecutionOp(
                source_index=source_index,
                op_name=op_name,
            )
        case list() | dict() | int() | float() | bool() | None:
            return _NormalizedSkipExecutionOp(
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
                        "projection_exec_plan.normalize_traverse_execution_op."
                        "field_exhaustive"
                    ),
                    "blocking_dependencies": ("PSF-007",),
                },
                source_index=source_index,
                op_name=op_name,
                field_value=unreachable_field,
            )
            return _NormalizedSkipExecutionOp(
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
                        "projection_exec_plan.normalize_traverse_execution_op."
                        "merge_exhaustive"
                    ),
                    "blocking_dependencies": ("PSF-007",),
                },
                source_index=source_index,
                op_name=op_name,
                merge_value=unreachable_merge,
            )
            return _NormalizedSkipExecutionOp(
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
                        "projection_exec_plan.normalize_traverse_execution_op."
                        "keep_exhaustive"
                    ),
                    "blocking_dependencies": ("PSF-007",),
                },
                source_index=source_index,
                op_name=op_name,
                keep_value=unreachable_keep,
            )
            return _NormalizedSkipExecutionOp(
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
                        "projection_exec_plan.normalize_traverse_execution_op."
                        "prefix_exhaustive"
                    ),
                    "blocking_dependencies": ("PSF-007",),
                },
                source_index=source_index,
                op_name=op_name,
                prefix_value=unreachable_prefix,
            )
            return _NormalizedSkipExecutionOp(
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
                        "projection_exec_plan.normalize_traverse_execution_op."
                        "as_field_exhaustive"
                    ),
                    "blocking_dependencies": ("PSF-007",),
                },
                source_index=source_index,
                op_name=op_name,
                as_value=unreachable_as,
            )
            return _NormalizedSkipExecutionOp(
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
                        "projection_exec_plan.normalize_traverse_execution_op."
                        "index_field_exhaustive"
                    ),
                    "blocking_dependencies": ("PSF-007",),
                },
                source_index=source_index,
                op_name=op_name,
                index_value=unreachable_index,
            )
            return _NormalizedSkipExecutionOp(
                source_index=source_index,
                op_name=op_name,
            )  # pragma: no cover - never() raises

    return _NormalizedTraverseExecutionOp(
        source_index=source_index,
        op_name=op_name,
        field=field,
        merge=normalized_merge,
        keep=normalized_keep,
        prefix=prefix,
        as_field=as_field,
        index_field=index_field,
    )


def _plan_execution_op(
    *,
    normalized_op: _NormalizedExecutionOp,
) -> _ExecutionPlanningDecision:
    match normalized_op:
        case _NormalizedSkipExecutionOp(
            source_index=source_index,
            op_name=op_name,
        ):
            return _SkipExecutionPlanningDecision(
                source_index=source_index,
                op_name=op_name,
            )
        case _NormalizedSelectExecutionOp(
            source_index=source_index,
            op_name=op_name,
            predicates=predicates,
        ):
            return _EmitExecutionPlanningDecision(
                source_index=source_index,
                op_name=op_name,
                execution_op=SelectExecutionOp(
                    source_index=source_index,
                    op_name=op_name,
                    predicates=predicates,
                ),
            )
        case _NormalizedProjectExecutionOp(
            source_index=source_index,
            op_name=op_name,
            fields=fields,
        ):
            return _EmitExecutionPlanningDecision(
                source_index=source_index,
                op_name=op_name,
                execution_op=ProjectExecutionOp(
                    source_index=source_index,
                    op_name=op_name,
                    fields=fields,
                ),
            )
        case _NormalizedCountByExecutionOp(
            source_index=source_index,
            op_name=op_name,
            fields=fields,
        ):
            return _EmitExecutionPlanningDecision(
                source_index=source_index,
                op_name=op_name,
                execution_op=CountByExecutionOp(
                    source_index=source_index,
                    op_name=op_name,
                    fields=fields,
                ),
            )
        case _NormalizedSortExecutionOp(
            source_index=source_index,
            op_name=op_name,
            keys=keys,
        ):
            return _EmitExecutionPlanningDecision(
                source_index=source_index,
                op_name=op_name,
                execution_op=SortExecutionOp(
                    source_index=source_index,
                    op_name=op_name,
                    keys=keys,
                ),
            )
        case _NormalizedLimitExecutionOp(
            source_index=source_index,
            op_name=op_name,
            count=count,
        ):
            return _EmitExecutionPlanningDecision(
                source_index=source_index,
                op_name=op_name,
                execution_op=LimitExecutionOp(
                    source_index=source_index,
                    op_name=op_name,
                    count=count,
                ),
            )
        case _NormalizedTraverseExecutionOp() as normalized_traverse:
            return _plan_traverse_execution_op(normalized_op=normalized_traverse)
        case _ as unreachable_normalized_op:
            never(
                reasoning={
                    "summary": (
                        "Normalized execution planning must consume only the "
                        "declared typed normalized carriers."
                    ),
                    "control": (
                        "projection_exec_plan.plan_execution_op."
                        "normalized_op_exhaustive"
                    ),
                    "blocking_dependencies": ("PSF-007",),
                },
                normalized_op=unreachable_normalized_op,
            )
            return _SkipExecutionPlanningDecision(
                source_index=normalized_op.source_index,
                op_name=normalized_op.op_name,
            )  # pragma: no cover - never() raises


def _plan_traverse_execution_op(
    *,
    normalized_op: _NormalizedTraverseExecutionOp,
) -> _ExecutionPlanningDecision:
    return _EmitExecutionPlanningDecision(
        source_index=normalized_op.source_index,
        op_name=normalized_op.op_name,
        execution_op=TraverseExecutionOp(
            source_index=normalized_op.source_index,
            op_name=normalized_op.op_name,
            field=normalized_op.field,
            merge=normalized_op.merge,
            keep=normalized_op.keep,
            prefix=normalized_op.prefix,
            as_field=normalized_op.as_field,
            index_field=normalized_op.index_field,
        ),
    )
