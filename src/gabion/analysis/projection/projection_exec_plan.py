from __future__ import annotations

from gabion.analysis.projection.projection_exec_plan_ingress import (
    _EmitExecutionPlanningDecision,
    _NormalizedCountByExecutionOp,
    _NormalizedExecutionOp,
    _NormalizedLimitExecutionOp,
    _NormalizedProjectExecutionOp,
    _NormalizedSelectExecutionOp,
    _NormalizedSkipExecutionOp,
    _NormalizedSortExecutionOp,
    _NormalizedTraverseExecutionOp,
    _SkipExecutionPlanningDecision,
    _normalize_execution_op,
    _normalize_execution_ops,
    _normalize_traverse_execution_op,
    _plan_execution_op,
    _plan_traverse_execution_op,
    execution_ops_from_spec as _execution_ops_from_spec_ingress,
)
from gabion.analysis.projection.projection_exec_protocol import ExecutionProjectionOp
from gabion.analysis.projection.projection_spec import ProjectionSpec

__all__ = [
    "_EmitExecutionPlanningDecision",
    "_NormalizedCountByExecutionOp",
    "_NormalizedExecutionOp",
    "_NormalizedLimitExecutionOp",
    "_NormalizedProjectExecutionOp",
    "_NormalizedSelectExecutionOp",
    "_NormalizedSkipExecutionOp",
    "_NormalizedSortExecutionOp",
    "_NormalizedTraverseExecutionOp",
    "_SkipExecutionPlanningDecision",
    "_normalize_execution_op",
    "_normalize_execution_ops",
    "_normalize_traverse_execution_op",
    "_plan_execution_op",
    "_plan_traverse_execution_op",
    "execution_ops_from_spec",
]


# gabion:grade_boundary kind=semantic_carrier_adapter name=projection_exec_plan.execution_ops_from_spec
def execution_ops_from_spec(spec: ProjectionSpec) -> tuple[ExecutionProjectionOp, ...]:
    return _execution_ops_from_spec_ingress(spec)
