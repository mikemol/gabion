from __future__ import annotations

from gabion.analysis.projection.projection_exec import apply_execution_ops
from gabion.analysis.projection.projection_exec_protocol import (
    CountByExecutionOp,
    LimitExecutionOp,
    ProjectExecutionOp,
    SelectExecutionOp,
    SortKey,
    SortExecutionOp,
    TraverseExecutionOp,
)
from gabion.analysis.projection.projection_exec_plan import (
    _EmitExecutionPlanningDecision,
    _SkipExecutionPlanningDecision,
    _plan_execution_op,
    _plan_traverse_execution_op,
    execution_ops_from_spec,
)
from gabion.analysis.projection.projection_spec import ProjectionOp, ProjectionSpec


def _apply_spec(
    spec: ProjectionSpec,
    rows,
    *,
    op_registry=None,
    params_override=None,
):
    runtime_params = dict(spec.params)
    if params_override:
        runtime_params.update(params_override)
    return apply_execution_ops(
        execution_ops_from_spec(spec),
        rows,
        op_registry=op_registry or {},
        runtime_params=runtime_params,
    )


def test_execution_ops_from_spec_normalizes_presentation_ops() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="demo",
        domain="tests",
        pipeline=(
            ProjectionOp(op="select", params={"predicates": ["keep", "", "keep"]}),
            ProjectionOp(op="project", params={"fields": ["status", "status", "id"]}),
            ProjectionOp(op="count_by", params={"field": "status"}),
            ProjectionOp(op="sort", params={"by": ["status"]}),
            ProjectionOp(op="limit", params={"count": 3}),
        ),
    )

    execution_ops = execution_ops_from_spec(spec)

    assert tuple(op.op_name for op in execution_ops) == (
        "select",
        "project",
        "count_by",
        "sort",
        "limit",
    )
    assert execution_ops[0] == SelectExecutionOp(
        source_index=0,
        op_name="select",
        predicates=("keep",),
    )
    assert execution_ops[1] == ProjectExecutionOp(
        source_index=1,
        op_name="project",
        fields=("status", "id"),
    )
    assert execution_ops[2] == CountByExecutionOp(
        source_index=2,
        op_name="count_by",
        fields=("status",),
    )
    assert execution_ops[3] == SortExecutionOp(
        source_index=3,
        op_name="sort",
        keys=(SortKey(field="status", order="asc"),),
    )
    assert execution_ops[4] == LimitExecutionOp(
        source_index=4,
        op_name="limit",
        count=3,
    )


def test_execution_ops_from_spec_erases_semantic_metadata_and_semantic_only_ops() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="demo",
        domain="tests",
        pipeline=(
            ProjectionOp(
                "project",
                {
                    "fields": ["id"],
                    "quotient_face": "projection_fiber.frontier",
                },
            ),
            ProjectionOp("reflect", {"surface": "projection_fiber"}),
            ProjectionOp("custom", {"a": 1}),
        ),
    )

    execution_ops = execution_ops_from_spec(spec)

    assert len(execution_ops) == 1
    assert execution_ops[0] == ProjectExecutionOp(
        source_index=0,
        op_name="project",
        fields=("id",),
    )


def test_execution_ops_from_spec_skips_negative_limit() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="demo",
        domain="tests",
        pipeline=(ProjectionOp(op="limit", params={"count": -1}),),
    )

    assert execution_ops_from_spec(spec) == ()


# gabion:evidence E:function_site::projection_exec_plan.py::gabion.analysis.projection.projection_exec_plan._plan_execution_op
# gabion:behavior primary=desired facets=edge
def test_plan_execution_op_emits_and_skips_with_typed_decisions() -> None:
    skipped = _plan_execution_op(
        source_index=1,
        op_name="project",
        params={"fields": []},
    )
    emitted = _plan_execution_op(
        source_index=2,
        op_name="project",
        params={"fields": ["status", "status", "id"]},
    )

    assert isinstance(skipped, _SkipExecutionPlanningDecision)
    assert skipped.source_index == 1
    assert skipped.op_name == "project"

    assert isinstance(emitted, _EmitExecutionPlanningDecision)
    assert emitted.source_index == 2
    assert emitted.op_name == "project"
    assert emitted.execution_op == ProjectExecutionOp(
        source_index=2,
        op_name="project",
        fields=("status", "id"),
    )


# gabion:evidence E:function_site::projection_exec_plan.py::gabion.analysis.projection.projection_exec_plan._plan_traverse_execution_op
# gabion:behavior primary=desired facets=edge
def test_plan_traverse_execution_op_normalizes_defaults_and_explicit_values() -> None:
    skipped = _plan_traverse_execution_op(
        source_index=3,
        op_name="traverse",
        params={"field": {"bad": True}},
    )
    defaulted = _plan_traverse_execution_op(
        source_index=4,
        op_name="traverse",
        params={
            "field": " items ",
            "merge": "bad",
            "keep": "bad",
            "prefix": ["bad"],
            "as": 0,
            "index": True,
        },
    )
    explicit = _plan_traverse_execution_op(
        source_index=5,
        op_name="traverse",
        params={
            "field": " items ",
            "merge": False,
            "keep": True,
            "prefix": " pre_ ",
            "as": " item ",
            "index": " idx ",
        },
    )

    assert isinstance(skipped, _SkipExecutionPlanningDecision)
    assert skipped.source_index == 3
    assert skipped.op_name == "traverse"

    assert isinstance(defaulted, _EmitExecutionPlanningDecision)
    assert defaulted.execution_op == TraverseExecutionOp(
        source_index=4,
        op_name="traverse",
        field="items",
        merge=True,
        keep=False,
        prefix="",
        as_field="items",
        index_field="",
    )

    assert isinstance(explicit, _EmitExecutionPlanningDecision)
    assert explicit.execution_op == TraverseExecutionOp(
        source_index=5,
        op_name="traverse",
        field="items",
        merge=False,
        keep=True,
        prefix="pre_",
        as_field="item",
        index_field="idx",
    )


def test_apply_spec_handles_invalid_and_semantic_only_ops_at_exec_ingress() -> None:
    rows = [
        {"group": ["a"], "value": 1},
        {"group": ["a"], "value": 2},
    ]

    def keep(_row, params):
        return int(params.get("threshold", 0)) <= 2

    spec = ProjectionSpec(
        spec_version=1,
        name="demo",
        domain="tests",
        params={"threshold": 2},
        pipeline=(
            ProjectionOp("select", {"predicates": "keep"}),
            ProjectionOp("select", {"predicates": {"bad": True}}),
            ProjectionOp("select", {"predicates": [1, "missing", "keep"]}),
            ProjectionOp("project", {"fields": 1}),
            ProjectionOp("project", {"fields": "group"}),
            ProjectionOp("count_by", {"fields": ["group", 5]}),
            ProjectionOp("sort", {"by": {"field": "count", "order": 1}}),
            ProjectionOp("sort", {"by": "bad"}),
            ProjectionOp("limit", {"count": "bad"}),
            ProjectionOp("reflect", {"surface": "projection_fiber"}),
            ProjectionOp("custom", {"bad": True}),
        ),
    )

    result = _apply_spec(
        spec,
        rows,
        op_registry={"keep": keep},
    )
    assert result == [{"group": ["a"], "count": 2}]


def test_apply_spec_traverse_skips_when_field_invalid() -> None:
    rows = [{"items": [1, 2, 3]}]
    spec = ProjectionSpec(
        spec_version=1,
        name="traverse",
        domain="tests",
        pipeline=(ProjectionOp("traverse", {"field": 123}),),
    )
    assert _apply_spec(spec, rows) == rows


def test_apply_spec_erases_semantic_projection_compatibility_at_exec_ingress() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="semantic-compat",
        domain="tests",
        pipeline=(
            ProjectionOp(
                "project",
                {
                    "fields": ["id"],
                    "quotient_face": "projection_fiber.frontier",
                },
            ),
            ProjectionOp(
                "reflect",
                {"surface": "projection_fiber"},
            ),
        ),
    )

    rows = [{"id": 1, "status": "ok"}]
    assert _apply_spec(spec, rows) == [{"id": 1}]
