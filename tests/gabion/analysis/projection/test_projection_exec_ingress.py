from __future__ import annotations

from gabion.analysis.projection.projection_exec_protocol import (
    CountByExecutionOp,
    LimitExecutionOp,
    ProjectExecutionOp,
    SelectExecutionOp,
    SortKey,
    SortExecutionOp,
)
from gabion.analysis.projection.projection_exec_ingress import execution_ops_from_spec
from gabion.analysis.projection.projection_spec import ProjectionOp, ProjectionSpec


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
