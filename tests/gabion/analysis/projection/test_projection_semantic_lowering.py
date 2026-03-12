from __future__ import annotations

from gabion.analysis.projection.projection_semantic_lowering import (
    BridgeProjectionKind,
    SemanticProjectionKind,
    lower_projection_spec_to_semantic_plan,
)
from gabion.analysis.projection.projection_spec import ProjectionOp, ProjectionSpec


def test_lower_projection_spec_classifies_rfc_layers() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="demo",
        domain="tests",
        pipeline=(
            ProjectionOp(op="select", params={"predicate": "stable"}),
            ProjectionOp(op="project", params={"fields": ["id", "status"]}),
            ProjectionOp(op="count_by", params={"fields": ["status"]}),
            ProjectionOp(op="traverse", params={"field": "items"}),
            ProjectionOp(op="sort", params={"by": ["status"]}),
            ProjectionOp(op="limit", params={"count": 3}),
        ),
    )

    lowered = lower_projection_spec_to_semantic_plan(spec)

    assert lowered.semantic_ops == ()
    assert tuple(op.source_op for op in lowered.presentation_ops) == (
        "project",
        "count_by",
        "sort",
        "limit",
    )
    assert tuple(op.bridge_kind for op in lowered.bridge_ops) == (
        BridgeProjectionKind.PREDICATE_FILTER,
        BridgeProjectionKind.TRAVERSE,
    )


def test_lower_projection_spec_promotes_declared_quotient_face() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="demo",
        domain="projection_fiber",
        pipeline=(
            ProjectionOp(
                op="project",
                params={
                    "fields": ["frontier_key", "projection_name"],
                    "quotient_face": "projection_fiber.frontier",
                },
            ),
            ProjectionOp(op="sort", params={"by": ["frontier_key"]}),
        ),
    )

    lowered = lower_projection_spec_to_semantic_plan(spec)

    assert len(lowered.semantic_ops) == 1
    semantic_op = lowered.semantic_ops[0]
    assert semantic_op.semantic_op is SemanticProjectionKind.QUOTIENT_FACE
    assert semantic_op.params["quotient_face"] == "projection_fiber.frontier"
    assert lowered.presentation_ops[0].source_op == "sort"


def test_lower_projection_spec_promotes_reflective_boundary_face() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="demo",
        domain="projection_fiber",
        pipeline=(
            ProjectionOp(
                op="project",
                params={
                    "fields": [
                        "frontier_key",
                        "data_anchor_site_identity",
                        "exec_frontier_site_identity",
                    ],
                    "quotient_face": "projection_fiber.reflective_boundary",
                },
            ),
            ProjectionOp(op="sort", params={"by": ["frontier_key"]}),
        ),
    )

    lowered = lower_projection_spec_to_semantic_plan(spec)

    assert len(lowered.semantic_ops) == 1
    semantic_op = lowered.semantic_ops[0]
    assert semantic_op.semantic_op is SemanticProjectionKind.QUOTIENT_FACE
    assert semantic_op.params["quotient_face"] == "projection_fiber.reflective_boundary"
    assert lowered.presentation_ops[0].source_op == "sort"


def test_lower_projection_spec_promotes_reflect_surface() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="demo",
        domain="projection_fiber",
        pipeline=(
            ProjectionOp(
                op="reflect",
                params={"surface": "projection_fiber"},
            ),
        ),
    )

    lowered = lower_projection_spec_to_semantic_plan(spec)

    assert len(lowered.semantic_ops) == 1
    semantic_op = lowered.semantic_ops[0]
    assert semantic_op.semantic_op is SemanticProjectionKind.REFLECT
    assert semantic_op.params["surface"] == "projection_fiber"
    assert lowered.presentation_ops == ()
    assert lowered.bridge_ops == ()


def test_lower_projection_spec_promotes_support_reflect_surface() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="demo",
        domain="projection_fiber",
        pipeline=(
            ProjectionOp(
                op="support_reflect",
                params={"surface": "projection_fiber"},
            ),
        ),
    )

    lowered = lower_projection_spec_to_semantic_plan(spec)

    assert len(lowered.semantic_ops) == 1
    semantic_op = lowered.semantic_ops[0]
    assert semantic_op.semantic_op is SemanticProjectionKind.SUPPORT_REFLECT
    assert semantic_op.params["surface"] == "projection_fiber"
    assert lowered.presentation_ops == ()
    assert lowered.bridge_ops == ()


def test_lower_projection_spec_promotes_synthesize_witness_surface() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="demo",
        domain="projection_fiber",
        pipeline=(
            ProjectionOp(
                op="synthesize_witness",
                params={"surface": "projection_fiber"},
            ),
        ),
    )

    lowered = lower_projection_spec_to_semantic_plan(spec)

    assert len(lowered.semantic_ops) == 1
    semantic_op = lowered.semantic_ops[0]
    assert semantic_op.semantic_op is SemanticProjectionKind.SYNTHESIZE_WITNESS
    assert semantic_op.params["surface"] == "projection_fiber"
    assert lowered.presentation_ops == ()
    assert lowered.bridge_ops == ()


def test_lower_projection_spec_promotes_wedge_surface() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="demo",
        domain="projection_fiber",
        pipeline=(
            ProjectionOp(
                op="wedge",
                params={"surface": "projection_fiber"},
            ),
        ),
    )

    lowered = lower_projection_spec_to_semantic_plan(spec)

    assert len(lowered.semantic_ops) == 1
    semantic_op = lowered.semantic_ops[0]
    assert semantic_op.semantic_op is SemanticProjectionKind.WEDGE
    assert semantic_op.params["surface"] == "projection_fiber"
    assert lowered.presentation_ops == ()
    assert lowered.bridge_ops == ()


def test_lower_projection_spec_promotes_reindex_surface() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="demo",
        domain="projection_fiber",
        pipeline=(
            ProjectionOp(
                op="reindex",
                params={"surface": "projection_fiber"},
            ),
        ),
    )

    lowered = lower_projection_spec_to_semantic_plan(spec)

    assert len(lowered.semantic_ops) == 1
    semantic_op = lowered.semantic_ops[0]
    assert semantic_op.semantic_op is SemanticProjectionKind.REINDEX
    assert semantic_op.params["surface"] == "projection_fiber"
    assert lowered.presentation_ops == ()
    assert lowered.bridge_ops == ()


def test_lower_projection_spec_promotes_existential_image_surface() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="demo",
        domain="projection_fiber",
        pipeline=(
            ProjectionOp(
                op="existential_image",
                params={"surface": "projection_fiber"},
            ),
        ),
    )

    lowered = lower_projection_spec_to_semantic_plan(spec)

    assert len(lowered.semantic_ops) == 1
    semantic_op = lowered.semantic_ops[0]
    assert semantic_op.semantic_op is SemanticProjectionKind.EXISTENTIAL_IMAGE
    assert semantic_op.params["surface"] == "projection_fiber"
    assert lowered.presentation_ops == ()
    assert lowered.bridge_ops == ()


def test_lower_projection_spec_promotes_negate_surface() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="demo",
        domain="projection_fiber",
        pipeline=(
            ProjectionOp(
                op="negate",
                params={"surface": "projection_fiber"},
            ),
        ),
    )

    lowered = lower_projection_spec_to_semantic_plan(spec)

    assert len(lowered.semantic_ops) == 1
    semantic_op = lowered.semantic_ops[0]
    assert semantic_op.semantic_op is SemanticProjectionKind.NEGATE
    assert semantic_op.params["surface"] == "projection_fiber"
    assert lowered.presentation_ops == ()
    assert lowered.bridge_ops == ()


def test_project_quotient_face_metadata_is_lowered() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="demo",
        domain="tests",
        pipeline=(
            ProjectionOp(
                op="project",
                params={
                    "fields": ["id"],
                    "quotient_face": "projection_fiber.frontier",
                },
            ),
        ),
    )

    lowered = lower_projection_spec_to_semantic_plan(spec)
    assert lowered.semantic_ops[0].params["quotient_face"] == "projection_fiber.frontier"


def test_reflect_semantic_metadata_is_lowered() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="demo",
        domain="tests",
        pipeline=(
            ProjectionOp(
                op="reflect",
                params={"surface": "projection_fiber"},
            ),
        ),
    )

    lowered = lower_projection_spec_to_semantic_plan(spec)
    assert lowered.semantic_ops[0].params["surface"] == "projection_fiber"
