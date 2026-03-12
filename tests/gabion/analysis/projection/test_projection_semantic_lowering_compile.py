from __future__ import annotations

from gabion.analysis.aspf.aspf_lattice_algebra import frontier_failure_witness
from gabion.analysis.projection.projection_semantic_lowering import (
    lower_projection_spec_to_semantic_plan,
)
from gabion.analysis.projection.projection_semantic_lowering_compile import (
    compile_projection_semantic_lowering_plan,
)
from gabion.analysis.projection.projection_spec import ProjectionOp, ProjectionSpec
from gabion.analysis.projection.semantic_fragment import (
    CanonicalWitnessedSemanticRow,
    ProjectionFiberRequestContext,
    reflect_projection_fiber_witness,
)


def _row() -> CanonicalWitnessedSemanticRow:
    witness = frontier_failure_witness(
        rel_path="demo.py",
        qualname="demo.f",
        line=10,
        column=5,
        node_kind="branch:if",
        reason="missing_exec_mapping",
    )
    context: ProjectionFiberRequestContext = {
        "path": "demo.py",
        "qualname": "demo.f",
        "structural_path": "demo.f::branch[0]::branch:if::x",
        "line": 10,
        "column": 5,
        "node_kind": "branch:if",
        "required_symbols": ("x",),
    }
    return reflect_projection_fiber_witness(context=context, witness=witness)


def _lowered_projection_fiber_frontier_plan():
    spec = ProjectionSpec(
        spec_version=1,
        name="projection_fiber_frontier",
        domain="projection_fiber",
        pipeline=(
            ProjectionOp(
                op="project",
                params={
                    "fields": ["frontier_key", "projection_name", "complete"],
                    "quotient_face": "projection_fiber.frontier",
                },
            ),
            ProjectionOp(op="sort", params={"by": ["frontier_key"]}),
        ),
    )
    return lower_projection_spec_to_semantic_plan(spec)


def _lowered_projection_fiber_reflective_boundary_plan():
    spec = ProjectionSpec(
        spec_version=1,
        name="projection_fiber_reflective_boundary",
        domain="projection_fiber",
        pipeline=(
            ProjectionOp(
                op="project",
                params={
                    "fields": [
                        "frontier_key",
                        "data_anchor_site_identity",
                        "exec_frontier_site_identity",
                        "obligation_state",
                    ],
                    "quotient_face": "projection_fiber.reflective_boundary",
                },
            ),
            ProjectionOp(op="sort", params={"by": ["frontier_key"]}),
        ),
    )
    return lower_projection_spec_to_semantic_plan(spec)


def _lowered_projection_fiber_reflection_plan():
    spec = ProjectionSpec(
        spec_version=1,
        name="projection_fiber_reflection",
        domain="projection_fiber",
        pipeline=(
            ProjectionOp(
                op="reflect",
                params={"surface": "projection_fiber"},
            ),
        ),
    )
    return lower_projection_spec_to_semantic_plan(spec)


def _lowered_projection_fiber_support_reflection_plan():
    spec = ProjectionSpec(
        spec_version=1,
        name="projection_fiber_support_reflection",
        domain="projection_fiber",
        pipeline=(
            ProjectionOp(
                op="support_reflect",
                params={"surface": "projection_fiber"},
            ),
        ),
    )
    return lower_projection_spec_to_semantic_plan(spec)


def test_projection_semantic_lowering_compilation_is_deterministic() -> None:
    row = _row()
    lowering_plan = _lowered_projection_fiber_frontier_plan()

    first = compile_projection_semantic_lowering_plan(lowering_plan, (row,))
    second = compile_projection_semantic_lowering_plan(lowering_plan, (row,))

    assert first == second


def test_projection_semantic_lowering_compilation_preserves_identity_and_trace() -> None:
    row = _row()
    lowering_plan = _lowered_projection_fiber_frontier_plan()

    compiled = compile_projection_semantic_lowering_plan(lowering_plan, (row,))

    assert len(compiled.bindings) == 1
    binding = compiled.bindings[0]
    shacl_plan = compiled.compiled_shacl_plans[0]
    sparql_plan = compiled.compiled_sparql_plans[0]

    assert binding.source_structural_identity == row["structural_identity"]
    assert shacl_plan["source_structural_identity"] == row["structural_identity"]
    assert sparql_plan["source_structural_identity"] == row["structural_identity"]
    assert shacl_plan["semantic_op"] == "quotient_face"
    assert sparql_plan["semantic_op"] == "quotient_face"
    assert shacl_plan["surface"] == "projection_fiber.frontier"
    assert sparql_plan["surface"] == "projection_fiber.frontier"
    assert shacl_plan["witness_trace"] == sparql_plan["witness_trace"]
    assert shacl_plan["constraints"][0]["focus_path"] == "structural_identity"
    assert sparql_plan["select_vars"] == [
        "?frontier_key",
        "?projection_name",
        "?complete",
    ]
    assert binding.shacl_plan_id == shacl_plan["plan_id"]
    assert binding.sparql_plan_id == sparql_plan["plan_id"]


def test_projection_semantic_lowering_compiles_reflective_boundary_face() -> None:
    row = _row()
    lowering_plan = _lowered_projection_fiber_reflective_boundary_plan()

    compiled = compile_projection_semantic_lowering_plan(lowering_plan, (row,))

    assert len(compiled.bindings) == 1
    binding = compiled.bindings[0]
    shacl_plan = compiled.compiled_shacl_plans[0]
    sparql_plan = compiled.compiled_sparql_plans[0]

    assert binding.quotient_face == "projection_fiber.reflective_boundary"
    assert shacl_plan["surface"] == "projection_fiber.reflective_boundary"
    assert sparql_plan["surface"] == "projection_fiber.reflective_boundary"
    assert sparql_plan["select_vars"] == [
        "?frontier_key",
        "?data_anchor_site_identity",
        "?exec_frontier_site_identity",
        "?obligation_state",
    ]


def test_projection_semantic_lowering_compiles_reflection_surface() -> None:
    row = _row()
    lowering_plan = _lowered_projection_fiber_reflection_plan()

    compiled = compile_projection_semantic_lowering_plan(lowering_plan, (row,))

    assert compiled.bindings == ()
    shacl_plan = compiled.compiled_shacl_plans[0]
    sparql_plan = compiled.compiled_sparql_plans[0]

    assert shacl_plan["semantic_op"] == "reflect"
    assert sparql_plan["semantic_op"] == "reflect"
    assert shacl_plan["surface"] == "projection_fiber"
    assert sparql_plan["surface"] == "projection_fiber"
    assert shacl_plan["source_structural_identity"] == row["structural_identity"]
    assert sparql_plan["source_structural_identity"] == row["structural_identity"]


def test_projection_semantic_lowering_compiles_support_reflection_surface() -> None:
    row = _row()
    lowering_plan = _lowered_projection_fiber_support_reflection_plan()

    compiled = compile_projection_semantic_lowering_plan(lowering_plan, (row,))

    assert compiled.bindings == ()
    shacl_plan = compiled.compiled_shacl_plans[0]
    sparql_plan = compiled.compiled_sparql_plans[0]

    assert shacl_plan["semantic_op"] == "support_reflect"
    assert sparql_plan["semantic_op"] == "support_reflect"
    assert shacl_plan["surface"] == "projection_fiber"
    assert sparql_plan["surface"] == "projection_fiber"
    assert shacl_plan["constraints"][0]["focus_path"] == "input_witnesses.kind"
    assert sparql_plan["select_vars"] == [
        "?inputWitnessKinds",
        "?synthesizedWitnessKinds",
        "?boundaryKinds",
    ]
