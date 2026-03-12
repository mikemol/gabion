from __future__ import annotations

from gabion.analysis.aspf.aspf_lattice_algebra import frontier_failure_witness
from gabion.analysis.projection.semantic_fragment import (
    CanonicalWitnessedSemanticRow,
    ProjectionFiberRequestContext,
    reflect_projection_fiber_witness,
)
from gabion.analysis.projection.semantic_fragment_compile import (
    compile_projection_fiber_reflect_to_shacl,
    compile_projection_fiber_reflect_to_sparql,
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


def test_projection_fiber_reflect_compilers_are_deterministic() -> None:
    row = _row()

    first_shacl = compile_projection_fiber_reflect_to_shacl(row)
    second_shacl = compile_projection_fiber_reflect_to_shacl(row)
    first_sparql = compile_projection_fiber_reflect_to_sparql(row)
    second_sparql = compile_projection_fiber_reflect_to_sparql(row)

    assert first_shacl == second_shacl
    assert first_sparql == second_sparql


def test_projection_fiber_reflect_compilers_preserve_carrier_identity_and_trace() -> None:
    row = _row()

    shacl_plan = compile_projection_fiber_reflect_to_shacl(row)
    sparql_plan = compile_projection_fiber_reflect_to_sparql(row)

    assert shacl_plan["source_structural_identity"] == row["structural_identity"]
    assert sparql_plan["source_structural_identity"] == row["structural_identity"]
    assert shacl_plan["semantic_op"] == "reflect"
    assert sparql_plan["semantic_op"] == "reflect"
    assert shacl_plan["constraints"][0]["focus_path"] == "payload.structural_path"
    assert sparql_plan["where_patterns"][0]["predicate"] == "gabion:siteIdentity"
    assert shacl_plan["witness_trace"] == sparql_plan["witness_trace"]
