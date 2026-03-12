from __future__ import annotations

from gabion.analysis.aspf.aspf_lattice_algebra import frontier_failure_witness
from gabion.analysis.projection.semantic_fragment import (
    ProjectionFiberRequestContext,
    close_canonical_semantic_row,
    reflect_projection_fiber_witness,
)


def test_projection_fiber_reflection_is_idempotent_and_bounded() -> None:
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

    first = reflect_projection_fiber_witness(context=context, witness=witness)
    second = close_canonical_semantic_row(first)

    assert second == first
    assert len(first["synthesized_witnesses"]) == 4
    assert len(second["synthesized_witnesses"]) == 4
    assert first["obligation_state"] == "unresolved"


def test_projection_fiber_structural_identity_ignores_line_motion_when_structure_is_stable() -> None:
    context: ProjectionFiberRequestContext = {
        "path": "demo.py",
        "qualname": "demo.f",
        "structural_path": "demo.f::branch[0]::branch:if::x",
        "line": 10,
        "column": 5,
        "node_kind": "branch:if",
        "required_symbols": ("x",),
    }
    moved_context: ProjectionFiberRequestContext = {
        "path": "demo.py",
        "qualname": "demo.f",
        "structural_path": "demo.f::branch[0]::branch:if::x",
        "line": 20,
        "column": 7,
        "node_kind": "branch:if",
        "required_symbols": ("x",),
    }
    first = reflect_projection_fiber_witness(
        context=context,
        witness=frontier_failure_witness(
            rel_path="demo.py",
            qualname="demo.f",
            line=10,
            column=5,
            node_kind="branch:if",
            reason="first",
        ),
    )
    second = reflect_projection_fiber_witness(
        context=moved_context,
        witness=frontier_failure_witness(
            rel_path="demo.py",
            qualname="demo.f",
            line=20,
            column=7,
            node_kind="branch:if",
            reason="second",
        ),
    )

    assert first["structural_identity"] == second["structural_identity"]
    assert first["site_identity"] != second["site_identity"]
