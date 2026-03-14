from __future__ import annotations

from gabion.analysis.aspf.aspf_lattice_algebra import frontier_failure_witness
from gabion.analysis.projection.semantic_fragment import (
    ProjectionFiberRequestContext,
    _canonical_value_materialization,
    _normalize_value,
    _stable_json_key,
    close_canonical_semantic_row,
    reflect_projection_fiber_witness,
)


def _projection_context(*, line: int = 10, column: int = 5) -> ProjectionFiberRequestContext:
    return {
        "path": "demo.py",
        "qualname": "demo.f",
        "structural_path": "demo.f::branch[0]::branch:if::x",
        "line": line,
        "column": column,
        "node_kind": "branch:if",
        "required_symbols": ("x",),
    }


def _projection_witness(*, line: int, column: int, reason: str):
    return frontier_failure_witness(
        rel_path="demo.py",
        qualname="demo.f",
        line=line,
        column=column,
        node_kind="branch:if",
        reason=reason,
    )


# gabion:evidence E:function_site::semantic_fragment.py::gabion.analysis.projection.semantic_fragment.reflect_projection_fiber_witness
# gabion:behavior primary=desired
def test_reflect_projection_fiber_witness_emits_canonical_row() -> None:
    row = reflect_projection_fiber_witness(
        context=_projection_context(),
        witness=_projection_witness(
            line=10,
            column=5,
            reason="missing_exec_mapping",
        ),
    )

    assert row["row_id"] == row["structural_identity"]
    assert row["surface"] == "projection_fiber"
    assert row["carrier_kind"] == "frontier_witness"
    assert row["payload"]["required_symbols"] == ["x"]
    assert len(row["synthesized_witnesses"]) == 4
    assert row["obligation_state"] == "unresolved"
    assert [item["op"] for item in row["transform_trace"]] == [
        "reflect",
        "synthesize_witness",
    ]


# gabion:evidence E:function_site::semantic_fragment.py::gabion.analysis.projection.semantic_fragment.close_canonical_semantic_row
# gabion:behavior primary=desired facets=edge
def test_close_canonical_semantic_row_sorts_and_deduplicates_nested_objects() -> None:
    row = {
        "row_id": "row",
        "structural_identity": "struct",
        "site_identity": "site",
        "surface": "projection_fiber",
        "carrier_kind": "frontier_witness",
        "payload": {
            "z": [{"b": 2, "a": 1}],
            "a": {"y": 2, "x": 1},
        },
        "input_witnesses": [
            {"kind": "branch_frontier", "b": 2, "a": 1},
            {"a": 1, "kind": "branch_frontier", "b": 2},
        ],
        "synthesized_witnesses": [
            {"kind": "join", "right_ids": ["r"], "left_ids": ["l"], "result_ids": ["o"]},
            {"result_ids": ["o"], "kind": "join", "left_ids": ["l"], "right_ids": ["r"]},
        ],
        "obligations": [
            {"kind": "completeness", "complete": True},
            {"complete": True, "kind": "completeness"},
        ],
        "boundary_trace": [
            {"kind": "boundary_crossing", "boundary_kind": "exec", "crossing_id": "x"},
            {"crossing_id": "x", "kind": "boundary_crossing", "boundary_kind": "exec"},
        ],
        "transform_trace": [
            {
                "op": "reflect",
                "details": {"b": {"y": 2, "x": 1}, "a": [{"d": 4, "c": 3}]},
            }
        ],
        "obligation_state": "erased",
    }

    closed = close_canonical_semantic_row(row)

    assert close_canonical_semantic_row(closed) == closed
    assert list(closed["payload"].keys()) == ["a", "z"]
    assert closed["payload"]["a"] == {"x": 1, "y": 2}
    assert closed["payload"]["z"] == [{"a": 1, "b": 2}]
    assert closed["input_witnesses"] == [{"a": 1, "b": 2, "kind": "branch_frontier"}]
    assert closed["synthesized_witnesses"] == [
        {"kind": "join", "left_ids": ["l"], "result_ids": ["o"], "right_ids": ["r"]}
    ]
    assert closed["obligations"] == [{"complete": True, "kind": "completeness"}]
    assert closed["boundary_trace"] == [
        {"boundary_kind": "exec", "crossing_id": "x", "kind": "boundary_crossing"}
    ]
    assert closed["transform_trace"] == [
        {
            "op": "reflect",
            "details": {"a": [{"c": 3, "d": 4}], "b": {"x": 1, "y": 2}},
        }
    ]


# gabion:evidence E:function_site::semantic_fragment.py::gabion.analysis.projection.semantic_fragment._normalize_value
# gabion:behavior primary=desired facets=edge
def test_normalize_value_recursively_orders_nested_mappings() -> None:
    value = {
        "b": [{"d": 4, "c": 3}],
        "a": {"y": 2, "x": 1},
        "scalar": 7,
    }

    assert _normalize_value(value) == {
        "a": {"x": 1, "y": 2},
        "b": [{"c": 3, "d": 4}],
        "scalar": 7,
    }
    assert _normalize_value(("keep", "tuple")) == ("keep", "tuple")


# gabion:evidence E:function_site::semantic_fragment.py::gabion.analysis.projection.semantic_fragment._canonical_value_materialization
# gabion:behavior primary=desired facets=edge
def test_canonical_value_materialization_preserves_normalized_value_and_stable_key() -> None:
    materialization = _canonical_value_materialization(
        {"b": [{"d": 4, "c": 3}], "a": {"y": 2, "x": 1}}
    )

    assert materialization.normalized_value == {
        "a": {"x": 1, "y": 2},
        "b": [{"c": 3, "d": 4}],
    }
    assert materialization.stable_key == (
        "{a:{x:scalar:1|y:scalar:2}|b:[{c:scalar:3|d:scalar:4}]}"
    )


# gabion:evidence E:function_site::semantic_fragment.py::gabion.analysis.projection.semantic_fragment._stable_json_key
# gabion:behavior primary=desired facets=edge
def test_stable_json_key_is_order_invariant_for_nested_json_shapes() -> None:
    left = {"b": [{"d": 4, "c": 3}], "a": {"y": 2, "x": 1}}
    right = {"a": {"x": 1, "y": 2}, "b": [{"c": 3, "d": 4}]}

    assert _stable_json_key(left) == _stable_json_key(right)
    assert _stable_json_key({"a": 1}) != _stable_json_key({"a": [1]})


def test_projection_fiber_structural_identity_ignores_line_motion_when_structure_is_stable() -> None:
    first = reflect_projection_fiber_witness(
        context=_projection_context(),
        witness=_projection_witness(
            line=10,
            column=5,
            reason="first",
        ),
    )
    second = reflect_projection_fiber_witness(
        context=_projection_context(line=20, column=7),
        witness=_projection_witness(
            line=20,
            column=7,
            reason="second",
        ),
    )

    assert first["structural_identity"] == second["structural_identity"]
    assert first["site_identity"] != second["site_identity"]
