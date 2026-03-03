from __future__ import annotations

import ast

from gabion.analysis.indexed_scan.handledness_decision import decide_handledness
from gabion.analysis.indexed_scan.obligation_decision import decide_exception_obligation


def _check_deadline() -> None:
    return None


def _handler_label(handler: ast.ExceptHandler) -> str:
    if handler.type is None:
        return "except:"
    return f"except {ast.unparse(handler.type)}"


def _handler_type_names(handler_type: object) -> tuple[str, ...]:
    if handler_type is None:
        return ()
    if isinstance(handler_type, ast.Tuple):
        out: list[str] = []
        for elt in handler_type.elts:
            if isinstance(elt, ast.Name):
                out.append(elt.id)
        return tuple(out)
    if isinstance(handler_type, ast.Name):
        return (handler_type.id,)
    return ()


def _compatibility(exception_name: object, handler_type: object) -> str:
    if handler_type is None:
        return "compatible"
    if isinstance(handler_type, ast.Name) and exception_name == handler_type.id:
        return "compatible"
    if isinstance(handler_type, ast.Name) and handler_type.id == "Unknown":
        return "unknown"
    return "incompatible"


def _sort_once(values, **_kwargs):
    return sorted(values)


def _names_in_expr(expr: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(expr) if isinstance(node, ast.Name)}


#
# gabion:evidence E:function_site::indexed_scan/handledness_decision.py::gabion.analysis.indexed_scan.handledness_decision.decide_handledness
def test_handledness_decision_paths_cover_catch_unknown_mismatch_and_convert() -> None:
    try_tree = ast.parse("try:\n    x()\nexcept ValueError:\n    pass\n").body[0]
    assert isinstance(try_tree, ast.Try)
    decision = decide_handledness(
        try_tree,
        exception_name="ValueError",
        exception_type_candidates=(),
        exception_handler_compatibility_fn=_compatibility,
        handler_label_fn=_handler_label,
        handler_type_names_fn=_handler_type_names,
        check_deadline_fn=_check_deadline,
    )
    assert decision.result == "HANDLED"
    assert decision.handledness_reason_code == "TYPED_MATCH"

    unknown_tree = ast.parse("try:\n    x()\nexcept Unknown:\n    pass\n").body[0]
    assert isinstance(unknown_tree, ast.Try)
    unknown = decide_handledness(
        unknown_tree,
        exception_name="ValueError",
        exception_type_candidates=("ValueError", "TypeError"),
        exception_handler_compatibility_fn=_compatibility,
        handler_label_fn=_handler_label,
        handler_type_names_fn=_handler_type_names,
        check_deadline_fn=_check_deadline,
    )
    assert unknown.result == "UNKNOWN"
    assert unknown.handledness_reason_code == "TYPE_UNRESOLVED"
    assert "narrow raised exception type" in unknown.type_refinement_opportunity

    mismatch_tree = ast.parse("try:\n    x()\nexcept KeyError:\n    pass\n").body[0]
    assert isinstance(mismatch_tree, ast.Try)
    mismatch = decide_handledness(
        mismatch_tree,
        exception_name="ValueError",
        exception_type_candidates=(),
        exception_handler_compatibility_fn=_compatibility,
        handler_label_fn=_handler_label,
        handler_type_names_fn=_handler_type_names,
        check_deadline_fn=_check_deadline,
    )
    assert mismatch.result == "UNKNOWN"
    assert mismatch.handledness_reason_code == "TYPED_MISMATCH"

    convert = decide_handledness(
        None,
        exception_name="SystemExit",
        exception_type_candidates=(),
        exception_handler_compatibility_fn=_compatibility,
        handler_label_fn=_handler_label,
        handler_type_names_fn=_handler_type_names,
        check_deadline_fn=_check_deadline,
    )
    assert convert.result == "HANDLED"
    assert convert.handler_kind == "convert"


#
# gabion:evidence E:function_site::indexed_scan/obligation_decision.py::gabion.analysis.indexed_scan.obligation_decision.decide_exception_obligation
def test_obligation_decision_paths_cover_unknown_handled_and_dead() -> None:
    tree = ast.parse("if flag:\n    raise ValueError()\n")
    if_node = tree.body[0]
    assert isinstance(if_node, ast.If)
    raise_node = if_node.body[0]
    assert isinstance(raise_node, ast.Raise)
    parents = {raise_node: if_node}

    unknown = decide_exception_obligation(
        kind="raise",
        handled={},
        has_handledness=False,
        node=raise_node,
        parents=parents,
        env_entries={},
        sequence_or_none_fn=lambda value: value if isinstance(value, list) else None,
        branch_reachability_under_env_fn=lambda _node, _parents, _env: "unknown",
        is_reachability_false_fn=lambda value: value == "false",
        names_in_expr_fn=_names_in_expr,
        sort_once_fn=_sort_once,
        order_policy_sort="sort",
        order_policy_enforce="enforce",
        check_deadline_fn=_check_deadline,
    )
    assert unknown.status == "UNKNOWN"
    assert unknown.remainder["exception_kind"] == "raise"

    handled = decide_exception_obligation(
        kind="raise",
        handled={
            "result": "HANDLED",
            "handledness_id": "handled:1",
            "environment": {"flag": 1},
        },
        has_handledness=True,
        node=raise_node,
        parents=parents,
        env_entries={},
        sequence_or_none_fn=lambda value: value if isinstance(value, list) else None,
        branch_reachability_under_env_fn=lambda _node, _parents, _env: "unknown",
        is_reachability_false_fn=lambda value: value == "false",
        names_in_expr_fn=_names_in_expr,
        sort_once_fn=_sort_once,
        order_policy_sort="sort",
        order_policy_enforce="enforce",
        check_deadline_fn=_check_deadline,
    )
    assert handled.status == "HANDLED"
    assert handled.remainder == {}

    dead = decide_exception_obligation(
        kind="raise",
        handled={},
        has_handledness=False,
        node=raise_node,
        parents=parents,
        env_entries={"flag": (0, {"deadness_id": "dead:1", "environment": {"flag": "0"}})},
        sequence_or_none_fn=lambda value: value if isinstance(value, list) else None,
        branch_reachability_under_env_fn=lambda _node, _parents, _env: "false",
        is_reachability_false_fn=lambda value: value == "false",
        names_in_expr_fn=_names_in_expr,
        sort_once_fn=_sort_once,
        order_policy_sort="sort",
        order_policy_enforce="enforce",
        check_deadline_fn=_check_deadline,
    )
    assert dead.status == "DEAD"
    assert dead.witness_ref == "dead:1"
