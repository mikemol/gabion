from __future__ import annotations

import ast

from gabion.analysis.indexed_scan.reachability import decide_never_reachability


def _check_deadline() -> None:
    return None


def _names_in_expr(expr: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(expr) if isinstance(node, ast.Name)}


def _sort_once(values, **_kwargs):
    return sorted(values)


#
# gabion:evidence E:function_site::indexed_scan/reachability.py::gabion.analysis.indexed_scan.reachability.decide_never_reachability
def test_decide_never_reachability_covers_no_env_true_false_and_undecidable() -> None:
    module = ast.parse("if a and b:\n    never('x')\n")
    if_node = module.body[0]
    assert isinstance(if_node, ast.If)
    call_node = if_node.body[0].value
    assert isinstance(call_node, ast.Call)
    parents = {call_node: if_node}

    no_env = decide_never_reachability(
        call_node,
        parents=parents,
        env_entries={},
        branch_reachability_under_env_fn=lambda _node, _parents, _env: "unknown",
        is_reachability_false_fn=lambda value: value == "false",
        is_reachability_true_fn=lambda value: value == "true",
        names_in_expr_fn=_names_in_expr,
        sort_once_fn=_sort_once,
        order_policy_sort="sort",
        order_policy_enforce="enforce",
        check_deadline_fn=_check_deadline,
    )
    assert no_env.status == "OBLIGATION"

    violation = decide_never_reachability(
        call_node,
        parents=parents,
        env_entries={"a": (1, {"deadness_id": "d", "environment": {"a": "1"}})},
        branch_reachability_under_env_fn=lambda _node, _parents, _env: "true",
        is_reachability_false_fn=lambda value: value == "false",
        is_reachability_true_fn=lambda value: value == "true",
        names_in_expr_fn=_names_in_expr,
        sort_once_fn=_sort_once,
        order_policy_sort="sort",
        order_policy_enforce="enforce",
        check_deadline_fn=_check_deadline,
    )
    assert violation.status == "VIOLATION"
    assert violation.environment_ref == {"a": 1}

    unreachable = decide_never_reachability(
        call_node,
        parents=parents,
        env_entries={"a": (0, {"deadness_id": "dead:1", "environment": {"a": "0"}})},
        branch_reachability_under_env_fn=lambda _node, _parents, _env: "false",
        is_reachability_false_fn=lambda value: value == "false",
        is_reachability_true_fn=lambda value: value == "true",
        names_in_expr_fn=_names_in_expr,
        sort_once_fn=_sort_once,
        order_policy_sort="sort",
        order_policy_enforce="enforce",
        check_deadline_fn=_check_deadline,
    )
    assert unreachable.status == "PROVEN_UNREACHABLE"
    assert unreachable.witness_ref == "dead:1"

    undecidable = decide_never_reachability(
        call_node,
        parents=parents,
        env_entries={"a": (1, {"deadness_id": "d", "environment": {"a": "1"}})},
        branch_reachability_under_env_fn=lambda _node, _parents, _env: "unknown",
        is_reachability_false_fn=lambda value: value == "false",
        is_reachability_true_fn=lambda value: value == "true",
        names_in_expr_fn=_names_in_expr,
        sort_once_fn=_sort_once,
        order_policy_sort="sort",
        order_policy_enforce="enforce",
        check_deadline_fn=_check_deadline,
    )
    assert undecidable.status == "OBLIGATION"
    assert undecidable.undecidable_reason == "depends on params: b"
