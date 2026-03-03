# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from gabion.analysis.json_types import JSONObject, JSONValue

from .ast_context import ancestor_if_names


@dataclass(frozen=True)
class NeverReachabilityDecision:
    status: str
    witness_ref: object
    environment_ref: JSONValue
    undecidable_reason: object


def _select_dead_witness(
    *,
    names: set[str],
    env_entries: Mapping[str, tuple[JSONValue, JSONObject]],
    sort_once_fn: Callable[..., object],
    order_policy_sort,
    order_policy_enforce,
    check_deadline_fn: Callable[[], None],
) -> tuple[object, JSONValue]:
    for name in sort_once_fn(
        sort_once_fn(
            names,
            source="indexed_scan.reachability.select_dead_witness.names",
            policy=order_policy_sort,
        ),
        source="indexed_scan.reachability.select_dead_witness.names.enforce",
        policy=order_policy_enforce,
    ):
        check_deadline_fn()
        if name in env_entries:
            _, witness = env_entries[name]
            return witness.get("deadness_id"), witness.get("environment") or {}
    return None, None


def decide_never_reachability(
    node: ast.AST,
    *,
    parents: dict[ast.AST, ast.AST],
    env_entries: Mapping[str, tuple[JSONValue, JSONObject]],
    branch_reachability_under_env_fn: Callable[..., object],
    is_reachability_false_fn: Callable[..., bool],
    is_reachability_true_fn: Callable[..., bool],
    names_in_expr_fn: Callable[..., set[str]],
    sort_once_fn: Callable[..., object],
    order_policy_sort,
    order_policy_enforce,
    check_deadline_fn: Callable[[], None],
) -> NeverReachabilityDecision:
    if not env_entries:
        return NeverReachabilityDecision(
            status="OBLIGATION",
            witness_ref=None,
            environment_ref=None,
            undecidable_reason=None,
        )

    env = {name: value for name, (value, _) in env_entries.items()}
    reachability = branch_reachability_under_env_fn(node, parents, env)
    if is_reachability_false_fn(reachability):
        names = ancestor_if_names(
            node,
            parents=parents,
            names_in_expr_fn=names_in_expr_fn,
            check_deadline_fn=check_deadline_fn,
        )
        witness_ref, environment_ref = _select_dead_witness(
            names=names,
            env_entries=env_entries,
            sort_once_fn=sort_once_fn,
            order_policy_sort=order_policy_sort,
            order_policy_enforce=order_policy_enforce,
            check_deadline_fn=check_deadline_fn,
        )
        return NeverReachabilityDecision(
            status="PROVEN_UNREACHABLE",
            witness_ref=witness_ref,
            environment_ref=environment_ref if environment_ref else env,
            undecidable_reason=None,
        )
    if is_reachability_true_fn(reachability):
        return NeverReachabilityDecision(
            status="VIOLATION",
            witness_ref=None,
            environment_ref=env,
            undecidable_reason=None,
        )

    names = ancestor_if_names(
        node,
        parents=parents,
        names_in_expr_fn=names_in_expr_fn,
        check_deadline_fn=check_deadline_fn,
    )
    undecidable_params = sort_once_fn(
        (name for name in names if name not in env_entries),
        source="indexed_scan.reachability.decide_never_reachability.undecidable_params",
        policy=order_policy_sort,
    )
    undecidable_reason = None
    if undecidable_params:
        undecidable_reason = f"depends on params: {', '.join(undecidable_params)}"

    return NeverReachabilityDecision(
        status="OBLIGATION",
        witness_ref=None,
        environment_ref=None,
        undecidable_reason=undecidable_reason,
    )


__all__ = ["NeverReachabilityDecision", "decide_never_reachability"]
