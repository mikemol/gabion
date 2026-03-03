# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from gabion.analysis.json_types import JSONObject, JSONValue

from .ast_context import ancestor_if_names


@dataclass(frozen=True)
class ExceptionObligationDecision:
    status: str
    witness_ref: object
    remainder: dict[str, object]
    environment_ref: JSONValue
    handledness_reason_code: str
    handledness_reason: str
    exception_type_source: JSONValue
    exception_type_candidates: list[str]
    type_refinement_opportunity: str


def _decision_from_handledness(
    handled: Mapping[str, JSONValue],
    *,
    has_handledness: bool,
    kind: str,
    sequence_or_none_fn: Callable[[JSONValue], object],
) -> ExceptionObligationDecision:
    if not has_handledness:
        return ExceptionObligationDecision(
            status="UNKNOWN",
            witness_ref=None,
            remainder={"exception_kind": kind},
            environment_ref=None,
            handledness_reason_code="NO_HANDLER",
            handledness_reason="no handledness witness",
            exception_type_source=None,
            exception_type_candidates=[],
            type_refinement_opportunity="",
        )

    witness_result = str(handled.get("result", ""))
    handledness_reason_code = str(handled.get("handledness_reason_code", "UNKNOWN_REASON"))
    handledness_reason = str(handled.get("handledness_reason", ""))
    exception_type_source = handled.get("exception_type_source")
    raw_candidates = sequence_or_none_fn(handled.get("exception_type_candidates") or [])
    candidates: list[str] = []
    if raw_candidates is not None:
        candidates = [str(value) for value in raw_candidates]
    type_refinement_opportunity = str(handled.get("type_refinement_opportunity", ""))

    if witness_result == "HANDLED":
        remainder: dict[str, object] = {}
        status = "HANDLED"
    else:
        status = "UNKNOWN"
        remainder = {
            "exception_kind": kind,
            "handledness_result": witness_result or "UNKNOWN",
            "type_compatibility": str(handled.get("type_compatibility", "unknown")),
            "handledness_reason_code": handledness_reason_code,
            "handledness_reason": handledness_reason,
        }
        if exception_type_source:
            remainder["exception_type_source"] = exception_type_source
        if candidates:
            remainder["exception_type_candidates"] = candidates
        if type_refinement_opportunity:
            remainder["type_refinement_opportunity"] = type_refinement_opportunity

    return ExceptionObligationDecision(
        status=status,
        witness_ref=handled.get("handledness_id"),
        remainder=remainder,
        environment_ref=handled.get("environment") or {},
        handledness_reason_code=handledness_reason_code,
        handledness_reason=handledness_reason,
        exception_type_source=exception_type_source,
        exception_type_candidates=candidates,
        type_refinement_opportunity=type_refinement_opportunity,
    )


def _apply_deadness(
    decision: ExceptionObligationDecision,
    *,
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    env_entries: Mapping[str, tuple[JSONValue, JSONObject]],
    branch_reachability_under_env_fn: Callable[..., object],
    is_reachability_false_fn: Callable[..., bool],
    names_in_expr_fn: Callable[..., set[str]],
    sort_once_fn: Callable[..., object],
    order_policy_sort,
    order_policy_enforce,
    check_deadline_fn: Callable[[], None],
) -> ExceptionObligationDecision:
    if decision.status == "HANDLED" or not env_entries:
        return decision

    env = {name: value for name, (value, _) in env_entries.items()}
    reachability = branch_reachability_under_env_fn(node, parents, env)
    if not is_reachability_false_fn(reachability):
        return decision

    names = ancestor_if_names(
        node,
        parents=parents,
        names_in_expr_fn=names_in_expr_fn,
        check_deadline_fn=check_deadline_fn,
    )
    ordered_names = sort_once_fn(
        names,
        source="indexed_scan.obligation_decision.apply_deadness.names",
        policy=order_policy_sort,
    )
    for name in sort_once_fn(
        ordered_names,
        source="indexed_scan.obligation_decision.apply_deadness.names.enforce",
        policy=order_policy_enforce,
    ):
        check_deadline_fn()
        if name in env_entries:
            _, witness = env_entries[name]
            return ExceptionObligationDecision(
                status="DEAD",
                witness_ref=witness.get("deadness_id"),
                remainder={},
                environment_ref=witness.get("environment") or {},
                handledness_reason_code=decision.handledness_reason_code,
                handledness_reason=decision.handledness_reason,
                exception_type_source=decision.exception_type_source,
                exception_type_candidates=decision.exception_type_candidates,
                type_refinement_opportunity=decision.type_refinement_opportunity,
            )

    return decision


def decide_exception_obligation(
    *,
    kind: str,
    handled: Mapping[str, JSONValue],
    has_handledness: bool,
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    env_entries: Mapping[str, tuple[JSONValue, JSONObject]],
    sequence_or_none_fn: Callable[[JSONValue], object],
    branch_reachability_under_env_fn: Callable[..., object],
    is_reachability_false_fn: Callable[..., bool],
    names_in_expr_fn: Callable[..., set[str]],
    sort_once_fn: Callable[..., object],
    order_policy_sort,
    order_policy_enforce,
    check_deadline_fn: Callable[[], None],
) -> ExceptionObligationDecision:
    decision = _decision_from_handledness(
        handled,
        has_handledness=has_handledness,
        kind=kind,
        sequence_or_none_fn=sequence_or_none_fn,
    )
    return _apply_deadness(
        decision,
        node=node,
        parents=parents,
        env_entries=env_entries,
        branch_reachability_under_env_fn=branch_reachability_under_env_fn,
        is_reachability_false_fn=is_reachability_false_fn,
        names_in_expr_fn=names_in_expr_fn,
        sort_once_fn=sort_once_fn,
        order_policy_sort=order_policy_sort,
        order_policy_enforce=order_policy_enforce,
        check_deadline_fn=check_deadline_fn,
    )


__all__ = ["ExceptionObligationDecision", "decide_exception_obligation"]
