# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from gabion.analysis.foundation.json_types import JSONValue


class EvalDecision(StrEnum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BoolEvalOutcome:
    decision: EvalDecision

    def is_unknown(self) -> bool:
        return self.decision is EvalDecision.UNKNOWN

    def as_bool(self) -> bool:
        return self.decision is EvalDecision.TRUE


@dataclass(frozen=True)
class ValueEvalOutcome:
    decision: EvalDecision
    value: JSONValue

    def is_unknown(self) -> bool:
        return self.decision is EvalDecision.UNKNOWN


def _bool_outcome(value: bool) -> BoolEvalOutcome:
    return BoolEvalOutcome(EvalDecision.TRUE if value else EvalDecision.FALSE)


def _unknown_bool_outcome() -> BoolEvalOutcome:
    return BoolEvalOutcome(EvalDecision.UNKNOWN)


def _known_value_outcome(value: JSONValue) -> ValueEvalOutcome:
    return ValueEvalOutcome(EvalDecision.TRUE, value)


def _unknown_value_outcome() -> ValueEvalOutcome:
    return ValueEvalOutcome(EvalDecision.UNKNOWN, False)


def _is_numeric_value(value: JSONValue) -> bool:
    return type(value) in {int, float}


def eval_value_expr(
    expr: ast.AST,
    env: Mapping[str, JSONValue],
    *,
    check_deadline_fn: Callable[[], None],
) -> ValueEvalOutcome:
    check_deadline_fn()
    match expr:
        case ast.Constant(value=value):
            if value is None or type(value) in {str, int, float, bool}:
                return _known_value_outcome(cast(JSONValue, value))
            return _unknown_value_outcome()
        case ast.Name(id=name):
            if name in env:
                return _known_value_outcome(env[name])
            return _unknown_value_outcome()
        case ast.UnaryOp(op=op, operand=operand):
            if type(op) not in {ast.USub, ast.UAdd}:
                return _unknown_value_outcome()
            value_outcome = eval_value_expr(
                operand,
                env,
                check_deadline_fn=check_deadline_fn,
            )
            if value_outcome.is_unknown() or not _is_numeric_value(value_outcome.value):
                return _unknown_value_outcome()
            if type(op) is ast.USub:
                return _known_value_outcome(-value_outcome.value)
            return _known_value_outcome(value_outcome.value)
        case _:
            return _unknown_value_outcome()


def _eval_boolop_values(
    values: Sequence[ast.expr],
    env: Mapping[str, JSONValue],
    *,
    check_deadline_fn: Callable[[], None],
    stop_on_decision: EvalDecision,
) -> BoolEvalOutcome:
    any_unknown = False
    for value in values:
        check_deadline_fn()
        outcome = eval_bool_expr(value, env, check_deadline_fn=check_deadline_fn)
        if outcome.decision is stop_on_decision:
            return outcome
        if outcome.is_unknown():
            any_unknown = True
    if any_unknown:
        return _unknown_bool_outcome()
    return _bool_outcome(stop_on_decision is EvalDecision.FALSE)


def _eval_compare(
    expr: ast.Compare,
    env: Mapping[str, JSONValue],
    *,
    check_deadline_fn: Callable[[], None],
) -> BoolEvalOutcome:
    if len(expr.ops) != 1 or len(expr.comparators) != 1:
        return _unknown_bool_outcome()
    left = eval_value_expr(expr.left, env, check_deadline_fn=check_deadline_fn)
    right = eval_value_expr(expr.comparators[0], env, check_deadline_fn=check_deadline_fn)
    if left.is_unknown() or right.is_unknown():
        return _unknown_bool_outcome()
    left_value = left.value
    right_value = right.value
    match type(expr.ops[0]):
        case ast.Eq:
            return _bool_outcome(left_value == right_value)
        case ast.NotEq:
            return _bool_outcome(left_value != right_value)
        case ast.Lt:
            if _is_numeric_value(left_value) and _is_numeric_value(right_value):
                return _bool_outcome(left_value < right_value)
        case ast.LtE:
            if _is_numeric_value(left_value) and _is_numeric_value(right_value):
                return _bool_outcome(left_value <= right_value)
        case ast.Gt:
            if _is_numeric_value(left_value) and _is_numeric_value(right_value):
                return _bool_outcome(left_value > right_value)
        case ast.GtE:
            if _is_numeric_value(left_value) and _is_numeric_value(right_value):
                return _bool_outcome(left_value >= right_value)
        case _:
            pass
    return _unknown_bool_outcome()


def eval_bool_expr(
    expr: ast.AST,
    env: Mapping[str, JSONValue],
    *,
    check_deadline_fn: Callable[[], None],
) -> BoolEvalOutcome:
    check_deadline_fn()
    match expr:
        case ast.Constant(value=value):
            return _bool_outcome(bool(value))
        case ast.Name(id=name):
            if name in env:
                return _bool_outcome(bool(env[name]))
            return _unknown_bool_outcome()
        case ast.UnaryOp(op=ast.Not(), operand=operand):
            inner = eval_bool_expr(operand, env, check_deadline_fn=check_deadline_fn)
            if inner.is_unknown():
                return _unknown_bool_outcome()
            return _bool_outcome(not inner.as_bool())
        case ast.BoolOp(op=ast.And(), values=values):
            return _eval_boolop_values(
                values,
                env,
                check_deadline_fn=check_deadline_fn,
                stop_on_decision=EvalDecision.FALSE,
            )
        case ast.BoolOp(op=ast.Or(), values=values):
            return _eval_boolop_values(
                values,
                env,
                check_deadline_fn=check_deadline_fn,
                stop_on_decision=EvalDecision.TRUE,
            )
        case ast.Compare() as compare_expr:
            return _eval_compare(compare_expr, env, check_deadline_fn=check_deadline_fn)
        case _:
            return _unknown_bool_outcome()


def branch_reachability_under_env(
    node: ast.AST,
    parents: Mapping[ast.AST, ast.AST],
    env: Mapping[str, JSONValue],
    *,
    check_deadline_fn: Callable[[], None],
    node_in_block_fn: Callable[[ast.AST, list[ast.stmt]], bool],
) -> EvalDecision:
    check_deadline_fn()
    constraints: list[tuple[ast.AST, bool]] = []
    current_node: ast.AST = node
    current = parents.get(current_node)
    while current is not None:
        check_deadline_fn()
        if type(current) is ast.If:
            if_node = cast(ast.If, current)
            if node_in_block_fn(current_node, if_node.body):
                constraints.append((if_node.test, True))
            elif node_in_block_fn(current_node, if_node.orelse):
                constraints.append((if_node.test, False))
        current_node = current
        current = parents.get(current_node)

    if not constraints:
        return EvalDecision.UNKNOWN

    any_unknown = False
    for test, want_true in constraints:
        check_deadline_fn()
        outcome = eval_bool_expr(test, env, check_deadline_fn=check_deadline_fn)
        if outcome.is_unknown():
            any_unknown = True
            continue
        if outcome.as_bool() != want_true:
            return EvalDecision.FALSE
    return EvalDecision.UNKNOWN if any_unknown else EvalDecision.TRUE


def is_reachability_false(reachability: EvalDecision) -> bool:
    return reachability is EvalDecision.FALSE


def is_reachability_true(reachability: EvalDecision) -> bool:
    return reachability is EvalDecision.TRUE


__all__ = [
    "BoolEvalOutcome",
    "EvalDecision",
    "ValueEvalOutcome",
    "branch_reachability_under_env",
    "eval_bool_expr",
    "eval_value_expr",
    "is_reachability_false",
    "is_reachability_true",
]
