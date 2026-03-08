from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable, cast


@dataclass(frozen=True)
class ValueEncodedDecisionParamsDeps:
    check_deadline_fn: Callable[[], None]
    param_names_fn: Callable[..., list[str]]
    mark_param_roots_fn: Callable[[ast.AST, set[str], set[str]], None]
    contains_boolish_fn: Callable[[ast.AST], bool]
    collect_param_roots_fn: Callable[[ast.AST, set[str]], set[str]]


def value_encoded_decision_params(
    fn: ast.AST,
    ignore_params=None,
    *,
    deps: ValueEncodedDecisionParamsDeps,
) -> tuple[set[str], set[str]]:
    deps.check_deadline_fn()
    params = set(deps.param_names_fn(fn, ignore_params))
    if not params:
        return set(), set()

    flagged: set[str] = set()
    reasons: set[str] = set()
    for node in ast.walk(fn):
        deps.check_deadline_fn()
        node_type = type(node)
        if node_type is ast.Call:
            call_node = cast(ast.Call, node)
            func = call_node.func
            func_type = type(func)
            if func_type is ast.Name and cast(ast.Name, func).id in {"min", "max"}:
                reasons.add("min/max")
                deps.mark_param_roots_fn(call_node, params, flagged)
            elif func_type is ast.Attribute and cast(ast.Attribute, func).attr in {
                "min",
                "max",
            }:
                reasons.add("min/max")
                deps.mark_param_roots_fn(call_node, params, flagged)
        elif node_type is ast.BinOp:
            binop_node = cast(ast.BinOp, node)
            op_type = type(binop_node.op)
            left_bool = deps.contains_boolish_fn(binop_node.left)
            right_bool = deps.contains_boolish_fn(binop_node.right)
            if op_type in {
                ast.Mult,
                ast.Add,
                ast.Sub,
                ast.FloorDiv,
                ast.Mod,
                ast.BitAnd,
                ast.BitOr,
                ast.BitXor,
                ast.LShift,
                ast.RShift,
            }:
                if left_bool or right_bool:
                    reasons.add("boolean arithmetic")
                    if left_bool:
                        flagged |= deps.collect_param_roots_fn(binop_node.left, params)
                    if right_bool:
                        flagged |= deps.collect_param_roots_fn(binop_node.right, params)
                if op_type in {
                    ast.BitAnd,
                    ast.BitOr,
                    ast.BitXor,
                    ast.LShift,
                    ast.RShift,
                } and not (left_bool or right_bool):
                    left_roots = deps.collect_param_roots_fn(binop_node.left, params)
                    right_roots = deps.collect_param_roots_fn(binop_node.right, params)
                    if left_roots or right_roots:
                        reasons.add("bitmask")
                        flagged |= left_roots | right_roots
    return flagged, reasons
