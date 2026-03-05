# gabion:boundary_normalization_module
from __future__ import annotations

"""Canonical runtime helpers for function-index callsite shaping."""

import ast
from dataclasses import replace
from collections.abc import Mapping, Sequence
from typing import cast

from gabion.analysis.dataflow.engine.dataflow_contracts import (
    CallArgs,
    FunctionInfo,
    ParamUse,
)
from gabion.analysis.dataflow.engine.dataflow_function_index_helpers import _node_span
from gabion.analysis.foundation.timeout_context import check_deadline


def _unused_params(use_map: dict[str, ParamUse]) -> tuple[set[str], set[str]]:
    check_deadline()
    unused: set[str] = set()
    unknown_key_carriers: set[str] = set()
    for name, info in use_map.items():
        check_deadline()
        if info.non_forward:
            continue
        if info.direct_forward:
            continue
        if info.unknown_key_carrier:
            unknown_key_carriers.add(name)
            continue
        unused.add(name)
    return unused, unknown_key_carriers


def _direct_lambda_callee_by_call_span(
    tree: ast.AST,
    *,
    lambda_infos: Sequence[FunctionInfo],
) -> dict[tuple[int, int, int, int], str]:
    check_deadline()
    lambda_qual_by_span = {
        info.function_span: info.qual
        for info in lambda_infos
        if info.function_span is not None
    }
    mapping: dict[tuple[int, int, int, int], str] = {}
    for node in ast.walk(tree):
        check_deadline()
        if type(node) is ast.Call:
            call_node = cast(ast.Call, node)
            if type(call_node.func) is ast.Lambda:
                call_span = _node_span(call_node)
                lambda_span = _node_span(call_node.func)
                if call_span is not None and lambda_span is not None:
                    callee = lambda_qual_by_span.get(lambda_span)
                    if callee is not None:
                        mapping[call_span] = callee
    return mapping


def _materialize_direct_lambda_callees(
    call_args: Sequence[CallArgs],
    *,
    direct_lambda_callee_by_call_span: Mapping[tuple[int, int, int, int], str],
) -> list[CallArgs]:
    out: list[CallArgs] = []
    for call in call_args:
        check_deadline()
        if call.span is not None and call.span in direct_lambda_callee_by_call_span:
            out.append(replace(call, callee=direct_lambda_callee_by_call_span[call.span]))
            continue
        out.append(call)
    return out


__all__ = [
    "_direct_lambda_callee_by_call_span",
    "_materialize_direct_lambda_callees",
    "_unused_params",
]
