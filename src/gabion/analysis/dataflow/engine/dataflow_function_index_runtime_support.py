from __future__ import annotations

"""Canonical runtime helpers for function-index callsite shaping."""

import ast
from dataclasses import replace
from collections.abc import Mapping, Sequence

from gabion.analysis.dataflow.engine.dataflow_contracts import (
    CallArgs,
    FunctionInfo,
    ParamUse,
)
from gabion.analysis.dataflow.engine.dataflow_function_index_helpers import _node_span
from gabion.analysis.foundation.timeout_context import check_deadline


def _unused_params(use_map: dict[str, ParamUse]) -> tuple[set[str], set[str]]:
    check_deadline()
    eligible = [
        (name, info)
        for name, info in use_map.items()
        if (not info.non_forward) and (not info.direct_forward)
    ]
    unknown_key_carriers = {
        name
        for name, info in eligible
        if info.unknown_key_carrier
    }
    unused = {
        name
        for name, info in eligible
        if not info.unknown_key_carrier
    }
    return unused, unknown_key_carriers


def _direct_lambda_callee_by_call_span(
    call_nodes: Sequence[ast.Call],
    *,
    lambda_infos: Sequence[FunctionInfo],
) -> dict[tuple[int, int, int, int], str]:
    check_deadline()
    lambda_qual_by_span = {
        info.function_span: info.qual
        for info in lambda_infos
        if info.function_span is not None
    }
    call_spans = [(_node_span(call_node), _node_span(call_node.func)) for call_node in call_nodes]
    return {
        call_span: lambda_qual_by_span[lambda_span]
        for call_span, lambda_span in call_spans
        if call_span is not None and lambda_span is not None and lambda_span in lambda_qual_by_span
    }


def _materialize_direct_lambda_callees(
    call_args: Sequence[CallArgs],
    *,
    direct_lambda_callee_by_call_span: Mapping[tuple[int, int, int, int], str],
) -> list[CallArgs]:
    return [
        (
            check_deadline(),
            replace(
                call,
                callee=direct_lambda_callee_by_call_span.get(call.span, call.callee),
            ),
        )[1]
        for call in call_args
    ]


__all__ = [
    "_direct_lambda_callee_by_call_span",
    "_materialize_direct_lambda_callees",
    "_unused_params",
]
