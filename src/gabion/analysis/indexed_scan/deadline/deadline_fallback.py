# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.order_contract import sort_once


class _DeadlineArgInfoFactory(Protocol):
    def __call__(
        self,
        *,
        kind: str,
        param: object = None,
        const: object = None,
    ) -> object: ...


def fallback_deadline_arg_info(
    call: object,
    callee: object,
    *,
    strictness: str,
    deadline_arg_info_factory: _DeadlineArgInfoFactory,
    check_deadline_fn: Callable[[], None] = check_deadline,
    sort_once_fn: Callable[..., list[str]] = sort_once,
) -> dict[str, object]:
    check_deadline_fn()
    pos_params = (
        list(callee.positional_params)
        if callee.positional_params
        else list(callee.params)
    )
    kwonly_params = set(callee.kwonly_params or ())
    named_params = set(pos_params) | kwonly_params
    mapping: dict[str, object] = {}

    for idx_str, caller_param in call.pos_map.items():
        check_deadline_fn()
        idx = int(idx_str)
        if idx < len(pos_params):
            mapping[pos_params[idx]] = deadline_arg_info_factory(
                kind="param",
                param=caller_param,
            )
        elif callee.vararg is not None:
            mapping.setdefault(
                callee.vararg,
                deadline_arg_info_factory(kind="param", param=caller_param),
            )

    for idx_str, const_val in call.const_pos.items():
        check_deadline_fn()
        idx = int(idx_str)
        kind = "none" if const_val == "None" else "const"
        if idx < len(pos_params):
            mapping[pos_params[idx]] = deadline_arg_info_factory(
                kind=kind,
                const=const_val,
            )
        elif callee.vararg is not None:
            mapping.setdefault(
                callee.vararg,
                deadline_arg_info_factory(kind=kind, const=const_val),
            )

    for idx_str in call.non_const_pos:
        check_deadline_fn()
        idx = int(idx_str)
        if idx < len(pos_params):
            mapping[pos_params[idx]] = deadline_arg_info_factory(kind="unknown")
        elif callee.vararg is not None:
            mapping.setdefault(callee.vararg, deadline_arg_info_factory(kind="unknown"))

    for kw_name, caller_param in call.kw_map.items():
        check_deadline_fn()
        if kw_name in named_params:
            mapping[kw_name] = deadline_arg_info_factory(
                kind="param",
                param=caller_param,
            )
        elif callee.kwarg is not None:
            mapping.setdefault(
                callee.kwarg,
                deadline_arg_info_factory(kind="param", param=caller_param),
            )

    for kw_name, const_val in call.const_kw.items():
        check_deadline_fn()
        kind = "none" if const_val == "None" else "const"
        if kw_name in named_params:
            mapping[kw_name] = deadline_arg_info_factory(
                kind=kind,
                const=const_val,
            )
        elif callee.kwarg is not None:
            mapping.setdefault(
                callee.kwarg,
                deadline_arg_info_factory(kind=kind, const=const_val),
            )

    for kw_name in call.non_const_kw:
        check_deadline_fn()
        if kw_name in named_params:
            mapping[kw_name] = deadline_arg_info_factory(kind="unknown")
        elif callee.kwarg is not None:
            mapping.setdefault(callee.kwarg, deadline_arg_info_factory(kind="unknown"))

    if strictness == "low":
        remaining = [
            param
            for param in sort_once_fn(
                named_params,
                source="indexed_scan.deadline_fallback.fallback_deadline_arg_info.remaining",
            )
            if param not in mapping
        ]
        if len(call.star_pos) == 1:
            _, star_param = call.star_pos[0]
            for param in remaining:
                check_deadline_fn()
                mapping.setdefault(
                    param,
                    deadline_arg_info_factory(kind="param", param=star_param),
                )
        if len(call.star_kw) == 1:
            star_param = call.star_kw[0]
            for param in remaining:
                check_deadline_fn()
                mapping.setdefault(
                    param,
                    deadline_arg_info_factory(kind="param", param=star_param),
                )

    return mapping


__all__ = ["fallback_deadline_arg_info"]
