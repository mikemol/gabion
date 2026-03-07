# gabion:decision_protocol_module
from __future__ import annotations

from typing import Callable, Iterator

CheckDeadlineFn = Callable[[], None]
EventCtor = Callable[..., object]


def iter_resolved_edge_param_events(
    *,
    edge: object,
    strictness: str,
    include_variadics_in_low_star: bool,
    check_deadline_fn: CheckDeadlineFn,
    event_ctor: EventCtor,
) -> Iterator[object]:
    check_deadline_fn()
    call = edge.call
    callee = edge.callee
    pos_params = list(callee.positional_params) if callee.positional_params else list(callee.params)
    kwonly_params = set(callee.kwonly_params or ())
    named_params = set(pos_params) | kwonly_params
    mapped_params: set[str] = set()

    for idx_str in call.pos_map:
        check_deadline_fn()
        idx = int(idx_str)
        if idx < len(pos_params):
            param = pos_params[idx]
            mapped_params.add(param)
            yield event_ctor(kind="non_const", param=param, value=None, countable=True)
        elif callee.vararg is not None:
            mapped_params.add(callee.vararg)
            yield event_ctor(
                kind="non_const",
                param=callee.vararg,
                value=None,
                countable=False,
            )
    for kw in call.kw_map:
        check_deadline_fn()
        if kw in named_params:
            mapped_params.add(kw)
            yield event_ctor(kind="non_const", param=kw, value=None, countable=True)
        elif callee.kwarg is not None:
            mapped_params.add(callee.kwarg)
            yield event_ctor(
                kind="non_const",
                param=callee.kwarg,
                value=None,
                countable=False,
            )

    for idx_str, value in call.const_pos.items():
        check_deadline_fn()
        idx = int(idx_str)
        if idx < len(pos_params):
            yield event_ctor(kind="const", param=pos_params[idx], value=value, countable=True)
        elif callee.vararg is not None:
            yield event_ctor(
                kind="non_const",
                param=callee.vararg,
                value=None,
                countable=False,
            )
    for idx_str in call.non_const_pos:
        check_deadline_fn()
        idx = int(idx_str)
        if idx < len(pos_params):
            yield event_ctor(
                kind="non_const",
                param=pos_params[idx],
                value=None,
                countable=True,
            )
        elif callee.vararg is not None:
            yield event_ctor(
                kind="non_const",
                param=callee.vararg,
                value=None,
                countable=False,
            )
    for kw, value in call.const_kw.items():
        check_deadline_fn()
        if kw in named_params:
            yield event_ctor(kind="const", param=kw, value=value, countable=True)
        elif callee.kwarg is not None:
            yield event_ctor(
                kind="non_const",
                param=callee.kwarg,
                value=None,
                countable=False,
            )
    for kw in call.non_const_kw:
        check_deadline_fn()
        if kw in named_params:
            yield event_ctor(kind="non_const", param=kw, value=None, countable=True)
        elif callee.kwarg is not None:
            yield event_ctor(
                kind="non_const",
                param=callee.kwarg,
                value=None,
                countable=False,
            )

    if strictness != "low":
        return

    remaining = [p for p in named_params if p not in mapped_params]
    if include_variadics_in_low_star:
        if callee.vararg is not None and callee.vararg not in mapped_params:
            remaining.append(callee.vararg)
        if callee.kwarg is not None and callee.kwarg not in mapped_params:
            remaining.append(callee.kwarg)

    if len(call.star_pos) == 1:
        for param in remaining:
            check_deadline_fn()
            yield event_ctor(
                kind="non_const",
                param=param,
                value=None,
                countable=param in named_params,
            )
        if not include_variadics_in_low_star and callee.vararg is not None:
            yield event_ctor(
                kind="non_const",
                param=callee.vararg,
                value=None,
                countable=False,
            )

    if len(call.star_kw) == 1:
        for param in remaining:
            check_deadline_fn()
            yield event_ctor(
                kind="non_const",
                param=param,
                value=None,
                countable=param in named_params,
            )
        if not include_variadics_in_low_star and callee.kwarg is not None:
            yield event_ctor(
                kind="non_const",
                param=callee.kwarg,
                value=None,
                countable=False,
            )
