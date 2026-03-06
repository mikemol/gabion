# gabion:decision_protocol_module
from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, cast


@dataclass(frozen=True)
class CollectDeadlineLocalInfoDeps:
    check_deadline_fn: Callable[[], None]
    is_deadline_origin_call_fn: Callable[[object], bool]
    target_names_fn: Callable[[ast.AST], set[str]]
    deadline_local_info_ctor: Callable[..., object]


def collect_deadline_local_info(
    assignments: list[tuple[list[ast.AST], object, object]],
    params: set[str],
    *,
    deps: CollectDeadlineLocalInfoDeps,
):
    deps.check_deadline_fn()
    origin_assign: set[str] = set()
    origin_spans: dict[str, tuple[int, int, int, int]] = {}
    for targets, value, span in assignments:
        deps.check_deadline_fn()
        if value is not None and deps.is_deadline_origin_call_fn(value):
            for target in targets:
                deps.check_deadline_fn()
                for name in deps.target_names_fn(target):
                    deps.check_deadline_fn()
                    origin_assign.add(name)
                    if span is not None and name not in origin_spans:
                        origin_spans[name] = cast(tuple[int, int, int, int], span)

    alias_assign: dict[str, set[str]] = defaultdict(set)
    origin_alias: set[str] = set()
    unknown_assign: set[str] = set()
    for targets, value, _ in assignments:
        deps.check_deadline_fn()
        if value is None:
            for target in targets:
                deps.check_deadline_fn()
                unknown_assign.update(deps.target_names_fn(target))
        elif not deps.is_deadline_origin_call_fn(value):
            alias_source = None
            propagate_origin_alias = False
            if type(value) is ast.Name:
                value_name = cast(ast.Name, value)
                if value_name.id in params:
                    alias_source = value_name.id
                elif value_name.id in origin_assign:
                    propagate_origin_alias = True
            for target in targets:
                deps.check_deadline_fn()
                for name in deps.target_names_fn(target):
                    deps.check_deadline_fn()
                    if propagate_origin_alias:
                        origin_alias.add(name)
                    elif alias_source is not None:
                        alias_assign[name].add(alias_source)
                    else:
                        unknown_assign.add(name)

    origin_candidates = origin_assign | origin_alias
    origin_vars = {
        name
        for name in origin_candidates
        if name not in unknown_assign and name not in alias_assign
    }

    alias_to_param: dict[str, str] = {}
    for name, sources in alias_assign.items():
        deps.check_deadline_fn()
        if name in unknown_assign or name in origin_candidates:
            continue
        if len(sources) == 1:
            alias_to_param[name] = next(iter(sources))
    for param in params:
        deps.check_deadline_fn()
        alias_to_param[param] = param

    return deps.deadline_local_info_ctor(
        origin_vars=origin_vars,
        origin_spans=origin_spans,
        alias_to_param=alias_to_param,
    )
