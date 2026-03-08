from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PropagateGroupsDeps:
    check_deadline_fn: Callable[[], None]


def propagate_groups(
    call_args: list[object],
    callee_groups: dict[str, list[set[str]]],
    callee_param_orders: dict[str, list[str]],
    strictness: str,
    *,
    opaque_callees=None,
    deps: PropagateGroupsDeps,
) -> list[set[str]]:
    deps.check_deadline_fn()
    groups: list[set[str]] = []
    for call in call_args:
        deps.check_deadline_fn()
        if opaque_callees and call.callee in opaque_callees:
            continue
        if call.callee not in callee_groups:
            continue
        callee_params = list(callee_param_orders[call.callee])
        mapping = call.argument_mapping()
        if (
            "." in call.callee
            and callee_params
            and callee_params[0] in {"self", "cls"}
            and len(mapping.positional) < len(callee_params)
        ):
            # Bound method calls (obj.method(...)) omit the receiver argument.
            callee_params = callee_params[1:]
        # Build mapping from callee param to caller param.
        callee_to_caller: dict[str, str] = {}
        for idx, pname in enumerate(callee_params):
            deps.check_deadline_fn()
            if idx in mapping.positional:
                callee_to_caller[pname] = mapping.positional[idx].value
        for kw, caller_param in mapping.keywords.items():
            deps.check_deadline_fn()
            callee_to_caller[kw] = caller_param.value
        if strictness == "low":
            mapped = set(callee_to_caller.keys())
            remaining = [p for p in callee_params if p not in mapped]
            if len(mapping.star_positional) == 1:
                _, star_param = mapping.star_positional[0]
                for param in remaining:
                    deps.check_deadline_fn()
                    callee_to_caller.setdefault(param, star_param.value)
            if len(mapping.star_keywords) == 1:
                star_param = mapping.star_keywords[0]
                for param in remaining:
                    deps.check_deadline_fn()
                    callee_to_caller.setdefault(param, star_param.value)
        for group in callee_groups[call.callee]:
            deps.check_deadline_fn()
            mapped = {callee_to_caller.get(p) for p in group}
            mapped.discard(None)
            if len(mapped) > 1:
                groups.append(set(mapped))
    return groups
