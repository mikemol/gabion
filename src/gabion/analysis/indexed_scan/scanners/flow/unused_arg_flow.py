from __future__ import annotations

from typing import Callable


def analyze_unused_arg_flow_indexed(
    context,
    *,
    analysis_index_resolved_call_edges_fn: Callable[..., tuple[object, ...]],
    check_deadline_fn: Callable[[], None],
    sort_once_fn: Callable[..., list[str]],
) -> list[str]:
    resolved_edges = analysis_index_resolved_call_edges_fn(
        context.analysis_index,
        project_root=context.project_root,
        require_transparent=True,
    )
    smells: set[str] = set()

    def _format(
        caller,
        callee_info,
        callee_param: str,
        arg_desc: str,
        *,
        category: str = "unused",
        call=None,
    ) -> str:
        prefix = f"{caller.path.name}:{caller.name}"
        if call is not None and call.span is not None:
            line, col, _, _ = call.span
            prefix = f"{caller.path.name}:{line + 1}:{col + 1}:{caller.name}"
        if category == "unknown_key_carrier":
            return (
                f"{prefix} passes {arg_desc} to {callee_info.path.name}:{callee_info.name}.{callee_param} "
                f"(unknown key carrier)"
            )
        return (
            f"{prefix} passes {arg_desc} "
            f"to unused {callee_info.path.name}:{callee_info.name}.{callee_param} "
            f"(no forwarding use)"
        )

    for edge in resolved_edges:
        check_deadline_fn()
        info = edge.caller
        call = edge.call
        callee = edge.callee
        if not callee.unused_params and not callee.unknown_key_carriers:
            continue
        callee_params = callee.params
        mapped_params = set()
        for idx_str in call.pos_map:
            check_deadline_fn()
            idx = int(idx_str)
            if idx >= len(callee_params):
                continue
            mapped_params.add(callee_params[idx])
        for kw in call.kw_map:
            check_deadline_fn()
            if kw in callee_params:
                mapped_params.add(kw)
        remaining = [
            (idx, name)
            for idx, name in enumerate(callee_params)
            if name not in mapped_params
        ]

        for idx_str, caller_param in call.pos_map.items():
            check_deadline_fn()
            idx = int(idx_str)
            if idx >= len(callee_params):
                continue
            callee_param = callee_params[idx]
            if callee_param in callee.unused_params | callee.unknown_key_carriers:
                smells.add(
                    _format(
                        info,
                        callee,
                        callee_param,
                        f"param {caller_param}",
                        category=(
                            "unknown_key_carrier"
                            if callee_param in callee.unknown_key_carriers
                            else "unused"
                        ),
                        call=call,
                    )
                )
        for idx_str in call.non_const_pos:
            check_deadline_fn()
            idx = int(idx_str)
            if idx >= len(callee_params):
                continue
            callee_param = callee_params[idx]
            if callee_param in callee.unused_params | callee.unknown_key_carriers:
                smells.add(
                    _format(
                        info,
                        callee,
                        callee_param,
                        f"non-constant arg at position {idx}",
                        category=(
                            "unknown_key_carrier"
                            if callee_param in callee.unknown_key_carriers
                            else "unused"
                        ),
                        call=call,
                    )
                )
        for kw, caller_param in call.kw_map.items():
            check_deadline_fn()
            if kw not in callee_params:
                continue
            if kw in callee.unused_params | callee.unknown_key_carriers:
                smells.add(
                    _format(
                        info,
                        callee,
                        kw,
                        f"param {caller_param}",
                        category=(
                            "unknown_key_carrier"
                            if kw in callee.unknown_key_carriers
                            else "unused"
                        ),
                        call=call,
                    )
                )
        for kw in call.non_const_kw:
            check_deadline_fn()
            if kw not in callee_params:
                continue
            if kw in callee.unused_params | callee.unknown_key_carriers:
                smells.add(
                    _format(
                        info,
                        callee,
                        kw,
                        f"non-constant kw '{kw}'",
                        category=(
                            "unknown_key_carrier"
                            if kw in callee.unknown_key_carriers
                            else "unused"
                        ),
                        call=call,
                    )
                )
        if context.strictness == "low":
            if len(call.star_pos) == 1:
                for idx, param in remaining:
                    check_deadline_fn()
                    if param in callee.unused_params | callee.unknown_key_carriers:
                        smells.add(
                            _format(
                                info,
                                callee,
                                param,
                                f"non-constant arg at position {idx}",
                                category=(
                                    "unknown_key_carrier"
                                    if param in callee.unknown_key_carriers
                                    else "unused"
                                ),
                                call=call,
                            )
                        )
            if len(call.star_kw) == 1:
                for _, param in remaining:
                    check_deadline_fn()
                    if param in callee.unused_params | callee.unknown_key_carriers:
                        smells.add(
                            _format(
                                info,
                                callee,
                                param,
                                f"non-constant kw '{param}'",
                                category=(
                                    "unknown_key_carrier"
                                    if param in callee.unknown_key_carriers
                                    else "unused"
                                ),
                                call=call,
                            )
                        )
    return sort_once_fn(
        smells,
        source="indexed_scan.unused_arg_flow.analyze_unused_arg_flow_indexed",
    )
