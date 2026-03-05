# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CalleeOutcomeDeps:
    check_deadline_fn: Callable[[], None]
    callee_resolution_context_core_ctor: Callable[..., object]
    resolve_callee_with_effects_fn: Callable[..., object]
    collect_callee_resolution_effects_fn: Callable[..., list[object]]
    module_name_fn: Callable[..., str]
    dedupe_resolution_candidates_fn: Callable[..., tuple[object, ...]]
    callee_key_fn: Callable[[str], str]
    is_dynamic_dispatch_callee_key_fn: Callable[[str], bool]
    outcome_ctor: Callable[..., object]
    default_resolve_callee_fn: Callable[..., object]


@dataclass(frozen=True)
class ResolveCalleeDeps:
    check_deadline_fn: Callable[[], None]
    callee_resolution_context_core_ctor: Callable[..., object]
    resolve_callee_with_effects_fn: Callable[..., object]
    collect_callee_resolution_effects_fn: Callable[..., list[object]]
    module_name_fn: Callable[..., str]


def resolve_callee(
    callee_key: str,
    caller: object,
    by_name: dict[str, list[object]],
    by_qual: dict[str, object],
    *,
    symbol_table=None,
    project_root=None,
    class_index=None,
    call=None,
    ambiguity_sink=None,
    local_lambda_bindings=None,
    deps: ResolveCalleeDeps,
):
    deps.check_deadline_fn()
    lambda_bindings = local_lambda_bindings
    if lambda_bindings is None:
        lambda_bindings = caller.local_lambda_bindings
    context = deps.callee_resolution_context_core_ctor(
        callee_key=callee_key,
        caller=caller,
        by_name=by_name,
        by_qual=by_qual,
        symbol_table=symbol_table,
        project_root=project_root,
        class_index=class_index,
        call=call,
        local_lambda_bindings=lambda_bindings,
        caller_module=deps.module_name_fn(caller.path, project_root=project_root),
    )
    resolution = deps.resolve_callee_with_effects_fn(context)
    if ambiguity_sink is not None:
        for effect in deps.collect_callee_resolution_effects_fn(resolution):
            deps.check_deadline_fn()
            ambiguity_sink(
                caller,
                call,
                list(effect.candidates),
                effect.phase,
                effect.callee_key,
            )
    return resolution.resolved


def resolve_callee_outcome(
    callee_key: str,
    caller: object,
    by_name: dict[str, list[object]],
    by_qual: dict[str, object],
    *,
    symbol_table=None,
    project_root=None,
    class_index=None,
    call=None,
    local_lambda_bindings=None,
    resolve_callee_fn=None,
    deps: CalleeOutcomeDeps,
):
    deps.check_deadline_fn()
    if resolve_callee_fn is None:
        resolve_callee_fn = deps.default_resolve_callee_fn
    ambiguous_candidates: list[object] = []
    ambiguity_phase = "unresolved"
    ambiguity_callee_key = callee_key

    def _sink(
        sink_caller: object,
        sink_call,
        candidates: list[object],
        phase: str,
        sink_callee_key: str,
    ) -> None:
        deps.check_deadline_fn()
        del sink_caller, sink_call
        ambiguous_candidates.extend(candidates)
        nonlocal ambiguity_phase, ambiguity_callee_key
        ambiguity_phase = phase
        ambiguity_callee_key = sink_callee_key

    if resolve_callee_fn is deps.default_resolve_callee_fn:
        lambda_bindings = local_lambda_bindings
        if lambda_bindings is None:
            lambda_bindings = caller.local_lambda_bindings
        context = deps.callee_resolution_context_core_ctor(
            callee_key=callee_key,
            caller=caller,
            by_name=by_name,
            by_qual=by_qual,
            symbol_table=symbol_table,
            project_root=project_root,
            class_index=class_index,
            call=call,
            local_lambda_bindings=lambda_bindings,
            caller_module=deps.module_name_fn(caller.path, project_root=project_root),
        )
        resolution = deps.resolve_callee_with_effects_fn(context)
        for effect in deps.collect_callee_resolution_effects_fn(resolution):
            deps.check_deadline_fn()
            _sink(caller, call, list(effect.candidates), effect.phase, effect.callee_key)
        resolved = resolution.resolved
    else:
        resolved = resolve_callee_fn(
            callee_key,
            caller,
            by_name,
            by_qual,
            symbol_table=symbol_table,
            project_root=project_root,
            class_index=class_index,
            call=call,
            ambiguity_sink=_sink,
            local_lambda_bindings=local_lambda_bindings,
        )
    if resolved is not None:
        return deps.outcome_ctor(
            status="resolved",
            phase="resolved",
            callee_key=callee_key,
            candidates=(resolved,),
        )
    ambiguous = deps.dedupe_resolution_candidates_fn(ambiguous_candidates)
    if ambiguous:
        return deps.outcome_ctor(
            status="ambiguous",
            phase=ambiguity_phase,
            callee_key=ambiguity_callee_key,
            candidates=ambiguous,
        )
    internal_pool = list(by_name.get(deps.callee_key_fn(callee_key), []))
    lambda_bindings = local_lambda_bindings
    if lambda_bindings is None:
        lambda_bindings = caller.local_lambda_bindings
    if "." not in callee_key:
        for qual in lambda_bindings.get(callee_key, ()):
            deps.check_deadline_fn()
            candidate = by_qual.get(qual)
            if candidate is not None:
                internal_pool.append(candidate)
    internal_candidates = deps.dedupe_resolution_candidates_fn(internal_pool)
    if internal_candidates:
        return deps.outcome_ctor(
            status="unresolved_internal",
            phase="unresolved_internal",
            callee_key=callee_key,
            candidates=internal_candidates,
        )
    if deps.is_dynamic_dispatch_callee_key_fn(callee_key):
        return deps.outcome_ctor(
            status="unresolved_dynamic",
            phase="unresolved_dynamic",
            callee_key=callee_key,
            candidates=(),
        )
    return deps.outcome_ctor(
        status="unresolved_external",
        phase="unresolved_external",
        callee_key=callee_key,
        candidates=(),
    )

