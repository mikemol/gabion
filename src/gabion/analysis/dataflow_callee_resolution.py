# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

_BOUND = False


def _bind_audit_symbols() -> None:
    global _BOUND
    if _BOUND:
        return
    from gabion.analysis import dataflow_audit as _audit

    module_globals = globals()
    for name, value in _audit.__dict__.items():
        module_globals.setdefault(name, value)
    _BOUND = True


@dataclass(frozen=True)
class CalleeResolutionContext:
    callee_key: str
    caller: FunctionInfo
    by_name: Mapping[str, list[FunctionInfo]]
    by_qual: Mapping[str, FunctionInfo]
    symbol_table: object
    project_root: object
    class_index: object
    call: object
    local_lambda_bindings: Mapping[str, tuple[str, ...]]
    caller_module: str


@dataclass(frozen=True)
class ResolutionOp:
    kind: str


@dataclass(frozen=True)
class CalleeResolutionEffect:
    phase: str
    callee_key: str
    candidates: tuple[FunctionInfo, ...]


@dataclass(frozen=True)
class CalleeResolutionOutcome:
    resolved: object
    effects: tuple[CalleeResolutionEffect, ...]
    phase: str


@dataclass
class _ResolutionState:
    candidates: list[FunctionInfo]
    effects: list[CalleeResolutionEffect]
    stop: bool = False
    phase: str = "unresolved"


def plan_callee_resolution(context: CalleeResolutionContext) -> tuple[ResolutionOp, ...]:
    check_deadline()
    ops: list[ResolutionOp] = [
        ResolutionOp(kind="guard_empty"),
        ResolutionOp(kind="resolve_local_lambda"),
    ]
    if "." not in context.callee_key:
        ops.append(ResolutionOp(kind="resolve_unqualified_local_or_global"))
    if context.symbol_table is not None:
        ops.append(ResolutionOp(kind="resolve_symbol_table"))
    ops.append(ResolutionOp(kind="resolve_exact_qualified"))
    if context.class_index is not None and "." in context.callee_key:
        ops.append(ResolutionOp(kind="resolve_class_hierarchy"))
    return tuple(ops)


def _candidate_pool(context: CalleeResolutionContext) -> list[FunctionInfo]:
    candidates = list(context.by_name.get(_callee_key(context.callee_key), []))
    if "." not in context.callee_key:
        # Unqualified Python name resolution is module-local unless an import
        # explicitly binds the symbol into scope.
        candidates = [info for info in candidates if info.path == context.caller.path]
    return candidates


def _emit_ambiguity(
    state: _ResolutionState,
    *,
    phase: str,
    callee_key: str,
    candidates: list[FunctionInfo],
) -> None:
    if not candidates:
        return
    state.effects.append(
        CalleeResolutionEffect(
            phase=phase,
            callee_key=callee_key,
            candidates=tuple(candidates),
        )
    )
    state.phase = phase


def _resolve_local_lambda(
    context: CalleeResolutionContext,
    state: _ResolutionState,
) -> object:
    bound_lambda_quals = tuple(context.local_lambda_bindings.get(context.callee_key, ()))
    if len(bound_lambda_quals) == 1:
        bound = context.by_qual.get(bound_lambda_quals[0])
        if bound is not None:
            state.phase = "local_lambda_binding"
            return bound
    elif len(bound_lambda_quals) > 1:
        bound_candidates = [
            context.by_qual[qual]
            for qual in bound_lambda_quals
            if qual in context.by_qual
        ]
        if len(bound_candidates) == 1:
            state.phase = "local_lambda_binding"
            return bound_candidates[0]
        if bound_candidates:
            _emit_ambiguity(
                state,
                phase="local_lambda_binding",
                callee_key=context.callee_key,
                candidates=bound_candidates,
            )
            state.stop = True
    return None


def _resolve_unqualified_local_or_global(
    context: CalleeResolutionContext,
    state: _ResolutionState,
) -> object:
    ambiguous = False
    effective_scope = list(context.caller.lexical_scope) + [context.caller.name]
    while True:
        check_deadline()
        scoped = [
            info
            for info in state.candidates
            if list(info.lexical_scope) == effective_scope
            and not (info.class_name and not info.lexical_scope)
        ]
        if len(scoped) == 1:
            state.phase = "local_resolution"
            return scoped[0]
        if len(scoped) > 1:
            ambiguous = True
            _emit_ambiguity(
                state,
                phase="local_resolution",
                callee_key=context.callee_key,
                candidates=scoped,
            )
            break
        if not effective_scope:
            break
        effective_scope = effective_scope[:-1]

    if ambiguous:
        pass

    globals_only = [
        info
        for info in state.candidates
        if not info.lexical_scope
        and not (info.class_name and not info.lexical_scope)
        and info.path == context.caller.path
    ]
    if len(globals_only) == 1:
        state.phase = "global_resolution"
        return globals_only[0]
    return None


def _resolve_symbol_table(
    context: CalleeResolutionContext,
    state: _ResolutionState,
) -> object:
    symbol_table = context.symbol_table
    if symbol_table is not None:
        if "." not in context.callee_key:
            if (context.caller_module, context.callee_key) in symbol_table.imports:
                fqn = symbol_table.resolve(context.caller_module, context.callee_key)
                if fqn is None:
                    state.phase = "import_resolution"
                    state.stop = True
                    return None
                if fqn in context.by_qual:
                    state.phase = "import_resolution"
                    return context.by_qual[fqn]
            resolved = symbol_table.resolve_star(context.caller_module, context.callee_key)
            if resolved is not None and resolved in context.by_qual:
                state.phase = "import_resolution"
                return context.by_qual[resolved]
            return None

        parts = context.callee_key.split(".")
        base = parts[0]
        if base in ("self", "cls") and len(parts) == 2:
            method = parts[-1]
            if context.caller.class_name:
                candidate = f"{context.caller_module}.{context.caller.class_name}.{method}"
                if candidate in context.by_qual:
                    state.phase = "class_self_resolution"
                    return context.by_qual[candidate]
        elif len(parts) == 2:
            candidate = f"{context.caller_module}.{base}.{parts[1]}"
            if candidate in context.by_qual:
                state.phase = "local_attribute_resolution"
                return context.by_qual[candidate]

        if (context.caller_module, base) in symbol_table.imports:
            base_fqn = symbol_table.resolve(context.caller_module, base)
            if base_fqn is None:
                state.phase = "import_resolution"
                state.stop = True
                return None
            candidate = base_fqn + "." + ".".join(parts[1:])
            if candidate in context.by_qual:
                state.phase = "import_resolution"
                return context.by_qual[candidate]
    return None


def _resolve_exact_qualified(
    context: CalleeResolutionContext,
    state: _ResolutionState,
) -> object:
    if context.callee_key in context.by_qual:
        state.phase = "exact_qualified"
        return context.by_qual[context.callee_key]
    return None


def _resolve_class_hierarchy(
    context: CalleeResolutionContext,
    state: _ResolutionState,
) -> object:
    class_index = context.class_index
    if class_index is not None and "." in context.callee_key:
        parts = context.callee_key.split(".")
        method = parts[-1]
        class_part = ".".join(parts[:-1])
        symbol_table = context.symbol_table
        if class_part in {"self", "cls"} and context.caller.class_name:
            class_candidates = _resolve_class_candidates(
                context.caller.class_name,
                module=context.caller_module,
                symbol_table=symbol_table,
                class_index=class_index,
            )
        else:
            class_candidates = _resolve_class_candidates(
                class_part,
                module=context.caller_module,
                symbol_table=symbol_table,
                class_index=class_index,
            )
        for class_qual in class_candidates:
            check_deadline()
            resolved = _resolve_method_in_hierarchy(
                class_qual,
                method,
                class_index=class_index,
                by_qual=context.by_qual,
                symbol_table=symbol_table,
                seen=set(),
            )
            if resolved is not None:
                state.phase = "class_hierarchy_resolution"
                return resolved
    return None


_OP_HANDLERS: dict[str, callable] = {
    "resolve_local_lambda": _resolve_local_lambda,
    "resolve_unqualified_local_or_global": _resolve_unqualified_local_or_global,
    "resolve_symbol_table": _resolve_symbol_table,
    "resolve_exact_qualified": _resolve_exact_qualified,
    "resolve_class_hierarchy": _resolve_class_hierarchy,
}


def apply_callee_resolution_ops(
    context: CalleeResolutionContext,
    ops: tuple[ResolutionOp, ...],
) -> CalleeResolutionOutcome:
    check_deadline()
    state = _ResolutionState(candidates=_candidate_pool(context), effects=[])
    if not context.callee_key:
        state.phase = "empty_callee_key"
        return CalleeResolutionOutcome(resolved=None, effects=tuple(), phase=state.phase)

    for op in ops:
        check_deadline()
        if op.kind == "guard_empty":
            continue
        handler = _OP_HANDLERS.get(op.kind)
        if handler is not None:
            resolved = handler(context, state)
            if resolved is not None:
                return CalleeResolutionOutcome(
                    resolved=resolved,
                    effects=tuple(state.effects),
                    phase=state.phase,
                )
            if state.stop:
                return CalleeResolutionOutcome(
                    resolved=None,
                    effects=tuple(state.effects),
                    phase=state.phase,
                )

    return CalleeResolutionOutcome(
        resolved=None,
        effects=tuple(state.effects),
        phase=state.phase,
    )


def collect_callee_resolution_effects(
    outcome: CalleeResolutionOutcome,
) -> tuple[CalleeResolutionEffect, ...]:
    check_deadline()
    return outcome.effects


def resolve_callee_with_effects(
    context: CalleeResolutionContext,
) -> CalleeResolutionOutcome:
    _bind_audit_symbols()
    ops = plan_callee_resolution(context)
    return apply_callee_resolution_ops(context, ops)
