from __future__ import annotations

from pathlib import Path


def _load():
    from gabion.analysis import dataflow_audit as da
    from gabion.analysis import dataflow_callee_resolution as resolution

    return da, resolution


def _fn(da, *, name: str, qual: str, path: Path, class_name: str | None = None):
    return da.FunctionInfo(
        name=name,
        qual=qual,
        path=path,
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        class_name=class_name,
        scope=(),
        lexical_scope=(),
        function_span=(0, 0, 0, 1),
    )


# gabion:evidence E:function_site::dataflow_callee_resolution.py::gabion.analysis.dataflow_callee_resolution.plan_callee_resolution
def test_callee_resolution_plan_and_effect_pipeline_idempotent() -> None:
    da, resolution = _load()
    resolution._bind_audit_symbols()

    path = Path("pkg/mod.py")
    caller = _fn(da, name="caller", qual="pkg.mod.caller", path=path)
    target = _fn(da, name="target", qual="pkg.mod.target", path=path)
    context = resolution.CalleeResolutionContext(
        callee_key="target",
        caller=caller,
        by_name={"target": [target]},
        by_qual={caller.qual: caller, target.qual: target},
        symbol_table=None,
        project_root=None,
        class_index=None,
        call=None,
        local_lambda_bindings={},
        caller_module="pkg.mod",
    )

    first_ops = resolution.plan_callee_resolution(context)
    second_ops = resolution.plan_callee_resolution(context)
    assert tuple(op.kind for op in first_ops) == tuple(op.kind for op in second_ops)

    first = resolution.apply_callee_resolution_ops(context, first_ops)
    second = resolution.apply_callee_resolution_ops(context, second_ops)
    assert first.resolved is target
    assert second.resolved is target
    assert first.phase == second.phase
    assert resolution.collect_callee_resolution_effects(first) == ()
    assert resolution.collect_callee_resolution_effects(second) == ()


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._resolve_callee
def test_resolve_callee_adapter_dispatches_ambiguity_effects() -> None:
    da, _ = _load()
    path = Path("pkg/mod.py")
    caller = _fn(da, name="caller", qual="pkg.mod.caller", path=path)
    bound_a = _fn(da, name="<lambda:a>", qual="pkg.mod.<lambda:a>", path=path)
    bound_b = _fn(da, name="<lambda:b>", qual="pkg.mod.<lambda:b>", path=path)
    by_qual = {
        caller.qual: caller,
        bound_a.qual: bound_a,
        bound_b.qual: bound_b,
    }

    sink_calls: list[tuple[str, str, tuple[str, ...]]] = []

    def _sink(_caller, _call, candidates, phase: str, callee_key: str) -> None:
        sink_calls.append((phase, callee_key, tuple(info.qual for info in candidates)))

    resolved = da._resolve_callee(
        "fn",
        caller,
        {},
        by_qual,
        ambiguity_sink=_sink,
        local_lambda_bindings={"fn": (bound_a.qual, bound_b.qual)},
    )
    assert resolved is None
    assert sink_calls == [
        (
            "local_lambda_binding",
            "fn",
            (bound_a.qual, bound_b.qual),
        )
    ]


# gabion:evidence E:function_site::dataflow_callee_resolution.py::gabion.analysis.dataflow_callee_resolution.apply_callee_resolution_ops
def test_callee_resolution_private_noop_paths_are_structural() -> None:
    da, resolution = _load()
    resolution._bind_audit_symbols()

    path = Path("pkg/mod.py")
    caller = _fn(da, name="caller", qual="pkg.mod.caller", path=path)
    state = resolution._ResolutionState(candidates=[], effects=[])
    # Empty candidate sets are valid no-op effects.
    resolution._emit_ambiguity(
        state,
        phase="local_resolution",
        callee_key="target",
        candidates=[],
    )
    assert state.effects == []

    context = resolution.CalleeResolutionContext(
        callee_key="target",
        caller=caller,
        by_name={},
        by_qual={caller.qual: caller},
        symbol_table=None,
        project_root=None,
        class_index=None,
        call=None,
        local_lambda_bindings={},
        caller_module="pkg.mod",
    )
    # Symbol/class resolvers return structural "not resolved" when carriers are absent.
    assert resolution._resolve_symbol_table(context, state) is None
    assert resolution._resolve_class_hierarchy(context, state) is None

    # Unknown operation tags are ignored structurally and keep the outcome unresolved.
    outcome = resolution.apply_callee_resolution_ops(
        context,
        (resolution.ResolutionOp(kind="unknown_op"),),
    )
    assert outcome.resolved is None
