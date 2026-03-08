from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class FunctionIndexAccumulatorDeps:
    check_deadline_fn: Callable[[], None]
    collect_functions_fn: Callable[[ast.Module], list[ast.AST]]
    parent_annotator_ctor: Callable[[], object]
    module_name_fn: Callable[..., str]
    collect_lambda_function_infos_fn: Callable[..., list[object]]
    collect_lambda_bindings_by_caller_fn: Callable[..., dict[str, dict[str, tuple[str, ...]]]]
    direct_lambda_callee_by_call_span_fn: Callable[..., dict[tuple[int, int, int, int], str]]
    collect_return_aliases_fn: Callable[..., dict[str, tuple[list[str], list[str]]]]
    enclosing_class_fn: Callable[..., object]
    enclosing_scopes_fn: Callable[..., list[str]]
    enclosing_function_scopes_fn: Callable[..., list[str]]
    analyze_function_fn: Callable[..., tuple[dict[str, object], list[object]]]
    is_test_path_fn: Callable[[Path], bool]
    materialize_direct_lambda_callees_fn: Callable[..., list[object]]
    unused_params_fn: Callable[..., tuple[set[str], set[str]]]
    decision_surface_reason_map_fn: Callable[..., dict[str, set[str]]]
    value_encoded_decision_params_fn: Callable[..., tuple[set[str], set[str]]]
    param_names_fn: Callable[..., list[str]]
    param_annotations_fn: Callable[..., dict[str, object]]
    param_defaults_fn: Callable[..., set[str]]
    decorators_transparent_fn: Callable[..., bool]
    param_spans_fn: Callable[..., dict[str, tuple[int, int, int, int]]]
    node_span_fn: Callable[[ast.AST], object]
    function_info_ctor: Callable[..., object]


def accumulate_function_index_for_tree(
    acc: object,
    path: Path,
    tree: ast.Module,
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    transparent_decorators,
    deps: FunctionIndexAccumulatorDeps,
) -> None:
    deps.check_deadline_fn()
    funcs = deps.collect_functions_fn(tree)
    if not funcs:
        return
    parents = deps.parent_annotator_ctor()
    parents.visit(tree)
    parent_map = parents.parents
    module = deps.module_name_fn(path, project_root)
    lambda_infos = deps.collect_lambda_function_infos_fn(
        tree,
        path=path,
        module=module,
        parent_map=parent_map,
        ignore_params=ignore_params,
    )
    lambda_bindings_by_caller = deps.collect_lambda_bindings_by_caller_fn(
        tree,
        module=module,
        parent_map=parent_map,
        lambda_infos=lambda_infos,
    )
    lambda_call_nodes: list[ast.Call] = []
    for node in ast.walk(tree):
        match node:
            case ast.Call(func=ast.Lambda()) as lambda_call:
                lambda_call_nodes.append(lambda_call)
            case _:
                pass
    direct_lambda_callee_by_call_span = deps.direct_lambda_callee_by_call_span_fn(
        lambda_call_nodes,
        lambda_infos=lambda_infos,
    )
    return_aliases = deps.collect_return_aliases_fn(
        funcs,
        parent_map,
        ignore_params=ignore_params,
    )
    for fn in funcs:
        deps.check_deadline_fn()
        class_name = deps.enclosing_class_fn(fn, parent_map)
        scopes = deps.enclosing_scopes_fn(fn, parent_map)
        lexical_scopes = deps.enclosing_function_scopes_fn(fn, parent_map)
        use_map, raw_call_args = deps.analyze_function_fn(
            fn,
            parent_map,
            is_test=deps.is_test_path_fn(path),
            ignore_params=ignore_params,
            strictness=strictness,
            class_name=class_name,
            return_aliases=return_aliases,
        )
        call_args = deps.materialize_direct_lambda_callees_fn(
            raw_call_args,
            direct_lambda_callee_by_call_span=direct_lambda_callee_by_call_span,
        )
        unused_params, unknown_key_carriers = deps.unused_params_fn(use_map)
        decision_reason_map = deps.decision_surface_reason_map_fn(fn, ignore_params)
        value_params, value_reasons = deps.value_encoded_decision_params_fn(
            fn, ignore_params
        )
        pos_args = [a.arg for a in (fn.args.posonlyargs + fn.args.args)]
        kwonly_args = [a.arg for a in fn.args.kwonlyargs]
        if pos_args and pos_args[0] in {"self", "cls"}:
            pos_args = pos_args[1:]
        if ignore_params:
            pos_args = [name for name in pos_args if name not in ignore_params]
            kwonly_args = [name for name in kwonly_args if name not in ignore_params]
        vararg = None
        if fn.args.vararg is not None:
            candidate = fn.args.vararg.arg
            if not ignore_params or candidate not in ignore_params:
                vararg = candidate
        kwarg = None
        if fn.args.kwarg is not None:
            candidate = fn.args.kwarg.arg
            if not ignore_params or candidate not in ignore_params:
                kwarg = candidate
        qual_parts = [module] if module else []
        if scopes:
            qual_parts.extend(scopes)
        qual_parts.append(fn.name)
        qual = ".".join(qual_parts)
        info = deps.function_info_ctor(
            name=fn.name,
            qual=qual,
            path=path,
            params=deps.param_names_fn(fn, ignore_params),
            annots=deps.param_annotations_fn(fn, ignore_params),
            defaults=deps.param_defaults_fn(fn, ignore_params),
            calls=call_args,
            unused_params=unused_params,
            unknown_key_carriers=unknown_key_carriers,
            transparent=deps.decorators_transparent_fn(fn, transparent_decorators),
            class_name=class_name,
            scope=tuple(scopes),
            lexical_scope=tuple(lexical_scopes),
            decision_params=set(decision_reason_map),
            decision_surface_reasons=decision_reason_map,
            value_decision_params=value_params,
            value_decision_reasons=value_reasons,
            positional_params=tuple(pos_args),
            kwonly_params=tuple(kwonly_args),
            vararg=vararg,
            kwarg=kwarg,
            param_spans=deps.param_spans_fn(fn, ignore_params),
            function_span=deps.node_span_fn(fn),
            local_lambda_bindings=lambda_bindings_by_caller.get(qual, {}),
        )
        acc.by_name[fn.name].append(info)
        acc.by_qual[info.qual] = info
    for info in lambda_infos:
        deps.check_deadline_fn()
        acc.by_name[info.name].append(info)
        acc.by_qual[info.qual] = info
