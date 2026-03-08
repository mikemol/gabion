from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence, cast


@dataclass(frozen=True)
class LambdaBindingsByCallerDeps:
    check_deadline_fn: Callable[[], None]
    require_not_none_fn: Callable[..., object]
    collect_closure_lambda_factories_fn: Callable[..., dict[str, set[str]]]
    node_span_fn: Callable[..., object]
    enclosing_scopes_fn: Callable[..., tuple[str, ...]]
    target_names_fn: Callable[..., set[str]]
    sort_once_fn: Callable[..., list[object]]


@dataclass(frozen=True)
class ClosureLambdaFactoriesDeps:
    check_deadline_fn: Callable[[], None]
    node_span_fn: Callable[..., object]
    target_names_fn: Callable[..., set[str]]
    enclosing_scopes_fn: Callable[..., tuple[str, ...]]
    function_key_fn: Callable[..., str]


@dataclass(frozen=True)
class CollectLambdaFunctionInfosDeps:
    check_deadline_fn: Callable[[], None]
    node_span_fn: Callable[..., object]
    enclosing_function_scopes_fn: Callable[..., tuple[str, ...]]
    enclosing_scopes_fn: Callable[..., tuple[str, ...]]
    enclosing_class_fn: Callable[..., object]
    synthetic_lambda_name_fn: Callable[..., str]
    function_info_ctor: Callable[..., object]


def collect_lambda_function_infos(
    tree: ast.AST,
    *,
    path: Path,
    module: str,
    parent_map: Mapping[ast.AST, ast.AST],
    ignore_params,
    deps: CollectLambdaFunctionInfosDeps,
) -> list[object]:
    deps.check_deadline_fn()
    lambda_infos: list[object] = []
    for node in ast.walk(tree):
        deps.check_deadline_fn()
        match node:
            case ast.Lambda() as lambda_node:
                span = deps.node_span_fn(lambda_node)
                if span is not None:
                    lexical_scopes = deps.enclosing_function_scopes_fn(lambda_node, parent_map)
                    scopes = deps.enclosing_scopes_fn(lambda_node, parent_map)
                    class_name = deps.enclosing_class_fn(lambda_node, parent_map)
                    synthetic_name = deps.synthetic_lambda_name_fn(
                        module=module,
                        lexical_scope=lexical_scopes,
                        span=span,
                    )
                    qual_parts = [module] if module else []
                    if scopes:
                        qual_parts.extend(scopes)
                    qual_parts.append(synthetic_name)
                    qual = ".".join(qual_parts)
                    params = [
                        arg.arg
                        for arg in (
                            lambda_node.args.posonlyargs
                            + lambda_node.args.args
                            + lambda_node.args.kwonlyargs
                        )
                    ]
                    if ignore_params:
                        params = [name for name in params if name not in ignore_params]
                    lambda_infos.append(
                        deps.function_info_ctor(
                            name=synthetic_name,
                            qual=qual,
                            path=path,
                            params=params,
                            annots={name: None for name in params},
                            calls=[],
                            unused_params=set(),
                            class_name=class_name,
                            scope=tuple(scopes),
                            lexical_scope=tuple(lexical_scopes),
                            positional_params=tuple(params),
                            function_span=span,
                        )
                    )
            case _:
                pass
    return lambda_infos


def collect_closure_lambda_factories(
    tree: ast.AST,
    *,
    module: str,
    parent_map: dict[ast.AST, ast.AST],
    lambda_qual_by_span: Mapping[tuple[int, int, int, int], str],
    deps: ClosureLambdaFactoriesDeps,
) -> dict[str, set[str]]:
    deps.check_deadline_fn()
    factories: dict[str, set[str]] = defaultdict(set)
    for node in ast.walk(tree):
        deps.check_deadline_fn()
        match node:
            case ast.FunctionDef() | ast.AsyncFunctionDef() as function_node:
                local_bindings: dict[str, set[str]] = {}
                for statement in function_node.body:
                    deps.check_deadline_fn()
                    assignment_targets = []
                    assignment_value = None
                    match statement:
                        case ast.Assign(value=value, targets=targets):
                            assignment_targets = list(targets)
                            assignment_value = value
                        case ast.AnnAssign(value=value, target=target):
                            assignment_targets = [target]
                            assignment_value = value
                        case ast.Return(value=ast.Name(id=return_name)):
                            returned = local_bindings.get(return_name, set())
                            if returned:
                                scopes = deps.enclosing_scopes_fn(function_node, parent_map)
                                keys = {function_node.name}
                                if scopes:
                                    keys.add(deps.function_key_fn(scopes, function_node.name))
                                qual_parts = [module] if module else []
                                if scopes:
                                    qual_parts.extend(scopes)
                                qual_parts.append(function_node.name)
                                keys.add(".".join(qual_parts))
                                for key in keys:
                                    deps.check_deadline_fn()
                                    factories[key].update(returned)
                        case _:
                            pass
                    if assignment_value is not None:
                        assigned_quals: set[str] = set()
                        value_span = deps.node_span_fn(assignment_value)
                        match assignment_value:
                            case ast.Lambda():
                                match value_span:
                                    case (
                                        int() as start_line,
                                        int() as start_column,
                                        int() as end_line,
                                        int() as end_column,
                                    ):
                                        qual = lambda_qual_by_span.get(
                                            (start_line, start_column, end_line, end_column)
                                        )
                                        if qual is not None:
                                            assigned_quals.add(qual)
                                    case _:
                                        pass
                            case ast.Name(id=name):
                                assigned_quals.update(local_bindings.get(name, set()))
                            case _:
                                pass
                        for target in assignment_targets:
                            deps.check_deadline_fn()
                            for name in deps.target_names_fn(target):
                                deps.check_deadline_fn()
                                if assigned_quals:
                                    local_bindings[name] = set(assigned_quals)
                                else:
                                    local_bindings.pop(name, None)
            case _:
                pass
    return factories


def collect_lambda_bindings_by_caller(
    tree: ast.AST,
    *,
    module: str,
    parent_map: dict[ast.AST, ast.AST],
    lambda_infos: Sequence[object],
    deps: LambdaBindingsByCallerDeps,
) -> dict[str, dict[str, tuple[str, ...]]]:
    deps.check_deadline_fn()
    lambda_qual_by_span: dict[tuple[int, int, int, int], str] = {}
    for info in lambda_infos:
        function_span = deps.require_not_none_fn(
            getattr(info, "function_span", None),
            reason="lambda function site requires span",
            strict=True,
        )
        lambda_qual_by_span[cast(tuple[int, int, int, int], function_span)] = str(
            getattr(info, "qual")
        )

    binding_sets: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    closure_factories = deps.collect_closure_lambda_factories_fn(
        tree,
        module=module,
        parent_map=parent_map,
        lambda_qual_by_span=lambda_qual_by_span,
    )

    for node in ast.walk(tree):
        deps.check_deadline_fn()
        assignment_node = None
        assignment_targets = []
        assignment_value = None
        match node:
            case ast.Assign(value=value, targets=targets):
                assignment_node = node
                assignment_targets = list(targets)
                assignment_value = value
            case ast.AnnAssign(value=value, target=target):
                assignment_node = node
                assignment_targets = [target]
                assignment_value = value
            case _:
                pass
        if assignment_node is not None and assignment_value is not None:
            fn_scope = deps.enclosing_scopes_fn(assignment_node, parent_map)
            if fn_scope:
                qual_parts = [module] if module else []
                qual_parts.extend(fn_scope)
                caller_key = ".".join(qual_parts)

                assigned_quals: set[str] = set()
                value_span = deps.node_span_fn(assignment_value)
                match assignment_value:
                    case ast.Lambda():
                        match value_span:
                            case (
                                int() as start_line,
                                int() as start_column,
                                int() as end_line,
                                int() as end_column,
                            ):
                                qual = lambda_qual_by_span.get(
                                    (start_line, start_column, end_line, end_column)
                                )
                                if qual is not None:
                                    assigned_quals.add(qual)
                            case _:
                                pass
                    case ast.Name(id=name):
                        assigned_quals.update(binding_sets.get(caller_key, {}).get(name, set()))
                    case ast.Call(func=ast.Name(id=called_name)):
                        assigned_quals.update(closure_factories.get(called_name, set()))
                    case _:
                        pass

                for target in assignment_targets:
                    deps.check_deadline_fn()
                    target_names = list(deps.target_names_fn(target))
                    match target:
                        case ast.Attribute(value=ast.Name(id=target_name), attr=attr):
                            target_names.append(f"{target_name}.{attr}")
                        case _:
                            pass
                    for name in target_names:
                        deps.check_deadline_fn()
                        if assigned_quals:
                            binding_sets[caller_key][name].update(assigned_quals)
                        else:
                            binding_sets[caller_key].pop(name, None)

    out: dict[str, dict[str, tuple[str, ...]]] = {}
    for caller_key, mapping in binding_sets.items():
        deps.check_deadline_fn()
        non_empty = {
            symbol: tuple(
                deps.sort_once_fn(
                    quals,
                    source="gabion.analysis.dataflow_indexed_file_scan._collect_lambda_bindings_by_caller.site_1",
                )
            )
            for symbol, quals in mapping.items()
            if quals
        }
        if non_empty:
            out[caller_key] = non_empty
    return out
