# gabion:ambiguity_boundary_module
from __future__ import annotations

"""Function-index helper boundary during runtime retirement."""


import ast
from collections import defaultdict
from pathlib import Path
from typing import cast

from gabion.analysis.dataflow.engine.dataflow_contracts import CallArgs, FunctionInfo
from gabion.analysis.dataflow.io.dataflow_parse_helpers import (
    _ParseModuleFailure,
    _ParseModuleStage,
    _ParseModuleSuccess,
    _parse_module_tree,
)
from gabion.analysis.foundation.json_types import JSONObject, ParseFailureWitnesses
from gabion.analysis.foundation.timeout_context import check_deadline


def _module_name(path: Path, project_root=None) -> str:
    rel = path.with_suffix("")
    if project_root is not None:
        try:
            rel = rel.relative_to(project_root)
        except ValueError:
            pass
    parts = list(rel.parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    return ".".join(parts)


def _is_test_path(path: Path) -> bool:
    if "tests" in path.parts:
        return True
    return path.name.startswith("test_")


def _node_span(node: ast.AST) -> tuple[int, int, int, int] | None:
    lineno = getattr(node, "lineno", None)
    col = getattr(node, "col_offset", None)
    end_lineno = getattr(node, "end_lineno", None)
    end_col = getattr(node, "end_col_offset", None)
    if not isinstance(lineno, int) or not isinstance(col, int):
        return None
    if not isinstance(end_lineno, int):
        end_lineno = lineno
    if not isinstance(end_col, int):
        end_col = col
    return (lineno - 1, col, end_lineno - 1, end_col)


def _build_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parent_map: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        check_deadline()
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent
    return parent_map


def _enclosing_scopes(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> list[str]:
    scopes: list[str] = []
    current = parents.get(node)
    while current is not None:
        check_deadline()
        if type(current) is ast.ClassDef:
            scopes.append(cast(ast.ClassDef, current).name)
        elif type(current) in {ast.FunctionDef, ast.AsyncFunctionDef}:
            scopes.append(cast(ast.FunctionDef | ast.AsyncFunctionDef, current).name)
        current = parents.get(current)
    return list(reversed(scopes))


def _enclosing_function_scopes(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> list[str]:
    scopes: list[str] = []
    current = parents.get(node)
    while current is not None:
        check_deadline()
        if type(current) in {ast.FunctionDef, ast.AsyncFunctionDef}:
            scopes.append(cast(ast.FunctionDef | ast.AsyncFunctionDef, current).name)
        current = parents.get(current)
    return list(reversed(scopes))


def _enclosing_class(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    current = parents.get(node)
    while current is not None:
        check_deadline()
        if type(current) is ast.ClassDef:
            return cast(ast.ClassDef, current).name
        current = parents.get(current)
    return None


def _collect_functions(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        check_deadline()
        if type(node) in {ast.FunctionDef, ast.AsyncFunctionDef}:
            functions.append(cast(ast.FunctionDef | ast.AsyncFunctionDef, node))
    return sorted(
        functions,
        key=lambda node: (
            int(getattr(node, "lineno", 0) or 0),
            int(getattr(node, "col_offset", 0) or 0),
            node.name,
        ),
    )


def _param_names(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, ignore_params: set[str]
) -> list[str]:
    names = [arg.arg for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs]
    if fn.args.vararg is not None:
        names.append(fn.args.vararg.arg)
    if fn.args.kwarg is not None:
        names.append(fn.args.kwarg.arg)
    if ignore_params:
        names = [name for name in names if name not in ignore_params]
    return names


def _annotation_text(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except (AttributeError, TypeError, ValueError, RecursionError):
        return None


def _param_annotations(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, ignore_params: set[str]
) -> dict[str, str | None]:
    annots: dict[str, str | None] = {}
    for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
        check_deadline()
        if arg.arg in ignore_params:
            continue
        annots[arg.arg] = _annotation_text(arg.annotation)
    if fn.args.vararg is not None and fn.args.vararg.arg not in ignore_params:
        annots[fn.args.vararg.arg] = _annotation_text(fn.args.vararg.annotation)
    if fn.args.kwarg is not None and fn.args.kwarg.arg not in ignore_params:
        annots[fn.args.kwarg.arg] = _annotation_text(fn.args.kwarg.annotation)
    return annots


def _param_defaults(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, ignore_params: set[str]
) -> set[str]:
    defaults: set[str] = set()
    positional = fn.args.posonlyargs + fn.args.args
    if fn.args.defaults:
        offset = len(positional) - len(fn.args.defaults)
        for idx, default in enumerate(fn.args.defaults):
            check_deadline()
            if default is None:
                continue
            param_name = positional[offset + idx].arg
            if param_name not in ignore_params:
                defaults.add(param_name)
    for arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        check_deadline()
        if default is None:
            continue
        if arg.arg not in ignore_params:
            defaults.add(arg.arg)
    return defaults


def _param_spans(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, ignore_params: set[str]
) -> dict[str, tuple[int, int, int, int]]:
    spans: dict[str, tuple[int, int, int, int]] = {}
    for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
        check_deadline()
        if arg.arg in ignore_params:
            continue
        span = _node_span(arg)
        if span is not None:
            spans[arg.arg] = span
    if fn.args.vararg is not None and fn.args.vararg.arg not in ignore_params:
        span = _node_span(fn.args.vararg)
        if span is not None:
            spans[fn.args.vararg.arg] = span
    if fn.args.kwarg is not None and fn.args.kwarg.arg not in ignore_params:
        span = _node_span(fn.args.kwarg)
        if span is not None:
            spans[fn.args.kwarg.arg] = span
    return spans


def _callee_name(call: ast.Call) -> str:
    if type(call.func) is ast.Name:
        return cast(ast.Name, call.func).id
    if type(call.func) is ast.Attribute:
        try:
            return ast.unparse(call.func)
        except (AttributeError, TypeError, ValueError, RecursionError):
            return "<call>"
    try:
        return ast.unparse(call.func)
    except (AttributeError, TypeError, ValueError, RecursionError):
        return "<call>"


def _collect_used_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    used: set[str] = set()

    class _Visitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if type(node.ctx) is ast.Load:
                used.add(node.id)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is fn:
                for stmt in node.body:
                    self.visit(stmt)
                return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if node is fn:
                for stmt in node.body:
                    self.visit(stmt)
                return

        def visit_Lambda(self, _node: ast.Lambda) -> None:
            return

        def visit_ClassDef(self, _node: ast.ClassDef) -> None:
            return

    _Visitor().visit(fn)
    return used


def _collect_calls(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, *, is_test: bool
) -> list[CallArgs]:
    calls: list[CallArgs] = []

    class _Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            check_deadline()
            pos_map: dict[str, str] = {}
            kw_map: dict[str, str] = {}
            const_pos: dict[str, str] = {}
            const_kw: dict[str, str] = {}
            non_const_pos: set[str] = set()
            non_const_kw: set[str] = set()
            star_pos: list[tuple[int, str]] = []
            star_kw: list[str] = []

            for idx, arg in enumerate(node.args):
                check_deadline()
                slot = str(idx)
                if type(arg) is ast.Name:
                    pos_map[slot] = cast(ast.Name, arg).id
                elif type(arg) is ast.Starred:
                    value = cast(ast.Starred, arg).value
                    if type(value) is ast.Name:
                        star_pos.append((idx, cast(ast.Name, value).id))
                    else:
                        non_const_pos.add(slot)
                elif type(arg) is ast.Constant:
                    const_pos[slot] = repr(cast(ast.Constant, arg).value)
                else:
                    non_const_pos.add(slot)

            for keyword in node.keywords:
                check_deadline()
                if keyword.arg is None:
                    value = keyword.value
                    if type(value) is ast.Name:
                        star_kw.append(cast(ast.Name, value).id)
                    continue
                name = str(keyword.arg)
                if type(keyword.value) is ast.Name:
                    kw_map[name] = cast(ast.Name, keyword.value).id
                elif type(keyword.value) is ast.Constant:
                    const_kw[name] = repr(cast(ast.Constant, keyword.value).value)
                else:
                    non_const_kw.add(name)

            calls.append(
                CallArgs(
                    callee=_callee_name(node),
                    pos_map=pos_map,
                    kw_map=kw_map,
                    const_pos=const_pos,
                    const_kw=const_kw,
                    non_const_pos=non_const_pos,
                    non_const_kw=non_const_kw,
                    star_pos=star_pos,
                    star_kw=star_kw,
                    is_test=is_test,
                    span=_node_span(node),
                )
            )
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is fn:
                for stmt in node.body:
                    self.visit(stmt)
                return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if node is fn:
                for stmt in node.body:
                    self.visit(stmt)
                return

        def visit_Lambda(self, _node: ast.Lambda) -> None:
            return

        def visit_ClassDef(self, _node: ast.ClassDef) -> None:
            return

    _Visitor().visit(fn)
    return calls


def _build_function_index(
    paths: list[Path],
    project_root,
    ignore_params: set[str],
    strictness: str,
    transparent_decorators = None,
    *,
    parse_failure_witnesses: ParseFailureWitnesses,
) -> tuple[dict[str, list[FunctionInfo]], dict[str, FunctionInfo]]:
    del strictness
    by_name: defaultdict[str, list[FunctionInfo]] = defaultdict(list)
    by_qual: dict[str, FunctionInfo] = {}
    for path in paths:
        check_deadline()
        parse_outcome = _parse_module_tree(
            path,
            stage=_ParseModuleStage.FUNCTION_INDEX,
            parse_failure_witnesses=parse_failure_witnesses,
        )
        match parse_outcome:
            case _ParseModuleSuccess(kind="parsed", tree=tree):
                parent_map = _build_parent_map(tree)
                module = _module_name(path, project_root)
                is_test = _is_test_path(path)
                for fn in _collect_functions(tree):
                    check_deadline()
                    scopes = _enclosing_scopes(fn, parent_map)
                    lexical_scopes = _enclosing_function_scopes(fn, parent_map)
                    params = _param_names(fn, ignore_params)
                    annots = _param_annotations(fn, ignore_params)
                    defaults = _param_defaults(fn, ignore_params)
                    used = _collect_used_names(fn)
                    positional_params = [arg.arg for arg in fn.args.posonlyargs + fn.args.args]
                    if positional_params and positional_params[0] in {"self", "cls"}:
                        positional_params = positional_params[1:]
                    if ignore_params:
                        positional_params = [
                            name for name in positional_params if name not in ignore_params
                        ]
                    kwonly_params = [
                        arg.arg
                        for arg in fn.args.kwonlyargs
                        if arg.arg not in ignore_params
                    ]
                    vararg = (
                        fn.args.vararg.arg
                        if fn.args.vararg is not None
                        and fn.args.vararg.arg not in ignore_params
                        else None
                    )
                    kwarg = (
                        fn.args.kwarg.arg
                        if fn.args.kwarg is not None
                        and fn.args.kwarg.arg not in ignore_params
                        else None
                    )
                    qual_parts = [module] if module else []
                    qual_parts.extend(scopes)
                    qual_parts.append(fn.name)
                    qual = ".".join(qual_parts)
                    info = FunctionInfo(
                        name=fn.name,
                        qual=qual,
                        path=path,
                        params=params,
                        annots=annots,
                        calls=_collect_calls(fn, is_test=is_test),
                        unused_params={name for name in params if name not in used},
                        defaults=defaults,
                        transparent=True if transparent_decorators is None else True,
                        class_name=_enclosing_class(fn, parent_map),
                        scope=tuple(scopes),
                        lexical_scope=tuple(lexical_scopes),
                        positional_params=tuple(positional_params),
                        kwonly_params=tuple(kwonly_params),
                        vararg=vararg,
                        kwarg=kwarg,
                        param_spans=_param_spans(fn, ignore_params),
                        function_span=_node_span(fn),
                    )
                    by_name[info.name].append(info)
                    by_qual[info.qual] = info
            case _ParseModuleFailure(kind="parse_failure"):
                pass
    return dict(by_name), by_qual

__all__ = [
    "_build_function_index",
    "_enclosing_class",
    "_enclosing_function_scopes",
    "_enclosing_scopes",
    "_is_test_path",
    "_param_names",
    "_param_spans",
]
