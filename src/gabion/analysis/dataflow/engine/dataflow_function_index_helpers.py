# gabion:ambiguity_boundary_module
from __future__ import annotations

"""Function-index helper boundary during runtime retirement."""


import ast
from collections import defaultdict
from functools import singledispatch
from pathlib import Path
from typing import cast

from gabion.analysis.dataflow.engine.dataflow_contracts import CallArgs, FunctionInfo
from gabion.analysis.dataflow.io.dataflow_parse_helpers import (
    ParseModuleFailure,
    ParseModuleStage,
    ParseModuleSuccess,
    parse_module_tree,
)
from gabion.analysis.foundation.json_types import JSONObject, ParseFailureWitnesses
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.foundation.timeout_context import deadline_loop_iter
from gabion.invariants import decision_protocol, grade_boundary, never


@grade_boundary(
    kind="semantic_carrier_adapter",
    name="dataflow_function_index_helpers.module_name",
)
@decision_protocol
def _module_name(path: Path, project_root: Path) -> str:
    rel = path.with_suffix("")
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


_NONE_TYPE = type(None)


def _leaf_ast_subclasses(base_type: type[ast.AST]) -> tuple[type[ast.AST], ...]:
    node_types: list[type[ast.AST]] = []
    for candidate in vars(ast).values():
        try:
            if issubclass(candidate, base_type) and candidate is not base_type:
                node_types.append(candidate)
        except TypeError:
            continue
    node_types_tuple = tuple(node_types)
    return tuple(
        node_type for node_type in node_types_tuple
        if not any(
            candidate is not node_type and issubclass(candidate, node_type)
            for candidate in node_types_tuple
        )
    )


_AST_LEAF_NODE_TYPES = _leaf_ast_subclasses(ast.AST)
_AST_LEAF_EXPR_TYPES = _leaf_ast_subclasses(ast.expr)
_AST_LEAF_EXPR_CONTEXT_TYPES = _leaf_ast_subclasses(ast.expr_context)


@singledispatch
def _int_optional(value: int | None) -> int | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_int_optional.register
def _sd_reg_1(value: int) -> int | None:
    return value


@_int_optional.register(_NONE_TYPE)
def _sd_reg_2(value: None) -> int | None:
    _ = value
    return None


@singledispatch
def _lexical_scope_name_optional(node: ast.AST) -> str | None:
    never("unregistered runtime type", value_type=type(node).__name__)


@_lexical_scope_name_optional.register(ast.ClassDef)
def _sd_reg_3(node: ast.ClassDef) -> str | None:
    return node.name


@_lexical_scope_name_optional.register(ast.FunctionDef)
def _sd_reg_4(node: ast.FunctionDef) -> str | None:
    return node.name


@_lexical_scope_name_optional.register(ast.AsyncFunctionDef)
def _sd_reg_5(node: ast.AsyncFunctionDef) -> str | None:
    return node.name


def _scope_name_none(_node: ast.AST) -> str | None:
    return None


for _runtime_type in _AST_LEAF_NODE_TYPES:
    if _runtime_type in {ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef}:
        continue
    _lexical_scope_name_optional.register(_runtime_type)(_scope_name_none)


@singledispatch
def _function_scope_name_optional(node: ast.AST) -> str | None:
    never("unregistered runtime type", value_type=type(node).__name__)


@_function_scope_name_optional.register(ast.FunctionDef)
def _sd_reg_6(node: ast.FunctionDef) -> str | None:
    return node.name


@_function_scope_name_optional.register(ast.AsyncFunctionDef)
def _sd_reg_7(node: ast.AsyncFunctionDef) -> str | None:
    return node.name


for _runtime_type in _AST_LEAF_NODE_TYPES:
    if _runtime_type in {ast.FunctionDef, ast.AsyncFunctionDef}:
        continue
    _function_scope_name_optional.register(_runtime_type)(_scope_name_none)


@singledispatch
def _class_scope_name_optional(node: ast.AST) -> str | None:
    never("unregistered runtime type", value_type=type(node).__name__)


@_class_scope_name_optional.register(ast.ClassDef)
def _sd_reg_8(node: ast.ClassDef) -> str | None:
    return node.name


for _runtime_type in _AST_LEAF_NODE_TYPES:
    if _runtime_type is ast.ClassDef:
        continue
    _class_scope_name_optional.register(_runtime_type)(_scope_name_none)


def _unparse_or_default(node: ast.AST, default: str) -> str:
    try:
        return ast.unparse(node)
    except (AttributeError, TypeError, ValueError, RecursionError):
        return default


@singledispatch
def _callee_name_from_expr(node: ast.expr) -> str:
    never("unregistered runtime type", value_type=type(node).__name__)


@_callee_name_from_expr.register(ast.Name)
def _sd_reg_9(node: ast.Name) -> str:
    return node.id


@_callee_name_from_expr.register(ast.Attribute)
def _sd_reg_10(node: ast.Attribute) -> str:
    return _unparse_or_default(node, "<call>")


def _callee_name_expr_fallback(node: ast.expr) -> str:
    return _unparse_or_default(node, "<call>")


for _runtime_type in _AST_LEAF_EXPR_TYPES:
    if _runtime_type in {ast.Name, ast.Attribute}:
        continue
    _callee_name_from_expr.register(_runtime_type)(_callee_name_expr_fallback)


@singledispatch
def _name_id_optional(node: ast.expr) -> str | None:
    never("unregistered runtime type", value_type=type(node).__name__)


@_name_id_optional.register(ast.Name)
def _sd_reg_11(node: ast.Name) -> str | None:
    return node.id


def _name_id_none(_node: ast.expr) -> str | None:
    return None


for _runtime_type in _AST_LEAF_EXPR_TYPES:
    if _runtime_type is ast.Name:
        continue
    _name_id_optional.register(_runtime_type)(_name_id_none)


@singledispatch
def _is_load_context(node: ast.expr_context) -> bool:
    never("unregistered runtime type", value_type=type(node).__name__)


@_is_load_context.register(ast.Load)
def _sd_reg_12(node: ast.Load) -> bool:
    _ = node
    return True


def _false_context(_node: ast.expr_context) -> bool:
    return False


for _runtime_type in _AST_LEAF_EXPR_CONTEXT_TYPES:
    if _runtime_type is ast.Load:
        continue
    _is_load_context.register(_runtime_type)(_false_context)


@singledispatch
def _route_call_arg(
    arg: ast.expr,
    *,
    slot: str,
    index: int,
    pos_map: dict[str, str],
    const_pos: dict[str, str],
    non_const_pos: set[str],
    star_pos: list[tuple[int, str]],
) -> None:
    never("unregistered runtime type", value_type=type(arg).__name__)


@_route_call_arg.register(ast.Name)
def _sd_reg_13(
    arg: ast.Name,
    *,
    slot: str,
    index: int,
    pos_map: dict[str, str],
    const_pos: dict[str, str],
    non_const_pos: set[str],
    star_pos: list[tuple[int, str]],
) -> None:
    _ = index, const_pos, non_const_pos, star_pos
    pos_map[slot] = arg.id


@_route_call_arg.register(ast.Starred)
def _sd_reg_14(
    arg: ast.Starred,
    *,
    slot: str,
    index: int,
    pos_map: dict[str, str],
    const_pos: dict[str, str],
    non_const_pos: set[str],
    star_pos: list[tuple[int, str]],
) -> None:
    _ = slot, pos_map, const_pos
    starred_name = _name_id_optional(arg.value)
    if starred_name is not None:
        star_pos.append((index, starred_name))
        return
    non_const_pos.add(slot)


@_route_call_arg.register(ast.Constant)
def _sd_reg_15(
    arg: ast.Constant,
    *,
    slot: str,
    index: int,
    pos_map: dict[str, str],
    const_pos: dict[str, str],
    non_const_pos: set[str],
    star_pos: list[tuple[int, str]],
) -> None:
    _ = index, pos_map, non_const_pos, star_pos
    const_pos[slot] = repr(arg.value)


def _route_call_arg_non_const(
    arg: ast.expr,
    *,
    slot: str,
    index: int,
    pos_map: dict[str, str],
    const_pos: dict[str, str],
    non_const_pos: set[str],
    star_pos: list[tuple[int, str]],
) -> None:
    _ = arg, index, pos_map, const_pos, star_pos
    non_const_pos.add(slot)


for _runtime_type in _AST_LEAF_EXPR_TYPES:
    if _runtime_type in {ast.Name, ast.Starred, ast.Constant}:
        continue
    _route_call_arg.register(_runtime_type)(_route_call_arg_non_const)


@singledispatch
def _route_keyword_value(
    value: ast.expr,
    *,
    name: str,
    kw_map: dict[str, str],
    const_kw: dict[str, str],
    non_const_kw: set[str],
) -> None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_route_keyword_value.register(ast.Name)
def _sd_reg_16(
    value: ast.Name,
    *,
    name: str,
    kw_map: dict[str, str],
    const_kw: dict[str, str],
    non_const_kw: set[str],
) -> None:
    _ = const_kw, non_const_kw
    kw_map[name] = value.id


@_route_keyword_value.register(ast.Constant)
def _sd_reg_17(
    value: ast.Constant,
    *,
    name: str,
    kw_map: dict[str, str],
    const_kw: dict[str, str],
    non_const_kw: set[str],
) -> None:
    _ = kw_map, non_const_kw
    const_kw[name] = repr(value.value)


def _route_keyword_value_non_const(
    value: ast.expr,
    *,
    name: str,
    kw_map: dict[str, str],
    const_kw: dict[str, str],
    non_const_kw: set[str],
) -> None:
    _ = value, kw_map, const_kw
    non_const_kw.add(name)


for _runtime_type in _AST_LEAF_EXPR_TYPES:
    if _runtime_type in {ast.Name, ast.Constant}:
        continue
    _route_keyword_value.register(_runtime_type)(_route_keyword_value_non_const)


def _node_span(node: ast.AST) -> tuple[int, int, int, int] | None:
    lineno = _int_optional(getattr(node, "lineno", None))
    col = _int_optional(getattr(node, "col_offset", None))
    end_lineno = _int_optional(getattr(node, "end_lineno", None))
    end_col = _int_optional(getattr(node, "end_col_offset", None))
    if lineno is None or col is None:
        return None
    if end_lineno is None:
        end_lineno = lineno
    if end_col is None:
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
        scope_name = _lexical_scope_name_optional(current)
        if scope_name is not None:
            scopes.append(scope_name)
        current = parents.get(current)
    return list(reversed(scopes))


def _enclosing_function_scopes(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> list[str]:
    scopes: list[str] = []
    current = parents.get(node)
    while current is not None:
        check_deadline()
        scope_name = _function_scope_name_optional(current)
        if scope_name is not None:
            scopes.append(scope_name)
        current = parents.get(current)
    return list(reversed(scopes))


def _enclosing_class(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    current = parents.get(node)
    while current is not None:
        check_deadline()
        scope_name = _class_scope_name_optional(current)
        if scope_name is not None:
            return scope_name
        current = parents.get(current)
    return None


def _collect_functions(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in deadline_loop_iter(ast.walk(tree)):
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
    return _callee_name_from_expr(call.func)


def _collect_used_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    used: set[str] = set()

    class _Visitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if _is_load_context(node.ctx):
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
                _route_call_arg(
                    arg,
                    slot=slot,
                    index=idx,
                    pos_map=pos_map,
                    const_pos=const_pos,
                    non_const_pos=non_const_pos,
                    star_pos=star_pos,
                )

            for keyword in node.keywords:
                check_deadline()
                if keyword.arg is None:
                    starred_kw_name = _name_id_optional(keyword.value)
                    if starred_kw_name is not None:
                        star_kw.append(starred_kw_name)
                    continue
                name = str(keyword.arg)
                _route_keyword_value(
                    keyword.value,
                    name=name,
                    kw_map=kw_map,
                    const_kw=const_kw,
                    non_const_kw=non_const_kw,
                )

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
    project_root: Path,
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
        parse_outcome = parse_module_tree(
            path,
            stage=ParseModuleStage.FUNCTION_INDEX,
            parse_failure_witnesses=parse_failure_witnesses,
        )
        match parse_outcome:
            case ParseModuleSuccess(kind="parsed", tree=tree):
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
            case ParseModuleFailure(kind="parse_failure"):
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
