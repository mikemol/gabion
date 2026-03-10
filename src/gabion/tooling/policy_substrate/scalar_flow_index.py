#!/usr/bin/env python3
from __future__ import annotations

import ast

_SCALAR_CASTS = {"str", "int", "float", "bool", "bytes", "repr"}
_KIND_STRING = "string"
_KIND_NON_STRING = "non_string"
_KIND_UNKNOWN = "unknown"


class ScalarFlowIndex(ast.NodeVisitor):
    def __init__(self) -> None:
        self._symbol_scopes: list[dict[str, str]] = [dict()]
        self._function_return_kinds: dict[str, str] = {}
        self._string_add_nodes: set[int] = set()
        self._string_join_call_nodes: set[int] = set()
        self._reduce_call_nodes: set[int] = set()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scoped(node=node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._register_function_return_kind(node=node)
        self._visit_scoped(node=node, initial_symbols=self._function_argument_kinds(arguments=node.args))

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._register_function_return_kind(node=node)
        self._visit_scoped(node=node, initial_symbols=self._function_argument_kinds(arguments=node.args))

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self._bind_target_from_value(target=target, value=node.value)
        for target in node.targets:
            self.visit(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
            value_kind = self._infer_expr_kind(node.value)
        else:
            value_kind = _KIND_UNKNOWN
        annotation_kind = _annotation_kind(node.annotation)
        assigned_kind = value_kind if value_kind != _KIND_UNKNOWN else annotation_kind
        self._bind_target_kind(target=node.target, kind=assigned_kind)
        self.visit(node.annotation)
        if not node.simple:
            self.visit(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        assigned_kind = self._infer_expr_kind(node.value)
        self._bind_target_kind(target=node.target, kind=assigned_kind)
        self.visit(node.target)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if self._is_string_add(node):
            self._string_add_nodes.add(id(node))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_string_join_call(node):
            self._string_join_call_nodes.add(id(node))
        if self._is_reduce_call(node):
            self._reduce_call_nodes.add(id(node))
        self.generic_visit(node)

    def is_string_add(self, *, node: ast.BinOp) -> bool:
        return id(node) in self._string_add_nodes

    def is_string_join_call(self, *, node: ast.Call) -> bool:
        return id(node) in self._string_join_call_nodes

    def is_reduce_call(self, *, node: ast.Call) -> bool:
        return id(node) in self._reduce_call_nodes

    def _visit_scoped(
        self,
        *,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        initial_symbols: dict[str, str] | None = None,
    ) -> None:
        self._symbol_scopes.append(dict(initial_symbols or {}))
        self.generic_visit(node)
        self._symbol_scopes.pop()

    def _function_argument_kinds(self, *, arguments: ast.arguments) -> dict[str, str]:
        symbols: dict[str, str] = {}
        for arg in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]:
            kind = _annotation_kind(arg.annotation)
            if kind != _KIND_UNKNOWN:
                symbols[arg.arg] = kind
        if arguments.vararg is not None:
            kind = _annotation_kind(arguments.vararg.annotation)
            if kind != _KIND_UNKNOWN:
                symbols[arguments.vararg.arg] = kind
        if arguments.kwarg is not None:
            kind = _annotation_kind(arguments.kwarg.annotation)
            if kind != _KIND_UNKNOWN:
                symbols[arguments.kwarg.arg] = kind
        return symbols

    def _register_function_return_kind(self, *, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        return_kind = _annotation_kind(node.returns)
        if return_kind != _KIND_UNKNOWN:
            self._function_return_kinds[node.name] = return_kind

    def _bind_target_from_value(self, *, target: ast.AST, value: ast.AST) -> None:
        match target, value:
            case (ast.Tuple(elts=target_elts) | ast.List(elts=target_elts), ast.Tuple(elts=value_elts) | ast.List(elts=value_elts)) if len(target_elts) == len(value_elts):
                for nested_target, nested_value in zip(target_elts, value_elts, strict=True):
                    self._bind_target_from_value(target=nested_target, value=nested_value)
            case _:
                self._bind_target_kind(target=target, kind=self._infer_expr_kind(value))

    def _bind_target_kind(self, *, target: ast.AST, kind: str) -> None:
        match target:
            case ast.Name(id=name):
                self._symbol_scopes[-1][name] = kind
            case ast.Tuple(elts=elts) | ast.List(elts=elts):
                for nested_target in elts:
                    self._bind_target_kind(target=nested_target, kind=_KIND_UNKNOWN)
            case _:
                return

    def _lookup_symbol_kind(self, *, name: str) -> str:
        for scope in reversed(self._symbol_scopes):
            kind = scope.get(name)
            if kind is not None:
                return kind
        return _KIND_UNKNOWN

    def _infer_expr_kind(self, node: ast.AST) -> str:
        match node:
            case ast.Constant(value=value):
                if isinstance(value, str):
                    return _KIND_STRING
                return _KIND_NON_STRING
            case ast.JoinedStr():
                return _KIND_STRING
            case ast.Name(id=name):
                return self._lookup_symbol_kind(name=name)
            case ast.Tuple() | ast.List() | ast.Set() | ast.Dict():
                return _KIND_NON_STRING
            case ast.Call(func=func):
                conversion = scalar_cast_name(func)
                if conversion in {"str", "repr"}:
                    return _KIND_STRING
                if conversion in {"int", "float", "bool", "bytes"}:
                    return _KIND_NON_STRING
                if is_dunder_str_call(func=func):
                    return _KIND_STRING
                if is_string_format_call(func=func):
                    return _KIND_STRING
                if self._is_join_call(func=func):
                    return _KIND_STRING
                return self._infer_call_result_kind(func=func)
            case ast.BinOp(op=ast.Add()):
                left_kind = self._infer_expr_kind(node.left)
                right_kind = self._infer_expr_kind(node.right)
                if left_kind == _KIND_STRING or right_kind == _KIND_STRING:
                    return _KIND_STRING
                if left_kind == _KIND_NON_STRING and right_kind == _KIND_NON_STRING:
                    return _KIND_NON_STRING
                return _KIND_UNKNOWN
            case _:
                return _KIND_UNKNOWN

    def _infer_call_result_kind(self, *, func: ast.AST) -> str:
        match func:
            case ast.Name(id=name):
                return self._function_return_kinds.get(name, _KIND_UNKNOWN)
            case _:
                return _KIND_UNKNOWN

    def _is_string_add(self, node: ast.BinOp) -> bool:
        if not isinstance(node.op, ast.Add):
            return False
        left_kind = self._infer_expr_kind(node.left)
        right_kind = self._infer_expr_kind(node.right)
        return left_kind == _KIND_STRING or right_kind == _KIND_STRING

    def _is_string_join_call(self, node: ast.Call) -> bool:
        if not self._is_join_call(func=node.func):
            return False
        match node.func:
            case ast.Attribute(value=receiver):
                receiver_kind = self._infer_expr_kind(receiver)
                return receiver_kind == _KIND_STRING
            case _:
                return False

    def _is_reduce_call(self, node: ast.Call) -> bool:
        return _is_reduce_call(func=node.func)

    def _is_join_call(self, *, func: ast.AST) -> bool:
        return _is_join_call(func=func)


def build_scalar_flow_index(*, tree: ast.AST) -> ScalarFlowIndex:
    index = ScalarFlowIndex()
    index.visit(tree)
    return index


def scalar_cast_name(func: ast.AST) -> str:
    match func:
        case ast.Name(id=name) if name in _SCALAR_CASTS:
            return name
        case _:
            return ""


def is_string_format_call(*, func: ast.AST) -> bool:
    return _is_format_call(func=func) or _is_builtin_format_call(func=func)


def is_dunder_str_call(*, func: ast.AST) -> bool:
    match func:
        case ast.Attribute(attr="__str__"):
            return True
        case _:
            return False


def is_join_call(*, func: ast.AST) -> bool:
    return _is_join_call(func=func)


def is_reduce_call(*, func: ast.AST) -> bool:
    return _is_reduce_call(func=func)


def _is_format_call(*, func: ast.AST) -> bool:
    match func:
        case ast.Attribute(attr="format", value=ast.Constant(value=value)) if isinstance(value, str):
            return True
        case _:
            return False


def _is_builtin_format_call(*, func: ast.AST) -> bool:
    match func:
        case ast.Name(id="format"):
            return True
        case _:
            return False


def _is_join_call(*, func: ast.AST) -> bool:
    match func:
        case ast.Attribute(attr="join"):
            return True
        case _:
            return False


def _is_reduce_call(*, func: ast.AST) -> bool:
    match func:
        case ast.Name(id="reduce"):
            return True
        case ast.Attribute(attr="reduce"):
            return True
        case _:
            return False


def _annotation_kind(annotation: ast.AST | None) -> str:
    match annotation:
        case ast.Name(id="str"):
            return _KIND_STRING
        case ast.Constant(value=value):
            if value == "str":
                return _KIND_STRING
            return _KIND_UNKNOWN
        case ast.Subscript(value=ast.Name(id="str")):
            return _KIND_STRING
        case _:
            return _KIND_UNKNOWN


__all__ = [
    "ScalarFlowIndex",
    "build_scalar_flow_index",
    "is_dunder_str_call",
    "is_join_call",
    "is_reduce_call",
    "is_string_format_call",
    "scalar_cast_name",
]
