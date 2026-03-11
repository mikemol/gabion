from __future__ import annotations

import ast
from collections.abc import Callable
from functools import singledispatch, singledispatchmethod

from gabion.invariants import never
from gabion.runtime_shape_dispatch import str_optional


@singledispatch
def _is_deadline_loop_iter_callee(func: ast.AST) -> bool:
    never("unregistered runtime type", value_type=type(func).__name__)


@_is_deadline_loop_iter_callee.register(ast.AST)
def _is_deadline_loop_iter_callee_default(func: ast.AST) -> bool:
    del func
    return False


@_is_deadline_loop_iter_callee.register(ast.Name)
def _is_deadline_loop_iter_callee_name(func: ast.Name) -> bool:
    return func.id == "deadline_loop_iter"


@_is_deadline_loop_iter_callee.register(ast.Attribute)
def _is_deadline_loop_iter_callee_attribute(func: ast.Attribute) -> bool:
    return func.attr == "deadline_loop_iter"


def make_deadline_function_collector(
    *,
    node_span_fn: Callable[[ast.AST], object],
    check_deadline_fn: Callable[[], None],
    deadline_loop_facts_ctor: Callable[..., object],
):
    deadline_check_methods = {"check", "expired"}

    class _DeadlineFunctionCollector(ast.NodeVisitor):
        def __init__(self, root: ast.AST, params: set[str]) -> None:
            self._root = root
            self._params = params
            self.loop = False
            self.check_params: set[str] = set()
            self.ambient_check = False
            self.loop_sites: list[object] = []
            self._loop_stack: list[object] = []
            self.assignments: list[tuple[list[ast.AST], object, object]] = []

        def _mark_param_check(self, name: str) -> None:
            if self._loop_stack:
                self._loop_stack[-1].check_params.add(name)
            else:
                self.check_params.add(name)

        def _mark_ambient_check(self) -> None:
            if self._loop_stack:
                self._loop_stack[-1].ambient_check = True
            else:
                self.ambient_check = True

        def _record_call_span(self, node: ast.AST) -> None:
            if self._loop_stack:
                span = node_span_fn(node)
                if span is not None:
                    self._loop_stack[-1].call_spans.add(span)

        def _iter_marks_ambient(self, expr: ast.AST) -> bool:
            match expr:
                case ast.Call(func=func):
                    return _is_deadline_loop_iter_callee(func)
                case _:
                    return False

                    never("unreachable wildcard match fall-through")
        def _visit_loop_body(
            self,
            node: ast.AST,
            kind: str,
            *,
            ambient_check: bool = False,
        ) -> None:
            self.loop = True
            loop_fact = deadline_loop_facts_ctor(
                span=node_span_fn(node),
                kind=kind,
                depth=len(self._loop_stack) + 1,
                ambient_check=ambient_check,
            )
            self._loop_stack.append(loop_fact)
            for stmt in getattr(node, "body", []):
                check_deadline_fn()
                self.visit(stmt)
            self._loop_stack.pop()
            self.loop_sites.append(loop_fact)
            for stmt in getattr(node, "orelse", []):
                check_deadline_fn()
                self.visit(stmt)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is not self._root:
                return
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if node is not self._root:
                return
            self.generic_visit(node)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            del node
            return

        def visit_For(self, node: ast.For) -> None:
            self.loop = True
            ambient_check = self._iter_marks_ambient(node.iter)
            self.visit(node.target)
            self.visit(node.iter)
            self._visit_loop_body(node, "for", ambient_check=ambient_check)

        def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
            self.loop = True
            ambient_check = self._iter_marks_ambient(node.iter)
            self.visit(node.target)
            self.visit(node.iter)
            self._visit_loop_body(node, "async_for", ambient_check=ambient_check)

        def visit_While(self, node: ast.While) -> None:
            self.loop = True
            self.visit(node.test)
            self._visit_loop_body(node, "while")

        def visit_Call(self, node: ast.Call) -> None:
            self._record_call_span(node)
            self._mark_deadline_call(node.func, node.args)
            self.generic_visit(node)

        @singledispatchmethod
        def _mark_deadline_call(self, func: ast.AST, args: list[ast.AST]) -> None:
            never("unregistered runtime type", value_type=type(func).__name__)

        @_mark_deadline_call.register(ast.AST)
        def _mark_deadline_call_default(self, func: ast.AST, args: list[ast.AST]) -> None:
            del func, args
            return

        @_mark_deadline_call.register(ast.Attribute)
        def _mark_deadline_call_attribute(
            self,
            func: ast.Attribute,
            args: list[ast.AST],
        ) -> None:
            if func.attr == "deadline_loop_iter":
                self._mark_ambient_check()

            owner_name = str_optional(func.value)
            if owner_name is not None and func.attr in deadline_check_methods and owner_name in self._params:
                self._mark_param_check(owner_name)

            if func.attr == "check_deadline":
                first_name = str_optional(args[0]) if args else None
                if first_name is not None and first_name in self._params:
                    self._mark_param_check(first_name)

            if func.attr in {"check_deadline", "require_deadline"} and not args:
                self._mark_ambient_check()

        @_mark_deadline_call.register(ast.Name)
        def _mark_deadline_call_name(self, func: ast.Name, args: list[ast.AST]) -> None:
            if func.id == "deadline_loop_iter":
                self._mark_ambient_check()

            if func.id == "check_deadline":
                first_name = str_optional(args[0]) if args else None
                if first_name is not None and first_name in self._params:
                    self._mark_param_check(first_name)

            if func.id in {"check_deadline", "require_deadline"} and not args:
                self._mark_ambient_check()

        def visit_Assign(self, node: ast.Assign) -> None:
            self.assignments.append((node.targets, node.value, node_span_fn(node)))
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            self.assignments.append(([node.target], node.value, node_span_fn(node)))
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            self.assignments.append(([node.target], node.value, node_span_fn(node)))
            self.generic_visit(node)

    return _DeadlineFunctionCollector


__all__ = ["make_deadline_function_collector"]
