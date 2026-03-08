from __future__ import annotations

import ast
from collections.abc import Callable
from typing import cast


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
            if type(expr) is ast.Call:
                func = cast(ast.Call, expr).func
                if type(func) is ast.Name:
                    return cast(ast.Name, func).id == "deadline_loop_iter"
                if type(func) is ast.Attribute:
                    return cast(ast.Attribute, func).attr == "deadline_loop_iter"
            return False

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
            func = node.func
            func_type = type(func)
            if func_type is ast.Attribute:
                attribute_func = cast(ast.Attribute, func)
                if attribute_func.attr == "deadline_loop_iter":
                    self._mark_ambient_check()
                if (
                    attribute_func.attr in deadline_check_methods
                    and type(attribute_func.value) is ast.Name
                    and cast(ast.Name, attribute_func.value).id in self._params
                ):
                    self._mark_param_check(cast(ast.Name, attribute_func.value).id)
                if attribute_func.attr == "check_deadline" and node.args:
                    first = node.args[0]
                    if type(first) is ast.Name and cast(ast.Name, first).id in self._params:
                        self._mark_param_check(cast(ast.Name, first).id)
                if attribute_func.attr in {"check_deadline", "require_deadline"} and not node.args:
                    self._mark_ambient_check()
            elif func_type is ast.Name:
                name_func = cast(ast.Name, func)
                if name_func.id == "deadline_loop_iter":
                    self._mark_ambient_check()
                if name_func.id == "check_deadline" and node.args:
                    first = node.args[0]
                    if type(first) is ast.Name and cast(ast.Name, first).id in self._params:
                        self._mark_param_check(cast(ast.Name, first).id)
                if name_func.id in {"check_deadline", "require_deadline"} and not node.args:
                    self._mark_ambient_check()
            self.generic_visit(node)

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
