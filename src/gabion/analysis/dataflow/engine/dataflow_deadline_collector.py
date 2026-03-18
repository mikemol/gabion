# gabion:ambiguity_boundary_module
# gabion:boundary_normalization_module
# gabion:grade_boundary kind=semantic_carrier_adapter name=dataflow_deadline_collector
from __future__ import annotations

import ast
from dataclasses import dataclass
from collections.abc import Callable

from gabion.invariants import never


def _is_deadline_loop_iter_callee(func: ast.AST) -> bool:
    match func:
        case ast.Name(id="deadline_loop_iter") | ast.Attribute(attr="deadline_loop_iter"):
            return True
        case ast.AST():
            return False
        case _:
            never("deadline callee classifier must receive AST nodes")


@dataclass(frozen=True)
class _DeadlineCallMark:
    ambient_check: bool = False
    checked_params: tuple[str, ...] = ()


def _classify_deadline_call(
    *,
    func: ast.AST,
    args: list[ast.AST],
    params: frozenset[str],
    deadline_check_methods: frozenset[str],
) -> _DeadlineCallMark:
    match func:
        case ast.Attribute(attr=attr, value=value):
            checked_params: list[str] = []
            if attr in deadline_check_methods:
                match value:
                    case ast.Name(id=owner_name) if owner_name in params:
                        checked_params.append(owner_name)
            if attr == "check_deadline" and args:
                match args[0]:
                    case ast.Name(id=first_name) if first_name in params:
                        checked_params.append(first_name)
            return _DeadlineCallMark(
                ambient_check=(attr == "deadline_loop_iter")
                or (attr in {"check_deadline", "require_deadline"} and not args),
                checked_params=tuple(checked_params),
            )
        case ast.Name(id=func_name):
            checked_params: tuple[str, ...] = ()
            if func_name == "check_deadline" and args:
                match args[0]:
                    case ast.Name(id=first_name) if first_name in params:
                        checked_params = (first_name,)
            return _DeadlineCallMark(
                ambient_check=(func_name == "deadline_loop_iter")
                or (func_name in {"check_deadline", "require_deadline"} and not args),
                checked_params=checked_params,
            )
        case ast.AST():
            return _DeadlineCallMark()
        case _:
            never("deadline call classifier must receive AST nodes")


# gabion:grade_boundary kind=semantic_carrier_adapter name=dataflow_deadline_collector._apply_deadline_call_mark
def _apply_deadline_call_mark(
    *,
    collector: object,
    node: ast.Call,
    params: set[str],
    deadline_check_methods: frozenset[str],
) -> None:
    collector._record_call_span(node)
    call_mark = _classify_deadline_call(
        func=node.func,
        args=node.args,
        params=params,
        deadline_check_methods=deadline_check_methods,
    )
    if call_mark.ambient_check:
        if collector._loop_stack:
            collector._loop_stack[-1].ambient_check = True
        else:
            collector.ambient_check = True
    for name in call_mark.checked_params:
        if collector._loop_stack:
            collector._loop_stack[-1].check_params.add(name)
        else:
            collector.check_params.add(name)


def make_deadline_function_collector(
    *,
    node_span_fn: Callable[[ast.AST], object],
    check_deadline_fn: Callable[[], None],
    deadline_loop_facts_ctor: Callable[..., object],
):
    deadline_check_methods = frozenset({"check", "expired"})

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

        # gabion:grade_boundary kind=semantic_carrier_adapter name=dataflow_deadline_collector.visit_Call
        def visit_Call(self, node: ast.Call) -> None:
            _apply_deadline_call_mark(
                collector=self,
                node=node,
                params=self._params,
                deadline_check_methods=deadline_check_methods,
            )
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
