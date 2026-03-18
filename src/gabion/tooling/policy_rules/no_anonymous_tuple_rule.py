#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass

from gabion.analysis.foundation.event_algebra import CanonicalRunContext
from gabion.tooling.policy_substrate.aspf_union_view import build_aspf_union_view
from gabion.tooling.policy_substrate.rule_runtime import (
    cst_failure_seeds,
    decorate_failure,
    decorate_site,
    new_run_context,
)
from gabion.tooling.policy_rules.fiber_diagnostics import (
    FiberApplicabilityBounds,
    FiberCounterfactualBoundary,
    FiberTraceEvent,
)
from gabion.tooling.runtime.policy_scan_batch import (
    PolicyScanBatch,
    ScanFailureSeed,
    iter_failure_seeds,
)

RULE_NAME = "no_anonymous_tuple"


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    column: int
    qualname: str
    kind: str
    message: str
    input_slot: str
    flow_identity: str
    fiber_trace: tuple[FiberTraceEvent, ...]
    applicability_bounds: FiberApplicabilityBounds
    counterfactual_boundary: FiberCounterfactualBoundary
    fiber_id: str
    taint_interval_id: str
    condition_overlap_id: str
    structured_hash: str

    @property
    def key(self) -> str:
        return f"{self.path}:{self.qualname}:{self.kind}:{self.structured_hash}"

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: {self.kind}: {self.message}"


class _AnonymousTupleVisitor(ast.NodeVisitor):
    def __init__(self, *, rel_path: str, run_context: CanonicalRunContext) -> None:
        self._path = rel_path
        self._run_context = run_context
        self._scope_parts: list[str] = []
        self.violations: list[Violation] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scoped(node=node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scoped(node=node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scoped(node=node)

    def visit_Return(self, node: ast.Return) -> None:
        if isinstance(node.value, ast.Tuple):
            self._record(
                node=node,
                kind="tuple_return",
                message="anonymous tuple return; define a DTO with named fields",
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Tuple):
            self._record(
                node=node,
                kind="tuple_assignment",
                message="anonymous tuple assignment; define a DTO with named fields",
            )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.value, ast.Tuple):
            self._record(
                node=node,
                kind="tuple_assignment",
                message="anonymous tuple assignment; define a DTO with named fields",
            )
        self.generic_visit(node)

    def visit_Yield(self, node: ast.Yield) -> None:
        if isinstance(node.value, ast.Tuple):
            self._record(
                node=node,
                kind="tuple_yield",
                message="anonymous tuple yield; define a DTO with named fields",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_tuple_constructor(node.func):
            self._record(
                node=node,
                kind="tuple_constructor",
                message="tuple(...) constructor creates anonymous tuple shape; define a DTO",
            )
        self.violations.extend(
            _tuple_argument_violations(
                path=self._path,
                qualname=self._qualname(),
                run_context=self._run_context,
                call_node=node,
            )
        )
        self.generic_visit(node)

    def _visit_scoped(self, *, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._scope_parts.append(node.name)
        self.generic_visit(node)
        self._scope_parts.pop()

    def _qualname(self) -> str:
        if not self._scope_parts:
            return "<module>"
        return ".".join(self._scope_parts)

    def _record(self, *, node: ast.AST, kind: str, message: str) -> None:
        line = int(getattr(node, "lineno", 1) or 1)
        column = int(getattr(node, "col_offset", 0) or 0) + 1
        self.violations.append(
            _violation(
                run_context=self._run_context,
                rel_path=self._path,
                qualname=self._qualname(),
                line=line,
                column=column,
                kind=kind,
                message=message,
            )
        )


def collect_violations(
    *,
    batch: PolicyScanBatch,
    run_context: CanonicalRunContext | None = None,
) -> list[Violation]:
    context = run_context if run_context is not None else new_run_context(rule_name=RULE_NAME)
    union_view = build_aspf_union_view(batch=batch)
    violations: list[Violation] = []
    for seed in (*iter_failure_seeds(batch=batch), *cst_failure_seeds(union_view=union_view)):
        violations.append(_failure_violation(run_context=context, seed=seed))

    for module in union_view.modules:
        visitor = _AnonymousTupleVisitor(rel_path=module.rel_path, run_context=context)
        visitor.visit(module.pyast_tree)
        violations.extend(visitor.violations)
    return violations


def _tuple_argument_violations(
    *,
    path: str,
    qualname: str,
    run_context: CanonicalRunContext,
    call_node: ast.Call,
) -> tuple[Violation, ...]:
    return tuple(
        _violation(
            run_context=run_context,
            rel_path=path,
            qualname=qualname,
            line=int(getattr(arg, "lineno", 1) or 1),
            column=int(getattr(arg, "col_offset", 0) or 0) + 1,
            kind="tuple_argument",
            message="anonymous tuple argument; define a DTO and pass named fields",
        )
        for arg in call_node.args
        if isinstance(arg, ast.Tuple)
    )


def _is_tuple_constructor(func: ast.AST) -> bool:
    return isinstance(func, ast.Name) and func.id == "tuple"


def _failure_violation(*, run_context: CanonicalRunContext, seed: ScanFailureSeed) -> Violation:
    decoration = decorate_failure(
        run_context=run_context,
        rule_name=RULE_NAME,
        seed=seed,
        rationale="Ensure module parse/read validity before no-anonymous-tuple evaluation.",
    )
    message = (
        "unable to read file while checking no-anonymous-tuple policy"
        if seed.kind == "read_error"
        else "syntax error while checking no-anonymous-tuple policy"
    )
    return Violation(
        path=seed.path,
        line=seed.line,
        column=seed.column,
        qualname="<module>",
        kind=seed.kind,
        message=message,
        input_slot="module_failure",
        flow_identity=decoration.flow_identity,
        fiber_trace=decoration.fiber_trace,
        applicability_bounds=decoration.applicability_bounds,
        counterfactual_boundary=decoration.counterfactual_boundary,
        fiber_id=decoration.fiber_id,
        taint_interval_id=decoration.taint_interval_id,
        condition_overlap_id=decoration.condition_overlap_id,
        structured_hash=_structured_hash(
            seed.path,
            "<module>",
            seed.kind,
            str(seed.line),
            str(seed.column),
            seed.detail,
        ),
    )


def _violation(
    *,
    run_context: CanonicalRunContext,
    rel_path: str,
    qualname: str,
    line: int,
    column: int,
    kind: str,
    message: str,
) -> Violation:
    input_slot = f"anonymous_tuple:{kind}"
    decoration = decorate_site(
        run_context=run_context,
        rule_name=RULE_NAME,
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        column=column,
        node_kind=f"anonymous_tuple:{kind}",
        input_slot=input_slot,
        taint_class="anonymous_tuple",
        intro_kind=f"syntax:anonymous_tuple_intro:{kind}",
        condition_kind=f"syntax:anonymous_tuple_condition:{kind}",
        erase_kind=f"syntax:anonymous_tuple_erase:{kind}",
        rationale="Anonymous tuple carriers are ambiguous and should be reified as DTOs.",
    )
    return Violation(
        path=rel_path,
        line=line,
        column=column,
        qualname=qualname,
        kind=kind,
        message=message,
        input_slot=input_slot,
        flow_identity=decoration.flow_identity,
        fiber_trace=decoration.fiber_trace,
        applicability_bounds=decoration.applicability_bounds,
        counterfactual_boundary=decoration.counterfactual_boundary,
        fiber_id=decoration.fiber_id,
        taint_interval_id=decoration.taint_interval_id,
        condition_overlap_id=decoration.condition_overlap_id,
        structured_hash=_structured_hash(
            rel_path,
            qualname,
            kind,
            str(line),
            str(column),
            message,
        ),
    )


def _structured_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


__all__ = [
    "Violation",
    "collect_violations",
]
