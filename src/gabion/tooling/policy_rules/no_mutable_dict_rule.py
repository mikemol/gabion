#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass

from gabion.analysis.foundation.event_algebra import CanonicalRunContext
from gabion.tooling.policy_substrate import (
    build_aspf_union_view,
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

RULE_NAME = "no_mutable_dict"


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


class _NoMutableDictVisitor(ast.NodeVisitor):
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

    def visit_Dict(self, node: ast.Dict) -> None:
        self._record(
            node=node,
            kind="dict_literal",
            message="dict literal is mutable; define a DTO carrier",
        )
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._record(
            node=node,
            kind="dict_comprehension",
            message="dict comprehension is mutable; define a DTO carrier",
        )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_dict_constructor(node.func):
            self._record(
                node=node,
                kind="dict_constructor",
                message="dict(...) constructor is mutable; define a DTO carrier",
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
        visitor = _NoMutableDictVisitor(rel_path=module.rel_path, run_context=context)
        visitor.visit(module.pyast_tree)
        violations.extend(visitor.violations)
    return violations


def _is_dict_constructor(func: ast.AST) -> bool:
    match func:
        case ast.Name(id="dict"):
            return True
        case _:
            return False


def _failure_violation(*, run_context: CanonicalRunContext, seed: ScanFailureSeed) -> Violation:
    decoration = decorate_failure(
        run_context=run_context,
        rule_name=RULE_NAME,
        seed=seed,
        rationale="Ensure module parse/read validity before no-mutable-dict evaluation.",
    )
    message = (
        "unable to read file while checking no-mutable-dict policy"
        if seed.kind == "read_error"
        else "syntax error while checking no-mutable-dict policy"
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
    input_slot = f"mutable_dict:{kind}"
    decoration = decorate_site(
        run_context=run_context,
        rule_name=RULE_NAME,
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        column=column,
        node_kind=f"mutable_dict:{kind}",
        input_slot=input_slot,
        taint_class="mutable_dict",
        intro_kind=f"syntax:mutable_dict_intro:{kind}",
        condition_kind=f"syntax:mutable_dict_condition:{kind}",
        erase_kind=f"syntax:mutable_dict_erase:{kind}",
        rationale="Mutable dict carriers should be replaced with explicit DTO structures.",
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
