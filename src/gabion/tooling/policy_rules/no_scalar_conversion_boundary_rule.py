#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass

from gabion.analysis.foundation.event_algebra import CanonicalRunContext
from gabion.tooling.policy_substrate import (
    ScalarFlowIndex,
    build_aspf_union_view,
    build_scalar_flow_index,
    cst_failure_seeds,
    decorate_failure,
    decorate_site,
    is_dunder_str_call,
    is_string_format_call,
    new_run_context,
    scalar_cast_name,
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

RULE_NAME = "no_scalar_conversion_boundary"


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    column: int
    qualname: str
    kind: str
    conversion: str
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


class _NoScalarConversionVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        rel_path: str,
        run_context: CanonicalRunContext,
        scalar_flow_index: ScalarFlowIndex,
    ) -> None:
        self._path = rel_path
        self._run_context = run_context
        self._scalar_flow_index = scalar_flow_index
        self._scope_parts: list[str] = []
        self.violations: list[Violation] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scoped(node=node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scoped(node=node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scoped(node=node)

    def visit_Call(self, node: ast.Call) -> None:
        conversion = scalar_cast_name(node.func)
        if conversion and not self._is_boundary_scope():
            self._record(
                node=node,
                kind="scalar_cast",
                conversion=conversion,
                message=(
                    "scalar conversion outside I/O boundary; move conversion to I/O "
                    "boundary or DTO __str__"
                ),
            )
        if is_dunder_str_call(func=node.func) and not self._is_boundary_scope():
            self._record(
                node=node,
                kind="dunder_str_call",
                conversion="__str__",
                message=(
                    "explicit __str__ call outside I/O boundary; move stringification "
                    "to I/O boundary or DTO __str__"
                ),
            )
        if is_string_format_call(func=node.func) and not self._is_boundary_scope():
            self._record(
                node=node,
                kind="string_format",
                conversion="format",
                message=(
                    "eager string formatting outside I/O boundary; move formatting to "
                    "I/O boundary or DTO __str__"
                ),
            )
        if self._scalar_flow_index.is_string_join_call(node=node) and not self._is_boundary_scope():
            self._record(
                node=node,
                kind="string_join",
                conversion="join",
                message=(
                    "join() scalarization outside I/O boundary; move joining to I/O "
                    "boundary or DTO __str__"
                ),
            )
        if self._scalar_flow_index.is_reduce_call(node=node) and not self._is_boundary_scope():
            self._record(
                node=node,
                kind="reduce_call",
                conversion="reduce",
                message=(
                    "reduce() scalarization outside I/O boundary; preserve stream shape "
                    "until I/O boundary"
                ),
            )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if self._scalar_flow_index.is_string_add(node=node) and not self._is_boundary_scope():
            self._record(
                node=node,
                kind="string_add",
                conversion="add",
                message=(
                    "string-add coercion outside I/O boundary; move stringification to "
                    "I/O boundary or DTO __str__"
                ),
            )
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        if not self._is_boundary_scope():
            self._record(
                node=node,
                kind="fstring_format",
                conversion="fstring",
                message=(
                    "f-string conversion outside I/O boundary; move formatting to I/O "
                    "boundary or DTO __str__"
                ),
            )
        self.generic_visit(node)

    def _visit_scoped(
        self,
        *,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        self._scope_parts.append(node.name)
        self.generic_visit(node)
        self._scope_parts.pop()

    def _qualname(self) -> str:
        if not self._scope_parts:
            return "<module>"
        return ".".join(self._scope_parts)

    def _is_boundary_scope(self) -> bool:
        if _is_path_boundary(self._path):
            return True
        return bool(self._scope_parts and self._scope_parts[-1] == "__str__")

    def _record(self, *, node: ast.AST, kind: str, conversion: str, message: str) -> None:
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
                conversion=conversion,
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
        scalar_flow_index = build_scalar_flow_index(tree=module.pyast_tree)
        visitor = _NoScalarConversionVisitor(
            rel_path=module.rel_path,
            run_context=context,
            scalar_flow_index=scalar_flow_index,
        )
        visitor.visit(module.pyast_tree)
        violations.extend(visitor.violations)
    return violations


def _is_path_boundary(rel_path: str) -> bool:
    normalized = rel_path.replace("\\", "/")
    if normalized.startswith("tests/"):
        return True
    if normalized.startswith("scripts/"):
        return True
    if normalized.startswith("src/gabion/commands/"):
        return True
    if normalized.startswith("src/gabion/cli"):
        return True
    return "/io/" in normalized


def _failure_violation(*, run_context: CanonicalRunContext, seed: ScanFailureSeed) -> Violation:
    decoration = decorate_failure(
        run_context=run_context,
        rule_name=RULE_NAME,
        seed=seed,
        rationale="Ensure module parse/read validity before scalar-conversion boundary evaluation.",
    )
    message = (
        "unable to read file while checking no-scalar-conversion-boundary policy"
        if seed.kind == "read_error"
        else "syntax error while checking no-scalar-conversion-boundary policy"
    )
    return Violation(
        path=seed.path,
        line=seed.line,
        column=seed.column,
        qualname="<module>",
        kind=seed.kind,
        conversion="module_failure",
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
    conversion: str,
    message: str,
) -> Violation:
    input_slot = f"scalar_conversion:{kind}:{conversion}"
    decoration = decorate_site(
        run_context=run_context,
        rule_name=RULE_NAME,
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        column=column,
        node_kind=f"scalar_conversion:{kind}",
        input_slot=input_slot,
        taint_class="scalar_conversion",
        intro_kind=f"syntax:scalar_conversion_intro:{kind}",
        condition_kind=f"syntax:scalar_conversion_condition:{kind}",
        erase_kind=f"syntax:scalar_conversion_erase:{kind}",
        rationale="Scalar conversion should be deferred to explicit I/O boundaries.",
    )
    return Violation(
        path=rel_path,
        line=line,
        column=column,
        qualname=qualname,
        kind=kind,
        conversion=conversion,
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
            conversion,
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
