#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

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
    build_policy_scan_batch,
    iter_failure_seeds,
    load_structured_violation_baseline_keys,
)

RULE_NAME = "defensive_fallback"
TARGET_GLOB = "src/gabion/**/*.py"
BASELINE_VERSION = 1
MODULE_MARKER = "gabion:boundary_normalization_module"
FUNCTION_MARKER = "gabion:boundary_normalization"
DECORATOR_NAMES = {
    "boundary_normalization",
    "invariants.boundary_normalization",
    "gabion.invariants.boundary_normalization",
}


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    column: int
    qualname: str
    kind: str
    message: str
    guard_form: str
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

    @property
    def legacy_key(self) -> str:
        return f"{self.path}:{self.qualname}:{self.line}:{self.kind}"

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: [{self.qualname}] {self.message}"


@dataclass(frozen=True)
class _Scope:
    qualname: str
    allow_fallbacks: bool


class _DefensiveFallbackVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        rel_path: str,
        source_lines: list[str],
        run_context: CanonicalRunContext,
    ) -> None:
        self.rel_path = rel_path
        self.source_lines = source_lines
        self.run_context = run_context
        self.violations: list[Violation] = []
        self.module_allows_fallbacks = _module_allows_fallbacks(source_lines)
        self.scope_stack: list[_Scope] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_node(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_node(node)

    def visit_If(self, node: ast.If) -> None:
        if self._scope_allows_fallbacks:
            self.generic_visit(node)
            return
        if _is_guard_condition(node.test):
            sentinel_stmt = _single_sentinel_stmt(node.body)
            if sentinel_stmt is not None:
                kind, message = sentinel_stmt
                self._report(node, kind=kind, guard_form="guard_sentinel", message=message)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if self._scope_allows_fallbacks:
            self.generic_visit(node)
            return
        if _is_broad_exception_handler(node):
            sentinel_stmt = _single_sentinel_stmt(node.body)
            if sentinel_stmt is not None:
                kind, message = sentinel_stmt
                self._report(
                    node,
                    kind=kind,
                    guard_form="broad_exception_sentinel",
                    message=f"broad exception handler {message}",
                )
        self.generic_visit(node)

    @property
    def _scope_allows_fallbacks(self) -> bool:
        if self.scope_stack:
            return self.scope_stack[-1].allow_fallbacks
        return self.module_allows_fallbacks

    def _visit_function_node(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        parent_qual = self.scope_stack[-1].qualname if self.scope_stack else "<module>"
        qualname = node.name if parent_qual == "<module>" else f"{parent_qual}.{node.name}"
        allow_fallbacks = self._function_allows_fallbacks(node)
        self.scope_stack.append(_Scope(qualname=qualname, allow_fallbacks=allow_fallbacks))
        self.generic_visit(node)
        self.scope_stack.pop()

    def _function_allows_fallbacks(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        if self.module_allows_fallbacks:
            return True
        if node.name.startswith("_normalize_"):
            return True
        if _has_preceding_marker(
            source_lines=self.source_lines,
            line=int(getattr(node, "lineno", 1)),
            marker=FUNCTION_MARKER,
        ):
            return True
        for decorator in node.decorator_list:
            dotted = _dotted_name(decorator)
            if dotted in DECORATOR_NAMES:
                return True
            match decorator:
                case ast.Call(func=func):
                    call_name = _dotted_name(func)
                    if call_name in DECORATOR_NAMES:
                        return True
                case _:
                    pass
        return False

    def _report(self, node: ast.AST, *, kind: str, guard_form: str, message: str) -> None:
        scope = self.scope_stack[-1] if self.scope_stack else _Scope("<module>", self.module_allows_fallbacks)
        input_slot = f"guard:{kind}"
        structural_identity = _structured_hash(
            self.rel_path,
            scope.qualname,
            kind,
            str(int(getattr(node, "col_offset", 0)) + 1),
            message,
        )
        decoration = decorate_site(
            run_context=self.run_context,
            rule_name=RULE_NAME,
            rel_path=self.rel_path,
            qualname=scope.qualname,
            line=int(getattr(node, "lineno", 1)),
            column=int(getattr(node, "col_offset", 0)) + 1,
            node_kind=f"fallback:{kind}",
            input_slot=input_slot,
            taint_class="fallback_guard",
            intro_kind=f"syntax:fallback_taint:{guard_form}",
            condition_kind=f"syntax:fallback_condition:{kind}",
            erase_kind=f"syntax:boundary_normalization:{kind}",
            rationale=(
                "Lift guard+sentinel fallbacks to boundary normalization so "
                "downstream fibers process only valid shapes."
            ),
        )
        self.violations.append(
            Violation(
                path=self.rel_path,
                line=int(getattr(node, "lineno", 1)),
                column=int(getattr(node, "col_offset", 0)) + 1,
                qualname=scope.qualname,
                kind=kind,
                message=message,
                guard_form=guard_form,
                input_slot=input_slot,
                flow_identity=decoration.flow_identity,
                fiber_trace=decoration.fiber_trace,
                applicability_bounds=decoration.applicability_bounds,
                counterfactual_boundary=decoration.counterfactual_boundary,
                fiber_id=decoration.fiber_id,
                taint_interval_id=decoration.taint_interval_id,
                condition_overlap_id=decoration.condition_overlap_id,
                structured_hash=structural_identity,
            )
        )


def _module_allows_fallbacks(source_lines: list[str]) -> bool:
    for raw_line in source_lines[:80]:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") and MODULE_MARKER in stripped:
            return True
        if stripped.startswith("\"\"\"") or stripped.startswith("'''"):
            continue
    return False


def _has_preceding_marker(*, source_lines: list[str], line: int, marker: str) -> bool:
    idx = max(0, line - 2)
    while idx >= 0:
        stripped = source_lines[idx].strip()
        if not stripped:
            idx -= 1
            continue
        return stripped.startswith("#") and marker in stripped
    return False


def _is_guard_condition(node: ast.AST) -> bool:
    for child in ast.walk(node):
        match child:
            case ast.Call(func=ast.Name(id="isinstance")):
                return True
            case ast.Compare(left=left, comparators=comparators):
                values = [left, *comparators]
                if any(_is_none_literal(value) for value in values):
                    return True
            case _:
                pass
    return False


def _is_broad_exception_handler(node: ast.ExceptHandler) -> bool:
    match node.type:
        case None:
            return True
        case ast.Name(id=identifier):
            return identifier in {"Exception", "BaseException"}
        case _:
            return False


def _single_sentinel_stmt(body: list[ast.stmt]) -> tuple[str, str] | None:
    if len(body) != 1:
        return None
    stmt = body[0]
    match stmt:
        case ast.Return(value=value) if _is_sentinel_return(value):
            return ("sentinel_return", "returns sentinel fallback")
        case ast.Continue():
            return ("sentinel_continue", "continues without explicit decision outcome")
        case ast.Pass():
            return ("sentinel_pass", "swallows invalid path with pass")
        case _:
            return None


def _is_none_literal(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _is_sentinel_return(node: ast.AST | None) -> bool:
    match node:
        case None:
            return True
        case ast.Constant(value=value):
            return value in {None, "", False, 0}
        case ast.List(elts=elts):
            return len(elts) == 0
        case ast.Tuple(elts=elts):
            return len(elts) == 0
        case ast.Set(elts=elts):
            return len(elts) == 0
        case ast.Dict(keys=keys):
            return len(keys) == 0
        case _:
            return False


def _dotted_name(node: ast.AST) -> str:
    return ".".join(_dotted_name_parts(node))


def _dotted_name_parts(node: ast.AST) -> tuple[str, ...]:
    match node:
        case ast.Name(id=identifier):
            return (identifier,)
        case ast.Attribute(value=value, attr=attr):
            return (*_dotted_name_parts(value), attr)
        case _:
            return ()


def _structured_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def collect_violations(
    *,
    batch: PolicyScanBatch,
    run_context: CanonicalRunContext | None = None,
) -> list[Violation]:
    context = run_context if run_context is not None else new_run_context(rule_name=RULE_NAME)
    return _collect_with_context(batch=batch, run_context=context)


def _collect_with_context(
    *,
    batch: PolicyScanBatch,
    run_context: CanonicalRunContext,
) -> list[Violation]:
    union_view = build_aspf_union_view(batch=batch)
    violations: list[Violation] = []
    for seed in (*iter_failure_seeds(batch=batch), *cst_failure_seeds(union_view=union_view)):
        violations.append(_failure_violation(run_context=run_context, seed=seed))
    for module in union_view.modules:
        visitor = _DefensiveFallbackVisitor(
            rel_path=module.rel_path,
            source_lines=module.source.splitlines(),
            run_context=run_context,
        )
        visitor.visit(module.pyast_tree)
        violations.extend(visitor.violations)
    return violations


def _failure_violation(*, run_context: CanonicalRunContext, seed: ScanFailureSeed) -> Violation:
    decoration = decorate_failure(
        run_context=run_context,
        rule_name=RULE_NAME,
        seed=seed,
        rationale="Ensure module parse/read validity before fallback substrate evaluation.",
    )
    structured_hash = _structured_hash(
        seed.path,
        "<module>",
        seed.kind,
        "module_failure",
        seed.detail,
    )
    return Violation(
        path=seed.path,
        line=seed.line,
        column=seed.column,
        qualname="<module>",
        kind=seed.kind,
        message="unable to read/parse file while checking defensive fallback policy",
        guard_form="module_failure",
        input_slot="module_failure",
        flow_identity=decoration.flow_identity,
        fiber_trace=decoration.fiber_trace,
        applicability_bounds=decoration.applicability_bounds,
        counterfactual_boundary=decoration.counterfactual_boundary,
        fiber_id=decoration.fiber_id,
        taint_interval_id=decoration.taint_interval_id,
        condition_overlap_id=decoration.condition_overlap_id,
        structured_hash=structured_hash,
    )


def _load_baseline(path: Path) -> set[str]:
    return load_structured_violation_baseline_keys(
        path=path,
        migrate_hash=lambda path_value, qualname, kind, column, message: _structured_hash(
            path_value,
            qualname,
            kind,
            str(column),
            message,
        ),
    )


def _write_baseline(*, path: Path, violations: list[Violation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": BASELINE_VERSION,
        "violations": [
            asdict(violation)
            for violation in sorted(
                violations,
                key=lambda item: (item.path, item.qualname, item.line, item.kind),
            )
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def run(*, root: Path, baseline: Path | None = None, baseline_write: bool = False) -> int:
    batch = build_policy_scan_batch(root=root, target_globs=(TARGET_GLOB,))
    violations = collect_violations(batch=batch)
    if baseline_write:
        if baseline is None:
            raise SystemExit("--baseline is required with --baseline-write")
        _write_baseline(path=baseline, violations=violations)
        print(f"wrote defensive fallback baseline to {baseline}")
        return 0

    if baseline is not None:
        allowed = _load_baseline(baseline)
        violations = [
            violation
            for violation in violations
            if violation.key not in allowed and violation.legacy_key not in allowed
        ]

    if not violations:
        print("defensive fallback policy check passed")
        return 0

    print("defensive fallback policy violations:")
    for violation in violations:
        print(f"  - {violation.render()}")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--baseline-write", action="store_true")
    args = parser.parse_args(argv)
    baseline = next(_iter_resolved_baseline_paths(args.baseline), None)
    return run(
        root=Path(args.root).resolve(),
        baseline=baseline,
        baseline_write=bool(args.baseline_write),
    )


def _iter_resolved_baseline_paths(raw_baseline: str | None) -> Iterable[Path]:
    if raw_baseline:
        yield Path(raw_baseline).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
