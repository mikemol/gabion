#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
from dataclasses import dataclass
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
from gabion.tooling.runtime.policy_result_schema import make_policy_result, write_policy_result
from gabion.tooling.runtime.policy_scan_batch import (
    PolicyScanBatch,
    ScanFailureSeed,
    build_policy_scan_batch,
    iter_failure_seeds,
)

RULE_NAME = "no_monkeypatch"
TARGET_GLOBS = (
    "tests/**/*.py",
    "src/**/*.py",
)
_PATCH_CALL_NAMES = {
    "patch",
    "patch.object",
    "patch.dict",
    "patch.multiple",
}
_PATCH_MODULES = {
    "mock",
    "unittest",
    "unittest.mock",
}


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
        return f"{self.path}:{self.line}:{self.column}: [{self.qualname}] {self.message}"


@dataclass
class _Scope:
    patch_names: set[str]
    patch_modules: set[str]


class _NoMonkeypatchVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        rel_path: str,
        run_context: CanonicalRunContext,
    ) -> None:
        self.rel_path = rel_path
        self.run_context = run_context
        self.violations: list[Violation] = []
        self.scope = _Scope(patch_names=set(), patch_modules=set(_PATCH_MODULES))
        self._qualname_stack: list[str] = []

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = str(node.module or "")
        if module in {"unittest.mock", "mock"}:
            for alias in node.names:
                imported_name = str(alias.name or "")
                local_name = str(alias.asname or imported_name)
                if imported_name.startswith("patch"):
                    self.scope.patch_names.add(local_name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            imported_name = str(alias.name or "")
            local_name = str(alias.asname or imported_name.split(".")[-1])
            if imported_name in _PATCH_MODULES:
                self.scope.patch_modules.add(local_name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._qualname_stack.append(node.name)
        self._check_monkeypatch_fixture(node)
        self._check_patch_decorators(node)
        self.generic_visit(node)
        self._qualname_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._qualname_stack.append(node.name)
        self._check_monkeypatch_fixture(node)
        self._check_patch_decorators(node)
        self.generic_visit(node)
        self._qualname_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        dotted = _dotted_name(node.func)
        if self._is_patch_call_name(dotted):
            self._report(
                node,
                kind="patch_call",
                input_slot="call:patch",
                message="patch-style runtime mutation is forbidden; use dependency injection",
            )
        match node.func:
            case ast.Attribute():
                owner = _dotted_name(node.func.value)
                if owner == "monkeypatch":
                    self._report(
                        node,
                        kind="monkeypatch_call",
                        input_slot="call:monkeypatch",
                        message=(
                            "pytest monkeypatch usage is forbidden; use dependency injection seams"
                        ),
                    )
            case _:
                pass
        self.generic_visit(node)

    def _check_monkeypatch_fixture(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        args = [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]
        if node.args.vararg is not None:
            args.append(node.args.vararg)
        if node.args.kwarg is not None:
            args.append(node.args.kwarg)
        for arg in args:
            if arg.arg == "monkeypatch":
                self._report(
                    arg,
                    kind="monkeypatch_fixture",
                    input_slot="fixture:monkeypatch",
                    message="monkeypatch fixture is forbidden; inject collaborators explicitly",
                )

    def _check_patch_decorators(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        for decorator in node.decorator_list:
            dotted = _dotted_name(decorator)
            if self._is_patch_call_name(dotted):
                self._report(
                    decorator,
                    kind="patch_decorator",
                    input_slot="decorator:patch",
                    message="patch decorator is forbidden; inject collaborators explicitly",
                )
                continue
            match decorator:
                case ast.Call():
                    dotted_call = _dotted_name(decorator.func)
                    if self._is_patch_call_name(dotted_call):
                        self._report(
                            decorator,
                            kind="patch_decorator_call",
                            input_slot="decorator:patch_call",
                            message="patch decorator call is forbidden; inject collaborators explicitly",
                        )
                case _:
                    pass

    def _is_patch_call_name(self, dotted: str) -> bool:
        if dotted in _PATCH_CALL_NAMES:
            return dotted in self.scope.patch_names or dotted.split(".", 1)[0] in self.scope.patch_names
        head = dotted.split(".", 1)[0]
        if dotted.endswith(".patch"):
            return head in self.scope.patch_modules
        if dotted.endswith(".patch.object") or dotted.endswith(".patch.dict") or dotted.endswith(
            ".patch.multiple"
        ):
            return head in self.scope.patch_modules
        return False

    @property
    def _qualname(self) -> str:
        return ".".join(self._qualname_stack) if self._qualname_stack else "<module>"

    def _report(self, node: ast.AST, *, kind: str, input_slot: str, message: str) -> None:
        line = int(getattr(node, "lineno", 1))
        column = int(getattr(node, "col_offset", 0)) + 1
        structured_hash = _structured_hash(
            self.rel_path,
            self._qualname,
            kind,
            input_slot,
            str(column),
            message,
        )
        decoration = decorate_site(
            run_context=self.run_context,
            rule_name=RULE_NAME,
            rel_path=self.rel_path,
            qualname=self._qualname,
            line=line,
            column=column,
            node_kind=f"monkeypatch:{kind}",
            input_slot=input_slot,
            taint_class="runtime_mutation",
            intro_kind=f"syntax:mutation_taint:{kind}",
            condition_kind=f"syntax:mutation_condition:{kind}",
            erase_kind=f"syntax:dependency_injection_boundary:{kind}",
            rationale=(
                "Move runtime mutation behavior behind explicit dependency-injection "
                "boundaries and keep test/runtime fibers mutation-free."
            ),
        )
        self.violations.append(
            Violation(
                path=self.rel_path,
                line=line,
                column=column,
                qualname=self._qualname,
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
                structured_hash=structured_hash,
            )
        )


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
    union_view = build_aspf_union_view(batch=batch)
    violations: list[Violation] = []
    for seed in (*iter_failure_seeds(batch=batch), *cst_failure_seeds(union_view=union_view)):
        if not _is_target_path(seed.path):
            continue
        violations.append(_failure_violation(run_context=context, seed=seed))
    for module in union_view.modules:
        if not _is_target_path(module.rel_path):
            continue
        visitor = _NoMonkeypatchVisitor(rel_path=module.rel_path, run_context=context)
        visitor.visit(module.pyast_tree)
        violations.extend(visitor.violations)
    return violations


def _failure_violation(*, run_context: CanonicalRunContext, seed: ScanFailureSeed) -> Violation:
    decoration = decorate_failure(
        run_context=run_context,
        rule_name=RULE_NAME,
        seed=seed,
        rationale="Ensure module parse/read validity before monkeypatch substrate evaluation.",
    )
    message = (
        "unable to read file while checking monkeypatch policy"
        if seed.kind == "read_error"
        else "syntax error while checking monkeypatch policy"
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
        structured_hash=_structured_hash(seed.path, "<module>", seed.kind, "module_failure", seed.detail),
    )


def _is_target_path(rel_path: str) -> bool:
    return rel_path.startswith("src/gabion/") or rel_path.startswith("tests/")


def _serialize_violation(violation: Violation) -> dict[str, object]:
    return {
        "path": violation.path,
        "line": violation.line,
        "column": violation.column,
        "qualname": violation.qualname,
        "kind": violation.kind,
        "message": violation.message,
        "input_slot": violation.input_slot,
        "flow_identity": violation.flow_identity,
        "fiber_id": violation.fiber_id,
        "taint_interval_id": violation.taint_interval_id,
        "condition_overlap_id": violation.condition_overlap_id,
        "structured_hash": violation.structured_hash,
        "render": violation.render(),
    }


def run(*, root: Path, output: Path | None = None) -> int:
    batch = build_policy_scan_batch(root=root, target_globs=TARGET_GLOBS)
    violations = collect_violations(batch=batch)
    status = "pass" if not violations else "fail"
    if output is not None:
        write_policy_result(
            path=output,
            result=make_policy_result(
                rule_id="no_monkeypatch",
                status=status,
                violations=[_serialize_violation(item) for item in violations],
                baseline_mode="none",
                source_tool="src/gabion/tooling/policy_rules/no_monkeypatch_rule.py",
                input_scope={"root": str(root)},
            ),
        )
    if not violations:
        print("no-monkeypatch policy check passed")
        return 0
    print("no-monkeypatch policy violations:")
    for violation in violations:
        print(f"  - {violation.render()}")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = next(_iter_resolved_output_paths(args.output), None)
    return run(root=Path(args.root).resolve(), output=output)


def _iter_resolved_output_paths(raw_output: Path | None) -> Iterable[Path]:
    if raw_output:
        yield raw_output.resolve()


if __name__ == "__main__":
    raise SystemExit(main())
