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
    load_path_allowlist,
)

RULE_NAME = "test_sleep_hygiene"
TARGET_GLOBS = ("tests/**/*.py",)


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    column: int
    kind: str
    message: str
    call: str
    key: str
    input_slot: str
    flow_identity: str
    fiber_trace: tuple[FiberTraceEvent, ...]
    applicability_bounds: FiberApplicabilityBounds
    counterfactual_boundary: FiberCounterfactualBoundary
    fiber_id: str
    taint_interval_id: str
    condition_overlap_id: str
    structured_hash: str

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: {self.kind}: {self.message}"


@dataclass
class _ImportScope:
    time_names: set[str]
    sleep_names: set[str]


class _SleepVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        rel_path: str,
        run_context: CanonicalRunContext,
    ) -> None:
        self.rel_path = rel_path
        self.scope = _ImportScope(time_names={"time"}, sleep_names=set())
        self.violations: list[Violation] = []
        self._run_context = run_context

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            imported_name = str(alias.name or "")
            if imported_name == "time":
                local_name = str(alias.asname or imported_name)
                self.scope.time_names.add(local_name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if str(node.module or "") != "time":
            self.generic_visit(node)
            return
        for alias in node.names:
            imported_name = str(alias.name or "")
            if imported_name == "sleep":
                local_name = str(alias.asname or imported_name)
                self.scope.sleep_names.add(local_name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        dotted = _dotted_name(node.func)

        if "." in dotted:
            head, attr = dotted.rsplit(".", 1)
            if head in self.scope.time_names and attr == "sleep":
                self._report(node=node, call=dotted, kind="time_sleep")
                self.generic_visit(node)
                return

        if dotted in self.scope.sleep_names:
            self._report(node=node, call=dotted, kind="time_sleep_import")
        self.generic_visit(node)

    def _report(self, *, node: ast.AST, call: str, kind: str) -> None:
        line = int(getattr(node, "lineno", 1))
        column = int(getattr(node, "col_offset", 0)) + 1
        message = (
            "wall-clock sleeping in tests is disallowed; "
            "prefer injected clock/process seams and keep only narrow allowlisted integration boundaries"
        )
        input_slot = f"sleep:{kind}"
        decoration = decorate_site(
            run_context=self._run_context,
            rule_name=RULE_NAME,
            rel_path=self.rel_path,
            qualname="<module>",
            line=line,
            column=column,
            node_kind=f"sleep:{kind}",
            input_slot=input_slot,
            taint_class="sleep_call",
            intro_kind=f"syntax:sleep_taint:{kind}",
            condition_kind=f"syntax:sleep_condition:{kind}",
            erase_kind=f"syntax:sleep_allowlist:{kind}",
            rationale=(
                "Move sleep behavior to explicit integration boundaries and keep "
                "core test fibers deterministic."
            ),
        )
        structured_hash = _structured_hash(
            self.rel_path,
            kind,
            str(line),
            str(column),
            call,
        )
        self.violations.append(
            Violation(
                path=self.rel_path,
                line=line,
                column=column,
                kind=kind,
                message=message,
                call=call,
                key=f"{self.rel_path}:{line}:{kind}",
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
            return (str(identifier),)
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
    allowlist_path: Path,
    run_context: CanonicalRunContext | None = None,
) -> list[Violation]:
    context = run_context if run_context is not None else new_run_context(rule_name=RULE_NAME)
    allowlisted_paths = load_path_allowlist(allowlist_path)
    union_view = build_aspf_union_view(batch=batch)
    violations: list[Violation] = []
    module_by_rel_path = {module.rel_path: module for module in union_view.modules}
    for seed in (*iter_failure_seeds(batch=batch), *cst_failure_seeds(union_view=union_view)):
        if not _is_target_path(seed.path):
            continue
        if seed.path in allowlisted_paths:
            continue
        violations.append(_failure_violation(run_context=context, seed=seed))
    for rel_path in sorted(module_by_rel_path):
        if not _is_target_path(rel_path):
            continue
        if rel_path in allowlisted_paths:
            continue
        module = module_by_rel_path[rel_path]
        visitor = _SleepVisitor(rel_path=rel_path, run_context=context)
        visitor.visit(module.pyast_tree)
        violations.extend(visitor.violations)
    return violations


def _failure_violation(*, run_context: CanonicalRunContext, seed: ScanFailureSeed) -> Violation:
    decoration = decorate_failure(
        run_context=run_context,
        rule_name=RULE_NAME,
        seed=seed,
        rationale="Ensure test module parse/read validity before sleep-hygiene substrate evaluation.",
    )
    message = (
        "unable to read test file while checking sleep hygiene"
        if seed.kind == "read_error"
        else "syntax error while checking sleep hygiene"
    )
    return Violation(
        path=seed.path,
        line=seed.line,
        column=seed.column,
        kind=seed.kind,
        message=message,
        call="<none>",
        key=f"{seed.path}:{seed.line}:{seed.kind}",
        input_slot="module_failure",
        flow_identity=decoration.flow_identity,
        fiber_trace=decoration.fiber_trace,
        applicability_bounds=decoration.applicability_bounds,
        counterfactual_boundary=decoration.counterfactual_boundary,
        fiber_id=decoration.fiber_id,
        taint_interval_id=decoration.taint_interval_id,
        condition_overlap_id=decoration.condition_overlap_id,
        structured_hash=_structured_hash(seed.path, seed.kind, str(seed.line), str(seed.column), seed.detail),
    )


def _is_target_path(rel_path: str) -> bool:
    return rel_path.startswith("tests/") and rel_path.endswith(".py")


def _serialize_violation(item: Violation) -> dict[str, object]:
    return {
        "path": item.path,
        "line": item.line,
        "column": item.column,
        "kind": item.kind,
        "call": item.call,
        "message": item.message,
        "key": item.key,
        "input_slot": item.input_slot,
        "flow_identity": item.flow_identity,
        "fiber_id": item.fiber_id,
        "taint_interval_id": item.taint_interval_id,
        "condition_overlap_id": item.condition_overlap_id,
        "structured_hash": item.structured_hash,
        "render": item.render(),
    }


def run(
    *,
    root: Path,
    allowlist_path: Path,
    output: Path | None = None,
) -> int:
    batch = build_policy_scan_batch(root=root, target_globs=TARGET_GLOBS)
    violations = collect_violations(batch=batch, allowlist_path=allowlist_path)
    status = "pass" if not violations else "fail"
    if output is not None:
        write_policy_result(
            path=output,
            result=make_policy_result(
                rule_id="test_sleep_hygiene",
                status=status,
                violations=[_serialize_violation(item) for item in violations],
                baseline_mode="allowlist",
                source_tool="src/gabion/tooling/policy_rules/test_sleep_hygiene_rule.py",
                input_scope={
                    "root": str(root),
                    "allowlist_path": str(allowlist_path),
                },
            ),
        )
    if not violations:
        print("test-sleep hygiene policy check passed")
        return 0
    print("test-sleep hygiene policy violations:")
    for violation in violations:
        print(f"  - {violation.render()}")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--allowlist",
        default="docs/policy/test_sleep_hygiene_allowlist.txt",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve() if args.output is not None else None
    return run(
        root=Path(args.root).resolve(),
        allowlist_path=Path(args.allowlist).resolve(),
        output=output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
