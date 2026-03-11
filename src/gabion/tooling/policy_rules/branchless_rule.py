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
    FiberBundle,
    FrontierWitness,
    branch_required_symbols,
    build_aspf_union_view,
    build_fiber_bundle_for_qualname,
    compute_lattice_witness,
    cst_failure_seeds,
    decorate_failure,
    decorate_site,
    frontier_failure_witness,
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

RULE_NAME = "branchless"
TARGET_GLOB = "src/gabion/**/*.py"
BASELINE_VERSION = 1


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
    lattice_witness: FrontierWitness
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


class _BranchlessVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        rel_path: str,
        module_tree: ast.AST,
        source_lines: list[str],
        run_context: CanonicalRunContext,
    ) -> None:
        self.rel_path = rel_path
        self.module_tree = module_tree
        self.source_lines = source_lines
        self.run_context = run_context
        self.violations: list[Violation] = []
        self.scope_stack: list[_Scope] = []
        self._fiber_bundle_by_qualname: dict[str, FiberBundle] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_node(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_node(node)

    def visit_If(self, node: ast.If) -> None:
        self._record_branch(node, kind="if")
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._record_branch(node, kind="ifexp")
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self._record_branch(node, kind="match")
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._record_branch(node, kind="for")
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._record_branch(node, kind="async_for")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._record_branch(node, kind="while")
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._record_branch(node, kind="try")
        self.generic_visit(node)

    def visit_TryStar(self, node: ast.TryStar) -> None:
        self._record_branch(node, kind="try_star")
        self.generic_visit(node)

    def _visit_function_node(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        parent_qual = self.scope_stack[-1].qualname if self.scope_stack else "<module>"
        qualname = node.name if parent_qual == "<module>" else f"{parent_qual}.{node.name}"
        self.scope_stack.append(_Scope(qualname=qualname))
        self.generic_visit(node)
        self.scope_stack.pop()

    def _record_branch(self, node: ast.AST, *, kind: str) -> None:
        scope = self.scope_stack[-1] if self.scope_stack else _Scope("<module>")
        line = int(getattr(node, "lineno", 1))
        column = int(getattr(node, "col_offset", 0)) + 1
        input_slot = f"branch:{kind}"
        structural_identity = _structured_hash(
            self.rel_path,
            scope.qualname,
            kind,
            str(column),
            "branch construct outside decision protocol",
        )
        decoration = decorate_site(
            run_context=self.run_context,
            rule_name=RULE_NAME,
            rel_path=self.rel_path,
            qualname=scope.qualname,
            line=line,
            column=column,
            node_kind=f"branch:{kind}",
            input_slot=input_slot,
            taint_class="branch_control",
            intro_kind=f"syntax:branch_taint:{kind}",
            condition_kind=f"syntax:branch_condition:{kind}",
            erase_kind=f"syntax:decision_boundary:{kind}",
            rationale=(
                "Move branch decisions into explicit decision protocol surfaces so "
                "core fibers remain branchless."
            ),
        )
        lattice_witness = self._lattice_witness(
            scope_qualname=scope.qualname,
            node=node,
            kind=kind,
            line=line,
            column=column,
        )
        self.violations.append(
            Violation(
                path=self.rel_path,
                line=line,
                column=column,
                qualname=scope.qualname,
                kind=kind,
                message="branch construct outside decision protocol",
                input_slot=input_slot,
                flow_identity=decoration.flow_identity,
                fiber_trace=decoration.fiber_trace,
                applicability_bounds=decoration.applicability_bounds,
                counterfactual_boundary=decoration.counterfactual_boundary,
                fiber_id=decoration.fiber_id,
                taint_interval_id=decoration.taint_interval_id,
                condition_overlap_id=decoration.condition_overlap_id,
                lattice_witness=lattice_witness,
                structured_hash=structural_identity,
            )
        )

    def _lattice_witness(
        self,
        *,
        scope_qualname: str,
        node: ast.AST,
        kind: str,
        line: int,
        column: int,
    ) -> FrontierWitness:
        bundle = self._fiber_bundle_by_qualname.get(scope_qualname)
        if bundle is None:
            bundle = build_fiber_bundle_for_qualname(
                rel_path=self.rel_path,
                module_tree=self.module_tree,
                qualname=scope_qualname,
            )
            self._fiber_bundle_by_qualname[scope_qualname] = bundle
        return compute_lattice_witness(
            rel_path=self.rel_path,
            qualname=scope_qualname,
            bundle=bundle,
            branch_line=line,
            branch_column=column,
            branch_node_kind=f"branch:{kind}",
            required_symbols=branch_required_symbols(node),
        )


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
        visitor = _BranchlessVisitor(
            rel_path=module.rel_path,
            module_tree=module.pyast_tree,
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
        rationale="Ensure module parse/read validity before branchless substrate evaluation.",
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
        message="unable to read/parse file while checking branchless policy",
        input_slot="module_failure",
        flow_identity=decoration.flow_identity,
        fiber_trace=decoration.fiber_trace,
        applicability_bounds=decoration.applicability_bounds,
        counterfactual_boundary=decoration.counterfactual_boundary,
        fiber_id=decoration.fiber_id,
        taint_interval_id=decoration.taint_interval_id,
        condition_overlap_id=decoration.condition_overlap_id,
        lattice_witness=frontier_failure_witness(
            rel_path=seed.path,
            qualname="<module>",
            line=seed.line,
            column=seed.column,
            node_kind="module_failure",
            reason=seed.detail,
        ),
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
            _violation_payload(violation)
            for violation in sorted(
                violations,
                key=lambda item: (item.path, item.qualname, item.line, item.kind),
            )
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _violation_payload(violation: Violation) -> dict[str, object]:
    payload = asdict(violation)
    payload["lattice_witness"] = _lattice_payload(violation.lattice_witness)
    return payload


def _lattice_payload(frontier: FrontierWitness) -> dict[str, object]:
    return {
        "branch_site_id": frontier.branch_site_id,
        "branch_site_identity": frontier.branch_site_identity,
        "branch_line": frontier.branch_line,
        "branch_column": frontier.branch_column,
        "branch_node_kind": frontier.branch_node_kind,
        "required_symbols": list(frontier.required_symbols),
        "unresolved_symbols": list(frontier.unresolved_symbols),
        "data_anchor_site_id": frontier.data_anchor_site_id,
        "data_anchor_site_identity": frontier.data_anchor_site_identity,
        "data_anchor_line": frontier.data_anchor_line,
        "data_anchor_column": frontier.data_anchor_column,
        "data_anchor_ordinal": frontier.data_anchor_ordinal,
        "data_upstream_site_ids": list(frontier.data_upstream_site_ids),
        "data_upstream_site_identities": list(frontier.data_upstream_site_identities),
        "data_upstream_edge_ids": list(frontier.data_upstream_edge_ids),
        "exec_frontier_site_id": frontier.exec_frontier_site_id,
        "exec_frontier_site_identity": frontier.exec_frontier_site_identity,
        "exec_frontier_line": frontier.exec_frontier_line,
        "exec_frontier_column": frontier.exec_frontier_column,
        "exec_frontier_ordinal": frontier.exec_frontier_ordinal,
        "exec_upstream_site_ids": list(frontier.exec_upstream_site_ids),
        "exec_upstream_site_identities": list(frontier.exec_upstream_site_identities),
        "exec_upstream_edge_ids": list(frontier.exec_upstream_edge_ids),
        "bundle_event_count": frontier.bundle_event_count,
        "bundle_edge_count": frontier.bundle_edge_count,
        "execution_event_count": frontier.execution_event_count,
        "execution_edge_count": frontier.execution_edge_count,
        "data_exec_join": {
            "left_ids": list(frontier.data_exec_join.left_ids),
            "right_ids": list(frontier.data_exec_join.right_ids),
            "result_ids": list(frontier.data_exec_join.result_ids),
            "deterministic": frontier.data_exec_join.deterministic,
        },
        "data_exec_meet": {
            "left_ids": list(frontier.data_exec_meet.left_ids),
            "right_ids": list(frontier.data_exec_meet.right_ids),
            "result_ids": list(frontier.data_exec_meet.result_ids),
            "deterministic": frontier.data_exec_meet.deterministic,
        },
        "eta_data_to_exec": _naturality_payload(frontier.eta_data_to_exec),
        "eta_exec_to_data": _naturality_payload(frontier.eta_exec_to_data),
        "complete": frontier.complete,
    }


def _naturality_payload(witness: object) -> dict[str, object]:
    unmapped_items = list(getattr(witness, "unmapped", ()))
    return {
        "direction": str(getattr(witness, "direction", "")),
        "mapped_source_site_ids": list(getattr(witness, "mapped_source_site_ids", ())),
        "mapped_target_site_ids": list(getattr(witness, "mapped_target_site_ids", ())),
        "unmapped": [
            {
                "source_kind": str(getattr(item, "source_kind", "")),
                "source_site_id": str(getattr(item, "source_site_id", "")),
                "source_site_identity": str(getattr(item, "source_site_identity", "")),
                "reason": str(getattr(item, "reason", "")),
            }
            for item in unmapped_items
        ],
        "complete": bool(getattr(witness, "complete", False)),
    }


def run(*, root: Path, baseline: Path | None = None, baseline_write: bool = False) -> int:
    batch = build_policy_scan_batch(root=root, target_globs=(TARGET_GLOB,))
    violations = collect_violations(batch=batch)
    if baseline_write:
        if baseline is None:
            raise SystemExit("--baseline is required with --baseline-write")
        _write_baseline(path=baseline, violations=violations)
        print(f"wrote branchless policy baseline to {baseline}")
        return 0

    if baseline is not None:
        allowed = _load_baseline(baseline)
        violations = [
            violation
            for violation in violations
            if violation.key not in allowed and violation.legacy_key not in allowed
        ]

    if not violations:
        print("branchless policy check passed")
        return 0

    print("branchless policy violations:")
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
