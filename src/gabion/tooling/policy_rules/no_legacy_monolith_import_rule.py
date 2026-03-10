#!/usr/bin/env python3
from __future__ import annotations

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
from gabion.tooling.runtime.policy_scan_batch import PolicyScanBatch, ScanFailureSeed, iter_failure_seeds

RULE_NAME = "no_legacy_monolith_import"
_LEGACY_MONOLITH_MODULE_PATH = Path("src/gabion/analysis/legacy_dataflow_monolith.py")


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    column: int
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
        return f"{self.path}:{self.kind}:{self.structured_hash}"

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: {self.kind}: {self.message}"


class _NoLegacyMonolithVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        rel_path: str,
        run_context: CanonicalRunContext,
    ) -> None:
        self._path = rel_path
        self._run_context = run_context
        self.violations: list[Violation] = []

    def visit_Import(self, node: ast.Import) -> None:
        self.violations.extend(
            _legacy_monolith_import_violations(
                path=self._path,
                node=node,
                run_context=self._run_context,
            )
        )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module_name = node.module or ""
        has_direct_legacy_alias = _has_alias_named(
            aliases=node.names,
            name="legacy_dataflow_monolith",
        )
        if module_name == "gabion.analysis.legacy_dataflow_monolith":
            self._record(
                node,
                kind="import_from",
                message="legacy_dataflow_monolith import is retired; use owned modules only",
            )
        elif module_name == "gabion.analysis" and has_direct_legacy_alias:
            self._record(
                node,
                kind="import_from",
                message="legacy_dataflow_monolith import is retired; use owned modules only",
            )
        elif node.level > 0 and module_name.endswith("legacy_dataflow_monolith"):
            self._record(
                node,
                kind="import_from",
                message="legacy_dataflow_monolith import is retired; use owned modules only",
            )
        elif node.level > 0 and module_name == "" and has_direct_legacy_alias:
            self._record(
                node,
                kind="import_from",
                message="legacy_dataflow_monolith import is retired; use owned modules only",
            )
        self.generic_visit(node)

    def _record(self, node: ast.AST, *, kind: str, message: str) -> None:
        self.violations.append(
            _violation(
                run_context=self._run_context,
                rel_path=self._path,
                line=int(getattr(node, "lineno", 1) or 1),
                column=int(getattr(node, "col_offset", 0) or 0) + 1,
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

    legacy_path = batch.root / _LEGACY_MONOLITH_MODULE_PATH
    if legacy_path.exists():
        violations.append(
            _violation(
                run_context=context,
                rel_path=_LEGACY_MONOLITH_MODULE_PATH.as_posix(),
                line=1,
                column=1,
                kind="module_present",
                message="retired legacy monolith module must not be present",
            )
        )

    for module in union_view.modules:
        visitor = _NoLegacyMonolithVisitor(
            rel_path=module.rel_path,
            run_context=context,
        )
        visitor.visit(module.pyast_tree)
        violations.extend(visitor.violations)
    return violations


def _failure_violation(*, run_context: CanonicalRunContext, seed: ScanFailureSeed) -> Violation:
    decoration = decorate_failure(
        run_context=run_context,
        rule_name=RULE_NAME,
        seed=seed,
        rationale="Ensure module parse/read validity before legacy-monolith substrate evaluation.",
    )
    message = (
        "unable to read file while checking legacy monolith import policy"
        if seed.kind == "read_error"
        else "syntax error while checking legacy monolith import policy"
    )
    return Violation(
        path=seed.path,
        line=seed.line,
        column=seed.column,
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
        structured_hash=_structured_hash(seed.path, seed.kind, str(seed.line), str(seed.column), seed.detail),
    )


def _legacy_monolith_import_violations(
    *,
    path: str,
    node: ast.Import,
    run_context: CanonicalRunContext,
) -> Iterable[Violation]:
    for alias in node.names:
        yield from _legacy_monolith_violation_from_alias(
            path=path,
            node=node,
            alias_name=alias.name,
            run_context=run_context,
        )


def _legacy_monolith_violation_from_alias(
    *,
    path: str,
    node: ast.Import,
    alias_name: str,
    run_context: CanonicalRunContext,
) -> tuple[Violation, ...]:
    if alias_name == "gabion.analysis.legacy_dataflow_monolith":
        return (
            _violation(
                run_context=run_context,
                rel_path=path,
                line=int(getattr(node, "lineno", 1) or 1),
                column=int(getattr(node, "col_offset", 0) or 0) + 1,
                kind="import",
                message="legacy_dataflow_monolith import is retired; use owned modules only",
            ),
        )
    return ()


def _has_alias_named(*, aliases: list[ast.alias], name: str) -> bool:
    return any(map(lambda alias: alias.name == name, aliases))


def _violation(
    *,
    run_context: CanonicalRunContext,
    rel_path: str,
    line: int,
    column: int,
    kind: str,
    message: str,
) -> Violation:
    input_slot = f"legacy:{kind}"
    decoration = decorate_site(
        run_context=run_context,
        rule_name=RULE_NAME,
        rel_path=rel_path,
        qualname="<module>",
        line=line,
        column=column,
        node_kind=f"legacy:{kind}",
        input_slot=input_slot,
        taint_class="legacy_monolith",
        intro_kind=f"syntax:legacy_taint:{kind}",
        condition_kind=f"syntax:legacy_condition:{kind}",
        erase_kind=f"syntax:legacy_erase:{kind}",
        rationale="Remove legacy monolith references from active fibers.",
    )
    return Violation(
        path=rel_path,
        line=line,
        column=column,
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
        structured_hash=_structured_hash(rel_path, kind, str(line), str(column), message),
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
