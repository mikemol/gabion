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

RULE_NAME = "orchestrator_primitive_barrel"
_ORCHESTRATOR_PRIMITIVE_BARREL_PATH = "src/gabion/server_core/command_orchestrator_primitives.py"
_ORCHESTRATOR_PRIMITIVE_MAX_LINES = 2400
_ORCHESTRATOR_PRIMITIVE_MAX_ALL_SYMBOLS = 220
_BRIDGE_CONTRACT_REGISTRY_IMPORT = "gabion.server_core.primitive_contract_registry"
_BRIDGE_CONTRACT_MODULE_PATHS = {
    "src/gabion/server_core/analysis_primitives.py",
    "src/gabion/server_core/timeout_runtime.py",
    "src/gabion/server_core/progress_contracts.py",
    "src/gabion/server_core/report_projection_runtime.py",
    "src/gabion/server_core/output_primitives.py",
    "src/gabion/server_core/progress_primitives.py",
    "src/gabion/server_core/timeout_primitives.py",
    "src/gabion/server_core/ingress_contracts.py",
}
_FORBIDDEN_BRIDGE_IMPORTS = {
    "gabion.server_core.command_orchestrator_primitives",
    "gabion.server_core.report_projection_runtime",
    "gabion.server_core.progress_contracts",
    "gabion.server_core.timeout_runtime",
}


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


def collect_violations(
    *,
    batch: PolicyScanBatch,
    run_context: CanonicalRunContext | None = None,
) -> list[Violation]:
    context = run_context if run_context is not None else new_run_context(rule_name=RULE_NAME)
    union_view = build_aspf_union_view(batch=batch)
    violations: list[Violation] = []

    for seed in (*iter_failure_seeds(batch=batch), *cst_failure_seeds(union_view=union_view)):
        if seed.path != _ORCHESTRATOR_PRIMITIVE_BARREL_PATH:
            continue
        violations.append(_failure_violation(run_context=context, seed=seed))

    target_module = _target_module(union_view=union_view)
    if target_module is None:
        return violations

    lines = target_module.source.splitlines()
    export_count = _export_count_from_tree(tree=target_module.pyast_tree)
    if len(lines) > _ORCHESTRATOR_PRIMITIVE_MAX_LINES:
        violations.append(
            _violation(
                run_context=context,
                rel_path=_ORCHESTRATOR_PRIMITIVE_BARREL_PATH,
                line=1,
                column=1,
                kind="line_threshold",
                message=(
                    "command_orchestrator_primitives.py exceeds line threshold "
                    f"{_ORCHESTRATOR_PRIMITIVE_MAX_LINES}"
                ),
            )
        )
    if export_count > _ORCHESTRATOR_PRIMITIVE_MAX_ALL_SYMBOLS:
        violations.append(
            _violation(
                run_context=context,
                rel_path=_ORCHESTRATOR_PRIMITIVE_BARREL_PATH,
                line=1,
                column=1,
                kind="export_threshold",
                message=(
                    "command_orchestrator_primitives.py __all__ exports exceed "
                    f"{_ORCHESTRATOR_PRIMITIVE_MAX_ALL_SYMBOLS}"
                ),
            )
        )
    for bridge_module in _iter_bridge_contract_modules(union_view=union_view):
        violations.extend(
            _bridge_contract_import_violations(
                run_context=context,
                rel_path=getattr(bridge_module, "rel_path", ""),
                tree=getattr(bridge_module, "pyast_tree", None),
            )
        )
    return violations


def _target_module(*, union_view: object):
    modules = getattr(union_view, "modules", ())
    for module in modules:
        if getattr(module, "rel_path", "") == _ORCHESTRATOR_PRIMITIVE_BARREL_PATH:
            return module
    return None


def _iter_bridge_contract_modules(*, union_view: object) -> Iterable[object]:
    for module in getattr(union_view, "modules", ()):
        rel_path = getattr(module, "rel_path", "")
        if rel_path in _BRIDGE_CONTRACT_MODULE_PATHS:
            yield module


def _bridge_contract_import_violations(
    *,
    run_context: CanonicalRunContext,
    rel_path: str,
    tree: ast.AST | None,
) -> list[Violation]:
    if tree is None:
        return []
    imported_modules = set(_iter_imported_module_names(tree))
    violations: list[Violation] = []
    if _BRIDGE_CONTRACT_REGISTRY_IMPORT not in imported_modules:
        violations.append(
            _violation(
                run_context=run_context,
                rel_path=rel_path,
                line=1,
                column=1,
                kind="bridge_contract_registry_import_missing",
                message=(
                    "bridge contract modules must import "
                    f"{_BRIDGE_CONTRACT_REGISTRY_IMPORT}"
                ),
            )
        )
    forbidden = sorted(imported_modules & _FORBIDDEN_BRIDGE_IMPORTS)
    if forbidden:
        violations.append(
            _violation(
                run_context=run_context,
                rel_path=rel_path,
                line=1,
                column=1,
                kind="bridge_contract_legacy_import",
                message=(
                    "bridge contract modules must not depend on legacy or peer bridge modules: "
                    + ", ".join(forbidden)
                ),
            )
        )
    return violations


def _iter_imported_module_names(tree: ast.AST) -> Iterable[str]:
    for node in ast.walk(tree):
        match node:
            case ast.Import(names=names):
                for alias in names:
                    yield alias.name
            case ast.ImportFrom(module=module):
                if module:
                    yield module


def _failure_violation(*, run_context: CanonicalRunContext, seed: ScanFailureSeed) -> Violation:
    decoration = decorate_failure(
        run_context=run_context,
        rule_name=RULE_NAME,
        seed=seed,
        rationale="Ensure barrel module parse/read validity before barrel threshold substrate evaluation.",
    )
    message = (
        "unable to read orchestrator barrel module"
        if seed.kind == "read_error"
        else "unable to parse orchestrator barrel module"
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


def _violation(
    *,
    run_context: CanonicalRunContext,
    rel_path: str,
    line: int,
    column: int,
    kind: str,
    message: str,
) -> Violation:
    input_slot = f"barrel:{kind}"
    decoration = decorate_site(
        run_context=run_context,
        rule_name=RULE_NAME,
        rel_path=rel_path,
        qualname="<module>",
        line=line,
        column=column,
        node_kind=f"barrel:{kind}",
        input_slot=input_slot,
        taint_class="barrel_growth",
        intro_kind=f"syntax:barrel_taint:{kind}",
        condition_kind=f"syntax:barrel_condition:{kind}",
        erase_kind=f"syntax:barrel_erase:{kind}",
        rationale="Split barrel surfaces into focused modules to preserve streamable orchestration fibers.",
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


def _export_count_from_tree(*, tree: ast.AST) -> int:
    return max(_iter_all_assignment_export_counts(getattr(tree, "body", [])), default=0)


def _iter_all_assignment_export_counts(body: list[ast.stmt]) -> Iterable[int]:
    for node in body:
        yield from _assignment_export_counts(node)


def _assignment_export_counts(node: ast.stmt) -> tuple[int, ...]:
    match node:
        case ast.Assign(targets=targets, value=ast.List(elts=elts)):
            if _assigns_all_target(targets):
                return (len(elts),)
            return ()
        case ast.Assign(targets=targets, value=ast.Tuple(elts=elts)):
            if _assigns_all_target(targets):
                return (len(elts),)
            return ()
        case _:
            return ()


def _assigns_all_target(targets: list[ast.expr]) -> bool:
    return any(_iter_target_is_all(targets))


def _iter_target_is_all(targets: list[ast.expr]) -> Iterable[bool]:
    for target in targets:
        yield _assign_target_name(target) == "__all__"


def _assign_target_name(target: ast.expr) -> str | None:
    match target:
        case ast.Name(id=identifier):
            return identifier
        case _:
            return None


def _structured_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


__all__ = ["Violation", "collect_violations"]
