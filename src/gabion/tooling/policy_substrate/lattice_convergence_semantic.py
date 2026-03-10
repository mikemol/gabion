from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from gabion.analysis.aspf import aspf_lattice_algebra
from gabion.order_contract import ordered_or_sorted
from gabion.tooling.runtime import policy_scanner_suite


_CANONICAL_CORPUS: tuple[str, ...] = (
    "src/gabion/tooling/policy_rules/branchless_rule.py",
    "src/gabion/tooling/runtime/policy_scanner_suite.py",
    "src/gabion/tooling/policy_substrate/dataflow_fibration.py",
    "src/gabion/analysis/aspf/aspf_lattice_algebra.py",
)


@dataclass(frozen=True)
class LatticeConvergenceDiagnostic:
    code: str
    message: str
    path: str = ""
    qualname: str = ""
    line: int = 0
    column: int = 0
    node_kind: str = ""
    detail: str = ""

    def as_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "qualname": self.qualname,
            "line": self.line,
            "column": self.column,
            "node_kind": self.node_kind,
            "detail": self.detail,
        }

    def render(self) -> str:
        location = (
            f"{self.path}:{self.line}:{self.column}:{self.qualname}:{self.node_kind}"
            if self.path
            else "<lattice_convergence>"
        )
        detail = f" [{self.detail}]" if self.detail else ""
        return f"{self.code}: {location}: {self.message}{detail}"


@dataclass(frozen=True)
class LatticeBranchRequest:
    path: str
    qualname: str
    line: int
    column: int
    node_kind: str
    required_symbols: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "qualname": self.qualname,
            "line": self.line,
            "column": self.column,
            "node_kind": self.node_kind,
            "required_symbols": list(self.required_symbols),
        }


@dataclass(frozen=True)
class SemanticLatticeConvergenceReport:
    corpus: tuple[str, ...]
    evaluated_request_count: int
    diagnostics: tuple[LatticeConvergenceDiagnostic, ...]
    evaluated_requests: tuple[LatticeBranchRequest, ...]

    @property
    def error_count(self) -> int:
        return len(self.diagnostics)

    def error_messages(self) -> tuple[str, ...]:
        return tuple(item.render() for item in self.diagnostics)

    def policy_data(self) -> dict[str, object]:
        return {
            "error_count": self.error_count,
            "evaluated_request_count": self.evaluated_request_count,
            "corpus": list(self.corpus),
            "evaluated_requests": [item.as_payload() for item in self.evaluated_requests],
            "diagnostics": [item.as_payload() for item in self.diagnostics],
        }


@dataclass(frozen=True)
class _Scope:
    qualname: str


class _RequestCollector(ast.NodeVisitor):
    def __init__(self, *, rel_path: str) -> None:
        self.rel_path = rel_path
        self.scope_stack: list[_Scope] = []
        self.requests: list[LatticeBranchRequest] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_If(self, node: ast.If) -> None:
        self._record_branch(node=node, node_kind="branch:if")
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._record_branch(node=node, node_kind="branch:ifexp")
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self._record_branch(node=node, node_kind="branch:match")
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._record_branch(node=node, node_kind="branch:for")
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._record_branch(node=node, node_kind="branch:async_for")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._record_branch(node=node, node_kind="branch:while")
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._record_branch(node=node, node_kind="branch:try")
        self.generic_visit(node)

    def visit_TryStar(self, node: ast.TryStar) -> None:
        self._record_branch(node=node, node_kind="branch:try_star")
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        parent = self.scope_stack[-1].qualname if self.scope_stack else "<module>"
        qualname = node.name if parent == "<module>" else f"{parent}.{node.name}"
        self.scope_stack.append(_Scope(qualname=qualname))
        self.generic_visit(node)
        self.scope_stack.pop()

    def _record_branch(self, *, node: ast.AST, node_kind: str) -> None:
        scope = self.scope_stack[-1].qualname if self.scope_stack else "<module>"
        self.requests.append(
            LatticeBranchRequest(
                path=self.rel_path,
                qualname=scope,
                line=int(getattr(node, "lineno", 1)),
                column=int(getattr(node, "col_offset", 0)) + 1,
                node_kind=node_kind,
                required_symbols=tuple(
                    sorted(dict.fromkeys(aspf_lattice_algebra.branch_required_symbols(node)))
                ),
            )
        )


def _sorted[T](values: Iterable[T], *, key=None) -> list[T]:
    return ordered_or_sorted(
        list(values),
        source="tooling.policy_substrate.lattice_convergence_semantic",
        key=key,
    )


def _collect_requests(*, rel_path: str, module_tree: ast.AST) -> tuple[LatticeBranchRequest, ...]:
    visitor = _RequestCollector(rel_path=rel_path)
    visitor.visit(module_tree)
    ordered = _sorted(
        visitor.requests,
        key=lambda item: (
            item.path,
            item.qualname,
            item.line,
            item.column,
            item.node_kind,
            item.required_symbols,
        ),
    )
    return tuple(ordered)


def _collect_linkage_diagnostics() -> tuple[LatticeConvergenceDiagnostic, ...]:
    diagnostics: list[LatticeConvergenceDiagnostic] = []
    try:
        from gabion.tooling import policy_substrate
        from gabion.tooling.policy_rules import branchless_rule
    except Exception as exc:  # pragma: no cover - fail-closed fallback
        diagnostics.append(
            LatticeConvergenceDiagnostic(
                code="lattice_linkage_import_failure",
                message="unable to import lattice linkage modules",
                detail=f"{type(exc).__name__}: {exc}",
            )
        )
        return tuple(diagnostics)

    if getattr(policy_substrate, "compute_lattice_witness", None) is not aspf_lattice_algebra.compute_lattice_witness:
        diagnostics.append(
            LatticeConvergenceDiagnostic(
                code="lattice_linkage_invalid_substrate_compute",
                message="policy_substrate.compute_lattice_witness must link to canonical algebra",
            )
        )
    if getattr(policy_substrate, "iter_lattice_witnesses", None) is not aspf_lattice_algebra.iter_lattice_witnesses:
        diagnostics.append(
            LatticeConvergenceDiagnostic(
                code="lattice_linkage_invalid_substrate_iter",
                message="policy_substrate.iter_lattice_witnesses must link to canonical algebra",
            )
        )
    if getattr(branchless_rule, "compute_lattice_witness", None) is not aspf_lattice_algebra.compute_lattice_witness:
        diagnostics.append(
            LatticeConvergenceDiagnostic(
                code="lattice_linkage_invalid_branchless_compute",
                message="branchless_rule.compute_lattice_witness must link to canonical algebra",
            )
        )
    serializer = getattr(policy_scanner_suite, "_lattice_witness_payload", None)
    if not callable(serializer):
        diagnostics.append(
            LatticeConvergenceDiagnostic(
                code="lattice_linkage_missing_scanner_serializer",
                message="policy_scanner_suite must expose lattice witness serializer",
            )
        )
    else:
        sample_witness = aspf_lattice_algebra.frontier_failure_witness(
            rel_path="lattice_probe.py",
            qualname="<module>",
            line=1,
            column=1,
            node_kind="branch:if",
            reason="linkage_probe",
        )

        class _Carrier:
            lattice_witness = sample_witness

        try:
            payload = serializer(_Carrier())
        except Exception as exc:  # pragma: no cover - fail-closed fallback
            diagnostics.append(
                LatticeConvergenceDiagnostic(
                    code="lattice_linkage_scanner_serializer_failure",
                    message="scanner lattice witness serializer raised",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            if not isinstance(payload, dict) or "complete" not in payload:
                diagnostics.append(
                    LatticeConvergenceDiagnostic(
                        code="lattice_linkage_scanner_payload_invalid",
                        message="scanner lattice witness serializer must emit dict with complete field",
                    )
                )
    return tuple(
        _sorted(
            diagnostics,
            key=lambda item: (
                item.code,
                item.path,
                item.qualname,
                item.line,
                item.column,
                item.node_kind,
                item.detail,
            ),
        )
    )


def collect_semantic_lattice_convergence(
    *,
    repo_root: Path,
    corpus: tuple[str, ...] | None = None,
) -> SemanticLatticeConvergenceReport:
    selected_corpus = tuple(
        _sorted(corpus or _CANONICAL_CORPUS)
    )
    diagnostics: list[LatticeConvergenceDiagnostic] = list(_collect_linkage_diagnostics())
    parsed_trees: dict[str, ast.AST] = {}
    requests: list[LatticeBranchRequest] = []
    for rel_path in selected_corpus:
        path = repo_root / rel_path
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            diagnostics.append(
                LatticeConvergenceDiagnostic(
                    code="lattice_corpus_read_failure",
                    message="unable to read canonical corpus file",
                    path=rel_path,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            diagnostics.append(
                LatticeConvergenceDiagnostic(
                    code="lattice_corpus_parse_failure",
                    message="unable to parse canonical corpus file",
                    path=rel_path,
                    line=int(exc.lineno or 0),
                    column=int(exc.offset or 0),
                    detail=str(exc.msg or ""),
                )
            )
            continue
        parsed_trees[rel_path] = tree
        requests.extend(_collect_requests(rel_path=rel_path, module_tree=tree))

    ordered_requests = tuple(
        _sorted(
            requests,
            key=lambda item: (
                item.path,
                item.qualname,
                item.line,
                item.column,
                item.node_kind,
                item.required_symbols,
            ),
        )
    )
    bundle_cache: dict[tuple[str, str], object] = {}
    evaluated_count = 0
    for request in ordered_requests:
        tree = parsed_trees.get(request.path)
        if tree is None:
            continue
        cache_key = (request.path, request.qualname)
        try:
            bundle = bundle_cache.get(cache_key)
            if bundle is None:
                bundle = aspf_lattice_algebra.build_fiber_bundle_for_qualname(
                    rel_path=request.path,
                    module_tree=tree,
                    qualname=request.qualname,
                )
                bundle_cache[cache_key] = bundle
            witness = aspf_lattice_algebra.compute_lattice_witness(
                rel_path=request.path,
                qualname=request.qualname,
                bundle=bundle,
                branch_line=request.line,
                branch_column=request.column,
                branch_node_kind=request.node_kind,
                required_symbols=request.required_symbols,
            )
            evaluated_count += 1
        except Exception as exc:  # pragma: no cover - fail-closed fallback
            diagnostics.append(
                LatticeConvergenceDiagnostic(
                    code="lattice_witness_compute_failure",
                    message="lattice witness computation failed",
                    path=request.path,
                    qualname=request.qualname,
                    line=request.line,
                    column=request.column,
                    node_kind=request.node_kind,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        incomplete = not bool(getattr(witness, "complete", False))
        has_violation = getattr(witness, "violation", None) is not None
        if incomplete or has_violation:
            diagnostics.append(
                LatticeConvergenceDiagnostic(
                    code="lattice_witness_incomplete_or_violation",
                    message="lattice witness did not converge cleanly",
                    path=request.path,
                    qualname=request.qualname,
                    line=request.line,
                    column=request.column,
                    node_kind=request.node_kind,
                    detail=f"incomplete={incomplete},has_violation={has_violation}",
                )
            )

    ordered_diagnostics = tuple(
        _sorted(
            diagnostics,
            key=lambda item: (
                item.code,
                item.path,
                item.qualname,
                item.line,
                item.column,
                item.node_kind,
                item.detail,
            ),
        )
    )
    return SemanticLatticeConvergenceReport(
        corpus=selected_corpus,
        evaluated_request_count=evaluated_count,
        diagnostics=ordered_diagnostics,
        evaluated_requests=ordered_requests,
    )


__all__ = [
    "LatticeBranchRequest",
    "LatticeConvergenceDiagnostic",
    "SemanticLatticeConvergenceReport",
    "collect_semantic_lattice_convergence",
]
