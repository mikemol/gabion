from __future__ import annotations

import ast
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Iterable, Iterator

from gabion.analysis.aspf import aspf_lattice_algebra
from gabion.analysis.projection.projection_registry import (
    iter_projection_fiber_semantic_specs,
)
from gabion.analysis.projection.projection_semantic_lowering import (
    ProjectionSemanticLoweringPlan,
    lower_projection_spec_to_semantic_plan,
)
from gabion.analysis.projection.projection_semantic_lowering_compile import (
    ProjectionSemanticCompiledPlanBundle,
    compile_projection_semantic_lowering_plan,
)
from gabion.analysis.projection.semantic_fragment_compile import (
    CompiledShaclPlan,
    CompiledSparqlPlan,
    compile_projection_fiber_reflect_to_shacl,
    compile_projection_fiber_reflect_to_sparql,
)
from gabion.analysis.projection.semantic_fragment import (
    CanonicalWitnessedSemanticRow,
    ProjectionFiberRequestContext,
    reflect_projection_fiber_witness,
)
from gabion.order_contract import ordered_or_sorted


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
    structural_path: str
    line: int
    column: int
    node_kind: str
    required_symbols: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "qualname": self.qualname,
            "structural_path": self.structural_path,
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
    semantic_rows: tuple[CanonicalWitnessedSemanticRow, ...]
    compiled_shacl_plans: tuple[CompiledShaclPlan, ...]
    compiled_sparql_plans: tuple[CompiledSparqlPlan, ...]
    compiled_projection_semantic_bundles: tuple[ProjectionSemanticCompiledPlanBundle, ...]

    @property
    def error_count(self) -> int:
        return len(self.diagnostics)

    def error_messages(self) -> tuple[str, ...]:
        return tuple(item.render() for item in self.diagnostics)

    def policy_data(self) -> dict[str, object]:
        witness_rows = [self._witness_row(item) for item in self.diagnostics]
        return {
            "error_count": self.error_count,
            "evaluated_request_count": self.evaluated_request_count,
            "corpus": list(self.corpus),
            "evaluated_requests": [item.as_payload() for item in self.evaluated_requests],
            "diagnostics": [item.as_payload() for item in self.diagnostics],
            "witness_rows": witness_rows,
            "semantic_rows": [item for item in self.semantic_rows],
            "compiled_shacl_plans": [item for item in self.compiled_shacl_plans],
            "compiled_sparql_plans": [item for item in self.compiled_sparql_plans],
            "compiled_projection_semantic_bundles": [
                item.policy_data() for item in self.compiled_projection_semantic_bundles
            ],
        }

    def _witness_row(self, diagnostic: LatticeConvergenceDiagnostic) -> dict[str, object]:
        code = diagnostic.code
        is_linkage = code.startswith("lattice_linkage_")
        is_ingress = code.startswith("lattice_corpus_")
        is_compute = code == "lattice_witness_compute_failure"
        is_witness = code == "lattice_witness_incomplete_or_violation"
        detail = diagnostic.detail
        has_violation = "has_violation=True" in detail if is_witness else False
        incomplete = "incomplete=True" in detail if is_witness else False
        return {
            "code": code,
            "path": diagnostic.path,
            "qualname": diagnostic.qualname,
            "line": diagnostic.line,
            "column": diagnostic.column,
            "node_kind": diagnostic.node_kind,
            "witness_kind": "unmapped_witness",
            "mapping_complete": False,
            "boundary_crossed": True,
            "collector_failure": is_linkage or is_ingress or is_compute,
            "witness_incomplete": incomplete,
            "witness_violation": has_violation,
        }


@dataclass(frozen=True)
class LatticeConvergenceEvent:
    request: LatticeBranchRequest | None = None
    diagnostic: LatticeConvergenceDiagnostic | None = None
    witness: object | None = None


@dataclass(frozen=True)
class _Scope:
    qualname: str


class _RequestCollector(ast.NodeVisitor):
    def __init__(self, *, rel_path: str) -> None:
        self.rel_path = rel_path
        self.scope_stack: list[_Scope] = []
        self.branch_counters: list[int] = []
        self.module_branch_counter = 0
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
        self.branch_counters.append(0)
        self.generic_visit(node)
        self.branch_counters.pop()
        self.scope_stack.pop()

    def _record_branch(self, *, node: ast.AST, node_kind: str) -> None:
        scope = self.scope_stack[-1].qualname if self.scope_stack else "<module>"
        if self.scope_stack:
            branch_index = self.branch_counters[-1]
            self.branch_counters[-1] += 1
        else:
            branch_index = self.module_branch_counter
            self.module_branch_counter += 1
        required_symbols = tuple(
            sorted(dict.fromkeys(aspf_lattice_algebra.branch_required_symbols(node)))
        )
        self.requests.append(
            LatticeBranchRequest(
                path=self.rel_path,
                qualname=scope,
                structural_path=_structural_path_identity(
                    (
                        scope,
                        f"branch[{branch_index}]",
                        node_kind,
                        *required_symbols,
                    )
                ),
                line=int(getattr(node, "lineno", 1)),
                column=int(getattr(node, "col_offset", 0)) + 1,
                node_kind=node_kind,
                required_symbols=required_symbols,
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
    sample_witness = aspf_lattice_algebra.frontier_failure_witness(
        rel_path="lattice_probe.py",
        qualname="<module>",
        line=1,
        column=1,
        node_kind="branch:if",
        reason="linkage_probe",
    )
    serializer = getattr(sample_witness, "as_payload", None)
    if not callable(serializer):
        diagnostics.append(
            LatticeConvergenceDiagnostic(
                code="lattice_linkage_missing_frontier_payload",
                message="FrontierWitness must expose canonical payload serializer",
            )
        )
    else:
        try:
            payload = serializer()
        except Exception as exc:  # pragma: no cover - fail-closed fallback
            diagnostics.append(
                LatticeConvergenceDiagnostic(
                    code="lattice_linkage_frontier_payload_failure",
                    message="FrontierWitness payload serializer raised",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            if not isinstance(payload, dict) or "complete" not in payload:
                diagnostics.append(
                    LatticeConvergenceDiagnostic(
                        code="lattice_linkage_frontier_payload_invalid",
                        message="FrontierWitness payload serializer must emit dict with complete field",
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


def iter_semantic_lattice_convergence(
    *,
    repo_root: Path,
    corpus: tuple[str, ...] | None = None,
) -> Iterator[LatticeConvergenceEvent]:
    selected_corpus = tuple(_sorted(corpus or _CANONICAL_CORPUS))

    def _events() -> Iterator[LatticeConvergenceEvent]:
        for diagnostic in _collect_linkage_diagnostics():
            yield LatticeConvergenceEvent(diagnostic=diagnostic)

        for rel_path in selected_corpus:
            path = repo_root / rel_path
            try:
                source = path.read_text(encoding="utf-8")
            except OSError as exc:
                yield LatticeConvergenceEvent(
                    diagnostic=LatticeConvergenceDiagnostic(
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
                yield LatticeConvergenceEvent(
                    diagnostic=LatticeConvergenceDiagnostic(
                        code="lattice_corpus_parse_failure",
                        message="unable to parse canonical corpus file",
                        path=rel_path,
                        line=int(exc.lineno or 0),
                        column=int(exc.offset or 0),
                        detail=str(exc.msg or ""),
                    )
                )
                continue

            requests = _collect_requests(rel_path=rel_path, module_tree=tree)
            for qualname, qual_request_iter in groupby(requests, key=lambda item: item.qualname):
                qual_requests = tuple(qual_request_iter)
                branch_requests = tuple(
                    aspf_lattice_algebra.BranchWitnessRequest(
                        branch_line=request.line,
                        branch_column=request.column,
                        branch_node_kind=request.node_kind,
                        required_symbols=request.required_symbols,
                    )
                    for request in qual_requests
                )
                try:
                    witness_iter = aspf_lattice_algebra.iter_lattice_witnesses(
                        rel_path=rel_path,
                        qualname=qualname,
                        module_tree=tree,
                        requests=branch_requests,
                    )
                except Exception as exc:  # pragma: no cover - fail-closed fallback
                    for request in qual_requests:
                        yield LatticeConvergenceEvent(
                            request=request,
                            diagnostic=LatticeConvergenceDiagnostic(
                                code="lattice_witness_compute_failure",
                                message="lattice witness computation failed",
                                path=request.path,
                                qualname=request.qualname,
                                line=request.line,
                                column=request.column,
                                node_kind=request.node_kind,
                                detail=f"{type(exc).__name__}: {exc}",
                            ),
                        )
                    continue
                for request in qual_requests:
                    try:
                        witness = next(witness_iter)
                    except StopIteration:
                        yield LatticeConvergenceEvent(
                            request=request,
                            diagnostic=LatticeConvergenceDiagnostic(
                                code="lattice_witness_stream_shortfall",
                                message="lattice witness iterator ended before all requests were evaluated",
                                path=request.path,
                                qualname=request.qualname,
                                line=request.line,
                                column=request.column,
                                node_kind=request.node_kind,
                            ),
                        )
                        continue
                    except Exception as exc:  # pragma: no cover - fail-closed fallback
                        yield LatticeConvergenceEvent(
                            request=request,
                            diagnostic=LatticeConvergenceDiagnostic(
                                code="lattice_witness_compute_failure",
                                message="lattice witness computation failed",
                                path=request.path,
                                qualname=request.qualname,
                                line=request.line,
                                column=request.column,
                                node_kind=request.node_kind,
                                detail=f"{type(exc).__name__}: {exc}",
                            ),
                        )
                        continue

                    incomplete = not bool(getattr(witness, "complete", False))
                    has_violation = getattr(witness, "violation", None) is not None
                    if incomplete or has_violation:
                        yield LatticeConvergenceEvent(
                            request=request,
                            witness=witness,
                            diagnostic=LatticeConvergenceDiagnostic(
                                code="lattice_witness_incomplete_or_violation",
                                message="lattice witness did not converge cleanly",
                                path=request.path,
                                qualname=request.qualname,
                                line=request.line,
                                column=request.column,
                                node_kind=request.node_kind,
                                detail=f"incomplete={incomplete},has_violation={has_violation}",
                            ),
                        )
                        continue
                    yield LatticeConvergenceEvent(request=request, witness=witness)

    return _events()


def materialize_semantic_lattice_convergence(
    *,
    corpus: tuple[str, ...] | None = None,
    events: Iterable[LatticeConvergenceEvent],
) -> SemanticLatticeConvergenceReport:
    diagnostics: list[LatticeConvergenceDiagnostic] = []
    evaluated_requests: list[LatticeBranchRequest] = []
    semantic_rows: list[CanonicalWitnessedSemanticRow] = []
    for event in events:
        request = event.request
        diagnostic = event.diagnostic
        witness = event.witness
        if request is not None:
            evaluated_requests.append(request)
            if isinstance(witness, aspf_lattice_algebra.FrontierWitness):
                semantic_rows.append(
                    reflect_projection_fiber_witness(
                        context={
                            "path": request.path,
                            "qualname": request.qualname,
                            "structural_path": request.structural_path,
                            "line": request.line,
                            "column": request.column,
                            "node_kind": request.node_kind,
                            "required_symbols": request.required_symbols,
                        },
                        witness=witness,
                    )
                )
        if diagnostic is not None:
            diagnostics.append(diagnostic)
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
    ordered_requests = tuple(
        _sorted(
            evaluated_requests,
            key=lambda item: (
                item.path,
                item.qualname,
                item.structural_path,
                item.line,
                item.column,
                item.node_kind,
                item.required_symbols,
            ),
        )
    )
    ordered_semantic_rows = tuple(
        _sorted(
            semantic_rows,
            key=lambda item: (
                item["structural_identity"],
                item["site_identity"],
            ),
        )
    )
    return SemanticLatticeConvergenceReport(
        corpus=tuple(_sorted(corpus or _CANONICAL_CORPUS)),
        evaluated_request_count=len(ordered_requests),
        diagnostics=ordered_diagnostics,
        evaluated_requests=ordered_requests,
        semantic_rows=ordered_semantic_rows,
        compiled_shacl_plans=tuple(
            _sorted(
                [compile_projection_fiber_reflect_to_shacl(item) for item in ordered_semantic_rows],
                key=lambda item: (
                    item["source_structural_identity"],
                    item["plan_id"],
                ),
            )
        ),
        compiled_sparql_plans=tuple(
            _sorted(
                [compile_projection_fiber_reflect_to_sparql(item) for item in ordered_semantic_rows],
                key=lambda item: (
                    item["source_structural_identity"],
                    item["plan_id"],
                ),
            )
        ),
        compiled_projection_semantic_bundles=tuple(
            _sorted(
                [
                    compile_projection_semantic_lowering_plan(
                        lowering_plan,
                        ordered_semantic_rows,
                    )
                    for lowering_plan in _projection_fiber_semantic_lowering_plans()
                ],
                key=lambda item: (
                    item.spec_name,
                    item.spec_identity,
                ),
            )
        ),
    )


def _structural_path_identity(path: tuple[str, ...]) -> str:
    return "::".join(path)


def _projection_fiber_semantic_lowering_plans() -> tuple[ProjectionSemanticLoweringPlan, ...]:
    return tuple(
        _sorted(
            [
                lower_projection_spec_to_semantic_plan(spec)
                for spec in iter_projection_fiber_semantic_specs()
            ],
            key=lambda item: (
                item.spec_name,
                item.spec_identity,
            ),
        )
    )


def collect_semantic_lattice_convergence(
    *,
    repo_root: Path,
    corpus: tuple[str, ...] | None = None,
) -> SemanticLatticeConvergenceReport:
    selected_corpus = tuple(_sorted(corpus or _CANONICAL_CORPUS))
    return materialize_semantic_lattice_convergence(
        corpus=selected_corpus,
        events=iter_semantic_lattice_convergence(
            repo_root=repo_root,
            corpus=selected_corpus,
        ),
    )


__all__ = [
    "LatticeBranchRequest",
    "LatticeConvergenceEvent",
    "LatticeConvergenceDiagnostic",
    "SemanticLatticeConvergenceReport",
    "collect_semantic_lattice_convergence",
    "iter_semantic_lattice_convergence",
    "materialize_semantic_lattice_convergence",
]
