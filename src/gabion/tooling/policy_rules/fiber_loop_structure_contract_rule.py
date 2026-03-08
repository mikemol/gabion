#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from gabion.analysis.core.prime_identity_adapter import PrimeIdentityAdapter
from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.analysis.foundation.event_algebra import (
    CanonicalRunContext,
    GlobalEventSequencer,
    derive_identity_projection_from_tokens,
)
from gabion.analysis.foundation.identity_space import GlobalIdentitySpace
from gabion.tooling.policy_rules.fiber_diagnostics import (
    FiberApplicabilityBounds,
    FiberCounterfactualBoundary,
    FiberTraceEvent,
    to_payload_bounds,
    to_payload_counterfactual,
    to_payload_trace,
)
from gabion.tooling.runtime.deadline_runtime import DeadlineBudget, deadline_scope_from_ticks

RULE_NAME = "fiber_loop_structure_contract"
TARGET_GLOB = "src/gabion/**/*.py"
_DEFAULT_POLICY_TIMEOUT_BUDGET = DeadlineBudget(ticks=120_000, tick_ns=1_000_000)
@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    column: int
    qualname: str
    kind: str
    message: str
    loop_form: str
    input_slot: str
    flow_identity: str
    structured_hash: str
    fiber_trace: tuple[FiberTraceEvent, ...]
    applicability_bounds: FiberApplicabilityBounds
    counterfactual_boundary: FiberCounterfactualBoundary

    @property
    def key(self) -> str:
        return f"{self.path}:{self.qualname}:{self.kind}:{self.structured_hash}"

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: [{self.qualname}] {self.message}"


@dataclass(frozen=True)
class _Scope:
    qualname: str


@dataclass(frozen=True)
class _LoopOccurrence:
    node: ast.AST
    loop_form: str
    line: int
    column: int
    input_slot: str


class _LoopStructureVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        rel_path: str,
        run_context: CanonicalRunContext,
    ) -> None:
        self._rel_path = rel_path
        self._run_context = run_context
        self.violations: list[Violation] = []
        self._scope_stack: list[_Scope] = [_Scope(qualname="<module>")]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_like(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_like(node)

    def _visit_function_like(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        parent = self._scope_stack[-1].qualname
        qualname = node.name if parent == "<module>" else f"{parent}.{node.name}"
        self._scope_stack.append(_Scope(qualname=qualname))

        scope_nodes, parent_map = _iter_scope_nodes(node)
        loop_occurrences = _loop_occurrences(scope_nodes=scope_nodes, parent_map=parent_map)
        nested_pair = _first_nested_pair(
            function_node=node,
            scope_nodes=scope_nodes,
            parent_map=parent_map,
            loops=loop_occurrences,
        )
        if nested_pair is not None:
            outer, inner = nested_pair
            self._record_violation(
                node=inner.node,
                kind="nested_loop",
                message=(
                    "nested loop forms inside one function are prohibited; "
                    "split into function-per-loop pipeline"
                ),
                loop_form=f"{outer.loop_form}->{inner.loop_form}",
                input_slot=f"{outer.input_slot}:{inner.input_slot}",
                event_kind="syntax:nested_loop",
                rationale=(
                    "Factor each loop into its own function and compose a loop pipeline "
                    "so nested iteration is structurally impossible."
                ),
            )

        has_yield = any(_is_yield_node(item) for item in scope_nodes)
        if len(loop_occurrences) == 1 and not has_yield:
            single_loop = loop_occurrences[0]
            self._record_violation(
                node=single_loop.node,
                kind="single_loop_non_generator",
                message=(
                    "function with exactly one loop must operate as a generator; "
                    "add yield/yield from and move collection to boundary"
                ),
                loop_form=single_loop.loop_form,
                input_slot=single_loop.input_slot,
                event_kind="syntax:single_loop_non_generator",
                rationale=(
                    "Model one-loop functions as generator processors so dataflow "
                    "commutes and downstream collection is explicit."
                ),
            )

        self.generic_visit(node)
        self._scope_stack.pop()

    def _record_violation(
        self,
        *,
        node: ast.AST,
        kind: str,
        message: str,
        loop_form: str,
        input_slot: str,
        event_kind: str,
        rationale: str,
    ) -> None:
        qualname = self._scope_stack[-1].qualname
        line = int(getattr(node, "lineno", 1) or 1)
        column = int(getattr(node, "col_offset", 0) or 0) + 1
        flow_identity = _derive_flow_identity(
            run_context=self._run_context,
            rel_path=self._rel_path,
            qualname=qualname,
            kind=kind,
            line=line,
            input_slot=input_slot,
            loop_form=loop_form,
        )
        structured_hash = _structured_hash(
            self._rel_path,
            qualname,
            kind,
            loop_form,
            input_slot,
            str(line),
            str(column),
        )
        self.violations.append(
            Violation(
                path=self._rel_path,
                line=line,
                column=column,
                qualname=qualname,
                kind=kind,
                message=message,
                loop_form=loop_form,
                input_slot=input_slot,
                flow_identity=flow_identity,
                structured_hash=structured_hash,
                fiber_trace=(
                    FiberTraceEvent(
                        ordinal=1,
                        line=line,
                        column=column,
                        event_kind=f"syntax:block_enter:{kind}",
                        normalization_class="narrow",
                        input_slot=input_slot,
                        phase_hint="syntax",
                        pre_core=True,
                    ),
                    FiberTraceEvent(
                        ordinal=2,
                        line=line,
                        column=column,
                        event_kind=event_kind,
                        normalization_class="narrow",
                        input_slot=input_slot,
                        phase_hint="syntax",
                        pre_core=True,
                    ),
                ),
                applicability_bounds=FiberApplicabilityBounds(
                    current_boundary_before_ordinal=2,
                    violation_applies_when_boundary_before_ordinal_gt=1,
                    violation_clears_when_boundary_before_ordinal_lte=1,
                    boundary_domain_max_before_ordinal=2,
                    core_entry_before_ordinal=None,
                ),
                counterfactual_boundary=FiberCounterfactualBoundary(
                    suggested_boundary_before_ordinal=1,
                    boundary_event_kind=f"syntax:block_enter:{kind}",
                    boundary_line=line,
                    boundary_column=column,
                    eliminates_violation_without_other_changes=True,
                    preserves_prior_normalization=True,
                    rationale=rationale,
                ),
            )
        )


def collect_violations(
    *,
    rel_path: str,
    source: str,
    tree: ast.AST,
    run_context: CanonicalRunContext | None = None,
) -> list[Violation]:
    with deadline_scope_from_ticks(_DEFAULT_POLICY_TIMEOUT_BUDGET):
        runtime_context = (
            run_context
            if run_context is not None
            else CanonicalRunContext(
                run_id=f"policy:{RULE_NAME}",
                sequencer=GlobalEventSequencer(),
                identity_space=GlobalIdentitySpace(
                    allocator=PrimeIdentityAdapter(registry=PrimeRegistry())
                ),
            )
        )
        visitor = _LoopStructureVisitor(
            rel_path=rel_path,
            run_context=runtime_context,
        )
        visitor.visit(tree)
        _ = source
        return visitor.violations


def collect_root_violations(*, root: Path, files: Sequence[Path] | None = None) -> list[Violation]:
    with deadline_scope_from_ticks(_DEFAULT_POLICY_TIMEOUT_BUDGET):
        candidates = files if files is not None else tuple(sorted(root.glob(TARGET_GLOB)))
        run_context = CanonicalRunContext(
            run_id=f"policy:{RULE_NAME}",
            sequencer=GlobalEventSequencer(),
            identity_space=GlobalIdentitySpace(
                allocator=PrimeIdentityAdapter(registry=PrimeRegistry())
            ),
        )
        violations: list[Violation] = []
        for path in candidates:
            if not path.is_file() or any(part == "__pycache__" for part in path.parts):
                continue
            rel_path = path.relative_to(root).as_posix()
            try:
                source = path.read_text(encoding="utf-8")
            except OSError:
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            violations.extend(
                collect_violations(
                    rel_path=rel_path,
                    source=source,
                    tree=tree,
                    run_context=run_context,
                )
            )
        return violations


def _iter_scope_nodes(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[tuple[ast.AST, ...], dict[ast.AST, ast.AST]]:
    nodes: list[ast.AST] = []
    parent_map: dict[ast.AST, ast.AST] = {}
    stack: list[ast.AST] = [function_node]
    while stack:
        node = stack.pop()
        nodes.append(node)
        for child in ast.iter_child_nodes(node):
            if _is_scope_break_node(child):
                continue
            parent_map[child] = node
            stack.append(child)
    return (tuple(nodes), parent_map)


def _loop_occurrences(
    *,
    scope_nodes: Sequence[ast.AST],
    parent_map: dict[ast.AST, ast.AST],
) -> tuple[_LoopOccurrence, ...]:
    raw: list[tuple[ast.AST, str, int, int]] = []
    for node in scope_nodes:
        loop_form = _covered_loop_form(node)
        if loop_form is None:
            continue
        line = _node_line(node, parent_map=parent_map)
        column = _node_column(node, parent_map=parent_map)
        raw.append((node, loop_form, line, column))
    ordered = sorted(raw, key=lambda item: (item[2], item[3], item[1]))
    return tuple(
        _LoopOccurrence(
            node=node,
            loop_form=loop_form,
            line=line,
            column=column,
            input_slot=f"loop_{index}",
        )
        for index, (node, loop_form, line, column) in enumerate(ordered, start=1)
    )


def _first_nested_pair(
    *,
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    scope_nodes: Sequence[ast.AST],
    parent_map: dict[ast.AST, ast.AST],
    loops: Sequence[_LoopOccurrence],
) -> tuple[_LoopOccurrence, _LoopOccurrence] | None:
    loop_by_node: dict[ast.AST, _LoopOccurrence] = {item.node: item for item in loops}
    for inner in loops:
        parent = parent_map.get(inner.node)
        while parent is not None and parent is not function_node:
            outer = loop_by_node.get(parent)
            if outer is not None:
                return (outer, inner)
            parent = parent_map.get(parent)

    for node in scope_nodes:
        if not _is_comprehension_expression(node):
            continue
        generators = tuple(node.generators)
        if len(generators) <= 1:
            continue
        outer = loop_by_node.get(generators[0])
        inner = loop_by_node.get(generators[1])
        if outer is not None and inner is not None:
            return (outer, inner)
    return None


def _covered_loop_form(node: ast.AST) -> str | None:
    match node:
        case ast.For():
            return "for"
        case ast.AsyncFor():
            return "async_for"
        case ast.comprehension(is_async=is_async):
            return "comprehension_async" if bool(is_async) else "comprehension"
        case _:
            return None


def _is_scope_break_node(node: ast.AST) -> bool:
    match node:
        case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.ClassDef() | ast.Lambda():
            return True
        case _:
            return False


def _is_comprehension_expression(node: ast.AST) -> bool:
    match node:
        case ast.ListComp() | ast.SetComp() | ast.DictComp() | ast.GeneratorExp():
            return True
        case _:
            return False


def _is_yield_node(node: ast.AST) -> bool:
    match node:
        case ast.Yield() | ast.YieldFrom():
            return True
        case _:
            return False


def _node_line(node: ast.AST, *, parent_map: dict[ast.AST, ast.AST]) -> int:
    line = getattr(node, "lineno", None)
    if line is not None:
        return int(line)
    parent = parent_map.get(node)
    if parent is None:
        return 1
    return _node_line(parent, parent_map=parent_map)


def _node_column(node: ast.AST, *, parent_map: dict[ast.AST, ast.AST]) -> int:
    column = getattr(node, "col_offset", None)
    if column is not None:
        return int(column) + 1
    parent = parent_map.get(node)
    if parent is None:
        return 1
    return _node_column(parent, parent_map=parent_map)


def _derive_flow_identity(
    *,
    run_context: CanonicalRunContext,
    rel_path: str,
    qualname: str,
    kind: str,
    line: int,
    input_slot: str,
    loop_form: str,
) -> str:
    projection = derive_identity_projection_from_tokens(
        run_context=run_context,
        tokens=(
            f"fiber.{RULE_NAME}",
            f"path:{rel_path}",
            f"qualname:{qualname}",
            f"kind:{kind}",
            f"line:{line}",
            f"slot:{input_slot}",
            f"loop:{loop_form}",
        ),
    )
    atoms = ".".join(str(atom) for atom in projection.basis_path.atoms)
    return f"{projection.basis_path.namespace}:{atoms}"


def _structured_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _serialize(violation: Violation) -> dict[str, object]:
    return {
        "path": violation.path,
        "line": violation.line,
        "column": violation.column,
        "qualname": violation.qualname,
        "kind": violation.kind,
        "message": violation.message,
        "loop_form": violation.loop_form,
        "input_slot": violation.input_slot,
        "flow_identity": violation.flow_identity,
        "structured_hash": violation.structured_hash,
        "fiber_trace": to_payload_trace(violation.fiber_trace),
        "applicability_bounds": to_payload_bounds(violation.applicability_bounds),
        "counterfactual_boundary": to_payload_counterfactual(
            violation.counterfactual_boundary
        ),
    }


def run(*, root: Path, out: Path | None = None) -> int:
    violations = collect_root_violations(root=root)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "violation_count": len(violations),
        "violations": [_serialize(item) for item in violations],
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
    print(f"{RULE_NAME}: violations={len(violations)}")
    for item in violations:
        print(f"  - {item.render()}")
    return 1 if violations else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Enforce function-per-loop policy and generator-only single-loop "
            "function contract."
        )
    )
    parser.add_argument("--root", default=".", help="Repository root (default: .)")
    parser.add_argument(
        "--out",
        default="artifacts/out/fiber_loop_structure_contract.json",
        help="Output JSON payload path",
    )
    args = parser.parse_args(argv)
    return run(root=Path(args.root), out=Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
