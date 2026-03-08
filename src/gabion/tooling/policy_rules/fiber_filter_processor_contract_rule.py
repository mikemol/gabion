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

RULE_NAME = "fiber_filter_processor_contract"
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
    branch_form: str
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


class _FilterProcessorVisitor(ast.NodeVisitor):
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
        loop_by_node = {item.node: item for item in loop_occurrences}

        self._record_explicit_loop_branches(
            function_node=node,
            scope_nodes=scope_nodes,
            parent_map=parent_map,
            loop_by_node=loop_by_node,
        )
        self._record_comprehension_branches(
            scope_nodes=scope_nodes,
            loop_by_node=loop_by_node,
        )

        self.generic_visit(node)
        self._scope_stack.pop()

    def _record_explicit_loop_branches(
        self,
        *,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
        scope_nodes: Sequence[ast.AST],
        parent_map: dict[ast.AST, ast.AST],
        loop_by_node: dict[ast.AST, _LoopOccurrence],
    ) -> None:
        for current in scope_nodes:
            branch_form = _branch_form(current)
            if branch_form == "branch":
                continue
            nearest_loop = _nearest_covered_loop(
                node=current,
                function_node=function_node,
                parent_map=parent_map,
            )
            if nearest_loop is None:
                continue
            nearest_occurrence = loop_by_node.get(nearest_loop)
            if nearest_occurrence is None:
                continue
            if nearest_occurrence.loop_form not in {"for", "async_for"}:
                continue
            self._record_violation(
                node=current,
                kind="branch_in_loop_processor",
                message=(
                    "loop processor body must be branchless; "
                    "split filter and processor into separate functions"
                ),
                branch_form=branch_form,
                input_slot=f"{nearest_occurrence.input_slot}:{branch_form}",
                event_kind=f"syntax:loop_branch:{branch_form}",
                rationale=(
                    "Move filtering to a dedicated upstream function so processor "
                    "functions operate on already-selected values."
                ),
            )

    def _record_comprehension_branches(
        self,
        *,
        scope_nodes: Sequence[ast.AST],
        loop_by_node: dict[ast.AST, _LoopOccurrence],
    ) -> None:
        for node in scope_nodes:
            if not _is_comprehension_expression(node):
                continue
            generators = tuple(node.generators)
            if not generators:
                continue
            root_occurrence = loop_by_node.get(generators[0])
            if root_occurrence is None:
                continue

            for generator_index, generator in enumerate(generators, start=1):
                for guard_index, guard in enumerate(generator.ifs, start=1):
                    self._record_violation(
                        node=guard,
                        kind="comprehension_filter_branch",
                        message=(
                            "comprehension-local filtering is prohibited; "
                            "split filter into a separate upstream function"
                        ),
                        branch_form="comprehension_if",
                        input_slot=(
                            f"{root_occurrence.input_slot}:gen_{generator_index}:if_{guard_index}"
                        ),
                        event_kind="syntax:comprehension_if",
                        rationale=(
                            "Hoist filtering out of comprehension clauses to preserve "
                            "a commute-friendly filter layer."
                        ),
                    )

            expression_index = 0
            for expression in _comprehension_value_expressions(node):
                for child in _iter_expression_nodes(expression):
                    if not _is_ifexp(child):
                        continue
                    expression_index += 1
                    self._record_violation(
                        node=child,
                        kind="comprehension_ifexp_branch",
                        message=(
                            "comprehension ternary branching is prohibited; "
                            "split transform paths into explicit processors"
                        ),
                        branch_form="comprehension_ifexp",
                        input_slot=f"{root_occurrence.input_slot}:ifexp_{expression_index}",
                        event_kind="syntax:comprehension_ifexp",
                        rationale=(
                            "Keep processor transformations structurally explicit by "
                            "removing ternary branching from comprehension bodies."
                        ),
                    )

    def _record_violation(
        self,
        *,
        node: ast.AST,
        kind: str,
        message: str,
        branch_form: str,
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
            branch_form=branch_form,
        )
        structured_hash = _structured_hash(
            self._rel_path,
            qualname,
            kind,
            branch_form,
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
                branch_form=branch_form,
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
        visitor = _FilterProcessorVisitor(
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


def _iter_expression_nodes(expr: ast.AST) -> tuple[ast.AST, ...]:
    nodes: list[ast.AST] = []
    stack: list[ast.AST] = [expr]
    while stack:
        current = stack.pop()
        nodes.append(current)
        for child in ast.iter_child_nodes(current):
            if _is_scope_break_node(child):
                continue
            stack.append(child)
    return tuple(nodes)


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


def _is_ifexp(node: ast.AST) -> bool:
    match node:
        case ast.IfExp():
            return True
        case _:
            return False


def _nearest_covered_loop(
    *,
    node: ast.AST,
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    parent_map: dict[ast.AST, ast.AST],
) -> ast.AST | None:
    parent = parent_map.get(node)
    while parent is not None and parent is not function_node:
        if _covered_loop_form(parent) is not None:
            return parent
        parent = parent_map.get(parent)
    return None


def _comprehension_value_expressions(node: ast.AST) -> tuple[ast.AST, ...]:
    match node:
        case ast.ListComp(elt=elt):
            return (elt,)
        case ast.SetComp(elt=elt):
            return (elt,)
        case ast.GeneratorExp(elt=elt):
            return (elt,)
        case ast.DictComp(key=key, value=value):
            return (key, value)
        case _:
            return ()


def _branch_form(node: ast.AST) -> str:
    match node:
        case ast.If():
            return "if"
        case ast.IfExp():
            return "ifexp"
        case ast.Match():
            return "match"
        case _:
            return "branch"


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
    branch_form: str,
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
            f"branch:{branch_form}",
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
        "branch_form": violation.branch_form,
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
            "Enforce filter/processor split by rejecting loop-local branching "
            "in processors and comprehensions."
        )
    )
    parser.add_argument("--root", default=".", help="Repository root (default: .)")
    parser.add_argument(
        "--out",
        default="artifacts/out/fiber_filter_processor_contract.json",
        help="Output JSON payload path",
    )
    args = parser.parse_args(argv)
    return run(root=Path(args.root), out=Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
