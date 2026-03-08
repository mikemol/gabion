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

TARGET_GLOB = "src/gabion/**/*.py"
RULE_NAME = "fiber_scalar_sentinel_contract"
_DEFAULT_POLICY_TIMEOUT_BUDGET = DeadlineBudget(ticks=120_000, tick_ns=1_000_000)


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    column: int
    qualname: str
    kind: str
    message: str
    scalar_literal: str
    comparison_operator: str
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


class _ScalarSentinelVisitor(ast.NodeVisitor):
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
        self._visit_scoped_node(node=node, name=node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scoped_node(node=node, name=node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scoped_node(node=node, name=node.name)

    def visit_If(self, node: ast.If) -> None:
        self._record_if_none_guard(node)
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        if _is_none_literal(node.orelse):
            self._record_ifexp_none(node=node, arm="else")
        if _is_none_literal(node.body):
            self._record_ifexp_none(node=node, arm="then")
        self.generic_visit(node)

    def _visit_scoped_node(
        self,
        *,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        name: str,
    ) -> None:
        parent = self._scope_stack[-1].qualname
        qualname = name if parent == "<module>" else f"{parent}.{name}"
        self._scope_stack.append(_Scope(qualname=qualname))
        self.generic_visit(node)
        self._scope_stack.pop()

    def _record_if_none_guard(self, node: ast.If) -> None:
        comparisons = _none_comparisons(node.test)
        if not comparisons:
            return

        noop_body = _is_noop_or_none_block(node.body)
        noop_else = _is_noop_or_none_block(node.orelse)
        if not noop_body and not noop_else:
            return

        branch_hint = "body" if noop_body else "orelse"
        for index, operator in enumerate(comparisons, start=1):
            input_slot = f"if_none_guard_{branch_hint}_{index}"
            self._record_violation(
                node=node.test,
                kind="none_comparison",
                message=(
                    "direct None guard driving noop/sentinel control flow is prohibited; "
                    "shift decision to boundary ingress"
                ),
                scalar_literal="None",
                comparison_operator=operator,
                input_slot=input_slot,
                event_kind=f"syntax:none_guard:{branch_hint}:{operator}",
            )

    def _record_ifexp_none(self, *, node: ast.IfExp, arm: str) -> None:
        self._record_violation(
            node=node,
            kind="ifexp_none_arm",
            message=(
                f"if-expression {arm} arm emits None sentinel; "
                "shift decision to boundary ingress"
            ),
            scalar_literal="None",
            comparison_operator="ifexp",
            input_slot=f"ifexp_{arm}_arm",
            event_kind=f"syntax:ifexp:{arm}_none",
        )

    def _record_violation(
        self,
        *,
        node: ast.AST,
        kind: str,
        message: str,
        scalar_literal: str,
        comparison_operator: str,
        input_slot: str,
        event_kind: str,
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
        )
        structured_hash = _structured_hash(
            self._rel_path,
            qualname,
            kind,
            scalar_literal,
            comparison_operator,
            input_slot,
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
                scalar_literal=scalar_literal,
                comparison_operator=comparison_operator,
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
                    rationale=(
                        "Discharge scalar/sentinel classification at boundary ingress "
                        "and pass typed outcomes into core."
                    ),
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
        visitor = _ScalarSentinelVisitor(rel_path=rel_path, run_context=runtime_context)
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


def _none_literal(node: ast.AST) -> str | None:
    match node:
        case ast.Constant(value=None):
            return "None"
        case _:
            return None


def _is_none_literal(node: ast.AST) -> bool:
    return _none_literal(node) == "None"


def _none_comparisons(test: ast.AST) -> tuple[str, ...]:
    match test:
        case ast.Compare(left=left, ops=ops, comparators=comparators):
            left_term = left
            operators: list[str] = []
            for operator_node, right_term in zip(ops, comparators):
                if _none_literal(left_term) == "None" or _none_literal(right_term) == "None":
                    operators.append(_operator_label(operator_node))
                left_term = right_term
            return tuple(operators)
        case _:
            return ()


def _is_noop_or_none_block(block: Sequence[ast.stmt]) -> bool:
    if len(block) != 1:
        return False
    stmt = block[0]
    match stmt:
        case ast.Pass():
            return True
        case ast.Continue():
            return True
        case ast.Return(value=value):
            return value is None or _is_none_literal(value)
        case _:
            return False


def _operator_label(operator: ast.cmpop) -> str:
    match operator:
        case ast.Is():
            return "is"
        case ast.IsNot():
            return "is_not"
        case ast.Eq():
            return "eq"
        case ast.NotEq():
            return "not_eq"
        case ast.Lt():
            return "lt"
        case ast.LtE():
            return "lt_eq"
        case ast.Gt():
            return "gt"
        case ast.GtE():
            return "gt_eq"
        case ast.In():
            return "in"
        case ast.NotIn():
            return "not_in"
        case _:
            return "compare"


def _derive_flow_identity(
    *,
    run_context: CanonicalRunContext,
    rel_path: str,
    qualname: str,
    kind: str,
    line: int,
    input_slot: str,
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
        "scalar_literal": violation.scalar_literal,
        "comparison_operator": violation.comparison_operator,
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
        out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"{RULE_NAME}: violations={len(violations)}")
    for item in violations:
        print(f"  - {item.render()}")
    return 1 if violations else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit direct scalar/sentinel comparisons and None-emitting if-expressions "
            "with fiber diagnostics payloads."
        )
    )
    parser.add_argument("--root", default=".", help="Repository root (default: .)")
    parser.add_argument(
        "--out",
        default="artifacts/out/fiber_scalar_sentinel_contract.json",
        help="Output JSON payload path",
    )
    args = parser.parse_args(argv)
    return run(root=Path(args.root), out=Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
