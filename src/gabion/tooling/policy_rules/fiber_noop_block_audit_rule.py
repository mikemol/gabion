#!/usr/bin/env python3
# gabion:decision_protocol_module
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
_DEFAULT_POLICY_TIMEOUT_BUDGET = DeadlineBudget(ticks=120_000, tick_ns=1_000_000)


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    column: int
    qualname: str
    kind: str
    message: str
    noop_kind: str
    block_kind: str
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


class _NoopBlockVisitor(ast.NodeVisitor):
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
        self._visit_scoped_node(node=node, name=node.name, block_kind="function_body")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scoped_node(node=node, name=node.name, block_kind="function_body")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scoped_node(node=node, name=node.name, block_kind="class_body")

    def visit_If(self, node: ast.If) -> None:
        self._record_block(
            block=node.body,
            block_kind="if_body",
            anchor_line=int(getattr(node, "lineno", 1) or 1),
            anchor_column=int(getattr(node, "col_offset", 0) or 0) + 1,
        )
        if node.orelse:
            self._record_block(
                block=node.orelse,
                block_kind="if_orelse",
                anchor_line=int(getattr(node, "lineno", 1) or 1),
                anchor_column=int(getattr(node, "col_offset", 0) or 0) + 1,
            )
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._visit_loop_like(
            body=node.body,
            orelse=node.orelse,
            block_kind="for",
            node=node,
        )
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_loop_like(
            body=node.body,
            orelse=node.orelse,
            block_kind="async_for",
            node=node,
        )
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._visit_loop_like(
            body=node.body,
            orelse=node.orelse,
            block_kind="while",
            node=node,
        )
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self._record_block(
            block=node.body,
            block_kind="with_body",
            anchor_line=int(getattr(node, "lineno", 1) or 1),
            anchor_column=int(getattr(node, "col_offset", 0) or 0) + 1,
        )
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._record_block(
            block=node.body,
            block_kind="async_with_body",
            anchor_line=int(getattr(node, "lineno", 1) or 1),
            anchor_column=int(getattr(node, "col_offset", 0) or 0) + 1,
        )
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._visit_try_like(
            body=node.body,
            handlers=node.handlers,
            orelse=node.orelse,
            finalbody=node.finalbody,
            node=node,
            block_kind="try",
        )
        self.generic_visit(node)

    def visit_TryStar(self, node: ast.TryStar) -> None:
        self._visit_try_like(
            body=node.body,
            handlers=node.handlers,
            orelse=node.orelse,
            finalbody=node.finalbody,
            node=node,
            block_kind="try_star",
        )
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        for index, case in enumerate(node.cases, start=1):
            self._record_block(
                block=case.body,
                block_kind=f"match_case_{index}",
                anchor_line=int(getattr(node, "lineno", 1) or 1),
                anchor_column=int(getattr(node, "col_offset", 0) or 0) + 1,
            )
        self.generic_visit(node)

    def _visit_loop_like(
        self,
        *,
        body: list[ast.stmt],
        orelse: list[ast.stmt],
        block_kind: str,
        node: ast.AST,
    ) -> None:
        anchor_line = int(getattr(node, "lineno", 1) or 1)
        anchor_column = int(getattr(node, "col_offset", 0) or 0) + 1
        self._record_block(
            block=body,
            block_kind=f"{block_kind}_body",
            anchor_line=anchor_line,
            anchor_column=anchor_column,
        )
        if orelse:
            self._record_block(
                block=orelse,
                block_kind=f"{block_kind}_orelse",
                anchor_line=anchor_line,
                anchor_column=anchor_column,
            )

    def _visit_try_like(
        self,
        *,
        body: list[ast.stmt],
        handlers: list[ast.ExceptHandler],
        orelse: list[ast.stmt],
        finalbody: list[ast.stmt],
        node: ast.AST,
        block_kind: str,
    ) -> None:
        anchor_line = int(getattr(node, "lineno", 1) or 1)
        anchor_column = int(getattr(node, "col_offset", 0) or 0) + 1
        self._record_block(
            block=body,
            block_kind=f"{block_kind}_body",
            anchor_line=anchor_line,
            anchor_column=anchor_column,
        )
        for index, handler in enumerate(handlers, start=1):
            self._record_block(
                block=handler.body,
                block_kind=f"{block_kind}_except_{index}",
                anchor_line=anchor_line,
                anchor_column=anchor_column,
            )
        if orelse:
            self._record_block(
                block=orelse,
                block_kind=f"{block_kind}_orelse",
                anchor_line=anchor_line,
                anchor_column=anchor_column,
            )
        if finalbody:
            self._record_block(
                block=finalbody,
                block_kind=f"{block_kind}_finalbody",
                anchor_line=anchor_line,
                anchor_column=anchor_column,
            )

    def _visit_scoped_node(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        name: str,
        block_kind: str,
    ) -> None:
        parent = self._scope_stack[-1].qualname
        qualname = name if parent == "<module>" else f"{parent}.{name}"
        self._scope_stack.append(_Scope(qualname=qualname))
        self._record_block(
            block=node.body,
            block_kind=block_kind,
            anchor_line=int(getattr(node, "lineno", 1) or 1),
            anchor_column=int(getattr(node, "col_offset", 0) or 0) + 1,
        )
        self.generic_visit(node)
        self._scope_stack.pop()

    def _record_block(
        self,
        *,
        block: list[ast.stmt],
        block_kind: str,
        anchor_line: int,
        anchor_column: int,
    ) -> None:
        if len(block) == 1:
            statement = block[0]
            noop_kind = _noop_kind(statement)
            if noop_kind is not None:
                qualname = self._scope_stack[-1].qualname
                line = int(getattr(statement, "lineno", anchor_line) or anchor_line)
                column = int(getattr(statement, "col_offset", 0) or 0) + 1
                flow_identity = _derive_flow_identity(
                    run_context=self._run_context,
                    rel_path=self._rel_path,
                    qualname=qualname,
                    block_kind=block_kind,
                    line=line,
                )
                event_kind = f"syntax:noop_block:{block_kind}:{noop_kind}"
                structured_hash = _structured_hash(
                    self._rel_path,
                    qualname,
                    block_kind,
                    noop_kind,
                    str(line),
                    str(column),
                )
                self.violations.append(
                    Violation(
                        path=self._rel_path,
                        line=line,
                        column=column,
                        qualname=qualname,
                        kind="singleton_noop_block",
                        message=f"{block_kind} contains only '{noop_kind}'",
                        noop_kind=noop_kind,
                        block_kind=block_kind,
                        flow_identity=flow_identity,
                        structured_hash=structured_hash,
                        fiber_trace=(
                            FiberTraceEvent(
                                ordinal=1,
                                line=anchor_line,
                                column=anchor_column,
                                event_kind=f"syntax:block_enter:{block_kind}",
                                normalization_class="noop",
                                input_slot=block_kind,
                                phase_hint="syntax",
                                pre_core=True,
                            ),
                            FiberTraceEvent(
                                ordinal=2,
                                line=line,
                                column=column,
                                event_kind=event_kind,
                                normalization_class="noop",
                                input_slot=block_kind,
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
                            boundary_event_kind=f"syntax:block_enter:{block_kind}",
                            boundary_line=anchor_line,
                            boundary_column=anchor_column,
                            eliminates_violation_without_other_changes=True,
                            preserves_prior_normalization=True,
                            rationale=(
                                "Replace singleton noop block with an explicit decision outcome "
                                "or a never(...) contract discharge."
                            ),
                        ),
                    )
                )


def collect_violations(*, root: Path, files: Sequence[Path] | None = None) -> list[Violation]:
    with deadline_scope_from_ticks(_DEFAULT_POLICY_TIMEOUT_BUDGET):
        candidates = files if files is not None else tuple(sorted(root.glob(TARGET_GLOB)))
        run_context = CanonicalRunContext(
            run_id="policy:fiber_noop_block_audit",
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
            visitor = _NoopBlockVisitor(rel_path=rel_path, run_context=run_context)
            visitor.visit(tree)
            violations.extend(visitor.violations)
        return violations


def _noop_kind(statement: ast.stmt) -> str | None:
    match statement:
        case ast.Pass():
            return "pass"
        case ast.Return(value=None):
            return "return_none"
        case ast.Return(value=ast.Constant(value=None)):
            return "return_none"
        case _:
            return None


def _derive_flow_identity(
    *,
    run_context: CanonicalRunContext,
    rel_path: str,
    qualname: str,
    block_kind: str,
    line: int,
) -> str:
    projection = derive_identity_projection_from_tokens(
        run_context=run_context,
        tokens=(
            "fiber.noop_block_audit",
            f"path:{rel_path}",
            f"qualname:{qualname}",
            f"block:{block_kind}",
            f"line:{line}",
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
        "noop_kind": violation.noop_kind,
        "block_kind": violation.block_kind,
        "flow_identity": violation.flow_identity,
        "structured_hash": violation.structured_hash,
        "fiber_trace": to_payload_trace(violation.fiber_trace),
        "applicability_bounds": to_payload_bounds(violation.applicability_bounds),
        "counterfactual_boundary": to_payload_counterfactual(
            violation.counterfactual_boundary
        ),
    }


def run(*, root: Path, out: Path | None = None) -> int:
    violations = collect_violations(root=root)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "violation_count": len(violations),
        "violations": [_serialize(item) for item in violations],
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"fiber-noop-block-audit: violations={len(violations)}")
    for item in violations:
        print(f"  - {item.render()}")
    return 1 if violations else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit singleton pass/return-none blocks with fiber diagnostics payloads."
    )
    parser.add_argument("--root", default=".", help="Repository root (default: .)")
    parser.add_argument(
        "--out",
        default="artifacts/out/fiber_noop_block_audit.json",
        help="Output JSON payload path",
    )
    args = parser.parse_args(argv)
    return run(root=Path(args.root), out=Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
