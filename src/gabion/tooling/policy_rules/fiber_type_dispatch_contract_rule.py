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
from gabion.tooling.runtime.policy_scan_batch import (
    PolicyScanBatch,
    ScanFailureSeed,
    build_policy_scan_batch,
    iter_failure_seeds,
)

RULE_NAME = "fiber_type_dispatch_contract"
TARGET_GLOB = "src/gabion/**/*.py"
_DEFAULT_POLICY_TIMEOUT_BUDGET = DeadlineBudget(ticks=120_000, tick_ns=1_000_000)
_ABSTRACT_REGISTER_TYPES = {
    "Mapping",
    "MutableMapping",
    "Sequence",
    "MutableSequence",
    "Iterable",
    "Iterator",
    "Collection",
    "Container",
    "Set",
    "AbstractSet",
    "Callable",
}


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    column: int
    qualname: str
    kind: str
    message: str
    guard_form: str
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


class _TypeDispatchVisitor(ast.NodeVisitor):
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

    def visit_If(self, node: ast.If) -> None:
        guard_forms = _guard_forms(node.test)
        for index, guard_form in enumerate(guard_forms, start=1):
            self._record_violation(
                node=node.test,
                kind="manual_type_guard",
                message=(
                    "manual type-driven control flow is prohibited; "
                    "use singledispatch and strict typed ingress"
                ),
                guard_form=guard_form,
                input_slot=f"if_type_guard_{index}",
                event_kind=f"syntax:if_guard:{guard_form}",
            )
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        guard_forms = _guard_forms(node.test)
        for index, guard_form in enumerate(guard_forms, start=1):
            self._record_violation(
                node=node.test,
                kind="manual_type_guard",
                message=(
                    "manual type-driven loop guard is prohibited; "
                    "use singledispatch and strict typed ingress"
                ),
                guard_form=guard_form,
                input_slot=f"while_type_guard_{index}",
                event_kind=f"syntax:while_guard:{guard_form}",
            )
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        if _is_type_call(node.subject):
            self._record_violation(
                node=node.subject,
                kind="match_type_guard",
                message=(
                    "match type(...) routing is prohibited; "
                    "use singledispatch-based routing"
                ),
                guard_form="match_type",
                input_slot="match_type_subject",
                event_kind="syntax:match_type_subject",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        dispatch_lookup = _dispatch_table_lookup_from_call(node)
        if dispatch_lookup is not None:
            table_name, call_arg = dispatch_lookup
            self._record_violation(
                node=call_arg,
                kind="dispatch_table_lookup",
                message=(
                    "manual type-keyed dispatch table lookup is prohibited; "
                    "use singledispatch registration"
                ),
                guard_form="dispatch_table_lookup",
                input_slot=f"dispatch_table:{table_name}",
                event_kind="syntax:dispatch_table_lookup",
            )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name):
            slice_expr = _subscript_slice_expr(node.slice)
            if _is_type_call(slice_expr):
                self._record_violation(
                    node=slice_expr,
                    kind="dispatch_table_lookup",
                    message=(
                        "manual type-keyed dispatch table lookup is prohibited; "
                        "use singledispatch registration"
                    ),
                    guard_form="dispatch_table_lookup",
                    input_slot=f"dispatch_table:{node.value.id}",
                    event_kind="syntax:dispatch_table_lookup",
                )
        self.generic_visit(node)

    def _visit_function_like(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        parent = self._scope_stack[-1].qualname
        qualname = node.name if parent == "<module>" else f"{parent}.{node.name}"
        self._scope_stack.append(_Scope(qualname=qualname))

        if _has_singledispatch_decorator(node):
            if not _has_never_call(node.body):
                self._record_violation(
                    node=node,
                    kind="missing_never_base",
                    message=(
                        "singledispatch base handler must call never(...) for "
                        "unregistered runtime types"
                    ),
                    guard_form="missing_never_base",
                    input_slot="singledispatch_base",
                    event_kind="syntax:singledispatch_base_missing_never",
                )

        for decorator in node.decorator_list:
            for register_type in _iter_dispatch_register_type_expr(
                decorator=decorator,
                function_node=node,
            ):
                if _is_abstract_register_type(register_type):
                    self._record_violation(
                        node=register_type,
                        kind="abstract_register_type",
                        message=(
                            "dispatch register type must be concrete; "
                            "ABCs are prohibited by dispatch contract"
                        ),
                        guard_form="abstract_register_type",
                        input_slot="register_type",
                        event_kind="syntax:abstract_register_type",
                    )

        self.generic_visit(node)
        self._scope_stack.pop()

    def _record_violation(
        self,
        *,
        node: ast.AST,
        kind: str,
        message: str,
        guard_form: str,
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
            guard_form=guard_form,
        )
        structured_hash = _structured_hash(
            self._rel_path,
            qualname,
            kind,
            guard_form,
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
                guard_form=guard_form,
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
                        "Move runtime-type routing to singledispatch ingress and "
                        "keep semantic core typed and branch-minimal."
                    ),
                ),
            )
        )


def collect_violations(
    *,
    batch: PolicyScanBatch,
    run_context: CanonicalRunContext | None = None,
) -> list[Violation]:
    if run_context is not None:
        return _collect_with_context(batch=batch, run_context=run_context)

    violations: list[Violation] = []
    for seed in iter_failure_seeds(batch=batch):
        with deadline_scope_from_ticks(_DEFAULT_POLICY_TIMEOUT_BUDGET):
            violations.append(
                _failure_violation(
                    run_context=_new_run_context(),
                    seed=seed,
                )
            )
    for module in batch.modules:
        with deadline_scope_from_ticks(_DEFAULT_POLICY_TIMEOUT_BUDGET):
            module_violations = _collect_with_context(
                batch=PolicyScanBatch(
                    root=batch.root,
                    modules=(module,),
                    read_failures=(),
                    parse_failures=(),
                ),
                run_context=_new_run_context(),
            )
            violations.extend(module_violations)
    return violations


def _new_run_context() -> CanonicalRunContext:
    return CanonicalRunContext(
        run_id=f"policy:{RULE_NAME}",
        sequencer=GlobalEventSequencer(),
        identity_space=GlobalIdentitySpace(
            allocator=PrimeIdentityAdapter(registry=PrimeRegistry())
        ),
    )


def _collect_with_context(
    *,
    batch: PolicyScanBatch,
    run_context: CanonicalRunContext,
) -> list[Violation]:
    violations: list[Violation] = []
    for seed in iter_failure_seeds(batch=batch):
        violations.append(
            _failure_violation(
                run_context=run_context,
                seed=seed,
            )
        )
    for module in batch.modules:
        visitor = _TypeDispatchVisitor(
            rel_path=module.rel_path,
            run_context=run_context,
        )
        visitor.visit(module.tree)
        violations.extend(visitor.violations)
    return violations


def _failure_violation(
    *,
    run_context: CanonicalRunContext,
    seed: ScanFailureSeed,
) -> Violation:
    flow_identity = _derive_flow_identity(
        run_context=run_context,
        rel_path=seed.path,
        qualname="<module>",
        kind=seed.kind,
        line=seed.line,
        input_slot="module_failure",
        guard_form="module_failure",
    )
    structured_hash = _structured_hash(
        seed.path,
        "<module>",
        seed.kind,
        "module_failure",
        "module_failure",
        str(seed.line),
        str(seed.column),
    )
    return Violation(
        path=seed.path,
        line=seed.line,
        column=seed.column,
        qualname="<module>",
        kind=seed.kind,
        message=seed.detail,
        guard_form="module_failure",
        input_slot="module_failure",
        flow_identity=flow_identity,
        structured_hash=structured_hash,
        fiber_trace=(
            FiberTraceEvent(
                ordinal=1,
                line=seed.line,
                column=seed.column,
                event_kind=f"syntax:block_enter:{seed.kind}",
                normalization_class="narrow",
                input_slot="module_failure",
                phase_hint="syntax",
                pre_core=True,
            ),
            FiberTraceEvent(
                ordinal=2,
                line=seed.line,
                column=seed.column,
                event_kind=f"syntax:{seed.kind}",
                normalization_class="narrow",
                input_slot="module_failure",
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
            boundary_event_kind=f"syntax:block_enter:{seed.kind}",
            boundary_line=seed.line,
            boundary_column=seed.column,
            eliminates_violation_without_other_changes=True,
            preserves_prior_normalization=True,
            rationale="Ensure module parse/read validity before type-dispatch analysis.",
        ),
    )


def _guard_forms(expr: ast.AST) -> tuple[str, ...]:
    forms: list[str] = []
    for node in ast.walk(expr):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
        ):
            forms.append("isinstance_guard")
            continue
        if isinstance(node, ast.Compare):
            terms = (node.left, *node.comparators)
            if any(_is_type_call(term) for term in terms):
                operators = tuple(_operator_label(op) for op in node.ops)
                if any(
                    label in {"is", "is_not", "eq", "not_eq"}
                    for label in operators
                ):
                    forms.append("type_compare_guard")
    return tuple(forms)


def _is_type_call(expr: ast.AST) -> bool:
    return (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "type"
        and bool(expr.args)
    )


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


def _dispatch_table_lookup_from_call(node: ast.Call) -> tuple[str, ast.AST] | None:
    if (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Name)
        and node.args
        and _is_type_call(node.args[0])
    ):
        return (node.func.value.id, node.args[0])
    return None


def _subscript_slice_expr(slice_node: ast.AST) -> ast.AST:
    if isinstance(slice_node, ast.Index):
        return slice_node.value
    return slice_node


def _has_singledispatch_decorator(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    for decorator in function_node.decorator_list:
        if isinstance(decorator, ast.Name):
            if decorator.id in {"singledispatch", "singledispatchmethod"}:
                return True
            continue
        if isinstance(decorator, ast.Attribute):
            if decorator.attr in {"singledispatch", "singledispatchmethod"}:
                return True
    return False


def _has_never_call(body: Sequence[ast.stmt]) -> bool:
    for statement in body:
        for node in ast.walk(statement):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "never"
            ):
                return True
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "never"
            ):
                return True
    return False


def _iter_dispatch_register_type_expr(
    *,
    decorator: ast.AST,
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.AST, ...]:
    if isinstance(decorator, ast.Attribute) and decorator.attr == "register":
        return _iter_dispatch_parameter_annotation(function_node)
    if (
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "register"
    ):
        if decorator.args:
            return (decorator.args[0],)
        return _iter_dispatch_parameter_annotation(function_node)
    return ()


def _iter_dispatch_parameter_annotation(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.AST, ...]:
    args = function_node.args.args
    if not args:
        return ()
    first_arg = args[0].arg
    dispatch_index = 1 if first_arg in {"self", "cls"} else 0
    if dispatch_index >= len(args):
        return ()
    annotation = args[dispatch_index].annotation
    if annotation is None:
        return ()
    return (annotation,)


def _is_abstract_register_type(type_expr: ast.AST) -> bool:
    if isinstance(type_expr, ast.Name):
        return type_expr.id in _ABSTRACT_REGISTER_TYPES
    if isinstance(type_expr, ast.Attribute):
        parts = _attribute_parts(type_expr)
        if not parts:
            return False
        head = parts[0]
        tail = parts[-1]
        if tail not in _ABSTRACT_REGISTER_TYPES:
            return False
        return head in {"typing", "collections", "abc"}
    if isinstance(type_expr, ast.Subscript):
        return _is_abstract_register_type(type_expr.value)
    if isinstance(type_expr, ast.BinOp) and isinstance(type_expr.op, ast.BitOr):
        return (
            _is_abstract_register_type(type_expr.left)
            or _is_abstract_register_type(type_expr.right)
        )
    if isinstance(type_expr, ast.Tuple):
        return any(_is_abstract_register_type(elt) for elt in type_expr.elts)
    return False


def _attribute_parts(expr: ast.AST) -> tuple[str, ...]:
    if isinstance(expr, ast.Name):
        return (expr.id,)
    if isinstance(expr, ast.Attribute):
        parent = _attribute_parts(expr.value)
        if parent:
            return (*parent, expr.attr)
        return (expr.attr,)
    return ()


def _derive_flow_identity(
    *,
    run_context: CanonicalRunContext,
    rel_path: str,
    qualname: str,
    kind: str,
    line: int,
    input_slot: str,
    guard_form: str,
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
            f"guard:{guard_form}",
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
        "guard_form": violation.guard_form,
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
    batch = build_policy_scan_batch(root=root, target_globs=(TARGET_GLOB,))
    violations = collect_violations(batch=batch)
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
            "Audit manual runtime-type dispatch branches and enforce "
            "singledispatch contract requirements."
        )
    )
    parser.add_argument("--root", default=".", help="Repository root (default: .)")
    parser.add_argument(
        "--out",
        default="artifacts/out/fiber_type_dispatch_contract.json",
        help="Output JSON payload path",
    )
    args = parser.parse_args(argv)
    return run(root=Path(args.root), out=Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
