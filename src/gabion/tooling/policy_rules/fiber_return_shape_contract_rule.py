#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
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

RULE_NAME = "fiber_return_shape_contract"
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
    return_form: str
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


class _ReturnShapeVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        rel_path: str,
        run_context: CanonicalRunContext,
        module_call_counts: Counter[str],
    ) -> None:
        self._rel_path = rel_path
        self._run_context = run_context
        self._module_call_counts = module_call_counts
        self._scope_stack: list[_Scope] = [_Scope(qualname="<module>")]
        self.violations: list[Violation] = []

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

        scope_nodes = _scope_nodes(function_node=node)
        for return_node in filter(_is_non_none_return, scope_nodes):
            return_form = _container_return_form(return_node.value)
            if return_form:
                self._record_violation(
                    node=return_node,
                    kind="container_return_prefer_iterator",
                    message=(
                        "do not return eager container values when an iterator can "
                        "encode the stream"
                    ),
                    return_form=return_form,
                    input_slot="return:eager_container",
                    event_kind="syntax:container_return_prefer_iterator",
                    rationale=(
                        "Expose iterator-producing semantics and collect only at the "
                        "boundary where materialization is required."
                    ),
                )
            iterator_form = _iterator_singleton_return_form(return_node.value)
            if iterator_form:
                self._record_violation(
                    node=return_node,
                    kind="iterator_return_prefer_item",
                    message=(
                        "do not return iterator wrappers when the result cardinality "
                        "is one"
                    ),
                    return_form=iterator_form,
                    input_slot="return:iterator_singleton",
                    event_kind="syntax:iterator_return_prefer_item",
                    rationale=(
                        "Return a scalar value directly when the function emits at "
                        "most one item."
                    ),
                )

        yield_nodes = tuple(filter(_is_yield_node, scope_nodes))
        loop_nodes = tuple(filter(_is_loop_node, scope_nodes))
        if len(yield_nodes) == 1 and not loop_nodes:
            self._record_violation(
                node=yield_nodes[0],
                kind="iterator_return_prefer_item",
                message=(
                    "generator-like function yields exactly one item without loop; "
                    "return scalar value"
                ),
                return_form="single_yield_no_loop",
                input_slot="yield:single_item",
                event_kind="syntax:iterator_return_prefer_item",
                rationale=(
                    "Collapse one-shot generators into scalar-return functions to "
                    "reduce unnecessary iterator carriers."
                ),
            )

        call_count = int(self._module_call_counts.get(node.name, 0))
        if _is_inline_passthrough_candidate(node=node, qualname=qualname, call_count=call_count):
            self._record_violation(
                node=node,
                kind="single_item_return_prefer_inline",
                message=(
                    "single-item passthrough helper should be inlined into its sole "
                    "callsite"
                ),
                return_form="single_return_passthrough",
                input_slot="return:inline_candidate",
                event_kind="syntax:single_item_return_prefer_inline",
                rationale=(
                    "Inline one-use passthrough helpers so control/dataflow remains "
                    "local and branch-free."
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
        return_form: str,
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
            return_form=return_form,
        )
        structured_hash = _structured_hash(
            self._rel_path,
            qualname,
            kind,
            return_form,
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
                return_form=return_form,
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
        visitor = _ReturnShapeVisitor(
            rel_path=module.rel_path,
            run_context=run_context,
            module_call_counts=_module_call_counts(module.tree),
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
        return_form="module_failure",
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
        return_form="module_failure",
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
            rationale="Ensure module parse/read validity before return-shape analysis.",
        ),
    )


def _module_call_counts(module_tree: ast.Module) -> Counter[str]:
    counts = Counter[str]()
    for node in ast.walk(module_tree):
        called_name = _call_func_name(node)
        if not called_name:
            continue
        counts[called_name] += 1
    return counts


def _call_func_name(node: ast.AST) -> str:
    match node:
        case ast.Call(func=func):
            return _called_name(func)
        case _:
            return ""


def _called_name(node: ast.AST) -> str:
    match node:
        case ast.Name(id=name):
            return name
        case _:
            return ""


def _scope_nodes(
    *,
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.AST, ...]:
    nodes: list[ast.AST] = []
    stack: list[ast.AST] = [function_node]
    while stack:
        node = stack.pop()
        nodes.append(node)
        for child in ast.iter_child_nodes(node):
            if _is_scope_break_node(child):
                continue
            stack.append(child)
    return tuple(nodes)


def _is_scope_break_node(node: ast.AST) -> bool:
    match node:
        case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.ClassDef() | ast.Lambda():
            return True
        case _:
            return False


def _is_non_none_return(node: ast.AST) -> bool:
    match node:
        case ast.Return(value=value):
            return value is not None
        case _:
            return False


def _is_loop_node(node: ast.AST) -> bool:
    match node:
        case ast.For() | ast.AsyncFor() | ast.While() | ast.comprehension():
            return True
        case _:
            return False


def _is_yield_node(node: ast.AST) -> bool:
    match node:
        case ast.Yield() | ast.YieldFrom():
            return True
        case _:
            return False


def _container_return_form(value: ast.AST) -> str:
    match value:
        case ast.ListComp():
            return "list_comprehension"
        case ast.SetComp():
            return "set_comprehension"
        case ast.DictComp():
            return "dict_comprehension"
        case ast.List(elts=elts) if elts:
            return "list_literal"
        case ast.Tuple(elts=elts) if elts:
            return "tuple_literal"
        case ast.Set(elts=elts) if elts:
            return "set_literal"
        case ast.Dict(keys=keys) if keys:
            return "dict_literal"
        case ast.Call(func=ast.Name(id=ctor_name), args=_args, keywords=keywords):
            if ctor_name in {"list", "tuple", "set", "dict"} and not keywords:
                return f"{ctor_name}_constructor"
            return ""
        case _:
            return ""


def _iterator_singleton_return_form(value: ast.AST) -> str:
    match value:
        case ast.Call(func=ast.Name(id="iter"), args=[iter_arg], keywords=[]):
            if _is_singleton_container(iter_arg):
                return "iter_singleton"
            return ""
        case ast.GeneratorExp(generators=[generator], elt=_):
            if generator.ifs:
                return ""
            if _is_singleton_container(generator.iter):
                return "generator_singleton"
            return ""
        case _:
            return ""


def _is_singleton_container(node: ast.AST) -> bool:
    match node:
        case ast.List(elts=elts) | ast.Tuple(elts=elts) | ast.Set(elts=elts):
            return len(elts) == 1
        case ast.Dict(keys=keys):
            return len(keys) == 1
        case _:
            return False


def _is_inline_passthrough_candidate(
    *,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    qualname: str,
    call_count: int,
) -> bool:
    if "." in qualname:
        return False
    if not node.name.startswith("_") or node.name.startswith("__"):
        return False
    if node.decorator_list:
        return False
    if len(node.body) != 1:
        return False
    call = _single_return_call(node.body[0])
    if call is None:
        return False
    if call_count > 1:
        return False
    return _is_passthrough_call(node=node, call=call)


def _single_return_call(statement: ast.stmt):
    match statement:
        case ast.Return(value=ast.Call() as call):
            return call
        case _:
            return None


def _is_passthrough_call(
    *,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    call: ast.Call,
) -> bool:
    parameter_names = _parameter_names(node.args)
    if not parameter_names:
        return False
    if not call.args and not call.keywords:
        return False
    return _all_passthrough_args(call=call, parameter_names=parameter_names)


def _parameter_names(arguments: ast.arguments) -> frozenset[str]:
    names = tuple(
        parameter.arg
        for parameter in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    )
    return frozenset(names)


def _all_passthrough_args(*, call: ast.Call, parameter_names: frozenset[str]) -> bool:
    positional_ok = all(_arg_is_parameter_reference(arg=arg, parameter_names=parameter_names) for arg in call.args)
    if not positional_ok:
        return False
    for keyword in call.keywords:
        if keyword.arg is None:
            return False
        if not _arg_is_parameter_reference(arg=keyword.value, parameter_names=parameter_names):
            return False
    return True


def _arg_is_parameter_reference(*, arg: ast.AST, parameter_names: frozenset[str]) -> bool:
    match arg:
        case ast.Name(id=name):
            return name in parameter_names
        case _:
            return False


def _derive_flow_identity(
    *,
    run_context: CanonicalRunContext,
    rel_path: str,
    qualname: str,
    kind: str,
    line: int,
    input_slot: str,
    return_form: str,
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
            f"return_form:{return_form}",
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
        "return_form": violation.return_form,
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
            "Enforce return-shape strictification: prefer iterator over eager "
            "containers, scalar over singleton iterator, and inline over one-use "
            "single-return passthrough helpers."
        )
    )
    parser.add_argument("--root", default=".", help="Repository root (default: .)")
    parser.add_argument(
        "--out",
        default="artifacts/out/fiber_return_shape_contract.json",
        help="Output JSON payload path",
    )
    args = parser.parse_args(argv)
    return run(root=Path(args.root), out=Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
