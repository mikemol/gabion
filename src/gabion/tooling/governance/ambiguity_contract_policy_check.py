#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass, field
from functools import singledispatch
from pathlib import Path
from typing import Mapping

from gabion.invariants import never
from gabion.policy_dsl import PolicyDomain, evaluate_policy
from gabion.policy_dsl.schema import PolicyOutcomeKind
from gabion.runtime.deadline_policy import DeadlineBudget, deadline_scope_from_ticks
from gabion.tooling.policy_substrate.grade_monotonicity_semantic import (
    SemanticGradeMonotonicityReport,
    collect_grade_monotonicity,
)
from gabion.tooling.runtime.policy_scan_batch import (
    PolicyScanBatch,
    build_policy_scan_batch,
    iter_failure_seeds,
)

TARGETS = (
    "src/gabion/analysis/**/*.py",
    "src/gabion/synthesis/**/*.py",
    "src/gabion/refactor/**/*.py",
)
BASELINE_VERSION = 1
MODULE_MARKER = "gabion:ambiguity_boundary_module"
FUNCTION_MARKER = "gabion:ambiguity_boundary"
DEFAULT_BASELINE_RELATIVE_PATH = Path("baselines/ambiguity_contract_policy_baseline.json")
ARTIFACT_RELATIVE_PATH = Path("artifacts/out/ambiguity_contract_policy_check.json")
_AMBIGUITY_CONTRACT_DEADLINE_BUDGET = DeadlineBudget(
    ticks=100_000_000,
    tick_ns=1_000_000,
)


@dataclass(frozen=True)
class Violation:
    rule_id: str
    path: str
    line: int
    column: int
    qualname: str
    message: str
    details: Mapping[str, object] = field(default_factory=dict)
    key_override: str | None = None

    @property
    def key(self) -> str:
        if self.key_override is not None:
            return self.key_override
        # Baseline identity is line-insensitive so routine line motion in
        # unchanged semantics does not churn policy deltas.
        return f"{self.rule_id}:{self.path}:{self.qualname}"

    def render(self) -> str:
        lines = [
            f"{self.path}:{self.line}:{self.column}: [{self.rule_id}] [{self.qualname}] {self.message}"
        ]
        lines.extend(_render_guidance_lines(_guidance_payload(self.details)))
        return "\n".join(lines)


@dataclass(frozen=True)
class _Scope:
    qualname: str
    is_boundary: bool


class _Visitor(ast.NodeVisitor):
    def __init__(self, *, rel_path: str, source_lines: list[str]) -> None:
        self.rel_path = rel_path
        self.source_lines = source_lines
        self.violations: list[Violation] = []
        self.module_boundary = _module_boundary(source_lines)
        self.scope_stack: list[_Scope] = []

    def visit_Module(self, node: ast.Module) -> None:
        if not self.module_boundary:
            self._scan_stmt_sequence(node.body)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._scope_boundary:
            self.generic_visit(node)
            return
        if _is_isinstance_call(node):
            self._report_event(node, event="runtime_isinstance_call")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not self._scope_boundary:
            self._check_annotation(node.annotation, node)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        if not self._scope_boundary and node.annotation is not None:
            self._check_annotation(node.annotation, node)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if self._scope_boundary:
            self.generic_visit(node)
            return
        if _looks_like_guard(node.test):
            sentinel = _single_sentinel_stmt(node.body)
            if sentinel is not None:
                if _contains_none_guard(node.test):
                    self._report_event(
                        node,
                        event="nullable_contract_control",
                        sentinel=sentinel,
                    )
                else:
                    self._report_event(node, event="sentinel_control", sentinel=sentinel)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        if self._scope_boundary:
            self.generic_visit(node)
            return
        for case in node.cases:
            if _is_fallthrough_case(case) and not _body_calls_never(case.body):
                self._report_event(case.pattern, event="match_fallthrough_without_never")
        self.generic_visit(node)

    @property
    def _scope_boundary(self) -> bool:
        if self.scope_stack:
            return self.scope_stack[-1].is_boundary
        return self.module_boundary

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        parent = self.scope_stack[-1].qualname if self.scope_stack else "<module>"
        qualname = node.name if parent == "<module>" else f"{parent}.{node.name}"
        is_boundary = self.module_boundary or _has_marker(self.source_lines, int(getattr(node, "lineno", 1)), FUNCTION_MARKER)
        self.scope_stack.append(_Scope(qualname=qualname, is_boundary=is_boundary))
        if not is_boundary and node.returns is not None:
            self._check_annotation(node.returns, node)
        if not is_boundary:
            self._scan_stmt_sequence(node.body)
        self.generic_visit(node)
        self.scope_stack.pop()

    def _scan_stmt_sequence(self, body: list[ast.stmt]) -> None:
        for index, stmt in enumerate(body):
            if _is_probe_state_recovery(body, probe_index=index):
                self._report_event(stmt, event="probe_state_recovery")
            for nested_body in _nested_stmt_bodies(stmt):
                self._scan_stmt_sequence(nested_body)

    def _check_annotation(self, annotation: ast.AST, node: ast.AST) -> None:
        if _annotation_is_dynamic(annotation):
            self._report_event(node, event="dynamic_annotation")

    def _report_event(self, node: ast.AST, *, event: str, sentinel: str | None = None) -> None:
        decision = evaluate_policy(
            domain=PolicyDomain.AMBIGUITY_CONTRACT_AST,
            data={"event": event},
        )
        if decision.outcome is PolicyOutcomeKind.PASS:
            return
        message = decision.message if sentinel is None else f"{decision.message} ({sentinel})"
        self._report(
            node,
            rule_id=decision.rule_id,
            message=message,
            details=decision.details,
        )

    def _report(
        self,
        node: ast.AST,
        *,
        rule_id: str,
        message: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        scope = self.scope_stack[-1] if self.scope_stack else _Scope("<module>", self.module_boundary)
        self.violations.append(
            Violation(
                rule_id=rule_id,
                path=self.rel_path,
                line=int(getattr(node, "lineno", 1)),
                column=int(getattr(node, "col_offset", 0)) + 1,
                qualname=scope.qualname,
                message=message,
                details={} if details is None else dict(details),
            )
        )


def _module_boundary(source_lines: list[str]) -> bool:
    for raw in source_lines[:80]:
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#") and MODULE_MARKER in stripped:
            return True
        if stripped.startswith("\"\"\"") or stripped.startswith("'''"):
            continue
    return False


def _has_marker(source_lines: list[str], line: int, marker: str) -> bool:
    idx = max(0, line - 2)
    while idx >= 0:
        stripped = source_lines[idx].strip()
        if not stripped:
            idx -= 1
            continue
        return stripped.startswith("#") and marker in stripped
    return False


@singledispatch
def _name_id_optional(node: object) -> str | None:
    never("unregistered runtime type", value_type=type(node).__name__)


@_name_id_optional.register(ast.Name)
def _sd_reg_1(node: ast.Name) -> str | None:
    return node.id


@_name_id_optional.register(ast.AST)
def _sd_reg_2(node: ast.AST) -> str | None:
    _ = node
    return None


@singledispatch
def _is_isinstance_call(node: object) -> bool:
    never("unregistered runtime type", value_type=type(node).__name__)


@_is_isinstance_call.register(ast.Call)
def _sd_reg_3(node: ast.Call) -> bool:
    return _name_id_optional(node.func) == "isinstance"


@_is_isinstance_call.register(ast.AST)
def _sd_reg_4(node: ast.AST) -> bool:
    _ = node
    return False


@singledispatch
def _is_none_constant(node: object) -> bool:
    never("unregistered runtime type", value_type=type(node).__name__)


@_is_none_constant.register(type(None))
def _sd_reg_5(node: None) -> bool:
    _ = node
    return True


@_is_none_constant.register(ast.Constant)
def _sd_reg_6(node: ast.Constant) -> bool:
    return node.value is None


@_is_none_constant.register(ast.AST)
def _sd_reg_7(node: ast.AST) -> bool:
    _ = node
    return False


@singledispatch
def _compare_contains_none_constant(node: object) -> bool:
    never("unregistered runtime type", value_type=type(node).__name__)


@_compare_contains_none_constant.register(ast.Compare)
def _sd_reg_8(node: ast.Compare) -> bool:
    values = [node.left, *node.comparators]
    for value in values:
        if _is_none_constant(value):
            return True
    return False


@_compare_contains_none_constant.register(ast.AST)
def _sd_reg_9(node: ast.AST) -> bool:
    _ = node
    return False


@singledispatch
def _subscript_value_is_dynamic_alias(node: object) -> bool:
    never("unregistered runtime type", value_type=type(node).__name__)


@_subscript_value_is_dynamic_alias.register(ast.Name)
def _sd_reg_10(node: ast.Name) -> bool:
    return node.id in {"Optional", "Union"}


@_subscript_value_is_dynamic_alias.register(ast.AST)
def _sd_reg_11(node: ast.AST) -> bool:
    _ = node
    return False


@singledispatch
def _is_bit_or_operator(node: object) -> bool:
    never("unregistered runtime type", value_type=type(node).__name__)


@_is_bit_or_operator.register(ast.BitOr)
def _sd_reg_12(node: ast.BitOr) -> bool:
    _ = node
    return True


@_is_bit_or_operator.register(ast.operator)
def _sd_reg_13(node: ast.operator) -> bool:
    _ = node
    return False


@singledispatch
def _annotation_is_dynamic(node: object) -> bool:
    never("unregistered runtime type", value_type=type(node).__name__)


@_annotation_is_dynamic.register(ast.Name)
def _sd_reg_14(node: ast.Name) -> bool:
    return node.id in {"Any", "Optional", "Union"}


@_annotation_is_dynamic.register(ast.Subscript)
def _sd_reg_15(node: ast.Subscript) -> bool:
    if _subscript_value_is_dynamic_alias(node.value):
        return True
    return _annotation_is_dynamic(node.value) or _annotation_is_dynamic(node.slice)


@_annotation_is_dynamic.register(ast.BinOp)
def _sd_reg_16(node: ast.BinOp) -> bool:
    if _is_bit_or_operator(node.op):
        return True
    return _annotation_is_dynamic(node.left) or _annotation_is_dynamic(node.right)


@_annotation_is_dynamic.register(ast.AST)
def _sd_reg_17(node: ast.AST) -> bool:
    for child in ast.iter_child_nodes(node):
        if _annotation_is_dynamic(child):
            return True
    return False


def _looks_like_guard(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if _is_isinstance_call(child):
            return True
        if _compare_contains_none_constant(child):
            return True
    return False


def _contains_none_guard(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if _compare_contains_none_constant(child):
            return True
    return False


@singledispatch
def _sentinel_return_value(value: object) -> str | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_sentinel_return_value.register(type(None))
def _sd_reg_18(value: None) -> str | None:
    _ = value
    return "return None"


@_sentinel_return_value.register(ast.Constant)
def _sd_reg_19(value: ast.Constant) -> str | None:
    if value.value is None:
        return "return None"
    return None


@_sentinel_return_value.register(ast.List)
def _sd_reg_20(value: ast.List) -> str | None:
    if len(value.elts) == 0:
        return "return []"
    return None


@_sentinel_return_value.register(ast.AST)
def _sd_reg_21(value: ast.AST) -> str | None:
    _ = value
    return None


@singledispatch
def _sentinel_stmt_value(stmt: object) -> str | None:
    never("unregistered runtime type", value_type=type(stmt).__name__)


@_sentinel_stmt_value.register(ast.Return)
def _sd_reg_22(stmt: ast.Return) -> str | None:
    return _sentinel_return_value(stmt.value)


@_sentinel_stmt_value.register(ast.Continue)
def _sd_reg_23(stmt: ast.Continue) -> str | None:
    _ = stmt
    return "continue"


@_sentinel_stmt_value.register(ast.Pass)
def _sd_reg_24(stmt: ast.Pass) -> str | None:
    _ = stmt
    return "pass"


@_sentinel_stmt_value.register(ast.stmt)
def _sd_reg_25(stmt: ast.stmt) -> str | None:
    _ = stmt
    return None


def _single_sentinel_stmt(body: list[ast.stmt]) -> str | None:
    if len(body) != 1:
        return None
    return _sentinel_stmt_value(body[0])


def _is_fallthrough_case(case: ast.match_case) -> bool:
    match case.pattern:
        case ast.MatchAs(pattern=None, name=None):
            return True
        case _:
            return False


def _body_calls_never(body: list[ast.stmt]) -> bool:
    for stmt in body:
        for node in ast.walk(stmt):
            match node:
                case ast.Call(func=ast.Name(id="never")):
                    return True
                case _:
                    continue
    return False


def _guidance_payload(details: Mapping[str, object]) -> dict[object, object] | None:
    guidance = _dict_optional(details.get("guidance"))
    if guidance is not None:
        return guidance
    direct_guidance = _dict_optional(details)
    if direct_guidance is None:
        return None
    if any(key in direct_guidance for key in ("why", "prefer", "avoid", "playbook_ref")):
        return direct_guidance
    return None


def _render_guidance_lines(guidance: Mapping[object, object] | None) -> list[str]:
    if guidance is None:
        return []
    lines: list[str] = []
    why = _str_optional(guidance.get("why"))
    prefer = _str_optional(guidance.get("prefer"))
    playbook = _str_optional(guidance.get("playbook_ref"))
    if why is not None:
        lines.append(f"      why: {why}")
    if prefer is not None:
        lines.append(f"      prefer: {prefer}")
    if playbook is not None:
        lines.append(f"      playbook: {playbook}")
    lines.extend(_render_avoid_guidance_lines(guidance.get("avoid")))
    return lines


def _render_avoid_guidance_lines(raw_avoid: object) -> list[str]:
    rendered: list[str] = []
    avoid_value = _str_optional(raw_avoid)
    if avoid_value is not None:
        return [f"      avoid: {avoid_value}"]
    avoid_entries = _list_optional(raw_avoid)
    if avoid_entries is None:
        return rendered
    for item in avoid_entries:
        text = _str_optional(item)
        if text is not None:
            rendered.append(f"      avoid: {text}")
    return rendered


def _nested_stmt_bodies(stmt: ast.stmt) -> tuple[list[ast.stmt], ...]:
    match stmt:
        case ast.If(body=body, orelse=orelse):
            return (body, orelse)
        case ast.For(body=body, orelse=orelse) | ast.AsyncFor(body=body, orelse=orelse):
            return (body, orelse)
        case ast.While(body=body, orelse=orelse):
            return (body, orelse)
        case ast.With(body=body) | ast.AsyncWith(body=body):
            return (body,)
        case ast.Try(body=body, handlers=handlers, orelse=orelse, finalbody=finalbody):
            handler_bodies = tuple(handler.body for handler in handlers)
            return (body, *handler_bodies, orelse, finalbody)
        case ast.Match(cases=cases):
            return tuple(case.body for case in cases)
    return ()


def _is_probe_state_recovery(body: list[ast.stmt], *, probe_index: int) -> bool:
    if probe_index + 1 >= len(body):
        return False
    tracked_names = _contiguous_provisional_seed_names(body, stop_index=probe_index)
    if not tracked_names:
        return False
    probe_stmt = body[probe_index]
    branch_stmt = body[probe_index + 1]
    if not isinstance(branch_stmt, ast.If):
        return False
    mutated_names = _probe_mutated_names(probe_stmt, tracked_names)
    if not mutated_names:
        return False
    branch_names = _immediate_branch_names(branch_stmt.test, mutated_names)
    return branch_names != frozenset()


def _contiguous_provisional_seed_names(
    body: list[ast.stmt],
    *,
    stop_index: int,
) -> tuple[str, ...]:
    provisional_names: list[str] = []
    for index in range(stop_index - 1, -1, -1):
        name = _provisional_seed_name(body[index])
        if name is None:
            break
        provisional_names.append(name)
    provisional_names.reverse()
    return tuple(provisional_names)


def _provisional_seed_name(stmt: ast.stmt) -> str | None:
    match stmt:
        case ast.Assign(targets=[ast.Name(id=name)], value=value):
            if _is_provisional_seed_expr(value):
                return name
        case ast.AnnAssign(target=ast.Name(id=name), value=value):
            if value is not None and _is_provisional_seed_expr(value):
                return name
    return None


def _is_provisional_seed_expr(value: ast.expr) -> bool:
    match value:
        case ast.Constant(value=seed_value):
            return isinstance(seed_value, (bool, int, str)) or seed_value is None
    return False


def _probe_mutated_names(
    stmt: ast.stmt,
    tracked_names: tuple[str, ...],
) -> frozenset[str]:
    tracked_name_set = frozenset(tracked_names)
    match stmt:
        case ast.Match(cases=cases):
            return frozenset().union(
                *(
                    _assigned_names_in_body(case.body, tracked_name_set)
                    for case in cases
                )
            )
        case ast.If(test=test, body=body, orelse=orelse):
            if not _contains_runtime_type_probe(test):
                return frozenset()
            return _assigned_names_in_body(body, tracked_name_set) | _assigned_names_in_body(
                orelse,
                tracked_name_set,
            )
    return frozenset()


def _assigned_names_in_body(
    body: list[ast.stmt],
    tracked_names: frozenset[str],
) -> frozenset[str]:
    assigned: set[str] = set()
    for stmt in body:
        for node in ast.walk(stmt):
            match node:
                case ast.Name(id=name, ctx=ast.Store()) if name in tracked_names:
                    assigned.add(name)
    return frozenset(assigned)


def _immediate_branch_names(
    test: ast.AST,
    candidate_names: frozenset[str],
) -> frozenset[str]:
    match test:
        case ast.Name(id=name) if name in candidate_names:
            return frozenset({name})
        case ast.UnaryOp(op=ast.Not(), operand=ast.Name(id=name)) if name in candidate_names:
            return frozenset({name})
    return frozenset()


def _contains_runtime_type_probe(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if _is_isinstance_call(child):
            return True
    return False


@singledispatch
def _dict_optional(value: object) -> dict[object, object] | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_dict_optional.register(dict)
def _sd_reg_26(value: dict[object, object]) -> dict[object, object] | None:
    return value


def _none_dict(value: object) -> dict[object, object] | None:
    _ = value
    return None


for _dict_none_type in (list, tuple, set, str, int, float, bool, type(None)):
    _dict_optional.register(_dict_none_type)(_none_dict)


@singledispatch
def _list_optional(value: object) -> list[object] | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_list_optional.register(list)
def _sd_reg_27(value: list[object]) -> list[object] | None:
    return value


@_list_optional.register(tuple)
def _sd_reg_28(value: tuple[object, ...]) -> list[object] | None:
    return list(value)


def _none_list(value: object) -> list[object] | None:
    _ = value
    return None


for _list_none_type in (dict, set, str, int, float, bool, type(None)):
    _list_optional.register(_list_none_type)(_none_list)


@singledispatch
def _str_optional(value: object) -> str | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_str_optional.register(str)
def _sd_reg_29(value: str) -> str | None:
    return value


def _none_str(value: object) -> str | None:
    _ = value
    return None


for _str_none_type in (dict, list, tuple, set, int, float, bool, type(None)):
    _str_optional.register(_str_none_type)(_none_str)


def _baseline_violation_entries(payload: object) -> list[dict[object, object]]:
    payload_mapping = _dict_optional(payload)
    if payload_mapping is None:
        return []
    raw = _list_optional(payload_mapping.get("violations"))
    if raw is None:
        return []
    entries: list[dict[object, object]] = []
    for item in raw:
        mapping = _dict_optional(item)
        if mapping is not None:
            entries.append(mapping)
    return entries


def collect_violations(*, batch: PolicyScanBatch) -> list[Violation]:
    violations: list[Violation] = []
    for seed in iter_failure_seeds(batch=batch):
        message = (
            "unable to read file while checking ambiguity contract"
            if seed.kind == "read_error"
            else "syntax error while checking ambiguity contract"
        )
        violations.append(
            Violation(
                rule_id="ACP-000",
                path=seed.path,
                line=seed.line,
                column=seed.column,
                qualname="<module>",
                message=message,
            )
        )
    for module in batch.modules:
        visitor = _Visitor(rel_path=module.rel_path, source_lines=module.source.splitlines())
        visitor.visit(module.tree)
        violations.extend(visitor.violations)
    return violations


def _load_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    keys: set[str] = set()
    for item in _baseline_violation_entries(payload):
        explicit_key = _str_optional(item.get("key"))
        if explicit_key is not None:
            keys.add(explicit_key)
            continue
        rule_id = _str_optional(item.get("rule_id"))
        path_value = _str_optional(item.get("path"))
        qualname = _str_optional(item.get("qualname"))
        if rule_id is not None and path_value is not None and qualname is not None:
            keys.add(f"{rule_id}:{path_value}:{qualname}")
    return keys


def _write_baseline(path: Path, violations: list[Violation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": BASELINE_VERSION,
        "violations": [
            {
                "key": item.key,
                "rule_id": item.rule_id,
                "path": item.path,
                "qualname": item.qualname,
            }
            for item in sorted(violations, key=lambda v: (v.rule_id, v.path, v.qualname, v.line, v.column))
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resolve_baseline_path(*, root: Path, baseline: Path | None) -> Path:
    if baseline is None:
        return (root / DEFAULT_BASELINE_RELATIVE_PATH).resolve()
    return baseline.resolve()


def _grade_violation_to_violation(item: object) -> Violation:
    if not hasattr(item, "rule_id"):
        never("grade violation payload missing rule_id")
    if not hasattr(item, "path"):
        never("grade violation payload missing path")
    if not hasattr(item, "line"):
        never("grade violation payload missing line")
    if not hasattr(item, "column"):
        never("grade violation payload missing column")
    if not hasattr(item, "qualname"):
        never("grade violation payload missing qualname")
    if not hasattr(item, "message"):
        never("grade violation payload missing message")
    if not hasattr(item, "details"):
        never("grade violation payload missing details")
    if not hasattr(item, "key"):
        never("grade violation payload missing key")
    return Violation(
        rule_id=str(getattr(item, "rule_id")),
        path=str(getattr(item, "path")),
        line=int(getattr(item, "line")),
        column=int(getattr(item, "column")),
        qualname=str(getattr(item, "qualname")),
        message=str(getattr(item, "message")),
        details=dict(getattr(item, "details")),
        key_override=str(getattr(item, "key")),
    )


def _decision_payload(decision: object) -> dict[str, object]:
    if not hasattr(decision, "rule_id"):
        never("decision payload missing rule_id")
    if not hasattr(decision, "outcome"):
        never("decision payload missing outcome")
    if not hasattr(decision, "message"):
        never("decision payload missing message")
    if not hasattr(decision, "details"):
        never("decision payload missing details")
    outcome = getattr(decision, "outcome")
    if not hasattr(outcome, "value"):
        never("decision outcome missing wire value")
    return {
        "rule_id": str(getattr(decision, "rule_id")),
        "outcome": str(getattr(outcome, "value")),
        "message": str(getattr(decision, "message")),
        "details": dict(getattr(decision, "details")),
    }


def _artifact_payload(
    *,
    root: Path,
    baseline_path: Path,
    ast_violations: list[Violation],
    new_ast_violations: list[Violation],
    grade_report: SemanticGradeMonotonicityReport,
    grade_violations: list[Violation],
    new_grade_violations: list[Violation],
    ambiguity_decision: object,
    grade_decision: object,
    baseline_keys: set[str],
) -> dict[str, object]:
    return {
        "version": BASELINE_VERSION,
        "root": root.as_posix(),
        "baseline_path": baseline_path.as_posix(),
        "ast": {
            "violation_count": len(ast_violations),
            "new_violation_count": len(new_ast_violations),
            "violations": [_violation_payload(item) for item in ast_violations],
        },
        "grade": {
            "violation_count": len(grade_violations),
            "new_violation_count": len(new_grade_violations),
            **grade_report.policy_data(),
        },
        "merged": {
            "violation_count": len(ast_violations) + len(grade_violations),
            "new_violation_count": len(new_ast_violations) + len(new_grade_violations),
            "baseline_key_count": len(baseline_keys),
        },
        "decisions": {
            "ambiguity_contract": _decision_payload(ambiguity_decision),
            "grade_monotonicity": _decision_payload(grade_decision),
        },
    }


def _violation_payload(item: Violation) -> dict[str, object]:
    return {
        "key": item.key,
        "rule_id": item.rule_id,
        "path": item.path,
        "line": item.line,
        "column": item.column,
        "qualname": item.qualname,
        "message": item.message,
        "details": dict(item.details),
    }


def _write_artifact(*, path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(root: Path, baseline: Path | None, baseline_write: bool) -> int:
    with deadline_scope_from_ticks(_AMBIGUITY_CONTRACT_DEADLINE_BUDGET):
        batch = build_policy_scan_batch(root=root, target_globs=TARGETS)
        ast_violations = collect_violations(batch=batch)
        grade_report = collect_grade_monotonicity(batch=batch)
        grade_violations = [_grade_violation_to_violation(item) for item in grade_report.violations]
        violations = sorted(
            [*ast_violations, *grade_violations],
            key=lambda v: (v.rule_id, v.path, v.qualname, v.line, v.column, v.message),
        )
        baseline_path = _resolve_baseline_path(root=root, baseline=baseline)
        baseline_keys = _load_baseline(baseline_path)
        new_ast_violations = [item for item in ast_violations if item.key not in baseline_keys]
        new_grade_violations = [item for item in grade_violations if item.key not in baseline_keys]
        ambiguity_decision = evaluate_policy(
            domain=PolicyDomain.AMBIGUITY_CONTRACT,
            data={"new_violations": len(new_ast_violations)},
        )
        grade_decision = evaluate_policy(
            domain=PolicyDomain.GRADE_MONOTONICITY,
            data={"new_violations": len(new_grade_violations)},
        )
        _write_artifact(
            path=root / ARTIFACT_RELATIVE_PATH,
            payload=_artifact_payload(
                root=root,
                baseline_path=baseline_path,
                ast_violations=ast_violations,
                new_ast_violations=new_ast_violations,
                grade_report=grade_report,
                grade_violations=grade_violations,
                new_grade_violations=new_grade_violations,
                ambiguity_decision=ambiguity_decision,
                grade_decision=grade_decision,
                baseline_keys=baseline_keys,
            ),
        )
    if baseline_write:
        _write_baseline(baseline_path, violations)
        print(f"wrote ambiguity-contract baseline: {baseline_path}")
        return 0

    blocked = False
    if ambiguity_decision.outcome is PolicyOutcomeKind.BLOCK:
        blocked = True
        print(f"{ambiguity_decision.message}:")
        for line in _render_guidance_lines(_guidance_payload(dict(ambiguity_decision.details))):
            print(line)
        for item in sorted(new_ast_violations, key=lambda v: (v.rule_id, v.path, v.line, v.column)):
            print(f"  - {item.render()}")
    if grade_decision.outcome is PolicyOutcomeKind.BLOCK:
        blocked = True
        print(f"{grade_decision.message}:")
        for line in _render_guidance_lines(_guidance_payload(dict(grade_decision.details))):
            print(line)
        for item in sorted(new_grade_violations, key=lambda v: (v.rule_id, v.path, v.line, v.column)):
            print(f"  - {item.render()}")
    if blocked:
        return 1
    print("ambiguity contract policy check passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shift-ambiguity-left contract policy check")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-write", action="store_true")
    args = parser.parse_args(argv)
    return run(root=args.root.resolve(), baseline=args.baseline, baseline_write=bool(args.baseline_write))


if __name__ == "__main__":
    raise SystemExit(main())
