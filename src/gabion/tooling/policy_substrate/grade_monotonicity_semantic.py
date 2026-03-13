from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from gabion.analysis.dataflow.engine.dataflow_contracts import CallArgs, FunctionInfo
from gabion.analysis.dataflow.engine.dataflow_deadline_helpers import (
    _build_analysis_index,
    _collect_call_edges,
    _collect_recursive_functions,
    _resolve_callee_outcome,
)
from gabion.analysis.dataflow.engine.dataflow_evidence_helpers import _module_name
from gabion.analysis.projection.decision_flow import (
    build_decision_tables,
    detect_repeated_guard_bundles,
)
from gabion.frontmatter import parse_strict_yaml_frontmatter
from gabion.order_contract import sort_once
from gabion.tooling.policy_substrate.dataflow_fibration import (
    CallEdgeGradeWitness,
    DeterminismCostGrade,
    GradeBoundaryKind,
    GradeBoundaryMarker,
    GradeMonotonicityViolation,
    OutputCardinalityClass,
    ProtocolDischargeLevel,
    WorkGrowthClass,
    canonical_structural_identity,
    canonical_site_identity,
)
from gabion.tooling.runtime.policy_scan_batch import PolicyScanBatch

_GRADE_BOUNDARY_RE = re.compile(r"gabion:grade_boundary\b(?P<fields>.*)$")
_ALLOWED_GRADE_BOUNDARY_KINDS = frozenset(item.value for item in GradeBoundaryKind)
_DOMAIN_TOP = 999
_DECISION_PROTOCOL_COMMENT = "gabion:decision_protocol"
_DECISION_PROTOCOL_MODULE_COMMENT = "gabion:decision_protocol_module"
_BOUNDARY_NORMALIZATION_COMMENT = "gabion:boundary_normalization"
_BOUNDARY_NORMALIZATION_MODULE_COMMENT = "gabion:boundary_normalization_module"
_REDUCER_NAMES = frozenset({"sum", "all", "any", "max", "min", "len", "reduce"})
_MATERIALIZER_NAMES = frozenset({"list", "tuple", "set", "dict", "sorted"})
_MARKDOWN_ANCHOR_RE = re.compile(r'<a\s+id="(?P<anchor>[a-z0-9][a-z0-9_-]*)"\s*></a>')
_SHAPE_TYPE_NAMES = frozenset(
    {
        "dict",
        "list",
        "tuple",
        "set",
        "frozenset",
        "mapping",
        "sequence",
        "iterable",
        "jsonobject",
        "jsonvalue",
        "json",
    }
)
_GRADE_MONOTONICITY_PLAYBOOK_RELATIVE_PATH = "docs/policy_rules/grade_monotonicity.md"
_VIOLATION_MESSAGES = {
    "GMP-001": "callee expands nullable contract beyond caller",
    "GMP-002": "callee expands runtime type domain beyond caller",
    "GMP-003": "callee expands structural shape domain beyond caller",
    "GMP-004": "callee reintroduces runtime classification work",
    "GMP-005": "callee regresses protocol discharge level",
    "GMP-006": "callee expands output cardinality without an explicit boundary",
    "GMP-007": "callee expands work growth without an explicit boundary",
}


@dataclass(frozen=True)
class _PlaybookGuidance:
    why: str
    prefer: tuple[str, ...]
    avoid: tuple[str, ...]

    def as_payload(self, *, playbook_ref: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "why": self.why,
            "playbook_ref": playbook_ref,
        }
        if self.prefer:
            payload["prefer"] = "; ".join(self.prefer)
        if self.avoid:
            payload["avoid"] = list(self.avoid)
        return payload


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


@lru_cache(maxsize=1)
def _grade_playbook_guidance_by_rule() -> dict[str, _PlaybookGuidance]:
    text = (_repo_root() / _GRADE_MONOTONICITY_PLAYBOOK_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    _, body = parse_strict_yaml_frontmatter(text, require_parser=True)
    sections = _playbook_sections(body)
    guidance_by_rule: dict[str, _PlaybookGuidance] = {}
    missing_anchors: list[str] = []
    for rule_id in sorted(_VIOLATION_MESSAGES):
        anchor = rule_id.lower()
        section = sections.get(anchor)
        if section is None:
            missing_anchors.append(anchor)
            continue
        guidance_by_rule[rule_id] = section
    if missing_anchors:
        raise ValueError(
            "missing grade-monotonicity playbook anchors: "
            + ", ".join(missing_anchors)
        )
    return guidance_by_rule


def _playbook_sections(body: str) -> dict[str, _PlaybookGuidance]:
    sections: dict[str, _PlaybookGuidance] = {}
    current_anchor: str | None = None
    current_lines: list[str] = []

    def _flush() -> None:
        if current_anchor is None:
            return
        section = _parse_playbook_section(current_lines)
        if section is not None:
            sections[current_anchor] = section

    for raw_line in body.splitlines():
        match = _MARKDOWN_ANCHOR_RE.fullmatch(raw_line.strip())
        if match is not None:
            _flush()
            current_anchor = match.group("anchor")
            current_lines = []
            continue
        if current_anchor is not None:
            current_lines.append(raw_line)
    _flush()
    return sections


def _parse_playbook_section(lines: list[str]) -> _PlaybookGuidance | None:
    meaning: str | None = None
    preferred: list[str] = []
    avoid: list[str] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("Meaning:"):
            meaning = stripped.removeprefix("Meaning:").strip()
            index += 1
            continue
        if stripped == "Preferred response:":
            bullets, index = _consume_bullet_block(lines, start=index + 1)
            preferred.extend(bullets)
            continue
        if stripped == "Avoid:":
            bullets, index = _consume_bullet_block(lines, start=index + 1)
            avoid.extend(bullets)
            continue
        index += 1
    if meaning is None:
        return None
    return _PlaybookGuidance(
        why=meaning,
        prefer=tuple(preferred),
        avoid=tuple(avoid),
    )


def _consume_bullet_block(
    lines: list[str], *, start: int
) -> tuple[list[str], int]:
    items: list[str] = []
    index = start
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if not stripped.startswith("- "):
            break
        items.append(stripped.removeprefix("- ").strip())
        index += 1
    return items, index


def _violation_guidance(rule_id: str) -> dict[str, object]:
    guidance = _grade_playbook_guidance_by_rule()[rule_id]
    return guidance.as_payload(
        playbook_ref=f"{_GRADE_MONOTONICITY_PLAYBOOK_RELATIVE_PATH}#{rule_id.lower()}"
    )


@dataclass(frozen=True)
class _ModuleContext:
    path: Path
    rel_path: str
    source_lines: tuple[str, ...]
    decision_protocol_module: bool
    boundary_marker: GradeBoundaryMarker | None


@dataclass(frozen=True)
class _FunctionAstContext:
    node: ast.FunctionDef | ast.AsyncFunctionDef
    module_context: _ModuleContext
    explicit_decision_protocol: bool
    boundary_marker: GradeBoundaryMarker | None
    callsite_contexts: dict[tuple[int, int, int, int], "_CallSiteAstContext"]
    ignored_call_spans: frozenset[tuple[int, int, int, int]]


@dataclass(frozen=True)
class _CallSiteAstContext:
    structural_path: str
    boundary_marker: GradeBoundaryMarker | None


@dataclass(frozen=True)
class _ClassificationFacts:
    runtime_classification_count: int
    type_domain_cardinality: int
    shape_domain_cardinality: int
    has_never_call: bool


@dataclass(frozen=True)
class _ComplexityFacts:
    output_cardinality_class: OutputCardinalityClass
    work_growth_class: WorkGrowthClass


@dataclass(frozen=True)
class SemanticGradeMonotonicityReport:
    witnesses: tuple[CallEdgeGradeWitness, ...]
    violations: tuple[GradeMonotonicityViolation, ...]
    callable_grades: tuple[tuple[str, DeterminismCostGrade], ...]

    def policy_data(self) -> dict[str, object]:
        counts: dict[str, int] = {}
        for violation in self.violations:
            counts[violation.rule_id] = counts.get(violation.rule_id, 0) + 1
        return {
            "witness_count": len(self.witnesses),
            "violation_count": len(self.violations),
            "counts": counts,
            "witness_rows": [item.as_payload() for item in self.witnesses],
            "violations": [item.as_payload() for item in self.violations],
            "callable_grades": [
                {"qualname": qualname, "grade": grade.as_payload()}
                for qualname, grade in self.callable_grades
            ],
        }


class _FunctionContextCollector(ast.NodeVisitor):
    def __init__(self, *, module_context: _ModuleContext, module_name: str) -> None:
        self.module_context = module_context
        self.module_name = module_name
        self.scope_stack: list[str] = []
        self.by_qual: dict[str, _FunctionAstContext] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qual_parts = [self.module_name] if self.module_name else []
        qual_parts.extend(self.scope_stack)
        qual_parts.append(node.name)
        qualname = ".".join(part for part in qual_parts if part)
        function_boundary = (
            _function_grade_boundary(self.module_context, qualname=qualname, node=node)
            or self.module_context.boundary_marker
        )
        callsite_collector = _CallSiteContextCollector(
            qualname=qualname,
            function_boundary=function_boundary,
        )
        callsite_collector.collect(node)
        self.by_qual[qualname] = _FunctionAstContext(
            node=node,
            module_context=self.module_context,
            explicit_decision_protocol=(
                self.module_context.decision_protocol_module
                or _has_marker(self.module_context.source_lines, node.lineno, _DECISION_PROTOCOL_COMMENT)
                or _has_decorator(node, "decision_protocol")
            ),
            boundary_marker=function_boundary,
            callsite_contexts=callsite_collector.by_span,
            ignored_call_spans=frozenset(callsite_collector.ignored_call_spans),
        )
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()


class _CallSiteContextCollector:
    def __init__(
        self,
        *,
        qualname: str,
        function_boundary: GradeBoundaryMarker | None,
    ) -> None:
        self.qualname = qualname
        self.function_boundary = function_boundary
        self.boundary_stack: list[GradeBoundaryMarker] = []
        self.by_span: dict[tuple[int, int, int, int], _CallSiteAstContext] = {}
        self.ignored_call_spans: set[tuple[int, int, int, int]] = set()

    def collect(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for index, stmt in enumerate(node.body):
            self._visit(stmt, path=(f"body[{index}]",))

    def _visit(self, node: ast.AST, *, path: tuple[str, ...]) -> None:
        match node:
            case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.ClassDef() | ast.Lambda():
                return
            case ast.With(items=items, body=body):
                self._visit_with(items=items, body=body, path=path)
                return
            case ast.AsyncWith(items=items, body=body):
                self._visit_with(items=items, body=body, path=path)
                return
            case ast.Call():
                self._visit_call(node, path=path)
        for field_name, value in ast.iter_fields(node):
            if isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        self._visit(
                            item,
                            path=path + (f"{field_name}[{index}]",),
                        )
            elif isinstance(value, ast.AST):
                self._visit(value, path=path + (field_name,))

    def _visit_with(
        self,
        *,
        items: list[ast.withitem],
        body: list[ast.stmt],
        path: tuple[str, ...],
    ) -> None:
        local_markers: list[GradeBoundaryMarker] = []
        for index, item in enumerate(items):
            marker = _grade_boundary_from_expr(
                item.context_expr,
                source="callsite_with",
                default_name=self.qualname,
            )
            if marker is not None:
                span = _node_span_optional(item.context_expr)
                if span is not None:
                    self.ignored_call_spans.add(span)
                local_markers.append(marker)
            else:
                self._visit(
                    item.context_expr,
                    path=path + (f"items[{index}].context_expr",),
                )
            if item.optional_vars is not None:
                self._visit(
                    item.optional_vars,
                    path=path + (f"items[{index}].optional_vars",),
                )
        previous_depth = len(self.boundary_stack)
        self.boundary_stack.extend(local_markers)
        try:
            for index, stmt in enumerate(body):
                self._visit(
                    stmt,
                    path=path + (f"body[{index}]",),
                )
        finally:
            del self.boundary_stack[previous_depth:]

    def _visit_call(self, node: ast.Call, *, path: tuple[str, ...]) -> None:
        if _grade_boundary_from_expr(node, source="callsite_with", default_name=self.qualname) is not None:
            span = _node_span_optional(node)
            if span is not None:
                self.ignored_call_spans.add(span)
            return
        span = _node_span_optional(node)
        if span is None:
            return
        self.by_span[span] = _CallSiteAstContext(
            structural_path=_structural_path_identity(path + ("call",)),
            boundary_marker=(
                self.boundary_stack[-1]
                if self.boundary_stack
                else self.function_boundary
            ),
        )


class _ClassificationVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.runtime_sites = 0
        self.type_domain_cardinality = 1
        self.shape_markers: set[str] = set()
        self.has_never_call = False

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name == "never":
            self.has_never_call = True
        if name == "isinstance":
            self.runtime_sites += 1
            alternative_count, shape_markers = _isinstance_type_alternatives(node)
            self.type_domain_cardinality = max(self.type_domain_cardinality, alternative_count)
            self.shape_markers.update(shape_markers)
        elif name == "hasattr":
            self.runtime_sites += 1
            self.shape_markers.add("attribute")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if _compare_contains_none(node):
            self.runtime_sites += 1
        elif _looks_like_shape_membership_probe(node):
            self.runtime_sites += 1
            self.shape_markers.add("mapping_key")
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self.runtime_sites += 1
        type_markers: set[str] = set()
        for case in node.cases:
            type_markers.update(_pattern_type_markers(case.pattern))
            self.shape_markers.update(_pattern_shape_markers(case.pattern))
        if type_markers:
            self.type_domain_cardinality = max(self.type_domain_cardinality, len(type_markers))
        self.generic_visit(node)


class _ComplexityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.loop_depth = 0
        self.max_loop_depth = 0
        self.has_sort = False
        self.has_reducer = False
        self.has_linear_work = False
        self.output_linear = False
        self.output_quadratic = False
        self.named_output_classes: dict[str, OutputCardinalityClass] = {}

    def visit_For(self, node: ast.For) -> None:
        self._visit_loop(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_loop(node)

    def visit_While(self, node: ast.While) -> None:
        self._visit_loop(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        output_class = self._output_class_for_expr(node.value)
        for target in node.targets:
            self._record_named_output_class(target, output_class)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._record_named_output_class(
                node.target,
                self._output_class_for_expr(node.value),
            )
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self._observe_output_expr(node.value)
        self.generic_visit(node)

    def visit_Yield(self, node: ast.Yield) -> None:
        self.output_linear = True
        self.generic_visit(node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self.output_linear = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name in _REDUCER_NAMES:
            self.has_reducer = True
            self.has_linear_work = True
        if name == "sorted" or name.endswith(".sort"):
            self.has_sort = True
            self.has_linear_work = True
        if name in _MATERIALIZER_NAMES and node.args:
            self.has_linear_work = True
        self.generic_visit(node)

    def _visit_loop(self, node: ast.For | ast.AsyncFor | ast.While) -> None:
        self.has_linear_work = True
        self.loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
        self.generic_visit(node)
        self.loop_depth -= 1

    def _visit_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
    ) -> None:
        self.has_linear_work = True
        comprehension_depth = self.loop_depth + len(node.generators)
        self.max_loop_depth = max(self.max_loop_depth, comprehension_depth)
        self.generic_visit(node)

    def _observe_output_expr(self, value: ast.expr | None) -> None:
        if value is None:
            return
        output_class = self._output_class_for_expr(value)
        if output_class >= OutputCardinalityClass.QUADRATIC:
            self.output_quadratic = True
            return
        if output_class >= OutputCardinalityClass.LINEAR:
            self.output_linear = True

    def _output_class_for_expr(self, value: ast.expr) -> OutputCardinalityClass:
        generators = _output_generator_depth(value)
        if generators >= 2:
            return OutputCardinalityClass.QUADRATIC
        if generators == 1:
            return OutputCardinalityClass.LINEAR
        match value:
            case ast.Name(id=name):
                return self.named_output_classes.get(
                    name,
                    OutputCardinalityClass.CONSTANT,
                )
            case ast.Call(func=func, args=args) if _call_name(func) in _MATERIALIZER_NAMES and args:
                nested_class = self._output_class_for_expr(args[0])
                if nested_class >= OutputCardinalityClass.QUADRATIC:
                    return OutputCardinalityClass.QUADRATIC
                return OutputCardinalityClass.LINEAR
            case _:
                return OutputCardinalityClass.CONSTANT

    def _record_named_output_class(
        self,
        target: ast.expr,
        output_class: OutputCardinalityClass,
    ) -> None:
        match target:
            case ast.Name(id=name):
                if output_class <= OutputCardinalityClass.CONSTANT:
                    self.named_output_classes.pop(name, None)
                    return
                self.named_output_classes[name] = output_class
            case ast.Tuple(elts=elts) | ast.List(elts=elts):
                for item in elts:
                    self._record_named_output_class(item, output_class)
            case _:
                return


def collect_grade_monotonicity(*, batch: PolicyScanBatch) -> SemanticGradeMonotonicityReport:
    if not batch.modules:
        return SemanticGradeMonotonicityReport(
            witnesses=(),
            violations=(),
            callable_grades=(),
        )
    analysis_index = _build_analysis_index(
        [module.path for module in batch.modules],
        project_root=batch.root,
        ignore_params=set(),
        strictness="high",
        external_filter=True,
        transparent_decorators=None,
        parse_failure_witnesses=[],
        decision_ignore_params=set(),
        decision_require_tiers=False,
    )
    module_contexts = _module_contexts(batch=batch)
    function_contexts = _function_contexts(batch=batch, module_contexts=module_contexts)
    decision_table_quals, decision_bundle_quals = _decision_flow_membership(
        infos=analysis_index.by_qual.values(),
        root=batch.root,
    )
    recursive_quals = _collect_recursive_functions(
        _collect_call_edges(
            by_name=analysis_index.by_name,
            by_qual=analysis_index.by_qual,
            symbol_table=analysis_index.symbol_table,
            project_root=batch.root,
            class_index=analysis_index.class_index,
        )
    )
    grade_by_qual: dict[str, DeterminismCostGrade] = {}
    for info in _sorted_infos(analysis_index.by_qual.values()):
        grade_by_qual[info.qual] = _infer_grade(
            info=info,
            function_context=function_contexts.get(info.qual),
            decision_table_quals=decision_table_quals,
            decision_bundle_quals=decision_bundle_quals,
            recursive_quals=recursive_quals,
        )
    witnesses: list[CallEdgeGradeWitness] = []
    violations: list[GradeMonotonicityViolation] = []
    for caller in _sorted_infos(analysis_index.by_qual.values()):
        caller_context = function_contexts.get(caller.qual)
        caller_grade = grade_by_qual.get(caller.qual, DeterminismCostGrade.unknown_top())
        caller_path = _rel_path_text(batch.root, caller.path)
        caller_line, caller_column = _span_location(caller.function_span)
        sorted_calls = sort_once(
            caller.calls,
            key=_call_sort_key,
            source="grade_monotonicity_semantic.collect_grade_monotonicity.calls",
        )
        for call in sorted_calls:
            if call.is_test:
                continue
            callsite_context = _callsite_context_optional(caller_context=caller_context, call=call)
            if callsite_context is None and caller_context is not None and call.span is not None:
                if tuple(int(item) for item in call.span) in caller_context.ignored_call_spans:
                    continue
            resolution = _resolve_callee_outcome(
                call.callee,
                caller,
                analysis_index.by_name,
                analysis_index.by_qual,
                symbol_table=analysis_index.symbol_table,
                project_root=batch.root,
                class_index=analysis_index.class_index,
                call=call,
            )
            if resolution.status == "resolved" and resolution.candidates:
                resolved_callee = resolution.candidates[0]
                callee_context = function_contexts.get(resolved_callee.qual)
                callee_grade = grade_by_qual.get(
                    resolved_callee.qual,
                    DeterminismCostGrade.unknown_top(),
                )
                callee_path = _rel_path_text(batch.root, resolved_callee.path)
                callee_qualname = resolved_callee.qual
                callee_line, callee_column = _span_location(resolved_callee.function_span)
            else:
                resolved_callee = None
                callee_context = None
                callee_grade = DeterminismCostGrade.unknown_top()
                callee_path = ""
                callee_qualname = f"<{resolution.status}:{resolution.callee_key}>"
                callee_line, callee_column = (0, 0)
            boundary_marker = _edge_boundary_marker(
                callsite_boundary_marker=(
                    None if callsite_context is None else callsite_context.boundary_marker
                ),
                caller_context=caller_context,
                callee_context=callee_context,
            )
            failure_rule_ids = _edge_failure_rule_ids(
                caller_grade=caller_grade,
                callee_grade=callee_grade,
                boundary_marker=boundary_marker,
            )
            call_line, call_column = _span_location(call.span)
            edge_structural_identity = _edge_structural_identity(
                caller_path=caller_path,
                caller_qualname=caller.qual,
                callee_qualname=callee_qualname,
                callsite_context=callsite_context,
                edge_resolution_status=resolution.status,
                edge_resolution_phase=resolution.phase,
            )
            edge_identity = canonical_site_identity(
                rel_path=caller_path,
                qualname=f"{caller.qual}->{callee_qualname}",
                line=call_line,
                column=call_column,
                node_kind="call_edge",
                surface="grade_monotonicity",
            )
            witness = CallEdgeGradeWitness(
                witness_id=edge_structural_identity,
                edge_site_identity=edge_identity,
                edge_structural_identity=edge_structural_identity,
                caller_path=caller_path,
                caller_qualname=caller.qual,
                caller_line=caller_line,
                caller_column=caller_column,
                callee_path=callee_path,
                callee_qualname=callee_qualname,
                callee_line=callee_line,
                callee_column=callee_column,
                call_line=call_line,
                call_column=call_column,
                edge_resolution_status=resolution.status,
                edge_resolution_phase=resolution.phase,
                edge_kind="boundary" if boundary_marker is not None else "ordinary",
                boundary_marker=boundary_marker,
                caller_grade=caller_grade,
                callee_grade=callee_grade,
                monotone=not failure_rule_ids,
                failure_rule_ids=failure_rule_ids,
            )
            witnesses.append(witness)
            for rule_id in failure_rule_ids:
                violations.append(
                    _build_violation(
                        rule_id=rule_id,
                        witness=witness,
                    )
                )
    return SemanticGradeMonotonicityReport(
        witnesses=tuple(_sorted_witnesses(witnesses)),
        violations=tuple(_sorted_violations(violations)),
        callable_grades=tuple(
            sort_once(
                grade_by_qual.items(),
                key=lambda item: item[0],
                source="grade_monotonicity_semantic.collect_grade_monotonicity.callable_grades",
            )
        ),
    )


def _module_contexts(*, batch: PolicyScanBatch) -> dict[Path, _ModuleContext]:
    contexts: dict[Path, _ModuleContext] = {}
    for module in batch.modules:
        source_lines = tuple(module.source.splitlines())
        contexts[module.path.resolve()] = _ModuleContext(
            path=module.path.resolve(),
            rel_path=module.rel_path,
            source_lines=source_lines,
            decision_protocol_module=_module_has_marker(source_lines, _DECISION_PROTOCOL_MODULE_COMMENT),
            boundary_marker=(
                _module_grade_boundary(source_lines)
                or _legacy_module_boundary_marker(
                    rel_path=module.rel_path,
                    source_lines=source_lines,
                )
            ),
        )
    return contexts


def _function_contexts(
    *,
    batch: PolicyScanBatch,
    module_contexts: dict[Path, _ModuleContext],
) -> dict[str, _FunctionAstContext]:
    contexts: dict[str, _FunctionAstContext] = {}
    for module in batch.modules:
        module_context = module_contexts[module.path.resolve()]
        collector = _FunctionContextCollector(
            module_context=module_context,
            module_name=_module_name(module.path, project_root=batch.root),
        )
        collector.visit(module.tree)
        contexts.update(collector.by_qual)
    return contexts


def _decision_flow_membership(
    *,
    infos: Iterable[FunctionInfo],
    root: Path,
) -> tuple[set[str], set[str]]:
    decision_surfaces: list[str] = []
    value_surfaces: list[str] = []
    for info in _sorted_infos(infos):
        rel_path = _rel_path_text(root, info.path)
        if info.decision_params:
            params = ", ".join(
                sort_once(
                    info.decision_params,
                    source="grade_monotonicity_semantic._decision_flow_membership.decision_params",
                )
            )
            decision_surfaces.append(f"{rel_path}:{info.qual} decision surface params: {params}")
        if info.value_decision_params:
            params = ", ".join(
                sort_once(
                    info.value_decision_params,
                    source="grade_monotonicity_semantic._decision_flow_membership.value_params",
                )
            )
            value_surfaces.append(
                f"{rel_path}:{info.qual} value-encoded decision params: {params}"
            )
    tables = build_decision_tables(
        decision_surfaces=decision_surfaces,
        value_decision_surfaces=value_surfaces,
    )
    table_quals = {str(item.get("qual", "")) for item in tables if item.get("qual")}
    decision_id_to_qual = {
        str(item.get("decision_id", "")): str(item.get("qual", ""))
        for item in tables
        if item.get("decision_id") and item.get("qual")
    }
    bundle_quals: set[str] = set()
    for bundle in detect_repeated_guard_bundles(tables):
        member_ids = bundle.get("member_decision_ids", [])
        if not isinstance(member_ids, list):
            continue
        for member_id in member_ids:
            qual = decision_id_to_qual.get(str(member_id))
            if qual:
                bundle_quals.add(qual)
    return table_quals, bundle_quals


def _infer_grade(
    *,
    info: FunctionInfo,
    function_context: _FunctionAstContext | None,
    decision_table_quals: set[str],
    decision_bundle_quals: set[str],
    recursive_quals: set[str],
) -> DeterminismCostGrade:
    classification = _classification_facts(function_context.node if function_context else None)
    complexity = _complexity_facts(
        node=function_context.node if function_context else None,
        recursive=info.qual in recursive_quals,
    )
    nullable_domain_cardinality = max(
        (
            _annotation_nullable_domain_cardinality_optional(text)
            for text in info.annots.values()
        ),
        default=1,
    )
    type_domain_cardinality = max(
        (
            _annotation_type_domain_cardinality_optional(text)
            for text in info.annots.values()
        ),
        default=1,
    )
    shape_domain_cardinality = max(
        (
            _annotation_shape_domain_cardinality_optional(text)
            for text in info.annots.values()
        ),
        default=1,
    )
    nullable_domain_cardinality = max(nullable_domain_cardinality, 1)
    type_domain_cardinality = max(type_domain_cardinality, classification.type_domain_cardinality, 1)
    shape_domain_cardinality = max(shape_domain_cardinality, classification.shape_domain_cardinality, 1)
    protocol_discharge_level = _protocol_discharge_level(
        info=info,
        function_context=function_context,
        decision_table_quals=decision_table_quals,
        decision_bundle_quals=decision_bundle_quals,
        classification=classification,
        nullable_domain_cardinality=nullable_domain_cardinality,
        type_domain_cardinality=type_domain_cardinality,
        shape_domain_cardinality=shape_domain_cardinality,
    )
    return DeterminismCostGrade(
        nullable_domain_cardinality=nullable_domain_cardinality,
        type_domain_cardinality=type_domain_cardinality,
        shape_domain_cardinality=shape_domain_cardinality,
        runtime_classification_count=classification.runtime_classification_count,
        protocol_discharge_level=protocol_discharge_level,
        output_cardinality_class=complexity.output_cardinality_class,
        work_growth_class=complexity.work_growth_class,
    )


def _protocol_discharge_level(
    *,
    info: FunctionInfo,
    function_context: _FunctionAstContext | None,
    decision_table_quals: set[str],
    decision_bundle_quals: set[str],
    classification: _ClassificationFacts,
    nullable_domain_cardinality: int,
    type_domain_cardinality: int,
    shape_domain_cardinality: int,
) -> ProtocolDischargeLevel:
    if (
        nullable_domain_cardinality <= 1
        and type_domain_cardinality <= 1
        and shape_domain_cardinality <= 1
        and classification.runtime_classification_count == 0
        and classification.has_never_call
    ):
        return ProtocolDischargeLevel.INVARIANT_DISCHARGED
    if function_context is not None and function_context.explicit_decision_protocol:
        return ProtocolDischargeLevel.DECISION_PROTOCOL
    if info.qual in decision_bundle_quals:
        return ProtocolDischargeLevel.DECISION_BUNDLE
    if info.qual in decision_table_quals or info.decision_params or info.value_decision_params:
        return ProtocolDischargeLevel.DECISION_TABLE
    return ProtocolDischargeLevel.RAW_INGRESS


def _classification_facts(
    node: ast.FunctionDef | ast.AsyncFunctionDef | None,
) -> _ClassificationFacts:
    if node is None:
        return _ClassificationFacts(
            runtime_classification_count=0,
            type_domain_cardinality=1,
            shape_domain_cardinality=1,
            has_never_call=False,
        )
    visitor = _ClassificationVisitor()
    visitor.visit(node)
    return _ClassificationFacts(
        runtime_classification_count=visitor.runtime_sites,
        type_domain_cardinality=visitor.type_domain_cardinality,
        shape_domain_cardinality=max(1, len(visitor.shape_markers)),
        has_never_call=visitor.has_never_call,
    )


def _complexity_facts(
    *,
    node: ast.FunctionDef | ast.AsyncFunctionDef | None,
    recursive: bool,
) -> _ComplexityFacts:
    if recursive:
        return _ComplexityFacts(
            output_cardinality_class=OutputCardinalityClass.UNKNOWN_TOP,
            work_growth_class=WorkGrowthClass.UNKNOWN_TOP,
        )
    if node is None:
        return _ComplexityFacts(
            output_cardinality_class=OutputCardinalityClass.CONSTANT,
            work_growth_class=WorkGrowthClass.CONSTANT,
        )
    visitor = _ComplexityVisitor()
    visitor.visit(node)
    output_cardinality_class = OutputCardinalityClass.CONSTANT
    if visitor.output_quadratic:
        output_cardinality_class = OutputCardinalityClass.QUADRATIC
    elif visitor.output_linear:
        output_cardinality_class = OutputCardinalityClass.LINEAR
    work_growth_class = WorkGrowthClass.CONSTANT
    if visitor.max_loop_depth >= 2:
        work_growth_class = WorkGrowthClass.QUADRATIC
    elif visitor.has_sort:
        work_growth_class = WorkGrowthClass.N_LOG_N
    elif visitor.has_linear_work or visitor.max_loop_depth == 1:
        work_growth_class = WorkGrowthClass.LINEAR
    return _ComplexityFacts(
        output_cardinality_class=output_cardinality_class,
        work_growth_class=work_growth_class,
    )


def _annotation_nullable_domain_cardinality_optional(annotation: str | None) -> int:
    if annotation is None:
        return 1
    lowered = annotation.lower()
    if "optional[" in lowered or "| none" in lowered or "none |" in lowered:
        return 2
    if "union[" in lowered and "none" in lowered:
        return 2
    return 1


def _annotation_type_domain_cardinality_optional(annotation: str | None) -> int:
    if annotation is None:
        return 1
    lowered = annotation.lower()
    if re.search(r"\b(any|object)\b", lowered):
        return _DOMAIN_TOP
    if "optional[" in lowered:
        inner = lowered.partition("optional[")[2].rpartition("]")[0]
        return max(2, _annotation_type_domain_cardinality_optional(inner or None))
    if "|" in lowered:
        return max(1, len([part for part in lowered.split("|") if part.strip()]))
    if "union[" in lowered:
        payload = lowered.partition("union[")[2].rpartition("]")[0]
        return max(1, len([part for part in payload.split(",") if part.strip()]))
    return 1


def _annotation_shape_domain_cardinality_optional(annotation: str | None) -> int:
    if annotation is None:
        return 1
    lowered = annotation.lower()
    if re.search(r"\b(any|object)\b", lowered):
        return _DOMAIN_TOP
    if "|" in lowered:
        markers = {_shape_marker_for_text(part) for part in lowered.split("|")}
        markers.discard("")
        return max(1, len(markers))
    if "union[" in lowered:
        payload = lowered.partition("union[")[2].rpartition("]")[0]
        markers = {_shape_marker_for_text(part) for part in payload.split(",")}
        markers.discard("")
        return max(1, len(markers))
    marker = _shape_marker_for_text(lowered)
    return 1 if not marker else 1


def _shape_marker_for_text(annotation: str) -> str:
    lowered = annotation.strip().lower()
    for marker in _SHAPE_TYPE_NAMES:
        if marker in lowered:
            return marker
    return ""


def _edge_failure_rule_ids(
    *,
    caller_grade: DeterminismCostGrade,
    callee_grade: DeterminismCostGrade,
    boundary_marker: GradeBoundaryMarker | None,
) -> tuple[str, ...]:
    if (
        boundary_marker is not None
        and boundary_marker.kind is GradeBoundaryKind.SEMANTIC_CARRIER_ADAPTER
    ):
        return ()
    failures: list[str] = []
    if callee_grade.nullable_domain_cardinality > caller_grade.nullable_domain_cardinality:
        failures.append("GMP-001")
    if callee_grade.type_domain_cardinality > caller_grade.type_domain_cardinality:
        failures.append("GMP-002")
    if callee_grade.shape_domain_cardinality > caller_grade.shape_domain_cardinality:
        failures.append("GMP-003")
    if callee_grade.runtime_classification_count > caller_grade.runtime_classification_count:
        failures.append("GMP-004")
    if callee_grade.protocol_discharge_level < caller_grade.protocol_discharge_level:
        failures.append("GMP-005")
    if _cost_expansion_requires_failure(
        caller_class=caller_grade.output_cardinality_class,
        callee_class=callee_grade.output_cardinality_class,
        boundary_marker=boundary_marker,
    ):
        failures.append("GMP-006")
    if _cost_expansion_requires_failure(
        caller_class=caller_grade.work_growth_class,
        callee_class=callee_grade.work_growth_class,
        boundary_marker=boundary_marker,
    ):
        failures.append("GMP-007")
    return tuple(failures)


def _callsite_context_optional(
    *,
    caller_context: _FunctionAstContext | None,
    call: CallArgs,
) -> _CallSiteAstContext | None:
    if caller_context is None or call.span is None:
        return None
    span = tuple(int(item) for item in call.span)
    return caller_context.callsite_contexts.get(span)


def _edge_structural_identity(
    *,
    caller_path: str,
    caller_qualname: str,
    callee_qualname: str,
    callsite_context: _CallSiteAstContext | None,
    edge_resolution_status: str,
    edge_resolution_phase: str,
) -> str:
    return canonical_structural_identity(
        rel_path=caller_path,
        qualname=f"{caller_qualname}->{callee_qualname}",
        structural_path=(
            callsite_context.structural_path
            if callsite_context is not None
            else _fallback_structural_path(
                callee_qualname=callee_qualname,
                edge_resolution_status=edge_resolution_status,
                edge_resolution_phase=edge_resolution_phase,
            )
        ),
        node_kind="call_edge",
        surface="grade_monotonicity",
    )


def _fallback_structural_path(
    *,
    callee_qualname: str,
    edge_resolution_status: str,
    edge_resolution_phase: str,
) -> str:
    return _structural_path_identity(
        (
            "call_edge",
            callee_qualname,
            edge_resolution_status,
            edge_resolution_phase,
        )
    )


def _cost_expansion_requires_failure(
    *,
    caller_class: OutputCardinalityClass | WorkGrowthClass,
    callee_class: OutputCardinalityClass | WorkGrowthClass,
    boundary_marker: GradeBoundaryMarker | None,
) -> bool:
    if int(callee_class) <= int(caller_class):
        return False
    if callee_class.name == "UNKNOWN_TOP":
        return True
    return boundary_marker is None


def _edge_boundary_marker(
    *,
    callsite_boundary_marker: GradeBoundaryMarker | None,
    caller_context: _FunctionAstContext | None,
    callee_context: _FunctionAstContext | None,
) -> GradeBoundaryMarker | None:
    if callsite_boundary_marker is not None:
        return callsite_boundary_marker
    if callee_context is not None and callee_context.boundary_marker is not None:
        return callee_context.boundary_marker
    if caller_context is not None and caller_context.boundary_marker is not None:
        return caller_context.boundary_marker
    return None


def _build_violation(
    *,
    rule_id: str,
    witness: CallEdgeGradeWitness,
) -> GradeMonotonicityViolation:
    message = (
        f"{_VIOLATION_MESSAGES[rule_id]} "
        f"({witness.caller_qualname} -> {witness.callee_qualname})"
    )
    violation_id = canonical_structural_identity(
        rel_path=witness.caller_path,
        qualname=f"{witness.caller_qualname}:{rule_id}:{witness.callee_qualname}",
        structural_path=witness.edge_structural_identity,
        node_kind="grade_violation",
        surface="grade_monotonicity",
    )
    details = {
        "guidance": _violation_guidance(rule_id),
        "edge_resolution_status": witness.edge_resolution_status,
        "edge_resolution_phase": witness.edge_resolution_phase,
        "edge_structural_identity": witness.edge_structural_identity,
        "boundary_marker": (
            None if witness.boundary_marker is None else witness.boundary_marker.as_payload()
        ),
        "caller_grade": witness.caller_grade.as_payload(),
        "callee_grade": witness.callee_grade.as_payload(),
    }
    return GradeMonotonicityViolation(
        violation_id=violation_id,
        witness_id=witness.witness_id,
        rule_id=rule_id,
        path=witness.caller_path,
        line=witness.call_line,
        column=witness.call_column,
        qualname=witness.caller_qualname,
        callee_qualname=witness.callee_qualname,
        message=message,
        details=details,
    )


def _module_has_marker(source_lines: tuple[str, ...], marker: str) -> bool:
    for raw in source_lines[:80]:
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#") and marker in stripped:
            return True
        if stripped.startswith("\"\"\"") or stripped.startswith("'''"):
            continue
    return False


def _module_grade_boundary(source_lines: tuple[str, ...]) -> GradeBoundaryMarker | None:
    for index, raw in enumerate(source_lines[:80], start=1):
        marker = _parse_grade_boundary_marker(
            raw,
            source="module_comment",
            line=index,
        )
        if marker is not None:
            return marker
    return None


def _legacy_module_boundary_marker(
    *,
    rel_path: str,
    source_lines: tuple[str, ...],
) -> GradeBoundaryMarker | None:
    if not _module_has_marker(source_lines, _BOUNDARY_NORMALIZATION_MODULE_COMMENT):
        return None
    return GradeBoundaryMarker(
        kind=GradeBoundaryKind.INGRESS_NORMALIZATION,
        name=rel_path,
        source="legacy_boundary_normalization_module",
        line=1,
    )


def _function_grade_boundary(
    module_context: _ModuleContext,
    *,
    qualname: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> GradeBoundaryMarker | None:
    line = int(getattr(node, "lineno", 1))
    marker = _nearest_comment_boundary_marker(
        source_lines=module_context.source_lines,
        line=line,
    )
    if marker is not None:
        return marker
    decorator_marker = _decorator_grade_boundary(node, default_name=qualname)
    if decorator_marker is not None:
        return decorator_marker
    if _has_marker(module_context.source_lines, line, _BOUNDARY_NORMALIZATION_COMMENT) or _has_decorator(
        node,
        "boundary_normalization",
    ):
        return GradeBoundaryMarker(
            kind=GradeBoundaryKind.INGRESS_NORMALIZATION,
            name=qualname,
            source="legacy_boundary_normalization",
            line=line,
        )
    return None


def _decorator_grade_boundary(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    default_name: str,
) -> GradeBoundaryMarker | None:
    for decorator in node.decorator_list:
        marker = _grade_boundary_from_expr(
            decorator,
            source="function_decorator",
            default_name=default_name,
        )
        if marker is not None:
            return marker
    return None


def _nearest_comment_boundary_marker(
    *,
    source_lines: tuple[str, ...],
    line: int,
) -> GradeBoundaryMarker | None:
    idx = max(0, line - 2)
    while idx >= 0:
        stripped = source_lines[idx].strip()
        if not stripped:
            idx -= 1
            continue
        if not stripped.startswith("#"):
            return None
        return _parse_grade_boundary_marker(
            stripped,
            source="function_comment",
            line=idx + 1,
        )
    return None


def _parse_grade_boundary_marker(
    raw: str,
    *,
    source: str,
    line: int,
) -> GradeBoundaryMarker | None:
    match = _GRADE_BOUNDARY_RE.search(raw)
    if match is None:
        return None
    fields: dict[str, str] = {}
    for token in match.group("fields").strip().split():
        key, separator, value = token.partition("=")
        if not separator or not key or not value:
            return None
        fields[key.strip()] = value.strip()
    kind_text = fields.get("kind", "")
    name_text = fields.get("name", "")
    if kind_text not in _ALLOWED_GRADE_BOUNDARY_KINDS or not name_text:
        return None
    return GradeBoundaryMarker(
        kind=GradeBoundaryKind(kind_text),
        name=name_text,
        source=source,
        line=line,
    )


def _grade_boundary_from_expr(
    node: ast.AST,
    *,
    source: str,
    default_name: str,
) -> GradeBoundaryMarker | None:
    match node:
        case ast.Call(func=func, keywords=keywords) if _call_name(func) == "grade_boundary":
            fields = _parse_grade_boundary_fields(keywords=keywords, default_name=default_name)
            if fields is None:
                return None
            kind_text, name_text = fields
            line = int(getattr(node, "lineno", 0))
            return GradeBoundaryMarker(
                kind=GradeBoundaryKind(kind_text),
                name=name_text,
                source=source,
                line=line,
            )
        case _:
            return None


def _parse_grade_boundary_fields(
    *,
    keywords: list[ast.keyword],
    default_name: str,
) -> tuple[str, str] | None:
    values: dict[str, str] = {}
    for keyword in keywords:
        if keyword.arg is None:
            return None
        value = _keyword_string_optional(keyword.value)
        if value is None:
            return None
        values[keyword.arg] = value
    kind_text = values.get("kind", "")
    name_text = values.get("name", default_name)
    if kind_text not in _ALLOWED_GRADE_BOUNDARY_KINDS or not name_text:
        return None
    return (kind_text, name_text)


def _keyword_string_optional(node: ast.AST) -> str | None:
    match node:
        case ast.Constant(value=value) if isinstance(value, str):
            return value.strip()
        case _:
            return None


def _has_marker(source_lines: tuple[str, ...], line: int, marker: str) -> bool:
    idx = max(0, line - 2)
    while idx >= 0:
        stripped = source_lines[idx].strip()
        if not stripped:
            idx -= 1
            continue
        return stripped.startswith("#") and marker in stripped
    return False


def _has_decorator(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    decorator_name: str,
) -> bool:
    return any(_decorator_name(item) == decorator_name for item in node.decorator_list)


def _decorator_name(node: ast.AST) -> str:
    match node:
        case ast.Name(id=name):
            return name
        case ast.Attribute(attr=attr):
            return attr
        case ast.Call(func=func):
            return _decorator_name(func)
        case _:
            return ""


def _call_name(node: ast.AST) -> str:
    match node:
        case ast.Name(id=name):
            return name
        case ast.Attribute(value=_, attr=attr):
            return attr
        case _:
            return ""


def _node_span_optional(node: ast.AST) -> tuple[int, int, int, int] | None:
    lineno = getattr(node, "lineno", None)
    col_offset = getattr(node, "col_offset", None)
    end_lineno = getattr(node, "end_lineno", None)
    end_col_offset = getattr(node, "end_col_offset", None)
    if None in (lineno, col_offset, end_lineno, end_col_offset):
        return None
    return (
        int(lineno) - 1,
        int(col_offset),
        int(end_lineno) - 1,
        int(end_col_offset),
    )


def _structural_path_identity(path: tuple[str, ...]) -> str:
    return "::".join(path)


def _compare_contains_none(node: ast.Compare) -> bool:
    values = [node.left, *node.comparators]
    return any(_ast_expr_is_none(value) for value in values)


def _looks_like_shape_membership_probe(node: ast.Compare) -> bool:
    if not node.ops:
        return False
    if not any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
        return False
    return isinstance(node.left, ast.Constant)


def _ast_expr_is_none(node: ast.AST | None) -> bool:
    match node:
        case None:
            return True
        case ast.Constant(value=None):
            return True
        case ast.Name(id="None"):
            return True
        case _:
            return False


def _isinstance_type_alternatives(node: ast.Call) -> tuple[int, set[str]]:
    if len(node.args) < 2:
        return 1, set()
    candidate = node.args[1]
    alternatives: list[str] = []
    shape_markers: set[str] = set()
    match candidate:
        case ast.Tuple(elts=elts):
            for item in elts:
                name = _type_name(item)
                if name:
                    alternatives.append(name)
        case _:
            name = _type_name(candidate)
            if name:
                alternatives.append(name)
    for item in alternatives:
        if item.lower() in _SHAPE_TYPE_NAMES:
            shape_markers.add(item.lower())
    return max(1, len(set(alternatives))), shape_markers


def _type_name(node: ast.AST) -> str:
    match node:
        case ast.Name(id=name):
            return name
        case ast.Attribute(attr=attr):
            return attr
        case _:
            return ""


def _pattern_type_markers(pattern: ast.pattern) -> set[str]:
    match pattern:
        case ast.MatchClass(cls=cls):
            name = _type_name(cls)
            return {name} if name else set()
        case ast.MatchOr(patterns=patterns):
            markers: set[str] = set()
            for item in patterns:
                markers.update(_pattern_type_markers(item))
            return markers
        case _:
            return set()


def _pattern_shape_markers(pattern: ast.pattern) -> set[str]:
    match pattern:
        case ast.MatchMapping():
            return {"mapping"}
        case ast.MatchSequence():
            return {"sequence"}
        case ast.MatchClass(cls=cls):
            name = _type_name(cls).lower()
            if name in _SHAPE_TYPE_NAMES:
                return {name}
            return set()
        case ast.MatchOr(patterns=patterns):
            markers: set[str] = set()
            for item in patterns:
                markers.update(_pattern_shape_markers(item))
            return markers
        case _:
            return set()


def _output_generator_depth(value: ast.expr) -> int:
    match value:
        case ast.ListComp(generators=generators):
            return len(generators)
        case ast.SetComp(generators=generators):
            return len(generators)
        case ast.DictComp(generators=generators):
            return len(generators)
        case ast.GeneratorExp(generators=generators):
            return len(generators)
        case ast.Call(args=args) if args:
            return _output_generator_depth(args[0])
        case _:
            return 0


def _sorted_infos(infos: Iterable[FunctionInfo]) -> list[FunctionInfo]:
    return sort_once(
        infos,
        key=lambda info: (_rel_path_text(Path.cwd(), info.path), info.qual),
        source="grade_monotonicity_semantic._sorted_infos",
    )


def _sorted_witnesses(
    witnesses: Iterable[CallEdgeGradeWitness],
) -> list[CallEdgeGradeWitness]:
    return sort_once(
        witnesses,
        key=lambda item: (
            item.caller_path,
            item.caller_qualname,
            item.call_line,
            item.call_column,
            item.callee_qualname,
            item.edge_resolution_status,
        ),
        source="grade_monotonicity_semantic._sorted_witnesses",
    )


def _sorted_violations(
    violations: Iterable[GradeMonotonicityViolation],
) -> list[GradeMonotonicityViolation]:
    return sort_once(
        violations,
        key=lambda item: (
            item.rule_id,
            item.path,
            item.qualname,
            item.line,
            item.column,
            item.callee_qualname,
        ),
        source="grade_monotonicity_semantic._sorted_violations",
    )


def _call_sort_key(call: CallArgs) -> tuple[int, int, str]:
    line, column = _span_location(call.span)
    return (line, column, call.callee)


def _span_location(span: tuple[int, int, int, int] | None) -> tuple[int, int]:
    if span is None:
        return (1, 1)
    return (int(span[0]), int(span[1]) + 1)


def _rel_path_text(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


__all__ = [
    "SemanticGradeMonotonicityReport",
    "collect_grade_monotonicity",
]
