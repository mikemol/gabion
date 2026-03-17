#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from gabion.analysis.foundation.event_algebra import CanonicalRunContext
from gabion.runtime_shape_dispatch import (
    json_list_optional as _json_list_optional,
    json_mapping_optional as _json_mapping_optional,
)
from gabion.tooling.policy_substrate.aspf_union_view import build_aspf_union_view
from gabion.tooling.policy_substrate.rule_runtime import (
    cst_failure_seeds,
    decorate_failure,
    decorate_site,
    new_run_context,
)
from gabion.tooling.policy_rules.fiber_diagnostics import (
    FiberApplicabilityBounds,
    FiberCounterfactualBoundary,
    FiberTraceEvent,
)
from gabion.tooling.runtime.policy_scan_batch import (
    PolicyScanBatch,
    ScanFailureSeed,
    iter_failure_seeds,
)

RULE_NAME = "typing_surface"
BASELINE_VERSION = 1
WAIVER_VERSION = 1


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    column: int
    qualname: str
    kind: str
    message: str
    scope: str
    annotation: str
    input_slot: str
    flow_identity: str
    fiber_trace: tuple[FiberTraceEvent, ...]
    applicability_bounds: FiberApplicabilityBounds
    counterfactual_boundary: FiberCounterfactualBoundary
    fiber_id: str
    taint_interval_id: str
    condition_overlap_id: str
    structured_hash: str

    @property
    def key(self) -> str:
        return f"{self.path}:{self.qualname}:{self.kind}:{self.structured_hash}"

    @property
    def legacy_key(self) -> str:
        return f"{self.path}:{self.qualname}:{self.line}:{self.kind}"

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: [{self.qualname}] {self.message}"


@dataclass(frozen=True)
class InvalidWaiver:
    index: int
    reason: str


@dataclass(frozen=True)
class WaiverLoadResult:
    allowed_keys: set[str]
    invalid_waivers: list[InvalidWaiver]


class _TypingSurfaceVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        rel_path: str,
        source: str,
        run_context: CanonicalRunContext,
    ) -> None:
        self._rel_path = rel_path
        self._source = source
        self._scope = _forbidden_scope(rel_path)
        self._run_context = run_context
        self.violations: list[Violation] = []
        self._qualname_stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_annotation(node.annotation, node=node)
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._qualname_stack.append(node.name)
        for annotation, owner in _iter_function_annotations(node):
            self._check_annotation(annotation, node=owner)
        self.generic_visit(node)
        self._qualname_stack.pop()

    def _check_annotation(self, annotation: ast.AST, *, node: ast.AST) -> None:
        match self._scope:
            case None:
                return
            case str() as scope:
                pass
        kind = _annotation_kind(annotation)
        match kind:
            case None:
                return
            case str() as kind_text:
                pass
        line = int(getattr(node, "lineno", 1) or 1)
        column = int(getattr(node, "col_offset", 0) or 0) + 1
        qualname = ".".join(self._qualname_stack) if self._qualname_stack else "<module>"
        annotation_text = ast.get_source_segment(self._source, annotation) or "<unknown>"
        input_slot = f"annotation:{kind_text}"
        structural_identity = _structured_hash(
            self._rel_path,
            qualname,
            kind_text,
            scope,
            annotation_text,
            str(column),
        )
        decoration = decorate_site(
            run_context=self._run_context,
            rule_name=RULE_NAME,
            rel_path=self._rel_path,
            qualname=qualname,
            line=line,
            column=column,
            node_kind=f"typing:{kind_text}",
            input_slot=input_slot,
            taint_class="typing_surface",
            intro_kind=f"syntax:typing_taint:{kind_text}",
            condition_kind=f"syntax:typing_condition:{kind_text}",
            erase_kind=f"syntax:typed_boundary:{kind_text}",
            rationale=(
                "Replace raw typing surfaces with explicit DTO/Protocol boundary "
                "types so typed fibers stay explicit-by-construction."
            ),
        )
        self.violations.append(
            Violation(
                path=self._rel_path,
                line=line,
                column=column,
                qualname=qualname,
                kind=kind_text,
                scope=scope,
                annotation=annotation_text,
                input_slot=input_slot,
                message=(
                    f"{annotation_text!r} annotation is forbidden in {scope}; "
                    "use DTO/Protocol/dataclass carrier or TypedDict/Pydantic model"
                ),
                flow_identity=decoration.flow_identity,
                fiber_trace=decoration.fiber_trace,
                applicability_bounds=decoration.applicability_bounds,
                counterfactual_boundary=decoration.counterfactual_boundary,
                fiber_id=decoration.fiber_id,
                taint_interval_id=decoration.taint_interval_id,
                condition_overlap_id=decoration.condition_overlap_id,
                structured_hash=structural_identity,
            )
        )


def _iter_function_annotations(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[tuple[ast.AST, ast.AST], ...]:
    annotation_pairs: list[tuple[ast.AST, ast.AST]] = []
    if node.returns is not None:
        annotation_pairs.append((node.returns, node))
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        if arg.annotation is not None:
            annotation_pairs.append((arg.annotation, arg))
    match node.args.vararg:
        case ast.arg(annotation=ast.AST() as annotation):
            annotation_pairs.append((annotation, node))
        case _:
            pass
    match node.args.kwarg:
        case ast.arg(annotation=ast.AST() as annotation):
            annotation_pairs.append((annotation, node))
        case _:
            pass
    return tuple(annotation_pairs)


def _forbidden_scope(rel_path: str) -> str | None:
    if rel_path.startswith("src/gabion/analysis/"):
        return "semantic_core"
    if "/stages/" in rel_path or "stage_contract" in rel_path or rel_path.endswith("_stage.py"):
        return "stage_contracts"
    if "/reducers/" in rel_path or rel_path.endswith("_reducer.py"):
        return "reducers"
    return None


def _annotation_kind(annotation: ast.AST) -> str | None:
    if _is_any(annotation):
        return "any_annotation"
    if _is_bare_object(annotation):
        return "bare_object_annotation"
    if _is_dict_str_object(annotation):
        return "dict_str_object_annotation"
    return None


def _is_any(node: ast.AST) -> bool:
    match node:
        case ast.Name(id="Any"):
            return True
        case ast.Attribute(attr="Any"):
            return True
        case _:
            return False


def _is_bare_object(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "object"


def _is_dict_str_object(node: ast.AST) -> bool:
    match node:
        case ast.Subscript(value=value, slice=slice_node):
            base_name = _dotted_name(value)
            if base_name not in {"dict", "typing.Dict", "Dict"}:
                return False
            match slice_node:
                case ast.Tuple(elts=[key_node, value_node]):
                    return _is_str_node(key_node) and _is_bare_object(value_node)
                case _:
                    return False
        case _:
            return False


def _is_str_node(node: ast.AST) -> bool:
    match node:
        case ast.Name(id="str"):
            return True
        case ast.Constant(value="str"):
            return True
        case _:
            return False


def _dotted_name(node: ast.AST) -> str:
    return ".".join(_dotted_name_parts(node))


def _dotted_name_parts(node: ast.AST) -> tuple[str, ...]:
    match node:
        case ast.Name(id=identifier):
            return (identifier,)
        case ast.Attribute(value=value, attr=attr):
            return (*_dotted_name_parts(value), attr)
        case _:
            return ()


def _structured_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def collect_violations(
    *,
    batch: PolicyScanBatch,
    run_context: CanonicalRunContext | None = None,
) -> list[Violation]:
    context = run_context if run_context is not None else new_run_context(rule_name=RULE_NAME)
    union_view = build_aspf_union_view(batch=batch)
    violations: list[Violation] = []
    for seed in (*iter_failure_seeds(batch=batch), *cst_failure_seeds(union_view=union_view)):
        violations.append(_failure_violation(run_context=context, seed=seed))
    for module in union_view.modules:
        visitor = _TypingSurfaceVisitor(
            rel_path=module.rel_path,
            source=module.source,
            run_context=context,
        )
        visitor.visit(module.pyast_tree)
        violations.extend(visitor.violations)
    return violations


def _failure_violation(*, run_context: CanonicalRunContext, seed: ScanFailureSeed) -> Violation:
    decoration = decorate_failure(
        run_context=run_context,
        rule_name=RULE_NAME,
        seed=seed,
        rationale="Ensure module parse/read validity before typing-surface substrate evaluation.",
    )
    return Violation(
        path=seed.path,
        line=seed.line,
        column=seed.column,
        qualname="<module>",
        kind=seed.kind,
        message=seed.detail,
        scope="module_failure",
        annotation="<none>",
        input_slot="module_failure",
        flow_identity=decoration.flow_identity,
        fiber_trace=decoration.fiber_trace,
        applicability_bounds=decoration.applicability_bounds,
        counterfactual_boundary=decoration.counterfactual_boundary,
        fiber_id=decoration.fiber_id,
        taint_interval_id=decoration.taint_interval_id,
        condition_overlap_id=decoration.condition_overlap_id,
        structured_hash=_structured_hash(
            seed.path,
            "<module>",
            seed.kind,
            "module_failure",
            "<none>",
            str(seed.column),
        ),
    )


def _load_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload_mapping = _json_mapping_optional(payload)
    match payload_mapping:
        case None:
            return set()
        case _:
            pass
    raw_items = _json_list_optional(payload_mapping.get("violations"))
    match raw_items:
        case None:
            return set()
        case _:
            pass
    keys: set[str] = set()
    for raw_item in raw_items:
        item = _json_mapping_optional(raw_item)
        match item:
            case None:
                continue
            case _:
                pass
        path_value = str(item.get("path", "") or "")
        qualname = str(item.get("qualname", "") or "")
        kind = str(item.get("kind", "") or "")
        structured_hash = item.get("structured_hash")
        scope = str(item.get("scope", "") or "")
        annotation = str(item.get("annotation", "") or "")
        column = item.get("column")
        line = item.get("line")
        if not path_value or not qualname or not kind:
            continue
        match structured_hash:
            case str() as structured_hash_value if structured_hash_value:
                keys.add(f"{path_value}:{qualname}:{kind}:{structured_hash_value}")
                continue
            case _:
                pass
        match column:
            case int() as column_value if scope and annotation:
                migrated_hash = _structured_hash(
                    path_value,
                    qualname,
                    kind,
                    scope,
                    annotation,
                    str(column_value),
                )
                keys.add(f"{path_value}:{qualname}:{kind}:{migrated_hash}")
            case _:
                pass
        match line:
            case int() as line_value:
                keys.add(f"{path_value}:{qualname}:{line_value}:{kind}")
            case _:
                pass
    return keys


def load_waivers(path: Path) -> WaiverLoadResult:
    if not path.exists():
        return WaiverLoadResult(allowed_keys=set(), invalid_waivers=[])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload_mapping = _json_mapping_optional(payload)
    match payload_mapping:
        case None:
            return WaiverLoadResult(allowed_keys=set(), invalid_waivers=[InvalidWaiver(index=0, reason="waiver_file_not_object")])
        case _:
            pass
    raw_waivers = _json_list_optional(payload_mapping.get("waivers"))
    match raw_waivers:
        case None:
            return WaiverLoadResult(allowed_keys=set(), invalid_waivers=[InvalidWaiver(index=0, reason="waivers_not_list")])
        case _:
            pass

    keys: set[str] = set()
    invalid: list[InvalidWaiver] = []
    required = ("path", "qualname", "kind", "rationale", "scope", "expiry", "owner")
    for index, raw in enumerate(raw_waivers, start=1):
        waiver = _json_mapping_optional(raw)
        match waiver:
            case None:
                invalid.append(InvalidWaiver(index=index, reason="waiver_not_object"))
                continue
            case _:
                pass
        missing = [field for field in required if waiver.get(field) in (None, "")]
        if missing:
            invalid.append(InvalidWaiver(index=index, reason=f"missing_fields:{','.join(missing)}"))
            continue
        structured_hash = waiver.get("structured_hash")
        line = waiver.get("line")
        match structured_hash:
            case str() as structured_hash_value if structured_hash_value:
                key = f"{waiver['path']}:{waiver['qualname']}:{waiver['kind']}:{structured_hash_value}"
                keys.add(key)
                continue
            case _:
                pass
        scope = str(waiver.get("scope", "") or "")
        annotation = str(waiver.get("annotation", "") or "")
        column = waiver.get("column")
        match column:
            case int() as column_value if scope and annotation:
                migrated_hash = _structured_hash(
                    str(waiver["path"]),
                    str(waiver["qualname"]),
                    str(waiver["kind"]),
                    scope,
                    annotation,
                    str(column_value),
                )
                keys.add(f"{waiver['path']}:{waiver['qualname']}:{waiver['kind']}:{migrated_hash}")
                continue
            case _:
                pass
        match line:
            case int() as line_value:
                key = f"{waiver['path']}:{waiver['qualname']}:{line_value}:{waiver['kind']}"
                keys.add(key)
            case _:
                invalid.append(InvalidWaiver(index=index, reason="line_or_structured_hash_required"))
                continue
    return WaiverLoadResult(allowed_keys=keys, invalid_waivers=invalid)
