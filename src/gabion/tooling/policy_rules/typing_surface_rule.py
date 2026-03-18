#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterable
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
        scope = self._scope
        if scope is None:
            return
        kind = _annotation_kind(annotation)
        if kind is None:
            return
        kind_text = kind
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
    if (
        isinstance(node.args.vararg, ast.arg)
        and isinstance(node.args.vararg.annotation, ast.AST)
    ):
        annotation_pairs.append((node.args.vararg.annotation, node))
    if (
        isinstance(node.args.kwarg, ast.arg)
        and isinstance(node.args.kwarg.annotation, ast.AST)
    ):
        annotation_pairs.append((node.args.kwarg.annotation, node))
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
    return (
        isinstance(node, ast.Name)
        and node.id == "Any"
        or isinstance(node, ast.Attribute)
        and node.attr == "Any"
    )


def _is_bare_object(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "object"


def _is_dict_str_object(node: ast.AST) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    base_name = _dotted_name(node.value)
    if base_name not in {"dict", "typing.Dict", "Dict"}:
        return False
    if (
        not isinstance(node.slice, ast.Tuple)
        or len(node.slice.elts) != 2
    ):
        return False
    key_node, value_node = node.slice.elts
    return _is_str_node(key_node) and _is_bare_object(value_node)


def _is_str_node(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "str"
    if isinstance(node, ast.Constant):
        return node.value == "str"
    return False


def _dotted_name(node: ast.AST) -> str:
    return ".".join(_dotted_name_parts(node))


def _dotted_name_parts(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        return (*_dotted_name_parts(node.value), node.attr)
    return ()


def _baseline_item_mappings(raw_items: list[object]) -> Iterable[dict[str, object]]:
    return (
        item
        for raw_item in raw_items
        for item in (_json_mapping_optional(raw_item),)
        if item is not None
    )


def _waiver_mappings(
    raw_waivers: list[object],
) -> Iterable[tuple[int, dict[str, object]]]:
    return (
        (index, waiver)
        for index, raw in enumerate(raw_waivers, start=1)
        for waiver in (_json_mapping_optional(raw),)
        if waiver is not None
    )


def _load_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload_mapping = _json_mapping_optional(payload)
    if payload_mapping is None:
        return set()
    raw_items = _json_list_optional(payload_mapping.get("violations"))
    if raw_items is None:
        return set()
    keys: set[str] = set()
    for item in _baseline_item_mappings(raw_items):
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
        if isinstance(structured_hash, str) and structured_hash:
            keys.add(f"{path_value}:{qualname}:{kind}:{structured_hash}")
            continue
        if isinstance(column, int) and scope and annotation:
            migrated_hash = _structured_hash(
                path_value,
                qualname,
                kind,
                scope,
                annotation,
                str(column),
            )
            keys.add(f"{path_value}:{qualname}:{kind}:{migrated_hash}")
        if isinstance(line, int):
            keys.add(f"{path_value}:{qualname}:{line}:{kind}")
    return keys


def load_waivers(path: Path) -> WaiverLoadResult:
    if not path.exists():
        return WaiverLoadResult(allowed_keys=set(), invalid_waivers=[])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload_mapping = _json_mapping_optional(payload)
    if payload_mapping is None:
        return WaiverLoadResult(allowed_keys=set(), invalid_waivers=[InvalidWaiver(index=0, reason="waiver_file_not_object")])
    raw_waivers = _json_list_optional(payload_mapping.get("waivers"))
    if raw_waivers is None:
        return WaiverLoadResult(allowed_keys=set(), invalid_waivers=[InvalidWaiver(index=0, reason="waivers_not_list")])

    keys: set[str] = set()
    invalid: list[InvalidWaiver] = []
    required = ("path", "qualname", "kind", "rationale", "scope", "expiry", "owner")
    invalid.extend(
        InvalidWaiver(index=index, reason="waiver_not_object")
        for index, raw in enumerate(raw_waivers, start=1)
        if _json_mapping_optional(raw) is None
    )
    for index, waiver in _waiver_mappings(raw_waivers):
        missing = [field for field in required if waiver.get(field) in (None, "")]
        if missing:
            invalid.append(InvalidWaiver(index=index, reason=f"missing_fields:{','.join(missing)}"))
            continue
        structured_hash = waiver.get("structured_hash")
        line = waiver.get("line")
        if isinstance(structured_hash, str) and structured_hash:
            key = f"{waiver['path']}:{waiver['qualname']}:{waiver['kind']}:{structured_hash}"
            keys.add(key)
            continue
        scope = str(waiver.get("scope", "") or "")
        annotation = str(waiver.get("annotation", "") or "")
        column = waiver.get("column")
        if isinstance(column, int) and scope and annotation:
            migrated_hash = _structured_hash(
                str(waiver["path"]),
                str(waiver["qualname"]),
                str(waiver["kind"]),
                scope,
                annotation,
                str(column),
            )
            keys.add(f"{waiver['path']}:{waiver['qualname']}:{waiver['kind']}:{migrated_hash}")
            continue
        if isinstance(line, int):
            key = f"{waiver['path']}:{waiver['qualname']}:{line}:{waiver['kind']}"
            keys.add(key)
            continue
        invalid.append(InvalidWaiver(index=index, reason="line_or_structured_hash_required"))
    return WaiverLoadResult(allowed_keys=keys, invalid_waivers=invalid)


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
