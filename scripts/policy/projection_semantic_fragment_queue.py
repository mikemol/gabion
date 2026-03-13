#!/usr/bin/env python3
from __future__ import annotations

import ast
import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

from gabion.analysis.aspf.aspf_lattice_algebra import canonical_structural_identity
from gabion.analysis.projection.projection_registry import (
    iter_projection_fiber_semantic_specs,
)
from gabion.policy_dsl.registry import build_registry
from gabion.policy_dsl.schema import PolicyDomain
from gabion.order_contract import ordered_or_sorted
from gabion.tooling.policy_substrate.projection_semantic_fragment_phase5_registry import (
    iter_phase5_subqueues,
    iter_phase5_touchpoints,
)
from gabion.tooling.policy_substrate.site_identity import canonical_site_identity
from gabion.tooling.runtime.projection_fiber_semantics_summary import (
    projection_fiber_decision_from_payload,
    projection_fiber_semantic_bundle_count_from_payload,
    projection_fiber_semantic_previews_from_payload,
    projection_fiber_semantic_row_count_from_payload,
    projection_fiber_semantic_spec_names_from_payload,
)

_FORMAT_VERSION = 1
_DEFAULT_SOURCE_ARTIFACT = "artifacts/out/policy_check_result.json"
_MAX_SEMANTIC_PREVIEW_SAMPLES = 20
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PHASE5_PROJECTION_ADAPTER_FILES = (
    "src/gabion/analysis/projection/projection_exec.py",
    "src/gabion/analysis/projection/projection_exec_plan.py",
    "src/gabion/analysis/projection/semantic_fragment.py",
    "src/gabion/analysis/projection/semantic_fragment_compile.py",
    "src/gabion/analysis/projection/projection_semantic_lowering.py",
    "src/gabion/analysis/projection/projection_semantic_lowering_compile.py",
)
_PHASE5_SURVIVING_TOUCHSITE_BOUNDARY_NAMES = frozenset(
    {
        "semantic_fragment.normalize_value",
        "semantic_fragment.stable_json_key",
        "projection_semantic_lowering.normalize_projection_op",
        "projection_semantic_lowering.lower_projection_op",
        "projection_semantic_lowering_compile.compile_semantic_projection_op",
        "projection_semantic_lowering_compile.semantic_rows_for_quotient_face",
        "projection_semantic_lowering_compile.semantic_rows_for_surface",
        "projection_exec.apply_execution_op",
        "projection_exec.sort_value",
        "projection_exec.canonical_group_reference",
    }
)


@lru_cache(maxsize=1)
def _declared_projection_fiber_semantic_spec_names() -> tuple[str, ...]:
    return tuple(
        str(spec.name)
        for spec in iter_projection_fiber_semantic_specs()
    )


def _sorted[T](values: list[T], *, key=None) -> list[T]:
    return ordered_or_sorted(
        values,
        source="scripts.policy.projection_semantic_fragment_queue",
        key=key,
    )


@dataclass(frozen=True)
class ProjectionSemanticFragmentCurrentState:
    decision: dict[str, Any]
    semantic_row_count: int
    compiled_projection_semantic_bundle_count: int
    compiled_projection_semantic_spec_names: tuple[str, ...]
    semantic_preview_count: int
    semantic_previews: tuple[dict[str, Any], ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "semantic_row_count": self.semantic_row_count,
            "compiled_projection_semantic_bundle_count": (
                self.compiled_projection_semantic_bundle_count
            ),
            "compiled_projection_semantic_spec_names": list(
                self.compiled_projection_semantic_spec_names
            ),
            "semantic_preview_count": self.semantic_preview_count,
            "semantic_previews": [item for item in self.semantic_previews],
        }


@dataclass(frozen=True)
class ProjectionSemanticFragmentQueueItem:
    queue_id: str
    phase: str
    status: str
    title: str
    summary: str
    next_action: str
    evidence_links: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "queue_id": self.queue_id,
            "phase": self.phase,
            "status": self.status,
            "title": self.title,
            "summary": self.summary,
            "next_action": self.next_action,
            "evidence_links": [item for item in self.evidence_links],
        }


@dataclass(frozen=True)
class ProjectionSemanticFragmentQueue:
    source_artifact: str
    current_state: ProjectionSemanticFragmentCurrentState
    next_queue_ids: tuple[str, ...]
    items: tuple[ProjectionSemanticFragmentQueueItem, ...]
    phase5_structure: ProjectionSemanticFragmentPhase5Structure

    def as_payload(self) -> dict[str, object]:
        return {
            "format_version": _FORMAT_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_artifact": self.source_artifact,
            "current_state": self.current_state.as_payload(),
            "next_queue_ids": [item for item in self.next_queue_ids],
            "items": [item.as_payload() for item in self.items],
            "phase5_structure": self.phase5_structure.as_payload(),
        }


@dataclass(frozen=True)
class ProjectionSemanticFragmentPhase5Touchsite:
    touchsite_id: str
    touchpoint_id: str
    subqueue_id: str
    rel_path: str
    qualname: str
    boundary_name: str
    line: int
    column: int
    node_kind: str
    site_identity: str
    structural_identity: str
    seam_class: str
    touchpoint_marker_identity: str
    touchpoint_structural_identity: str
    subqueue_marker_identity: str
    subqueue_structural_identity: str

    def as_payload(self) -> dict[str, object]:
        return {
            "touchsite_id": self.touchsite_id,
            "touchpoint_id": self.touchpoint_id,
            "subqueue_id": self.subqueue_id,
            "rel_path": self.rel_path,
            "qualname": self.qualname,
            "boundary_name": self.boundary_name,
            "line": self.line,
            "column": self.column,
            "node_kind": self.node_kind,
            "site_identity": self.site_identity,
            "structural_identity": self.structural_identity,
            "seam_class": self.seam_class,
            "touchpoint_marker_identity": self.touchpoint_marker_identity,
            "touchpoint_structural_identity": self.touchpoint_structural_identity,
            "subqueue_marker_identity": self.subqueue_marker_identity,
            "subqueue_structural_identity": self.subqueue_structural_identity,
        }


@dataclass(frozen=True)
class ProjectionSemanticFragmentPhase5Touchpoint:
    touchpoint_id: str
    subqueue_id: str
    title: str
    rel_path: str
    site_identity: str
    structural_identity: str
    marker_identity: str
    marker_reason: str
    reasoning_summary: str
    reasoning_control: str
    blocking_dependencies: tuple[str, ...]
    object_ids: tuple[str, ...]
    touchsite_count: int
    collapsible_touchsite_count: int
    surviving_touchsite_count: int
    touchsites: tuple[ProjectionSemanticFragmentPhase5Touchsite, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "touchpoint_id": self.touchpoint_id,
            "subqueue_id": self.subqueue_id,
            "title": self.title,
            "rel_path": self.rel_path,
            "site_identity": self.site_identity,
            "structural_identity": self.structural_identity,
            "marker_identity": self.marker_identity,
            "marker_reason": self.marker_reason,
            "reasoning_summary": self.reasoning_summary,
            "reasoning_control": self.reasoning_control,
            "blocking_dependencies": list(self.blocking_dependencies),
            "object_ids": list(self.object_ids),
            "touchsite_count": self.touchsite_count,
            "collapsible_touchsite_count": self.collapsible_touchsite_count,
            "surviving_touchsite_count": self.surviving_touchsite_count,
            "touchsites": [item.as_payload() for item in self.touchsites],
        }


@dataclass(frozen=True)
class ProjectionSemanticFragmentPhase5Subqueue:
    subqueue_id: str
    title: str
    site_identity: str
    structural_identity: str
    marker_identity: str
    marker_reason: str
    reasoning_summary: str
    reasoning_control: str
    blocking_dependencies: tuple[str, ...]
    object_ids: tuple[str, ...]
    touchpoint_ids: tuple[str, ...]
    touchsite_count: int
    collapsible_touchsite_count: int
    surviving_touchsite_count: int

    def as_payload(self) -> dict[str, object]:
        return {
            "subqueue_id": self.subqueue_id,
            "title": self.title,
            "site_identity": self.site_identity,
            "structural_identity": self.structural_identity,
            "marker_identity": self.marker_identity,
            "marker_reason": self.marker_reason,
            "reasoning_summary": self.reasoning_summary,
            "reasoning_control": self.reasoning_control,
            "blocking_dependencies": list(self.blocking_dependencies),
            "object_ids": list(self.object_ids),
            "touchpoint_ids": list(self.touchpoint_ids),
            "touchsite_count": self.touchsite_count,
            "collapsible_touchsite_count": self.collapsible_touchsite_count,
            "surviving_touchsite_count": self.surviving_touchsite_count,
        }


@dataclass(frozen=True)
class ProjectionSemanticFragmentPhase5Structure:
    queue_id: str
    title: str
    remaining_touchsite_count: int
    collapsible_touchsite_count: int
    surviving_touchsite_count: int
    subqueues: tuple[ProjectionSemanticFragmentPhase5Subqueue, ...]
    touchpoints: tuple[ProjectionSemanticFragmentPhase5Touchpoint, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "queue_id": self.queue_id,
            "title": self.title,
            "remaining_touchsite_count": self.remaining_touchsite_count,
            "collapsible_touchsite_count": self.collapsible_touchsite_count,
            "surviving_touchsite_count": self.surviving_touchsite_count,
            "subqueues": [item.as_payload() for item in self.subqueues],
            "touchpoints": [item.as_payload() for item in self.touchpoints],
        }

@lru_cache(maxsize=1)
def _projection_fiber_policy_direct_carrier_judgment_landed() -> bool:
    registry = build_registry()
    blocking_rule = next(
        (
            rule
            for rule in registry.program.rules
            if rule.domain is PolicyDomain.PROJECTION_FIBER
            and rule.rule_id == "projection_fiber.convergence.blocking"
        ),
        None,
    )
    if blocking_rule is None:
        return False
    has_legacy_transform = any(
        transform.transform_id == "projection.unmapped_intro"
        for transform in registry.program.transforms
    )
    return _predicate_reads_semantic_rows(blocking_rule.predicate) and not has_legacy_transform


def _predicate_reads_semantic_rows(predicate: Mapping[str, Any]) -> bool:
    op = predicate.get("op")
    path = predicate.get("path")
    if op == "rows_any" and path == ["semantic_rows"]:
        return True
    for value in predicate.values():
        if isinstance(value, Mapping) and _predicate_reads_semantic_rows(value):
            return True
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping) and _predicate_reads_semantic_rows(item):
                    return True
    return False


@lru_cache(maxsize=1)
def _legacy_projection_exec_ingress_retired() -> bool:
    return not (
        _REPO_ROOT / "src/gabion/analysis/projection/projection_exec_ingress.py"
    ).exists()


@lru_cache(maxsize=1)
def _remaining_phase5_projection_adapter_markers() -> int:
    return _phase5_structure().remaining_touchsite_count


def _dotted_name(node: ast.AST) -> str | None:
    match node:
        case ast.Name(id=name):
            return name
        case ast.Attribute(value=value, attr=attr):
            parent = _dotted_name(value)
            if parent is None:
                return None
            return f"{parent}.{attr}"
        case _:
            return None


def _keyword_string_literal(call: ast.Call, key: str) -> str:
    for keyword in call.keywords:
        if keyword.arg != key:
            continue
        if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            return str(keyword.value.value).strip()
    return ""


def _object_ids(payload: object) -> tuple[str, ...]:
    raw_links = getattr(payload, "links", ())
    values = [
        str(link.value)
        for link in raw_links
        if str(getattr(link.kind, "value", "")) == "object_id" and str(link.value).strip()
    ]
    return tuple(_sorted(values))


def _touchsite_seam_class(*, qualname: str, boundary_name: str) -> str:
    function_name = qualname.rsplit(".", 1)[-1]
    if (
        not function_name.startswith("_")
        or boundary_name in _PHASE5_SURVIVING_TOUCHSITE_BOUNDARY_NAMES
    ):
        return "surviving_carrier_seam"
    return "collapsible_helper_seam"


class _Phase5TouchsiteScanner(ast.NodeVisitor):
    def __init__(
        self,
        *,
        rel_path: str,
        touchpoint_id: str,
        subqueue_id: str,
        touchpoint_marker_identity: str,
        touchpoint_structural_identity: str,
        subqueue_marker_identity: str,
        subqueue_structural_identity: str,
    ) -> None:
        self.rel_path = rel_path
        self.touchpoint_id = touchpoint_id
        self.subqueue_id = subqueue_id
        self.touchpoint_marker_identity = touchpoint_marker_identity
        self.touchpoint_structural_identity = touchpoint_structural_identity
        self.subqueue_marker_identity = subqueue_marker_identity
        self.subqueue_structural_identity = subqueue_structural_identity
        self._scope: list[str] = []
        self.touchsites: list[ProjectionSemanticFragmentPhase5Touchsite] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node=node, node_kind="function_def")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node=node, node_kind="async_function_def")

    def _visit_function(
        self,
        *,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        node_kind: str,
    ) -> None:
        self._scope.append(node.name)
        qualname = ".".join(self._scope)
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if _dotted_name(decorator.func) != "grade_boundary":
                continue
            if _keyword_string_literal(decorator, "kind") != "semantic_carrier_adapter":
                continue
            boundary_name = _keyword_string_literal(decorator, "name") or qualname
            line = int(node.lineno)
            column = int(node.col_offset) + 1
            site_identity = canonical_site_identity(
                rel_path=self.rel_path,
                qualname=qualname,
                line=line,
                column=column,
                node_kind=node_kind,
                surface="semantic_carrier_adapter",
            )
            structural_identity = canonical_structural_identity(
                rel_path=self.rel_path,
                qualname=qualname,
                structural_path=f"{qualname}::grade_boundary[{boundary_name}]",
                node_kind=node_kind,
                surface="semantic_carrier_adapter",
            )
            seam_class = _touchsite_seam_class(
                qualname=qualname,
                boundary_name=boundary_name,
            )
            self.touchsites.append(
                ProjectionSemanticFragmentPhase5Touchsite(
                    touchsite_id=structural_identity,
                    touchpoint_id=self.touchpoint_id,
                    subqueue_id=self.subqueue_id,
                    rel_path=self.rel_path,
                    qualname=qualname,
                    boundary_name=boundary_name,
                    line=line,
                    column=column,
                    node_kind=node_kind,
                    site_identity=site_identity,
                    structural_identity=structural_identity,
                    seam_class=seam_class,
                    touchpoint_marker_identity=self.touchpoint_marker_identity,
                    touchpoint_structural_identity=self.touchpoint_structural_identity,
                    subqueue_marker_identity=self.subqueue_marker_identity,
                    subqueue_structural_identity=self.subqueue_structural_identity,
                )
            )
        self.generic_visit(node)
        self._scope.pop()


def _scan_phase5_touchsites(
    *,
    rel_path: str,
    touchpoint_id: str,
    subqueue_id: str,
    touchpoint_marker_identity: str,
    touchpoint_structural_identity: str,
    subqueue_marker_identity: str,
    subqueue_structural_identity: str,
) -> tuple[ProjectionSemanticFragmentPhase5Touchsite, ...]:
    source = (_REPO_ROOT / rel_path).read_text(encoding="utf-8")
    scanner = _Phase5TouchsiteScanner(
        rel_path=rel_path,
        touchpoint_id=touchpoint_id,
        subqueue_id=subqueue_id,
        touchpoint_marker_identity=touchpoint_marker_identity,
        touchpoint_structural_identity=touchpoint_structural_identity,
        subqueue_marker_identity=subqueue_marker_identity,
        subqueue_structural_identity=subqueue_structural_identity,
    )
    scanner.visit(ast.parse(source, filename=rel_path))
    return tuple(
        _sorted(
            scanner.touchsites,
            key=lambda item: (item.rel_path, item.line, item.column, item.qualname),
        )
    )


@lru_cache(maxsize=1)
def _phase5_structure() -> ProjectionSemanticFragmentPhase5Structure:
    subqueue_definitions = tuple(iter_phase5_subqueues())
    touchpoint_definitions = tuple(iter_phase5_touchpoints())
    subqueue_by_id = {item.subqueue_id: item for item in subqueue_definitions}
    touchpoints_by_subqueue: dict[str, list[ProjectionSemanticFragmentPhase5Touchpoint]] = defaultdict(list)
    touchpoint_states: list[ProjectionSemanticFragmentPhase5Touchpoint] = []
    all_touchsites: list[ProjectionSemanticFragmentPhase5Touchsite] = []

    for touchpoint_definition in touchpoint_definitions:
        subqueue_definition = subqueue_by_id[touchpoint_definition.subqueue_id]
        touchsites = _scan_phase5_touchsites(
            rel_path=touchpoint_definition.rel_path,
            touchpoint_id=touchpoint_definition.touchpoint_id,
            subqueue_id=touchpoint_definition.subqueue_id,
            touchpoint_marker_identity=touchpoint_definition.marker_identity,
            touchpoint_structural_identity=touchpoint_definition.structural_identity,
            subqueue_marker_identity=subqueue_definition.marker_identity,
            subqueue_structural_identity=subqueue_definition.structural_identity,
        )
        collapsible_touchsite_count = sum(
            1 for item in touchsites if item.seam_class == "collapsible_helper_seam"
        )
        touchpoint_state = ProjectionSemanticFragmentPhase5Touchpoint(
            touchpoint_id=touchpoint_definition.touchpoint_id,
            subqueue_id=touchpoint_definition.subqueue_id,
            title=touchpoint_definition.title,
            rel_path=touchpoint_definition.rel_path,
            site_identity=touchpoint_definition.site_identity,
            structural_identity=touchpoint_definition.structural_identity,
            marker_identity=touchpoint_definition.marker_identity,
            marker_reason=touchpoint_definition.marker_payload.reason,
            reasoning_summary=touchpoint_definition.marker_payload.reasoning.summary,
            reasoning_control=touchpoint_definition.marker_payload.reasoning.control,
            blocking_dependencies=(
                touchpoint_definition.marker_payload.reasoning.blocking_dependencies
            ),
            object_ids=_object_ids(touchpoint_definition.marker_payload),
            touchsite_count=len(touchsites),
            collapsible_touchsite_count=collapsible_touchsite_count,
            surviving_touchsite_count=len(touchsites) - collapsible_touchsite_count,
            touchsites=touchsites,
        )
        touchpoint_states.append(touchpoint_state)
        touchpoints_by_subqueue[touchpoint_definition.subqueue_id].append(touchpoint_state)
        all_touchsites.extend(touchsites)

    subqueue_states = tuple(
        _sorted(
            [
                ProjectionSemanticFragmentPhase5Subqueue(
                    subqueue_id=subqueue_definition.subqueue_id,
                    title=subqueue_definition.title,
                    site_identity=subqueue_definition.site_identity,
                    structural_identity=subqueue_definition.structural_identity,
                    marker_identity=subqueue_definition.marker_identity,
                    marker_reason=subqueue_definition.marker_payload.reason,
                    reasoning_summary=subqueue_definition.marker_payload.reasoning.summary,
                    reasoning_control=subqueue_definition.marker_payload.reasoning.control,
                    blocking_dependencies=(
                        subqueue_definition.marker_payload.reasoning.blocking_dependencies
                    ),
                    object_ids=_object_ids(subqueue_definition.marker_payload),
                    touchpoint_ids=subqueue_definition.touchpoint_ids,
                    touchsite_count=sum(
                        item.touchsite_count
                        for item in touchpoints_by_subqueue[subqueue_definition.subqueue_id]
                    ),
                    collapsible_touchsite_count=sum(
                        item.collapsible_touchsite_count
                        for item in touchpoints_by_subqueue[subqueue_definition.subqueue_id]
                    ),
                    surviving_touchsite_count=sum(
                        item.surviving_touchsite_count
                        for item in touchpoints_by_subqueue[subqueue_definition.subqueue_id]
                    ),
                )
                for subqueue_definition in subqueue_definitions
            ],
            key=lambda item: item.subqueue_id,
        )
    )
    touchpoint_states = _sorted(touchpoint_states, key=lambda item: item.touchpoint_id)
    collapsible_touchsite_count = sum(
        1 for item in all_touchsites if item.seam_class == "collapsible_helper_seam"
    )
    return ProjectionSemanticFragmentPhase5Structure(
        queue_id="PSF-007",
        title="Cut over legacy adapters and retire semantic_carrier_adapter boundaries",
        remaining_touchsite_count=len(all_touchsites),
        collapsible_touchsite_count=collapsible_touchsite_count,
        surviving_touchsite_count=len(all_touchsites) - collapsible_touchsite_count,
        subqueues=subqueue_states,
        touchpoints=tuple(touchpoint_states),
    )


def analyze(
    *,
    payload: Mapping[str, object],
    source_artifact: str = _DEFAULT_SOURCE_ARTIFACT,
) -> ProjectionSemanticFragmentQueue:
    current_state = _current_state(payload)
    phase5_structure = _phase5_structure()
    items = _queue_items(current_state, phase5_structure=phase5_structure)
    next_queue_ids = tuple(
        item.queue_id for item in items if item.status != "landed"
    )
    return ProjectionSemanticFragmentQueue(
        source_artifact=source_artifact,
        current_state=current_state,
        next_queue_ids=next_queue_ids,
        items=items,
        phase5_structure=phase5_structure,
    )


def _current_state(
    payload: Mapping[str, object],
) -> ProjectionSemanticFragmentCurrentState:
    semantic_previews = tuple(
        _preview_payload(item)
        for item in projection_fiber_semantic_previews_from_payload(payload)
    )
    return ProjectionSemanticFragmentCurrentState(
        decision=projection_fiber_decision_from_payload(payload),
        semantic_row_count=projection_fiber_semantic_row_count_from_payload(payload),
        compiled_projection_semantic_bundle_count=(
            projection_fiber_semantic_bundle_count_from_payload(payload)
        ),
        compiled_projection_semantic_spec_names=(
            projection_fiber_semantic_spec_names_from_payload(payload)
        ),
        semantic_preview_count=len(semantic_previews),
        semantic_previews=semantic_previews[:_MAX_SEMANTIC_PREVIEW_SAMPLES],
    )


def _queue_items(
    current_state: ProjectionSemanticFragmentCurrentState,
    *,
    phase5_structure: ProjectionSemanticFragmentPhase5Structure,
) -> tuple[ProjectionSemanticFragmentQueueItem, ...]:
    spec_names = current_state.compiled_projection_semantic_spec_names
    spec_names_summary = ", ".join(spec_names) if spec_names else "<none>"
    row_count = current_state.semantic_row_count
    bundle_count = current_state.compiled_projection_semantic_bundle_count
    preview_count = current_state.semantic_preview_count
    semantic_lowering_landed = bundle_count > 0 and bool(spec_names)
    declared_spec_names = _declared_projection_fiber_semantic_spec_names()
    friendly_surface_convergence_landed = semantic_lowering_landed and all(
        name in spec_names for name in declared_spec_names
    )
    policy_direct_carrier_judgment_landed = (
        _projection_fiber_policy_direct_carrier_judgment_landed()
    )
    semantic_op_expansion_landed = "projection_fiber_witness_synthesis" in spec_names
    semantic_expansion_spec_names = tuple(
        name
        for name in spec_names
        if name
        not in {
            "projection_fiber_frontier",
            "projection_fiber_reflective_boundary",
        }
    )
    semantic_expansion_summary = (
        ", ".join(semantic_expansion_spec_names)
        if semantic_expansion_spec_names
        else "<none>"
    )
    phase5_cutover_criteria_met = (
        row_count > 0
        and semantic_lowering_landed
        and preview_count > 0
        and policy_direct_carrier_judgment_landed
    )
    legacy_projection_exec_ingress_retired = _legacy_projection_exec_ingress_retired()
    remaining_phase5_adapter_markers = _remaining_phase5_projection_adapter_markers()
    phase5_subqueue_count = len(phase5_structure.subqueues)
    phase5_touchpoint_count = len(phase5_structure.touchpoints)
    phase5_collapsible_touchsite_count = phase5_structure.collapsible_touchsite_count
    phase5_surviving_touchsite_count = phase5_structure.surviving_touchsite_count
    phase5_landed = (
        phase5_cutover_criteria_met
        and legacy_projection_exec_ingress_retired
        and remaining_phase5_adapter_markers == 0
    )
    phase5_in_progress = (
        phase5_cutover_criteria_met
        and legacy_projection_exec_ingress_retired
        and remaining_phase5_adapter_markers > 0
    )
    items = (
        ProjectionSemanticFragmentQueueItem(
            queue_id="PSF-001",
            phase="Phase 2",
            status="landed" if row_count > 0 else "queued",
            title="Carrier-first reflection over ASPF/fibration witnesses",
            summary=(
                f"{row_count} canonical semantic row(s) emitted into the policy surface."
                if row_count > 0
                else "Canonical semantic rows are not yet present in the source artifact payload."
            ),
            next_action=(
                "Keep structural identity / site identity continuity stable as new semantic surfaces land."
                if row_count > 0
                else "Land the canonical witnessed carrier on a real semantic path before expanding authoring surfaces."
            ),
            evidence_links=(
                "src/gabion/analysis/projection/semantic_fragment.py",
                "src/gabion/tooling/policy_substrate/lattice_convergence_semantic.py",
                "tests/gabion/tooling/runtime_policy/test_lattice_convergence_semantic.py",
            ),
        ),
        ProjectionSemanticFragmentQueueItem(
            queue_id="PSF-002",
            phase="Phase 3",
            status="landed" if semantic_lowering_landed else "queued",
            title="Deterministic SHACL/SPARQL lowering for declared quotient faces",
            summary=(
                f"{bundle_count} compiled semantic bundle(s) emitted for {spec_names_summary}."
                if semantic_lowering_landed
                else "Compiled semantic lowering is not yet present for declared quotient-face specs."
            ),
            next_action=(
                "Preserve lowering determinism and identity/witness trace continuity as additional faces are promoted."
                if semantic_lowering_landed
                else "Compile the first declared quotient-face authoring surface into SHACL/SPARQL plans."
            ),
            evidence_links=(
                "src/gabion/analysis/projection/projection_semantic_lowering.py",
                "src/gabion/analysis/projection/projection_semantic_lowering_compile.py",
                "src/gabion/analysis/projection/semantic_fragment_compile.py",
            ),
        ),
        ProjectionSemanticFragmentQueueItem(
            queue_id="PSF-003",
            phase="Phase 4",
            status="landed" if preview_count > 0 else "queued",
            title="Reporting-layer propagation of canonical semantic previews",
            summary=(
                f"{preview_count} semantic preview row(s) are now carried into queue/report artifacts."
                if preview_count > 0
                else "Reporting artifacts are not yet carrying canonical semantic previews."
            ),
            next_action=(
                "Use preview propagation as the continuity surface while broader carrier consumers are added."
                if preview_count > 0
                else "Thread canonical semantic previews into reporting surfaces that currently only see aggregate counts."
            ),
            evidence_links=(
                "src/gabion/tooling/runtime/policy_scanner_suite.py",
                "scripts/policy/hotspot_neighborhood_queue.py",
                "tests/gabion/tooling/policy/test_hotspot_neighborhood_queue.py",
            ),
        ),
        ProjectionSemanticFragmentQueueItem(
            queue_id="PSF-004",
            phase="Phase 4",
            status=(
                "landed"
                if friendly_surface_convergence_landed
                else "in_progress" if semantic_lowering_landed else "queued"
            ),
            title="Friendly-surface convergence via typed ProjectionSpec lowering",
            summary=(
                f"Typed lowering exists for {spec_names_summary}, projection_exec remains the compatibility runtime, and the projection history ledger now records per-spec lowering status."
                if semantic_lowering_landed
                else "Friendly-surface lowering has not yet been anchored to a canonical semantic path."
            ),
            next_action=(
                "Keep the declared projection_fiber semantic-spec set closed under typed lowering; add new authoring faces as semantic ops first and preserve projection history alignment."
                if friendly_surface_convergence_landed
                else "Promote additional declared semantic ops through lowering without adding new semantic behavior directly to projection_exec."
            ),
            evidence_links=(
                "src/gabion/analysis/projection/projection_registry.py",
                "scripts/policy/build_projection_spec_history.py",
                "artifacts/out/projection_spec_history_ledger.json",
                "src/gabion/analysis/projection/projection_exec.py",
                "docs/projection_semantic_fragment_rfc.md",
            ),
        ),
        ProjectionSemanticFragmentQueueItem(
            queue_id="PSF-005",
            phase="Phase 4",
            status="landed" if semantic_op_expansion_landed else "queued",
            title="Expand the semantic op set beyond declared quotient-face slices",
            summary=(
                "The typed lowering path now includes non-quotient semantic bundles for "
                f"{semantic_expansion_summary} alongside the declared quotient-face slices."
                if semantic_op_expansion_landed
                else "The reflect + declared quotient_face projection_fiber slices are executable today; the remaining RFC ops are still design-only."
            ),
            next_action=(
                "Preserve semantic-op growth through explicit typed operators; do not widen generic bridge transforms in place of lawful semantic ops."
                if semantic_op_expansion_landed
                else "Add the next smallest lawful semantic op on top of the same carrier instead of widening generic row-shaping operators."
            ),
            evidence_links=(
                "src/gabion/analysis/projection/projection_registry.py",
                "src/gabion/analysis/projection/projection_semantic_lowering.py",
                "src/gabion/analysis/projection/projection_semantic_lowering_compile.py",
                "docs/projection_semantic_fragment_rfc.md",
                "docs/ttl_kernel_semantics.md",
            ),
        ),
        ProjectionSemanticFragmentQueueItem(
            queue_id="PSF-006",
            phase="Phase 4",
            status="landed" if policy_direct_carrier_judgment_landed else "queued",
            title="Move policy and authoring consumers toward direct canonical-carrier judgment",
            summary=(
                "The projection-fiber policy DSL now judges canonical semantic_rows directly, and the retired projection.unmapped_intro bridge is no longer present in the registry."
                if policy_direct_carrier_judgment_landed
                else "The projection-fiber policy path is carrier-backed, but broader policy/authoring surfaces still depend on compatibility and summary bridges."
            ),
            next_action=(
                "Preserve direct canonical-carrier judgment for the next consumer; do not reintroduce witness-row transforms once semantic_rows exist."
                if policy_direct_carrier_judgment_landed
                else "Shift the next consumer from row-shape inference to direct canonical-carrier reads, then preserve that path with policy tests."
            ),
            evidence_links=(
                "docs/projection_fiber_rules.yaml",
                "tests/test_policy_dsl.py",
                "docs/projection_semantic_fragment_rfc.md",
            ),
        ),
        ProjectionSemanticFragmentQueueItem(
            queue_id="PSF-007",
            phase="Phase 5",
            status=(
                "landed"
                if phase5_landed
                else "in_progress" if phase5_in_progress else "queued"
            ),
            title="Cut over legacy adapters and retire semantic_carrier_adapter boundaries",
            summary=(
                "Phase 5 cutover criteria are satisfied for at least one end-to-end path, "
                "projection_exec_ingress.py is retired, and no temporary "
                "semantic_carrier_adapter markers remain on the core projection path."
                if phase5_landed
                else (
                    "Phase 5 cutover criteria are satisfied for at least one end-to-end path, "
                    "projection_exec_ingress.py is retired, and "
                    f"{remaining_phase5_adapter_markers} temporary semantic_carrier_adapter "
                    "marker(s) remain on the core projection path across "
                    f"{phase5_subqueue_count} subqueue(s), {phase5_touchpoint_count} touchpoint(s), "
                    f"{phase5_collapsible_touchsite_count} collapsible helper seam(s), and "
                    f"{phase5_surviving_touchsite_count} surviving carrier seam(s)."
                )
                if phase5_in_progress
                else "The semantic fragment has not yet satisfied the RFC Phase 5 cutover criteria on a stable end-to-end path."
            ),
            next_action=(
                "Keep the core projection path free of new temporary adapter grading and remove the remaining function-local semantic_carrier_adapter markers."
                if phase5_landed
                else (
                    "Use the RFC cutover criteria and ratchet rules to keep shrinking the remaining "
                    "function-local semantic_carrier_adapter markers until the core projection path is fully cut over."
                )
                if phase5_in_progress
                else "Use the RFC cutover criteria and ratchet rules to remove temporary adapter status only after end-to-end semantic paths are stable."
            ),
            evidence_links=(
                "src/gabion/analysis/projection/projection_exec.py",
                "src/gabion/analysis/projection/projection_exec_plan.py",
                "src/gabion/analysis/projection/semantic_fragment.py",
                "docs/projection_semantic_fragment_rfc.md",
            ),
        ),
    )
    return items


def _preview_payload(value: Mapping[str, object]) -> dict[str, Any]:
    return {
        "spec_name": _string_value(value.get("spec_name")),
        "quotient_face": _string_value(value.get("quotient_face")),
        "source_structural_identity": _string_value(
            value.get("source_structural_identity")
        ),
        "path": _string_value(value.get("path")),
        "qualname": _string_value(value.get("qualname")),
        "structural_path": _string_value(value.get("structural_path")),
        "obligation_state": _string_value(value.get("obligation_state")),
        "complete": bool(value.get("complete")) if isinstance(value.get("complete"), bool) else False,
    }

def _string_value(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _markdown_summary(queue: ProjectionSemanticFragmentQueue) -> str:
    current_state = queue.current_state
    phase5_structure = queue.phase5_structure
    spec_names = ", ".join(current_state.compiled_projection_semantic_spec_names) or "<none>"
    lines = [
        "# Projection Semantic Fragment Queue",
        "",
        f"- source_artifact: `{queue.source_artifact}`",
        f"- decision_rule: `{_string_value(current_state.decision.get('rule_id')) or '<none>'}`",
        f"- semantic_rows: `{current_state.semantic_row_count}`",
        (
            "- compiled_projection_semantic_bundles: "
            f"`{current_state.compiled_projection_semantic_bundle_count}`"
        ),
        f"- compiled_specs: `{spec_names}`",
        f"- semantic_preview_count: `{current_state.semantic_preview_count}`",
        (
            "- semantic_preview_samples: "
            f"`{len(current_state.semantic_previews)}`"
        ),
        "",
        "## Next Queue",
    ]
    if queue.next_queue_ids:
        lines.extend(f"- `{item}`" for item in queue.next_queue_ids)
    else:
        lines.append("- `<none>`")
    lines.extend(
        [
            "",
            "## Queue",
            "",
            "| id | phase | status | title |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in queue.items:
        lines.append(
            f"| {item.queue_id} | {item.phase} | {item.status} | {item.title} |"
        )
    lines.extend(
        [
            "",
            "## Phase 5 Structure",
            "",
            f"- remaining_touchsites: `{phase5_structure.remaining_touchsite_count}`",
            f"- collapsible_touchsites: `{phase5_structure.collapsible_touchsite_count}`",
            f"- surviving_touchsites: `{phase5_structure.surviving_touchsite_count}`",
            "",
            "### Subqueues",
            "",
            "| id | touchpoints | touchsites | collapsible | surviving | control |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for subqueue in phase5_structure.subqueues:
        lines.append(
            "| {subqueue_id} | {touchpoints} | {touchsites} | {collapsible} | {surviving} | {control} |".format(
                subqueue_id=subqueue.subqueue_id,
                touchpoints=len(subqueue.touchpoint_ids),
                touchsites=subqueue.touchsite_count,
                collapsible=subqueue.collapsible_touchsite_count,
                surviving=subqueue.surviving_touchsite_count,
                control=subqueue.reasoning_control or "<none>",
            )
        )
    lines.extend(
        [
            "",
            "### Touchpoints",
            "",
            "| id | subqueue | path | touchsites | collapsible | surviving |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for touchpoint in phase5_structure.touchpoints:
        lines.append(
            "| {touchpoint_id} | {subqueue_id} | {rel_path} | {touchsites} | {collapsible} | {surviving} |".format(
                touchpoint_id=touchpoint.touchpoint_id,
                subqueue_id=touchpoint.subqueue_id,
                rel_path=touchpoint.rel_path,
                touchsites=touchpoint.touchsite_count,
                collapsible=touchpoint.collapsible_touchsite_count,
                surviving=touchpoint.surviving_touchsite_count,
            )
        )
    lines.extend(
        [
            "",
            "### Touchsites",
            "",
            "| touchpoint | qualname | boundary | class | line |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for touchpoint in phase5_structure.touchpoints:
        for touchsite in touchpoint.touchsites:
            lines.append(
                "| {touchpoint_id} | {qualname} | {boundary_name} | {seam_class} | {line} |".format(
                    touchpoint_id=touchsite.touchpoint_id,
                    qualname=touchsite.qualname,
                    boundary_name=touchsite.boundary_name,
                    seam_class=touchsite.seam_class,
                    line=touchsite.line,
                )
            )
    if current_state.semantic_previews:
        lines.extend(
            [
                "",
                "## Semantic Previews",
                "",
                "| spec | quotient_face | path | qualname | structural_path |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for preview in current_state.semantic_previews:
            lines.append(
                "| {spec} | {face} | {path} | {qualname} | {structural_path} |".format(
                    spec=_string_value(preview.get("spec_name")),
                    face=_string_value(preview.get("quotient_face")),
                    path=_string_value(preview.get("path")),
                    qualname=_string_value(preview.get("qualname")),
                    structural_path=_string_value(preview.get("structural_path")),
                )
            )
    return "\n".join(lines) + "\n"


def run(
    *,
    source_artifact_path: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> int:
    payload = json.loads(source_artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("projection semantic fragment source payload must be a mapping")
    queue = analyze(
        payload=payload,
        source_artifact=str(source_artifact_path),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(queue.as_payload(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(
            _markdown_summary(queue),
            encoding="utf-8",
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-artifact",
        default=_DEFAULT_SOURCE_ARTIFACT,
    )
    parser.add_argument(
        "--out",
        default="artifacts/out/projection_semantic_fragment_queue.json",
    )
    parser.add_argument(
        "--markdown-out",
        default="artifacts/out/projection_semantic_fragment_queue.md",
    )
    args = parser.parse_args(argv)
    return run(
        source_artifact_path=Path(args.source_artifact).resolve(),
        out_path=Path(args.out).resolve(),
        markdown_out=Path(args.markdown_out).resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
