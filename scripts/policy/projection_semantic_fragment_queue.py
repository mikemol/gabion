#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

from gabion.analysis.projection.projection_registry import (
    iter_projection_fiber_semantic_specs,
)
from gabion.policy_dsl.registry import build_registry
from gabion.policy_dsl.schema import PolicyDomain
from gabion.order_contract import ordered_or_sorted
from gabion.invariants import never
from gabion.tooling.policy_substrate.invariant_graph import (
    build_invariant_graph,
    build_invariant_workstreams,
    load_invariant_workstreams,
)
from gabion.tooling.policy_substrate.policy_artifact_stream import (
    ArtifactSourceRef,
    mapping_document,
    render_markdown,
    write_json,
    write_markdown,
)
from gabion.tooling.policy_substrate.policy_queue_identity import (
    PolicyQueueIdentitySpace,
    SiteReferenceId,
    StructuralReferenceId,
    SubqueueId,
    TouchpointId,
    TouchsiteId,
    WorkstreamId,
    encode_policy_queue_identity,
)
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
    planning_chain: "ProjectionSemanticFragmentPlanningChain | None"
    evidence_links: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "queue_id": self.queue_id,
            "phase": self.phase,
            "status": self.status,
            "title": self.title,
            "summary": self.summary,
            "next_action": self.next_action,
            "planning_chain": (
                None if self.planning_chain is None else self.planning_chain.as_payload()
            ),
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
    touchsite_id: TouchsiteId
    touchpoint_id: TouchpointId
    subqueue_id: SubqueueId
    rel_path: str
    qualname: str
    boundary_name: str
    line: int
    column: int
    node_kind: str
    site_identity: SiteReferenceId
    structural_identity: StructuralReferenceId
    seam_class: str
    touchpoint_marker_identity: str
    touchpoint_structural_identity: StructuralReferenceId
    subqueue_marker_identity: str
    subqueue_structural_identity: StructuralReferenceId

    def as_payload(self) -> dict[str, object]:
        return {
            "touchsite_id": encode_policy_queue_identity(self.touchsite_id),
            "touchpoint_id": encode_policy_queue_identity(self.touchpoint_id),
            "subqueue_id": encode_policy_queue_identity(self.subqueue_id),
            "rel_path": self.rel_path,
            "qualname": self.qualname,
            "boundary_name": self.boundary_name,
            "line": self.line,
            "column": self.column,
            "node_kind": self.node_kind,
            "site_identity": encode_policy_queue_identity(self.site_identity),
            "structural_identity": encode_policy_queue_identity(self.structural_identity),
            "seam_class": self.seam_class,
            "touchpoint_marker_identity": self.touchpoint_marker_identity,
            "touchpoint_structural_identity": encode_policy_queue_identity(
                self.touchpoint_structural_identity
            ),
            "subqueue_marker_identity": self.subqueue_marker_identity,
            "subqueue_structural_identity": encode_policy_queue_identity(
                self.subqueue_structural_identity
            ),
        }


@dataclass(frozen=True)
class ProjectionSemanticFragmentPhase5Touchpoint:
    touchpoint_id: TouchpointId
    subqueue_id: SubqueueId
    title: str
    rel_path: str
    site_identity: SiteReferenceId
    structural_identity: StructuralReferenceId
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
            "touchpoint_id": encode_policy_queue_identity(self.touchpoint_id),
            "subqueue_id": encode_policy_queue_identity(self.subqueue_id),
            "title": self.title,
            "rel_path": self.rel_path,
            "site_identity": encode_policy_queue_identity(self.site_identity),
            "structural_identity": encode_policy_queue_identity(self.structural_identity),
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
    subqueue_id: SubqueueId
    title: str
    site_identity: SiteReferenceId
    structural_identity: StructuralReferenceId
    marker_identity: str
    marker_reason: str
    reasoning_summary: str
    reasoning_control: str
    blocking_dependencies: tuple[str, ...]
    object_ids: tuple[str, ...]
    touchpoint_ids: tuple[TouchpointId, ...]
    touchsite_count: int
    collapsible_touchsite_count: int
    surviving_touchsite_count: int

    def as_payload(self) -> dict[str, object]:
        return {
            "subqueue_id": encode_policy_queue_identity(self.subqueue_id),
            "title": self.title,
            "site_identity": encode_policy_queue_identity(self.site_identity),
            "structural_identity": encode_policy_queue_identity(self.structural_identity),
            "marker_identity": self.marker_identity,
            "marker_reason": self.marker_reason,
            "reasoning_summary": self.reasoning_summary,
            "reasoning_control": self.reasoning_control,
            "blocking_dependencies": list(self.blocking_dependencies),
            "object_ids": list(self.object_ids),
            "touchpoint_ids": [
                encode_policy_queue_identity(item) for item in self.touchpoint_ids
            ],
            "touchsite_count": self.touchsite_count,
            "collapsible_touchsite_count": self.collapsible_touchsite_count,
            "surviving_touchsite_count": self.surviving_touchsite_count,
        }


@dataclass(frozen=True)
class ProjectionSemanticFragmentPlanningChain:
    observed_state: str
    next_slice: str
    stabilization_goal: str

    def as_payload(self) -> dict[str, object]:
        return {
            "observed_state": self.observed_state,
            "next_slice": self.next_slice,
            "stabilization_goal": self.stabilization_goal,
        }


@dataclass(frozen=True)
class ProjectionSemanticFragmentPhase5Frontier:
    action_kind: str
    object_id: str
    owner_object_id: str
    title: str
    blocker_class: str
    touchsite_count: int
    surviving_touchsite_count: int
    decision_mode: str
    decision_reason: str
    same_kind_pressure: str
    cross_kind_pressure: str

    def as_payload(self) -> dict[str, object]:
        return {
            "action_kind": self.action_kind,
            "object_id": self.object_id,
            "owner_object_id": self.owner_object_id,
            "title": self.title,
            "blocker_class": self.blocker_class,
            "touchsite_count": self.touchsite_count,
            "surviving_touchsite_count": self.surviving_touchsite_count,
            "decision_mode": self.decision_mode,
            "decision_reason": self.decision_reason,
            "same_kind_pressure": self.same_kind_pressure,
            "cross_kind_pressure": self.cross_kind_pressure,
        }


@dataclass(frozen=True)
class ProjectionSemanticFragmentPhase5Structure:
    queue_id: WorkstreamId
    title: str
    remaining_touchsite_count: int
    collapsible_touchsite_count: int
    surviving_touchsite_count: int
    subqueues: tuple[ProjectionSemanticFragmentPhase5Subqueue, ...]
    touchpoints: tuple[ProjectionSemanticFragmentPhase5Touchpoint, ...]
    current_frontier: ProjectionSemanticFragmentPhase5Frontier | None = None

    def as_payload(self) -> dict[str, object]:
        return {
            "queue_id": encode_policy_queue_identity(self.queue_id),
            "title": self.title,
            "remaining_touchsite_count": self.remaining_touchsite_count,
            "collapsible_touchsite_count": self.collapsible_touchsite_count,
            "surviving_touchsite_count": self.surviving_touchsite_count,
            "subqueues": [item.as_payload() for item in self.subqueues],
            "touchpoints": [item.as_payload() for item in self.touchpoints],
            "current_frontier": (
                None if self.current_frontier is None else self.current_frontier.as_payload()
            ),
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


def _phase5_frontier_from_workstream_payload(
    raw_workstream: Mapping[str, object],
) -> ProjectionSemanticFragmentPhase5Frontier | None:
    raw_next_actions = raw_workstream.get("next_actions", {})
    if not isinstance(raw_next_actions, Mapping):
        return None
    raw_frontier = raw_next_actions.get("recommended_followup")
    raw_decision_protocol = raw_next_actions.get("recommended_cut_decision_protocol")
    if not isinstance(raw_frontier, Mapping) or not isinstance(
        raw_decision_protocol, Mapping
    ):
        return None
    frontier_object_id = _string_value(raw_frontier.get("object_id"))
    frontier_owner_object_id = _string_value(raw_frontier.get("owner_object_id"))
    frontier_title = _string_value(raw_frontier.get("title"))
    frontier_blocker_class = _string_value(raw_frontier.get("blocker_class"))
    frontier_action_kind = _string_value(raw_frontier.get("action_kind"))
    decision_mode = _string_value(raw_decision_protocol.get("decision_mode"))
    decision_reason = _string_value(raw_decision_protocol.get("decision_reason"))
    same_kind_pressure = _string_value(raw_decision_protocol.get("same_kind_pressure"))
    cross_kind_pressure = _string_value(
        raw_decision_protocol.get("cross_kind_pressure")
    )
    if not (
        frontier_object_id
        and frontier_owner_object_id
        and frontier_title
        and frontier_blocker_class
        and frontier_action_kind
        and decision_mode
        and decision_reason
        and same_kind_pressure
        and cross_kind_pressure
    ):
        return None
    return ProjectionSemanticFragmentPhase5Frontier(
        action_kind=frontier_action_kind,
        object_id=frontier_object_id,
        owner_object_id=frontier_owner_object_id,
        title=frontier_title,
        blocker_class=frontier_blocker_class,
        touchsite_count=int(raw_frontier.get("touchsite_count", 0)),
        surviving_touchsite_count=int(
            raw_frontier.get("surviving_touchsite_count", 0)
        ),
        decision_mode=decision_mode,
        decision_reason=decision_reason,
        same_kind_pressure=same_kind_pressure,
        cross_kind_pressure=cross_kind_pressure,
    )


def _phase5_planning_chain(
    *,
    phase5_structure: ProjectionSemanticFragmentPhase5Structure,
    phase5_in_progress: bool,
) -> ProjectionSemanticFragmentPlanningChain | None:
    if not phase5_in_progress:
        return None
    frontier = phase5_structure.current_frontier
    if frontier is None:
        never(
            reasoning={
                "summary": (
                    "Phase-5 queue planning must resolve a concrete frontier before "
                    "emitting an in-progress queue row."
                ),
                "control": "projection_semantic_fragment_queue.phase5_requires_frontier",
                "blocking_dependencies": (phase5_structure.queue_id.wire(),),
            },
            queue_id=phase5_structure.queue_id.wire(),
            remaining_touchsite_count=phase5_structure.remaining_touchsite_count,
            collapsible_touchsite_count=phase5_structure.collapsible_touchsite_count,
            surviving_touchsite_count=phase5_structure.surviving_touchsite_count,
        )
        return None  # pragma: no cover - never() raises
    return ProjectionSemanticFragmentPlanningChain(
        observed_state=(
            "X happened: "
            f"Phase 5 still has {phase5_structure.remaining_touchsite_count} remaining "
            f"marker(s), and the planner currently ranks {frontier.object_id} "
            f"({frontier.title}) as the active {frontier.blocker_class} "
            f"{frontier.action_kind} frontier."
        ),
        next_slice=(
            "So Y: "
            f"take {frontier.object_id} next, because it keeps the cut on the smallest "
            f"currently-ranked surface at {frontier.touchsite_count} touchsite(s) and "
            f"{frontier.surviving_touchsite_count} surviving seam(s)."
        ),
        stabilization_goal=(
            "So that Z: "
            f"the queue can hold {frontier.decision_mode} with same-kind pressure "
            f"{frontier.same_kind_pressure} and cross-kind pressure "
            f"{frontier.cross_kind_pressure} while the next correction unit ratchets the "
            "frontier before widening to sibling cuts."
        ),
    )


def _resolve_source_artifact_path(source_artifact: str) -> Path:
    source_path = Path(source_artifact)
    if not source_path.is_absolute():
        source_path = (_REPO_ROOT / source_path).resolve()
    return source_path


def _load_phase5_workstreams_projection(source_artifact: str) -> Mapping[str, object]:
    source_path = _resolve_source_artifact_path(source_artifact)
    workstreams_artifact = source_path.parent / "invariant_workstreams.json"
    if workstreams_artifact.exists():
        return load_invariant_workstreams(workstreams_artifact)
    return build_invariant_workstreams(
        build_invariant_graph(_REPO_ROOT),
        root=_REPO_ROOT,
    ).as_payload()


def _phase5_structure_from_projection(
    phase5_workstreams_projection: Mapping[str, object],
) -> ProjectionSemanticFragmentPhase5Structure:
    identity_space = PolicyQueueIdentitySpace()
    current_frontier: ProjectionSemanticFragmentPhase5Frontier | None = None
    workstreams = phase5_workstreams_projection.get("workstreams", [])
    raw_workstream = next(
        (
            item
            for item in workstreams
            if isinstance(item, Mapping) and str(item.get("object_id", "")) == "PSF-007"
        ),
        None,
    )
    if not isinstance(raw_workstream, Mapping):
        raise KeyError("PSF-007")
    current_frontier = _phase5_frontier_from_workstream_payload(raw_workstream)
    projection = {
        "title": str(raw_workstream.get("title", "")),
        "remaining_touchsite_count": int(raw_workstream.get("touchsite_count", 0)),
        "collapsible_touchsite_count": int(
            raw_workstream.get("collapsible_touchsite_count", 0)
        ),
        "surviving_touchsite_count": int(
            raw_workstream.get("surviving_touchsite_count", 0)
        ),
        "subqueues": [
            {
                "subqueue_id": str(item.get("object_id", "")),
                "title": str(item.get("title", "")),
                "site_identity": str(item.get("site_identity", "")),
                "structural_identity": str(item.get("structural_identity", "")),
                "marker_identity": str(item.get("marker_identity", "")),
                "marker_reason": str(item.get("reasoning_summary", "")),
                "reasoning_summary": str(item.get("reasoning_summary", "")),
                "reasoning_control": str(item.get("reasoning_control", "")),
                "blocking_dependencies": [
                    str(value) for value in item.get("blocking_dependencies", [])
                ],
                "object_ids": [str(value) for value in item.get("object_ids", [])],
                "touchpoint_ids": [
                    str(value) for value in item.get("touchpoint_ids", [])
                ],
                "touchsite_count": int(item.get("touchsite_count", 0)),
                "collapsible_touchsite_count": int(
                    item.get("collapsible_touchsite_count", 0)
                ),
                "surviving_touchsite_count": int(
                    item.get("surviving_touchsite_count", 0)
                ),
            }
            for item in raw_workstream.get("subqueues", [])
            if isinstance(item, Mapping)
        ],
        "touchpoints": [
            {
                "touchpoint_id": str(item.get("object_id", "")),
                "subqueue_id": str(item.get("subqueue_id", "")),
                "title": str(item.get("title", "")),
                "rel_path": str(item.get("rel_path", "")),
                "site_identity": str(item.get("site_identity", "")),
                "structural_identity": str(item.get("structural_identity", "")),
                "marker_identity": str(item.get("marker_identity", "")),
                "marker_reason": str(item.get("reasoning_summary", "")),
                "reasoning_summary": str(item.get("reasoning_summary", "")),
                "reasoning_control": str(item.get("reasoning_control", "")),
                "blocking_dependencies": [
                    str(value) for value in item.get("blocking_dependencies", [])
                ],
                "object_ids": [str(value) for value in item.get("object_ids", [])],
                "touchsite_count": int(item.get("touchsite_count", 0)),
                "collapsible_touchsite_count": int(
                    item.get("collapsible_touchsite_count", 0)
                ),
                "surviving_touchsite_count": int(
                    item.get("surviving_touchsite_count", 0)
                ),
                "touchsites": [
                    {
                        "touchsite_id": str(touchsite.get("object_id", "")),
                        "touchpoint_id": str(touchsite.get("touchpoint_id", "")),
                        "subqueue_id": str(touchsite.get("subqueue_id", "")),
                        "rel_path": str(touchsite.get("rel_path", "")),
                        "qualname": str(touchsite.get("qualname", "")),
                        "boundary_name": str(touchsite.get("boundary_name", "")),
                        "line": int(touchsite.get("line", 0)),
                        "column": int(touchsite.get("column", 0)),
                        "node_kind": str(touchsite.get("node_kind", "")),
                        "site_identity": str(touchsite.get("site_identity", "")),
                        "structural_identity": str(
                            touchsite.get("structural_identity", "")
                        ),
                        "seam_class": str(touchsite.get("seam_class", "")),
                        "touchpoint_marker_identity": str(
                            touchsite.get("touchpoint_marker_identity", "")
                        ),
                        "touchpoint_structural_identity": str(
                            touchsite.get("touchpoint_structural_identity", "")
                        ),
                        "subqueue_marker_identity": str(
                            touchsite.get("subqueue_marker_identity", "")
                        ),
                        "subqueue_structural_identity": str(
                            touchsite.get("subqueue_structural_identity", "")
                        ),
                    }
                    for touchsite in item.get("touchsites", [])
                    if isinstance(touchsite, Mapping)
                ],
            }
            for item in raw_workstream.get("touchpoints", [])
            if isinstance(item, Mapping)
        ],
    }
    subqueue_states = tuple(
        ProjectionSemanticFragmentPhase5Subqueue(
            subqueue_id=identity_space.subqueue_id(str(item.get("subqueue_id", ""))),
            title=str(item.get("title", "")),
            site_identity=identity_space.site_ref_id(str(item.get("site_identity", ""))),
            structural_identity=identity_space.structural_ref_id(
                str(item.get("structural_identity", ""))
            ),
            marker_identity=str(item.get("marker_identity", "")),
            marker_reason=str(item.get("marker_reason", "")),
            reasoning_summary=str(item.get("reasoning_summary", "")),
            reasoning_control=str(item.get("reasoning_control", "")),
            blocking_dependencies=tuple(
                str(value) for value in item.get("blocking_dependencies", [])
            ),
            object_ids=tuple(str(value) for value in item.get("object_ids", [])),
            touchpoint_ids=tuple(
                identity_space.touchpoint_id(str(value))
                for value in item.get("touchpoint_ids", [])
            ),
            touchsite_count=int(item.get("touchsite_count", 0)),
            collapsible_touchsite_count=int(item.get("collapsible_touchsite_count", 0)),
            surviving_touchsite_count=int(item.get("surviving_touchsite_count", 0)),
        )
        for item in projection.get("subqueues", [])
        if isinstance(item, Mapping)
    )
    touchpoint_states = tuple(
        ProjectionSemanticFragmentPhase5Touchpoint(
            touchpoint_id=identity_space.touchpoint_id(
                str(item.get("touchpoint_id", ""))
            ),
            subqueue_id=identity_space.subqueue_id(str(item.get("subqueue_id", ""))),
            title=str(item.get("title", "")),
            rel_path=str(item.get("rel_path", "")),
            site_identity=identity_space.site_ref_id(str(item.get("site_identity", ""))),
            structural_identity=identity_space.structural_ref_id(
                str(item.get("structural_identity", ""))
            ),
            marker_identity=str(item.get("marker_identity", "")),
            marker_reason=str(item.get("marker_reason", "")),
            reasoning_summary=str(item.get("reasoning_summary", "")),
            reasoning_control=str(item.get("reasoning_control", "")),
            blocking_dependencies=tuple(
                str(value) for value in item.get("blocking_dependencies", [])
            ),
            object_ids=tuple(str(value) for value in item.get("object_ids", [])),
            touchsite_count=int(item.get("touchsite_count", 0)),
            collapsible_touchsite_count=int(item.get("collapsible_touchsite_count", 0)),
            surviving_touchsite_count=int(item.get("surviving_touchsite_count", 0)),
            touchsites=tuple(
                ProjectionSemanticFragmentPhase5Touchsite(
                    touchsite_id=identity_space.touchsite_id(
                        str(touchsite.get("touchsite_id", ""))
                    ),
                    touchpoint_id=identity_space.touchpoint_id(
                        str(touchsite.get("touchpoint_id", ""))
                    ),
                    subqueue_id=identity_space.subqueue_id(
                        str(touchsite.get("subqueue_id", ""))
                    ),
                    rel_path=str(touchsite.get("rel_path", "")),
                    qualname=str(touchsite.get("qualname", "")),
                    boundary_name=str(touchsite.get("boundary_name", "")),
                    line=int(touchsite.get("line", 0)),
                    column=int(touchsite.get("column", 0)),
                    node_kind=str(touchsite.get("node_kind", "")),
                    site_identity=identity_space.site_ref_id(
                        str(touchsite.get("site_identity", ""))
                    ),
                    structural_identity=identity_space.structural_ref_id(
                        str(touchsite.get("structural_identity", ""))
                    ),
                    seam_class=str(touchsite.get("seam_class", "")),
                    touchpoint_marker_identity=str(
                        touchsite.get("touchpoint_marker_identity", "")
                    ),
                    touchpoint_structural_identity=identity_space.structural_ref_id(
                        str(touchsite.get("touchpoint_structural_identity", ""))
                    ),
                    subqueue_marker_identity=str(
                        touchsite.get("subqueue_marker_identity", "")
                    ),
                    subqueue_structural_identity=identity_space.structural_ref_id(
                        str(touchsite.get("subqueue_structural_identity", ""))
                    ),
                )
                for touchsite in item.get("touchsites", [])
                if isinstance(touchsite, Mapping)
            ),
        )
        for item in projection.get("touchpoints", [])
        if isinstance(item, Mapping)
    )
    return ProjectionSemanticFragmentPhase5Structure(
        queue_id=identity_space.workstream_id("PSF-007"),
        title=str(projection.get("title", "")),
        remaining_touchsite_count=int(projection.get("remaining_touchsite_count", 0)),
        collapsible_touchsite_count=int(projection.get("collapsible_touchsite_count", 0)),
        surviving_touchsite_count=int(projection.get("surviving_touchsite_count", 0)),
        subqueues=subqueue_states,
        touchpoints=touchpoint_states,
        current_frontier=current_frontier,
    )


@lru_cache(maxsize=8)
def _phase5_structure(source_artifact: str) -> ProjectionSemanticFragmentPhase5Structure:
    return _phase5_structure_from_projection(
        _load_phase5_workstreams_projection(source_artifact)
    )


def analyze(
    *,
    payload: Mapping[str, object],
    source_artifact: str = _DEFAULT_SOURCE_ARTIFACT,
    phase5_workstreams_projection: Mapping[str, object] | None = None,
) -> ProjectionSemanticFragmentQueue:
    current_state = _current_state(payload)
    phase5_structure = (
        _phase5_structure(source_artifact)
        if phase5_workstreams_projection is None
        else _phase5_structure_from_projection(phase5_workstreams_projection)
    )
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
    remaining_phase5_adapter_markers = phase5_structure.remaining_touchsite_count
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
    phase5_planning_chain = _phase5_planning_chain(
        phase5_structure=phase5_structure,
        phase5_in_progress=phase5_in_progress,
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
            planning_chain=None,
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
            planning_chain=None,
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
            planning_chain=None,
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
            planning_chain=None,
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
            planning_chain=None,
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
            planning_chain=None,
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
            planning_chain=phase5_planning_chain,
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
    return render_markdown(_queue_document(queue))


def _queue_document(queue: ProjectionSemanticFragmentQueue):
    return mapping_document(
        identity=ArtifactSourceRef(
            rel_path="<synthetic>",
            qualname="projection_semantic_fragment_queue",
        ),
        title="Projection Semantic Fragment Queue",
        payload=queue.as_payload(),
    )


def run(
    *,
    source_artifact_path: Path,
    out_path: Path,
    markdown_out: Path | None = None,
    phase5_workstreams_projection: Mapping[str, object] | None = None,
) -> int:
    payload = json.loads(source_artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("projection semantic fragment source payload must be a mapping")
    queue = analyze(
        payload=payload,
        source_artifact=str(source_artifact_path),
        phase5_workstreams_projection=phase5_workstreams_projection,
    )
    write_json(out_path, _queue_document(queue))
    if markdown_out is not None:
        write_markdown(markdown_out, _queue_document(queue))
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
