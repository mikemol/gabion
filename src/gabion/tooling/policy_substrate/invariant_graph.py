from __future__ import annotations

import ast
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import cached_property
import json
from pathlib import Path
from typing import cast

from gabion.analysis.aspf.aspf_lattice_algebra import (
    ReplayableStream,
    canonical_structural_identity,
)
from gabion.analysis.foundation.marker_protocol import SemanticLinkKind
from gabion.frontmatter import parse_strict_yaml_frontmatter
from gabion.order_contract import ordered_or_sorted
from gabion.tooling.policy_substrate.invariant_marker_scan import (
    InvariantMarkerScanNode,
    scan_invariant_markers,
)
from gabion.tooling.policy_substrate.grade_monotonicity_semantic import (
    grade_monotonicity_governance_priority_rank,
)
from gabion.tooling.policy_substrate.policy_artifact_stream import (
    ArtifactColumn,
    ArtifactSourceRef,
    ArtifactUnit,
    bullet_list,
    cell,
    document,
    list_item,
    paragraph,
    render_json_value,
    row,
    scalar,
    section,
    table,
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
from gabion.tooling.policy_substrate.policy_rule_frontmatter_migration_registry import (
    PolicyRuleFrontmatterMigrationQueueDefinition,
    PolicyRuleFrontmatterMigrationSubqueueDefinition,
    iter_prf_queues,
    iter_prf_subqueues,
)
from gabion.tooling.policy_substrate.projection_semantic_fragment_phase5_registry import (
    ProjectionSemanticFragmentPhase5QueueDefinition,
    ProjectionSemanticFragmentPhase5SubqueueDefinition,
    ProjectionSemanticFragmentPhase5TouchpointDefinition,
    iter_phase5_queues,
    iter_phase5_subqueues,
    iter_phase5_touchpoints,
)
from gabion.tooling.policy_substrate.site_identity import (
    canonical_site_identity,
    stable_hash,
)

_FORMAT_VERSION = 1
_REPO_ROOT = Path(__file__).resolve().parents[4]
_AMBIGUITY_ARTIFACT = Path("artifacts/out/ambiguity_contract_policy_check.json")
_TEST_EVIDENCE_ARTIFACT = Path("out/test_evidence.json")
_DEFAULT_GOVERNANCE_PRIORITY_RANK = 999


def _sorted[T](values: list[T], *, key=None) -> list[T]:
    return ordered_or_sorted(
        values,
        source="tooling.policy_substrate.invariant_graph",
        key=key,
    )


def _stream_from_iterable[T](factory) -> ReplayableStream[T]:
    return ReplayableStream(factory=factory)


@dataclass(frozen=True)
class InvariantGraphNode:
    node_id: str
    node_kind: str
    title: str
    marker_name: str
    marker_kind: str
    marker_id: str
    site_identity: str
    structural_identity: str
    object_ids: tuple[str, ...]
    doc_ids: tuple[str, ...]
    policy_ids: tuple[str, ...]
    invariant_ids: tuple[str, ...]
    reasoning_summary: str
    reasoning_control: str
    blocking_dependencies: tuple[str, ...]
    rel_path: str
    qualname: str
    line: int
    column: int
    ast_node_kind: str
    seam_class: str
    source_marker_node_id: str
    status_hint: str

    def matches_raw_id(self, raw_id: str) -> bool:
        values = {
            self.node_id,
            self.marker_id,
            self.site_identity,
            self.structural_identity,
            *self.object_ids,
            *self.doc_ids,
            *self.policy_ids,
            *self.invariant_ids,
        }
        return raw_id in values

    def as_payload(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "node_kind": self.node_kind,
            "title": self.title,
            "marker_name": self.marker_name,
            "marker_kind": self.marker_kind,
            "marker_id": self.marker_id,
            "site_identity": self.site_identity,
            "structural_identity": self.structural_identity,
            "object_ids": list(self.object_ids),
            "doc_ids": list(self.doc_ids),
            "policy_ids": list(self.policy_ids),
            "invariant_ids": list(self.invariant_ids),
            "reasoning_summary": self.reasoning_summary,
            "reasoning_control": self.reasoning_control,
            "blocking_dependencies": list(self.blocking_dependencies),
            "rel_path": self.rel_path,
            "qualname": self.qualname,
            "line": self.line,
            "column": self.column,
            "ast_node_kind": self.ast_node_kind,
            "seam_class": self.seam_class,
            "source_marker_node_id": self.source_marker_node_id,
            "status_hint": self.status_hint,
        }


@dataclass(frozen=True)
class InvariantGraphEdge:
    edge_id: str
    edge_kind: str
    source_id: str
    target_id: str

    def as_payload(self) -> dict[str, object]:
        return {
            "edge_id": self.edge_id,
            "edge_kind": self.edge_kind,
            "source_id": self.source_id,
            "target_id": self.target_id,
        }


@dataclass(frozen=True)
class InvariantGraphDiagnostic:
    diagnostic_id: str
    severity: str
    code: str
    node_id: str
    raw_dependency: str
    message: str

    def as_payload(self) -> dict[str, object]:
        return {
            "diagnostic_id": self.diagnostic_id,
            "severity": self.severity,
            "code": self.code,
            "node_id": self.node_id,
            "raw_dependency": self.raw_dependency,
            "message": self.message,
        }


@dataclass(frozen=True)
class InvariantGraph:
    root: str
    workstream_root_ids: tuple[str, ...]
    nodes: tuple[InvariantGraphNode, ...]
    edges: tuple[InvariantGraphEdge, ...]
    diagnostics: tuple[InvariantGraphDiagnostic, ...]

    def node_by_id(self) -> dict[str, InvariantGraphNode]:
        return {node.node_id: node for node in self.nodes}

    def edges_from(self) -> dict[str, tuple[InvariantGraphEdge, ...]]:
        grouped: defaultdict[str, list[InvariantGraphEdge]] = defaultdict(list)
        for edge in self.edges:
            grouped[edge.source_id].append(edge)
        return {
            key: tuple(_sorted(values, key=lambda item: (item.edge_kind, item.target_id)))
            for key, values in grouped.items()
        }

    def edges_to(self) -> dict[str, tuple[InvariantGraphEdge, ...]]:
        grouped: defaultdict[str, list[InvariantGraphEdge]] = defaultdict(list)
        for edge in self.edges:
            grouped[edge.target_id].append(edge)
        return {
            key: tuple(_sorted(values, key=lambda item: (item.edge_kind, item.source_id)))
            for key, values in grouped.items()
        }

    def as_payload(self) -> dict[str, object]:
        node_kind_counts: dict[str, int] = defaultdict(int)
        edge_kind_counts: dict[str, int] = defaultdict(int)
        for node in self.nodes:
            node_kind_counts[node.node_kind] += 1
        for edge in self.edges:
            edge_kind_counts[edge.edge_kind] += 1
        return {
            "format_version": _FORMAT_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "root": self.root,
            "workstream_root_ids": list(self.workstream_root_ids),
            "counts": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "diagnostic_count": len(self.diagnostics),
                "node_kind_counts": dict(_sorted(list(node_kind_counts.items()))),
                "edge_kind_counts": dict(_sorted(list(edge_kind_counts.items()))),
            },
            "nodes": [node.as_payload() for node in self.nodes],
            "edges": [edge.as_payload() for edge in self.edges],
            "diagnostics": [item.as_payload() for item in self.diagnostics],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> InvariantGraph:
        nodes = tuple(
            InvariantGraphNode(
                node_id=str(item.get("node_id", "")),
                node_kind=str(item.get("node_kind", "")),
                title=str(item.get("title", "")),
                marker_name=str(item.get("marker_name", "")),
                marker_kind=str(item.get("marker_kind", "")),
                marker_id=str(item.get("marker_id", "")),
                site_identity=str(item.get("site_identity", "")),
                structural_identity=str(item.get("structural_identity", "")),
                object_ids=tuple(str(value) for value in item.get("object_ids", [])),
                doc_ids=tuple(str(value) for value in item.get("doc_ids", [])),
                policy_ids=tuple(str(value) for value in item.get("policy_ids", [])),
                invariant_ids=tuple(str(value) for value in item.get("invariant_ids", [])),
                reasoning_summary=str(item.get("reasoning_summary", "")),
                reasoning_control=str(item.get("reasoning_control", "")),
                blocking_dependencies=tuple(
                    str(value) for value in item.get("blocking_dependencies", [])
                ),
                rel_path=str(item.get("rel_path", "")),
                qualname=str(item.get("qualname", "")),
                line=int(item.get("line", 0)),
                column=int(item.get("column", 0)),
                ast_node_kind=str(item.get("ast_node_kind", "")),
                seam_class=str(item.get("seam_class", "")),
                source_marker_node_id=str(item.get("source_marker_node_id", "")),
                status_hint=str(item.get("status_hint", "")),
            )
            for item in payload.get("nodes", [])
            if isinstance(item, Mapping)
        )
        edges = tuple(
            InvariantGraphEdge(
                edge_id=str(item.get("edge_id", "")),
                edge_kind=str(item.get("edge_kind", "")),
                source_id=str(item.get("source_id", "")),
                target_id=str(item.get("target_id", "")),
            )
            for item in payload.get("edges", [])
            if isinstance(item, Mapping)
        )
        diagnostics = tuple(
            InvariantGraphDiagnostic(
                diagnostic_id=str(item.get("diagnostic_id", "")),
                severity=str(item.get("severity", "")),
                code=str(item.get("code", "")),
                node_id=str(item.get("node_id", "")),
                raw_dependency=str(item.get("raw_dependency", "")),
                message=str(item.get("message", "")),
            )
            for item in payload.get("diagnostics", [])
            if isinstance(item, Mapping)
        )
        return cls(
            root=str(payload.get("root", "")),
            workstream_root_ids=tuple(
                str(value) for value in payload.get("workstream_root_ids", [])
            ),
            nodes=nodes,
            edges=edges,
            diagnostics=diagnostics,
        )


@dataclass(frozen=True)
class InvariantTouchsiteProjection:
    object_id: TouchsiteId
    touchpoint_id: TouchpointId
    subqueue_id: SubqueueId
    title: str
    status: str
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
    policy_signal_count: int
    coverage_count: int
    diagnostic_count: int
    object_ids: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "object_id": self.object_id.wire(),
            "touchpoint_id": self.touchpoint_id.wire(),
            "subqueue_id": self.subqueue_id.wire(),
            "title": self.title,
            "status": self.status,
            "rel_path": self.rel_path,
            "qualname": self.qualname,
            "boundary_name": self.boundary_name,
            "line": self.line,
            "column": self.column,
            "node_kind": self.node_kind,
            "site_identity": self.site_identity.wire(),
            "structural_identity": self.structural_identity.wire(),
            "seam_class": self.seam_class,
            "touchpoint_marker_identity": self.touchpoint_marker_identity,
            "touchpoint_structural_identity": self.touchpoint_structural_identity.wire(),
            "subqueue_marker_identity": self.subqueue_marker_identity,
            "subqueue_structural_identity": self.subqueue_structural_identity.wire(),
            "policy_signal_count": self.policy_signal_count,
            "coverage_count": self.coverage_count,
            "diagnostic_count": self.diagnostic_count,
            "object_ids": list(self.object_ids),
        }


@dataclass(frozen=True)
class InvariantCutCandidate:
    cut_kind: str
    object_id: TouchpointId | SubqueueId
    owner_object_id: WorkstreamId | SubqueueId
    title: str
    touchsite_count: int
    collapsible_touchsite_count: int
    surviving_touchsite_count: int
    policy_signal_count: int
    coverage_count: int
    diagnostic_count: int
    covered_touchsite_count: int
    uncovered_touchsite_count: int
    readiness_class: str
    touchsite_ids: tuple[TouchsiteId, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "cut_kind": self.cut_kind,
            "object_id": encode_policy_queue_identity(self.object_id),
            "owner_object_id": encode_policy_queue_identity(self.owner_object_id),
            "title": self.title,
            "touchsite_count": self.touchsite_count,
            "collapsible_touchsite_count": self.collapsible_touchsite_count,
            "surviving_touchsite_count": self.surviving_touchsite_count,
            "policy_signal_count": self.policy_signal_count,
            "coverage_count": self.coverage_count,
            "diagnostic_count": self.diagnostic_count,
            "covered_touchsite_count": self.covered_touchsite_count,
            "uncovered_touchsite_count": self.uncovered_touchsite_count,
            "readiness_class": self.readiness_class,
            "touchsite_ids": [
                encode_policy_queue_identity(item) for item in self.touchsite_ids
            ],
        }


_READINESS_PRIORITY = {
    "ready_structural": 0,
    "coverage_gap": 1,
    "policy_blocked": 2,
    "diagnostic_blocked": 3,
}

_CUT_KIND_PRIORITY = {
    "touchpoint_cut": 0,
    "subqueue_cut": 1,
}


def _cut_readiness_class(
    *,
    policy_signal_count: int,
    diagnostic_count: int,
    uncovered_touchsite_count: int,
) -> str:
    if diagnostic_count > 0:
        return "diagnostic_blocked"
    if policy_signal_count > 0:
        return "policy_blocked"
    if uncovered_touchsite_count > 0:
        return "coverage_gap"
    return "ready_structural"


def _touchsite_blocker_class(touchsite: InvariantTouchsiteProjection) -> str:
    return _cut_readiness_class(
        policy_signal_count=touchsite.policy_signal_count,
        diagnostic_count=touchsite.diagnostic_count,
        uncovered_touchsite_count=1 if touchsite.coverage_count <= 0 else 0,
    )


def _cut_priority_rank(candidate: InvariantCutCandidate) -> int:
    return (
        _READINESS_PRIORITY.get(candidate.readiness_class, 99) * 2
        + _CUT_KIND_PRIORITY.get(candidate.cut_kind, 99)
    )


def _cut_sort_key(candidate: InvariantCutCandidate) -> tuple[int, int, int, int, int, str]:
    return (
        _READINESS_PRIORITY.get(candidate.readiness_class, 99),
        _CUT_KIND_PRIORITY.get(candidate.cut_kind, 99),
        candidate.touchsite_count,
        candidate.surviving_touchsite_count,
        candidate.policy_signal_count,
        candidate.diagnostic_count,
        candidate.uncovered_touchsite_count,
        encode_policy_queue_identity(candidate.object_id),
    )


@dataclass(frozen=True)
class InvariantWorkstreamHealthSummary:
    touchsite_count: int
    covered_touchsite_count: int
    uncovered_touchsite_count: int
    governed_touchsite_count: int
    diagnosed_touchsite_count: int
    ready_touchsite_count: int
    coverage_gap_touchsite_count: int
    policy_blocked_touchsite_count: int
    diagnostic_blocked_touchsite_count: int
    ready_touchpoint_cut_count: int
    coverage_gap_touchpoint_cut_count: int
    policy_blocked_touchpoint_cut_count: int
    diagnostic_blocked_touchpoint_cut_count: int
    ready_subqueue_cut_count: int
    coverage_gap_subqueue_cut_count: int
    policy_blocked_subqueue_cut_count: int
    diagnostic_blocked_subqueue_cut_count: int

    def as_payload(self) -> dict[str, object]:
        return {
            "touchsite_count": self.touchsite_count,
            "covered_touchsite_count": self.covered_touchsite_count,
            "uncovered_touchsite_count": self.uncovered_touchsite_count,
            "governed_touchsite_count": self.governed_touchsite_count,
            "diagnosed_touchsite_count": self.diagnosed_touchsite_count,
            "ready_touchsite_count": self.ready_touchsite_count,
            "coverage_gap_touchsite_count": self.coverage_gap_touchsite_count,
            "policy_blocked_touchsite_count": self.policy_blocked_touchsite_count,
            "diagnostic_blocked_touchsite_count": self.diagnostic_blocked_touchsite_count,
            "ready_touchpoint_cut_count": self.ready_touchpoint_cut_count,
            "coverage_gap_touchpoint_cut_count": self.coverage_gap_touchpoint_cut_count,
            "policy_blocked_touchpoint_cut_count": self.policy_blocked_touchpoint_cut_count,
            "diagnostic_blocked_touchpoint_cut_count": self.diagnostic_blocked_touchpoint_cut_count,
            "ready_subqueue_cut_count": self.ready_subqueue_cut_count,
            "coverage_gap_subqueue_cut_count": self.coverage_gap_subqueue_cut_count,
            "policy_blocked_subqueue_cut_count": self.policy_blocked_subqueue_cut_count,
            "diagnostic_blocked_subqueue_cut_count": self.diagnostic_blocked_subqueue_cut_count,
        }


@dataclass(frozen=True)
class InvariantRemediationLane:
    remediation_family: str
    blocker_class: str
    touchsite_count: int
    touchpoint_cut_count: int
    subqueue_cut_count: int
    best_touchpoint_cut: InvariantCutCandidate | None
    best_subqueue_cut: InvariantCutCandidate | None
    best_cut: InvariantCutCandidate | None

    def as_payload(self) -> dict[str, object]:
        return {
            "remediation_family": self.remediation_family,
            "blocker_class": self.blocker_class,
            "touchsite_count": self.touchsite_count,
            "touchpoint_cut_count": self.touchpoint_cut_count,
            "subqueue_cut_count": self.subqueue_cut_count,
            "best_touchpoint_cut": (
                None
                if self.best_touchpoint_cut is None
                else self.best_touchpoint_cut.as_payload()
            ),
            "best_subqueue_cut": (
                None if self.best_subqueue_cut is None else self.best_subqueue_cut.as_payload()
            ),
            "best_cut": None if self.best_cut is None else self.best_cut.as_payload(),
        }


@dataclass(frozen=True)
class InvariantDocumentationFollowupLane:
    followup_family: str
    alignment_status: str
    target_doc_count: int
    misaligned_target_doc_count: int
    target_doc_ids: tuple[str, ...]
    misaligned_target_doc_ids: tuple[str, ...]
    recommended_action: str
    best_target_doc_id: str | None

    def as_payload(self) -> dict[str, object]:
        return {
            "followup_family": self.followup_family,
            "alignment_status": self.alignment_status,
            "target_doc_count": self.target_doc_count,
            "misaligned_target_doc_count": self.misaligned_target_doc_count,
            "target_doc_ids": list(self.target_doc_ids),
            "misaligned_target_doc_ids": list(self.misaligned_target_doc_ids),
            "recommended_action": self.recommended_action,
            "best_target_doc_id": self.best_target_doc_id,
        }


@dataclass(frozen=True)
class InvariantFollowupAction:
    followup_family: str
    action_kind: str
    priority_rank: int
    object_id: str | None
    owner_object_id: str | None
    target_doc_id: str | None
    title: str
    blocker_class: str | None
    readiness_class: str | None
    alignment_status: str | None
    recommended_action: str | None
    touchsite_count: int
    collapsible_touchsite_count: int
    surviving_touchsite_count: int

    def as_payload(self) -> dict[str, object]:
        return {
            "followup_family": self.followup_family,
            "action_kind": self.action_kind,
            "priority_rank": self.priority_rank,
            "object_id": self.object_id,
            "owner_object_id": self.owner_object_id,
            "target_doc_id": self.target_doc_id,
            "title": self.title,
            "blocker_class": self.blocker_class,
            "readiness_class": self.readiness_class,
            "alignment_status": self.alignment_status,
            "recommended_action": self.recommended_action,
            "touchsite_count": self.touchsite_count,
            "collapsible_touchsite_count": self.collapsible_touchsite_count,
            "surviving_touchsite_count": self.surviving_touchsite_count,
        }


@dataclass(frozen=True)
class InvariantDiagnosticBucket:
    code: str
    severity: str
    count: int

    def as_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "count": self.count,
        }


@dataclass(frozen=True)
class InvariantDiagnosticSummary:
    diagnostic_count: int
    unmatched_policy_signal_count: int
    unresolved_blocking_dependency_count: int
    buckets: tuple[InvariantDiagnosticBucket, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "diagnostic_count": self.diagnostic_count,
            "unmatched_policy_signal_count": self.unmatched_policy_signal_count,
            "unresolved_blocking_dependency_count": self.unresolved_blocking_dependency_count,
            "buckets": [item.as_payload() for item in self.buckets],
        }


@dataclass(frozen=True)
class InvariantScoreComponent:
    kind: str
    score: int
    rationale: str

    def as_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "score": self.score,
            "rationale": self.rationale,
        }


def _margin_strength(score: int | None) -> str:
    if score is None:
        return "none"
    if score <= 0:
        return "tied"
    if score < 25:
        return "weak"
    if score < 100:
        return "moderate"
    return "strong"


@dataclass(frozen=True)
class InvariantRepoFollowupCohortMember:
    followup_family: str
    followup_class: str
    action_kind: str
    object_id: str | None
    diagnostic_code: str | None
    target_doc_id: str | None
    policy_ids: tuple[str, ...]
    title: str
    utility_score: int
    selection_rank: int
    selection_reason: str

    def as_payload(self) -> dict[str, object]:
        return {
            "followup_family": self.followup_family,
            "followup_class": self.followup_class,
            "action_kind": self.action_kind,
            "object_id": self.object_id,
            "diagnostic_code": self.diagnostic_code,
            "target_doc_id": self.target_doc_id,
            "policy_ids": list(self.policy_ids),
            "title": self.title,
            "utility_score": self.utility_score,
            "selection_rank": self.selection_rank,
            "selection_reason": self.selection_reason,
        }


@dataclass(frozen=True)
class InvariantRepoFollowupAction:
    followup_family: str
    action_kind: str
    priority_rank: int
    object_id: str | None
    owner_object_id: str | None
    diagnostic_code: str | None
    target_doc_id: str | None
    policy_ids: tuple[str, ...]
    title: str
    blocker_class: str | None
    readiness_class: str | None
    alignment_status: str | None
    recommended_action: str | None
    owner_seed_path: str | None
    owner_seed_object_id: str | None
    owner_resolution_kind: str | None
    owner_resolution_score: int | None
    owner_resolution_options: tuple["InvariantOwnerCandidateOption", ...]
    runner_up_owner_object_id: str | None
    runner_up_owner_resolution_kind: str | None
    runner_up_owner_resolution_score: int | None
    owner_choice_margin_score: int | None
    owner_choice_margin_reason: str | None
    owner_choice_margin_components: tuple[InvariantScoreComponent, ...]
    owner_option_tradeoff_score: int | None
    owner_option_tradeoff_reason: str | None
    owner_option_tradeoff_components: tuple[InvariantScoreComponent, ...]
    utility_score: int
    utility_reason: str
    utility_components: tuple[InvariantScoreComponent, ...]
    selection_certainty_kind: str
    cofrontier_followup_count: int
    cofrontier_followup_cohort: tuple[InvariantRepoFollowupCohortMember, ...]
    selection_scope_kind: str
    selection_scope_id: str | None
    runner_up_followup_family: str | None
    runner_up_followup_class: str | None
    runner_up_followup_object_id: str | None
    runner_up_followup_utility_score: int | None
    frontier_choice_margin_score: int | None
    frontier_choice_margin_reason: str | None
    frontier_choice_margin_components: tuple[InvariantScoreComponent, ...]
    selection_rank: int
    opportunity_cost_score: int
    opportunity_cost_reason: str
    opportunity_cost_components: tuple[InvariantScoreComponent, ...]
    count: int

    def as_payload(self) -> dict[str, object]:
        return {
            "followup_family": self.followup_family,
            "action_kind": self.action_kind,
            "priority_rank": self.priority_rank,
            "object_id": self.object_id,
            "owner_object_id": self.owner_object_id,
            "diagnostic_code": self.diagnostic_code,
            "target_doc_id": self.target_doc_id,
            "policy_ids": list(self.policy_ids),
            "title": self.title,
            "blocker_class": self.blocker_class,
            "readiness_class": self.readiness_class,
            "alignment_status": self.alignment_status,
            "recommended_action": self.recommended_action,
            "owner_seed_path": self.owner_seed_path,
            "owner_seed_object_id": self.owner_seed_object_id,
            "owner_resolution_kind": self.owner_resolution_kind,
            "owner_resolution_score": self.owner_resolution_score,
            "owner_resolution_options": [
                item.as_payload() for item in self.owner_resolution_options
            ],
            "runner_up_owner_object_id": self.runner_up_owner_object_id,
            "runner_up_owner_resolution_kind": self.runner_up_owner_resolution_kind,
            "runner_up_owner_resolution_score": self.runner_up_owner_resolution_score,
            "owner_choice_margin_score": self.owner_choice_margin_score,
            "owner_choice_margin_reason": self.owner_choice_margin_reason,
            "owner_choice_margin_components": [
                item.as_payload() for item in self.owner_choice_margin_components
            ],
            "owner_option_tradeoff_score": self.owner_option_tradeoff_score,
            "owner_option_tradeoff_reason": self.owner_option_tradeoff_reason,
            "owner_option_tradeoff_components": [
                item.as_payload() for item in self.owner_option_tradeoff_components
            ],
            "utility_score": self.utility_score,
            "utility_reason": self.utility_reason,
            "utility_components": [
                item.as_payload() for item in self.utility_components
            ],
            "selection_certainty_kind": self.selection_certainty_kind,
            "cofrontier_followup_count": self.cofrontier_followup_count,
            "cofrontier_followup_cohort": [
                item.as_payload() for item in self.cofrontier_followup_cohort
            ],
            "selection_scope_kind": self.selection_scope_kind,
            "selection_scope_id": self.selection_scope_id,
            "runner_up_followup_family": self.runner_up_followup_family,
            "runner_up_followup_class": self.runner_up_followup_class,
            "runner_up_followup_object_id": self.runner_up_followup_object_id,
            "runner_up_followup_utility_score": self.runner_up_followup_utility_score,
            "frontier_choice_margin_score": self.frontier_choice_margin_score,
            "frontier_choice_margin_reason": self.frontier_choice_margin_reason,
            "frontier_choice_margin_components": [
                item.as_payload() for item in self.frontier_choice_margin_components
            ],
            "selection_rank": self.selection_rank,
            "opportunity_cost_score": self.opportunity_cost_score,
            "opportunity_cost_reason": self.opportunity_cost_reason,
            "opportunity_cost_components": [
                item.as_payload() for item in self.opportunity_cost_components
            ],
            "count": self.count,
        }


@dataclass(frozen=True)
class InvariantRepoFollowupLane:
    followup_family: str
    followup_class: str
    action_count: int
    strongest_owner_resolution_kind: str | None
    strongest_owner_resolution_score: int | None
    strongest_utility_score: int
    strongest_utility_reason: str
    lane_utility_score: int
    lane_utility_reason: str
    lane_utility_components: tuple[InvariantScoreComponent, ...]
    selection_rank: int
    opportunity_cost_score: int
    opportunity_cost_reason: str
    opportunity_cost_components: tuple[InvariantScoreComponent, ...]
    best_followup: InvariantRepoFollowupAction

    def as_payload(self) -> dict[str, object]:
        return {
            "followup_family": self.followup_family,
            "followup_class": self.followup_class,
            "action_count": self.action_count,
            "strongest_owner_resolution_kind": self.strongest_owner_resolution_kind,
            "strongest_owner_resolution_score": self.strongest_owner_resolution_score,
            "strongest_utility_score": self.strongest_utility_score,
            "strongest_utility_reason": self.strongest_utility_reason,
            "lane_utility_score": self.lane_utility_score,
            "lane_utility_reason": self.lane_utility_reason,
            "lane_utility_components": [
                item.as_payload() for item in self.lane_utility_components
            ],
            "selection_rank": self.selection_rank,
            "opportunity_cost_score": self.opportunity_cost_score,
            "opportunity_cost_reason": self.opportunity_cost_reason,
            "opportunity_cost_components": [
                item.as_payload() for item in self.opportunity_cost_components
            ],
            "best_followup": self.best_followup.as_payload(),
        }


@dataclass(frozen=True)
class InvariantRepoFrontierTradeoff:
    frontier_followup_family: str
    frontier_followup_class: str
    runner_up_followup_family: str
    runner_up_followup_class: str
    frontier_lane_utility_score: int
    frontier_lane_utility_reason: str
    runner_up_lane_utility_score: int
    runner_up_lane_utility_reason: str
    margin_score: int
    margin_reason: str
    margin_components: tuple[InvariantScoreComponent, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "frontier_followup_family": self.frontier_followup_family,
            "frontier_followup_class": self.frontier_followup_class,
            "runner_up_followup_family": self.runner_up_followup_family,
            "runner_up_followup_class": self.runner_up_followup_class,
            "frontier_lane_utility_score": self.frontier_lane_utility_score,
            "frontier_lane_utility_reason": self.frontier_lane_utility_reason,
            "runner_up_lane_utility_score": self.runner_up_lane_utility_score,
            "runner_up_lane_utility_reason": self.runner_up_lane_utility_reason,
            "margin_score": self.margin_score,
            "margin_reason": self.margin_reason,
            "margin_components": [
                item.as_payload() for item in self.margin_components
            ],
        }


@dataclass(frozen=True)
class InvariantRepoFollowupCrossClassTradeoff:
    frontier_followup_family: str
    frontier_followup_class: str
    frontier_action_kind: str
    frontier_object_id: str | None
    frontier_diagnostic_code: str | None
    frontier_target_doc_id: str | None
    frontier_utility_score: int
    frontier_utility_reason: str
    runner_up_followup_family: str
    runner_up_followup_class: str
    runner_up_action_kind: str
    runner_up_object_id: str | None
    runner_up_diagnostic_code: str | None
    runner_up_target_doc_id: str | None
    runner_up_utility_score: int
    runner_up_utility_reason: str
    margin_score: int
    margin_reason: str
    margin_components: tuple[InvariantScoreComponent, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "frontier_followup_family": self.frontier_followup_family,
            "frontier_followup_class": self.frontier_followup_class,
            "frontier_action_kind": self.frontier_action_kind,
            "frontier_object_id": self.frontier_object_id,
            "frontier_diagnostic_code": self.frontier_diagnostic_code,
            "frontier_target_doc_id": self.frontier_target_doc_id,
            "frontier_utility_score": self.frontier_utility_score,
            "frontier_utility_reason": self.frontier_utility_reason,
            "runner_up_followup_family": self.runner_up_followup_family,
            "runner_up_followup_class": self.runner_up_followup_class,
            "runner_up_action_kind": self.runner_up_action_kind,
            "runner_up_object_id": self.runner_up_object_id,
            "runner_up_diagnostic_code": self.runner_up_diagnostic_code,
            "runner_up_target_doc_id": self.runner_up_target_doc_id,
            "runner_up_utility_score": self.runner_up_utility_score,
            "runner_up_utility_reason": self.runner_up_utility_reason,
            "margin_score": self.margin_score,
            "margin_reason": self.margin_reason,
            "margin_components": [
                item.as_payload() for item in self.margin_components
            ],
        }


@dataclass(frozen=True)
class InvariantRepoFollowupSameClassTradeoff:
    frontier_followup_family: str
    frontier_followup_class: str
    frontier_action_kind: str
    frontier_object_id: str | None
    frontier_diagnostic_code: str | None
    frontier_target_doc_id: str | None
    frontier_policy_ids: tuple[str, ...]
    frontier_utility_score: int
    frontier_utility_reason: str
    runner_up_followup_family: str
    runner_up_followup_class: str
    runner_up_action_kind: str
    runner_up_object_id: str | None
    runner_up_diagnostic_code: str | None
    runner_up_target_doc_id: str | None
    runner_up_policy_ids: tuple[str, ...]
    runner_up_utility_score: int
    runner_up_utility_reason: str
    margin_score: int
    margin_reason: str
    margin_components: tuple[InvariantScoreComponent, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "frontier_followup_family": self.frontier_followup_family,
            "frontier_followup_class": self.frontier_followup_class,
            "frontier_action_kind": self.frontier_action_kind,
            "frontier_object_id": self.frontier_object_id,
            "frontier_diagnostic_code": self.frontier_diagnostic_code,
            "frontier_target_doc_id": self.frontier_target_doc_id,
            "frontier_policy_ids": list(self.frontier_policy_ids),
            "frontier_utility_score": self.frontier_utility_score,
            "frontier_utility_reason": self.frontier_utility_reason,
            "runner_up_followup_family": self.runner_up_followup_family,
            "runner_up_followup_class": self.runner_up_followup_class,
            "runner_up_action_kind": self.runner_up_action_kind,
            "runner_up_object_id": self.runner_up_object_id,
            "runner_up_diagnostic_code": self.runner_up_diagnostic_code,
            "runner_up_target_doc_id": self.runner_up_target_doc_id,
            "runner_up_policy_ids": list(self.runner_up_policy_ids),
            "runner_up_utility_score": self.runner_up_utility_score,
            "runner_up_utility_reason": self.runner_up_utility_reason,
            "margin_score": self.margin_score,
            "margin_reason": self.margin_reason,
            "margin_components": [
                item.as_payload() for item in self.margin_components
            ],
        }


@dataclass(frozen=True)
class InvariantRepoDiagnosticLane:
    diagnostic_code: str
    severity: str
    title: str
    recommended_action: str
    count: int
    node_ids: tuple[str, ...]
    policy_ids: tuple[str, ...]
    rel_path: str
    qualname: str
    line: int
    column: int
    candidate_owner_status: str
    candidate_owner_object_id: str | None
    candidate_owner_object_ids: tuple[str, ...]
    candidate_owner_seed_path: str | None
    candidate_owner_seed_object_id: str | None
    candidate_owner_options: tuple["InvariantOwnerCandidateOption", ...]
    runner_up_candidate_owner_option: "InvariantOwnerCandidateOption" | None
    candidate_owner_choice_margin_score: int | None
    candidate_owner_choice_margin_reason: str | None
    candidate_owner_choice_margin_components: tuple[InvariantScoreComponent, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "diagnostic_code": self.diagnostic_code,
            "severity": self.severity,
            "title": self.title,
            "recommended_action": self.recommended_action,
            "count": self.count,
            "node_ids": list(self.node_ids),
            "policy_ids": list(self.policy_ids),
            "rel_path": self.rel_path,
            "qualname": self.qualname,
            "line": self.line,
            "column": self.column,
            "candidate_owner_status": self.candidate_owner_status,
            "candidate_owner_object_id": self.candidate_owner_object_id,
            "candidate_owner_object_ids": list(self.candidate_owner_object_ids),
            "candidate_owner_seed_path": self.candidate_owner_seed_path,
            "candidate_owner_seed_object_id": self.candidate_owner_seed_object_id,
            "candidate_owner_options": [
                item.as_payload() for item in self.candidate_owner_options
            ],
            "runner_up_candidate_owner_option": (
                None
                if self.runner_up_candidate_owner_option is None
                else self.runner_up_candidate_owner_option.as_payload()
            ),
            "candidate_owner_choice_margin_score": self.candidate_owner_choice_margin_score,
            "candidate_owner_choice_margin_reason": self.candidate_owner_choice_margin_reason,
            "candidate_owner_choice_margin_components": [
                item.as_payload() for item in self.candidate_owner_choice_margin_components
            ],
        }


@dataclass(frozen=True)
class InvariantRepoFollowupFrontierTriad:
    frontier_followup_family: str
    frontier_followup_class: str
    frontier_action_kind: str
    frontier_object_id: str | None
    frontier_diagnostic_code: str | None
    frontier_target_doc_id: str | None
    frontier_policy_ids: tuple[str, ...]
    frontier_utility_score: int
    frontier_utility_reason: str
    same_class_tradeoff: InvariantRepoFollowupSameClassTradeoff | None
    cross_class_tradeoff: InvariantRepoFollowupCrossClassTradeoff | None

    def as_payload(self) -> dict[str, object]:
        return {
            "frontier_followup_family": self.frontier_followup_family,
            "frontier_followup_class": self.frontier_followup_class,
            "frontier_action_kind": self.frontier_action_kind,
            "frontier_object_id": self.frontier_object_id,
            "frontier_diagnostic_code": self.frontier_diagnostic_code,
            "frontier_target_doc_id": self.frontier_target_doc_id,
            "frontier_policy_ids": list(self.frontier_policy_ids),
            "frontier_utility_score": self.frontier_utility_score,
            "frontier_utility_reason": self.frontier_utility_reason,
            "same_class_tradeoff": (
                None if self.same_class_tradeoff is None else self.same_class_tradeoff.as_payload()
            ),
            "cross_class_tradeoff": (
                None
                if self.cross_class_tradeoff is None
                else self.cross_class_tradeoff.as_payload()
            ),
        }


@dataclass(frozen=True)
class InvariantRepoFollowupFrontierExplanation:
    frontier_followup_family: str
    frontier_followup_class: str
    frontier_action_kind: str
    frontier_object_id: str | None
    frontier_diagnostic_code: str | None
    frontier_target_doc_id: str | None
    frontier_policy_ids: tuple[str, ...]
    frontier_utility_score: int
    frontier_utility_reason: str
    same_class_runner_up_followup_family: str | None
    same_class_runner_up_followup_class: str | None
    same_class_runner_up_action_kind: str | None
    same_class_runner_up_object_id: str | None
    same_class_runner_up_diagnostic_code: str | None
    same_class_runner_up_target_doc_id: str | None
    same_class_runner_up_policy_ids: tuple[str, ...]
    same_class_runner_up_utility_score: int | None
    same_class_runner_up_utility_reason: str | None
    same_class_margin_score: int | None
    same_class_margin_reason: str | None
    same_class_margin_components: tuple[InvariantScoreComponent, ...]
    cross_class_runner_up_followup_family: str | None
    cross_class_runner_up_followup_class: str | None
    cross_class_runner_up_action_kind: str | None
    cross_class_runner_up_object_id: str | None
    cross_class_runner_up_diagnostic_code: str | None
    cross_class_runner_up_target_doc_id: str | None
    cross_class_runner_up_utility_score: int | None
    cross_class_runner_up_utility_reason: str | None
    cross_class_margin_score: int | None
    cross_class_margin_reason: str | None
    cross_class_margin_components: tuple[InvariantScoreComponent, ...]
    recommendation_rationale_kind: str
    recommendation_rationale_reason: str
    recommendation_rationale_components: tuple[InvariantScoreComponent, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "frontier_followup_family": self.frontier_followup_family,
            "frontier_followup_class": self.frontier_followup_class,
            "frontier_action_kind": self.frontier_action_kind,
            "frontier_object_id": self.frontier_object_id,
            "frontier_diagnostic_code": self.frontier_diagnostic_code,
            "frontier_target_doc_id": self.frontier_target_doc_id,
            "frontier_policy_ids": list(self.frontier_policy_ids),
            "frontier_utility_score": self.frontier_utility_score,
            "frontier_utility_reason": self.frontier_utility_reason,
            "same_class_runner_up_followup_family": self.same_class_runner_up_followup_family,
            "same_class_runner_up_followup_class": self.same_class_runner_up_followup_class,
            "same_class_runner_up_action_kind": self.same_class_runner_up_action_kind,
            "same_class_runner_up_object_id": self.same_class_runner_up_object_id,
            "same_class_runner_up_diagnostic_code": self.same_class_runner_up_diagnostic_code,
            "same_class_runner_up_target_doc_id": self.same_class_runner_up_target_doc_id,
            "same_class_runner_up_policy_ids": list(self.same_class_runner_up_policy_ids),
            "same_class_runner_up_utility_score": self.same_class_runner_up_utility_score,
            "same_class_runner_up_utility_reason": self.same_class_runner_up_utility_reason,
            "same_class_margin_score": self.same_class_margin_score,
            "same_class_margin_reason": self.same_class_margin_reason,
            "same_class_margin_components": [
                item.as_payload() for item in self.same_class_margin_components
            ],
            "cross_class_runner_up_followup_family": self.cross_class_runner_up_followup_family,
            "cross_class_runner_up_followup_class": self.cross_class_runner_up_followup_class,
            "cross_class_runner_up_action_kind": self.cross_class_runner_up_action_kind,
            "cross_class_runner_up_object_id": self.cross_class_runner_up_object_id,
            "cross_class_runner_up_diagnostic_code": self.cross_class_runner_up_diagnostic_code,
            "cross_class_runner_up_target_doc_id": self.cross_class_runner_up_target_doc_id,
            "cross_class_runner_up_utility_score": self.cross_class_runner_up_utility_score,
            "cross_class_runner_up_utility_reason": self.cross_class_runner_up_utility_reason,
            "cross_class_margin_score": self.cross_class_margin_score,
            "cross_class_margin_reason": self.cross_class_margin_reason,
            "cross_class_margin_components": [
                item.as_payload() for item in self.cross_class_margin_components
            ],
            "recommendation_rationale_kind": self.recommendation_rationale_kind,
            "recommendation_rationale_reason": self.recommendation_rationale_reason,
            "recommendation_rationale_components": [
                item.as_payload() for item in self.recommendation_rationale_components
            ],
        }


@dataclass(frozen=True)
class InvariantOwnerCandidateOption:
    resolution_kind: str
    owner_status: str
    object_id: str
    score: int
    rationale: str
    score_components: tuple[InvariantScoreComponent, ...]
    selection_rank: int = 0
    opportunity_cost_score: int = 0
    opportunity_cost_reason: str = "frontier"
    opportunity_cost_components: tuple[InvariantScoreComponent, ...] = ()

    def as_payload(self) -> dict[str, object]:
        return {
            "resolution_kind": self.resolution_kind,
            "owner_status": self.owner_status,
            "object_id": self.object_id,
            "score": self.score,
            "rationale": self.rationale,
            "score_components": [
                item.as_payload() for item in self.score_components
            ],
            "selection_rank": self.selection_rank,
            "opportunity_cost_score": self.opportunity_cost_score,
            "opportunity_cost_reason": self.opportunity_cost_reason,
            "opportunity_cost_components": [
                item.as_payload() for item in self.opportunity_cost_components
            ],
        }


@dataclass(frozen=True)
class InvariantTouchpointProjection:
    object_id: TouchpointId
    subqueue_id: SubqueueId
    title: str
    status: str
    rel_path: str
    site_identity: SiteReferenceId
    structural_identity: StructuralReferenceId
    marker_identity: str
    reasoning_summary: str
    reasoning_control: str
    blocking_dependencies: tuple[str, ...]
    object_ids: tuple[str, ...]
    touchsite_count: int
    collapsible_touchsite_count: int
    surviving_touchsite_count: int
    policy_signal_count: int
    coverage_count: int
    diagnostic_count: int
    touchsites: ReplayableStream[InvariantTouchsiteProjection]

    def iter_touchsites(self) -> Iterator[InvariantTouchsiteProjection]:
        return iter(self.touchsites)

    def as_payload(self) -> dict[str, object]:
        return {
            "object_id": self.object_id.wire(),
            "subqueue_id": self.subqueue_id.wire(),
            "title": self.title,
            "status": self.status,
            "rel_path": self.rel_path,
            "site_identity": self.site_identity.wire(),
            "structural_identity": self.structural_identity.wire(),
            "marker_identity": self.marker_identity,
            "reasoning_summary": self.reasoning_summary,
            "reasoning_control": self.reasoning_control,
            "blocking_dependencies": list(self.blocking_dependencies),
            "object_ids": list(self.object_ids),
            "touchsite_count": self.touchsite_count,
            "collapsible_touchsite_count": self.collapsible_touchsite_count,
            "surviving_touchsite_count": self.surviving_touchsite_count,
            "policy_signal_count": self.policy_signal_count,
            "coverage_count": self.coverage_count,
            "diagnostic_count": self.diagnostic_count,
            "touchsites": [item.as_payload() for item in self.iter_touchsites()],
        }


@dataclass(frozen=True)
class InvariantSubqueueProjection:
    object_id: SubqueueId
    title: str
    status: str
    site_identity: SiteReferenceId
    structural_identity: StructuralReferenceId
    marker_identity: str
    reasoning_summary: str
    reasoning_control: str
    blocking_dependencies: tuple[str, ...]
    object_ids: tuple[str, ...]
    touchpoint_ids: tuple[TouchpointId, ...]
    touchsite_count: int
    collapsible_touchsite_count: int
    surviving_touchsite_count: int
    policy_signal_count: int
    coverage_count: int
    diagnostic_count: int

    def as_payload(self) -> dict[str, object]:
        return {
            "object_id": self.object_id.wire(),
            "title": self.title,
            "status": self.status,
            "site_identity": self.site_identity.wire(),
            "structural_identity": self.structural_identity.wire(),
            "marker_identity": self.marker_identity,
            "reasoning_summary": self.reasoning_summary,
            "reasoning_control": self.reasoning_control,
            "blocking_dependencies": list(self.blocking_dependencies),
            "object_ids": list(self.object_ids),
            "touchpoint_ids": [item.wire() for item in self.touchpoint_ids],
            "touchsite_count": self.touchsite_count,
            "collapsible_touchsite_count": self.collapsible_touchsite_count,
            "surviving_touchsite_count": self.surviving_touchsite_count,
            "policy_signal_count": self.policy_signal_count,
            "coverage_count": self.coverage_count,
            "diagnostic_count": self.diagnostic_count,
        }


@dataclass(frozen=True)
class InvariantWorkstreamProjection:
    object_id: WorkstreamId
    title: str
    status: str
    site_identity: SiteReferenceId
    structural_identity: StructuralReferenceId
    marker_identity: str
    reasoning_summary: str
    reasoning_control: str
    blocking_dependencies: tuple[str, ...]
    object_ids: tuple[str, ...]
    doc_ids: tuple[str, ...]
    policy_ids: tuple[str, ...]
    touchsite_count: int
    collapsible_touchsite_count: int
    surviving_touchsite_count: int
    policy_signal_count: int
    coverage_count: int
    diagnostic_count: int
    subqueues: ReplayableStream[InvariantSubqueueProjection]
    touchpoints: ReplayableStream[InvariantTouchpointProjection]
    doc_alignment_summary: InvariantLedgerAlignmentSummary | None = None

    def iter_subqueues(self) -> Iterator[InvariantSubqueueProjection]:
        return iter(self.subqueues)

    def iter_touchpoints(self) -> Iterator[InvariantTouchpointProjection]:
        return iter(self.touchpoints)

    @cached_property
    def _subqueue_cache(self) -> tuple[InvariantSubqueueProjection, ...]:
        return tuple(self.iter_subqueues())

    @cached_property
    def _touchpoint_cache(self) -> tuple[InvariantTouchpointProjection, ...]:
        return tuple(self.iter_touchpoints())

    @cached_property
    def _touchsite_cache(self) -> tuple[InvariantTouchsiteProjection, ...]:
        return tuple(
            touchsite
            for touchpoint in self._touchpoint_cache
            for touchsite in touchpoint.iter_touchsites()
        )

    def ranked_touchpoint_cuts(self) -> tuple[InvariantCutCandidate, ...]:
        return self._ranked_touchpoint_cuts

    @cached_property
    def _ranked_touchpoint_cuts(self) -> tuple[InvariantCutCandidate, ...]:
        candidates = [
            InvariantCutCandidate(
                cut_kind="touchpoint_cut",
                object_id=touchpoint.object_id,
                owner_object_id=touchpoint.subqueue_id,
                title=touchpoint.title,
                touchsite_count=touchpoint.touchsite_count,
                collapsible_touchsite_count=touchpoint.collapsible_touchsite_count,
                surviving_touchsite_count=touchpoint.surviving_touchsite_count,
                policy_signal_count=touchpoint.policy_signal_count,
                coverage_count=touchpoint.coverage_count,
                diagnostic_count=touchpoint.diagnostic_count,
                covered_touchsite_count=sum(
                    1 for touchsite in touchpoint.iter_touchsites() if touchsite.coverage_count > 0
                ),
                uncovered_touchsite_count=sum(
                    1 for touchsite in touchpoint.iter_touchsites() if touchsite.coverage_count <= 0
                ),
                readiness_class=_cut_readiness_class(
                    policy_signal_count=touchpoint.policy_signal_count,
                    diagnostic_count=touchpoint.diagnostic_count,
                    uncovered_touchsite_count=sum(
                        1
                        for touchsite in touchpoint.iter_touchsites()
                        if touchsite.coverage_count <= 0
                    ),
                ),
                touchsite_ids=tuple(
                    touchsite.object_id for touchsite in touchpoint.iter_touchsites()
                ),
            )
            for touchpoint in self._touchpoint_cache
            if touchpoint.touchsite_count > 0
        ]
        return tuple(_sorted(candidates, key=_cut_sort_key))

    def ranked_subqueue_cuts(self) -> tuple[InvariantCutCandidate, ...]:
        return self._ranked_subqueue_cuts

    @cached_property
    def _ranked_subqueue_cuts(self) -> tuple[InvariantCutCandidate, ...]:
        touchpoint_groups: defaultdict[str, list[InvariantTouchpointProjection]] = defaultdict(list)
        for touchpoint in self._touchpoint_cache:
            touchpoint_groups[touchpoint.subqueue_id.wire()].append(touchpoint)
        candidates = []
        for subqueue in self._subqueue_cache:
            if subqueue.touchsite_count <= 0:
                continue
            touchsites = tuple(
                touchsite
                for touchpoint in _sorted(
                    touchpoint_groups.get(subqueue.object_id.wire(), []),
                    key=lambda item: item.object_id.wire(),
                )
                for touchsite in touchpoint.iter_touchsites()
            )
            touchsite_ids = tuple(
                touchsite.object_id for touchsite in touchsites
            )
            uncovered_touchsite_count = sum(
                1 for touchsite in touchsites if touchsite.coverage_count <= 0
            )
            candidates.append(
                InvariantCutCandidate(
                    cut_kind="subqueue_cut",
                    object_id=subqueue.object_id,
                    owner_object_id=self.object_id,
                    title=subqueue.title,
                    touchsite_count=subqueue.touchsite_count,
                    collapsible_touchsite_count=subqueue.collapsible_touchsite_count,
                    surviving_touchsite_count=subqueue.surviving_touchsite_count,
                    policy_signal_count=subqueue.policy_signal_count,
                    coverage_count=subqueue.coverage_count,
                    diagnostic_count=subqueue.diagnostic_count,
                    covered_touchsite_count=len(touchsites) - uncovered_touchsite_count,
                    uncovered_touchsite_count=uncovered_touchsite_count,
                    readiness_class=_cut_readiness_class(
                        policy_signal_count=subqueue.policy_signal_count,
                        diagnostic_count=subqueue.diagnostic_count,
                        uncovered_touchsite_count=uncovered_touchsite_count,
                    ),
                    touchsite_ids=touchsite_ids,
                )
            )
        return tuple(_sorted(candidates, key=_cut_sort_key))

    def _recommended_cut_for_readiness(
        self,
        readiness_class: str,
    ) -> InvariantCutCandidate | None:
        candidates = [
            candidate
            for candidate in (
                *self.ranked_touchpoint_cuts(),
                *self.ranked_subqueue_cuts(),
            )
            if candidate.readiness_class == readiness_class
        ]
        if not candidates:
            return None
        return _sorted(candidates, key=_cut_sort_key)[0]

    def recommended_ready_cut(self) -> InvariantCutCandidate | None:
        return self._recommended_cut_for_readiness("ready_structural")

    def recommended_coverage_gap_cut(self) -> InvariantCutCandidate | None:
        return self._recommended_cut_for_readiness("coverage_gap")

    def recommended_policy_blocked_cut(self) -> InvariantCutCandidate | None:
        return self._recommended_cut_for_readiness("policy_blocked")

    def recommended_diagnostic_blocked_cut(self) -> InvariantCutCandidate | None:
        return self._recommended_cut_for_readiness("diagnostic_blocked")

    def recommended_cut(self) -> InvariantCutCandidate | None:
        candidates: list[InvariantCutCandidate] = []
        touchpoint_cuts = self.ranked_touchpoint_cuts()
        subqueue_cuts = self.ranked_subqueue_cuts()
        if touchpoint_cuts:
            candidates.append(touchpoint_cuts[0])
        if subqueue_cuts:
            candidates.append(subqueue_cuts[0])
        if not candidates:
            return None
        return _sorted(candidates, key=_cut_sort_key)[0]

    def dominant_doc_alignment_status(self) -> str:
        if self.doc_alignment_summary is None:
            return "none"
        return self.doc_alignment_summary.dominant_alignment_status

    def recommended_doc_alignment_action(self) -> str:
        if self.doc_alignment_summary is None:
            return "none"
        return self.doc_alignment_summary.recommended_doc_alignment_action

    def misaligned_target_doc_ids(self) -> tuple[str, ...]:
        if self.doc_alignment_summary is None:
            return ()
        return self.doc_alignment_summary.misaligned_target_doc_ids

    def next_human_followup_family(self) -> str:
        if self.documentation_followup_lane() is None:
            return "none"
        return "documentation_alignment"

    def recommended_doc_followup_target_doc_id(self) -> str | None:
        lane = self.documentation_followup_lane()
        if lane is None:
            return None
        return lane.best_target_doc_id

    def _documentation_followup_priority_rank(
        self,
        lane: InvariantDocumentationFollowupLane,
    ) -> int:
        priority = {
            "missing_target_doc": 20,
            "ambiguous_target_doc": 21,
            "unassigned_target_doc": 22,
            "append_pending_new_object": 23,
            "append_pending_existing_object": 24,
        }
        return priority.get(lane.alignment_status, 49)

    def ranked_followups(self) -> tuple[InvariantFollowupAction, ...]:
        return self._ranked_followups

    @cached_property
    def _ranked_followups(self) -> tuple[InvariantFollowupAction, ...]:
        actions: list[InvariantFollowupAction] = []
        for lane in self.remediation_lanes():
            if lane.best_cut is None:
                continue
            actions.append(
                InvariantFollowupAction(
                    followup_family=lane.remediation_family,
                    action_kind=lane.best_cut.cut_kind,
                    priority_rank=_cut_priority_rank(lane.best_cut),
                    object_id=lane.best_cut.object_id.wire(),
                    owner_object_id=lane.best_cut.owner_object_id.wire(),
                    target_doc_id=None,
                    title=lane.best_cut.title,
                    blocker_class=lane.blocker_class,
                    readiness_class=lane.best_cut.readiness_class,
                    alignment_status=None,
                    recommended_action=None,
                    touchsite_count=lane.best_cut.touchsite_count,
                    collapsible_touchsite_count=lane.best_cut.collapsible_touchsite_count,
                    surviving_touchsite_count=lane.best_cut.surviving_touchsite_count,
                )
            )
        documentation_followup_lane = self.documentation_followup_lane()
        if documentation_followup_lane is not None:
            actions.append(
                InvariantFollowupAction(
                    followup_family=documentation_followup_lane.followup_family,
                    action_kind="doc_alignment",
                    priority_rank=self._documentation_followup_priority_rank(
                        documentation_followup_lane
                    ),
                    object_id=None,
                    owner_object_id=self.object_id.wire(),
                    target_doc_id=documentation_followup_lane.best_target_doc_id,
                    title=documentation_followup_lane.best_target_doc_id or "<none>",
                    blocker_class=None,
                    readiness_class=None,
                    alignment_status=documentation_followup_lane.alignment_status,
                    recommended_action=documentation_followup_lane.recommended_action,
                    touchsite_count=self.touchsite_count,
                    collapsible_touchsite_count=self.collapsible_touchsite_count,
                    surviving_touchsite_count=self.surviving_touchsite_count,
                )
            )
        return tuple(
            _sorted(
                actions,
                key=lambda item: (
                    item.priority_rank,
                    item.touchsite_count,
                    item.surviving_touchsite_count,
                    item.followup_family,
                    item.object_id or "",
                    item.target_doc_id or "",
                ),
            )
        )

    def recommended_followup(self) -> InvariantFollowupAction | None:
        ranked_followups = self.ranked_followups()
        if not ranked_followups:
            return None
        return ranked_followups[0]

    def health_summary(self) -> InvariantWorkstreamHealthSummary:
        return self._health_summary

    @cached_property
    def _health_summary(self) -> InvariantWorkstreamHealthSummary:
        touchsites = self._touchsite_cache
        touchsite_blocker_classes = tuple(
            _touchsite_blocker_class(touchsite) for touchsite in touchsites
        )
        touchpoint_cuts = self.ranked_touchpoint_cuts()
        subqueue_cuts = self.ranked_subqueue_cuts()
        return InvariantWorkstreamHealthSummary(
            touchsite_count=len(touchsites),
            covered_touchsite_count=sum(
                1 for touchsite in touchsites if touchsite.coverage_count > 0
            ),
            uncovered_touchsite_count=sum(
                1 for touchsite in touchsites if touchsite.coverage_count <= 0
            ),
            governed_touchsite_count=sum(
                1 for touchsite in touchsites if touchsite.policy_signal_count > 0
            ),
            diagnosed_touchsite_count=sum(
                1 for touchsite in touchsites if touchsite.diagnostic_count > 0
            ),
            ready_touchsite_count=sum(
                1
                for blocker_class in touchsite_blocker_classes
                if blocker_class == "ready_structural"
            ),
            coverage_gap_touchsite_count=sum(
                1
                for blocker_class in touchsite_blocker_classes
                if blocker_class == "coverage_gap"
            ),
            policy_blocked_touchsite_count=sum(
                1
                for blocker_class in touchsite_blocker_classes
                if blocker_class == "policy_blocked"
            ),
            diagnostic_blocked_touchsite_count=sum(
                1
                for blocker_class in touchsite_blocker_classes
                if blocker_class == "diagnostic_blocked"
            ),
            ready_touchpoint_cut_count=sum(
                1
                for candidate in touchpoint_cuts
                if candidate.readiness_class == "ready_structural"
            ),
            coverage_gap_touchpoint_cut_count=sum(
                1 for candidate in touchpoint_cuts if candidate.readiness_class == "coverage_gap"
            ),
            policy_blocked_touchpoint_cut_count=sum(
                1
                for candidate in touchpoint_cuts
                if candidate.readiness_class == "policy_blocked"
            ),
            diagnostic_blocked_touchpoint_cut_count=sum(
                1
                for candidate in touchpoint_cuts
                if candidate.readiness_class == "diagnostic_blocked"
            ),
            ready_subqueue_cut_count=sum(
                1
                for candidate in subqueue_cuts
                if candidate.readiness_class == "ready_structural"
            ),
            coverage_gap_subqueue_cut_count=sum(
                1 for candidate in subqueue_cuts if candidate.readiness_class == "coverage_gap"
            ),
            policy_blocked_subqueue_cut_count=sum(
                1
                for candidate in subqueue_cuts
                if candidate.readiness_class == "policy_blocked"
            ),
            diagnostic_blocked_subqueue_cut_count=sum(
                1
                for candidate in subqueue_cuts
                if candidate.readiness_class == "diagnostic_blocked"
            ),
        )

    def dominant_blocker_class(self) -> str:
        return self._dominant_blocker_class

    @cached_property
    def _dominant_blocker_class(self) -> str:
        health_summary = self.health_summary()
        blocked_counts = (
            ("diagnostic_blocked", health_summary.diagnostic_blocked_touchsite_count),
            ("policy_blocked", health_summary.policy_blocked_touchsite_count),
            ("coverage_gap", health_summary.coverage_gap_touchsite_count),
        )
        blocker_priority = {
            "diagnostic_blocked": 0,
            "policy_blocked": 1,
            "coverage_gap": 2,
        }
        dominant = _sorted(
            [
                (blocker_class, count)
                for blocker_class, count in blocked_counts
                if count > 0
            ],
            key=lambda item: (-item[1], blocker_priority.get(item[0], 99), item[0]),
        )
        if dominant:
            return dominant[0][0]
        if health_summary.ready_touchsite_count > 0:
            return "ready_structural"
        return "none"

    def recommended_remediation_family(self) -> str:
        return self._recommended_remediation_family

    @cached_property
    def _recommended_remediation_family(self) -> str:
        if self.recommended_ready_cut() is not None:
            return "structural_cut"
        if self.recommended_diagnostic_blocked_cut() is not None:
            return "diagnostic_blocked"
        if self.recommended_policy_blocked_cut() is not None:
            return "policy_blocked"
        if self.recommended_coverage_gap_cut() is not None:
            return "coverage_gap"
        return "none"

    def remediation_lanes(self) -> tuple[InvariantRemediationLane, ...]:
        return self._remediation_lanes

    @cached_property
    def _remediation_lanes(self) -> tuple[InvariantRemediationLane, ...]:
        health_summary = self.health_summary()
        touchpoint_cuts = self.ranked_touchpoint_cuts()
        subqueue_cuts = self.ranked_subqueue_cuts()
        lane_specs = (
            ("structural_cut", "ready_structural", health_summary.ready_touchsite_count),
            (
                "diagnostic_blocked",
                "diagnostic_blocked",
                health_summary.diagnostic_blocked_touchsite_count,
            ),
            (
                "policy_blocked",
                "policy_blocked",
                health_summary.policy_blocked_touchsite_count,
            ),
            ("coverage_gap", "coverage_gap", health_summary.coverage_gap_touchsite_count),
        )
        lanes: list[InvariantRemediationLane] = []
        for remediation_family, blocker_class, touchsite_count in lane_specs:
            matching_touchpoint_cuts = tuple(
                candidate
                for candidate in touchpoint_cuts
                if candidate.readiness_class == blocker_class
            )
            matching_subqueue_cuts = tuple(
                candidate
                for candidate in subqueue_cuts
                if candidate.readiness_class == blocker_class
            )
            if (
                touchsite_count <= 0
                and not matching_touchpoint_cuts
                and not matching_subqueue_cuts
            ):
                continue
            best_touchpoint_cut = (
                matching_touchpoint_cuts[0] if matching_touchpoint_cuts else None
            )
            best_subqueue_cut = (
                matching_subqueue_cuts[0] if matching_subqueue_cuts else None
            )
            best_cut_candidates = [
                candidate
                for candidate in (best_touchpoint_cut, best_subqueue_cut)
                if candidate is not None
            ]
            best_cut = (
                None
                if not best_cut_candidates
                else _sorted(best_cut_candidates, key=_cut_sort_key)[0]
            )
            lanes.append(
                InvariantRemediationLane(
                    remediation_family=remediation_family,
                    blocker_class=blocker_class,
                    touchsite_count=touchsite_count,
                    touchpoint_cut_count=len(matching_touchpoint_cuts),
                    subqueue_cut_count=len(matching_subqueue_cuts),
                    best_touchpoint_cut=best_touchpoint_cut,
                    best_subqueue_cut=best_subqueue_cut,
                    best_cut=best_cut,
                )
            )
        return tuple(lanes)

    def documentation_followup_lane(self) -> InvariantDocumentationFollowupLane | None:
        return self._documentation_followup_lane

    @cached_property
    def _documentation_followup_lane(self) -> InvariantDocumentationFollowupLane | None:
        if self.doc_alignment_summary is None:
            return None
        if self.doc_alignment_summary.recommended_doc_alignment_action == "none":
            return None
        misaligned_target_doc_ids = self.doc_alignment_summary.misaligned_target_doc_ids
        return InvariantDocumentationFollowupLane(
            followup_family="documentation_alignment",
            alignment_status=self.doc_alignment_summary.dominant_alignment_status,
            target_doc_count=self.doc_alignment_summary.target_doc_count,
            misaligned_target_doc_count=len(misaligned_target_doc_ids),
            target_doc_ids=self.doc_ids,
            misaligned_target_doc_ids=misaligned_target_doc_ids,
            recommended_action=self.doc_alignment_summary.recommended_doc_alignment_action,
            best_target_doc_id=(
                misaligned_target_doc_ids[0] if misaligned_target_doc_ids else None
            ),
        )

    def as_payload(self) -> dict[str, object]:
        ranked_touchpoint_cuts = self.ranked_touchpoint_cuts()
        ranked_subqueue_cuts = self.ranked_subqueue_cuts()
        recommended_cut = self.recommended_cut()
        recommended_ready_cut = self.recommended_ready_cut()
        recommended_coverage_gap_cut = self.recommended_coverage_gap_cut()
        recommended_policy_blocked_cut = self.recommended_policy_blocked_cut()
        recommended_diagnostic_blocked_cut = self.recommended_diagnostic_blocked_cut()
        health_summary = self.health_summary()
        remediation_lanes = self.remediation_lanes()
        documentation_followup_lane = self.documentation_followup_lane()
        ranked_followups = self.ranked_followups()
        recommended_followup = self.recommended_followup()
        return {
            "object_id": self.object_id.wire(),
            "title": self.title,
            "status": self.status,
            "site_identity": self.site_identity.wire(),
            "structural_identity": self.structural_identity.wire(),
            "marker_identity": self.marker_identity,
            "reasoning_summary": self.reasoning_summary,
            "reasoning_control": self.reasoning_control,
            "blocking_dependencies": list(self.blocking_dependencies),
            "object_ids": list(self.object_ids),
            "doc_ids": list(self.doc_ids),
            "policy_ids": list(self.policy_ids),
            "touchsite_count": self.touchsite_count,
            "collapsible_touchsite_count": self.collapsible_touchsite_count,
            "surviving_touchsite_count": self.surviving_touchsite_count,
            "policy_signal_count": self.policy_signal_count,
            "coverage_count": self.coverage_count,
            "diagnostic_count": self.diagnostic_count,
            "doc_alignment_summary": (
                None
                if self.doc_alignment_summary is None
                else self.doc_alignment_summary.as_payload()
            ),
            "subqueues": [item.as_payload() for item in self.iter_subqueues()],
            "touchpoints": [item.as_payload() for item in self.iter_touchpoints()],
            "health_summary": health_summary.as_payload(),
            "next_actions": {
                "dominant_blocker_class": self.dominant_blocker_class(),
                "recommended_remediation_family": self.recommended_remediation_family(),
                "dominant_doc_alignment_status": self.dominant_doc_alignment_status(),
                "recommended_doc_alignment_action": self.recommended_doc_alignment_action(),
                "misaligned_target_doc_ids": list(self.misaligned_target_doc_ids()),
                "next_human_followup_family": self.next_human_followup_family(),
                "recommended_doc_followup_target_doc_id": (
                    self.recommended_doc_followup_target_doc_id()
                ),
                "recommended_followup": (
                    None
                    if recommended_followup is None
                    else recommended_followup.as_payload()
                ),
                "recommended_cut": (
                    recommended_cut.as_payload() if recommended_cut is not None else None
                ),
                "recommended_ready_cut": (
                    recommended_ready_cut.as_payload()
                    if recommended_ready_cut is not None
                    else None
                ),
                "recommended_coverage_gap_cut": (
                    recommended_coverage_gap_cut.as_payload()
                    if recommended_coverage_gap_cut is not None
                    else None
                ),
                "recommended_policy_blocked_cut": (
                    recommended_policy_blocked_cut.as_payload()
                    if recommended_policy_blocked_cut is not None
                    else None
                ),
                "recommended_diagnostic_blocked_cut": (
                    recommended_diagnostic_blocked_cut.as_payload()
                    if recommended_diagnostic_blocked_cut is not None
                    else None
                ),
                "remediation_lanes": [
                    item.as_payload() for item in remediation_lanes
                ],
                "documentation_followup_lanes": (
                    []
                    if documentation_followup_lane is None
                    else [documentation_followup_lane.as_payload()]
                ),
                "ranked_followups": [
                    item.as_payload() for item in ranked_followups
                ],
                "ranked_touchpoint_cuts": [
                    item.as_payload() for item in ranked_touchpoint_cuts
                ],
                "ranked_subqueue_cuts": [
                    item.as_payload() for item in ranked_subqueue_cuts
                ],
            },
        }


@dataclass(frozen=True)
class InvariantWorkstreamsProjection:
    root: str
    generated_at_utc: str
    workstreams: ReplayableStream[InvariantWorkstreamProjection]
    diagnostics: tuple[InvariantGraphDiagnostic, ...] = ()
    node_lookup: Mapping[str, InvariantGraphNode] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def iter_workstreams(self) -> Iterator[InvariantWorkstreamProjection]:
        return iter(self.workstreams)

    @cached_property
    def _workstream_cache(self) -> tuple[InvariantWorkstreamProjection, ...]:
        return tuple(self.iter_workstreams())

    def diagnostic_summary(self) -> InvariantDiagnosticSummary:
        return self._diagnostic_summary

    @cached_property
    def _diagnostic_summary(self) -> InvariantDiagnosticSummary:
        bucket_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
        unmatched_policy_signal_count = 0
        unresolved_blocking_dependency_count = 0
        for diagnostic in self.diagnostics:
            bucket_counts[(diagnostic.code, diagnostic.severity)] += 1
            if diagnostic.code == "unmatched_policy_signal":
                unmatched_policy_signal_count += 1
            if diagnostic.code == "unresolved_blocking_dependency":
                unresolved_blocking_dependency_count += 1
        buckets = tuple(
            InvariantDiagnosticBucket(code=code, severity=severity, count=count)
            for (code, severity), count in _sorted(
                list(bucket_counts.items()),
                key=lambda item: (item[0][0], item[0][1]),
            )
        )
        return InvariantDiagnosticSummary(
            diagnostic_count=len(self.diagnostics),
            unmatched_policy_signal_count=unmatched_policy_signal_count,
            unresolved_blocking_dependency_count=unresolved_blocking_dependency_count,
            buckets=buckets,
        )

    def _repo_diagnostic_candidate_owner_ids(
        self,
        *,
        rel_path: str,
        exact_only: bool = False,
    ) -> tuple[str, ...]:
        if not rel_path:
            return ()
        exact_matches: set[str] = set()
        family_matches: set[str] = set()
        diagnostic_parent = Path(rel_path).parent.as_posix()
        for workstream in self.iter_workstreams():
            exact_match = False
            family_match = False
            for touchpoint in workstream.iter_touchpoints():
                if touchpoint.rel_path:
                    if touchpoint.rel_path == rel_path:
                        exact_match = True
                    elif Path(touchpoint.rel_path).parent.as_posix() == diagnostic_parent:
                        family_match = True
                for touchsite in touchpoint.iter_touchsites():
                    if touchsite.rel_path:
                        if touchsite.rel_path == rel_path:
                            exact_match = True
                        elif (
                            Path(touchsite.rel_path).parent.as_posix()
                            == diagnostic_parent
                        ):
                            family_match = True
            if exact_match:
                exact_matches.add(workstream.object_id.wire())
            elif family_match:
                family_matches.add(workstream.object_id.wire())
        if exact_matches:
            return tuple(_sorted(list(exact_matches)))
        if exact_only:
            return ()
        family_owner_object_ids = self._repo_diagnostic_family_owner_ids(
            rel_path=rel_path
        )
        if family_owner_object_ids:
            return family_owner_object_ids
        proximity_scores = self._repo_diagnostic_proximity_owner_scores(rel_path=rel_path)
        if proximity_scores:
            return tuple(_sorted(list(proximity_scores)))
        return ()

    def _repo_diagnostic_family_owner_ids(
        self,
        *,
        rel_path: str,
    ) -> tuple[str, ...]:
        if not rel_path:
            return ()
        family_matches: set[str] = set()
        diagnostic_parent = Path(rel_path).parent.as_posix()
        for workstream in self.iter_workstreams():
            family_match = False
            for touchpoint in workstream.iter_touchpoints():
                if (
                    touchpoint.rel_path
                    and Path(touchpoint.rel_path).parent.as_posix() == diagnostic_parent
                ):
                    family_match = True
                for touchsite in touchpoint.iter_touchsites():
                    if (
                        touchsite.rel_path
                        and Path(touchsite.rel_path).parent.as_posix() == diagnostic_parent
                    ):
                        family_match = True
            if family_match:
                family_matches.add(workstream.object_id.wire())
        return tuple(_sorted(list(family_matches)))

    @staticmethod
    def _shared_path_prefix_depth(
        left: tuple[str, ...],
        right: tuple[str, ...],
    ) -> int:
        depth = 0
        for left_part, right_part in zip(left, right):
            if left_part != right_part:
                break
            depth += 1
        return depth

    def _repo_diagnostic_proximity_owner_scores(
        self,
        *,
        rel_path: str,
        minimum_prefix_depth: int = 4,
    ) -> dict[str, int]:
        if not rel_path:
            return {}
        diagnostic_parent_parts = Path(rel_path).parent.parts
        if not diagnostic_parent_parts:
            return {}
        scores: dict[str, int] = {}
        for workstream in self.iter_workstreams():
            best_depth = 0
            for touchpoint in workstream.iter_touchpoints():
                if touchpoint.rel_path:
                    best_depth = max(
                        best_depth,
                        self._shared_path_prefix_depth(
                            diagnostic_parent_parts,
                            Path(touchpoint.rel_path).parent.parts,
                        ),
                    )
                for touchsite in touchpoint.iter_touchsites():
                    if touchsite.rel_path:
                        best_depth = max(
                            best_depth,
                            self._shared_path_prefix_depth(
                                diagnostic_parent_parts,
                                Path(touchsite.rel_path).parent.parts,
                            ),
                        )
            if best_depth >= minimum_prefix_depth:
                scores[workstream.object_id.wire()] = best_depth
        return scores

    def _repo_diagnostic_candidate_owner_seed_path(self, *, rel_path: str) -> str | None:
        if not rel_path:
            return None
        parent = Path(rel_path).parent.as_posix()
        if not parent or parent == ".":
            return None
        return parent

    def _repo_diagnostic_candidate_owner_seed_object_id(
        self,
        *,
        seed_path: str | None,
    ) -> str | None:
        if seed_path is None:
            return None
        seed_token = seed_path.removeprefix("src/").replace("/", ".")
        if not seed_token:
            return None
        identity_space = PolicyQueueIdentitySpace()
        return identity_space.workstream_id(f"WS-SEED:{seed_token}").wire()

    def _repo_diagnostic_candidate_owner_status(
        self,
        *,
        rel_path: str,
        candidate_owner_object_ids: tuple[str, ...],
        candidate_owner_seed_path: str | None,
    ) -> str:
        if not candidate_owner_object_ids:
            if candidate_owner_seed_path is not None:
                return "source_family_seed_owner"
            return "unassigned"
        exact_owner_object_ids = self._repo_diagnostic_candidate_owner_ids(
            rel_path=rel_path,
            exact_only=True,
        )
        if exact_owner_object_ids:
            if len(exact_owner_object_ids) == 1:
                return "exact_path_owner"
            return "ambiguous_exact_path_owner"
        family_owner_object_ids = self._repo_diagnostic_family_owner_ids(rel_path=rel_path)
        if family_owner_object_ids and family_owner_object_ids == candidate_owner_object_ids:
            if len(candidate_owner_object_ids) == 1:
                return "path_family_owner"
            return "ambiguous_path_family_owner"
        proximity_owner_scores = self._repo_diagnostic_proximity_owner_scores(
            rel_path=rel_path
        )
        if proximity_owner_scores and tuple(
            _sorted(list(proximity_owner_scores))
        ) == candidate_owner_object_ids:
            if len(candidate_owner_object_ids) == 1:
                return "structural_proximity_owner"
            return "ambiguous_structural_proximity_owner"
        if len(candidate_owner_object_ids) == 1:
            return "candidate_owner"
        return "ambiguous_candidate_owner"

    def _repo_diagnostic_candidate_owner_options(
        self,
        *,
        rel_path: str,
        candidate_owner_object_ids: tuple[str, ...],
        candidate_owner_seed_object_id: str | None,
    ) -> tuple[InvariantOwnerCandidateOption, ...]:
        exact_owner_object_ids = self._repo_diagnostic_candidate_owner_ids(
            rel_path=rel_path,
            exact_only=True,
        )
        family_owner_object_ids = self._repo_diagnostic_family_owner_ids(rel_path=rel_path)
        proximity_owner_scores = self._repo_diagnostic_proximity_owner_scores(
            rel_path=rel_path
        )
        options: list[InvariantOwnerCandidateOption] = []
        if exact_owner_object_ids:
            for object_id in exact_owner_object_ids:
                options.append(
                    InvariantOwnerCandidateOption(
                        resolution_kind="attach_existing_owner",
                        owner_status=(
                            "exact_path_owner"
                            if len(exact_owner_object_ids) == 1
                            else "ambiguous_exact_path_owner"
                        ),
                        object_id=object_id,
                        score=300,
                        rationale="exact_path_match",
                        score_components=(
                            InvariantScoreComponent(
                                kind="attach_existing_owner_base",
                                score=200,
                                rationale="attach_existing_owner",
                            ),
                            InvariantScoreComponent(
                                kind="exact_path_bonus",
                                score=100,
                                rationale="exact_path_match",
                            ),
                        ),
                    )
                )
        elif family_owner_object_ids:
            for object_id in family_owner_object_ids:
                options.append(
                    InvariantOwnerCandidateOption(
                        resolution_kind="attach_existing_owner",
                        owner_status=(
                            "path_family_owner"
                            if len(family_owner_object_ids) == 1
                            else "ambiguous_path_family_owner"
                        ),
                        object_id=object_id,
                        score=200,
                        rationale="same_parent_path",
                        score_components=(
                            InvariantScoreComponent(
                                kind="attach_existing_owner_base",
                                score=200,
                                rationale="attach_existing_owner",
                            ),
                        ),
                    )
                )
        elif proximity_owner_scores:
            owner_status = (
                "structural_proximity_owner"
                if len(proximity_owner_scores) == 1
                else "ambiguous_structural_proximity_owner"
            )
            for object_id, depth in proximity_owner_scores.items():
                options.append(
                    InvariantOwnerCandidateOption(
                        resolution_kind="attach_existing_owner",
                        owner_status=owner_status,
                        object_id=object_id,
                        score=120 + (depth * 10),
                        rationale=f"shared_source_family_prefix:{depth}",
                        score_components=(
                            InvariantScoreComponent(
                                kind="attach_existing_owner_base",
                                score=120,
                                rationale="attach_existing_owner",
                            ),
                            InvariantScoreComponent(
                                kind="structural_proximity_bonus",
                                score=depth * 10,
                                rationale=f"shared_source_family_prefix:{depth}",
                            ),
                        ),
                    )
                )
        if candidate_owner_seed_object_id is not None:
            options.append(
                InvariantOwnerCandidateOption(
                    resolution_kind="seed_new_owner",
                    owner_status="source_family_seed_owner",
                    object_id=candidate_owner_seed_object_id,
                    score=100,
                    rationale="source_family_seed",
                    score_components=(
                        InvariantScoreComponent(
                            kind="seed_new_owner_base",
                            score=100,
                            rationale="source_family_seed",
                        ),
                    ),
                )
            )
        ranked_options = tuple(
            _sorted(
                options,
                key=lambda item: (-item.score, item.resolution_kind, item.object_id),
            )
        )
        if not ranked_options:
            return ()
        frontier_option = ranked_options[0]
        projected_options: list[InvariantOwnerCandidateOption] = []
        for index, option in enumerate(ranked_options, start=1):
            if option.object_id == frontier_option.object_id:
                projected_options.append(
                    replace(
                        option,
                        selection_rank=index,
                        opportunity_cost_score=0,
                        opportunity_cost_reason="frontier",
                        opportunity_cost_components=(),
                    )
                )
                continue
            opportunity_cost_score = max(0, frontier_option.score - option.score)
            projected_options.append(
                replace(
                    option,
                    selection_rank=index,
                    opportunity_cost_score=opportunity_cost_score,
                    opportunity_cost_reason=(
                        "cofrontier"
                        if opportunity_cost_score == 0
                        else f"{frontier_option.rationale}->{option.rationale}"
                    ),
                    opportunity_cost_components=(
                        ()
                        if opportunity_cost_score == 0
                        else self._owner_choice_margin_components(
                            best_option=frontier_option,
                            runner_up_option=option,
                        )
                    ),
                )
            )
        return tuple(projected_options)

    @staticmethod
    def _owner_choice_margin_components(
        *,
        best_option: InvariantOwnerCandidateOption | None,
        runner_up_option: InvariantOwnerCandidateOption | None,
    ) -> tuple[InvariantScoreComponent, ...]:
        if best_option is None:
            return ()
        components: list[InvariantScoreComponent] = list(best_option.score_components)
        if runner_up_option is None:
            return tuple(components)
        runner_up_components = runner_up_option.score_components
        if runner_up_components:
            components.extend(
                InvariantScoreComponent(
                    kind=f"runner_up_offset:{component.kind}",
                    score=-component.score,
                    rationale=component.rationale,
                )
                for component in runner_up_components
            )
        else:
            components.append(
                InvariantScoreComponent(
                    kind="runner_up_offset",
                    score=-runner_up_option.score,
                    rationale=runner_up_option.rationale,
                )
            )
        return tuple(components)

    def _owner_option_tradeoff(
        self,
        options: tuple[InvariantOwnerCandidateOption, ...],
    ) -> tuple[int | None, str | None, tuple[InvariantScoreComponent, ...]]:
        if not options:
            return None, None, ()
        best_option = options[0]
        if len(options) == 1:
            return (
                best_option.score,
                "uncontested_best_option",
                best_option.score_components,
            )
        if len(options) == 2:
            runner_up_option = options[1]
            return (
                runner_up_option.opportunity_cost_score,
                runner_up_option.opportunity_cost_reason,
                self._owner_choice_margin_components(
                    best_option=best_option,
                    runner_up_option=runner_up_option,
                ),
            )
        tradeoff_components = tuple(
            InvariantScoreComponent(
                kind="owner_option_gap_bonus",
                score=option.opportunity_cost_score,
                rationale=option.opportunity_cost_reason or option.rationale,
            )
            for option in options[1:]
            if option.opportunity_cost_score > 0
        )
        tradeoff_score = sum(component.score for component in tradeoff_components)
        if tradeoff_score <= 0:
            return 0, "cofrontier_options", ()
        return (
            tradeoff_score,
            "aggregate_owner_option_gap",
            tradeoff_components,
        )

    def _repo_followup_utility(
        self,
        followup: InvariantRepoFollowupAction,
    ) -> tuple[int, str, tuple[InvariantScoreComponent, ...]]:
        if followup.diagnostic_code == "unmatched_policy_signal":
            owner_score = followup.owner_resolution_score or 0
            owner_option_tradeoff = followup.owner_option_tradeoff_score or 0
            governance_priority_identity = self._governance_priority_identity(
                followup.policy_ids
            )
            governance_priority_bonus = self._governance_priority_bonus(
                followup.policy_ids
            )
            score = (
                900
                + owner_score
                + owner_option_tradeoff
                + governance_priority_bonus
            )
            reason = "governance_orphan"
            if followup.owner_resolution_kind is not None:
                reason = f"{reason}:{followup.owner_resolution_kind}"
            if owner_option_tradeoff > 0:
                reason = (
                    f"{reason}+owner_option_tradeoff:{owner_option_tradeoff}"
                )
            if (
                governance_priority_bonus > 0
                and governance_priority_identity is not None
            ):
                priority_rank, policy_id = governance_priority_identity
                reason = (
                    f"{reason}+governance_priority:{policy_id}:{priority_rank}"
                )
            return (
                score,
                reason,
                tuple(
                    component
                    for component in (
                        InvariantScoreComponent(
                            kind="governance_orphan_base",
                            score=900,
                            rationale="governance_orphan",
                        ),
                        InvariantScoreComponent(
                            kind="owner_resolution_bonus",
                            score=owner_score,
                            rationale=(
                                followup.owner_resolution_kind or "owner_resolution:none"
                            ),
                        ),
                        InvariantScoreComponent(
                            kind="owner_option_tradeoff_bonus",
                            score=owner_option_tradeoff,
                            rationale=(
                                followup.owner_option_tradeoff_reason
                                or "owner_option_tradeoff:none"
                            ),
                        ),
                        (
                            InvariantScoreComponent(
                                kind="governance_priority_bonus",
                                score=governance_priority_bonus,
                                rationale=(
                                    "governance_priority:none"
                                    if governance_priority_identity is None
                                    else (
                                        "governance_priority:"
                                        f"{governance_priority_identity[1]}:"
                                        f"{governance_priority_identity[0]}"
                                    )
                                ),
                            )
                            if governance_priority_bonus > 0
                            else None
                        ),
                    )
                    if component is not None
                ),
            )
        if followup.diagnostic_code == "unresolved_blocking_dependency":
            return (
                850,
                "dependency_orphan",
                (
                    InvariantScoreComponent(
                        kind="dependency_orphan_base",
                        score=850,
                        rationale="dependency_orphan",
                    ),
                ),
            )
        if followup.action_kind == "diagnostic_resolution":
            return (
                800,
                "diagnostic_backlog",
                (
                    InvariantScoreComponent(
                        kind="diagnostic_backlog_base",
                        score=800,
                        rationale="diagnostic_backlog",
                    ),
                ),
            )
        if followup.action_kind == "touchpoint_cut":
            readiness = followup.readiness_class or "none"
            score_map = {
                "ready_structural": 700,
                "policy_blocked": 520,
                "diagnostic_blocked": 500,
                "coverage_gap": 480,
            }
            score = score_map.get(readiness, 450)
            base_score = 450
            return (
                score,
                f"code:{readiness}",
                (
                    InvariantScoreComponent(
                        kind="code_touchpoint_base",
                        score=base_score,
                        rationale="code:touchpoint_cut",
                    ),
                    InvariantScoreComponent(
                        kind="readiness_bonus",
                        score=score - base_score,
                        rationale=f"readiness:{readiness}",
                    ),
                ),
            )
        if followup.action_kind == "subqueue_cut":
            readiness = followup.readiness_class or "none"
            score_map = {
                "ready_structural": 680,
                "policy_blocked": 500,
                "diagnostic_blocked": 480,
                "coverage_gap": 460,
            }
            score = score_map.get(readiness, 430)
            base_score = 430
            return (
                score,
                f"code:{readiness}",
                (
                    InvariantScoreComponent(
                        kind="code_subqueue_base",
                        score=base_score,
                        rationale="code:subqueue_cut",
                    ),
                    InvariantScoreComponent(
                        kind="readiness_bonus",
                        score=score - base_score,
                        rationale=f"readiness:{readiness}",
                    ),
                ),
            )
        if followup.action_kind == "doc_alignment":
            alignment = followup.alignment_status or "none"
            score_map = {
                "missing_target_doc": 420,
                "append_pending_new_object": 390,
                "append_pending_existing_object": 380,
                "ambiguous_target_doc": 360,
                "unassigned_target_doc": 340,
            }
            score = score_map.get(alignment, 320)
            base_score = 320
            return (
                score,
                f"documentation:{alignment}",
                (
                    InvariantScoreComponent(
                        kind="documentation_alignment_base",
                        score=base_score,
                        rationale="documentation",
                    ),
                    InvariantScoreComponent(
                        kind="alignment_bonus",
                        score=score - base_score,
                        rationale=f"alignment:{alignment}",
                    ),
                ),
            )
        return (
            250,
            followup.followup_family,
            (
                InvariantScoreComponent(
                    kind="generic_followup_base",
                    score=250,
                    rationale=followup.followup_family,
                ),
            ),
        )

    @staticmethod
    def _repo_followup_opportunity_components(
        *,
        frontier_followup: InvariantRepoFollowupAction,
        followup: InvariantRepoFollowupAction,
    ) -> tuple[InvariantScoreComponent, ...]:
        if (
            frontier_followup.followup_family == followup.followup_family
            and frontier_followup.action_kind == followup.action_kind
            and frontier_followup.object_id == followup.object_id
            and frontier_followup.diagnostic_code == followup.diagnostic_code
            and frontier_followup.target_doc_id == followup.target_doc_id
            and frontier_followup.policy_ids == followup.policy_ids
        ):
            return ()
        components: list[InvariantScoreComponent] = list(
            frontier_followup.utility_components
        )
        if followup.utility_components:
            components.extend(
                InvariantScoreComponent(
                    kind=f"runner_up_offset:{component.kind}",
                    score=-component.score,
                    rationale=component.rationale,
                )
                for component in followup.utility_components
            )
        else:
            components.append(
                InvariantScoreComponent(
                    kind="runner_up_offset",
                    score=-followup.utility_score,
                    rationale=followup.utility_reason,
                )
            )
        return tuple(components)

    def _repo_followup_cohort_members(
        self,
        *,
        cofrontier_followups: tuple[InvariantRepoFollowupAction, ...],
    ) -> tuple[InvariantRepoFollowupCohortMember, ...]:
        frontier_followup = cofrontier_followups[0]
        return tuple(
            InvariantRepoFollowupCohortMember(
                followup_family=item.followup_family,
                followup_class=self._repo_followup_class(item),
                action_kind=item.action_kind,
                object_id=item.object_id,
                diagnostic_code=item.diagnostic_code,
                target_doc_id=item.target_doc_id,
                policy_ids=item.policy_ids,
                title=item.title,
                utility_score=item.utility_score,
                selection_rank=index,
                selection_reason=self._repo_followup_cohort_selection_reason(
                    frontier_followup=frontier_followup,
                    followup=item,
                    selection_rank=index,
                ),
            )
            for index, item in enumerate(cofrontier_followups, start=1)
        )

    def _repo_followup_cohort_selection_reason(
        self,
        *,
        frontier_followup: InvariantRepoFollowupAction,
        followup: InvariantRepoFollowupAction,
        selection_rank: int,
    ) -> str:
        if selection_rank == 1:
            return "frontier_tiebreak_winner"
        followup_governance_priority = self._governance_priority_identity(
            followup.policy_ids
        )
        frontier_governance_priority = self._governance_priority_identity(
            frontier_followup.policy_ids
        )
        if (
            followup_governance_priority != frontier_governance_priority
            and followup_governance_priority is not None
        ):
            priority_rank, policy_id = followup_governance_priority
            return f"governance_priority:{policy_id}:{priority_rank}"
        if followup.policy_ids != frontier_followup.policy_ids and followup.policy_ids:
            return "policy_ids:" + ",".join(followup.policy_ids)
        if followup.priority_rank != frontier_followup.priority_rank:
            return f"priority_rank:{followup.priority_rank}"
        if (followup.owner_resolution_score or 0) != (
            frontier_followup.owner_resolution_score or 0
        ):
            return (
                "owner_resolution_score:"
                f"{followup.owner_resolution_score or 0}"
            )
        if followup.count != frontier_followup.count:
            return f"count:{followup.count}"
        if followup.followup_family != frontier_followup.followup_family:
            return f"followup_family:{followup.followup_family}"
        if followup.title != frontier_followup.title:
            return f"title:{followup.title}"
        if (followup.object_id or "") != (frontier_followup.object_id or ""):
            return f"object_id:{followup.object_id or '<none>'}"
        if (followup.target_doc_id or "") != (frontier_followup.target_doc_id or ""):
            return f"target_doc_id:{followup.target_doc_id or '<none>'}"
        if (followup.diagnostic_code or "") != (
            frontier_followup.diagnostic_code or ""
        ):
            return f"diagnostic_code:{followup.diagnostic_code or '<none>'}"
        return "cofrontier_peer"

    def ranked_repo_followups(self) -> tuple[InvariantRepoFollowupAction, ...]:
        return self._ranked_repo_followups

    def _governance_priority_identity(
        self,
        policy_ids: tuple[str, ...],
    ) -> tuple[int, str] | None:
        ranked_policy_ids = [
            (priority_rank, policy_id)
            for policy_id in policy_ids
            if (
                priority_rank := grade_monotonicity_governance_priority_rank(policy_id)
            )
            is not None
        ]
        if not ranked_policy_ids:
            return None
        return tuple(
            _sorted(
                ranked_policy_ids,
                key=lambda item: (item[0], item[1]),
            )[0]
        )

    def _governance_priority_bonus(
        self,
        policy_ids: tuple[str, ...],
    ) -> int:
        governance_priority_identity = self._governance_priority_identity(policy_ids)
        if governance_priority_identity is None:
            return 0
        priority_rank, _ = governance_priority_identity
        return max(0, 100 - priority_rank)

    @cached_property
    def _ranked_repo_followups(self) -> tuple[InvariantRepoFollowupAction, ...]:
        actions: list[InvariantRepoFollowupAction] = []
        for lane in self.repo_diagnostic_lanes():
            if lane.diagnostic_code == "unmatched_policy_signal":
                best_option = (
                    lane.candidate_owner_options[0]
                    if lane.candidate_owner_options
                    else None
                )
                runner_up_option = lane.runner_up_candidate_owner_option
                (
                    owner_option_tradeoff_score,
                    owner_option_tradeoff_reason,
                    owner_option_tradeoff_components,
                ) = self._owner_option_tradeoff(lane.candidate_owner_options)
                if (
                    best_option is not None
                    and best_option.resolution_kind == "attach_existing_owner"
                ):
                    title = (
                        f"resolve {lane.title} ownership via "
                        f"{best_option.object_id}"
                    )
                elif lane.candidate_owner_seed_path:
                    title = (
                        f"seed ownership for {lane.title} from "
                        f"{lane.candidate_owner_seed_path}"
                    )
                else:
                    title = f"resolve {lane.title} ownership"
                if lane.rel_path and not lane.candidate_owner_seed_path:
                    title = (
                        f"{title} at {lane.rel_path}"
                        + (f"::{lane.qualname}" if lane.qualname else "")
                    )
                actions.append(
                    InvariantRepoFollowupAction(
                        followup_family="governance_orphan_resolution",
                        action_kind="diagnostic_resolution",
                        priority_rank=0,
                        object_id=None,
                        owner_object_id=lane.candidate_owner_object_id,
                        diagnostic_code=lane.diagnostic_code,
                        target_doc_id=None,
                        policy_ids=lane.policy_ids,
                        title=title,
                        blocker_class="policy_orphan",
                        readiness_class=None,
                        alignment_status=None,
                        recommended_action=lane.recommended_action,
                        owner_seed_path=lane.candidate_owner_seed_path,
                        owner_seed_object_id=lane.candidate_owner_seed_object_id,
                        owner_resolution_kind=(
                            best_option.resolution_kind if best_option is not None else None
                        ),
                        owner_resolution_score=(
                            best_option.score if best_option is not None else None
                        ),
                        owner_resolution_options=lane.candidate_owner_options,
                        runner_up_owner_object_id=(
                            runner_up_option.object_id
                            if runner_up_option is not None
                            else None
                        ),
                        runner_up_owner_resolution_kind=(
                            runner_up_option.resolution_kind
                            if runner_up_option is not None
                            else None
                        ),
                        runner_up_owner_resolution_score=(
                            runner_up_option.score
                            if runner_up_option is not None
                            else None
                        ),
                        owner_choice_margin_score=lane.candidate_owner_choice_margin_score,
                        owner_choice_margin_reason=lane.candidate_owner_choice_margin_reason,
                        owner_choice_margin_components=(
                            lane.candidate_owner_choice_margin_components
                        ),
                        owner_option_tradeoff_score=owner_option_tradeoff_score,
                        owner_option_tradeoff_reason=owner_option_tradeoff_reason,
                        owner_option_tradeoff_components=(
                            owner_option_tradeoff_components
                        ),
                        utility_score=0,
                        utility_reason="",
                        utility_components=(),
                        selection_certainty_kind="ranked_unique",
                        cofrontier_followup_count=1,
                        cofrontier_followup_cohort=(),
                        selection_scope_kind="singleton",
                        selection_scope_id=None,
                        runner_up_followup_family=None,
                        runner_up_followup_class=None,
                        runner_up_followup_object_id=None,
                        runner_up_followup_utility_score=None,
                        frontier_choice_margin_score=None,
                        frontier_choice_margin_reason=None,
                        frontier_choice_margin_components=(),
                        selection_rank=0,
                        opportunity_cost_score=0,
                        opportunity_cost_reason="frontier",
                        opportunity_cost_components=(),
                        count=lane.count,
                    )
                )
            elif lane.diagnostic_code == "unresolved_blocking_dependency":
                title = "resolve unresolved blocking dependencies"
                if lane.rel_path:
                    title = (
                        f"{title} at {lane.rel_path}"
                        + (f"::{lane.qualname}" if lane.qualname else "")
                    )
                actions.append(
                    InvariantRepoFollowupAction(
                        followup_family="dependency_resolution",
                        action_kind="diagnostic_resolution",
                        priority_rank=10,
                        object_id=None,
                        owner_object_id=None,
                        diagnostic_code=lane.diagnostic_code,
                        target_doc_id=None,
                        policy_ids=lane.policy_ids,
                        title=title,
                        blocker_class="dependency_orphan",
                        readiness_class=None,
                        alignment_status=None,
                        recommended_action=lane.recommended_action,
                        owner_seed_path=None,
                        owner_seed_object_id=None,
                        owner_resolution_kind=None,
                        owner_resolution_score=None,
                        owner_resolution_options=(),
                        runner_up_owner_object_id=None,
                        runner_up_owner_resolution_kind=None,
                        runner_up_owner_resolution_score=None,
                        owner_choice_margin_score=None,
                        owner_choice_margin_reason=None,
                        owner_choice_margin_components=(),
                        owner_option_tradeoff_score=None,
                        owner_option_tradeoff_reason=None,
                        owner_option_tradeoff_components=(),
                        utility_score=0,
                        utility_reason="",
                        utility_components=(),
                        selection_certainty_kind="ranked_unique",
                        cofrontier_followup_count=1,
                        cofrontier_followup_cohort=(),
                        selection_scope_kind="singleton",
                        selection_scope_id=None,
                        runner_up_followup_family=None,
                        runner_up_followup_class=None,
                        runner_up_followup_object_id=None,
                        runner_up_followup_utility_score=None,
                        frontier_choice_margin_score=None,
                        frontier_choice_margin_reason=None,
                        frontier_choice_margin_components=(),
                        selection_rank=0,
                        opportunity_cost_score=0,
                        opportunity_cost_reason="frontier",
                        opportunity_cost_components=(),
                        count=lane.count,
                    )
                )
            else:
                title = lane.title
                if lane.rel_path:
                    title = (
                        f"{title} at {lane.rel_path}"
                        + (f"::{lane.qualname}" if lane.qualname else "")
                    )
                actions.append(
                    InvariantRepoFollowupAction(
                        followup_family="diagnostic_backlog",
                        action_kind="diagnostic_resolution",
                        priority_rank=20,
                        object_id=None,
                        owner_object_id=None,
                        diagnostic_code=lane.diagnostic_code,
                        target_doc_id=None,
                        policy_ids=lane.policy_ids,
                        title=title,
                        blocker_class="diagnostic_backlog",
                        readiness_class=None,
                        alignment_status=None,
                        recommended_action=lane.recommended_action,
                        owner_seed_path=None,
                        owner_seed_object_id=None,
                        owner_resolution_kind=None,
                        owner_resolution_score=None,
                        owner_resolution_options=(),
                        runner_up_owner_object_id=None,
                        runner_up_owner_resolution_kind=None,
                        runner_up_owner_resolution_score=None,
                        owner_choice_margin_score=None,
                        owner_choice_margin_reason=None,
                        owner_choice_margin_components=(),
                        owner_option_tradeoff_score=None,
                        owner_option_tradeoff_reason=None,
                        owner_option_tradeoff_components=(),
                        utility_score=0,
                        utility_reason="",
                        utility_components=(),
                        selection_certainty_kind="ranked_unique",
                        cofrontier_followup_count=1,
                        cofrontier_followup_cohort=(),
                        selection_scope_kind="singleton",
                        selection_scope_id=None,
                        runner_up_followup_family=None,
                        runner_up_followup_class=None,
                        runner_up_followup_object_id=None,
                        runner_up_followup_utility_score=None,
                        frontier_choice_margin_score=None,
                        frontier_choice_margin_reason=None,
                        frontier_choice_margin_components=(),
                        selection_rank=0,
                        opportunity_cost_score=0,
                        opportunity_cost_reason="frontier",
                        opportunity_cost_components=(),
                        count=lane.count,
                    )
                )
        for workstream in self.iter_workstreams():
            for followup in workstream.ranked_followups():
                actions.append(
                    InvariantRepoFollowupAction(
                        followup_family=followup.followup_family,
                        action_kind=followup.action_kind,
                        priority_rank=100 + followup.priority_rank,
                        object_id=followup.object_id,
                        owner_object_id=workstream.object_id.wire(),
                        diagnostic_code=None,
                        target_doc_id=followup.target_doc_id,
                        policy_ids=(),
                        title=followup.title,
                        blocker_class=followup.blocker_class,
                        readiness_class=followup.readiness_class,
                        alignment_status=followup.alignment_status,
                        recommended_action=followup.recommended_action,
                        owner_seed_path=None,
                        owner_seed_object_id=None,
                        owner_resolution_kind=None,
                        owner_resolution_score=None,
                        owner_resolution_options=(),
                        runner_up_owner_object_id=None,
                        runner_up_owner_resolution_kind=None,
                        runner_up_owner_resolution_score=None,
                        owner_choice_margin_score=None,
                        owner_choice_margin_reason=None,
                        owner_choice_margin_components=(),
                        owner_option_tradeoff_score=None,
                        owner_option_tradeoff_reason=None,
                        owner_option_tradeoff_components=(),
                        utility_score=0,
                        utility_reason="",
                        utility_components=(),
                        selection_certainty_kind="ranked_unique",
                        cofrontier_followup_count=1,
                        cofrontier_followup_cohort=(),
                        selection_scope_kind="singleton",
                        selection_scope_id=None,
                        runner_up_followup_family=None,
                        runner_up_followup_class=None,
                        runner_up_followup_object_id=None,
                        runner_up_followup_utility_score=None,
                        frontier_choice_margin_score=None,
                        frontier_choice_margin_reason=None,
                        frontier_choice_margin_components=(),
                        selection_rank=0,
                        opportunity_cost_score=0,
                        opportunity_cost_reason="frontier",
                        opportunity_cost_components=(),
                        count=followup.touchsite_count,
                    )
                )
        scored_actions: list[InvariantRepoFollowupAction] = []
        for action in actions:
            utility_score, utility_reason, utility_components = (
                self._repo_followup_utility(action)
            )
            scored_actions.append(
                replace(
                    action,
                    utility_score=utility_score,
                    utility_reason=utility_reason,
                    utility_components=utility_components,
                )
            )
        actions = scored_actions
        ranked_actions = tuple(
            _sorted(
                actions,
                key=lambda item: (
                    -item.utility_score,
                    item.priority_rank,
                    (
                        self._governance_priority_identity(item.policy_ids)[0]
                        if self._governance_priority_identity(item.policy_ids) is not None
                        else _DEFAULT_GOVERNANCE_PRIORITY_RANK
                    ),
                    (
                        self._governance_priority_identity(item.policy_ids)[1]
                        if self._governance_priority_identity(item.policy_ids) is not None
                        else ""
                    ),
                    -(item.owner_resolution_score or 0),
                    -item.count,
                    item.followup_family,
                    item.policy_ids,
                    item.title,
                    item.object_id or "",
                    item.target_doc_id or "",
                    item.diagnostic_code or "",
                ),
            )
        )
        if not ranked_actions:
            return ()
        frontier_followup = ranked_actions[0]
        projected_actions: list[InvariantRepoFollowupAction] = []
        for index, action in enumerate(ranked_actions, start=1):
            runner_up_followup = (
                ranked_actions[index] if index < len(ranked_actions) else None
            )
            cofrontier_followup_count = sum(
                1 for item in ranked_actions if item.utility_score == action.utility_score
            )
            cofrontier_followups = tuple(
                item for item in ranked_actions if item.utility_score == action.utility_score
            )
            cofrontier_followup_cohort = self._repo_followup_cohort_members(
                cofrontier_followups=cofrontier_followups,
            )
            selection_certainty_kind = (
                "frontier_plateau"
                if index == 1 and cofrontier_followup_count > 1
                else (
                    "frontier_unique"
                    if index == 1
                    else (
                        "ranked_plateau"
                        if cofrontier_followup_count > 1
                        else "ranked_unique"
                    )
                )
            )
            selection_scope_kind, selection_scope_id = (
                self._repo_followup_selection_scope(
                    cofrontier_followups=cofrontier_followups,
                )
            )
            frontier_choice_margin_score = (
                None
                if runner_up_followup is None
                else max(0, action.utility_score - runner_up_followup.utility_score)
            )
            frontier_choice_margin_reason = (
                None
                if runner_up_followup is None
                else (
                    "cofrontier"
                    if frontier_choice_margin_score == 0
                    else (
                        f"{action.utility_reason}"
                        f"->{runner_up_followup.utility_reason}"
                    )
                )
            )
            frontier_choice_margin_components = (
                ()
                if runner_up_followup is None or frontier_choice_margin_score == 0
                else self._repo_followup_opportunity_components(
                    frontier_followup=action,
                    followup=runner_up_followup,
                )
            )
            if index == 1:
                projected_actions.append(
                    replace(
                        action,
                        runner_up_followup_family=(
                            None
                            if runner_up_followup is None
                            else runner_up_followup.followup_family
                        ),
                        runner_up_followup_class=(
                            None
                            if runner_up_followup is None
                            else self._repo_followup_class(runner_up_followup)
                        ),
                        runner_up_followup_object_id=(
                            None
                            if runner_up_followup is None
                            else runner_up_followup.object_id
                        ),
                        runner_up_followup_utility_score=(
                            None
                            if runner_up_followup is None
                            else runner_up_followup.utility_score
                        ),
                        frontier_choice_margin_score=frontier_choice_margin_score,
                        frontier_choice_margin_reason=frontier_choice_margin_reason,
                        frontier_choice_margin_components=(
                            frontier_choice_margin_components
                        ),
                        selection_certainty_kind=selection_certainty_kind,
                        cofrontier_followup_count=cofrontier_followup_count,
                        cofrontier_followup_cohort=cofrontier_followup_cohort,
                        selection_scope_kind=selection_scope_kind,
                        selection_scope_id=selection_scope_id,
                        selection_rank=index,
                        opportunity_cost_score=0,
                        opportunity_cost_reason="frontier",
                        opportunity_cost_components=(),
                    )
                )
                continue
            opportunity_cost_score = max(
                0, frontier_followup.utility_score - action.utility_score
            )
            projected_actions.append(
                replace(
                    action,
                    runner_up_followup_family=(
                        None
                        if runner_up_followup is None
                        else runner_up_followup.followup_family
                    ),
                    runner_up_followup_class=(
                        None
                        if runner_up_followup is None
                        else self._repo_followup_class(runner_up_followup)
                    ),
                    runner_up_followup_object_id=(
                        None
                        if runner_up_followup is None
                        else runner_up_followup.object_id
                    ),
                    runner_up_followup_utility_score=(
                        None
                        if runner_up_followup is None
                        else runner_up_followup.utility_score
                    ),
                    frontier_choice_margin_score=frontier_choice_margin_score,
                    frontier_choice_margin_reason=frontier_choice_margin_reason,
                    frontier_choice_margin_components=frontier_choice_margin_components,
                    selection_certainty_kind=selection_certainty_kind,
                    cofrontier_followup_count=cofrontier_followup_count,
                    cofrontier_followup_cohort=cofrontier_followup_cohort,
                    selection_scope_kind=selection_scope_kind,
                    selection_scope_id=selection_scope_id,
                    selection_rank=index,
                    opportunity_cost_score=opportunity_cost_score,
                    opportunity_cost_reason=(
                        "cofrontier"
                        if opportunity_cost_score == 0
                        else (
                            f"{frontier_followup.utility_reason}"
                            f"->{action.utility_reason}"
                        )
                    ),
                    opportunity_cost_components=(
                        ()
                        if opportunity_cost_score == 0
                        else self._repo_followup_opportunity_components(
                            frontier_followup=frontier_followup,
                            followup=action,
                        )
                    ),
                )
            )
        return tuple(projected_actions)

    def recommended_repo_followup(self) -> InvariantRepoFollowupAction | None:
        return self._recommended_repo_followup

    @cached_property
    def _recommended_repo_followup(self) -> InvariantRepoFollowupAction | None:
        ranked = self.ranked_repo_followups()
        if not ranked:
            return None
        return ranked[0]

    def _repo_followup_class(self, followup: InvariantRepoFollowupAction) -> str:
        if followup.diagnostic_code is not None or followup.action_kind == "diagnostic_resolution":
            return "governance"
        if followup.action_kind == "doc_alignment":
            return "documentation"
        return "code"

    @staticmethod
    def _repo_followup_selection_scope(
        *,
        cofrontier_followups: tuple[InvariantRepoFollowupAction, ...],
    ) -> tuple[str, str | None]:
        if len(cofrontier_followups) <= 1:
            return ("singleton", None)
        owner_resolution_kinds = {
            item.owner_resolution_kind
            for item in cofrontier_followups
            if item.owner_resolution_kind is not None
        }
        owner_resolution_targets = {
            item.owner_object_id or item.owner_seed_object_id
            for item in cofrontier_followups
            if (item.owner_object_id or item.owner_seed_object_id) is not None
        }
        if len(owner_resolution_kinds) == 1 and len(owner_resolution_targets) == 1:
            return (
                "shared_owner_resolution_surface",
                (
                    f"{next(iter(owner_resolution_kinds))}:"
                    f"{next(iter(owner_resolution_targets))}"
                ),
            )
        followup_families = {item.followup_family for item in cofrontier_followups}
        if len(followup_families) == 1:
            return ("shared_followup_family", next(iter(followup_families)))
        return ("mixed_plateau", None)

    def _repo_followup_lane_utility(
        self,
        *,
        followup_class: str,
        followup_family: str,
        items: tuple[InvariantRepoFollowupAction, ...],
    ) -> tuple[int, str, tuple[InvariantScoreComponent, ...]]:
        best_followup = items[0]
        breadth_bonus = min(len(items), 9) * 5
        class_bonus = {
            "governance": 25,
            "code": 10,
            "documentation": 5,
        }.get(followup_class, 0)
        utility_components = (
            InvariantScoreComponent(
                kind="best_followup_utility",
                score=best_followup.utility_score,
                rationale=best_followup.utility_reason,
            ),
            InvariantScoreComponent(
                kind="lane_breadth_bonus",
                score=breadth_bonus,
                rationale=f"lane_breadth:{len(items)}",
            ),
            InvariantScoreComponent(
                kind="lane_class_bonus",
                score=class_bonus,
                rationale=f"lane_class:{followup_class}",
            ),
        )
        utility_score = best_followup.utility_score + breadth_bonus + class_bonus
        utility_reason = (
            f"{best_followup.utility_reason}+lane_breadth:{len(items)}+lane:{followup_family}"
        )
        return (utility_score, utility_reason, utility_components)

    def _repo_followup_lane_opportunity_components(
        self,
        *,
        frontier_lane: InvariantRepoFollowupLane,
        lane: InvariantRepoFollowupLane,
    ) -> tuple[InvariantScoreComponent, ...]:
        if frontier_lane.followup_family == lane.followup_family:
            return ()
        frontier_by_kind = {
            item.kind: item for item in frontier_lane.lane_utility_components
        }
        lane_by_kind = {item.kind: item for item in lane.lane_utility_components}
        components: list[InvariantScoreComponent] = []
        for kind in (
            "best_followup_utility",
            "lane_breadth_bonus",
            "lane_class_bonus",
        ):
            frontier_component = frontier_by_kind.get(kind)
            lane_component = lane_by_kind.get(kind)
            if frontier_component is None or lane_component is None:
                continue
            gap = max(0, frontier_component.score - lane_component.score)
            if gap == 0:
                continue
            components.append(
                InvariantScoreComponent(
                    kind=f"{kind}_gap",
                    score=gap,
                    rationale=(
                        f"{frontier_component.rationale}->{lane_component.rationale}"
                    ),
                )
            )
        return tuple(components)

    def recommended_repo_followup_lane(self) -> InvariantRepoFollowupLane | None:
        return self._recommended_repo_followup_lane

    @cached_property
    def _recommended_repo_followup_lane(self) -> InvariantRepoFollowupLane | None:
        lanes = self.repo_followup_lanes()
        if not lanes:
            return None
        return lanes[0]

    def recommended_repo_code_followup_lane(self) -> InvariantRepoFollowupLane | None:
        for lane in self.repo_followup_lanes():
            if lane.followup_class == "code":
                return lane
        return None

    def recommended_repo_human_followup_lane(self) -> InvariantRepoFollowupLane | None:
        for lane in self.repo_followup_lanes():
            if lane.followup_class != "code":
                return lane
        return None

    def recommended_repo_followup_cross_class_tradeoff(
        self,
    ) -> InvariantRepoFollowupCrossClassTradeoff | None:
        return self._recommended_repo_followup_cross_class_tradeoff

    @cached_property
    def _recommended_repo_followup_cross_class_tradeoff(
        self,
    ) -> InvariantRepoFollowupCrossClassTradeoff | None:
        frontier = self.recommended_repo_followup()
        if frontier is None:
            return None
        frontier_class = self._repo_followup_class(frontier)
        runner_up = next(
            (
                item
                for item in self.ranked_repo_followups()
                if self._repo_followup_class(item) != frontier_class
            ),
            None,
        )
        if runner_up is None:
            return None
        return InvariantRepoFollowupCrossClassTradeoff(
            frontier_followup_family=frontier.followup_family,
            frontier_followup_class=frontier_class,
            frontier_action_kind=frontier.action_kind,
            frontier_object_id=frontier.object_id,
            frontier_diagnostic_code=frontier.diagnostic_code,
            frontier_target_doc_id=frontier.target_doc_id,
            frontier_utility_score=frontier.utility_score,
            frontier_utility_reason=frontier.utility_reason,
            runner_up_followup_family=runner_up.followup_family,
            runner_up_followup_class=self._repo_followup_class(runner_up),
            runner_up_action_kind=runner_up.action_kind,
            runner_up_object_id=runner_up.object_id,
            runner_up_diagnostic_code=runner_up.diagnostic_code,
            runner_up_target_doc_id=runner_up.target_doc_id,
            runner_up_utility_score=runner_up.utility_score,
            runner_up_utility_reason=runner_up.utility_reason,
            margin_score=max(0, frontier.utility_score - runner_up.utility_score),
            margin_reason=f"{frontier.utility_reason}->{runner_up.utility_reason}",
            margin_components=self._repo_followup_opportunity_components(
                frontier_followup=frontier,
                followup=runner_up,
            ),
        )

    def recommended_repo_followup_same_class_tradeoff(
        self,
    ) -> InvariantRepoFollowupSameClassTradeoff | None:
        return self._recommended_repo_followup_same_class_tradeoff

    @cached_property
    def _recommended_repo_followup_same_class_tradeoff(
        self,
    ) -> InvariantRepoFollowupSameClassTradeoff | None:
        frontier = self.recommended_repo_followup()
        if frontier is None:
            return None
        frontier_class = self._repo_followup_class(frontier)
        runner_up = next(
            (
                item
                for item in self.ranked_repo_followups()[1:]
                if self._repo_followup_class(item) == frontier_class
            ),
            None,
        )
        if runner_up is None:
            return None
        return InvariantRepoFollowupSameClassTradeoff(
            frontier_followup_family=frontier.followup_family,
            frontier_followup_class=frontier_class,
            frontier_action_kind=frontier.action_kind,
            frontier_object_id=frontier.object_id,
            frontier_diagnostic_code=frontier.diagnostic_code,
            frontier_target_doc_id=frontier.target_doc_id,
            frontier_policy_ids=frontier.policy_ids,
            frontier_utility_score=frontier.utility_score,
            frontier_utility_reason=frontier.utility_reason,
            runner_up_followup_family=runner_up.followup_family,
            runner_up_followup_class=self._repo_followup_class(runner_up),
            runner_up_action_kind=runner_up.action_kind,
            runner_up_object_id=runner_up.object_id,
            runner_up_diagnostic_code=runner_up.diagnostic_code,
            runner_up_target_doc_id=runner_up.target_doc_id,
            runner_up_policy_ids=runner_up.policy_ids,
            runner_up_utility_score=runner_up.utility_score,
            runner_up_utility_reason=runner_up.utility_reason,
            margin_score=max(0, frontier.utility_score - runner_up.utility_score),
            margin_reason=f"{frontier.utility_reason}->{runner_up.utility_reason}",
            margin_components=self._repo_followup_opportunity_components(
                frontier_followup=frontier,
                followup=runner_up,
            ),
        )

    def recommended_repo_followup_frontier_triad(
        self,
    ) -> InvariantRepoFollowupFrontierTriad | None:
        return self._recommended_repo_followup_frontier_triad

    @cached_property
    def _recommended_repo_followup_frontier_triad(
        self,
    ) -> InvariantRepoFollowupFrontierTriad | None:
        frontier = self.recommended_repo_followup()
        if frontier is None:
            return None
        return InvariantRepoFollowupFrontierTriad(
            frontier_followup_family=frontier.followup_family,
            frontier_followup_class=self._repo_followup_class(frontier),
            frontier_action_kind=frontier.action_kind,
            frontier_object_id=frontier.object_id,
            frontier_diagnostic_code=frontier.diagnostic_code,
            frontier_target_doc_id=frontier.target_doc_id,
            frontier_policy_ids=frontier.policy_ids,
            frontier_utility_score=frontier.utility_score,
            frontier_utility_reason=frontier.utility_reason,
            same_class_tradeoff=self.recommended_repo_followup_same_class_tradeoff(),
            cross_class_tradeoff=self.recommended_repo_followup_cross_class_tradeoff(),
        )

    def recommended_repo_followup_frontier_explanation(
        self,
    ) -> InvariantRepoFollowupFrontierExplanation | None:
        return self._recommended_repo_followup_frontier_explanation

    @cached_property
    def _recommended_repo_followup_frontier_explanation(
        self,
    ) -> InvariantRepoFollowupFrontierExplanation | None:
        triad = self.recommended_repo_followup_frontier_triad()
        if triad is None:
            return None
        same_class_tradeoff = triad.same_class_tradeoff
        cross_class_tradeoff = triad.cross_class_tradeoff
        same_class_strength = _margin_strength(
            None if same_class_tradeoff is None else same_class_tradeoff.margin_score
        )
        cross_class_strength = _margin_strength(
            None if cross_class_tradeoff is None else cross_class_tradeoff.margin_score
        )
        return InvariantRepoFollowupFrontierExplanation(
            frontier_followup_family=triad.frontier_followup_family,
            frontier_followup_class=triad.frontier_followup_class,
            frontier_action_kind=triad.frontier_action_kind,
            frontier_object_id=triad.frontier_object_id,
            frontier_diagnostic_code=triad.frontier_diagnostic_code,
            frontier_target_doc_id=triad.frontier_target_doc_id,
            frontier_policy_ids=triad.frontier_policy_ids,
            frontier_utility_score=triad.frontier_utility_score,
            frontier_utility_reason=triad.frontier_utility_reason,
            same_class_runner_up_followup_family=(
                None if same_class_tradeoff is None else same_class_tradeoff.runner_up_followup_family
            ),
            same_class_runner_up_followup_class=(
                None if same_class_tradeoff is None else same_class_tradeoff.runner_up_followup_class
            ),
            same_class_runner_up_action_kind=(
                None if same_class_tradeoff is None else same_class_tradeoff.runner_up_action_kind
            ),
            same_class_runner_up_object_id=(
                None if same_class_tradeoff is None else same_class_tradeoff.runner_up_object_id
            ),
            same_class_runner_up_diagnostic_code=(
                None if same_class_tradeoff is None else same_class_tradeoff.runner_up_diagnostic_code
            ),
            same_class_runner_up_target_doc_id=(
                None if same_class_tradeoff is None else same_class_tradeoff.runner_up_target_doc_id
            ),
            same_class_runner_up_policy_ids=(
                () if same_class_tradeoff is None else same_class_tradeoff.runner_up_policy_ids
            ),
            same_class_runner_up_utility_score=(
                None if same_class_tradeoff is None else same_class_tradeoff.runner_up_utility_score
            ),
            same_class_runner_up_utility_reason=(
                None if same_class_tradeoff is None else same_class_tradeoff.runner_up_utility_reason
            ),
            same_class_margin_score=(
                None if same_class_tradeoff is None else same_class_tradeoff.margin_score
            ),
            same_class_margin_reason=(
                None if same_class_tradeoff is None else same_class_tradeoff.margin_reason
            ),
            same_class_margin_components=(
                () if same_class_tradeoff is None else same_class_tradeoff.margin_components
            ),
            cross_class_runner_up_followup_family=(
                None if cross_class_tradeoff is None else cross_class_tradeoff.runner_up_followup_family
            ),
            cross_class_runner_up_followup_class=(
                None if cross_class_tradeoff is None else cross_class_tradeoff.runner_up_followup_class
            ),
            cross_class_runner_up_action_kind=(
                None if cross_class_tradeoff is None else cross_class_tradeoff.runner_up_action_kind
            ),
            cross_class_runner_up_object_id=(
                None if cross_class_tradeoff is None else cross_class_tradeoff.runner_up_object_id
            ),
            cross_class_runner_up_diagnostic_code=(
                None if cross_class_tradeoff is None else cross_class_tradeoff.runner_up_diagnostic_code
            ),
            cross_class_runner_up_target_doc_id=(
                None if cross_class_tradeoff is None else cross_class_tradeoff.runner_up_target_doc_id
            ),
            cross_class_runner_up_utility_score=(
                None if cross_class_tradeoff is None else cross_class_tradeoff.runner_up_utility_score
            ),
            cross_class_runner_up_utility_reason=(
                None if cross_class_tradeoff is None else cross_class_tradeoff.runner_up_utility_reason
            ),
            cross_class_margin_score=(
                None if cross_class_tradeoff is None else cross_class_tradeoff.margin_score
            ),
            cross_class_margin_reason=(
                None if cross_class_tradeoff is None else cross_class_tradeoff.margin_reason
            ),
            cross_class_margin_components=(
                () if cross_class_tradeoff is None else cross_class_tradeoff.margin_components
            ),
            recommendation_rationale_kind=(
                f"same_class_{same_class_strength}__cross_class_{cross_class_strength}"
            ),
            recommendation_rationale_reason=(
                f"same_class_margin:{same_class_strength}:"
                f"{None if same_class_tradeoff is None else same_class_tradeoff.margin_score}"
                f"|cross_class_margin:{cross_class_strength}:"
                f"{None if cross_class_tradeoff is None else cross_class_tradeoff.margin_score}"
            ),
            recommendation_rationale_components=(
                InvariantScoreComponent(
                    kind="same_class_margin_strength",
                    score=(
                        0 if same_class_tradeoff is None else same_class_tradeoff.margin_score
                    ),
                    rationale=same_class_strength,
                ),
                InvariantScoreComponent(
                    kind="cross_class_margin_strength",
                    score=(
                        0
                        if cross_class_tradeoff is None
                        else cross_class_tradeoff.margin_score
                    ),
                    rationale=cross_class_strength,
                ),
            ),
        )

    def recommended_repo_followup_frontier_tradeoff(
        self,
    ) -> InvariantRepoFrontierTradeoff | None:
        return self._recommended_repo_followup_frontier_tradeoff

    @cached_property
    def _recommended_repo_followup_frontier_tradeoff(
        self,
    ) -> InvariantRepoFrontierTradeoff | None:
        lanes = self.repo_followup_lanes()
        if len(lanes) < 2:
            return None
        frontier_lane = lanes[0]
        runner_up_lane = lanes[1]
        return InvariantRepoFrontierTradeoff(
            frontier_followup_family=frontier_lane.followup_family,
            frontier_followup_class=frontier_lane.followup_class,
            runner_up_followup_family=runner_up_lane.followup_family,
            runner_up_followup_class=runner_up_lane.followup_class,
            frontier_lane_utility_score=frontier_lane.lane_utility_score,
            frontier_lane_utility_reason=frontier_lane.lane_utility_reason,
            runner_up_lane_utility_score=runner_up_lane.lane_utility_score,
            runner_up_lane_utility_reason=runner_up_lane.lane_utility_reason,
            margin_score=max(
                0,
                frontier_lane.lane_utility_score - runner_up_lane.lane_utility_score,
            ),
            margin_reason=(
                f"{frontier_lane.lane_utility_reason}"
                f"->{runner_up_lane.lane_utility_reason}"
            ),
            margin_components=runner_up_lane.opportunity_cost_components,
        )

    def repo_followup_lanes(self) -> tuple[InvariantRepoFollowupLane, ...]:
        return self._repo_followup_lanes

    @cached_property
    def _repo_followup_lanes(self) -> tuple[InvariantRepoFollowupLane, ...]:
        grouped: defaultdict[tuple[str, str], list[InvariantRepoFollowupAction]] = defaultdict(list)
        for followup in self.ranked_repo_followups():
            followup_class = self._repo_followup_class(followup)
            grouped[(followup_class, followup.followup_family)].append(followup)
        lanes: list[InvariantRepoFollowupLane] = []
        for (followup_class, followup_family), grouped_items in grouped.items():
            items = tuple(grouped_items)
            lane_utility_score, lane_utility_reason, lane_utility_components = (
                self._repo_followup_lane_utility(
                    followup_class=followup_class,
                    followup_family=followup_family,
                    items=items,
                )
            )
            lanes.append(
                InvariantRepoFollowupLane(
                    followup_family=followup_family,
                    followup_class=followup_class,
                    action_count=len(items),
                    strongest_owner_resolution_kind=items[0].owner_resolution_kind,
                    strongest_owner_resolution_score=items[0].owner_resolution_score,
                    strongest_utility_score=items[0].utility_score,
                    strongest_utility_reason=items[0].utility_reason,
                    lane_utility_score=lane_utility_score,
                    lane_utility_reason=lane_utility_reason,
                    lane_utility_components=lane_utility_components,
                    selection_rank=0,
                    opportunity_cost_score=0,
                    opportunity_cost_reason="",
                    opportunity_cost_components=(),
                    best_followup=items[0],
                )
            )
        ranked_lanes = _sorted(
            lanes,
            key=lambda item: (
                -item.lane_utility_score,
                item.best_followup.priority_rank,
                item.action_count,
                item.followup_class,
                item.followup_family,
            ),
        )
        frontier_lane = ranked_lanes[0] if ranked_lanes else None
        top_score = frontier_lane.lane_utility_score if frontier_lane is not None else 0
        return tuple(
            replace(
                lane,
                selection_rank=index,
                opportunity_cost_score=max(0, top_score - lane.lane_utility_score),
                opportunity_cost_reason=(
                    "frontier"
                    if index == 1
                    else (
                        f"deferred_by:{ranked_lanes[0].followup_family}"
                        if ranked_lanes
                        else "none"
                    )
                ),
                opportunity_cost_components=(
                    ()
                    if frontier_lane is None
                    else self._repo_followup_lane_opportunity_components(
                        frontier_lane=frontier_lane,
                        lane=lane,
                    )
                ),
            )
            for index, lane in enumerate(ranked_lanes, start=1)
        )

    def recommended_repo_code_followup(self) -> InvariantRepoFollowupAction | None:
        for followup in self.ranked_repo_followups():
            if self._repo_followup_class(followup) == "code":
                return followup
        return None

    def recommended_repo_human_followup(self) -> InvariantRepoFollowupAction | None:
        for followup in self.ranked_repo_followups():
            if self._repo_followup_class(followup) != "code":
                return followup
        return None

    def dominant_repo_followup_class(self) -> str:
        recommended = self.recommended_repo_followup()
        if recommended is None:
            return "none"
        return self._repo_followup_class(recommended)

    def next_repo_human_followup_family(self) -> str:
        recommended = self.recommended_repo_human_followup()
        if recommended is None:
            return "none"
        return recommended.followup_family

    def repo_diagnostic_lanes(self) -> tuple[InvariantRepoDiagnosticLane, ...]:
        return self._repo_diagnostic_lanes

    @cached_property
    def _repo_diagnostic_lanes(self) -> tuple[InvariantRepoDiagnosticLane, ...]:
        grouped: defaultdict[
            tuple[str, str, str, str, str, str],
            list[InvariantGraphDiagnostic],
        ] = defaultdict(list)
        for diagnostic in self.diagnostics:
            node = self.node_lookup.get(diagnostic.node_id)
            rel_path = "" if node is None else node.rel_path
            qualname = "" if node is None else node.qualname
            if diagnostic.code == "unmatched_policy_signal":
                title = "" if node is None else node.title
                if not title:
                    title = diagnostic.message.split(" did not resolve", 1)[0].strip() or diagnostic.code
                recommended_action = "attribute_policy_signals_to_owned_workstreams"
            elif diagnostic.code == "unresolved_blocking_dependency":
                title = diagnostic.code
                recommended_action = "resolve_or_reassign_blocking_dependencies"
            else:
                title = diagnostic.message.strip() or diagnostic.code
                recommended_action = "investigate_diagnostic_backlog"
            grouped[
                (
                    diagnostic.code,
                    diagnostic.severity,
                    title,
                    recommended_action,
                    rel_path,
                    qualname,
                )
            ].append(diagnostic)
        lanes = []
        for (
            diagnostic_code,
            severity,
            title,
            _base_recommended_action,
            rel_path,
            qualname,
        ), diagnostics in grouped.items():
            node_ids = tuple(_sorted([item.node_id for item in diagnostics]))
            nodes = [
                self.node_lookup[item.node_id]
                for item in diagnostics
                if item.node_id in self.node_lookup
            ]
            policy_ids = tuple(
                _sorted(
                    list(
                        {
                            policy_id
                            for node in nodes
                            for policy_id in node.policy_ids
                            if policy_id
                        }
                    )
                )
            )
            line = min((node.line for node in nodes if node.line > 0), default=0)
            column = min((node.column for node in nodes if node.column > 0), default=0)
            candidate_owner_object_ids = self._repo_diagnostic_candidate_owner_ids(
                rel_path=rel_path
            )
            candidate_owner_seed_path = self._repo_diagnostic_candidate_owner_seed_path(
                rel_path=rel_path
            )
            candidate_owner_seed_object_id = (
                self._repo_diagnostic_candidate_owner_seed_object_id(
                    seed_path=candidate_owner_seed_path
                )
            )
            candidate_owner_options = self._repo_diagnostic_candidate_owner_options(
                rel_path=rel_path,
                candidate_owner_object_ids=candidate_owner_object_ids,
                candidate_owner_seed_object_id=candidate_owner_seed_object_id,
            )
            candidate_owner_status = self._repo_diagnostic_candidate_owner_status(
                rel_path=rel_path,
                candidate_owner_object_ids=candidate_owner_object_ids,
                candidate_owner_seed_path=candidate_owner_seed_path,
            )
            best_candidate_owner_option = (
                candidate_owner_options[0] if candidate_owner_options else None
            )
            runner_up_candidate_owner_option = (
                candidate_owner_options[1]
                if len(candidate_owner_options) > 1
                else None
            )
            recommended_action = _base_recommended_action
            if diagnostic_code == "unmatched_policy_signal":
                if candidate_owner_status in {"exact_path_owner", "path_family_owner"}:
                    recommended_action = "attach_policy_signals_to_candidate_owner"
                elif candidate_owner_object_ids:
                    recommended_action = "choose_candidate_owner_from_ranked_options"
                elif candidate_owner_seed_path is not None:
                    recommended_action = "seed_owned_workstream_from_source_family"
            lanes.append(
                InvariantRepoDiagnosticLane(
                    diagnostic_code=diagnostic_code,
                    severity=severity,
                    title=title,
                    recommended_action=recommended_action,
                    count=len(diagnostics),
                    node_ids=node_ids,
                    policy_ids=policy_ids,
                    rel_path=rel_path,
                    qualname=qualname,
                    line=line,
                    column=column,
                    candidate_owner_status=candidate_owner_status,
                    candidate_owner_object_id=(
                        candidate_owner_object_ids[0]
                        if len(candidate_owner_object_ids) == 1
                        else None
                    ),
                    candidate_owner_object_ids=candidate_owner_object_ids,
                    candidate_owner_seed_path=candidate_owner_seed_path,
                    candidate_owner_seed_object_id=candidate_owner_seed_object_id,
                    candidate_owner_options=candidate_owner_options,
                    runner_up_candidate_owner_option=runner_up_candidate_owner_option,
                    candidate_owner_choice_margin_score=(
                        None
                        if best_candidate_owner_option is None
                        else (
                            best_candidate_owner_option.score
                            if runner_up_candidate_owner_option is None
                            else max(
                                0,
                                best_candidate_owner_option.score
                                - runner_up_candidate_owner_option.score,
                            )
                        )
                    ),
                    candidate_owner_choice_margin_reason=(
                        None
                        if best_candidate_owner_option is None
                        else (
                            "uncontested_best_option"
                            if runner_up_candidate_owner_option is None
                            else (
                                f"{best_candidate_owner_option.rationale}->"
                                f"{runner_up_candidate_owner_option.rationale}"
                            )
                        )
                    ),
                    candidate_owner_choice_margin_components=(
                        self._owner_choice_margin_components(
                            best_option=best_candidate_owner_option,
                            runner_up_option=runner_up_candidate_owner_option,
                        )
                    ),
                )
            )
        return tuple(
            _sorted(
                lanes,
                key=lambda item: (
                    item.severity,
                    item.diagnostic_code,
                    -item.count,
                    item.rel_path,
                    item.qualname,
                    item.title,
                ),
            )
        )

    def as_payload(self) -> dict[str, object]:
        workstreams = self._workstream_cache
        diagnostic_summary = self.diagnostic_summary()
        recommended_repo_followup = self.recommended_repo_followup()
        recommended_repo_code_followup = self.recommended_repo_code_followup()
        recommended_repo_human_followup = self.recommended_repo_human_followup()
        recommended_repo_followup_lane = self.recommended_repo_followup_lane()
        recommended_repo_code_followup_lane = self.recommended_repo_code_followup_lane()
        recommended_repo_human_followup_lane = self.recommended_repo_human_followup_lane()
        recommended_repo_followup_frontier_tradeoff = (
            self.recommended_repo_followup_frontier_tradeoff()
        )
        recommended_repo_followup_frontier_explanation = (
            self.recommended_repo_followup_frontier_explanation()
        )
        recommended_repo_followup_frontier_triad = (
            self.recommended_repo_followup_frontier_triad()
        )
        recommended_repo_followup_same_class_tradeoff = (
            self.recommended_repo_followup_same_class_tradeoff()
        )
        recommended_repo_followup_cross_class_tradeoff = (
            self.recommended_repo_followup_cross_class_tradeoff()
        )
        ranked_repo_followups = self.ranked_repo_followups()
        repo_followup_lanes = self.repo_followup_lanes()
        repo_diagnostic_lanes = self.repo_diagnostic_lanes()
        return {
            "format_version": _FORMAT_VERSION,
            "generated_at_utc": self.generated_at_utc,
            "root": self.root,
            "workstreams": [item.as_payload() for item in workstreams],
            "diagnostic_summary": diagnostic_summary.as_payload(),
            "repo_next_actions": {
                "dominant_followup_class": self.dominant_repo_followup_class(),
                "next_human_followup_family": self.next_repo_human_followup_family(),
                "recommended_followup": (
                    None
                    if recommended_repo_followup is None
                    else recommended_repo_followup.as_payload()
                ),
                "recommended_code_followup": (
                    None
                    if recommended_repo_code_followup is None
                    else recommended_repo_code_followup.as_payload()
                ),
                "recommended_human_followup": (
                    None
                    if recommended_repo_human_followup is None
                    else recommended_repo_human_followup.as_payload()
                ),
                "recommended_followup_lane": (
                    None
                    if recommended_repo_followup_lane is None
                    else recommended_repo_followup_lane.as_payload()
                ),
                "recommended_code_followup_lane": (
                    None
                    if recommended_repo_code_followup_lane is None
                    else recommended_repo_code_followup_lane.as_payload()
                ),
                "recommended_human_followup_lane": (
                    None
                    if recommended_repo_human_followup_lane is None
                    else recommended_repo_human_followup_lane.as_payload()
                ),
                "recommended_followup_frontier_tradeoff": (
                    None
                    if recommended_repo_followup_frontier_tradeoff is None
                    else recommended_repo_followup_frontier_tradeoff.as_payload()
                ),
                "recommended_followup_frontier_explanation": (
                    None
                    if recommended_repo_followup_frontier_explanation is None
                    else recommended_repo_followup_frontier_explanation.as_payload()
                ),
                "recommended_followup_frontier_triad": (
                    None
                    if recommended_repo_followup_frontier_triad is None
                    else recommended_repo_followup_frontier_triad.as_payload()
                ),
                "recommended_followup_same_class_tradeoff": (
                    None
                    if recommended_repo_followup_same_class_tradeoff is None
                    else recommended_repo_followup_same_class_tradeoff.as_payload()
                ),
                "recommended_followup_cross_class_tradeoff": (
                    None
                    if recommended_repo_followup_cross_class_tradeoff is None
                    else recommended_repo_followup_cross_class_tradeoff.as_payload()
                ),
                "ranked_followups": [
                    item.as_payload() for item in ranked_repo_followups
                ],
                "followup_lanes": [
                    item.as_payload() for item in repo_followup_lanes
                ],
                "diagnostic_lanes": [
                    item.as_payload() for item in repo_diagnostic_lanes
                ],
            },
            "counts": {
                "workstream_count": len(workstreams),
                "diagnostic_count": diagnostic_summary.diagnostic_count,
            },
        }

    def artifact_document(self) -> ArtifactUnit:
        def _workstream_items() -> Iterator[ArtifactUnit]:
            for workstream in self._workstream_cache:
                yield list_item(
                    identity=workstream.object_id,
                    title=workstream.object_id.wire(),
                    children=lambda workstream=workstream: iter(
                        (
                            scalar(
                                identity=workstream.object_id,
                                key="object_id",
                                title="object_id",
                                value=workstream.object_id.wire(),
                            ),
                            scalar(
                                identity=workstream.site_identity,
                                key="title",
                                title="title",
                                value=workstream.title,
                            ),
                            scalar(
                                identity=workstream.object_id,
                                key="status",
                                title="status",
                                value=workstream.status,
                            ),
                            scalar(
                                identity=workstream.site_identity,
                                key="site_identity",
                                title="site_identity",
                                value=workstream.site_identity.wire(),
                            ),
                            scalar(
                                identity=workstream.structural_identity,
                                key="structural_identity",
                                title="structural_identity",
                                value=workstream.structural_identity.wire(),
                            ),
                            scalar(
                                identity=workstream.object_id,
                                key="marker_identity",
                                title="marker_identity",
                                value=workstream.marker_identity,
                            ),
                            scalar(
                                identity=workstream.object_id,
                                key="reasoning_summary",
                                title="reasoning_summary",
                                value=workstream.reasoning_summary,
                            ),
                            scalar(
                                identity=workstream.object_id,
                                key="reasoning_control",
                                title="reasoning_control",
                                value=workstream.reasoning_control,
                            ),
                            bullet_list(
                                identity=workstream.object_id,
                                key="blocking_dependencies",
                                children=lambda workstream=workstream: (
                                    list_item(
                                        identity=ArtifactSourceRef(
                                            rel_path="<synthetic>",
                                            qualname=dependency,
                                        ),
                                        value=dependency,
                                    )
                                    for dependency in workstream.blocking_dependencies
                                ),
                            ),
                            bullet_list(
                                identity=workstream.object_id,
                                key="object_ids",
                                children=lambda workstream=workstream: (
                                    list_item(
                                        identity=ArtifactSourceRef(
                                            rel_path="<synthetic>",
                                            qualname=object_id,
                                        ),
                                        value=object_id,
                                    )
                                    for object_id in workstream.object_ids
                                ),
                            ),
                            bullet_list(
                                identity=workstream.object_id,
                                key="doc_ids",
                                children=lambda workstream=workstream: (
                                    list_item(
                                        identity=ArtifactSourceRef(
                                            rel_path="<synthetic>",
                                            qualname=doc_id,
                                        ),
                                        value=doc_id,
                                    )
                                    for doc_id in workstream.doc_ids
                                ),
                            ),
                            bullet_list(
                                identity=workstream.object_id,
                                key="policy_ids",
                                children=lambda workstream=workstream: (
                                    list_item(
                                        identity=ArtifactSourceRef(
                                            rel_path="<synthetic>",
                                            qualname=policy_id,
                                        ),
                                        value=policy_id,
                                    )
                                    for policy_id in workstream.policy_ids
                                ),
                            ),
                            scalar(identity=workstream.object_id, key="touchsite_count", title="touchsite_count", value=workstream.touchsite_count),
                            scalar(identity=workstream.object_id, key="collapsible_touchsite_count", title="collapsible_touchsite_count", value=workstream.collapsible_touchsite_count),
                            scalar(identity=workstream.object_id, key="surviving_touchsite_count", title="surviving_touchsite_count", value=workstream.surviving_touchsite_count),
                            scalar(identity=workstream.object_id, key="policy_signal_count", title="policy_signal_count", value=workstream.policy_signal_count),
                            scalar(identity=workstream.object_id, key="coverage_count", title="coverage_count", value=workstream.coverage_count),
                            scalar(identity=workstream.object_id, key="diagnostic_count", title="diagnostic_count", value=workstream.diagnostic_count),
                            section(
                                identity=workstream.object_id,
                                key="doc_alignment_summary",
                                title="doc_alignment_summary",
                                children=lambda workstream=workstream: iter(
                                    ()
                                    if workstream.doc_alignment_summary is None
                                    else _payload_to_units(
                                        workstream.doc_alignment_summary.as_payload()
                                    )
                                ),
                            ),
                            section(
                                identity=workstream.object_id,
                                key="health_summary",
                                title="health_summary",
                                children=lambda workstream=workstream: iter(
                                    _payload_to_units(workstream.health_summary().as_payload())
                                ),
                            ),
                            bullet_list(
                                identity=workstream.object_id,
                                key="subqueues",
                                children=lambda workstream=workstream: (
                                    list_item(
                                        identity=subqueue.object_id,
                                        children=lambda subqueue=subqueue: iter(_payload_to_units(subqueue.as_payload())),
                                    )
                                    for subqueue in workstream.iter_subqueues()
                                ),
                            ),
                            bullet_list(
                                identity=workstream.object_id,
                                key="touchpoints",
                                children=lambda workstream=workstream: (
                                    list_item(
                                        identity=touchpoint.object_id,
                                        children=lambda touchpoint=touchpoint: iter(_payload_to_units(touchpoint.as_payload())),
                                    )
                                    for touchpoint in workstream.iter_touchpoints()
                                ),
                            ),
                            section(
                                identity=workstream.object_id,
                                key="next_actions",
                                title="next_actions",
                                children=lambda workstream=workstream: iter(
                                    (
                                        scalar(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="dominant_blocker_class",
                                            ),
                                            key="dominant_blocker_class",
                                            title="dominant_blocker_class",
                                            value=workstream.dominant_blocker_class(),
                                        ),
                                        scalar(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="recommended_remediation_family",
                                            ),
                                            key="recommended_remediation_family",
                                            title="recommended_remediation_family",
                                            value=workstream.recommended_remediation_family(),
                                        ),
                                        scalar(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="dominant_doc_alignment_status",
                                            ),
                                            key="dominant_doc_alignment_status",
                                            title="dominant_doc_alignment_status",
                                            value=workstream.dominant_doc_alignment_status(),
                                        ),
                                        scalar(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="recommended_doc_alignment_action",
                                            ),
                                            key="recommended_doc_alignment_action",
                                            title="recommended_doc_alignment_action",
                                            value=workstream.recommended_doc_alignment_action(),
                                        ),
                                        scalar(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="next_human_followup_family",
                                            ),
                                            key="next_human_followup_family",
                                            title="next_human_followup_family",
                                            value=workstream.next_human_followup_family(),
                                        ),
                                        scalar(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="recommended_doc_followup_target_doc_id",
                                            ),
                                            key="recommended_doc_followup_target_doc_id",
                                            title="recommended_doc_followup_target_doc_id",
                                            value=workstream.recommended_doc_followup_target_doc_id(),
                                        ),
                                        scalar(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="recommended_followup",
                                            ),
                                            key="recommended_followup",
                                            title="recommended_followup",
                                            value=(
                                                None
                                                if workstream.recommended_followup() is None
                                                else workstream.recommended_followup().as_payload()
                                            ),
                                        ),
                                        bullet_list(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="misaligned_target_doc_ids",
                                            ),
                                            key="misaligned_target_doc_ids",
                                            children=lambda workstream=workstream: (
                                                list_item(
                                                    identity=ArtifactSourceRef(
                                                        rel_path="<synthetic>",
                                                        qualname=doc_id,
                                                    ),
                                                    value=doc_id,
                                                )
                                                for doc_id in workstream.misaligned_target_doc_ids()
                                            ),
                                        ),
                                        scalar(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="recommended_cut",
                                            ),
                                            key="recommended_cut",
                                            title="recommended_cut",
                                            value=(
                                                None
                                                if workstream.recommended_cut() is None
                                                else workstream.recommended_cut().as_payload()
                                            ),
                                        ),
                                        scalar(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="recommended_ready_cut",
                                            ),
                                            key="recommended_ready_cut",
                                            title="recommended_ready_cut",
                                            value=(
                                                None
                                                if workstream.recommended_ready_cut() is None
                                                else workstream.recommended_ready_cut().as_payload()
                                            ),
                                        ),
                                        scalar(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="recommended_coverage_gap_cut",
                                            ),
                                            key="recommended_coverage_gap_cut",
                                            title="recommended_coverage_gap_cut",
                                            value=(
                                                None
                                                if workstream.recommended_coverage_gap_cut() is None
                                                else workstream.recommended_coverage_gap_cut().as_payload()
                                            ),
                                        ),
                                        scalar(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="recommended_policy_blocked_cut",
                                            ),
                                            key="recommended_policy_blocked_cut",
                                            title="recommended_policy_blocked_cut",
                                            value=(
                                                None
                                                if workstream.recommended_policy_blocked_cut() is None
                                                else workstream.recommended_policy_blocked_cut().as_payload()
                                            ),
                                        ),
                                        scalar(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="recommended_diagnostic_blocked_cut",
                                            ),
                                            key="recommended_diagnostic_blocked_cut",
                                            title="recommended_diagnostic_blocked_cut",
                                            value=(
                                                None
                                                if workstream.recommended_diagnostic_blocked_cut() is None
                                                else workstream.recommended_diagnostic_blocked_cut().as_payload()
                                            ),
                                        ),
                                        bullet_list(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="ranked_followups",
                                            ),
                                            key="ranked_followups",
                                            children=lambda workstream=workstream: (
                                                list_item(
                                                    identity=ArtifactSourceRef(
                                                        rel_path="<synthetic>",
                                                        qualname=(
                                                            item.object_id
                                                            or item.target_doc_id
                                                            or item.followup_family
                                                        ),
                                                    ),
                                                    children=lambda item=item: iter(
                                                        _payload_to_units(item.as_payload())
                                                    ),
                                                )
                                                for item in workstream.ranked_followups()
                                            ),
                                        ),
                                        bullet_list(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="remediation_lanes",
                                            ),
                                            key="remediation_lanes",
                                            children=lambda workstream=workstream: (
                                                list_item(
                                                    identity=ArtifactSourceRef(
                                                        rel_path="<synthetic>",
                                                        qualname=item.remediation_family,
                                                    ),
                                                    children=lambda item=item: iter(
                                                        _payload_to_units(item.as_payload())
                                                    ),
                                                )
                                                for item in workstream.remediation_lanes()
                                            ),
                                        ),
                                        bullet_list(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="documentation_followup_lanes",
                                            ),
                                            key="documentation_followup_lanes",
                                            children=lambda workstream=workstream: iter(
                                                ()
                                                if workstream.documentation_followup_lane() is None
                                                else (
                                                    list_item(
                                                        identity=ArtifactSourceRef(
                                                            rel_path="<synthetic>",
                                                            qualname=workstream.documentation_followup_lane().followup_family,
                                                        ),
                                                        children=lambda workstream=workstream: iter(
                                                            _payload_to_units(
                                                                workstream.documentation_followup_lane().as_payload()
                                                            )
                                                        ),
                                                    ),
                                                )
                                            ),
                                        ),
                                        bullet_list(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="ranked_touchpoint_cuts",
                                            ),
                                            key="ranked_touchpoint_cuts",
                                            children=lambda workstream=workstream: (
                                                list_item(
                                                    identity=item.object_id,
                                                    children=lambda item=item: iter(
                                                        _payload_to_units(item.as_payload())
                                                    ),
                                                )
                                                for item in workstream.ranked_touchpoint_cuts()
                                            ),
                                        ),
                                        bullet_list(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="ranked_subqueue_cuts",
                                            ),
                                            key="ranked_subqueue_cuts",
                                            children=lambda workstream=workstream: (
                                                list_item(
                                                    identity=item.object_id,
                                                    children=lambda item=item: iter(
                                                        _payload_to_units(item.as_payload())
                                                    ),
                                                )
                                                for item in workstream.ranked_subqueue_cuts()
                                            ),
                                        ),
                                    )
                                ),
                            ),
                        )
                    ),
                )

        return document(
            identity=ArtifactSourceRef(rel_path="<synthetic>", qualname="invariant_workstreams"),
            children=lambda: iter(
                (
                    scalar(
                        identity=ArtifactSourceRef(rel_path="<synthetic>", qualname="format_version"),
                        key="format_version",
                        title="format_version",
                        value=_FORMAT_VERSION,
                    ),
                    scalar(
                        identity=ArtifactSourceRef(rel_path="<synthetic>", qualname="generated_at_utc"),
                        key="generated_at_utc",
                        title="generated_at_utc",
                        value=self.generated_at_utc,
                    ),
                    scalar(
                        identity=ArtifactSourceRef(rel_path="<synthetic>", qualname="root"),
                        key="root",
                        title="root",
                        value=self.root,
                    ),
                    bullet_list(
                        identity=ArtifactSourceRef(rel_path="<synthetic>", qualname="workstreams"),
                        key="workstreams",
                        children=_workstream_items,
                    ),
                    section(
                        identity=ArtifactSourceRef(
                            rel_path="<synthetic>",
                            qualname="diagnostic_summary",
                        ),
                        key="diagnostic_summary",
                        title="diagnostic_summary",
                        children=lambda self=self: iter(
                            _payload_to_units(self.diagnostic_summary().as_payload())
                        ),
                    ),
                    section(
                        identity=ArtifactSourceRef(
                            rel_path="<synthetic>",
                            qualname="repo_next_actions",
                        ),
                        key="repo_next_actions",
                        title="repo_next_actions",
                        children=lambda self=self: iter(
                            (
                                scalar(
                                    identity=ArtifactSourceRef(
                                        rel_path="<synthetic>",
                                        qualname="dominant_followup_class",
                                    ),
                                    key="dominant_followup_class",
                                    title="dominant_followup_class",
                                    value=self.dominant_repo_followup_class(),
                                ),
                                scalar(
                                    identity=ArtifactSourceRef(
                                        rel_path="<synthetic>",
                                        qualname="next_human_followup_family",
                                    ),
                                    key="next_human_followup_family",
                                    title="next_human_followup_family",
                                    value=self.next_repo_human_followup_family(),
                                ),
                                scalar(
                                    identity=ArtifactSourceRef(
                                        rel_path="<synthetic>",
                                        qualname="recommended_followup",
                                    ),
                                    key="recommended_followup",
                                    title="recommended_followup",
                                    value=(
                                        None
                                        if self.recommended_repo_followup() is None
                                        else self.recommended_repo_followup().as_payload()
                                    ),
                                ),
                                scalar(
                                    identity=ArtifactSourceRef(
                                        rel_path="<synthetic>",
                                        qualname="recommended_code_followup",
                                    ),
                                    key="recommended_code_followup",
                                    title="recommended_code_followup",
                                    value=(
                                        None
                                        if self.recommended_repo_code_followup() is None
                                        else self.recommended_repo_code_followup().as_payload()
                                    ),
                                ),
                                scalar(
                                    identity=ArtifactSourceRef(
                                        rel_path="<synthetic>",
                                        qualname="recommended_human_followup",
                                    ),
                                    key="recommended_human_followup",
                                    title="recommended_human_followup",
                                    value=(
                                        None
                                        if self.recommended_repo_human_followup() is None
                                        else self.recommended_repo_human_followup().as_payload()
                                    ),
                                ),
                                scalar(
                                    identity=ArtifactSourceRef(
                                        rel_path="<synthetic>",
                                        qualname="recommended_followup_lane",
                                    ),
                                    key="recommended_followup_lane",
                                    title="recommended_followup_lane",
                                    value=(
                                        None
                                        if self.recommended_repo_followup_lane() is None
                                        else self.recommended_repo_followup_lane().as_payload()
                                    ),
                                ),
                                scalar(
                                    identity=ArtifactSourceRef(
                                        rel_path="<synthetic>",
                                        qualname="recommended_code_followup_lane",
                                    ),
                                    key="recommended_code_followup_lane",
                                    title="recommended_code_followup_lane",
                                    value=(
                                        None
                                        if self.recommended_repo_code_followup_lane() is None
                                        else self.recommended_repo_code_followup_lane().as_payload()
                                    ),
                                ),
                                scalar(
                                    identity=ArtifactSourceRef(
                                        rel_path="<synthetic>",
                                        qualname="recommended_human_followup_lane",
                                    ),
                                    key="recommended_human_followup_lane",
                                    title="recommended_human_followup_lane",
                                    value=(
                                        None
                                        if self.recommended_repo_human_followup_lane() is None
                                        else self.recommended_repo_human_followup_lane().as_payload()
                                    ),
                                ),
                                bullet_list(
                                    identity=ArtifactSourceRef(
                                        rel_path="<synthetic>",
                                        qualname="ranked_followups",
                                    ),
                                    key="ranked_followups",
                                    children=lambda self=self: (
                                        list_item(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname=(
                                                    item.object_id
                                                    or item.target_doc_id
                                                    or item.diagnostic_code
                                                    or item.followup_family
                                                ),
                                            ),
                                            children=lambda item=item: iter(
                                                _payload_to_units(item.as_payload())
                                            ),
                                        )
                                        for item in self.ranked_repo_followups()
                                    ),
                                ),
                                bullet_list(
                                    identity=ArtifactSourceRef(
                                        rel_path="<synthetic>",
                                        qualname="followup_lanes",
                                    ),
                                    key="followup_lanes",
                                    children=lambda self=self: (
                                        list_item(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname=item.followup_family,
                                            ),
                                            children=lambda item=item: iter(
                                                _payload_to_units(item.as_payload())
                                            ),
                                        )
                                        for item in self.repo_followup_lanes()
                                    ),
                                ),
                                bullet_list(
                                    identity=ArtifactSourceRef(
                                        rel_path="<synthetic>",
                                        qualname="diagnostic_lanes",
                                    ),
                                    key="diagnostic_lanes",
                                    children=lambda self=self: (
                                        list_item(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname=item.title,
                                            ),
                                            children=lambda item=item: iter(
                                                _payload_to_units(item.as_payload())
                                            ),
                                        )
                                        for item in self.repo_diagnostic_lanes()
                                    ),
                                ),
                            )
                        ),
                    ),
                    section(
                        identity=ArtifactSourceRef(rel_path="<synthetic>", qualname="counts"),
                        key="counts",
                        title="counts",
                        children=lambda self=self: iter(
                            (
                                scalar(
                                    identity=ArtifactSourceRef(
                                        rel_path="<synthetic>",
                                        qualname="workstream_count",
                                    ),
                                    key="workstream_count",
                                    title="workstream_count",
                                    value=len(self._workstream_cache),
                                ),
                                scalar(
                                    identity=ArtifactSourceRef(
                                        rel_path="<synthetic>",
                                        qualname="diagnostic_count",
                                    ),
                                    key="diagnostic_count",
                                    title="diagnostic_count",
                                    value=self.diagnostic_summary().diagnostic_count,
                                ),
                            )
                        ),
                    ),
                )
            ),
        )


@dataclass(frozen=True)
class InvariantLedgerTargetDocAlignment:
    target_doc_id: str
    target_doc_path: str
    alignment_status: str
    object_reference_present: bool
    summary_present: bool

    def as_payload(self) -> dict[str, object]:
        return {
            "target_doc_id": self.target_doc_id,
            "target_doc_path": self.target_doc_path,
            "alignment_status": self.alignment_status,
            "object_reference_present": self.object_reference_present,
            "summary_present": self.summary_present,
        }


@dataclass(frozen=True)
class InvariantLedgerAlignmentSummary:
    target_doc_count: int
    reflected_target_doc_count: int
    append_pending_existing_target_doc_count: int
    append_pending_new_target_doc_count: int
    missing_target_doc_count: int
    ambiguous_target_doc_count: int
    unassigned_target_doc_count: int
    dominant_alignment_status: str
    recommended_doc_alignment_action: str
    misaligned_target_doc_ids: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "target_doc_count": self.target_doc_count,
            "reflected_target_doc_count": self.reflected_target_doc_count,
            "append_pending_existing_target_doc_count": self.append_pending_existing_target_doc_count,
            "append_pending_new_target_doc_count": self.append_pending_new_target_doc_count,
            "missing_target_doc_count": self.missing_target_doc_count,
            "ambiguous_target_doc_count": self.ambiguous_target_doc_count,
            "unassigned_target_doc_count": self.unassigned_target_doc_count,
            "dominant_alignment_status": self.dominant_alignment_status,
            "recommended_doc_alignment_action": self.recommended_doc_alignment_action,
            "misaligned_target_doc_ids": list(self.misaligned_target_doc_ids),
        }


@dataclass(frozen=True)
class InvariantLedgerProjection:
    object_id: str
    title: str
    status: str
    target_doc_ids: tuple[str, ...]
    target_policy_ids: tuple[str, ...]
    dominant_blocker_class: str
    recommended_remediation_family: str
    recommended_ledger_action: str
    summary: str
    current_snapshot: Mapping[str, object]
    target_doc_alignments: tuple[InvariantLedgerTargetDocAlignment, ...] = ()
    alignment_summary: InvariantLedgerAlignmentSummary | None = None

    def as_payload(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "title": self.title,
            "status": self.status,
            "target_doc_ids": list(self.target_doc_ids),
            "target_policy_ids": list(self.target_policy_ids),
            "dominant_blocker_class": self.dominant_blocker_class,
            "recommended_remediation_family": self.recommended_remediation_family,
            "recommended_ledger_action": self.recommended_ledger_action,
            "summary": self.summary,
            "current_snapshot": dict(self.current_snapshot),
            "target_doc_alignments": [
                item.as_payload() for item in self.target_doc_alignments
            ],
            "alignment_summary": (
                None if self.alignment_summary is None else self.alignment_summary.as_payload()
            ),
        }


@dataclass(frozen=True)
class InvariantLedgerProjections:
    root: str
    generated_at_utc: str
    ledgers: ReplayableStream[InvariantLedgerProjection]

    def iter_ledgers(self) -> Iterator[InvariantLedgerProjection]:
        return iter(self.ledgers)

    def as_payload(self) -> dict[str, object]:
        ledgers = tuple(self.iter_ledgers())
        return {
            "format_version": _FORMAT_VERSION,
            "generated_at_utc": self.generated_at_utc,
            "root": self.root,
            "ledgers": [item.as_payload() for item in ledgers],
            "counts": {
                "ledger_count": len(ledgers),
            },
        }

    def artifact_document(self) -> ArtifactUnit:
        ledgers = tuple(self.iter_ledgers())

        def _ledger_items() -> Iterator[ArtifactUnit]:
            for ledger in ledgers:
                yield list_item(
                    identity=ArtifactSourceRef(rel_path="<synthetic>", qualname=ledger.object_id),
                    title=ledger.object_id,
                    children=lambda ledger=ledger: iter(_payload_to_units(ledger.as_payload())),
                )

        return document(
            identity=ArtifactSourceRef(
                rel_path="<synthetic>",
                qualname="invariant_ledger_projections",
            ),
            children=lambda: iter(
                (
                    scalar(
                        identity=ArtifactSourceRef(
                            rel_path="<synthetic>",
                            qualname="format_version",
                        ),
                        key="format_version",
                        title="format_version",
                        value=_FORMAT_VERSION,
                    ),
                    scalar(
                        identity=ArtifactSourceRef(
                            rel_path="<synthetic>",
                            qualname="generated_at_utc",
                        ),
                        key="generated_at_utc",
                        title="generated_at_utc",
                        value=self.generated_at_utc,
                    ),
                    scalar(
                        identity=ArtifactSourceRef(rel_path="<synthetic>", qualname="root"),
                        key="root",
                        title="root",
                        value=self.root,
                    ),
                    bullet_list(
                        identity=ArtifactSourceRef(
                            rel_path="<synthetic>",
                            qualname="ledgers",
                        ),
                        key="ledgers",
                        children=_ledger_items,
                    ),
                    section(
                        identity=ArtifactSourceRef(
                            rel_path="<synthetic>",
                            qualname="counts",
                        ),
                        key="counts",
                        title="counts",
                        children=lambda: iter(
                            (
                                scalar(
                                    identity=ArtifactSourceRef(
                                        rel_path="<synthetic>",
                                        qualname="ledger_count",
                                    ),
                                    key="ledger_count",
                                    title="ledger_count",
                                    value=len(ledgers),
                                ),
                            )
                        ),
                    ),
                )
            ),
        )


@dataclass(frozen=True)
class InvariantLedgerDelta:
    object_id: str
    title: str
    target_doc_ids: tuple[str, ...]
    target_policy_ids: tuple[str, ...]
    classification: str
    recommended_ledger_action: str
    summary: str
    before_status: str
    after_status: str
    before_recommended_cut_object_id: str | None
    after_recommended_cut_object_id: str | None

    def as_payload(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "title": self.title,
            "target_doc_ids": list(self.target_doc_ids),
            "target_policy_ids": list(self.target_policy_ids),
            "classification": self.classification,
            "recommended_ledger_action": self.recommended_ledger_action,
            "summary": self.summary,
            "before_status": self.before_status,
            "after_status": self.after_status,
            "before_recommended_cut_object_id": self.before_recommended_cut_object_id,
            "after_recommended_cut_object_id": self.after_recommended_cut_object_id,
            "append_entry": {
                "object_id": self.object_id,
                "title": self.title,
                "classification": self.classification,
                "recommended_ledger_action": self.recommended_ledger_action,
                "summary": self.summary,
                "before_status": self.before_status,
                "after_status": self.after_status,
                "before_recommended_cut_object_id": self.before_recommended_cut_object_id,
                "after_recommended_cut_object_id": self.after_recommended_cut_object_id,
                "target_policy_ids": list(self.target_policy_ids),
            },
        }


@dataclass(frozen=True)
class InvariantLedgerDeltaProjections:
    root: str
    generated_at_utc: str
    before_workstreams_artifact: str
    after_workstreams_artifact: str
    deltas: ReplayableStream[InvariantLedgerDelta]

    def iter_deltas(self) -> Iterator[InvariantLedgerDelta]:
        return iter(self.deltas)

    def as_payload(self) -> dict[str, object]:
        deltas = tuple(self.iter_deltas())
        classification_counts: dict[str, int] = defaultdict(int)
        target_doc_ids: set[str] = set()
        for item in deltas:
            classification_counts[item.classification] += 1
            target_doc_ids.update(item.target_doc_ids)
        return {
            "format_version": _FORMAT_VERSION,
            "generated_at_utc": self.generated_at_utc,
            "root": self.root,
            "before_workstreams_artifact": self.before_workstreams_artifact,
            "after_workstreams_artifact": self.after_workstreams_artifact,
            "deltas": [item.as_payload() for item in deltas],
            "counts": {
                "delta_count": len(deltas),
                "classification_counts": dict(
                    _sorted(list(classification_counts.items()), key=lambda item: item[0])
                ),
                "target_doc_count": len(target_doc_ids),
            },
        }

    def artifact_document(self) -> ArtifactUnit:
        deltas = tuple(self.iter_deltas())

        def _delta_items() -> Iterator[ArtifactUnit]:
            for delta in deltas:
                yield list_item(
                    identity=ArtifactSourceRef(rel_path="<synthetic>", qualname=delta.object_id),
                    title=delta.object_id,
                    children=lambda delta=delta: iter(_payload_to_units(delta.as_payload())),
                )

        return document(
            identity=ArtifactSourceRef(
                rel_path="<synthetic>",
                qualname="invariant_ledger_delta_projections",
            ),
            children=lambda: iter(
                (
                    scalar(
                        identity=ArtifactSourceRef(
                            rel_path="<synthetic>",
                            qualname="format_version",
                        ),
                        key="format_version",
                        title="format_version",
                        value=_FORMAT_VERSION,
                    ),
                    scalar(
                        identity=ArtifactSourceRef(
                            rel_path="<synthetic>",
                            qualname="generated_at_utc",
                        ),
                        key="generated_at_utc",
                        title="generated_at_utc",
                        value=self.generated_at_utc,
                    ),
                    scalar(
                        identity=ArtifactSourceRef(rel_path="<synthetic>", qualname="root"),
                        key="root",
                        title="root",
                        value=self.root,
                    ),
                    scalar(
                        identity=ArtifactSourceRef(
                            rel_path="<synthetic>",
                            qualname="before_workstreams_artifact",
                        ),
                        key="before_workstreams_artifact",
                        title="before_workstreams_artifact",
                        value=self.before_workstreams_artifact,
                    ),
                    scalar(
                        identity=ArtifactSourceRef(
                            rel_path="<synthetic>",
                            qualname="after_workstreams_artifact",
                        ),
                        key="after_workstreams_artifact",
                        title="after_workstreams_artifact",
                        value=self.after_workstreams_artifact,
                    ),
                    bullet_list(
                        identity=ArtifactSourceRef(
                            rel_path="<synthetic>",
                            qualname="deltas",
                        ),
                        key="deltas",
                        children=_delta_items,
                    ),
                    section(
                        identity=ArtifactSourceRef(
                            rel_path="<synthetic>",
                            qualname="counts",
                        ),
                        key="counts",
                        title="counts",
                        children=lambda: iter(
                            (
                                scalar(
                                    identity=ArtifactSourceRef(
                                        rel_path="<synthetic>",
                                        qualname="delta_count",
                                    ),
                                    key="delta_count",
                                    title="delta_count",
                                    value=len(deltas),
                                ),
                            )
                        ),
                    ),
                )
            ),
        )

    def markdown_document(self) -> ArtifactUnit:
        grouped = self.grouped_by_target_doc_id()

        def _doc_sections() -> Iterator[ArtifactUnit]:
            for doc_id, deltas in grouped:
                yield section(
                    identity=ArtifactSourceRef(rel_path="<synthetic>", qualname=doc_id),
                    key=doc_id,
                    title=doc_id,
                    children=lambda deltas=deltas: _ledger_delta_markdown_units(
                        deltas=deltas
                    ),
                )

        return document(
            identity=ArtifactSourceRef(
                rel_path="<synthetic>",
                qualname="invariant_ledger_delta_markdown",
            ),
            title="Invariant Ledger Deltas",
            children=_doc_sections,
        )

    def grouped_by_target_doc_id(self) -> tuple[tuple[str, tuple[InvariantLedgerDelta, ...]], ...]:
        grouped: defaultdict[str, list[InvariantLedgerDelta]] = defaultdict(list)
        for delta in self.iter_deltas():
            target_doc_ids = delta.target_doc_ids or ("<unassigned>",)
            for doc_id in target_doc_ids:
                grouped[doc_id].append(delta)
        return tuple(
            (
                doc_id,
                tuple(_sorted(deltas, key=lambda item: (item.object_id, item.classification))),
            )
            for doc_id, deltas in _sorted(list(grouped.items()), key=lambda item: item[0])
        )


@dataclass(frozen=True)
class InvariantLedgerAlignment:
    object_id: str
    title: str
    target_doc_id: str
    target_doc_path: str
    classification: str
    recommended_ledger_action: str
    alignment_status: str
    summary: str
    object_reference_present: bool
    summary_present: bool

    def as_payload(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "title": self.title,
            "target_doc_id": self.target_doc_id,
            "target_doc_path": self.target_doc_path,
            "classification": self.classification,
            "recommended_ledger_action": self.recommended_ledger_action,
            "alignment_status": self.alignment_status,
            "summary": self.summary,
            "object_reference_present": self.object_reference_present,
            "summary_present": self.summary_present,
        }


@dataclass(frozen=True)
class InvariantLedgerAlignments:
    root: str
    generated_at_utc: str
    alignments: ReplayableStream[InvariantLedgerAlignment]

    def iter_alignments(self) -> Iterator[InvariantLedgerAlignment]:
        return iter(self.alignments)

    def as_payload(self) -> dict[str, object]:
        alignments = tuple(self.iter_alignments())
        status_counts: dict[str, int] = defaultdict(int)
        for item in alignments:
            status_counts[item.alignment_status] += 1
        return {
            "format_version": _FORMAT_VERSION,
            "generated_at_utc": self.generated_at_utc,
            "root": self.root,
            "alignments": [item.as_payload() for item in alignments],
            "counts": {
                "alignment_count": len(alignments),
                "status_counts": dict(
                    _sorted(list(status_counts.items()), key=lambda item: item[0])
                ),
            },
        }

    def artifact_document(self) -> ArtifactUnit:
        alignments = tuple(self.iter_alignments())
        status_counts: dict[str, int] = defaultdict(int)
        for item in alignments:
            status_counts[item.alignment_status] += 1

        def _alignment_items() -> Iterator[ArtifactUnit]:
            for alignment in alignments:
                yield list_item(
                    identity=ArtifactSourceRef(
                        rel_path="<synthetic>",
                        qualname=f"{alignment.target_doc_id}:{alignment.object_id}",
                    ),
                    title=f"{alignment.target_doc_id} :: {alignment.object_id}",
                    children=lambda alignment=alignment: iter(
                        _payload_to_units(alignment.as_payload())
                    ),
                )

        return document(
            identity=ArtifactSourceRef(
                rel_path="<synthetic>",
                qualname="invariant_ledger_alignments",
            ),
            children=lambda: iter(
                (
                    scalar(
                        identity=ArtifactSourceRef(
                            rel_path="<synthetic>",
                            qualname="format_version",
                        ),
                        key="format_version",
                        title="format_version",
                        value=_FORMAT_VERSION,
                    ),
                    scalar(
                        identity=ArtifactSourceRef(
                            rel_path="<synthetic>",
                            qualname="generated_at_utc",
                        ),
                        key="generated_at_utc",
                        title="generated_at_utc",
                        value=self.generated_at_utc,
                    ),
                    scalar(
                        identity=ArtifactSourceRef(rel_path="<synthetic>", qualname="root"),
                        key="root",
                        title="root",
                        value=self.root,
                    ),
                    bullet_list(
                        identity=ArtifactSourceRef(
                            rel_path="<synthetic>",
                            qualname="alignments",
                        ),
                        key="alignments",
                        children=_alignment_items,
                    ),
                    section(
                        identity=ArtifactSourceRef(
                            rel_path="<synthetic>",
                            qualname="counts",
                        ),
                        key="counts",
                        title="counts",
                        children=lambda alignments=alignments: iter(
                            _payload_to_units(
                                {
                                    "alignment_count": len(alignments),
                                    "status_counts": dict(
                                        _sorted(
                                            list(status_counts.items()),
                                            key=lambda item: item[0],
                                        )
                                    ),
                                }
                            )
                        ),
                    ),
                )
            ),
        )


@dataclass(frozen=True)
class InvariantWorkstreamDrift:
    object_id: str
    classification: str
    before_status: str
    after_status: str
    before_touchsite_count: int
    after_touchsite_count: int
    touchsite_delta: int
    before_surviving_touchsite_count: int
    after_surviving_touchsite_count: int
    surviving_touchsite_delta: int
    before_dominant_blocker_class: str
    after_dominant_blocker_class: str
    before_recommended_cut_object_id: str | None
    after_recommended_cut_object_id: str | None
    blocker_deltas: Mapping[str, int]
    added_touchsite_ids: tuple[str, ...]
    removed_touchsite_ids: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "classification": self.classification,
            "before_status": self.before_status,
            "after_status": self.after_status,
            "before_touchsite_count": self.before_touchsite_count,
            "after_touchsite_count": self.after_touchsite_count,
            "touchsite_delta": self.touchsite_delta,
            "before_surviving_touchsite_count": self.before_surviving_touchsite_count,
            "after_surviving_touchsite_count": self.after_surviving_touchsite_count,
            "surviving_touchsite_delta": self.surviving_touchsite_delta,
            "before_dominant_blocker_class": self.before_dominant_blocker_class,
            "after_dominant_blocker_class": self.after_dominant_blocker_class,
            "before_recommended_cut_object_id": self.before_recommended_cut_object_id,
            "after_recommended_cut_object_id": self.after_recommended_cut_object_id,
            "blocker_deltas": dict(self.blocker_deltas),
            "added_touchsite_ids": list(self.added_touchsite_ids),
            "removed_touchsite_ids": list(self.removed_touchsite_ids),
        }


def _workstream_payloads_by_object_id(
    payload: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    workstreams = payload.get("workstreams", [])
    if not isinstance(workstreams, list):
        return {}
    mapping: dict[str, Mapping[str, object]] = {}
    for item in workstreams:
        if not isinstance(item, Mapping):
            continue
        object_id = item.get("object_id")
        if isinstance(object_id, str) and object_id:
            mapping[object_id] = item
    return mapping


def _extract_touchsite_ids(workstream_payload: Mapping[str, object]) -> tuple[str, ...]:
    touchpoints = workstream_payload.get("touchpoints", [])
    if not isinstance(touchpoints, list):
        return ()
    touchsite_ids: list[str] = []
    for touchpoint in touchpoints:
        if not isinstance(touchpoint, Mapping):
            continue
        touchsites = touchpoint.get("touchsites", [])
        if not isinstance(touchsites, list):
            continue
        for touchsite in touchsites:
            if not isinstance(touchsite, Mapping):
                continue
            object_id = touchsite.get("object_id")
            if isinstance(object_id, str) and object_id:
                touchsite_ids.append(object_id)
    return tuple(_sorted(touchsite_ids))


def _recommended_cut_object_id(workstream_payload: Mapping[str, object]) -> str | None:
    next_actions = workstream_payload.get("next_actions")
    if not isinstance(next_actions, Mapping):
        return None
    recommended_cut = next_actions.get("recommended_cut")
    if not isinstance(recommended_cut, Mapping):
        return None
    object_id = recommended_cut.get("object_id")
    if not isinstance(object_id, str) or not object_id:
        return None
    return object_id


def _dominant_blocker_class_payload(workstream_payload: Mapping[str, object]) -> str:
    next_actions = workstream_payload.get("next_actions")
    if not isinstance(next_actions, Mapping):
        return "none"
    dominant = next_actions.get("dominant_blocker_class")
    if not isinstance(dominant, str) or not dominant:
        return "none"
    return dominant


def _health_summary_payload(workstream_payload: Mapping[str, object]) -> Mapping[str, object]:
    health_summary = workstream_payload.get("health_summary")
    if not isinstance(health_summary, Mapping):
        return {}
    return health_summary


def _doc_ids_payload(workstream_payload: Mapping[str, object]) -> tuple[str, ...]:
    doc_ids = workstream_payload.get("doc_ids", [])
    if not isinstance(doc_ids, list):
        return ()
    return tuple(_sorted([str(item) for item in doc_ids if isinstance(item, str)]))


def _policy_ids_payload(workstream_payload: Mapping[str, object]) -> tuple[str, ...]:
    policy_ids = workstream_payload.get("policy_ids", [])
    if not isinstance(policy_ids, list):
        return ()
    return tuple(_sorted([str(item) for item in policy_ids if isinstance(item, str)]))


def _title_payload(workstream_payload: Mapping[str, object]) -> str:
    title = workstream_payload.get("title")
    if isinstance(title, str) and title:
        return title
    object_id = workstream_payload.get("object_id")
    return object_id if isinstance(object_id, str) else ""


def _int_field(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key, 0)
    return int(value) if isinstance(value, int) else 0


def _recommended_ledger_action_for_status(status: str) -> str:
    if status == "landed":
        return "record_landed_state"
    if status == "in_progress":
        return "record_progress_state"
    if status == "queued":
        return "record_queued_state"
    return "record_state"


def _recommended_ledger_action_for_classification(classification: str) -> str:
    if classification == "stable":
        return "no_ledger_change"
    if classification == "reduced":
        return "append_reduction_delta"
    if classification == "widened":
        return "append_widening_delta"
    if classification == "relocated":
        return "append_relocation_delta"
    if classification == "introduced":
        return "append_introduction_delta"
    if classification == "retired":
        return "append_retirement_delta"
    return "append_delta"


def _ledger_delta_markdown_units(
    *,
    deltas: tuple[InvariantLedgerDelta, ...],
) -> Iterator[ArtifactUnit]:
    for delta in deltas:
        identity = ArtifactSourceRef(rel_path="<synthetic>", qualname=delta.object_id)
        yield section(
            identity=identity,
            key=delta.object_id,
            title=f"{delta.object_id} :: {delta.classification}",
            children=lambda delta=delta, identity=identity: iter(
                (
                    paragraph(
                        identity=identity,
                        value=delta.summary,
                    ),
                    bullet_list(
                        identity=identity,
                        key="append_entry",
                        title="append_entry",
                        children=lambda delta=delta: iter(
                            (
                                list_item(
                                    identity=identity,
                                    title="recommended_ledger_action",
                                    value=delta.recommended_ledger_action,
                                ),
                                list_item(
                                    identity=identity,
                                    title="before_status",
                                    value=delta.before_status,
                                ),
                                list_item(
                                    identity=identity,
                                    title="after_status",
                                    value=delta.after_status,
                                ),
                                list_item(
                                    identity=identity,
                                    title="before_recommended_cut_object_id",
                                    value=delta.before_recommended_cut_object_id,
                                ),
                                list_item(
                                    identity=identity,
                                    title="after_recommended_cut_object_id",
                                    value=delta.after_recommended_cut_object_id,
                                ),
                                list_item(
                                    identity=identity,
                                    title="target_policy_ids",
                                    children=lambda delta=delta: iter(
                                        list_item(
                                            identity=identity,
                                            value=policy_id,
                                        )
                                        for policy_id in delta.target_policy_ids
                                    ),
                                ),
                            )
                        ),
                    ),
                )
            ),
        )


def _ledger_alignment_markdown_units(
    *,
    alignments: tuple[InvariantLedgerAlignment, ...],
) -> Iterator[ArtifactUnit]:
    for alignment in alignments:
        identity = ArtifactSourceRef(
            rel_path="<synthetic>",
            qualname=f"{alignment.target_doc_id}:{alignment.object_id}",
        )
        yield section(
            identity=identity,
            key=alignment.object_id,
            title=f"{alignment.object_id} :: {alignment.alignment_status}",
            children=lambda alignment=alignment, identity=identity: iter(
                (
                    paragraph(
                        identity=identity,
                        value=alignment.summary,
                    ),
                    bullet_list(
                        identity=identity,
                        key="alignment",
                        title="alignment",
                        children=lambda alignment=alignment, identity=identity: iter(
                            (
                                list_item(
                                    identity=identity,
                                    title="target_doc_path",
                                    value=alignment.target_doc_path,
                                ),
                                list_item(
                                    identity=identity,
                                    title="classification",
                                    value=alignment.classification,
                                ),
                                list_item(
                                    identity=identity,
                                    title="recommended_ledger_action",
                                    value=alignment.recommended_ledger_action,
                                ),
                                list_item(
                                    identity=identity,
                                    title="object_reference_present",
                                    value=alignment.object_reference_present,
                                ),
                                list_item(
                                    identity=identity,
                                    title="summary_present",
                                    value=alignment.summary_present,
                                ),
                            )
                        ),
                    ),
                )
            ),
        )


def compare_invariant_workstreams(
    before_payload: Mapping[str, object],
    after_payload: Mapping[str, object],
) -> tuple[InvariantWorkstreamDrift, ...]:
    before_by_object_id = _workstream_payloads_by_object_id(before_payload)
    after_by_object_id = _workstream_payloads_by_object_id(after_payload)
    object_ids = _sorted(
        list(set(before_by_object_id.keys()) | set(after_by_object_id.keys()))
    )
    blocker_keys = (
        "ready_touchsite_count",
        "coverage_gap_touchsite_count",
        "policy_blocked_touchsite_count",
        "diagnostic_blocked_touchsite_count",
    )
    drifts: list[InvariantWorkstreamDrift] = []
    for object_id in object_ids:
        before = before_by_object_id.get(object_id)
        after = after_by_object_id.get(object_id)
        if before is None:
            after_health = _health_summary_payload(after or {})
            after_touchsite_ids = _extract_touchsite_ids(after or {})
            drifts.append(
                InvariantWorkstreamDrift(
                    object_id=object_id,
                    classification="introduced",
                    before_status="missing",
                    after_status=str((after or {}).get("status", "")),
                    before_touchsite_count=0,
                    after_touchsite_count=_int_field(after or {}, "touchsite_count"),
                    touchsite_delta=_int_field(after or {}, "touchsite_count"),
                    before_surviving_touchsite_count=0,
                    after_surviving_touchsite_count=_int_field(
                        after or {}, "surviving_touchsite_count"
                    ),
                    surviving_touchsite_delta=_int_field(
                        after or {}, "surviving_touchsite_count"
                    ),
                    before_dominant_blocker_class="none",
                    after_dominant_blocker_class=_dominant_blocker_class_payload(after or {}),
                    before_recommended_cut_object_id=None,
                    after_recommended_cut_object_id=_recommended_cut_object_id(after or {}),
                    blocker_deltas={
                        key: _int_field(after_health, key) for key in blocker_keys
                    },
                    added_touchsite_ids=after_touchsite_ids,
                    removed_touchsite_ids=(),
                )
            )
            continue
        if after is None:
            before_health = _health_summary_payload(before)
            before_touchsite_ids = _extract_touchsite_ids(before)
            drifts.append(
                InvariantWorkstreamDrift(
                    object_id=object_id,
                    classification="retired",
                    before_status=str(before.get("status", "")),
                    after_status="missing",
                    before_touchsite_count=_int_field(before, "touchsite_count"),
                    after_touchsite_count=0,
                    touchsite_delta=-_int_field(before, "touchsite_count"),
                    before_surviving_touchsite_count=_int_field(
                        before, "surviving_touchsite_count"
                    ),
                    after_surviving_touchsite_count=0,
                    surviving_touchsite_delta=-_int_field(
                        before, "surviving_touchsite_count"
                    ),
                    before_dominant_blocker_class=_dominant_blocker_class_payload(before),
                    after_dominant_blocker_class="none",
                    before_recommended_cut_object_id=_recommended_cut_object_id(before),
                    after_recommended_cut_object_id=None,
                    blocker_deltas={
                        key: -_int_field(before_health, key) for key in blocker_keys
                    },
                    added_touchsite_ids=(),
                    removed_touchsite_ids=before_touchsite_ids,
                )
            )
            continue

        before_health = _health_summary_payload(before)
        after_health = _health_summary_payload(after)
        before_touchsite_ids = set(_extract_touchsite_ids(before))
        after_touchsite_ids = set(_extract_touchsite_ids(after))
        added_touchsite_ids = tuple(_sorted(list(after_touchsite_ids - before_touchsite_ids)))
        removed_touchsite_ids = tuple(
            _sorted(list(before_touchsite_ids - after_touchsite_ids))
        )
        touchsite_delta = _int_field(after, "touchsite_count") - _int_field(
            before, "touchsite_count"
        )
        surviving_touchsite_delta = _int_field(
            after, "surviving_touchsite_count"
        ) - _int_field(before, "surviving_touchsite_count")
        blocker_deltas = {
            key: _int_field(after_health, key) - _int_field(before_health, key)
            for key in blocker_keys
        }
        blocking_pressure_deltas = (
            blocker_deltas["coverage_gap_touchsite_count"],
            blocker_deltas["policy_blocked_touchsite_count"],
            blocker_deltas["diagnostic_blocked_touchsite_count"],
        )
        delta_vector = (
            touchsite_delta,
            surviving_touchsite_delta,
            *blocker_deltas.values(),
        )
        if (
            delta_vector == (0, 0, 0, 0, 0, 0)
            and not added_touchsite_ids
            and not removed_touchsite_ids
            and _dominant_blocker_class_payload(before)
            == _dominant_blocker_class_payload(after)
            and _recommended_cut_object_id(before) == _recommended_cut_object_id(after)
        ):
            classification = "stable"
        elif (
            touchsite_delta <= 0
            and surviving_touchsite_delta <= 0
            and any(value < 0 for value in blocking_pressure_deltas)
            and not any(value > 0 for value in blocking_pressure_deltas)
        ):
            classification = "reduced"
        elif (
            touchsite_delta >= 0
            and surviving_touchsite_delta >= 0
            and any(value > 0 for value in blocking_pressure_deltas)
            and not any(value < 0 for value in blocking_pressure_deltas)
        ):
            classification = "widened"
        else:
            classification = "relocated"
        drifts.append(
            InvariantWorkstreamDrift(
                object_id=object_id,
                classification=classification,
                before_status=str(before.get("status", "")),
                after_status=str(after.get("status", "")),
                before_touchsite_count=_int_field(before, "touchsite_count"),
                after_touchsite_count=_int_field(after, "touchsite_count"),
                touchsite_delta=touchsite_delta,
                before_surviving_touchsite_count=_int_field(
                    before, "surviving_touchsite_count"
                ),
                after_surviving_touchsite_count=_int_field(
                    after, "surviving_touchsite_count"
                ),
                surviving_touchsite_delta=surviving_touchsite_delta,
                before_dominant_blocker_class=_dominant_blocker_class_payload(before),
                after_dominant_blocker_class=_dominant_blocker_class_payload(after),
                before_recommended_cut_object_id=_recommended_cut_object_id(before),
                after_recommended_cut_object_id=_recommended_cut_object_id(after),
                blocker_deltas=blocker_deltas,
                added_touchsite_ids=added_touchsite_ids,
                removed_touchsite_ids=removed_touchsite_ids,
            )
        )
    return tuple(_sorted(drifts, key=lambda item: (item.object_id, item.classification)))


def build_invariant_ledger_projections(
    workstreams: InvariantWorkstreamsProjection,
    *,
    root: Path | None = None,
) -> InvariantLedgerProjections:
    doc_paths_by_id = None if root is None else _doc_id_paths(root)

    def _ledger_items() -> Iterator[InvariantLedgerProjection]:
        for workstream in workstreams.iter_workstreams():
            yield _ledger_projection_for_workstream(
                workstream=workstream,
                root=root,
                doc_paths_by_id=doc_paths_by_id,
            )

    return InvariantLedgerProjections(
        root=workstreams.root,
        generated_at_utc=workstreams.generated_at_utc,
        ledgers=_stream_from_iterable(_ledger_items),
    )


def compare_invariant_ledger_projections(
    before_payload: Mapping[str, object],
    after_payload: Mapping[str, object],
) -> tuple[InvariantLedgerDelta, ...]:
    before_by_object_id = _workstream_payloads_by_object_id(before_payload)
    after_by_object_id = _workstream_payloads_by_object_id(after_payload)
    deltas: list[InvariantLedgerDelta] = []
    for drift in compare_invariant_workstreams(before_payload, after_payload):
        target_payload = after_by_object_id.get(drift.object_id) or before_by_object_id.get(
            drift.object_id
        )
        if target_payload is None:
            target_doc_ids = ()
            target_policy_ids = ()
            title = drift.object_id
        else:
            target_doc_ids = _doc_ids_payload(target_payload)
            target_policy_ids = _policy_ids_payload(target_payload)
            title = _title_payload(target_payload)
        summary = (
            f"{title or drift.object_id} {drift.classification}: touchsites "
            f"{drift.before_touchsite_count}->{drift.after_touchsite_count}, "
            f"dominant blocker {drift.before_dominant_blocker_class}->"
            f"{drift.after_dominant_blocker_class}, recommended cut "
            f"{drift.before_recommended_cut_object_id or '<none>'}->"
            f"{drift.after_recommended_cut_object_id or '<none>'}."
        )
        deltas.append(
            InvariantLedgerDelta(
                object_id=drift.object_id,
                title=title,
                target_doc_ids=target_doc_ids,
                target_policy_ids=target_policy_ids,
                classification=drift.classification,
                recommended_ledger_action=_recommended_ledger_action_for_classification(
                    drift.classification
                ),
                summary=summary,
                before_status=drift.before_status,
                after_status=drift.after_status,
                before_recommended_cut_object_id=drift.before_recommended_cut_object_id,
                after_recommended_cut_object_id=drift.after_recommended_cut_object_id,
            )
        )
    return tuple(_sorted(deltas, key=lambda item: (item.object_id, item.classification)))


def build_invariant_ledger_delta_projections(
    *,
    root: str,
    before_workstreams_artifact: str,
    after_workstreams_artifact: str,
    before_payload: Mapping[str, object],
    after_payload: Mapping[str, object],
) -> InvariantLedgerDeltaProjections:
    deltas = compare_invariant_ledger_projections(before_payload, after_payload)

    return InvariantLedgerDeltaProjections(
        root=root,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        before_workstreams_artifact=before_workstreams_artifact,
        after_workstreams_artifact=after_workstreams_artifact,
        deltas=_stream_from_iterable(lambda: iter(deltas)),
    )


def _doc_id_paths(root: Path) -> dict[str, tuple[Path, ...]]:
    grouped: defaultdict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*.md"):
        rel_parts = path.relative_to(root).parts
        if not rel_parts:
            continue
        if rel_parts[0] in {"artifacts", "out", ".git", ".venv", "__pycache__"}:
            continue
        frontmatter, _body = parse_strict_yaml_frontmatter(
            path.read_text(encoding="utf-8"),
            require_parser=False,
        )
        doc_id = frontmatter.get("doc_id")
        if isinstance(doc_id, str) and doc_id:
            grouped[doc_id].append(path)
    return {
        doc_id: tuple(_sorted(paths, key=lambda item: str(item)))
        for doc_id, paths in grouped.items()
    }


def _current_target_doc_alignments(
    *,
    root: Path,
    doc_paths_by_id: Mapping[str, tuple[Path, ...]],
    object_id: str,
    target_doc_ids: tuple[str, ...],
    summary: str,
) -> tuple[InvariantLedgerTargetDocAlignment, ...]:
    alignments: list[InvariantLedgerTargetDocAlignment] = []
    normalized_doc_ids = target_doc_ids or ("<unassigned>",)
    for target_doc_id in normalized_doc_ids:
        if target_doc_id == "<unassigned>":
            alignments.append(
                InvariantLedgerTargetDocAlignment(
                    target_doc_id=target_doc_id,
                    target_doc_path="",
                    alignment_status="unassigned_target_doc",
                    object_reference_present=False,
                    summary_present=False,
                )
            )
            continue
        resolved_paths = doc_paths_by_id.get(target_doc_id, ())
        if not resolved_paths:
            alignments.append(
                InvariantLedgerTargetDocAlignment(
                    target_doc_id=target_doc_id,
                    target_doc_path="",
                    alignment_status="missing_target_doc",
                    object_reference_present=False,
                    summary_present=False,
                )
            )
            continue
        if len(resolved_paths) > 1:
            alignments.append(
                InvariantLedgerTargetDocAlignment(
                    target_doc_id=target_doc_id,
                    target_doc_path="",
                    alignment_status="ambiguous_target_doc",
                    object_reference_present=False,
                    summary_present=False,
                )
            )
            continue
        target_path = resolved_paths[0]
        text = target_path.read_text(encoding="utf-8")
        object_reference_present = object_id in text
        summary_present = summary in text
        if summary_present:
            alignment_status = "projection_reflected"
        elif object_reference_present:
            alignment_status = "append_pending_existing_object"
        else:
            alignment_status = "append_pending_new_object"
        alignments.append(
            InvariantLedgerTargetDocAlignment(
                target_doc_id=target_doc_id,
                target_doc_path=str(target_path.relative_to(root)),
                alignment_status=alignment_status,
                object_reference_present=object_reference_present,
                summary_present=summary_present,
            )
        )
    return tuple(
        _sorted(alignments, key=lambda item: (item.target_doc_id, item.alignment_status))
    )


def _recommended_doc_alignment_action(
    *,
    status_counts: Mapping[str, int],
) -> str:
    if status_counts.get("missing_target_doc", 0) > 0:
        return "repair_missing_target_doc"
    if status_counts.get("ambiguous_target_doc", 0) > 0:
        return "repair_ambiguous_target_doc"
    if status_counts.get("unassigned_target_doc", 0) > 0:
        return "assign_target_doc"
    if status_counts.get("append_pending_new_object", 0) > 0:
        return "append_new_ledger_entry"
    if status_counts.get("append_pending_existing_object", 0) > 0:
        return "append_existing_ledger_entry"
    return "none"


def _dominant_doc_alignment_status(
    *,
    status_counts: Mapping[str, int],
) -> str:
    priorities = (
        "missing_target_doc",
        "ambiguous_target_doc",
        "unassigned_target_doc",
        "append_pending_new_object",
        "append_pending_existing_object",
        "projection_reflected",
    )
    for status in priorities:
        if status_counts.get(status, 0) > 0:
            return status
    return "none"


def _alignment_summary(
    alignments: tuple[InvariantLedgerTargetDocAlignment, ...],
) -> InvariantLedgerAlignmentSummary:
    status_counts: dict[str, int] = defaultdict(int)
    misaligned_target_doc_ids = [
        item.target_doc_id
        for item in alignments
        if item.alignment_status != "projection_reflected"
    ]
    for item in alignments:
        status_counts[item.alignment_status] += 1
    return InvariantLedgerAlignmentSummary(
        target_doc_count=len(alignments),
        reflected_target_doc_count=status_counts.get("projection_reflected", 0),
        append_pending_existing_target_doc_count=status_counts.get(
            "append_pending_existing_object", 0
        ),
        append_pending_new_target_doc_count=status_counts.get(
            "append_pending_new_object", 0
        ),
        missing_target_doc_count=status_counts.get("missing_target_doc", 0),
        ambiguous_target_doc_count=status_counts.get("ambiguous_target_doc", 0),
        unassigned_target_doc_count=status_counts.get("unassigned_target_doc", 0),
        dominant_alignment_status=_dominant_doc_alignment_status(
            status_counts=status_counts
        ),
        recommended_doc_alignment_action=_recommended_doc_alignment_action(
            status_counts=status_counts
        ),
        misaligned_target_doc_ids=tuple(_sorted(misaligned_target_doc_ids)),
    )


def _ledger_projection_for_workstream(
    *,
    workstream: InvariantWorkstreamProjection,
    root: Path | None,
    doc_paths_by_id: Mapping[str, tuple[Path, ...]] | None,
) -> InvariantLedgerProjection:
    recommended_cut = workstream.recommended_cut()
    recommended_ready_cut = workstream.recommended_ready_cut()
    recommended_coverage_gap_cut = workstream.recommended_coverage_gap_cut()
    summary = (
        f"{workstream.object_id.wire()} is {workstream.status} with "
        f"{workstream.touchsite_count} touchsites; dominant blocker "
        f"{workstream.dominant_blocker_class()}; recommended cut "
        f"{recommended_cut.object_id.wire() if recommended_cut is not None else '<none>'}."
    )
    target_doc_alignments: tuple[InvariantLedgerTargetDocAlignment, ...] = ()
    alignment_summary: InvariantLedgerAlignmentSummary | None = None
    if root is not None and doc_paths_by_id is not None:
        target_doc_alignments = _current_target_doc_alignments(
            root=root,
            doc_paths_by_id=doc_paths_by_id,
            object_id=workstream.object_id.wire(),
            target_doc_ids=workstream.doc_ids,
            summary=summary,
        )
        alignment_summary = _alignment_summary(target_doc_alignments)
    return InvariantLedgerProjection(
        object_id=workstream.object_id.wire(),
        title=workstream.title,
        status=workstream.status,
        target_doc_ids=workstream.doc_ids,
        target_policy_ids=workstream.policy_ids,
        dominant_blocker_class=workstream.dominant_blocker_class(),
        recommended_remediation_family=workstream.recommended_remediation_family(),
        recommended_ledger_action=_recommended_ledger_action_for_status(
            workstream.status
        ),
        summary=summary,
        current_snapshot={
            "touchsite_count": workstream.touchsite_count,
            "collapsible_touchsite_count": workstream.collapsible_touchsite_count,
            "surviving_touchsite_count": workstream.surviving_touchsite_count,
            "policy_signal_count": workstream.policy_signal_count,
            "coverage_count": workstream.coverage_count,
            "diagnostic_count": workstream.diagnostic_count,
            "recommended_cut_object_id": (
                None if recommended_cut is None else recommended_cut.object_id.wire()
            ),
            "recommended_ready_cut_object_id": (
                None
                if recommended_ready_cut is None
                else recommended_ready_cut.object_id.wire()
            ),
            "recommended_coverage_gap_cut_object_id": (
                None
                if recommended_coverage_gap_cut is None
                else recommended_coverage_gap_cut.object_id.wire()
            ),
        },
        target_doc_alignments=target_doc_alignments,
        alignment_summary=alignment_summary,
    )


def build_invariant_ledger_alignments(
    *,
    root: Path,
    ledger_deltas: InvariantLedgerDeltaProjections,
) -> InvariantLedgerAlignments:
    doc_paths_by_id = _doc_id_paths(root)

    def _alignment_items() -> Iterator[InvariantLedgerAlignment]:
        for delta in ledger_deltas.iter_deltas():
            target_doc_ids = delta.target_doc_ids or ("<unassigned>",)
            for target_doc_id in target_doc_ids:
                if target_doc_id == "<unassigned>":
                    yield InvariantLedgerAlignment(
                        object_id=delta.object_id,
                        title=delta.title,
                        target_doc_id=target_doc_id,
                        target_doc_path="",
                        classification=delta.classification,
                        recommended_ledger_action=delta.recommended_ledger_action,
                        alignment_status="unassigned_target_doc",
                        summary=delta.summary,
                        object_reference_present=False,
                        summary_present=False,
                    )
                    continue
                resolved_paths = doc_paths_by_id.get(target_doc_id, ())
                if not resolved_paths:
                    yield InvariantLedgerAlignment(
                        object_id=delta.object_id,
                        title=delta.title,
                        target_doc_id=target_doc_id,
                        target_doc_path="",
                        classification=delta.classification,
                        recommended_ledger_action=delta.recommended_ledger_action,
                        alignment_status="missing_target_doc",
                        summary=delta.summary,
                        object_reference_present=False,
                        summary_present=False,
                    )
                    continue
                if len(resolved_paths) > 1:
                    yield InvariantLedgerAlignment(
                        object_id=delta.object_id,
                        title=delta.title,
                        target_doc_id=target_doc_id,
                        target_doc_path="",
                        classification=delta.classification,
                        recommended_ledger_action=delta.recommended_ledger_action,
                        alignment_status="ambiguous_target_doc",
                        summary=delta.summary,
                        object_reference_present=False,
                        summary_present=False,
                    )
                    continue
                target_path = resolved_paths[0]
                text = target_path.read_text(encoding="utf-8")
                object_reference_present = delta.object_id in text
                summary_present = delta.summary in text
                if summary_present:
                    alignment_status = "delta_reflected"
                elif object_reference_present:
                    alignment_status = "append_pending_existing_object"
                else:
                    alignment_status = "append_pending_new_object"
                yield InvariantLedgerAlignment(
                    object_id=delta.object_id,
                    title=delta.title,
                    target_doc_id=target_doc_id,
                    target_doc_path=str(target_path.relative_to(root)),
                    classification=delta.classification,
                    recommended_ledger_action=delta.recommended_ledger_action,
                    alignment_status=alignment_status,
                    summary=delta.summary,
                    object_reference_present=object_reference_present,
                    summary_present=summary_present,
                )

    return InvariantLedgerAlignments(
        root=str(root),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        alignments=_stream_from_iterable(_alignment_items),
    )


def _payload_to_units(payload: Mapping[str, object]) -> tuple[ArtifactUnit, ...]:
    units: list[ArtifactUnit] = []
    for key, value in payload.items():
        if isinstance(value, Mapping):
            units.append(
                section(
                    identity=ArtifactSourceRef(rel_path="<synthetic>", qualname=str(key)),
                    key=str(key),
                    title=str(key),
                    children=lambda value=value: iter(_payload_to_units(value)),
                )
            )
            continue
        if isinstance(value, list):
            units.append(
                bullet_list(
                    identity=ArtifactSourceRef(rel_path="<synthetic>", qualname=str(key)),
                    key=str(key),
                    children=lambda value=value: (
                        list_item(
                            identity=ArtifactSourceRef(rel_path="<synthetic>", qualname=str(key)),
                            value=item,
                        )
                        for item in value
                    ),
                )
            )
            continue
        units.append(
            scalar(
                identity=ArtifactSourceRef(rel_path="<synthetic>", qualname=str(key)),
                key=str(key),
                title=str(key),
                value=cast(object, value),
            )
        )
    return tuple(units)


def _synthetic_identity(
    *,
    ref_kind: str,
    value: str,
) -> tuple[str, str]:
    return (
        canonical_site_identity(
            rel_path="<synthetic>",
            qualname=f"{ref_kind}:{value}",
            line=0,
            column=0,
            node_kind="synthetic",
            surface="invariant_graph",
        ),
        canonical_structural_identity(
            rel_path="<synthetic>",
            qualname=f"{ref_kind}:{value}",
            structural_path=f"{ref_kind}:{value}",
            node_kind="synthetic",
            surface="invariant_graph",
        ),
    )


def _edge_id(edge_kind: str, source_id: str, target_id: str) -> str:
    return stable_hash("invariant_graph_edge", edge_kind, source_id, target_id)


def _marker_node_id(entry: InvariantMarkerScanNode) -> str:
    return f"{entry.scan_kind}:{entry.structural_identity}"


def _synthetic_node(
    *,
    node_id: str,
    title: str,
    ref_kind: str,
    value: str,
    object_ids: tuple[str, ...] = (),
    doc_ids: tuple[str, ...] = (),
    policy_ids: tuple[str, ...] = (),
    invariant_ids: tuple[str, ...] = (),
    reasoning_summary: str = "",
    reasoning_control: str = "",
    blocking_dependencies: tuple[str, ...] = (),
    rel_path: str = "",
    qualname: str = "",
    line: int = 0,
    column: int = 0,
    seam_class: str = "",
    marker_id: str = "",
    marker_name: str = "",
    marker_kind: str = "",
    source_marker_node_id: str = "",
    node_kind: str = "synthetic_work_item",
    status_hint: str = "",
) -> InvariantGraphNode:
    site_identity, structural_identity = _synthetic_identity(
        ref_kind=ref_kind,
        value=value,
    )
    return InvariantGraphNode(
        node_id=node_id,
        node_kind=node_kind,
        title=title,
        marker_name=marker_name,
        marker_kind=marker_kind,
        marker_id=marker_id,
        site_identity=site_identity,
        structural_identity=structural_identity,
        object_ids=object_ids,
        doc_ids=doc_ids,
        policy_ids=policy_ids,
        invariant_ids=invariant_ids,
        reasoning_summary=reasoning_summary,
        reasoning_control=reasoning_control,
        blocking_dependencies=blocking_dependencies,
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        column=column,
        ast_node_kind="synthetic",
        seam_class=seam_class,
        source_marker_node_id=source_marker_node_id,
        status_hint=status_hint,
    )


def _node_from_scan_entry(entry: InvariantMarkerScanNode) -> InvariantGraphNode:
    return InvariantGraphNode(
        node_id=_marker_node_id(entry),
        node_kind=entry.scan_kind,
        title=entry.qualname or entry.marker_name,
        marker_name=entry.marker_name,
        marker_kind=entry.marker_kind,
        marker_id=entry.marker_id,
        site_identity=entry.site_identity,
        structural_identity=entry.structural_identity,
        object_ids=entry.object_ids,
        doc_ids=entry.doc_ids,
        policy_ids=entry.policy_ids,
        invariant_ids=entry.invariant_ids,
        reasoning_summary=entry.reasoning_summary,
        reasoning_control=entry.reasoning_control,
        blocking_dependencies=entry.blocking_dependencies,
        rel_path=entry.rel_path,
        qualname=entry.qualname,
        line=entry.line,
        column=entry.column,
        ast_node_kind=entry.ast_node_kind,
        seam_class="",
        source_marker_node_id="",
        status_hint="",
    )


def _synthetic_ref_node_id(kind: str, value: str) -> str:
    return f"{kind}:{value}"


def _primary_object_id(node: InvariantGraphNode) -> str:
    _prefix, _separator, primary_object_id = node.node_id.partition(":")
    if primary_object_id:
        return primary_object_id
    if node.object_ids:
        return node.object_ids[0]
    return node.title


def _semantic_ref_node(
    *,
    ref_kind: SemanticLinkKind,
    value: str,
) -> InvariantGraphNode:
    node_id = _synthetic_ref_node_id(ref_kind.value, value)
    object_ids = (value,) if ref_kind is SemanticLinkKind.OBJECT_ID else ()
    doc_ids = (value,) if ref_kind is SemanticLinkKind.DOC_ID else ()
    policy_ids = (value,) if ref_kind is SemanticLinkKind.POLICY_ID else ()
    invariant_ids = (value,) if ref_kind is SemanticLinkKind.INVARIANT_ID else ()
    return _synthetic_node(
        node_id=node_id,
        title=f"{ref_kind.value}:{value}",
        ref_kind=ref_kind.value,
        value=value,
        object_ids=object_ids,
        doc_ids=doc_ids,
        policy_ids=policy_ids,
        invariant_ids=invariant_ids,
    )


def _touchsite_seam_class(
    *,
    qualname: str,
    boundary_name: str,
    touchpoint_definition: ProjectionSemanticFragmentPhase5TouchpointDefinition,
) -> str:
    function_name = qualname.rsplit(".", 1)[-1]
    if not touchpoint_definition.collapse_private_helpers:
        return "surviving_carrier_seam"
    if (
        not function_name.startswith("_")
        or boundary_name in touchpoint_definition.surviving_boundary_names
    ):
        return "surviving_carrier_seam"
    return "collapsible_helper_seam"


@dataclass(frozen=True)
class _Phase5Touchsite:
    touchsite_object_id: str
    touchpoint_object_id: str
    subqueue_object_id: str
    rel_path: str
    qualname: str
    boundary_name: str
    line: int
    column: int
    node_kind: str
    site_identity: str
    structural_identity: str
    seam_class: str


class _Phase5TouchsiteScanner(ast.NodeVisitor):
    def __init__(
        self,
        *,
        rel_path: str,
        touchpoint_definition: ProjectionSemanticFragmentPhase5TouchpointDefinition,
        touchpoint_object_id: str,
        subqueue_object_id: str,
    ) -> None:
        self.rel_path = rel_path
        self.touchpoint_definition = touchpoint_definition
        self.touchpoint_object_id = touchpoint_object_id
        self.subqueue_object_id = subqueue_object_id
        self._scope: list[str] = []
        self.touchsites: list[_Phase5Touchsite] = []

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
                touchpoint_definition=self.touchpoint_definition,
            )
            touchsite_object_id = f"PSF-007-TS:{structural_identity}"
            self.touchsites.append(
                _Phase5Touchsite(
                    touchsite_object_id=touchsite_object_id,
                    touchpoint_object_id=self.touchpoint_object_id,
                    subqueue_object_id=self.subqueue_object_id,
                    rel_path=self.rel_path,
                    qualname=qualname,
                    boundary_name=boundary_name,
                    line=line,
                    column=column,
                    node_kind=node_kind,
                    site_identity=site_identity,
                    structural_identity=structural_identity,
                    seam_class=seam_class,
                )
            )
        self.generic_visit(node)
        self._scope.pop()


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


def _scan_phase5_touchsites(
    *,
    touchpoint_definition: ProjectionSemanticFragmentPhase5TouchpointDefinition,
    rel_path: str,
    touchpoint_object_id: str,
    subqueue_object_id: str,
) -> tuple[_Phase5Touchsite, ...]:
    source = (_REPO_ROOT / rel_path).read_text(encoding="utf-8")
    scanner = _Phase5TouchsiteScanner(
        rel_path=rel_path,
        touchpoint_definition=touchpoint_definition,
        touchpoint_object_id=touchpoint_object_id,
        subqueue_object_id=subqueue_object_id,
    )
    scanner.visit(ast.parse(source, filename=rel_path))
    return tuple(
        _sorted(
            scanner.touchsites,
            key=lambda item: (
                item.rel_path,
                item.line,
                item.column,
                item.qualname,
                item.boundary_name,
            ),
        )
    )


def _phase5_primary_object_id(values: tuple[str, ...], fallback: str) -> str:
    for value in values:
        if value == fallback:
            return value
    return fallback


@dataclass
class _InvariantGraphBuildState:
    root: Path
    nodes_by_id: dict[str, InvariantGraphNode]
    diagnostics: list[InvariantGraphDiagnostic]
    edge_keys: set[tuple[str, str, str]]
    edges: list[InvariantGraphEdge]
    object_node_ids: dict[str, str]
    object_owner_node_ids: dict[str, str]
    workstream_root_ids: list[str]
    declared_workstream_ids: set[str]


def _new_build_state(root: Path) -> _InvariantGraphBuildState:
    return _InvariantGraphBuildState(
        root=root,
        nodes_by_id={},
        diagnostics=[],
        edge_keys=set(),
        edges=[],
        object_node_ids={},
        object_owner_node_ids={},
        workstream_root_ids=[],
        declared_workstream_ids=set(),
    )


def _add_node(
    state: _InvariantGraphBuildState,
    node: InvariantGraphNode,
    *,
    replace: bool = False,
) -> None:
    existing = state.nodes_by_id.get(node.node_id)
    if existing is not None and not replace:
        raise ValueError(f"duplicate invariant graph node id: {node.node_id}")
    state.nodes_by_id[node.node_id] = node


def _add_edge(
    state: _InvariantGraphBuildState,
    edge_kind: str,
    source_id: str,
    target_id: str,
) -> None:
    key = (edge_kind, source_id, target_id)
    if source_id == target_id or key in state.edge_keys:
        return
    state.edge_keys.add(key)
    state.edges.append(
        InvariantGraphEdge(
            edge_id=_edge_id(edge_kind, source_id, target_id),
            edge_kind=edge_kind,
            source_id=source_id,
            target_id=target_id,
        )
    )


def _ensure_ref_node(
    state: _InvariantGraphBuildState,
    ref_kind: SemanticLinkKind,
    value: str,
) -> str:
    node_id = _synthetic_ref_node_id(ref_kind.value, value)
    if node_id not in state.nodes_by_id:
        _add_node(state, _semantic_ref_node(ref_kind=ref_kind, value=value))
    if ref_kind is SemanticLinkKind.OBJECT_ID:
        state.object_node_ids.setdefault(value, node_id)
    return node_id


def _claim_object_id(
    state: _InvariantGraphBuildState,
    node: InvariantGraphNode,
    *,
    object_id: str,
) -> None:
    claimed_by = state.object_owner_node_ids.get(object_id)
    if claimed_by is not None and claimed_by != node.node_id:
        raise ValueError(
            "duplicate invariant graph object_id ownership: "
            f"{object_id} claimed by {claimed_by} and {node.node_id}"
        )
    state.object_owner_node_ids[object_id] = node.node_id
    state.object_node_ids[object_id] = node.node_id


def _link_node_refs(state: _InvariantGraphBuildState, node: InvariantGraphNode) -> None:
    for value in node.object_ids:
        _add_edge(
            state,
            "links_to",
            node.node_id,
            _ensure_ref_node(state, SemanticLinkKind.OBJECT_ID, value),
        )
    for value in node.doc_ids:
        _add_edge(
            state,
            "links_to",
            node.node_id,
            _ensure_ref_node(state, SemanticLinkKind.DOC_ID, value),
        )
    for value in node.policy_ids:
        _add_edge(
            state,
            "links_to",
            node.node_id,
            _ensure_ref_node(state, SemanticLinkKind.POLICY_ID, value),
        )
    for value in node.invariant_ids:
        _add_edge(
            state,
            "links_to",
            node.node_id,
            _ensure_ref_node(state, SemanticLinkKind.INVARIANT_ID, value),
        )


def _work_item_node(
    *,
    object_id: str,
    title: str,
    rel_path: str,
    qualname: str,
    line: int,
    marker_id: str,
    marker_name: str,
    marker_kind: str,
    site_identity: str,
    structural_identity: str,
    object_ids: tuple[str, ...],
    doc_ids: tuple[str, ...],
    policy_ids: tuple[str, ...],
    invariant_ids: tuple[str, ...],
    reasoning_summary: str,
    reasoning_control: str,
    blocking_dependencies: tuple[str, ...],
    source_marker_node_id: str,
    status_hint: str = "",
) -> InvariantGraphNode:
    return InvariantGraphNode(
        node_id=_synthetic_ref_node_id("object_id", object_id),
        node_kind="synthetic_work_item",
        title=title,
        marker_name=marker_name,
        marker_kind=marker_kind,
        marker_id=marker_id,
        site_identity=site_identity,
        structural_identity=structural_identity,
        object_ids=object_ids,
        doc_ids=doc_ids,
        policy_ids=policy_ids,
        invariant_ids=invariant_ids,
        reasoning_summary=reasoning_summary,
        reasoning_control=reasoning_control,
        blocking_dependencies=blocking_dependencies,
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        column=1,
        ast_node_kind="function_def",
        seam_class="",
        source_marker_node_id=source_marker_node_id,
        status_hint=status_hint,
    )


def _add_work_item(
    state: _InvariantGraphBuildState,
    *,
    object_id: str,
    title: str,
    rel_path: str,
    qualname: str,
    line: int,
    marker_id: str,
    marker_name: str,
    marker_kind: str,
    site_identity: str,
    structural_identity: str,
    object_ids: tuple[str, ...],
    doc_ids: tuple[str, ...],
    policy_ids: tuple[str, ...],
    invariant_ids: tuple[str, ...],
    reasoning_summary: str,
    reasoning_control: str,
    blocking_dependencies: tuple[str, ...],
    source_marker_node_id: str,
    status_hint: str = "",
) -> str:
    node = _work_item_node(
        object_id=object_id,
        title=title,
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        marker_id=marker_id,
        marker_name=marker_name,
        marker_kind=marker_kind,
        site_identity=site_identity,
        structural_identity=structural_identity,
        object_ids=object_ids,
        doc_ids=doc_ids,
        policy_ids=policy_ids,
        invariant_ids=invariant_ids,
        reasoning_summary=reasoning_summary,
        reasoning_control=reasoning_control,
        blocking_dependencies=blocking_dependencies,
        source_marker_node_id=source_marker_node_id,
        status_hint=status_hint,
    )
    _add_node(state, node, replace=True)
    _claim_object_id(state, node, object_id=object_id)
    state.declared_workstream_ids.add(object_id)
    _link_node_refs(state, node)
    return node.node_id


def _register_root_workstream(
    state: _InvariantGraphBuildState,
    *,
    object_id: str,
) -> None:
    if object_id not in state.workstream_root_ids:
        state.workstream_root_ids.append(object_id)
    state.declared_workstream_ids.add(object_id)


def _normalize_rel_path(root: Path, raw_path: object) -> str:
    value = str(raw_path or "").strip().replace("\\", "/")
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute():
        try:
            return str(path.resolve().relative_to(root)).replace("\\", "/")
        except ValueError:
            return path.name
    return value.lstrip("./")


def _boundary_key(rel_path: str, line: int, boundary_name: str) -> tuple[str, int, str]:
    return (rel_path, line, boundary_name)


def _scan_phase5_touchsites_for_definition(
    *,
    touchpoint_definition: ProjectionSemanticFragmentPhase5TouchpointDefinition,
) -> tuple[_Phase5Touchsite, ...]:
    return _scan_phase5_touchsites(
        touchpoint_definition=touchpoint_definition,
        rel_path=touchpoint_definition.rel_path,
        touchpoint_object_id=touchpoint_definition.touchpoint_id,
        subqueue_object_id=touchpoint_definition.subqueue_id,
    )


def _enrich_psf_phase5_workstream(
    state: _InvariantGraphBuildState,
    *,
    marker_node_id_by_marker_id: Mapping[str, str],
) -> None:
    queue_definitions = tuple(iter_phase5_queues())
    subqueue_definitions = tuple(iter_phase5_subqueues())
    touchpoint_definitions = tuple(iter_phase5_touchpoints())

    for queue_definition in queue_definitions:
        primary_object_id = _phase5_primary_object_id(
            tuple(
                link.value
                for link in queue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.OBJECT_ID
            ),
            queue_definition.queue_id,
        )
        _add_work_item(
            state,
            object_id=primary_object_id,
            title=queue_definition.title,
            rel_path=queue_definition.rel_path,
            qualname=queue_definition.qualname,
            line=queue_definition.line,
            marker_id=queue_definition.marker_identity,
            marker_name="gabion.invariants.todo_decorator",
            marker_kind=queue_definition.marker_payload.marker_kind.value,
            site_identity=queue_definition.site_identity,
            structural_identity=queue_definition.structural_identity,
            object_ids=tuple(
                link.value
                for link in queue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.OBJECT_ID
            ),
            doc_ids=tuple(
                link.value
                for link in queue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.DOC_ID
            ),
            policy_ids=tuple(
                link.value
                for link in queue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.POLICY_ID
            ),
            invariant_ids=tuple(
                link.value
                for link in queue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.INVARIANT_ID
            ),
            reasoning_summary=queue_definition.marker_payload.reasoning.summary,
            reasoning_control=queue_definition.marker_payload.reasoning.control,
            blocking_dependencies=queue_definition.marker_payload.reasoning.blocking_dependencies,
            source_marker_node_id=marker_node_id_by_marker_id.get(
                queue_definition.marker_identity,
                "",
            ),
        )
        _register_root_workstream(state, object_id=primary_object_id)

    for subqueue_definition in subqueue_definitions:
        primary_object_id = _phase5_primary_object_id(
            tuple(
                link.value
                for link in subqueue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.OBJECT_ID
                and link.value.startswith(subqueue_definition.subqueue_id)
            ),
            subqueue_definition.subqueue_id,
        )
        _add_work_item(
            state,
            object_id=primary_object_id,
            title=subqueue_definition.title,
            rel_path=subqueue_definition.rel_path,
            qualname=subqueue_definition.qualname,
            line=subqueue_definition.line,
            marker_id=subqueue_definition.marker_identity,
            marker_name="gabion.invariants.todo_decorator",
            marker_kind=subqueue_definition.marker_payload.marker_kind.value,
            site_identity=subqueue_definition.site_identity,
            structural_identity=subqueue_definition.structural_identity,
            object_ids=tuple(
                link.value
                for link in subqueue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.OBJECT_ID
            ),
            doc_ids=tuple(
                link.value
                for link in subqueue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.DOC_ID
            ),
            policy_ids=tuple(
                link.value
                for link in subqueue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.POLICY_ID
            ),
            invariant_ids=tuple(
                link.value
                for link in subqueue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.INVARIANT_ID
            ),
            reasoning_summary=subqueue_definition.marker_payload.reasoning.summary,
            reasoning_control=subqueue_definition.marker_payload.reasoning.control,
            blocking_dependencies=subqueue_definition.marker_payload.reasoning.blocking_dependencies,
            source_marker_node_id=marker_node_id_by_marker_id.get(
                subqueue_definition.marker_identity,
                "",
            ),
        )

    for touchpoint_definition in touchpoint_definitions:
        primary_object_id = _phase5_primary_object_id(
            tuple(
                link.value
                for link in touchpoint_definition.marker_payload.links
                if link.kind is SemanticLinkKind.OBJECT_ID
                and link.value.startswith(touchpoint_definition.touchpoint_id)
            ),
            touchpoint_definition.touchpoint_id,
        )
        _add_work_item(
            state,
            object_id=primary_object_id,
            title=touchpoint_definition.title,
            rel_path=touchpoint_definition.rel_path,
            qualname=touchpoint_definition.qualname,
            line=touchpoint_definition.line,
            marker_id=touchpoint_definition.marker_identity,
            marker_name="gabion.invariants.todo_decorator",
            marker_kind=touchpoint_definition.marker_payload.marker_kind.value,
            site_identity=touchpoint_definition.site_identity,
            structural_identity=touchpoint_definition.structural_identity,
            object_ids=tuple(
                link.value
                for link in touchpoint_definition.marker_payload.links
                if link.kind is SemanticLinkKind.OBJECT_ID
            ),
            doc_ids=tuple(
                link.value
                for link in touchpoint_definition.marker_payload.links
                if link.kind is SemanticLinkKind.DOC_ID
            ),
            policy_ids=tuple(
                link.value
                for link in touchpoint_definition.marker_payload.links
                if link.kind is SemanticLinkKind.POLICY_ID
            ),
            invariant_ids=tuple(
                link.value
                for link in touchpoint_definition.marker_payload.links
                if link.kind is SemanticLinkKind.INVARIANT_ID
            ),
            reasoning_summary=touchpoint_definition.marker_payload.reasoning.summary,
            reasoning_control=touchpoint_definition.marker_payload.reasoning.control,
            blocking_dependencies=touchpoint_definition.marker_payload.reasoning.blocking_dependencies,
            source_marker_node_id=marker_node_id_by_marker_id.get(
                touchpoint_definition.marker_identity,
                "",
            ),
        )

    for queue_definition in queue_definitions:
        queue_node_id = state.object_node_ids[queue_definition.queue_id]
        for subqueue_id in queue_definition.subqueue_ids:
            subqueue_node_id = state.object_node_ids[subqueue_id]
            _add_edge(state, "contains", queue_node_id, subqueue_node_id)
            _add_edge(state, "blocks", subqueue_node_id, queue_node_id)

    for subqueue_definition in subqueue_definitions:
        subqueue_node_id = state.object_node_ids[subqueue_definition.subqueue_id]
        for touchpoint_id in subqueue_definition.touchpoint_ids:
            touchpoint_node_id = state.object_node_ids[touchpoint_id]
            _add_edge(state, "contains", subqueue_node_id, touchpoint_node_id)
            _add_edge(state, "blocks", touchpoint_node_id, subqueue_node_id)

    for touchpoint_definition in touchpoint_definitions:
        touchpoint_node_id = state.object_node_ids[touchpoint_definition.touchpoint_id]
        touchsites = _scan_phase5_touchsites_for_definition(
            touchpoint_definition=touchpoint_definition
        )
        for touchsite in touchsites:
            node_id = _synthetic_ref_node_id("object_id", touchsite.touchsite_object_id)
            node = InvariantGraphNode(
                node_id=node_id,
                node_kind="synthetic_touchsite",
                title=touchsite.boundary_name,
                marker_name="grade_boundary",
                marker_kind="",
                marker_id="",
                site_identity=touchsite.site_identity,
                structural_identity=touchsite.structural_identity,
                object_ids=(touchsite.touchsite_object_id,),
                doc_ids=(),
                policy_ids=(),
                invariant_ids=(),
                reasoning_summary="PSF-007 touchsite remains active.",
                reasoning_control=touchpoint_definition.marker_payload.reasoning.control,
                blocking_dependencies=(touchpoint_definition.touchpoint_id,),
                rel_path=touchsite.rel_path,
                qualname=touchsite.qualname,
                line=touchsite.line,
                column=touchsite.column,
                ast_node_kind=touchsite.node_kind,
                seam_class=touchsite.seam_class,
                source_marker_node_id=touchpoint_node_id,
                status_hint="",
            )
            _add_node(state, node, replace=True)
            _claim_object_id(state, node, object_id=touchsite.touchsite_object_id)
            state.declared_workstream_ids.add(touchsite.touchsite_object_id)
            _add_edge(state, "contains", touchpoint_node_id, node_id)
            _add_edge(state, "blocks", node_id, touchpoint_node_id)


def _enrich_prf_workstream(
    state: _InvariantGraphBuildState,
    *,
    marker_node_id_by_marker_id: Mapping[str, str],
) -> None:
    queue_definitions = tuple(iter_prf_queues())
    subqueue_definitions = tuple(iter_prf_subqueues())
    for queue_definition in queue_definitions:
        _add_work_item(
            state,
            object_id=queue_definition.queue_id,
            title=queue_definition.title,
            rel_path=queue_definition.rel_path,
            qualname=queue_definition.qualname,
            line=queue_definition.line,
            marker_id=queue_definition.marker_identity,
            marker_name="gabion.invariants.todo_decorator",
            marker_kind=queue_definition.marker_payload.marker_kind.value,
            site_identity=queue_definition.site_identity,
            structural_identity=queue_definition.structural_identity,
            object_ids=tuple(
                link.value
                for link in queue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.OBJECT_ID
            ),
            doc_ids=tuple(
                link.value
                for link in queue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.DOC_ID
            ),
            policy_ids=tuple(
                link.value
                for link in queue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.POLICY_ID
            ),
            invariant_ids=tuple(
                link.value
                for link in queue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.INVARIANT_ID
            ),
            reasoning_summary=queue_definition.marker_payload.reasoning.summary,
            reasoning_control=queue_definition.marker_payload.reasoning.control,
            blocking_dependencies=queue_definition.marker_payload.reasoning.blocking_dependencies,
            source_marker_node_id=marker_node_id_by_marker_id.get(
                queue_definition.marker_identity,
                "",
            ),
            status_hint=queue_definition.status_hint,
        )
        _register_root_workstream(state, object_id=queue_definition.queue_id)
    for subqueue_definition in subqueue_definitions:
        _add_work_item(
            state,
            object_id=subqueue_definition.subqueue_id,
            title=subqueue_definition.title,
            rel_path=subqueue_definition.rel_path,
            qualname=subqueue_definition.qualname,
            line=subqueue_definition.line,
            marker_id=subqueue_definition.marker_identity,
            marker_name="gabion.invariants.todo_decorator",
            marker_kind=subqueue_definition.marker_payload.marker_kind.value,
            site_identity=subqueue_definition.site_identity,
            structural_identity=subqueue_definition.structural_identity,
            object_ids=tuple(
                link.value
                for link in subqueue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.OBJECT_ID
            ),
            doc_ids=tuple(
                link.value
                for link in subqueue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.DOC_ID
            ),
            policy_ids=tuple(
                link.value
                for link in subqueue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.POLICY_ID
            ),
            invariant_ids=tuple(
                link.value
                for link in subqueue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.INVARIANT_ID
            ),
            reasoning_summary=subqueue_definition.marker_payload.reasoning.summary,
            reasoning_control=subqueue_definition.marker_payload.reasoning.control,
            blocking_dependencies=subqueue_definition.marker_payload.reasoning.blocking_dependencies,
            source_marker_node_id=marker_node_id_by_marker_id.get(
                subqueue_definition.marker_identity,
                "",
            ),
            status_hint=subqueue_definition.status_hint,
        )
    for queue_definition in queue_definitions:
        queue_node_id = state.object_node_ids[queue_definition.queue_id]
        for subqueue_id in queue_definition.subqueue_ids:
            subqueue_node_id = state.object_node_ids[subqueue_id]
            _add_edge(state, "contains", queue_node_id, subqueue_node_id)
            _add_edge(state, "blocks", subqueue_node_id, queue_node_id)


def _path_variants(raw_path: str) -> tuple[str, ...]:
    normalized = raw_path.strip().replace("\\", "/")
    if not normalized:
        return ()
    basename = Path(normalized).name
    if basename and basename != normalized:
        return (normalized, basename)
    return (normalized,)


@dataclass(frozen=True)
class _InvariantGraphNodeIndex:
    by_structural_identity: Mapping[str, str]
    touchsite_by_boundary: Mapping[tuple[str, int, str], str]
    by_path_line_qualname: Mapping[tuple[str, int, str], str]
    by_path_exact: Mapping[str, tuple[str, ...]]
    by_path_basename: Mapping[str, tuple[str, ...]]


def _node_index(state: _InvariantGraphBuildState) -> _InvariantGraphNodeIndex:
    by_structural_identity: dict[str, str] = {}
    touchsite_by_boundary: dict[tuple[str, int, str], str] = {}
    by_path_line_qualname: dict[tuple[str, int, str], str] = {}
    by_path_exact: defaultdict[str, list[str]] = defaultdict(list)
    by_path_basename: defaultdict[str, list[str]] = defaultdict(list)
    for node in state.nodes_by_id.values():
        if node.structural_identity:
            by_structural_identity.setdefault(node.structural_identity, node.node_id)
        if node.rel_path:
            by_path_exact[node.rel_path].append(node.node_id)
            by_path_basename[Path(node.rel_path).name].append(node.node_id)
        if node.rel_path and node.line > 0 and node.qualname:
            by_path_line_qualname.setdefault(
                (node.rel_path, node.line, node.qualname),
                node.node_id,
            )
        if node.node_kind == "synthetic_touchsite":
            touchsite_by_boundary.setdefault(
                _boundary_key(node.rel_path, node.line, node.title),
                node.node_id,
            )
    return _InvariantGraphNodeIndex(
        by_structural_identity=by_structural_identity,
        touchsite_by_boundary=touchsite_by_boundary,
        by_path_line_qualname=by_path_line_qualname,
        by_path_exact={
            key: tuple(_sorted(value))
            for key, value in by_path_exact.items()
        },
        by_path_basename={
            key: tuple(_sorted(value))
            for key, value in by_path_basename.items()
        },
    )


def _match_policy_signal_target(
    *,
    root: Path,
    index: _InvariantGraphNodeIndex,
    violation: Mapping[str, object],
) -> str:
    details = violation.get("details")
    if isinstance(details, Mapping):
        structural_identity = str(details.get("edge_structural_identity", "")).strip()
        if structural_identity:
            node_id = index.by_structural_identity.get(structural_identity)
            if node_id is not None:
                return node_id
        boundary_marker = details.get("boundary_marker")
        if isinstance(boundary_marker, Mapping):
            boundary_name = str(boundary_marker.get("name", "")).strip()
            boundary_line = int(boundary_marker.get("line", 0) or 0)
            for path_variant in _path_variants(
                _normalize_rel_path(root, boundary_marker.get("source"))
            ):
                node_id = index.touchsite_by_boundary.get(
                    _boundary_key(path_variant, boundary_line, boundary_name)
                )
                if node_id is not None:
                    return node_id
    qualname = str(violation.get("qualname", "")).strip()
    line = int(violation.get("line", 0) or 0)
    for path_variant in _path_variants(_normalize_rel_path(root, violation.get("path"))):
        node_id = index.by_path_line_qualname.get((path_variant, line, qualname))
        if node_id is not None:
            return node_id
    return ""


def _add_policy_signal_node(
    state: _InvariantGraphBuildState,
    *,
    domain: str,
    rule_id: str,
    target_node_id: str,
    count: int,
    message: str,
    rel_path: str = "",
    qualname: str = "",
    line: int = 0,
    column: int = 0,
) -> None:
    target_suffix = target_node_id or f"orphan:{rel_path}:{qualname}"
    node_id = f"policy_signal:{stable_hash(domain, rule_id, target_suffix)}"
    title = f"{domain}:{rule_id}"
    node = _synthetic_node(
        node_id=node_id,
        title=title,
        ref_kind="policy_signal",
        value=f"{title}:{target_suffix}",
        policy_ids=(rule_id,),
        reasoning_summary=f"{count} aggregated violation(s): {message}",
        reasoning_control=f"policy_signal.{domain}.{rule_id}",
        node_kind="policy_signal",
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        column=column,
    )
    _add_node(state, node, replace=True)
    _link_node_refs(state, node)
    if target_node_id:
        _add_edge(state, "blocks", node_id, target_node_id)
    else:
        state.diagnostics.append(
            InvariantGraphDiagnostic(
                diagnostic_id=stable_hash("invariant_graph_policy_signal_orphan", node_id),
                severity="warning",
                code="unmatched_policy_signal",
                node_id=node_id,
                raw_dependency="",
                message=(
                    f"{title} at {rel_path}::{qualname} did not resolve to a graph touchsite or work item."
                    if rel_path
                    else f"{title} did not resolve to a graph touchsite or work item."
                ),
            )
        )


def _join_policy_signals(state: _InvariantGraphBuildState) -> None:
    artifact_path = state.root / _AMBIGUITY_ARTIFACT
    if not artifact_path.exists():
        return
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return
    index = _node_index(state)
    aggregates: dict[tuple[str, str, str], dict[str, object]] = {}
    for domain in ("ast", "grade"):
        section = payload.get(domain)
        if not isinstance(section, Mapping):
            continue
        violations = section.get("violations", [])
        if not isinstance(violations, list):
            continue
        for raw_violation in violations:
            if not isinstance(raw_violation, Mapping):
                continue
            rule_id = str(raw_violation.get("rule_id", "")).strip() or f"{domain}.signal"
            target_node_id = _match_policy_signal_target(
                root=state.root,
                index=index,
                violation=raw_violation,
            )
            rel_path = _normalize_rel_path(state.root, raw_violation.get("path"))
            qualname = str(raw_violation.get("qualname", "")).strip()
            aggregate_key = (domain, rule_id, target_node_id)
            entry = aggregates.setdefault(
                aggregate_key,
                {
                    "count": 0,
                    "message": str(raw_violation.get("message", "")).strip(),
                    "source_counts": defaultdict(int),
                    "source_lines": {},
                    "source_columns": {},
                },
            )
            entry["count"] = int(entry["count"]) + 1
            source_key = (rel_path, qualname)
            cast(defaultdict[tuple[str, str], int], entry["source_counts"])[source_key] += 1
            line = int(raw_violation.get("line", 0) or 0)
            column = int(raw_violation.get("column", 0) or 0)
            existing_line = cast(dict[tuple[str, str], int], entry["source_lines"]).get(
                source_key,
                0,
            )
            existing_column = cast(
                dict[tuple[str, str], int],
                entry["source_columns"],
            ).get(source_key, 0)
            cast(dict[tuple[str, str], int], entry["source_lines"])[source_key] = (
                min(existing_line, line) if existing_line > 0 and line > 0 else existing_line or line
            )
            cast(dict[tuple[str, str], int], entry["source_columns"])[source_key] = (
                min(existing_column, column)
                if existing_column > 0 and column > 0
                else existing_column or column
            )
    for (domain, rule_id, target_node_id), data in _sorted(
        list(aggregates.items()),
        key=lambda item: item[0],
    ):
        source_counts = cast(defaultdict[tuple[str, str], int], data["source_counts"])
        source_lines = cast(dict[tuple[str, str], int], data["source_lines"])
        source_columns = cast(dict[tuple[str, str], int], data["source_columns"])
        dominant_source = ("", "")
        if source_counts:
            dominant_source = _sorted(
                list(source_counts),
                key=lambda item: (-source_counts[item], item[0], item[1]),
            )[0]
        rel_path, qualname = dominant_source
        _add_policy_signal_node(
            state,
            domain=domain,
            rule_id=rule_id,
            target_node_id=target_node_id,
            count=int(data["count"]),
            message=str(data["message"]),
            rel_path=rel_path,
            qualname=qualname,
            line=source_lines.get(dominant_source, 0),
            column=source_columns.get(dominant_source, 0),
        )


def _site_span_contains_line(site: Mapping[str, object], line: int) -> bool:
    raw_span = site.get("span")
    if not isinstance(raw_span, list) or len(raw_span) != 4:
        return True
    start = raw_span[0]
    end = raw_span[2]
    if not isinstance(start, int) or not isinstance(end, int):
        return True
    lower = min(start, end)
    upper = max(start, end)
    return lower <= line <= upper


def _match_test_evidence_node_ids(
    *,
    root: Path,
    index: _InvariantGraphNodeIndex,
    state: _InvariantGraphBuildState,
    site: Mapping[str, object],
) -> tuple[str, ...]:
    rel_path = _normalize_rel_path(root, site.get("path"))
    site_qual = str(site.get("qual", "")).strip()
    candidate_ids: list[str] = []
    for path_variant in _path_variants(rel_path):
        candidate_ids.extend(index.by_path_exact.get(path_variant, ()))
        candidate_ids.extend(index.by_path_basename.get(path_variant, ()))
    if not candidate_ids:
        return ()
    matched: list[str] = []
    for node_id in _sorted(list(set(candidate_ids))):
        node = state.nodes_by_id[node_id]
        if site_qual and node.qualname and not site_qual.endswith(node.qualname):
            continue
        if node.line > 0 and not _site_span_contains_line(site, node.line):
            continue
        matched.append(node_id)
    return tuple(matched)


def _join_test_coverage(state: _InvariantGraphBuildState) -> None:
    evidence_path = state.root / _TEST_EVIDENCE_ARTIFACT
    if not evidence_path.exists():
        return
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return
    tests = payload.get("tests", [])
    if not isinstance(tests, list):
        return
    index = _node_index(state)
    for raw_test in tests:
        if not isinstance(raw_test, Mapping):
            continue
        test_id = str(raw_test.get("test_id", "")).strip()
        if not test_id:
            continue
        evidence = raw_test.get("evidence", [])
        if not isinstance(evidence, list):
            continue
        matched_node_ids: set[str] = set()
        for raw_item in evidence:
            if not isinstance(raw_item, Mapping):
                continue
            key = raw_item.get("key")
            if not isinstance(key, Mapping):
                continue
            site = key.get("site")
            if not isinstance(site, Mapping):
                continue
            matched_node_ids.update(
                _match_test_evidence_node_ids(
                    root=state.root,
                    index=index,
                    state=state,
                    site=site,
                )
            )
        if not matched_node_ids:
            continue
        node_id = f"test_case:{stable_hash(test_id)}"
        test_file = _normalize_rel_path(state.root, raw_test.get("file"))
        test_line = int(raw_test.get("line", 0) or 0)
        node = _synthetic_node(
            node_id=node_id,
            title=test_id,
            ref_kind="test_case",
            value=test_id,
            rel_path=test_file,
            qualname=test_id,
            line=test_line,
            reasoning_summary=str(raw_test.get("status", "")).strip(),
            node_kind="test_case",
        )
        _add_node(state, node, replace=True)
        for matched_node_id in _sorted(list(matched_node_ids)):
            _add_edge(state, "covered_by", matched_node_id, node_id)


def _resolve_blocking_dependencies(state: _InvariantGraphBuildState) -> None:
    for node in list(state.nodes_by_id.values()):
        for dependency in node.blocking_dependencies:
            target_id = state.object_node_ids.get(dependency)
            if target_id is None:
                if (
                    dependency in state.declared_workstream_ids
                    or dependency.startswith("PSF-")
                    or dependency.startswith("PRF-")
                ):
                    raise ValueError(
                        "declared workstream blocking dependency did not resolve: "
                        f"{dependency} (from {node.node_id})"
                    )
                state.diagnostics.append(
                    InvariantGraphDiagnostic(
                        diagnostic_id=stable_hash(
                            "invariant_graph_diagnostic",
                            node.node_id,
                            dependency,
                        ),
                        severity="warning",
                        code="unresolved_blocking_dependency",
                        node_id=node.node_id,
                        raw_dependency=dependency,
                        message=(
                            f"{node.node_id} declares blocking dependency {dependency} "
                            "that does not resolve to a graph object_id node."
                        ),
                    )
                )
                continue
            _add_edge(state, "depends_on", node.node_id, target_id)


def build_invariant_graph(root: Path) -> InvariantGraph:
    root = root.resolve()
    state = _new_build_state(root)
    marker_nodes = scan_invariant_markers(root)
    marker_node_id_by_marker_id: dict[str, str] = {}
    for marker_node in marker_nodes:
        graph_node = _node_from_scan_entry(marker_node)
        _add_node(state, graph_node)
        marker_node_id_by_marker_id[graph_node.marker_id] = graph_node.node_id
        _link_node_refs(state, graph_node)
    _enrich_psf_phase5_workstream(
        state,
        marker_node_id_by_marker_id=marker_node_id_by_marker_id,
    )
    _enrich_prf_workstream(
        state,
        marker_node_id_by_marker_id=marker_node_id_by_marker_id,
    )
    _resolve_blocking_dependencies(state)
    _join_policy_signals(state)
    _join_test_coverage(state)
    return InvariantGraph(
        root=str(root),
        workstream_root_ids=tuple(_sorted(state.workstream_root_ids)),
        nodes=tuple(
            _sorted(
                list(state.nodes_by_id.values()),
                key=lambda item: (item.node_kind, item.rel_path, item.line, item.node_id),
            )
        ),
        edges=tuple(
            _sorted(
                state.edges,
                key=lambda item: (item.edge_kind, item.source_id, item.target_id),
            )
        ),
        diagnostics=tuple(
            _sorted(
                state.diagnostics,
                key=lambda item: (item.severity, item.code, item.node_id, item.raw_dependency),
            )
        ),
    )


def _descendant_node_ids(graph: InvariantGraph, node_id: str) -> tuple[str, ...]:
    node_by_id = graph.node_by_id()
    edges_from = graph.edges_from()
    pending = deque([node_id])
    seen: set[str] = set()
    while pending:
        current = pending.popleft()
        if current in seen:
            continue
        seen.add(current)
        for edge in edges_from.get(current, ()):
            if edge.edge_kind != "contains" or edge.target_id not in node_by_id:
                continue
            pending.append(edge.target_id)
    return tuple(_sorted(list(seen)))


def _status_for_projection(
    node: InvariantGraphNode,
    *,
    touchsite_count: int,
) -> str:
    if node.status_hint:
        return node.status_hint
    if touchsite_count > 0:
        return "in_progress"
    return "landed"


def _workstream_signal_ids(graph: InvariantGraph, node_ids: set[str]) -> tuple[str, ...]:
    edges_to = graph.edges_to()
    node_by_id = graph.node_by_id()
    signal_ids = {
        edge.source_id
        for node_id in node_ids
        for edge in edges_to.get(node_id, ())
        if edge.edge_kind == "blocks"
        and node_by_id.get(edge.source_id, None) is not None
        and node_by_id[edge.source_id].node_kind == "policy_signal"
    }
    return tuple(_sorted(list(signal_ids)))


def _workstream_test_case_ids(graph: InvariantGraph, node_ids: set[str]) -> tuple[str, ...]:
    edges_from = graph.edges_from()
    node_by_id = graph.node_by_id()
    test_case_ids = {
        edge.target_id
        for node_id in node_ids
        for edge in edges_from.get(node_id, ())
        if edge.edge_kind == "covered_by"
        and node_by_id.get(edge.target_id, None) is not None
        and node_by_id[edge.target_id].node_kind == "test_case"
    }
    return tuple(_sorted(list(test_case_ids)))


def build_invariant_workstreams(
    graph: InvariantGraph,
    *,
    root: Path | None = None,
) -> InvariantWorkstreamsProjection:
    node_by_id = graph.node_by_id()
    edges_from = graph.edges_from()
    edges_to = graph.edges_to()
    identity_space = PolicyQueueIdentitySpace()
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    doc_paths_by_id = None if root is None else _doc_id_paths(root)
    diagnostics_by_node_id: dict[str, tuple[InvariantGraphDiagnostic, ...]] = {}
    for diagnostic in graph.diagnostics:
        diagnostics_by_node_id.setdefault(diagnostic.node_id, tuple())
    if graph.diagnostics:
        grouped_diagnostics: defaultdict[str, list[InvariantGraphDiagnostic]] = defaultdict(list)
        for diagnostic in graph.diagnostics:
            grouped_diagnostics[diagnostic.node_id].append(diagnostic)
        diagnostics_by_node_id = {
            node_id: tuple(items) for node_id, items in grouped_diagnostics.items()
        }
    descendant_cache: dict[str, tuple[str, ...]] = {}

    def _descendants(node_id: str) -> tuple[str, ...]:
        cached = descendant_cache.get(node_id)
        if cached is not None:
            return cached
        pending = deque([node_id])
        seen: set[str] = set()
        while pending:
            current = pending.popleft()
            if current in seen:
                continue
            seen.add(current)
            for edge in edges_from.get(current, ()):
                if edge.edge_kind != "contains" or edge.target_id not in node_by_id:
                    continue
                pending.append(edge.target_id)
        resolved = tuple(_sorted(list(seen)))
        descendant_cache[node_id] = resolved
        return resolved

    def _signal_ids(node_ids: Iterable[str]) -> tuple[str, ...]:
        signal_ids = {
            edge.source_id
            for node_id in node_ids
            for edge in edges_to.get(node_id, ())
            if edge.edge_kind == "blocks"
            and node_by_id.get(edge.source_id, None) is not None
            and node_by_id[edge.source_id].node_kind == "policy_signal"
        }
        return tuple(_sorted(list(signal_ids)))

    def _test_case_ids(node_ids: Iterable[str]) -> tuple[str, ...]:
        test_case_ids = {
            edge.target_id
            for node_id in node_ids
            for edge in edges_from.get(node_id, ())
            if edge.edge_kind == "covered_by"
            and node_by_id.get(edge.target_id, None) is not None
            and node_by_id[edge.target_id].node_kind == "test_case"
        }
        return tuple(_sorted(list(test_case_ids)))

    def _diagnostic_count(node_ids: Iterable[str]) -> int:
        return sum(len(diagnostics_by_node_id.get(node_id, ())) for node_id in node_ids)

    def _site_ref(token: str) -> SiteReferenceId:
        return identity_space.site_ref_id(token)

    def _structural_ref(token: str) -> StructuralReferenceId:
        return identity_space.structural_ref_id(token)

    def _touchsite_projection(
        *,
        node: InvariantGraphNode,
        touchpoint_node: InvariantGraphNode,
        subqueue_node: InvariantGraphNode,
    ) -> InvariantTouchsiteProjection:
        return InvariantTouchsiteProjection(
            object_id=identity_space.touchsite_id(_primary_object_id(node)),
            touchpoint_id=identity_space.touchpoint_id(_primary_object_id(touchpoint_node)),
            subqueue_id=identity_space.subqueue_id(_primary_object_id(subqueue_node)),
            title=node.title,
            status=_status_for_projection(node, touchsite_count=1),
            rel_path=node.rel_path,
            qualname=node.qualname,
            boundary_name=node.title,
            line=node.line,
            column=node.column,
            node_kind=node.ast_node_kind,
            site_identity=_site_ref(node.site_identity),
            structural_identity=_structural_ref(node.structural_identity),
            seam_class=node.seam_class,
            touchpoint_marker_identity=touchpoint_node.marker_id,
            touchpoint_structural_identity=_structural_ref(
                touchpoint_node.structural_identity
            ),
            subqueue_marker_identity=subqueue_node.marker_id,
            subqueue_structural_identity=_structural_ref(subqueue_node.structural_identity),
            policy_signal_count=len(_signal_ids((node.node_id,))),
            coverage_count=len(_test_case_ids((node.node_id,))),
            diagnostic_count=_diagnostic_count((node.node_id,)),
            object_ids=node.object_ids,
        )

    def _touchpoint_projection(
        *,
        touchpoint_node: InvariantGraphNode,
        subqueue_node: InvariantGraphNode,
    ) -> InvariantTouchpointProjection:
        touchpoint_descendants = set(_descendants(touchpoint_node.node_id))
        touchsites = _sorted(
            [
                node_by_id[node_id]
                for node_id in touchpoint_descendants
                if node_by_id[node_id].node_kind == "synthetic_touchsite"
            ],
            key=lambda item: (
                item.rel_path,
                item.line,
                item.column,
                item.qualname,
            ),
        )

        def _iter_touchsites() -> Iterator[InvariantTouchsiteProjection]:
            for node in touchsites:
                yield _touchsite_projection(
                    node=node,
                    touchpoint_node=touchpoint_node,
                    subqueue_node=subqueue_node,
                )

        return InvariantTouchpointProjection(
            object_id=identity_space.touchpoint_id(_primary_object_id(touchpoint_node)),
            subqueue_id=identity_space.subqueue_id(_primary_object_id(subqueue_node)),
            title=touchpoint_node.title,
            status=_status_for_projection(
                touchpoint_node,
                touchsite_count=len(touchsites),
            ),
            rel_path=touchpoint_node.rel_path,
            site_identity=_site_ref(touchpoint_node.site_identity),
            structural_identity=_structural_ref(touchpoint_node.structural_identity),
            marker_identity=touchpoint_node.marker_id,
            reasoning_summary=touchpoint_node.reasoning_summary,
            reasoning_control=touchpoint_node.reasoning_control,
            blocking_dependencies=touchpoint_node.blocking_dependencies,
            object_ids=touchpoint_node.object_ids,
            touchsite_count=len(touchsites),
            collapsible_touchsite_count=sum(
                1 for node in touchsites if node.seam_class == "collapsible_helper_seam"
            ),
            surviving_touchsite_count=sum(
                1 for node in touchsites if node.seam_class == "surviving_carrier_seam"
            ),
            policy_signal_count=len(_signal_ids(touchpoint_descendants)),
            coverage_count=len(_test_case_ids(touchpoint_descendants)),
            diagnostic_count=_diagnostic_count(touchpoint_descendants),
            touchsites=_stream_from_iterable(_iter_touchsites),
        )

    def _subqueue_projection(
        *,
        subqueue_node: InvariantGraphNode,
    ) -> InvariantSubqueueProjection:
        touchpoint_nodes = _sorted(
            [
                node_by_id[edge.target_id]
                for edge in edges_from.get(subqueue_node.node_id, ())
                if edge.edge_kind == "contains" and edge.target_id in node_by_id
            ],
            key=lambda item: _primary_object_id(item),
        )
        subqueue_descendants = set(_descendants(subqueue_node.node_id))
        subqueue_touchsites = [
            node_by_id[node_id]
            for node_id in subqueue_descendants
            if node_by_id[node_id].node_kind == "synthetic_touchsite"
        ]
        return InvariantSubqueueProjection(
            object_id=identity_space.subqueue_id(_primary_object_id(subqueue_node)),
            title=subqueue_node.title,
            status=_status_for_projection(
                subqueue_node,
                touchsite_count=len(subqueue_touchsites),
            ),
            site_identity=_site_ref(subqueue_node.site_identity),
            structural_identity=_structural_ref(subqueue_node.structural_identity),
            marker_identity=subqueue_node.marker_id,
            reasoning_summary=subqueue_node.reasoning_summary,
            reasoning_control=subqueue_node.reasoning_control,
            blocking_dependencies=subqueue_node.blocking_dependencies,
            object_ids=subqueue_node.object_ids,
            touchpoint_ids=tuple(
                identity_space.touchpoint_id(_primary_object_id(node))
                for node in touchpoint_nodes
            ),
            touchsite_count=len(subqueue_touchsites),
            collapsible_touchsite_count=sum(
                1
                for node in subqueue_touchsites
                if node.seam_class == "collapsible_helper_seam"
            ),
            surviving_touchsite_count=sum(
                1
                for node in subqueue_touchsites
                if node.seam_class == "surviving_carrier_seam"
            ),
            policy_signal_count=len(_signal_ids(subqueue_descendants)),
            coverage_count=len(_test_case_ids(subqueue_descendants)),
            diagnostic_count=_diagnostic_count(subqueue_descendants),
        )

    def _workstream_projection(root_object_id: str) -> InvariantWorkstreamProjection:
        root_node_id = _synthetic_ref_node_id("object_id", root_object_id)
        root_node = node_by_id[root_node_id]
        subqueue_nodes = _sorted(
            [
                node_by_id[edge.target_id]
                for edge in edges_from.get(root_node_id, ())
                if edge.edge_kind == "contains" and edge.target_id in node_by_id
            ],
            key=lambda item: _primary_object_id(item),
        )
        root_descendants = set(_descendants(root_node_id))
        touchsite_nodes = [
            node_by_id[node_id]
            for node_id in root_descendants
            if node_by_id[node_id].node_kind == "synthetic_touchsite"
        ]

        def _iter_subqueues() -> Iterator[InvariantSubqueueProjection]:
            for subqueue_node in subqueue_nodes:
                yield _subqueue_projection(subqueue_node=subqueue_node)

        def _iter_touchpoints() -> Iterator[InvariantTouchpointProjection]:
            for subqueue_node in subqueue_nodes:
                touchpoint_nodes = _sorted(
                    [
                        node_by_id[edge.target_id]
                        for edge in edges_from.get(subqueue_node.node_id, ())
                        if edge.edge_kind == "contains" and edge.target_id in node_by_id
                    ],
                    key=lambda item: _primary_object_id(item),
                )
                for touchpoint_node in touchpoint_nodes:
                    yield _touchpoint_projection(
                        touchpoint_node=touchpoint_node,
                        subqueue_node=subqueue_node,
                    )

        workstream = InvariantWorkstreamProjection(
            object_id=identity_space.workstream_id(root_object_id),
            title=root_node.title,
            status=_status_for_projection(
                root_node,
                touchsite_count=len(touchsite_nodes),
            ),
            site_identity=_site_ref(root_node.site_identity),
            structural_identity=_structural_ref(root_node.structural_identity),
            marker_identity=root_node.marker_id,
            reasoning_summary=root_node.reasoning_summary,
            reasoning_control=root_node.reasoning_control,
            blocking_dependencies=root_node.blocking_dependencies,
            object_ids=root_node.object_ids,
            doc_ids=root_node.doc_ids,
            policy_ids=root_node.policy_ids,
            touchsite_count=len(touchsite_nodes),
            collapsible_touchsite_count=sum(
                1
                for node in touchsite_nodes
                if node.seam_class == "collapsible_helper_seam"
            ),
            surviving_touchsite_count=sum(
                1
                for node in touchsite_nodes
                if node.seam_class == "surviving_carrier_seam"
            ),
            policy_signal_count=len(_signal_ids(root_descendants)),
            coverage_count=len(_test_case_ids(root_descendants)),
            diagnostic_count=_diagnostic_count(root_descendants),
            doc_alignment_summary=None,
            subqueues=_stream_from_iterable(_iter_subqueues),
            touchpoints=_stream_from_iterable(_iter_touchpoints),
        )
        if root is None or doc_paths_by_id is None:
            return workstream
        summary = (
            f"{workstream.object_id.wire()} is {workstream.status} with "
            f"{workstream.touchsite_count} touchsites."
        )
        return replace(
            workstream,
            doc_alignment_summary=_alignment_summary(
                _current_target_doc_alignments(
                    root=root,
                    doc_paths_by_id=doc_paths_by_id,
                    object_id=workstream.object_id.wire(),
                    target_doc_ids=workstream.doc_ids,
                    summary=summary,
                )
            ),
        )

    return InvariantWorkstreamsProjection(
        root=graph.root,
        generated_at_utc=generated_at_utc,
        diagnostics=graph.diagnostics,
        node_lookup=node_by_id,
        workstreams=_stream_from_iterable(
            lambda: (
                _workstream_projection(root_object_id)
                for root_object_id in _sorted(list(graph.workstream_root_ids))
            )
        ),
    )


def build_psf_phase5_projection(
    graph: InvariantGraph,
    *,
    queue_object_id: str = "PSF-007",
) -> dict[str, object]:
    workstreams = build_invariant_workstreams(graph)
    workstream = next(
        (
            item
            for item in workstreams.iter_workstreams()
            if item.object_id.wire() == queue_object_id
        ),
        None,
    )
    if workstream is None:
        raise KeyError(queue_object_id)
    return {
        "queue_id": queue_object_id,
        "title": workstream.title,
        "remaining_touchsite_count": workstream.touchsite_count,
        "collapsible_touchsite_count": workstream.collapsible_touchsite_count,
        "surviving_touchsite_count": workstream.surviving_touchsite_count,
        "subqueues": [
            {
                "subqueue_id": item.object_id.wire(),
                "title": item.title,
                "site_identity": item.site_identity.wire(),
                "structural_identity": item.structural_identity.wire(),
                "marker_identity": item.marker_identity,
                "marker_reason": item.reasoning_summary,
                "reasoning_summary": item.reasoning_summary,
                "reasoning_control": item.reasoning_control,
                "blocking_dependencies": list(item.blocking_dependencies),
                "object_ids": list(item.object_ids),
                "touchpoint_ids": [touchpoint_id.wire() for touchpoint_id in item.touchpoint_ids],
                "touchsite_count": item.touchsite_count,
                "collapsible_touchsite_count": item.collapsible_touchsite_count,
                "surviving_touchsite_count": item.surviving_touchsite_count,
            }
            for item in workstream.iter_subqueues()
        ],
        "touchpoints": [
            {
                "touchpoint_id": item.object_id.wire(),
                "subqueue_id": item.subqueue_id.wire(),
                "title": item.title,
                "rel_path": item.rel_path,
                "site_identity": item.site_identity.wire(),
                "structural_identity": item.structural_identity.wire(),
                "marker_identity": item.marker_identity,
                "marker_reason": item.reasoning_summary,
                "reasoning_summary": item.reasoning_summary,
                "reasoning_control": item.reasoning_control,
                "blocking_dependencies": list(item.blocking_dependencies),
                "object_ids": list(item.object_ids),
                "touchsite_count": item.touchsite_count,
                "collapsible_touchsite_count": item.collapsible_touchsite_count,
                "surviving_touchsite_count": item.surviving_touchsite_count,
                "touchsites": [
                    {
                        "touchsite_id": touchsite.object_id.wire(),
                        "touchpoint_id": touchsite.touchpoint_id.wire(),
                        "subqueue_id": touchsite.subqueue_id.wire(),
                        "rel_path": touchsite.rel_path,
                        "qualname": touchsite.qualname,
                        "boundary_name": touchsite.boundary_name,
                        "line": touchsite.line,
                        "column": touchsite.column,
                        "node_kind": touchsite.node_kind,
                        "site_identity": touchsite.site_identity.wire(),
                        "structural_identity": touchsite.structural_identity.wire(),
                        "seam_class": touchsite.seam_class,
                        "touchpoint_marker_identity": touchsite.touchpoint_marker_identity,
                        "touchpoint_structural_identity": touchsite.touchpoint_structural_identity.wire(),
                        "subqueue_marker_identity": touchsite.subqueue_marker_identity,
                        "subqueue_structural_identity": touchsite.subqueue_structural_identity.wire(),
                        "object_ids": list(touchsite.object_ids),
                    }
                    for touchsite in item.iter_touchsites()
                ],
            }
            for item in workstream.iter_touchpoints()
        ],
    }


def trace_nodes(graph: InvariantGraph, raw_id: str) -> tuple[InvariantGraphNode, ...]:
    nodes = [node for node in graph.nodes if node.matches_raw_id(raw_id)]
    return tuple(_sorted(nodes, key=lambda item: (item.node_kind, item.node_id)))


def blocker_chains(
    graph: InvariantGraph,
    *,
    object_id: str,
) -> dict[str, list[list[str]]]:
    queue_node_id = _synthetic_ref_node_id("object_id", object_id)
    node_by_id = graph.node_by_id()
    edges_from = graph.edges_from()
    if queue_node_id not in node_by_id:
        return {}
    grouped: defaultdict[str, list[list[str]]] = defaultdict(list)
    for subqueue_edge in edges_from.get(queue_node_id, ()):
        if subqueue_edge.edge_kind != "contains":
            continue
        subqueue_node = node_by_id[subqueue_edge.target_id]
        for touchpoint_edge in edges_from.get(subqueue_node.node_id, ()):
            if touchpoint_edge.edge_kind != "contains":
                continue
            touchpoint_node = node_by_id[touchpoint_edge.target_id]
            for touchsite_edge in edges_from.get(touchpoint_node.node_id, ()):
                if touchsite_edge.edge_kind != "contains":
                    continue
                touchsite_node = node_by_id[touchsite_edge.target_id]
                grouped[touchsite_node.seam_class or "unknown"].append(
                    [
                        queue_node_id,
                        subqueue_node.node_id,
                        touchpoint_node.node_id,
                        touchsite_node.node_id,
                    ]
                )
    return {
        key: _sorted(value, key=lambda item: tuple(item))
        for key, value in grouped.items()
    }


def load_invariant_workstreams(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("invariant workstreams payload must be a mapping")
    return payload


def load_invariant_ledger_projections(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("invariant ledger projections payload must be a mapping")
    return payload


def load_invariant_ledger_deltas(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("invariant ledger deltas payload must be a mapping")
    return payload


def load_invariant_ledger_alignments(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("invariant ledger alignments payload must be a mapping")
    return payload


def write_invariant_workstreams(
    path: Path,
    workstreams: InvariantWorkstreamsProjection,
) -> None:
    write_json(path, workstreams.artifact_document())


def write_invariant_ledger_projections(
    path: Path,
    ledger_projections: InvariantLedgerProjections,
) -> None:
    write_json(path, ledger_projections.artifact_document())


def write_invariant_ledger_deltas(
    path: Path,
    ledger_deltas: InvariantLedgerDeltaProjections,
) -> None:
    write_json(path, ledger_deltas.artifact_document())


def write_invariant_ledger_deltas_markdown(
    path: Path,
    ledger_deltas: InvariantLedgerDeltaProjections,
) -> None:
    write_markdown(path, ledger_deltas.markdown_document())


def write_invariant_ledger_alignments(
    path: Path,
    ledger_alignments: InvariantLedgerAlignments,
) -> None:
    write_json(path, ledger_alignments.artifact_document())


def write_invariant_ledger_alignments_markdown(
    path: Path,
    ledger_alignments: InvariantLedgerAlignments,
) -> None:
    def _doc_sections() -> Iterator[ArtifactUnit]:
        grouped: defaultdict[str, list[InvariantLedgerAlignment]] = defaultdict(list)
        for item in ledger_alignments.iter_alignments():
            grouped[item.target_doc_id].append(item)
        for doc_id, items in _sorted(list(grouped.items()), key=lambda item: item[0]):
            yield section(
                identity=ArtifactSourceRef(rel_path="<synthetic>", qualname=doc_id),
                key=doc_id,
                title=doc_id,
                children=lambda items=tuple(
                    _sorted(items, key=lambda item: (item.object_id, item.alignment_status))
                ): _ledger_alignment_markdown_units(alignments=items),
            )

    write_markdown(
        path,
        document(
            identity=ArtifactSourceRef(
                rel_path="<synthetic>",
                qualname="invariant_ledger_alignment_markdown",
            ),
            title="Invariant Ledger Alignments",
            children=_doc_sections,
        ),
    )


def load_invariant_graph(path: Path) -> InvariantGraph:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return InvariantGraph.from_payload(payload)


def write_invariant_graph(path: Path, graph: InvariantGraph) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(graph.as_payload(), indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


__all__ = [
    "InvariantGraph",
    "InvariantGraphDiagnostic",
    "InvariantGraphEdge",
    "InvariantLedgerDelta",
    "InvariantLedgerDeltaProjections",
    "InvariantLedgerAlignment",
    "InvariantLedgerAlignments",
    "InvariantLedgerProjection",
    "InvariantLedgerProjections",
    "InvariantGraphNode",
    "InvariantWorkstreamDrift",
    "blocker_chains",
    "build_invariant_graph",
    "build_invariant_ledger_alignments",
    "build_invariant_ledger_delta_projections",
    "build_invariant_ledger_projections",
    "build_invariant_workstreams",
    "build_psf_phase5_projection",
    "compare_invariant_ledger_projections",
    "compare_invariant_workstreams",
    "load_invariant_workstreams",
    "load_invariant_ledger_alignments",
    "load_invariant_ledger_deltas",
    "load_invariant_ledger_projections",
    "load_invariant_graph",
    "trace_nodes",
    "write_invariant_graph",
    "write_invariant_ledger_alignments",
    "write_invariant_ledger_alignments_markdown",
    "write_invariant_ledger_deltas",
    "write_invariant_ledger_deltas_markdown",
    "write_invariant_ledger_projections",
    "write_invariant_workstreams",
]
