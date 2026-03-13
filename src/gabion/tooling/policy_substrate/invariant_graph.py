from __future__ import annotations

import ast
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import cast

from gabion.analysis.aspf.aspf_lattice_algebra import (
    ReplayableStream,
    canonical_structural_identity,
)
from gabion.analysis.foundation.marker_protocol import SemanticLinkKind
from gabion.order_contract import ordered_or_sorted
from gabion.tooling.policy_substrate.invariant_marker_scan import (
    InvariantMarkerScanNode,
    scan_invariant_markers,
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

    def iter_subqueues(self) -> Iterator[InvariantSubqueueProjection]:
        return iter(self.subqueues)

    def iter_touchpoints(self) -> Iterator[InvariantTouchpointProjection]:
        return iter(self.touchpoints)

    def ranked_touchpoint_cuts(self) -> tuple[InvariantCutCandidate, ...]:
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
            for touchpoint in self.iter_touchpoints()
            if touchpoint.touchsite_count > 0
        ]
        return tuple(_sorted(candidates, key=_cut_sort_key))

    def ranked_subqueue_cuts(self) -> tuple[InvariantCutCandidate, ...]:
        touchpoint_groups: defaultdict[str, list[InvariantTouchpointProjection]] = defaultdict(list)
        for touchpoint in self.iter_touchpoints():
            touchpoint_groups[touchpoint.subqueue_id.wire()].append(touchpoint)
        candidates = []
        for subqueue in self.iter_subqueues():
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

    def health_summary(self) -> InvariantWorkstreamHealthSummary:
        touchpoints = tuple(self.iter_touchpoints())
        touchsites = tuple(
            touchsite
            for touchpoint in touchpoints
            for touchsite in touchpoint.iter_touchsites()
        )
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
            "subqueues": [item.as_payload() for item in self.iter_subqueues()],
            "touchpoints": [item.as_payload() for item in self.iter_touchpoints()],
            "health_summary": health_summary.as_payload(),
            "next_actions": {
                "dominant_blocker_class": self.dominant_blocker_class(),
                "recommended_remediation_family": self.recommended_remediation_family(),
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

    def iter_workstreams(self) -> Iterator[InvariantWorkstreamProjection]:
        return iter(self.workstreams)

    def as_payload(self) -> dict[str, object]:
        workstreams = tuple(self.iter_workstreams())
        return {
            "format_version": _FORMAT_VERSION,
            "generated_at_utc": self.generated_at_utc,
            "root": self.root,
            "workstreams": [item.as_payload() for item in workstreams],
            "counts": {
                "workstream_count": len(workstreams),
            },
        }

    def artifact_document(self) -> ArtifactUnit:
        def _workstream_items() -> Iterator[ArtifactUnit]:
            for workstream in self.iter_workstreams():
                recommended_cut = workstream.recommended_cut()
                recommended_ready_cut = workstream.recommended_ready_cut()
                recommended_coverage_gap_cut = workstream.recommended_coverage_gap_cut()
                recommended_policy_blocked_cut = workstream.recommended_policy_blocked_cut()
                recommended_diagnostic_blocked_cut = (
                    workstream.recommended_diagnostic_blocked_cut()
                )
                remediation_lanes = workstream.remediation_lanes()
                ranked_touchpoint_cuts = workstream.ranked_touchpoint_cuts()
                ranked_subqueue_cuts = workstream.ranked_subqueue_cuts()
                health_summary = workstream.health_summary()
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
                                key="health_summary",
                                title="health_summary",
                                children=lambda health_summary=health_summary: iter(
                                    _payload_to_units(health_summary.as_payload())
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
                                children=lambda recommended_cut=recommended_cut,
                                recommended_ready_cut=recommended_ready_cut,
                                recommended_coverage_gap_cut=recommended_coverage_gap_cut,
                                recommended_policy_blocked_cut=recommended_policy_blocked_cut,
                                recommended_diagnostic_blocked_cut=recommended_diagnostic_blocked_cut,
                                remediation_lanes=remediation_lanes,
                                ranked_touchpoint_cuts=ranked_touchpoint_cuts,
                                ranked_subqueue_cuts=ranked_subqueue_cuts: iter(
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
                                                qualname="recommended_cut",
                                            ),
                                            key="recommended_cut",
                                            title="recommended_cut",
                                            value=(
                                                None
                                                if recommended_cut is None
                                                else recommended_cut.as_payload()
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
                                                if recommended_ready_cut is None
                                                else recommended_ready_cut.as_payload()
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
                                                if recommended_coverage_gap_cut is None
                                                else recommended_coverage_gap_cut.as_payload()
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
                                                if recommended_policy_blocked_cut is None
                                                else recommended_policy_blocked_cut.as_payload()
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
                                                if recommended_diagnostic_blocked_cut is None
                                                else recommended_diagnostic_blocked_cut.as_payload()
                                            ),
                                        ),
                                        bullet_list(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="remediation_lanes",
                                            ),
                                            key="remediation_lanes",
                                            children=lambda remediation_lanes=remediation_lanes: (
                                                list_item(
                                                    identity=ArtifactSourceRef(
                                                        rel_path="<synthetic>",
                                                        qualname=item.remediation_family,
                                                    ),
                                                    children=lambda item=item: iter(
                                                        _payload_to_units(item.as_payload())
                                                    ),
                                                )
                                                for item in remediation_lanes
                                            ),
                                        ),
                                        bullet_list(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="ranked_touchpoint_cuts",
                                            ),
                                            key="ranked_touchpoint_cuts",
                                            children=lambda ranked_touchpoint_cuts=ranked_touchpoint_cuts: (
                                                list_item(
                                                    identity=item.object_id,
                                                    children=lambda item=item: iter(
                                                        _payload_to_units(item.as_payload())
                                                    ),
                                                )
                                                for item in ranked_touchpoint_cuts
                                            ),
                                        ),
                                        bullet_list(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="ranked_subqueue_cuts",
                                            ),
                                            key="ranked_subqueue_cuts",
                                            children=lambda ranked_subqueue_cuts=ranked_subqueue_cuts: (
                                                list_item(
                                                    identity=item.object_id,
                                                    children=lambda item=item: iter(
                                                        _payload_to_units(item.as_payload())
                                                    ),
                                                )
                                                for item in ranked_subqueue_cuts
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
                        identity=ArtifactSourceRef(rel_path="<synthetic>", qualname="counts"),
                        key="counts",
                        title="counts",
                        children=lambda: iter(
                            (
                                scalar(
                                    identity=ArtifactSourceRef(
                                        rel_path="<synthetic>",
                                        qualname="workstream_count",
                                    ),
                                    key="workstream_count",
                                    title="workstream_count",
                                    value=sum(1 for _ in self.iter_workstreams()),
                                ),
                            )
                        ),
                    ),
                )
            ),
        )


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
        def _ledger_items() -> Iterator[ArtifactUnit]:
            for ledger in self.iter_ledgers():
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
                                    value=sum(1 for _ in self.iter_ledgers()),
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
        def _delta_items() -> Iterator[ArtifactUnit]:
            for delta in self.iter_deltas():
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
                                    value=sum(1 for _ in self.iter_deltas()),
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
) -> InvariantLedgerProjections:
    def _ledger_items() -> Iterator[InvariantLedgerProjection]:
        for workstream in workstreams.iter_workstreams():
            recommended_cut = workstream.recommended_cut()
            recommended_ready_cut = workstream.recommended_ready_cut()
            recommended_coverage_gap_cut = workstream.recommended_coverage_gap_cut()
            summary = (
                f"{workstream.object_id.wire()} is {workstream.status} with "
                f"{workstream.touchsite_count} touchsites; dominant blocker "
                f"{workstream.dominant_blocker_class()}; recommended cut "
                f"{recommended_cut.object_id.wire() if recommended_cut is not None else '<none>'}."
            )
            yield InvariantLedgerProjection(
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
                        None
                        if recommended_cut is None
                        else recommended_cut.object_id.wire()
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
) -> None:
    target_suffix = target_node_id or "orphan"
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
                message=f"{title} did not resolve to a graph touchsite or work item.",
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
            aggregate_key = (domain, rule_id, target_node_id)
            entry = aggregates.setdefault(
                aggregate_key,
                {
                    "count": 0,
                    "message": str(raw_violation.get("message", "")).strip(),
                },
            )
            entry["count"] = int(entry["count"]) + 1
    for (domain, rule_id, target_node_id), data in _sorted(
        list(aggregates.items()),
        key=lambda item: item[0],
    ):
        _add_policy_signal_node(
            state,
            domain=domain,
            rule_id=rule_id,
            target_node_id=target_node_id,
            count=int(data["count"]),
            message=str(data["message"]),
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


def build_invariant_workstreams(graph: InvariantGraph) -> InvariantWorkstreamsProjection:
    node_by_id = graph.node_by_id()
    edges_from = graph.edges_from()
    edges_to = graph.edges_to()
    identity_space = PolicyQueueIdentitySpace()
    generated_at_utc = datetime.now(timezone.utc).isoformat()
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

        return InvariantWorkstreamProjection(
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
            subqueues=_stream_from_iterable(_iter_subqueues),
            touchpoints=_stream_from_iterable(_iter_touchpoints),
        )

    return InvariantWorkstreamsProjection(
        root=graph.root,
        generated_at_utc=generated_at_utc,
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
    "InvariantLedgerProjection",
    "InvariantLedgerProjections",
    "InvariantGraphNode",
    "InvariantWorkstreamDrift",
    "blocker_chains",
    "build_invariant_graph",
    "build_invariant_ledger_delta_projections",
    "build_invariant_ledger_projections",
    "build_invariant_workstreams",
    "build_psf_phase5_projection",
    "compare_invariant_ledger_projections",
    "compare_invariant_workstreams",
    "load_invariant_workstreams",
    "load_invariant_ledger_deltas",
    "load_invariant_ledger_projections",
    "load_invariant_graph",
    "trace_nodes",
    "write_invariant_graph",
    "write_invariant_ledger_deltas",
    "write_invariant_ledger_deltas_markdown",
    "write_invariant_ledger_projections",
    "write_invariant_workstreams",
]
