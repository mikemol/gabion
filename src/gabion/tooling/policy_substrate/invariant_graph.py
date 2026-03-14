from __future__ import annotations

import ast
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import cached_property
import json
from pathlib import Path
import re
from typing import cast

from gabion.analysis.aspf.aspf_lattice_algebra import (
    ReplayableStream,
    canonical_structural_identity,
)
from gabion.analysis.foundation.marker_protocol import SemanticLinkKind
from gabion.frontmatter import parse_strict_yaml_frontmatter
from gabion.invariants import never
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
from gabion.tooling.policy_substrate.ranking_signal_dsl import (
    RankingSignalCapture,
    RankingSignalPredicate,
    RankingSignalRule,
    evaluate_ranking_signal_rules,
)
from gabion.tooling.policy_substrate.policy_rule_frontmatter_migration_registry import (
    prf_workstream_registry,
)
from gabion.tooling.policy_substrate.projection_semantic_fragment_phase5_registry import (
    phase5_workstream_registry,
)
from gabion.tooling.policy_substrate.structured_artifact_ingress import (
    StructuredArtifactIdentitySpace,
    TestEvidenceSite,
    load_controller_drift_artifact,
    load_identity_grammar_completion_artifact,
    load_kernel_vm_alignment_artifact,
    load_local_ci_repro_contract_artifact,
    load_docflow_compliance_artifact,
    load_cross_origin_witness_contract_artifact,
    load_docflow_packet_enforcement_artifact,
    load_git_state_artifact,
    load_ingress_merge_parity_artifact,
    load_junit_failure_artifact,
    load_local_repro_closure_ledger_artifact,
    load_test_evidence_artifact,
)
from gabion.tooling.policy_substrate.connectivity_synergy_registry import (
    connectivity_synergy_workstream_registries,
)
from gabion.tooling.policy_substrate.planning_chart import (
    PlanningChartRule,
    PlanningChartSummary,
    build_planning_chart_summary,
)
from gabion.tooling.policy_substrate.site_identity import (
    canonical_site_identity,
    stable_hash,
)
from gabion.tooling.policy_substrate.workstream_registry import (
    RegisteredCounterfactualActionDefinition,
    RegisteredRootDefinition,
    RegisteredSubqueueDefinition,
    RegisteredTouchpointDefinition,
    RegisteredTouchsiteDefinition,
    WorkstreamRegistry,
)

_FORMAT_VERSION = 1
_REPO_ROOT = Path(__file__).resolve().parents[4]
_AMBIGUITY_ARTIFACT = Path("artifacts/out/ambiguity_contract_policy_check.json")
_TEST_EVIDENCE_ARTIFACT = Path("out/test_evidence.json")
_JUNIT_TEST_RESULTS_ARTIFACT = Path("artifacts/test_runs/junit.xml")
_SPPF_DEPENDENCY_GRAPH_ARTIFACT = Path("artifacts/sppf_dependency_graph.json")
_DOCFLOW_COMPLIANCE_ARTIFACT = Path("artifacts/out/docflow_compliance.json")
_DOCFLOW_PACKET_ENFORCEMENT_ARTIFACT = Path(
    "artifacts/out/docflow_packet_enforcement.json"
)
_CONTROLLER_DRIFT_ARTIFACT = Path("artifacts/out/controller_drift.json")
_LOCAL_REPRO_CLOSURE_LEDGER_ARTIFACT = Path(
    "artifacts/out/local_repro_closure_ledger.json"
)
_LOCAL_CI_REPRO_CONTRACT_ARTIFACT = Path(
    "artifacts/out/local_ci_repro_contract.json"
)
_KERNEL_VM_ALIGNMENT_ARTIFACT = Path("artifacts/out/kernel_vm_alignment.json")
_IDENTITY_GRAMMAR_COMPLETION_ARTIFACT = Path(
    "artifacts/out/identity_grammar_completion.json"
)
_GIT_STATE_ARTIFACT = Path("artifacts/out/git_state.json")
_DOCFLOW_REQUIRED_ISSUE_LIFECYCLE_LABELS = (
    "done-on-stage",
    "status/pending-release",
)
_COUNTERFACTUAL_BLOCKED_READINESS_CLASSES = frozenset(
    {"policy_blocked", "diagnostic_blocked", "counterfactual_blocked"}
)
_DIRTY_GIT_STATE_CLASSES = frozenset({"staged", "unstaged", "untracked"})
_IDENTITY_GRAMMAR_TOUCHSITE_IDS = {
    "raw_string_grouping_in_core_queue_logic": "CSA-IDR-TS-013",
    "partial_file_quotient_reification": "CSA-IDR-TS-014",
    "partial_scope_quotient_reification": "CSA-IDR-TS-015",
    "planning_chart_identity_grammar_unintegrated": "CSA-IDR-TS-016",
    "coherence_witness_emission_missing": "CSA-IDR-TS-017",
}
_MECHANICALLY_RECONSTRUCTABLE_PATH_ROOTS = frozenset(
    {
        "artifacts",
        "out",
        "build",
        "dist",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
    }
)
_MECHANICALLY_RECONSTRUCTABLE_PATH_NAMES = frozenset(
    {
        ".coverage",
        "coverage.xml",
        "junit.xml",
    }
)
_GIT_STATE_NODE_KINDS = frozenset(
    {
        "git_state_report",
        "git_head_commit",
        "git_state_entry",
    }
)
_CROSS_ORIGIN_WITNESS_CONTRACT_ARTIFACT = Path(
    "artifacts/out/cross_origin_witness_contract.json"
)
_INGRESS_MERGE_PARITY_ARTIFACT = Path("artifacts/out/ingress_merge_parity.json")
_SPPF_CHECKLIST_DOC = Path("docs/sppf_checklist.md")
_INFLUENCE_INDEX_DOC = Path("docs/influence_index.md")
_DEFAULT_GOVERNANCE_PRIORITY_RANK = 999
_GH_ISSUE_RE = re.compile(r"\bGH-\d+\b")
_TRACEBACK_FRAME_RE = re.compile(
    r'File "(?P<path>[^"]+)", line (?P<line>\d+)(?:, in (?P<qual>.+))?'
)
_ORDERED_LIST_RE = re.compile(r"^\s*(?P<ordinal>\d+)\.\s+(?P<text>.+?)\s*$")
_CHECKLIST_ITEM_RE = re.compile(
    r"^\s*-\s+\[(?P<marker>[ xX~])\]\s+(?P<text>.+?)\s*$"
)
_HEADING_RE = re.compile(r"^(?P<level>#{2,6})\s+(?P<title>.+?)\s*$")


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
    phase_kind: str = ""
    item_kind: str = ""
    source_kind: str = ""
    selection_rank: int = 0
    tracked_node_ids: tuple[str, ...] = ()
    tracked_object_ids: tuple[str, ...] = ()

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
            "phase_kind": self.phase_kind,
            "item_kind": self.item_kind,
            "source_kind": self.source_kind,
            "selection_rank": self.selection_rank,
            "tracked_node_ids": list(self.tracked_node_ids),
            "tracked_object_ids": list(self.tracked_object_ids),
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
class InvariantGraphRankingSignal:
    signal_id: str
    code: str
    node_id: str
    touchpoint_object_id: str
    touchsite_object_id: str
    raw_dependency: str
    score: int
    message: str

    def as_payload(self) -> dict[str, object]:
        return {
            "signal_id": self.signal_id,
            "code": self.code,
            "node_id": self.node_id,
            "touchpoint_object_id": self.touchpoint_object_id,
            "touchsite_object_id": self.touchsite_object_id,
            "raw_dependency": self.raw_dependency,
            "score": self.score,
            "message": self.message,
        }


@dataclass(frozen=True)
class InvariantGraph:
    root: str
    workstream_root_ids: tuple[str, ...]
    nodes: tuple[InvariantGraphNode, ...]
    edges: tuple[InvariantGraphEdge, ...]
    diagnostics: tuple[InvariantGraphDiagnostic, ...]
    ranking_signals: tuple[InvariantGraphRankingSignal, ...] = ()
    planning_chart_summary: PlanningChartSummary | None = None

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
                "ranking_signal_count": len(self.ranking_signals),
                "ranking_signal_score_total": sum(
                    signal.score for signal in self.ranking_signals
                ),
                "node_kind_counts": dict(_sorted(list(node_kind_counts.items()))),
                "edge_kind_counts": dict(_sorted(list(edge_kind_counts.items()))),
            },
            "nodes": [node.as_payload() for node in self.nodes],
            "edges": [edge.as_payload() for edge in self.edges],
            "diagnostics": [item.as_payload() for item in self.diagnostics],
            "ranking_signals": [item.as_payload() for item in self.ranking_signals],
            "planning_chart_summary": (
                None
                if self.planning_chart_summary is None
                else self.planning_chart_summary.as_payload()
            ),
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
                phase_kind=str(item.get("phase_kind", "")),
                item_kind=str(item.get("item_kind", "")),
                source_kind=str(item.get("source_kind", "")),
                selection_rank=int(item.get("selection_rank", 0) or 0),
                tracked_node_ids=tuple(
                    str(value) for value in item.get("tracked_node_ids", [])
                ),
                tracked_object_ids=tuple(
                    str(value) for value in item.get("tracked_object_ids", [])
                ),
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
        ranking_signals = tuple(
            InvariantGraphRankingSignal(
                signal_id=str(item.get("signal_id", "")),
                code=str(item.get("code", "")),
                node_id=str(item.get("node_id", "")),
                touchpoint_object_id=str(item.get("touchpoint_object_id", "")),
                touchsite_object_id=str(item.get("touchsite_object_id", "")),
                raw_dependency=str(item.get("raw_dependency", "")),
                score=int(item.get("score", 0)),
                message=str(item.get("message", "")),
            )
            for item in payload.get("ranking_signals", [])
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
            ranking_signals=ranking_signals,
            planning_chart_summary=(
                None
                if not isinstance(payload.get("planning_chart_summary"), Mapping)
                else PlanningChartSummary.from_payload(
                    cast(Mapping[str, object], payload["planning_chart_summary"])
                )
            ),
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
    failing_test_case_count: int = 0
    test_failure_count: int = 0

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
            "failing_test_case_count": self.failing_test_case_count,
            "test_failure_count": self.test_failure_count,
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
    ranking_signal_count: int = 0
    ranking_signal_score: int = 0
    counterfactual_action_count: int = 0
    viable_counterfactual_action_count: int = 0
    blocked_counterfactual_action_count: int = 0

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
            "ranking_signal_count": self.ranking_signal_count,
            "ranking_signal_score": self.ranking_signal_score,
            "counterfactual_action_count": self.counterfactual_action_count,
            "viable_counterfactual_action_count": self.viable_counterfactual_action_count,
            "blocked_counterfactual_action_count": self.blocked_counterfactual_action_count,
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
    "counterfactual_blocked": 2,
    "policy_blocked": 3,
    "diagnostic_blocked": 4,
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
    counterfactual_action_count: int = 0,
    viable_counterfactual_action_count: int = 0,
    blocked_counterfactual_action_count: int = 0,
) -> str:
    if diagnostic_count > 0:
        return "diagnostic_blocked"
    if policy_signal_count > 0:
        return "policy_blocked"
    if uncovered_touchsite_count > 0:
        return "coverage_gap"
    if (
        counterfactual_action_count > 0
        and viable_counterfactual_action_count <= 0
        and blocked_counterfactual_action_count > 0
    ):
        return "counterfactual_blocked"
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


def _cut_sort_key(
    candidate: InvariantCutCandidate,
) -> tuple[int, int, int, int, int, int, int, int, str]:
    return (
        _READINESS_PRIORITY.get(candidate.readiness_class, 99),
        _CUT_KIND_PRIORITY.get(candidate.cut_kind, 99),
        -candidate.ranking_signal_score,
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
    counterfactual_blocked_touchpoint_cut_count: int
    policy_blocked_touchpoint_cut_count: int
    diagnostic_blocked_touchpoint_cut_count: int
    ready_subqueue_cut_count: int
    coverage_gap_subqueue_cut_count: int
    counterfactual_blocked_subqueue_cut_count: int
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
            "counterfactual_blocked_touchpoint_cut_count": self.counterfactual_blocked_touchpoint_cut_count,
            "policy_blocked_touchpoint_cut_count": self.policy_blocked_touchpoint_cut_count,
            "diagnostic_blocked_touchpoint_cut_count": self.diagnostic_blocked_touchpoint_cut_count,
            "ready_subqueue_cut_count": self.ready_subqueue_cut_count,
            "coverage_gap_subqueue_cut_count": self.coverage_gap_subqueue_cut_count,
            "counterfactual_blocked_subqueue_cut_count": self.counterfactual_blocked_subqueue_cut_count,
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
    workspace_preservation_count: int
    orphaned_workspace_change_count: int
    buckets: tuple[InvariantDiagnosticBucket, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "diagnostic_count": self.diagnostic_count,
            "unmatched_policy_signal_count": self.unmatched_policy_signal_count,
            "unresolved_blocking_dependency_count": self.unresolved_blocking_dependency_count,
            "workspace_preservation_count": self.workspace_preservation_count,
            "orphaned_workspace_change_count": self.orphaned_workspace_change_count,
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


def _margin_pressure(score: int | None) -> str:
    if score is None:
        return "unknown"
    if score <= 0:
        return "blocking"
    if score < 25:
        return "high"
    if score < 100:
        return "moderate"
    return "low"


def _frontier_decision_mode(
    same_class_pressure: str,
    cross_class_pressure: str,
) -> str:
    if same_class_pressure in {"blocking", "high"} and cross_class_pressure in {
        "blocking",
        "high",
    }:
        return "frontier_contested"
    if cross_class_pressure in {"blocking", "high"}:
        return "frontier_watch_cross_class"
    if same_class_pressure in {"blocking", "high"}:
        return "frontier_watch_same_class"
    if same_class_pressure == "moderate" and cross_class_pressure == "moderate":
        return "frontier_watch_balanced"
    if cross_class_pressure == "moderate":
        return "frontier_hold_with_cross_class_watch"
    if same_class_pressure == "moderate":
        return "frontier_hold_with_same_class_watch"
    return "frontier_hold"


def _is_mechanically_reconstructable_git_path(rel_path: str) -> bool:
    if not rel_path:
        return True
    path = Path(rel_path)
    if path.name in _MECHANICALLY_RECONSTRUCTABLE_PATH_NAMES:
        return True
    if path.parts and path.parts[0] in _MECHANICALLY_RECONSTRUCTABLE_PATH_ROOTS:
        return True
    if path.suffix in {".pyc", ".pyo", ".log", ".tmp"}:
        return True
    return False


def _workspace_preservation_action_for_state(state_class: str) -> str:
    return {
        "staged": "validate_commit_graph_participating_change",
        "unstaged": "stage_validate_commit_graph_participating_change",
        "untracked": "adopt_stage_validate_commit_graph_participating_change",
    }.get(state_class, "preserve_graph_participating_change")


def _workspace_preservation_state_from_action(recommended_action: str | None) -> str:
    return {
        "validate_commit_graph_participating_change": "staged",
        "stage_validate_commit_graph_participating_change": "unstaged",
        "adopt_stage_validate_commit_graph_participating_change": "untracked",
    }.get(recommended_action or "", "")


def _workspace_preservation_followup_title(
    *,
    rel_path: str,
    recommended_action: str | None,
) -> str:
    path_label = rel_path or "<unknown>"
    if recommended_action == "validate_commit_graph_participating_change":
        return f"validate and commit graph change at {path_label}"
    if recommended_action == "stage_validate_commit_graph_participating_change":
        return f"stage, validate, and commit graph change at {path_label}"
    if recommended_action == "adopt_stage_validate_commit_graph_participating_change":
        return f"adopt, stage, validate, and commit graph change at {path_label}"
    return f"preserve graph change at {path_label}"


def _workspace_orphan_action_for_state(state_class: str) -> str:
    return {
        "staged": "attribute_validate_commit_orphaned_change",
        "unstaged": "attribute_stage_validate_commit_orphaned_change",
        "untracked": "adopt_attribute_stage_validate_commit_orphaned_change",
    }.get(state_class, "attribute_preserve_orphaned_change")


def _workspace_orphan_state_from_action(recommended_action: str | None) -> str:
    return {
        "attribute_validate_commit_orphaned_change": "staged",
        "attribute_stage_validate_commit_orphaned_change": "unstaged",
        "adopt_attribute_stage_validate_commit_orphaned_change": "untracked",
    }.get(recommended_action or "", "")


def _workspace_orphan_followup_title(
    *,
    rel_path: str,
    recommended_action: str | None,
) -> str:
    path_label = rel_path or "<unknown>"
    if recommended_action == "attribute_validate_commit_orphaned_change":
        return f"attribute, validate, and commit orphaned change at {path_label}"
    if recommended_action == "attribute_stage_validate_commit_orphaned_change":
        return f"attribute, stage, validate, and commit orphaned change at {path_label}"
    if recommended_action == "adopt_attribute_stage_validate_commit_orphaned_change":
        return (
            f"adopt, attribute, stage, validate, and commit orphaned change at {path_label}"
        )
    return f"attribute and preserve orphaned change at {path_label}"


def _cut_frontier_stability_kind(
    same_kind_pressure: str,
    cross_kind_pressure: str,
) -> str:
    if same_kind_pressure in {"blocking", "high"} and cross_kind_pressure in {
        "blocking",
        "high",
    }:
        return "same_kind_and_cross_kind_contested"
    if cross_kind_pressure in {"blocking", "high"}:
        return "cross_kind_contested"
    if same_kind_pressure in {"blocking", "high"}:
        return "same_kind_contested"
    if same_kind_pressure == "moderate" and cross_kind_pressure == "moderate":
        return "balanced_watch"
    if cross_kind_pressure == "moderate":
        return "cross_kind_watch"
    if same_kind_pressure == "moderate":
        return "same_kind_watch"
    return "stable_frontier"


@dataclass(frozen=True)
class InvariantWorkstreamCutTradeoff:
    frontier_cut_kind: str
    frontier_object_id: str
    frontier_title: str
    frontier_readiness_class: str
    frontier_touchsite_count: int
    frontier_surviving_touchsite_count: int
    runner_up_cut_kind: str
    runner_up_object_id: str
    runner_up_title: str
    runner_up_readiness_class: str
    runner_up_touchsite_count: int
    runner_up_surviving_touchsite_count: int
    margin_kind: str
    margin_score: int
    margin_reason: str
    margin_components: tuple[InvariantScoreComponent, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "frontier_cut_kind": self.frontier_cut_kind,
            "frontier_object_id": self.frontier_object_id,
            "frontier_title": self.frontier_title,
            "frontier_readiness_class": self.frontier_readiness_class,
            "frontier_touchsite_count": self.frontier_touchsite_count,
            "frontier_surviving_touchsite_count": self.frontier_surviving_touchsite_count,
            "runner_up_cut_kind": self.runner_up_cut_kind,
            "runner_up_object_id": self.runner_up_object_id,
            "runner_up_title": self.runner_up_title,
            "runner_up_readiness_class": self.runner_up_readiness_class,
            "runner_up_touchsite_count": self.runner_up_touchsite_count,
            "runner_up_surviving_touchsite_count": self.runner_up_surviving_touchsite_count,
            "margin_kind": self.margin_kind,
            "margin_score": self.margin_score,
            "margin_reason": self.margin_reason,
            "margin_components": [
                item.as_payload() for item in self.margin_components
            ],
        }


@dataclass(frozen=True)
class InvariantWorkstreamCutFrontierExplanation:
    frontier_cut_kind: str
    frontier_object_id: str
    frontier_title: str
    frontier_readiness_class: str
    frontier_touchsite_count: int
    frontier_surviving_touchsite_count: int
    same_kind_tradeoff: InvariantWorkstreamCutTradeoff | None
    cross_kind_tradeoff: InvariantWorkstreamCutTradeoff | None
    recommendation_rationale_kind: str
    recommendation_rationale_reason: str
    recommendation_rationale_components: tuple[InvariantScoreComponent, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "frontier_cut_kind": self.frontier_cut_kind,
            "frontier_object_id": self.frontier_object_id,
            "frontier_title": self.frontier_title,
            "frontier_readiness_class": self.frontier_readiness_class,
            "frontier_touchsite_count": self.frontier_touchsite_count,
            "frontier_surviving_touchsite_count": self.frontier_surviving_touchsite_count,
            "same_kind_tradeoff": (
                None if self.same_kind_tradeoff is None else self.same_kind_tradeoff.as_payload()
            ),
            "cross_kind_tradeoff": (
                None
                if self.cross_kind_tradeoff is None
                else self.cross_kind_tradeoff.as_payload()
            ),
            "recommendation_rationale_kind": self.recommendation_rationale_kind,
            "recommendation_rationale_reason": self.recommendation_rationale_reason,
            "recommendation_rationale_components": [
                item.as_payload() for item in self.recommendation_rationale_components
            ],
        }


@dataclass(frozen=True)
class InvariantWorkstreamCutDecisionProtocol:
    frontier_cut_kind: str
    frontier_object_id: str
    frontier_title: str
    frontier_readiness_class: str
    frontier_touchsite_count: int
    frontier_surviving_touchsite_count: int
    decision_mode: str
    decision_reason: str
    same_kind_pressure: str
    cross_kind_pressure: str
    decision_components: tuple[InvariantScoreComponent, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "frontier_cut_kind": self.frontier_cut_kind,
            "frontier_object_id": self.frontier_object_id,
            "frontier_title": self.frontier_title,
            "frontier_readiness_class": self.frontier_readiness_class,
            "frontier_touchsite_count": self.frontier_touchsite_count,
            "frontier_surviving_touchsite_count": self.frontier_surviving_touchsite_count,
            "decision_mode": self.decision_mode,
            "decision_reason": self.decision_reason,
            "same_kind_pressure": self.same_kind_pressure,
            "cross_kind_pressure": self.cross_kind_pressure,
            "decision_components": [
                item.as_payload() for item in self.decision_components
            ],
        }


@dataclass(frozen=True)
class InvariantWorkstreamCutFrontierStability:
    frontier_cut_kind: str
    frontier_object_id: str
    frontier_title: str
    frontier_readiness_class: str
    frontier_touchsite_count: int
    frontier_surviving_touchsite_count: int
    stability_kind: str
    stability_reason: str
    same_kind_pressure: str
    cross_kind_pressure: str
    stability_components: tuple[InvariantScoreComponent, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "frontier_cut_kind": self.frontier_cut_kind,
            "frontier_object_id": self.frontier_object_id,
            "frontier_title": self.frontier_title,
            "frontier_readiness_class": self.frontier_readiness_class,
            "frontier_touchsite_count": self.frontier_touchsite_count,
            "frontier_surviving_touchsite_count": self.frontier_surviving_touchsite_count,
            "stability_kind": self.stability_kind,
            "stability_reason": self.stability_reason,
            "same_kind_pressure": self.same_kind_pressure,
            "cross_kind_pressure": self.cross_kind_pressure,
            "stability_components": [
                item.as_payload() for item in self.stability_components
            ],
        }


def _cut_tradeoff_components(
    *,
    frontier: InvariantCutCandidate,
    runner_up: InvariantCutCandidate,
) -> tuple[InvariantScoreComponent, ...]:
    components: list[InvariantScoreComponent] = []

    readiness_gap = max(
        0,
        _READINESS_PRIORITY.get(runner_up.readiness_class, 99)
        - _READINESS_PRIORITY.get(frontier.readiness_class, 99),
    )
    if readiness_gap > 0:
        components.append(
            InvariantScoreComponent(
                kind="readiness_priority_gap",
                score=readiness_gap,
                rationale=f"{frontier.readiness_class}->{runner_up.readiness_class}",
            )
        )

    cut_kind_gap = max(
        0,
        _CUT_KIND_PRIORITY.get(runner_up.cut_kind, 99)
        - _CUT_KIND_PRIORITY.get(frontier.cut_kind, 99),
    )
    if cut_kind_gap > 0:
        components.append(
            InvariantScoreComponent(
                kind="cut_kind_priority_gap",
                score=cut_kind_gap,
                rationale=f"{frontier.cut_kind}->{runner_up.cut_kind}",
            )
        )

    ranking_signal_gap = max(
        0,
        frontier.ranking_signal_score - runner_up.ranking_signal_score,
    )
    if ranking_signal_gap > 0:
        components.append(
            InvariantScoreComponent(
                kind="ranking_signal_score_gap",
                score=ranking_signal_gap,
                rationale=(
                    f"{frontier.ranking_signal_score}->"
                    f"{runner_up.ranking_signal_score}"
                ),
            )
        )

    count_specs = (
        ("touchsite_count_gap", frontier.touchsite_count, runner_up.touchsite_count),
        (
            "surviving_touchsite_count_gap",
            frontier.surviving_touchsite_count,
            runner_up.surviving_touchsite_count,
        ),
        ("policy_signal_count_gap", frontier.policy_signal_count, runner_up.policy_signal_count),
        ("diagnostic_count_gap", frontier.diagnostic_count, runner_up.diagnostic_count),
        (
            "uncovered_touchsite_count_gap",
            frontier.uncovered_touchsite_count,
            runner_up.uncovered_touchsite_count,
        ),
    )
    for kind, frontier_value, runner_up_value in count_specs:
        gap = max(0, runner_up_value - frontier_value)
        if gap <= 0:
            continue
        components.append(
            InvariantScoreComponent(
                kind=kind,
                score=gap,
                rationale=f"{frontier_value}->{runner_up_value}",
            )
        )

    if components:
        return tuple(components)
    return (
        InvariantScoreComponent(
            kind="identity_tiebreak",
            score=1,
            rationale=(
                f"{encode_policy_queue_identity(frontier.object_id)}->"
                f"{encode_policy_queue_identity(runner_up.object_id)}"
            ),
        ),
    )


def _cut_tradeoff(
    *,
    frontier: InvariantCutCandidate,
    runner_up: InvariantCutCandidate,
) -> InvariantWorkstreamCutTradeoff:
    components = _cut_tradeoff_components(frontier=frontier, runner_up=runner_up)
    lead_component = components[0]
    return InvariantWorkstreamCutTradeoff(
        frontier_cut_kind=frontier.cut_kind,
        frontier_object_id=encode_policy_queue_identity(frontier.object_id),
        frontier_title=frontier.title,
        frontier_readiness_class=frontier.readiness_class,
        frontier_touchsite_count=frontier.touchsite_count,
        frontier_surviving_touchsite_count=frontier.surviving_touchsite_count,
        runner_up_cut_kind=runner_up.cut_kind,
        runner_up_object_id=encode_policy_queue_identity(runner_up.object_id),
        runner_up_title=runner_up.title,
        runner_up_readiness_class=runner_up.readiness_class,
        runner_up_touchsite_count=runner_up.touchsite_count,
        runner_up_surviving_touchsite_count=runner_up.surviving_touchsite_count,
        margin_kind=lead_component.kind,
        margin_score=lead_component.score,
        margin_reason=lead_component.rationale,
        margin_components=components,
    )


def _cut_tradeoff_pressure(
    tradeoff: InvariantWorkstreamCutTradeoff | None,
) -> str:
    if tradeoff is None:
        return "none"
    if tradeoff.margin_kind == "identity_tiebreak":
        return "blocking"
    if tradeoff.margin_kind in {"readiness_priority_gap", "cut_kind_priority_gap"}:
        return "low"
    if tradeoff.margin_score <= 1:
        return "high"
    if tradeoff.margin_score <= 3:
        return "moderate"
    return "low"


@dataclass(frozen=True)
class InvariantRepoFollowupCohortMember:
    followup_family: str
    followup_class: str
    action_kind: str
    object_id: str | None
    owner_root_object_id: str | None
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
            "owner_root_object_id": self.owner_root_object_id,
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
    owner_root_object_id: str | None
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
    rel_path: str = ""
    qualname: str = ""

    def as_payload(self) -> dict[str, object]:
        return {
            "followup_family": self.followup_family,
            "action_kind": self.action_kind,
            "priority_rank": self.priority_rank,
            "object_id": self.object_id,
            "owner_object_id": self.owner_object_id,
            "owner_root_object_id": self.owner_root_object_id,
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
            "rel_path": self.rel_path,
            "qualname": self.qualname,
            "count": self.count,
        }


@dataclass(frozen=True)
class InvariantRepoFollowupLane:
    followup_family: str
    followup_class: str
    action_count: int
    root_object_ids: tuple[str, ...]
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
            "root_object_ids": list(self.root_object_ids),
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
class InvariantWorkspaceCommitUnit:
    followup_family: str
    diagnostic_code: str
    recommended_action: str | None
    state_class: str
    owner_scope_kind: str
    root_object_ids: tuple[str, ...]
    owner_seed_path: str | None
    owner_resolution_kind: str | None
    action_count: int
    rel_paths: tuple[str, ...]
    qualnames: tuple[str, ...]
    utility_score: int
    utility_reason: str
    selection_rank: int
    opportunity_cost_score: int
    opportunity_cost_reason: str
    best_followup: InvariantRepoFollowupAction

    def as_payload(self) -> dict[str, object]:
        return {
            "followup_family": self.followup_family,
            "diagnostic_code": self.diagnostic_code,
            "recommended_action": self.recommended_action,
            "state_class": self.state_class,
            "owner_scope_kind": self.owner_scope_kind,
            "root_object_ids": list(self.root_object_ids),
            "owner_seed_path": self.owner_seed_path,
            "owner_resolution_kind": self.owner_resolution_kind,
            "action_count": self.action_count,
            "rel_paths": list(self.rel_paths),
            "qualnames": list(self.qualnames),
            "utility_score": self.utility_score,
            "utility_reason": self.utility_reason,
            "selection_rank": self.selection_rank,
            "opportunity_cost_score": self.opportunity_cost_score,
            "opportunity_cost_reason": self.opportunity_cost_reason,
            "best_followup": self.best_followup.as_payload(),
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
class InvariantRepoFollowupDecisionProtocol:
    frontier_followup_family: str
    frontier_followup_class: str
    frontier_action_kind: str
    frontier_object_id: str | None
    frontier_diagnostic_code: str | None
    frontier_target_doc_id: str | None
    frontier_policy_ids: tuple[str, ...]
    frontier_utility_score: int
    frontier_utility_reason: str
    decision_mode: str
    decision_reason: str
    same_class_pressure: str
    cross_class_pressure: str
    decision_components: tuple[InvariantScoreComponent, ...]

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
            "decision_mode": self.decision_mode,
            "decision_reason": self.decision_reason,
            "same_class_pressure": self.same_class_pressure,
            "cross_class_pressure": self.cross_class_pressure,
            "decision_components": [
                item.as_payload() for item in self.decision_components
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
    ranking_signal_count: int = 0
    ranking_signal_score: int = 0
    failing_test_case_count: int = 0
    test_failure_count: int = 0
    counterfactual_action_count: int = 0
    viable_counterfactual_action_count: int = 0
    blocked_counterfactual_action_count: int = 0

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
            "ranking_signal_count": self.ranking_signal_count,
            "ranking_signal_score": self.ranking_signal_score,
            "failing_test_case_count": self.failing_test_case_count,
            "test_failure_count": self.test_failure_count,
            "counterfactual_action_count": self.counterfactual_action_count,
            "viable_counterfactual_action_count": self.viable_counterfactual_action_count,
            "blocked_counterfactual_action_count": self.blocked_counterfactual_action_count,
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
    ranking_signal_count: int = 0
    ranking_signal_score: int = 0
    failing_test_case_count: int = 0
    test_failure_count: int = 0
    counterfactual_action_count: int = 0
    viable_counterfactual_action_count: int = 0
    blocked_counterfactual_action_count: int = 0

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
            "ranking_signal_count": self.ranking_signal_count,
            "ranking_signal_score": self.ranking_signal_score,
            "failing_test_case_count": self.failing_test_case_count,
            "test_failure_count": self.test_failure_count,
            "counterfactual_action_count": self.counterfactual_action_count,
            "viable_counterfactual_action_count": self.viable_counterfactual_action_count,
            "blocked_counterfactual_action_count": self.blocked_counterfactual_action_count,
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
    ranking_signal_count: int = 0
    ranking_signal_score: int = 0
    doc_alignment_summary: InvariantLedgerAlignmentSummary | None = None
    failing_test_case_count: int = 0
    test_failure_count: int = 0
    counterfactual_action_count: int = 0
    viable_counterfactual_action_count: int = 0
    blocked_counterfactual_action_count: int = 0

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
                ranking_signal_count=touchpoint.ranking_signal_count,
                ranking_signal_score=touchpoint.ranking_signal_score,
                counterfactual_action_count=touchpoint.counterfactual_action_count,
                viable_counterfactual_action_count=touchpoint.viable_counterfactual_action_count,
                blocked_counterfactual_action_count=touchpoint.blocked_counterfactual_action_count,
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
                    counterfactual_action_count=touchpoint.counterfactual_action_count,
                    viable_counterfactual_action_count=touchpoint.viable_counterfactual_action_count,
                    blocked_counterfactual_action_count=touchpoint.blocked_counterfactual_action_count,
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
                ranking_signal_count=subqueue.ranking_signal_count,
                ranking_signal_score=subqueue.ranking_signal_score,
                counterfactual_action_count=subqueue.counterfactual_action_count,
                viable_counterfactual_action_count=subqueue.viable_counterfactual_action_count,
                blocked_counterfactual_action_count=subqueue.blocked_counterfactual_action_count,
                covered_touchsite_count=len(touchsites) - uncovered_touchsite_count,
                uncovered_touchsite_count=uncovered_touchsite_count,
                readiness_class=_cut_readiness_class(
                    policy_signal_count=subqueue.policy_signal_count,
                    diagnostic_count=subqueue.diagnostic_count,
                    uncovered_touchsite_count=uncovered_touchsite_count,
                    counterfactual_action_count=subqueue.counterfactual_action_count,
                    viable_counterfactual_action_count=subqueue.viable_counterfactual_action_count,
                    blocked_counterfactual_action_count=subqueue.blocked_counterfactual_action_count,
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

    def recommended_counterfactual_blocked_cut(self) -> InvariantCutCandidate | None:
        return self._recommended_cut_for_readiness("counterfactual_blocked")

    def recommended_policy_blocked_cut(self) -> InvariantCutCandidate | None:
        return self._recommended_cut_for_readiness("policy_blocked")

    def recommended_diagnostic_blocked_cut(self) -> InvariantCutCandidate | None:
        return self._recommended_cut_for_readiness("diagnostic_blocked")

    def ranked_cuts(self) -> tuple[InvariantCutCandidate, ...]:
        return self._ranked_cuts

    @cached_property
    def _ranked_cuts(self) -> tuple[InvariantCutCandidate, ...]:
        return tuple(
            _sorted(
                [
                    *self.ranked_touchpoint_cuts(),
                    *self.ranked_subqueue_cuts(),
                ],
                key=_cut_sort_key,
            )
        )

    def recommended_cut(self) -> InvariantCutCandidate | None:
        ranked_cuts = self.ranked_cuts()
        if not ranked_cuts:
            if (
                self.touchsite_count > 0
                or self.policy_signal_count > 0
                or self.diagnostic_count > 0
            ):
                never(
                    reasoning={
                        "summary": (
                            "Active workstream planning must resolve a concrete cut "
                            "frontier instead of emitting an empty ranked-cut set."
                        ),
                        "control": "invariant_graph.workstream.requires_recommended_cut",
                        "blocking_dependencies": (self.object_id.wire(),),
                    },
                    workstream_id=self.object_id.wire(),
                    touchsite_count=self.touchsite_count,
                    policy_signal_count=self.policy_signal_count,
                    diagnostic_count=self.diagnostic_count,
                )
                return None  # pragma: no cover - never() raises
            return None
        return ranked_cuts[0]

    def recommended_cut_same_kind_tradeoff(
        self,
    ) -> InvariantWorkstreamCutTradeoff | None:
        return self._recommended_cut_same_kind_tradeoff

    @cached_property
    def _recommended_cut_same_kind_tradeoff(
        self,
    ) -> InvariantWorkstreamCutTradeoff | None:
        frontier = self.recommended_cut()
        if frontier is None:
            return None
        runner_up = next(
            (
                item
                for item in self.ranked_cuts()
                if item.object_id != frontier.object_id
                and item.cut_kind == frontier.cut_kind
            ),
            None,
        )
        if runner_up is None:
            return None
        return _cut_tradeoff(frontier=frontier, runner_up=runner_up)

    def recommended_cut_cross_kind_tradeoff(
        self,
    ) -> InvariantWorkstreamCutTradeoff | None:
        return self._recommended_cut_cross_kind_tradeoff

    @cached_property
    def _recommended_cut_cross_kind_tradeoff(
        self,
    ) -> InvariantWorkstreamCutTradeoff | None:
        frontier = self.recommended_cut()
        if frontier is None:
            return None
        runner_up = next(
            (
                item
                for item in self.ranked_cuts()
                if item.cut_kind != frontier.cut_kind
            ),
            None,
        )
        if runner_up is None:
            return None
        return _cut_tradeoff(frontier=frontier, runner_up=runner_up)

    def recommended_cut_frontier_explanation(
        self,
    ) -> InvariantWorkstreamCutFrontierExplanation | None:
        return self._recommended_cut_frontier_explanation

    @cached_property
    def _recommended_cut_frontier_explanation(
        self,
    ) -> InvariantWorkstreamCutFrontierExplanation | None:
        frontier = self.recommended_cut()
        if frontier is None:
            return None
        same_kind_tradeoff = self.recommended_cut_same_kind_tradeoff()
        cross_kind_tradeoff = self.recommended_cut_cross_kind_tradeoff()
        same_kind_pressure = _cut_tradeoff_pressure(same_kind_tradeoff)
        cross_kind_pressure = _cut_tradeoff_pressure(cross_kind_tradeoff)
        return InvariantWorkstreamCutFrontierExplanation(
            frontier_cut_kind=frontier.cut_kind,
            frontier_object_id=encode_policy_queue_identity(frontier.object_id),
            frontier_title=frontier.title,
            frontier_readiness_class=frontier.readiness_class,
            frontier_touchsite_count=frontier.touchsite_count,
            frontier_surviving_touchsite_count=frontier.surviving_touchsite_count,
            same_kind_tradeoff=same_kind_tradeoff,
            cross_kind_tradeoff=cross_kind_tradeoff,
            recommendation_rationale_kind=(
                f"same_kind_{same_kind_pressure}__cross_kind_{cross_kind_pressure}"
            ),
            recommendation_rationale_reason=(
                f"same_kind_pressure:{same_kind_pressure}:"
                f"{None if same_kind_tradeoff is None else same_kind_tradeoff.margin_kind}"
                f"|cross_kind_pressure:{cross_kind_pressure}:"
                f"{None if cross_kind_tradeoff is None else cross_kind_tradeoff.margin_kind}"
            ),
            recommendation_rationale_components=(
                InvariantScoreComponent(
                    kind="same_kind_pressure",
                    score=0 if same_kind_tradeoff is None else same_kind_tradeoff.margin_score,
                    rationale=same_kind_pressure,
                ),
                InvariantScoreComponent(
                    kind="cross_kind_pressure",
                    score=0 if cross_kind_tradeoff is None else cross_kind_tradeoff.margin_score,
                    rationale=cross_kind_pressure,
                ),
            ),
        )

    def recommended_cut_decision_protocol(
        self,
    ) -> InvariantWorkstreamCutDecisionProtocol | None:
        return self._recommended_cut_decision_protocol

    @cached_property
    def _recommended_cut_decision_protocol(
        self,
    ) -> InvariantWorkstreamCutDecisionProtocol | None:
        explanation = self.recommended_cut_frontier_explanation()
        if explanation is None:
            return None
        same_kind_pressure = _cut_tradeoff_pressure(explanation.same_kind_tradeoff)
        cross_kind_pressure = _cut_tradeoff_pressure(explanation.cross_kind_tradeoff)
        return InvariantWorkstreamCutDecisionProtocol(
            frontier_cut_kind=explanation.frontier_cut_kind,
            frontier_object_id=explanation.frontier_object_id,
            frontier_title=explanation.frontier_title,
            frontier_readiness_class=explanation.frontier_readiness_class,
            frontier_touchsite_count=explanation.frontier_touchsite_count,
            frontier_surviving_touchsite_count=explanation.frontier_surviving_touchsite_count,
            decision_mode=_frontier_decision_mode(
                same_class_pressure=same_kind_pressure,
                cross_class_pressure=cross_kind_pressure,
            ),
            decision_reason=(
                f"same_kind_pressure:{same_kind_pressure}:"
                f"{None if explanation.same_kind_tradeoff is None else explanation.same_kind_tradeoff.margin_kind}"
                f"|cross_kind_pressure:{cross_kind_pressure}:"
                f"{None if explanation.cross_kind_tradeoff is None else explanation.cross_kind_tradeoff.margin_kind}"
            ),
            same_kind_pressure=same_kind_pressure,
            cross_kind_pressure=cross_kind_pressure,
            decision_components=(
                InvariantScoreComponent(
                    kind="same_kind_pressure",
                    score=(
                        0
                        if explanation.same_kind_tradeoff is None
                        else explanation.same_kind_tradeoff.margin_score
                    ),
                    rationale=same_kind_pressure,
                ),
                InvariantScoreComponent(
                    kind="cross_kind_pressure",
                    score=(
                        0
                        if explanation.cross_kind_tradeoff is None
                        else explanation.cross_kind_tradeoff.margin_score
                    ),
                    rationale=cross_kind_pressure,
                ),
            ),
        )

    def recommended_cut_frontier_stability(
        self,
    ) -> InvariantWorkstreamCutFrontierStability | None:
        return self._recommended_cut_frontier_stability

    @cached_property
    def _recommended_cut_frontier_stability(
        self,
    ) -> InvariantWorkstreamCutFrontierStability | None:
        explanation = self.recommended_cut_frontier_explanation()
        if explanation is None:
            return None
        same_kind_pressure = _cut_tradeoff_pressure(explanation.same_kind_tradeoff)
        cross_kind_pressure = _cut_tradeoff_pressure(explanation.cross_kind_tradeoff)
        return InvariantWorkstreamCutFrontierStability(
            frontier_cut_kind=explanation.frontier_cut_kind,
            frontier_object_id=explanation.frontier_object_id,
            frontier_title=explanation.frontier_title,
            frontier_readiness_class=explanation.frontier_readiness_class,
            frontier_touchsite_count=explanation.frontier_touchsite_count,
            frontier_surviving_touchsite_count=explanation.frontier_surviving_touchsite_count,
            stability_kind=_cut_frontier_stability_kind(
                same_kind_pressure=same_kind_pressure,
                cross_kind_pressure=cross_kind_pressure,
            ),
            stability_reason=(
                f"same_kind_pressure:{same_kind_pressure}"
                f"|cross_kind_pressure:{cross_kind_pressure}"
            ),
            same_kind_pressure=same_kind_pressure,
            cross_kind_pressure=cross_kind_pressure,
            stability_components=(
                InvariantScoreComponent(
                    kind="same_kind_pressure",
                    score=(
                        0
                        if explanation.same_kind_tradeoff is None
                        else explanation.same_kind_tradeoff.margin_score
                    ),
                    rationale=same_kind_pressure,
                ),
                InvariantScoreComponent(
                    kind="cross_kind_pressure",
                    score=(
                        0
                        if explanation.cross_kind_tradeoff is None
                        else explanation.cross_kind_tradeoff.margin_score
                    ),
                    rationale=cross_kind_pressure,
                ),
            ),
        )

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
        if lane.best_target_doc_id is None:
            never(
                reasoning={
                    "summary": (
                        "Documentation followup lanes must resolve a concrete target "
                        "doc before they are projected into next-action outputs."
                    ),
                    "control": "invariant_graph.workstream.documentation_lane_requires_target_doc",
                    "blocking_dependencies": (self.object_id.wire(),),
                },
                workstream_id=self.object_id.wire(),
                alignment_status=lane.alignment_status,
                recommended_action=lane.recommended_action,
                misaligned_target_doc_count=lane.misaligned_target_doc_count,
            )
            return None  # pragma: no cover - never() raises
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
                never(
                    reasoning={
                        "summary": (
                            "Remediation lanes must resolve a best cut before they are "
                            "projected into workstream followups."
                        ),
                        "control": "invariant_graph.workstream.remediation_lane_requires_best_cut",
                        "blocking_dependencies": (self.object_id.wire(),),
                    },
                    workstream_id=self.object_id.wire(),
                    remediation_family=lane.remediation_family,
                    blocker_class=lane.blocker_class,
                    touchsite_count=lane.touchsite_count,
                    touchpoint_cut_count=lane.touchpoint_cut_count,
                    subqueue_cut_count=lane.subqueue_cut_count,
                )
                continue  # pragma: no cover - never() raises
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
            documentation_lane = self.documentation_followup_lane()
            if self.touchsite_count > 0 or documentation_lane is not None:
                never(
                    reasoning={
                        "summary": (
                            "Active workstream planning must resolve a recommended "
                            "followup instead of emitting an empty followup set."
                        ),
                        "control": "invariant_graph.workstream.requires_recommended_followup",
                        "blocking_dependencies": (self.object_id.wire(),),
                    },
                    workstream_id=self.object_id.wire(),
                    touchsite_count=self.touchsite_count,
                    has_documentation_lane=documentation_lane is not None,
                )
                return None  # pragma: no cover - never() raises
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
            counterfactual_blocked_touchpoint_cut_count=sum(
                1
                for candidate in touchpoint_cuts
                if candidate.readiness_class == "counterfactual_blocked"
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
            counterfactual_blocked_subqueue_cut_count=sum(
                1
                for candidate in subqueue_cuts
                if candidate.readiness_class == "counterfactual_blocked"
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
            (
                "counterfactual_blocked",
                (
                    health_summary.counterfactual_blocked_touchpoint_cut_count
                    + health_summary.counterfactual_blocked_subqueue_cut_count
                ),
            ),
        )
        blocker_priority = {
            "diagnostic_blocked": 0,
            "policy_blocked": 1,
            "coverage_gap": 2,
            "counterfactual_blocked": 3,
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
        if self.recommended_counterfactual_blocked_cut() is not None:
            return "counterfactual_blocked"
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
            (
                "counterfactual_blocked",
                "counterfactual_blocked",
                0,
            ),
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
            if not matching_touchpoint_cuts and not matching_subqueue_cuts:
                continue
            if blocker_class == "counterfactual_blocked":
                touchsite_count = len(
                    {
                        touchsite_id.wire()
                        for candidate in (*matching_touchpoint_cuts, *matching_subqueue_cuts)
                        for touchsite_id in candidate.touchsite_ids
                    }
                )
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
        if not misaligned_target_doc_ids:
            never(
                reasoning={
                    "summary": (
                        "Documentation alignment planning must resolve at least one "
                        "misaligned target doc before it emits a documentation lane."
                    ),
                    "control": "invariant_graph.workstream.documentation_lane_requires_misaligned_target",
                    "blocking_dependencies": (self.object_id.wire(),),
                },
                workstream_id=self.object_id.wire(),
                recommended_action=(
                    self.doc_alignment_summary.recommended_doc_alignment_action
                ),
                target_doc_count=self.doc_alignment_summary.target_doc_count,
            )
            return None  # pragma: no cover - never() raises
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
        recommended_counterfactual_blocked_cut = (
            self.recommended_counterfactual_blocked_cut()
        )
        recommended_policy_blocked_cut = self.recommended_policy_blocked_cut()
        recommended_diagnostic_blocked_cut = self.recommended_diagnostic_blocked_cut()
        recommended_cut_frontier_explanation = (
            self.recommended_cut_frontier_explanation()
        )
        recommended_cut_decision_protocol = self.recommended_cut_decision_protocol()
        recommended_cut_frontier_stability = (
            self.recommended_cut_frontier_stability()
        )
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
            "ranking_signal_count": self.ranking_signal_count,
            "ranking_signal_score": self.ranking_signal_score,
            "failing_test_case_count": self.failing_test_case_count,
            "test_failure_count": self.test_failure_count,
            "counterfactual_action_count": self.counterfactual_action_count,
            "viable_counterfactual_action_count": self.viable_counterfactual_action_count,
            "blocked_counterfactual_action_count": self.blocked_counterfactual_action_count,
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
                "recommended_counterfactual_blocked_cut": (
                    recommended_counterfactual_blocked_cut.as_payload()
                    if recommended_counterfactual_blocked_cut is not None
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
                "recommended_cut_frontier_explanation": (
                    None
                    if recommended_cut_frontier_explanation is None
                    else recommended_cut_frontier_explanation.as_payload()
                ),
                "recommended_cut_decision_protocol": (
                    None
                    if recommended_cut_decision_protocol is None
                    else recommended_cut_decision_protocol.as_payload()
                ),
                "recommended_cut_frontier_stability": (
                    None
                    if recommended_cut_frontier_stability is None
                    else recommended_cut_frontier_stability.as_payload()
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
    planning_chart_summary: PlanningChartSummary | None = None
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
        workspace_preservation_count = 0
        orphaned_workspace_change_count = 0
        for diagnostic in self.diagnostics:
            bucket_counts[(diagnostic.code, diagnostic.severity)] += 1
            if diagnostic.code == "unmatched_policy_signal":
                unmatched_policy_signal_count += 1
            if diagnostic.code == "unresolved_blocking_dependency":
                unresolved_blocking_dependency_count += 1
            if diagnostic.code == "workspace_preservation_needed":
                workspace_preservation_count += 1
            if diagnostic.code == "orphaned_workspace_change":
                orphaned_workspace_change_count += 1
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
            workspace_preservation_count=workspace_preservation_count,
            orphaned_workspace_change_count=orphaned_workspace_change_count,
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
        if followup.diagnostic_code == "workspace_preservation_needed":
            state_class = _workspace_preservation_state_from_action(
                followup.recommended_action
            )
            base_score = {
                "untracked": 960,
                "unstaged": 950,
                "staged": 940,
            }.get(state_class, 930)
            owner_resolution_bonus = {
                "attach_existing_owner": 30,
                "seed_new_owner": 10,
            }.get(followup.owner_resolution_kind or "", 0)
            owner_option_tradeoff_bonus = (
                10 if (followup.owner_option_tradeoff_score or 0) > 0 else 0
            )
            score = (
                base_score
                + owner_resolution_bonus
                + owner_option_tradeoff_bonus
            )
            reason = f"workspace_preservation:{state_class or 'dirty'}"
            if followup.owner_resolution_kind is not None:
                reason = f"{reason}:{followup.owner_resolution_kind}"
            return (
                score,
                reason,
                tuple(
                    component
                    for component in (
                        InvariantScoreComponent(
                            kind="workspace_preservation_base",
                            score=base_score,
                            rationale=f"workspace_preservation:{state_class or 'dirty'}",
                        ),
                        (
                            InvariantScoreComponent(
                                kind="owner_resolution_bonus",
                                score=owner_resolution_bonus,
                                rationale=(
                                    followup.owner_resolution_kind
                                    or "owner_resolution:none"
                                ),
                            )
                            if owner_resolution_bonus > 0
                            else None
                        ),
                        (
                            InvariantScoreComponent(
                                kind="owner_option_tradeoff_bonus",
                                score=owner_option_tradeoff_bonus,
                                rationale=(
                                    followup.owner_option_tradeoff_reason
                                    or "owner_option_tradeoff:none"
                                ),
                            )
                            if owner_option_tradeoff_bonus > 0
                            else None
                        ),
                    )
                    if component is not None
                ),
            )
        if followup.diagnostic_code == "orphaned_workspace_change":
            state_class = _workspace_orphan_state_from_action(
                followup.recommended_action
            )
            base_score = {
                "untracked": 930,
                "unstaged": 920,
                "staged": 910,
            }.get(state_class, 900)
            owner_resolution_bonus = {
                "attach_existing_owner": 20,
                "seed_new_owner": 10,
            }.get(followup.owner_resolution_kind or "", 0)
            owner_option_tradeoff_bonus = (
                10 if (followup.owner_option_tradeoff_score or 0) > 0 else 0
            )
            score = (
                base_score
                + owner_resolution_bonus
                + owner_option_tradeoff_bonus
            )
            reason = f"workspace_orphan:{state_class or 'dirty'}"
            if followup.owner_resolution_kind is not None:
                reason = f"{reason}:{followup.owner_resolution_kind}"
            return (
                score,
                reason,
                tuple(
                    component
                    for component in (
                        InvariantScoreComponent(
                            kind="workspace_orphan_base",
                            score=base_score,
                            rationale=f"workspace_orphan:{state_class or 'dirty'}",
                        ),
                        (
                            InvariantScoreComponent(
                                kind="owner_resolution_bonus",
                                score=owner_resolution_bonus,
                                rationale=(
                                    followup.owner_resolution_kind
                                    or "owner_resolution:none"
                                ),
                            )
                            if owner_resolution_bonus > 0
                            else None
                        ),
                        (
                            InvariantScoreComponent(
                                kind="owner_option_tradeoff_bonus",
                                score=owner_option_tradeoff_bonus,
                                rationale=(
                                    followup.owner_option_tradeoff_reason
                                    or "owner_option_tradeoff:none"
                                ),
                            )
                            if owner_option_tradeoff_bonus > 0
                            else None
                        ),
                    )
                    if component is not None
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
                owner_root_object_id=item.owner_root_object_id,
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

    @staticmethod
    def _resolved_repo_followup_owner_root_object_id(
        followup: InvariantRepoFollowupAction,
    ) -> str | None:
        if followup.owner_root_object_id is not None:
            return followup.owner_root_object_id
        if followup.owner_object_id is not None:
            return followup.owner_object_id
        if followup.owner_seed_object_id is not None:
            return followup.owner_seed_object_id
        return None

    @staticmethod
    def _resolved_repo_diagnostic_lane_owner_root_object_id(
        *,
        lane: InvariantRepoDiagnosticLane,
        best_option: InvariantOwnerCandidateOption | None,
    ) -> str | None:
        if best_option is not None:
            return best_option.object_id
        if lane.candidate_owner_object_id is not None:
            return lane.candidate_owner_object_id
        if lane.candidate_owner_seed_object_id is not None:
            return lane.candidate_owner_seed_object_id
        return None

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
                        owner_root_object_id=(
                            self._resolved_repo_diagnostic_lane_owner_root_object_id(
                                lane=lane,
                                best_option=best_option,
                            )
                        ),
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
                        rel_path=lane.rel_path,
                        qualname=lane.qualname,
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
                        owner_object_id=lane.candidate_owner_object_id,
                        owner_root_object_id=(
                            self._resolved_repo_diagnostic_lane_owner_root_object_id(
                                lane=lane,
                                best_option=None,
                            )
                        ),
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
                        rel_path=lane.rel_path,
                        qualname=lane.qualname,
                        selection_rank=0,
                        opportunity_cost_score=0,
                        opportunity_cost_reason="frontier",
                        opportunity_cost_components=(),
                        count=lane.count,
                    )
                )
            elif lane.diagnostic_code == "workspace_preservation_needed":
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
                title = _workspace_preservation_followup_title(
                    rel_path=lane.rel_path,
                    recommended_action=lane.recommended_action,
                )
                actions.append(
                    InvariantRepoFollowupAction(
                        followup_family="workspace_preservation",
                        action_kind="state_preservation",
                        priority_rank=5,
                        object_id=None,
                        owner_object_id=lane.candidate_owner_object_id,
                        owner_root_object_id=(
                            self._resolved_repo_diagnostic_lane_owner_root_object_id(
                                lane=lane,
                                best_option=best_option,
                            )
                        ),
                        diagnostic_code=lane.diagnostic_code,
                        target_doc_id=None,
                        policy_ids=lane.policy_ids,
                        title=title,
                        blocker_class=(
                            f"workspace_{_workspace_preservation_state_from_action(lane.recommended_action) or 'dirty'}"
                        ),
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
                        rel_path=lane.rel_path,
                        qualname=lane.qualname,
                        selection_rank=0,
                        opportunity_cost_score=0,
                        opportunity_cost_reason="frontier",
                        opportunity_cost_components=(),
                        count=lane.count,
                    )
                )
            elif lane.diagnostic_code == "orphaned_workspace_change":
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
                title = _workspace_orphan_followup_title(
                    rel_path=lane.rel_path,
                    recommended_action=lane.recommended_action,
                )
                actions.append(
                    InvariantRepoFollowupAction(
                        followup_family="workspace_orphan_resolution",
                        action_kind="state_preservation",
                        priority_rank=6,
                        object_id=None,
                        owner_object_id=lane.candidate_owner_object_id,
                        owner_root_object_id=(
                            self._resolved_repo_diagnostic_lane_owner_root_object_id(
                                lane=lane,
                                best_option=best_option,
                            )
                        ),
                        diagnostic_code=lane.diagnostic_code,
                        target_doc_id=None,
                        policy_ids=lane.policy_ids,
                        title=title,
                        blocker_class=(
                            "workspace_orphan_"
                            f"{_workspace_orphan_state_from_action(lane.recommended_action) or 'dirty'}"
                        ),
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
                        rel_path=lane.rel_path,
                        qualname=lane.qualname,
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
                        owner_object_id=lane.candidate_owner_object_id,
                        owner_root_object_id=(
                            self._resolved_repo_diagnostic_lane_owner_root_object_id(
                                lane=lane,
                                best_option=None,
                            )
                        ),
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
                        rel_path=lane.rel_path,
                        qualname=lane.qualname,
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
                        owner_root_object_id=workstream.object_id.wire(),
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
                        rel_path="",
                        qualname="",
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
            if self.diagnostics or any(
                workstream.touchsite_count > 0
                or workstream.documentation_followup_lane() is not None
                for workstream in self._workstream_cache
            ):
                never(
                    reasoning={
                        "summary": (
                            "Repo-level planning must resolve a frontier followup "
                            "when diagnostics or active workstream queues exist."
                        ),
                        "control": "invariant_graph.repo.requires_recommended_followup",
                        "blocking_dependencies": tuple(
                            workstream.object_id.wire()
                            for workstream in self._workstream_cache
                            if workstream.touchsite_count > 0
                            or workstream.documentation_followup_lane() is not None
                        ),
                    },
                    diagnostic_count=len(self.diagnostics),
                    active_workstream_count=sum(
                        1
                        for workstream in self._workstream_cache
                        if workstream.touchsite_count > 0
                        or workstream.documentation_followup_lane() is not None
                    ),
                )
                return None  # pragma: no cover - never() raises
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
        owner_root_object_ids = {
            item.owner_root_object_id
            for item in cofrontier_followups
            if item.owner_root_object_id is not None
        }
        if len(owner_root_object_ids) == 1:
            return ("shared_root_workstream", next(iter(owner_root_object_ids)))
        followup_families = {item.followup_family for item in cofrontier_followups}
        if len(followup_families) == 1:
            followup_family = next(iter(followup_families))
            if len(owner_root_object_ids) > 1:
                return (
                    "mixed_root_followup_family",
                    f"{followup_family}:{','.join(_sorted(list(owner_root_object_ids)))}",
                )
            return ("shared_followup_family", followup_family)
        if len(owner_root_object_ids) > 1:
            return (
                "mixed_root_plateau",
                ",".join(_sorted(list(owner_root_object_ids))),
            )
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
            if self.ranked_repo_followups():
                never(
                    reasoning={
                        "summary": (
                            "Repo followup ranking must resolve at least one lane "
                            "whenever repo followups exist."
                        ),
                        "control": "invariant_graph.repo.requires_recommended_followup_lane",
                        "blocking_dependencies": tuple(
                            workstream.object_id.wire()
                            for workstream in self._workstream_cache
                            if workstream.touchsite_count > 0
                            or workstream.documentation_followup_lane() is not None
                        ),
                    },
                    followup_count=len(self.ranked_repo_followups()),
                )
                return None  # pragma: no cover - never() raises
            return None
        return lanes[0]

    def recommended_repo_code_followup_lane(self) -> InvariantRepoFollowupLane | None:
        for lane in self.repo_followup_lanes():
            if lane.followup_class == "code":
                return lane
        if self.recommended_repo_code_followup() is not None:
            never(
                reasoning={
                    "summary": (
                        "Repo-level code followups must project a code lane once "
                        "a recommended code followup exists."
                    ),
                    "control": "invariant_graph.repo.requires_recommended_code_followup_lane",
                    "blocking_dependencies": tuple(
                        workstream.object_id.wire()
                        for workstream in self._workstream_cache
                        if workstream.touchsite_count > 0
                    ),
                },
                followup_family=self.recommended_repo_code_followup().followup_family,
            )
            return None  # pragma: no cover - never() raises
        return None

    def recommended_repo_human_followup_lane(self) -> InvariantRepoFollowupLane | None:
        for lane in self.repo_followup_lanes():
            if lane.followup_class != "code":
                return lane
        if self.recommended_repo_human_followup() is not None:
            never(
                reasoning={
                    "summary": (
                        "Repo-level human followups must project a human lane once "
                        "a recommended human followup exists."
                    ),
                    "control": "invariant_graph.repo.requires_recommended_human_followup_lane",
                    "blocking_dependencies": tuple(
                        workstream.object_id.wire()
                        for workstream in self._workstream_cache
                        if workstream.documentation_followup_lane() is not None
                    ),
                },
                followup_family=self.recommended_repo_human_followup().followup_family,
            )
            return None  # pragma: no cover - never() raises
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

    def recommended_repo_followup_decision_protocol(
        self,
    ) -> InvariantRepoFollowupDecisionProtocol | None:
        return self._recommended_repo_followup_decision_protocol

    @cached_property
    def _recommended_repo_followup_decision_protocol(
        self,
    ) -> InvariantRepoFollowupDecisionProtocol | None:
        explanation = self.recommended_repo_followup_frontier_explanation()
        if explanation is None:
            return None
        same_class_pressure = _margin_pressure(explanation.same_class_margin_score)
        cross_class_pressure = _margin_pressure(explanation.cross_class_margin_score)
        return InvariantRepoFollowupDecisionProtocol(
            frontier_followup_family=explanation.frontier_followup_family,
            frontier_followup_class=explanation.frontier_followup_class,
            frontier_action_kind=explanation.frontier_action_kind,
            frontier_object_id=explanation.frontier_object_id,
            frontier_diagnostic_code=explanation.frontier_diagnostic_code,
            frontier_target_doc_id=explanation.frontier_target_doc_id,
            frontier_policy_ids=explanation.frontier_policy_ids,
            frontier_utility_score=explanation.frontier_utility_score,
            frontier_utility_reason=explanation.frontier_utility_reason,
            decision_mode=_frontier_decision_mode(
                same_class_pressure=same_class_pressure,
                cross_class_pressure=cross_class_pressure,
            ),
            decision_reason=(
                f"same_class_pressure:{same_class_pressure}:"
                f"{explanation.same_class_margin_score}"
                f"|cross_class_pressure:{cross_class_pressure}:"
                f"{explanation.cross_class_margin_score}"
            ),
            same_class_pressure=same_class_pressure,
            cross_class_pressure=cross_class_pressure,
            decision_components=(
                InvariantScoreComponent(
                    kind="same_class_pressure",
                    score=0
                    if explanation.same_class_margin_score is None
                    else explanation.same_class_margin_score,
                    rationale=same_class_pressure,
                ),
                InvariantScoreComponent(
                    kind="cross_class_pressure",
                    score=0
                    if explanation.cross_class_margin_score is None
                    else explanation.cross_class_margin_score,
                    rationale=cross_class_pressure,
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
                    root_object_ids=tuple(
                        _sorted(
                            list(
                                {
                                self._resolved_repo_followup_owner_root_object_id(
                                    item
                                )
                                for item in items
                                if self._resolved_repo_followup_owner_root_object_id(
                                    item
                                )
                                is not None
                                }
                            )
                        )
                    ),
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
            elif diagnostic.code == "workspace_preservation_needed":
                state_class = "" if node is None else node.status_hint
                title = "" if node is None else node.title
                if not title:
                    title = f"{state_class}:{rel_path}" if state_class and rel_path else diagnostic.code
                recommended_action = _workspace_preservation_action_for_state(state_class)
            elif diagnostic.code == "orphaned_workspace_change":
                state_class = "" if node is None else node.status_hint
                title = "" if node is None else node.title
                if not title:
                    title = f"{state_class}:{rel_path}" if state_class and rel_path else diagnostic.code
                recommended_action = _workspace_orphan_action_for_state(state_class)
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

    @staticmethod
    def _workspace_commit_unit_owner_scope(
        followup: InvariantRepoFollowupAction,
    ) -> tuple[str, tuple[str, ...]]:
        attach_existing_owner_ids = tuple(
            _sorted(
                list(
                    {
                        item.object_id
                        for item in followup.owner_resolution_options
                        if item.object_id
                        and item.resolution_kind == "attach_existing_owner"
                    }
                )
            )
        )
        if len(attach_existing_owner_ids) > 1:
            return ("ambiguous_owner_set", attach_existing_owner_ids)
        if len(attach_existing_owner_ids) == 1:
            return ("resolved_owner", attach_existing_owner_ids)
        owner_option_object_ids = tuple(
            _sorted(
                list(
                    {
                        item.object_id
                        for item in followup.owner_resolution_options
                        if item.object_id
                    }
                )
            )
        )
        if len(owner_option_object_ids) > 1:
            return ("ambiguous_owner_set", owner_option_object_ids)
        resolved_owner_root_object_id = (
            InvariantWorkstreamsProjection._resolved_repo_followup_owner_root_object_id(
                followup
            )
        )
        if resolved_owner_root_object_id is not None:
            return ("resolved_owner", (resolved_owner_root_object_id,))
        if followup.owner_seed_object_id is not None:
            return ("seed_owner", (followup.owner_seed_object_id,))
        return ("unassigned", ())

    def workspace_commit_units(self) -> tuple[InvariantWorkspaceCommitUnit, ...]:
        return self._workspace_commit_units

    @cached_property
    def _workspace_commit_units(self) -> tuple[InvariantWorkspaceCommitUnit, ...]:
        workspace_followups = [
            item
            for item in self.ranked_repo_followups()
            if item.diagnostic_code
            in {"workspace_preservation_needed", "orphaned_workspace_change"}
        ]
        grouped: defaultdict[
            tuple[str, str, str, str, str, tuple[str, ...], str, str],
            list[InvariantRepoFollowupAction],
        ] = defaultdict(list)
        for followup in workspace_followups:
            if followup.diagnostic_code == "workspace_preservation_needed":
                state_class = _workspace_preservation_state_from_action(
                    followup.recommended_action
                )
            else:
                state_class = _workspace_orphan_state_from_action(
                    followup.recommended_action
                )
            owner_scope_kind, root_object_ids = self._workspace_commit_unit_owner_scope(
                followup
            )
            grouped[
                (
                    followup.followup_family,
                    followup.diagnostic_code or "",
                    followup.recommended_action or "",
                    state_class,
                    owner_scope_kind,
                    root_object_ids,
                    followup.owner_seed_path or "",
                    followup.owner_resolution_kind or "",
                )
            ].append(followup)

        units: list[InvariantWorkspaceCommitUnit] = []
        for (
            followup_family,
            diagnostic_code,
            recommended_action,
            state_class,
            owner_scope_kind,
            root_object_ids,
            owner_seed_path,
            owner_resolution_kind,
        ), items in grouped.items():
            best_followup = items[0]
            action_count = sum(item.count for item in items)
            rel_paths = tuple(
                _sorted(list({item.rel_path for item in items if item.rel_path}))
            )
            qualnames = tuple(
                _sorted(list({item.qualname for item in items if item.qualname}))
            )
            breadth_bonus = 0 if len(items) <= 1 else min(len(items), 9) * 5
            utility_score = best_followup.utility_score + breadth_bonus
            utility_reason = (
                best_followup.utility_reason
                if breadth_bonus <= 0
                else f"{best_followup.utility_reason}+workspace_breadth:{len(items)}"
            )
            units.append(
                InvariantWorkspaceCommitUnit(
                    followup_family=followup_family,
                    diagnostic_code=diagnostic_code,
                    recommended_action=(
                        None if not recommended_action else recommended_action
                    ),
                    state_class=state_class,
                    owner_scope_kind=owner_scope_kind,
                    root_object_ids=root_object_ids,
                    owner_seed_path=(
                        None if not owner_seed_path else owner_seed_path
                    ),
                    owner_resolution_kind=(
                        None if not owner_resolution_kind else owner_resolution_kind
                    ),
                    action_count=action_count,
                    rel_paths=rel_paths,
                    qualnames=qualnames,
                    utility_score=utility_score,
                    utility_reason=utility_reason,
                    selection_rank=0,
                    opportunity_cost_score=0,
                    opportunity_cost_reason="frontier",
                    best_followup=best_followup,
                )
            )

        ranked_units = _sorted(
            units,
            key=lambda item: (
                -item.utility_score,
                item.best_followup.priority_rank,
                -item.action_count,
                item.followup_family,
                item.owner_scope_kind,
                item.root_object_ids,
                item.rel_paths,
            ),
        )
        frontier_unit = ranked_units[0] if ranked_units else None
        frontier_score = frontier_unit.utility_score if frontier_unit is not None else 0
        return tuple(
            replace(
                item,
                selection_rank=index,
                opportunity_cost_score=max(0, frontier_score - item.utility_score),
                opportunity_cost_reason=(
                    "frontier"
                    if index == 1
                    else (
                        "cofrontier"
                        if frontier_unit is not None
                        and frontier_score == item.utility_score
                        else (
                            f"deferred_by:{frontier_unit.followup_family}"
                            if frontier_unit is not None
                            else "none"
                        )
                    )
                ),
            )
            for index, item in enumerate(ranked_units, start=1)
        )

    def recommended_workspace_commit_unit(self) -> InvariantWorkspaceCommitUnit | None:
        units = self.workspace_commit_units()
        if not units:
            return None
        return units[0]

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
        recommended_repo_followup_decision_protocol = (
            self.recommended_repo_followup_decision_protocol()
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
        workspace_commit_units = self.workspace_commit_units()
        recommended_workspace_commit_unit = self.recommended_workspace_commit_unit()
        return {
            "format_version": _FORMAT_VERSION,
            "generated_at_utc": self.generated_at_utc,
            "root": self.root,
            "workstreams": [item.as_payload() for item in workstreams],
            "diagnostic_summary": diagnostic_summary.as_payload(),
            "planning_chart_summary": (
                None
                if self.planning_chart_summary is None
                else self.planning_chart_summary.as_payload()
            ),
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
                "recommended_followup_decision_protocol": (
                    None
                    if recommended_repo_followup_decision_protocol is None
                    else recommended_repo_followup_decision_protocol.as_payload()
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
                "recommended_workspace_commit_unit": (
                    None
                    if recommended_workspace_commit_unit is None
                    else recommended_workspace_commit_unit.as_payload()
                ),
                "workspace_commit_units": [
                    item.as_payload() for item in workspace_commit_units
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
                            scalar(identity=workstream.object_id, key="failing_test_case_count", title="failing_test_case_count", value=workstream.failing_test_case_count),
                            scalar(identity=workstream.object_id, key="test_failure_count", title="test_failure_count", value=workstream.test_failure_count),
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
                                                qualname="recommended_counterfactual_blocked_cut",
                                            ),
                                            key="recommended_counterfactual_blocked_cut",
                                            title="recommended_counterfactual_blocked_cut",
                                            value=(
                                                None
                                                if workstream.recommended_counterfactual_blocked_cut()
                                                is None
                                                else workstream.recommended_counterfactual_blocked_cut().as_payload()
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
                                        scalar(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="recommended_cut_frontier_explanation",
                                            ),
                                            key="recommended_cut_frontier_explanation",
                                            title="recommended_cut_frontier_explanation",
                                            value=(
                                                None
                                                if workstream.recommended_cut_frontier_explanation() is None
                                                else workstream.recommended_cut_frontier_explanation().as_payload()
                                            ),
                                        ),
                                        scalar(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="recommended_cut_decision_protocol",
                                            ),
                                            key="recommended_cut_decision_protocol",
                                            title="recommended_cut_decision_protocol",
                                            value=(
                                                None
                                                if workstream.recommended_cut_decision_protocol() is None
                                                else workstream.recommended_cut_decision_protocol().as_payload()
                                            ),
                                        ),
                                        scalar(
                                            identity=ArtifactSourceRef(
                                                rel_path="<synthetic>",
                                                qualname="recommended_cut_frontier_stability",
                                            ),
                                            key="recommended_cut_frontier_stability",
                                            title="recommended_cut_frontier_stability",
                                            value=(
                                                None
                                                if workstream.recommended_cut_frontier_stability() is None
                                                else workstream.recommended_cut_frontier_stability().as_payload()
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
    phase_kind: str = "",
    item_kind: str = "",
    source_kind: str = "",
    selection_rank: int = 0,
    tracked_node_ids: tuple[str, ...] = (),
    tracked_object_ids: tuple[str, ...] = (),
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
        phase_kind=phase_kind,
        item_kind=item_kind,
        source_kind=source_kind,
        selection_rank=selection_rank,
        tracked_node_ids=tracked_node_ids,
        tracked_object_ids=tracked_object_ids,
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
    touchpoint_definition: RegisteredTouchpointDefinition,
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
        touchpoint_definition: RegisteredTouchpointDefinition,
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
            touchsite_object_id = _scanned_touchsite_object_id(
                touchpoint_object_id=self.touchpoint_object_id,
                structural_identity=structural_identity,
            )
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
    touchpoint_definition: RegisteredTouchpointDefinition,
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


def _scanned_touchsite_object_id(
    *,
    touchpoint_object_id: str,
    structural_identity: str,
) -> str:
    prefix, _, _suffix = touchpoint_object_id.partition("-TP-")
    if prefix:
        return f"{prefix}-TS:{structural_identity}"
    return f"TS:{structural_identity}"


def _primary_workstream_object_id(values: tuple[str, ...], fallback: str) -> str:
    for value in values:
        if value == fallback:
            return value
    return fallback


@dataclass
class _InvariantGraphBuildState:
    root: Path
    nodes_by_id: dict[str, InvariantGraphNode]
    diagnostics: list[InvariantGraphDiagnostic]
    ranking_signals: list[InvariantGraphRankingSignal]
    edge_keys: set[tuple[str, str, str]]
    edges: list[InvariantGraphEdge]
    object_node_ids: dict[str, str]
    object_owner_node_ids: dict[str, str]
    workstream_root_ids: list[str]
    declared_workstream_ids: set[str]
    structured_artifact_identities: StructuredArtifactIdentitySpace


def _new_build_state(root: Path) -> _InvariantGraphBuildState:
    return _InvariantGraphBuildState(
        root=root,
        nodes_by_id={},
        diagnostics=[],
        ranking_signals=[],
        edge_keys=set(),
        edges=[],
        object_node_ids={},
        object_owner_node_ids={},
        workstream_root_ids=[],
        declared_workstream_ids=set(),
        structured_artifact_identities=StructuredArtifactIdentitySpace(),
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
    touchpoint_definition: RegisteredTouchpointDefinition,
) -> tuple[_Phase5Touchsite, ...]:
    return _scan_phase5_touchsites(
        touchpoint_definition=touchpoint_definition,
        rel_path=touchpoint_definition.rel_path,
        touchpoint_object_id=touchpoint_definition.touchpoint_id,
        subqueue_object_id=touchpoint_definition.subqueue_id,
    )


def _semantic_link_values(
    payload,
    *,
    kind: SemanticLinkKind,
) -> tuple[str, ...]:
    return tuple(link.value for link in payload.links if link.kind is kind)


def _registered_touchsite_node(
    *,
    touchsite_id: str,
    title: str,
    rel_path: str,
    qualname: str,
    line: int,
    column: int,
    node_kind: str,
    site_identity: str,
    structural_identity: str,
    seam_class: str,
    source_marker_node_id: str,
    reasoning_summary: str,
    reasoning_control: str,
    blocking_dependencies: tuple[str, ...],
    object_ids: tuple[str, ...] = (),
    status_hint: str = "",
) -> InvariantGraphNode:
    return InvariantGraphNode(
        node_id=_synthetic_ref_node_id("object_id", touchsite_id),
        node_kind="synthetic_touchsite",
        title=title,
        marker_name="grade_boundary",
        marker_kind="",
        marker_id="",
        site_identity=site_identity,
        structural_identity=structural_identity,
        object_ids=(touchsite_id, *object_ids),
        doc_ids=(),
        policy_ids=(),
        invariant_ids=(),
        reasoning_summary=reasoning_summary,
        reasoning_control=reasoning_control,
        blocking_dependencies=blocking_dependencies,
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        column=column,
        ast_node_kind=node_kind,
        seam_class=seam_class,
        source_marker_node_id=source_marker_node_id,
        status_hint=status_hint,
    )


def _add_registered_touchsite(
    state: _InvariantGraphBuildState,
    *,
    touchpoint_definition: RegisteredTouchpointDefinition,
    touchpoint_node_id: str,
    touchsite_definition: RegisteredTouchsiteDefinition,
) -> None:
    node = _registered_touchsite_node(
        touchsite_id=touchsite_definition.touchsite_id,
        title=touchsite_definition.boundary_name,
        rel_path=touchsite_definition.rel_path,
        qualname=touchsite_definition.qualname,
        line=touchsite_definition.line,
        column=touchsite_definition.column,
        node_kind=touchsite_definition.node_kind,
        site_identity=touchsite_definition.site_identity,
        structural_identity=touchsite_definition.structural_identity,
        seam_class=touchsite_definition.seam_class,
        source_marker_node_id=touchpoint_node_id,
        reasoning_summary=f"{touchpoint_definition.touchpoint_id} touchsite remains active.",
        reasoning_control=touchpoint_definition.marker_payload.reasoning.control,
        blocking_dependencies=(touchpoint_definition.touchpoint_id,),
        object_ids=touchsite_definition.object_ids,
        status_hint=touchsite_definition.status_hint,
    )
    _add_node(state, node, replace=True)
    _claim_object_id(state, node, object_id=touchsite_definition.touchsite_id)
    state.declared_workstream_ids.add(touchsite_definition.touchsite_id)
    _add_edge(state, "contains", touchpoint_node_id, node.node_id)
    _add_edge(state, "blocks", node.node_id, touchpoint_node_id)


def _touchpoint_touchsite_node_ids(
    state: _InvariantGraphBuildState,
    *,
    touchpoint_node_id: str,
) -> tuple[str, ...]:
    return tuple(
        _sorted(
            [
                edge.target_id
                for edge in state.edges
                if edge.edge_kind == "contains"
                and edge.source_id == touchpoint_node_id
                and state.nodes_by_id.get(edge.target_id, None) is not None
                and state.nodes_by_id[edge.target_id].node_kind == "synthetic_touchsite"
            ]
        )
    )


def _resolve_counterfactual_target_touchsite_node_id(
    state: _InvariantGraphBuildState,
    *,
    touchpoint_node_id: str,
    target_boundary_name: str,
) -> str:
    if not target_boundary_name:
        return ""
    for node_id in _touchpoint_touchsite_node_ids(
        state,
        touchpoint_node_id=touchpoint_node_id,
    ):
        node = state.nodes_by_id[node_id]
        if node.title == target_boundary_name:
            return node_id
    return ""


def _counterfactual_action_signal_score(
    action_definition: RegisteredCounterfactualActionDefinition,
) -> int:
    if action_definition.score > 0:
        return action_definition.score
    match action_definition.predicted_readiness_class:
        case "diagnostic_blocked":
            return 10
        case "policy_blocked":
            return 8
        case "counterfactual_blocked":
            return 6
        case "coverage_gap":
            return 3
        case "ready_structural":
            return 1
        case _:
            return 0


def _add_registered_counterfactual_action(
    state: _InvariantGraphBuildState,
    *,
    touchpoint_definition: RegisteredTouchpointDefinition,
    touchpoint_node_id: str,
    action_definition: RegisteredCounterfactualActionDefinition,
) -> None:
    action_node_id = f"counterfactual_action:{stable_hash(action_definition.action_id)}"
    action_node = _synthetic_node(
        node_id=action_node_id,
        title=action_definition.title,
        ref_kind="counterfactual_action",
        value=action_definition.action_id,
        object_ids=tuple(
            item
            for item in (
                action_definition.action_id,
                touchpoint_definition.touchpoint_id,
                action_definition.target_boundary_name,
                *action_definition.object_ids,
            )
            if item
        ),
        reasoning_summary=action_definition.rationale or action_definition.title,
        reasoning_control=(
            "invariant_graph.counterfactual_action."
            f"{action_definition.action_kind or 'declared'}"
        ),
        rel_path=touchpoint_definition.rel_path,
        qualname=action_definition.action_id,
        node_kind="counterfactual_action",
        status_hint=action_definition.predicted_readiness_class,
    )
    _add_node(state, action_node, replace=True)
    _link_node_refs(state, action_node)
    _add_edge(state, "contains", touchpoint_node_id, action_node_id)
    target_touchsite_node_id = _resolve_counterfactual_target_touchsite_node_id(
        state,
        touchpoint_node_id=touchpoint_node_id,
        target_boundary_name=action_definition.target_boundary_name,
    )
    if target_touchsite_node_id:
        _add_edge(state, "predicts", action_node_id, target_touchsite_node_id)
        _add_edge(state, "tracks", action_node_id, target_touchsite_node_id)
    if action_definition.predicted_readiness_class in _COUNTERFACTUAL_BLOCKED_READINESS_CLASSES:
        target_touchsite_object_id = (
            _primary_object_id(state.nodes_by_id[target_touchsite_node_id])
            if target_touchsite_node_id
            else touchpoint_definition.touchpoint_id
        )
        _append_ranking_signal(
            state,
            node_id=action_node_id,
            touchpoint_object_id=touchpoint_definition.touchpoint_id,
            touchsite_object_id=target_touchsite_object_id,
            code=f"counterfactual_{action_definition.predicted_readiness_class}",
            score=_counterfactual_action_signal_score(action_definition),
            raw_dependency=action_definition.action_id,
            message=action_definition.rationale or action_definition.title,
        )


def _add_registered_work_item(
    state: _InvariantGraphBuildState,
    *,
    object_id: str,
    title: str,
    rel_path: str,
    qualname: str,
    line: int,
    marker_identity: str,
    marker_payload,
    site_identity: str,
    structural_identity: str,
    marker_node_id_by_marker_id: Mapping[str, str],
    status_hint: str = "",
) -> None:
    _add_work_item(
        state,
        object_id=object_id,
        title=title,
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        marker_id=marker_identity,
        marker_name="gabion.invariants.todo_decorator",
        marker_kind=marker_payload.marker_kind.value,
        site_identity=site_identity,
        structural_identity=structural_identity,
        object_ids=_semantic_link_values(
            marker_payload,
            kind=SemanticLinkKind.OBJECT_ID,
        ),
        doc_ids=_semantic_link_values(
            marker_payload,
            kind=SemanticLinkKind.DOC_ID,
        ),
        policy_ids=_semantic_link_values(
            marker_payload,
            kind=SemanticLinkKind.POLICY_ID,
        ),
        invariant_ids=_semantic_link_values(
            marker_payload,
            kind=SemanticLinkKind.INVARIANT_ID,
        ),
        reasoning_summary=marker_payload.reasoning.summary,
        reasoning_control=marker_payload.reasoning.control,
        blocking_dependencies=marker_payload.reasoning.blocking_dependencies,
        source_marker_node_id=marker_node_id_by_marker_id.get(marker_identity, ""),
        status_hint=status_hint,
    )


def _enrich_workstream_registry(
    state: _InvariantGraphBuildState,
    *,
    registry: WorkstreamRegistry,
    marker_node_id_by_marker_id: Mapping[str, str],
) -> None:
    root_definition = registry.root
    primary_root_object_id = _primary_workstream_object_id(
        _semantic_link_values(
            root_definition.marker_payload,
            kind=SemanticLinkKind.OBJECT_ID,
        ),
        root_definition.root_id,
    )
    _add_registered_work_item(
        state,
        object_id=primary_root_object_id,
        title=root_definition.title,
        rel_path=root_definition.rel_path,
        qualname=root_definition.qualname,
        line=root_definition.line,
        marker_identity=root_definition.marker_identity,
        marker_payload=root_definition.marker_payload,
        site_identity=root_definition.site_identity,
        structural_identity=root_definition.structural_identity,
        marker_node_id_by_marker_id=marker_node_id_by_marker_id,
        status_hint=root_definition.status_hint,
    )
    _register_root_workstream(state, object_id=primary_root_object_id)

    for subqueue_definition in registry.subqueues:
        primary_subqueue_object_id = _primary_workstream_object_id(
            _semantic_link_values(
                subqueue_definition.marker_payload,
                kind=SemanticLinkKind.OBJECT_ID,
            ),
            subqueue_definition.subqueue_id,
        )
        _add_registered_work_item(
            state,
            object_id=primary_subqueue_object_id,
            title=subqueue_definition.title,
            rel_path=subqueue_definition.rel_path,
            qualname=subqueue_definition.qualname,
            line=subqueue_definition.line,
            marker_identity=subqueue_definition.marker_identity,
            marker_payload=subqueue_definition.marker_payload,
            site_identity=subqueue_definition.site_identity,
            structural_identity=subqueue_definition.structural_identity,
            marker_node_id_by_marker_id=marker_node_id_by_marker_id,
            status_hint=subqueue_definition.status_hint,
        )

    for touchpoint_definition in registry.touchpoints:
        primary_touchpoint_object_id = _primary_workstream_object_id(
            _semantic_link_values(
                touchpoint_definition.marker_payload,
                kind=SemanticLinkKind.OBJECT_ID,
            ),
            touchpoint_definition.touchpoint_id,
        )
        _add_registered_work_item(
            state,
            object_id=primary_touchpoint_object_id,
            title=touchpoint_definition.title,
            rel_path=touchpoint_definition.rel_path,
            qualname=touchpoint_definition.qualname,
            line=touchpoint_definition.line,
            marker_identity=touchpoint_definition.marker_identity,
            marker_payload=touchpoint_definition.marker_payload,
            site_identity=touchpoint_definition.site_identity,
            structural_identity=touchpoint_definition.structural_identity,
            marker_node_id_by_marker_id=marker_node_id_by_marker_id,
            status_hint=touchpoint_definition.status_hint,
        )

    root_node_id = state.object_node_ids[primary_root_object_id]
    for subqueue_id in root_definition.subqueue_ids:
        subqueue_node_id = state.object_node_ids[subqueue_id]
        _add_edge(state, "contains", root_node_id, subqueue_node_id)
        _add_edge(state, "blocks", subqueue_node_id, root_node_id)

    for subqueue_definition in registry.subqueues:
        subqueue_node_id = state.object_node_ids[subqueue_definition.subqueue_id]
        for touchpoint_id in subqueue_definition.touchpoint_ids:
            touchpoint_node_id = state.object_node_ids[touchpoint_id]
            _add_edge(state, "contains", subqueue_node_id, touchpoint_node_id)
            _add_edge(state, "blocks", touchpoint_node_id, subqueue_node_id)

    for touchpoint_definition in registry.touchpoints:
        touchpoint_node_id = state.object_node_ids[touchpoint_definition.touchpoint_id]
        for touchsite_definition in touchpoint_definition.declared_touchsites:
            _add_registered_touchsite(
                state,
                touchpoint_definition=touchpoint_definition,
                touchpoint_node_id=touchpoint_node_id,
                touchsite_definition=touchsite_definition,
            )
        for action_definition in touchpoint_definition.declared_counterfactual_actions:
            _add_registered_counterfactual_action(
                state,
                touchpoint_definition=touchpoint_definition,
                touchpoint_node_id=touchpoint_node_id,
                action_definition=action_definition,
            )
        if not touchpoint_definition.scan_touchsites:
            continue
        for touchsite in _scan_phase5_touchsites_for_definition(
            touchpoint_definition=touchpoint_definition
        ):
            _add_registered_touchsite(
                state,
                touchpoint_definition=touchpoint_definition,
                touchpoint_node_id=touchpoint_node_id,
                touchsite_definition=RegisteredTouchsiteDefinition(
                    touchsite_id=touchsite.touchsite_object_id,
                    rel_path=touchsite.rel_path,
                    qualname=touchsite.qualname,
                    boundary_name=touchsite.boundary_name,
                    line=touchsite.line,
                    column=touchsite.column,
                    node_kind=touchsite.node_kind,
                    site_identity=touchsite.site_identity,
                    structural_identity=touchsite.structural_identity,
                    seam_class=touchsite.seam_class,
                ),
            )


def _iter_declared_workstream_registries() -> tuple[WorkstreamRegistry, ...]:
    registries = []
    phase5_registry = phase5_workstream_registry()
    if phase5_registry is not None:
        registries.append(phase5_registry)
    prf_registry = prf_workstream_registry()
    if prf_registry is not None:
        registries.append(prf_registry)
    registries.extend(connectivity_synergy_workstream_registries())
    return tuple(registries)


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


def _policy_signal_orphan_group_key(*, rel_path: str, qualname: str) -> str:
    if rel_path:
        parent = Path(rel_path).parent.as_posix()
        if parent and parent != ".":
            return f"seed:{parent}"
        return f"path:{rel_path}"
    if qualname:
        return f"qualname:{qualname}"
    return "global"


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
            rel_path = _normalize_rel_path(state.root, raw_violation.get("path"))
            qualname = str(raw_violation.get("qualname", "")).strip()
            target_node_id = _match_policy_signal_target(
                root=state.root,
                index=index,
                violation=raw_violation,
            )
            aggregate_target = target_node_id or _policy_signal_orphan_group_key(
                rel_path=rel_path,
                qualname=qualname,
            )
            aggregate_key = (domain, rule_id, aggregate_target)
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


def _site_span_contains_line(site: TestEvidenceSite, line: int) -> bool:
    raw_span = site.span
    if len(raw_span) != 4:
        return True
    start = raw_span[0]
    end = raw_span[2]
    lower = min(start, end)
    upper = max(start, end)
    return lower <= line <= upper


def _match_test_evidence_node_ids(
    *,
    root: Path,
    index: _InvariantGraphNodeIndex,
    state: _InvariantGraphBuildState,
    site: TestEvidenceSite,
) -> tuple[str, ...]:
    rel_path = _normalize_rel_path(root, site.path)
    site_qual = site.qualname
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
    artifact = load_test_evidence_artifact(
        root=state.root,
        rel_path=_TEST_EVIDENCE_ARTIFACT.as_posix(),
        identities=state.structured_artifact_identities,
    )
    if artifact is None:
        return
    index = _node_index(state)
    for test_case in artifact.cases:
        matched_node_ids: set[str] = set()
        for site in test_case.evidence_sites:
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
        node_id = f"test_case:{stable_hash(test_case.identity.wire())}"
        node = _synthetic_node(
            node_id=node_id,
            title=test_case.test_id,
            ref_kind="test_case",
            value=test_case.identity.wire(),
            object_ids=(test_case.identity.wire(),),
            rel_path=test_case.rel_path,
            qualname=test_case.test_id,
            line=test_case.line,
            reasoning_summary=test_case.status,
            node_kind="test_case",
        )
        _add_node(state, node, replace=True)
        for matched_node_id in _sorted(list(matched_node_ids)):
            _add_edge(state, "covered_by", matched_node_id, node_id)


def _match_existing_test_case_node_id(
    state: _InvariantGraphBuildState,
    *,
    rel_path: str,
    test_name: str,
    classname: str,
) -> str:
    matches: list[str] = []
    class_suffix = classname.split(".")[-1].strip() if classname else ""
    for node in state.nodes_by_id.values():
        if node.node_kind != "test_case":
            continue
        if rel_path and node.rel_path != rel_path:
            continue
        if node.title.endswith(f"::{test_name}"):
            matches.append(node.node_id)
            continue
        if class_suffix and node.title.endswith(f"::{class_suffix}::{test_name}"):
            matches.append(node.node_id)
    if not matches:
        return ""
    return _sorted(matches)[0]


def _ensure_test_case_node(
    state: _InvariantGraphBuildState,
    *,
    test_id: str,
    rel_path: str,
    line: int,
    status: str,
) -> str:
    node_id = f"test_case:{stable_hash(test_id)}"
    node = _synthetic_node(
        node_id=node_id,
        title=test_id,
        ref_kind="test_case",
        value=test_id,
        rel_path=rel_path,
        qualname=test_id,
        line=line,
        reasoning_summary=status,
        node_kind="test_case",
    )
    _add_node(state, node, replace=True)
    return node_id


def _traceback_matched_node_ids(
    *,
    root: Path,
    index: _InvariantGraphNodeIndex,
    state: _InvariantGraphBuildState,
    traceback_text: str,
) -> tuple[str, ...]:
    matched: set[str] = set()
    for match in _TRACEBACK_FRAME_RE.finditer(traceback_text):
        rel_path = _normalize_rel_path(root, match.group("path"))
        line = int(match.group("line") or 0)
        qualname = str(match.group("qual") or "").strip()
        for path_variant in _path_variants(rel_path):
            if qualname:
                node_id = index.by_path_line_qualname.get((path_variant, line, qualname))
                if node_id is not None:
                    matched.add(node_id)
            for candidate_id in index.by_path_exact.get(path_variant, ()):
                candidate = state.nodes_by_id[candidate_id]
                if candidate.line > 0 and candidate.line == line:
                    matched.add(candidate_id)
    return tuple(_sorted(list(matched)))


def _join_test_failures(state: _InvariantGraphBuildState) -> None:
    artifact = load_junit_failure_artifact(
        root=state.root,
        rel_path=_JUNIT_TEST_RESULTS_ARTIFACT.as_posix(),
        identities=state.structured_artifact_identities,
    )
    if artifact is None:
        return
    index = _node_index(state)
    for failure in artifact.failures:
        test_case_node_id = _match_existing_test_case_node_id(
            state,
            rel_path=failure.rel_path,
            test_name=failure.raw_name,
            classname=failure.classname,
        )
        test_id = failure.test_id
        if test_case_node_id:
            test_id = state.nodes_by_id[test_case_node_id].title
        else:
            test_case_node_id = _ensure_test_case_node(
                state,
                test_id=test_id,
                rel_path=failure.rel_path,
                line=failure.line,
                status="failed",
            )
        failure_node_id = f"test_failure:{stable_hash(failure.identity.wire())}"
        failure_node = _synthetic_node(
            node_id=failure_node_id,
            title=failure.title,
            ref_kind="test_failure",
            value=failure.identity.wire(),
            object_ids=tuple(
                item
                for item in (failure.identity.wire(), failure.title)
                if item
            ),
            rel_path=failure.rel_path,
            qualname=test_id,
            line=failure.line,
            reasoning_summary=failure.message or failure.raw_name,
            reasoning_control="invariant_graph.test_failure",
            node_kind="test_failure",
            status_hint=failure.failure_kind,
        )
        _add_node(state, failure_node, replace=True)
        _add_edge(state, "fails_with", test_case_node_id, failure_node_id)
        for matched_node_id in _traceback_matched_node_ids(
            root=state.root,
            index=index,
            state=state,
            traceback_text=failure.traceback_text,
        ):
            _add_edge(state, "fails_on", failure_node_id, matched_node_id)


def _markdown_frontmatter(
    *,
    root: Path,
    rel_path: str,
) -> tuple[dict[str, object], str]:
    path = root / rel_path
    if not path.exists():
        return {}, ""
    frontmatter, body = parse_strict_yaml_frontmatter(path.read_text(encoding="utf-8"))
    return cast(dict[str, object], dict(frontmatter)), body


def _governance_doc_aliases(
    *,
    rel_path: str,
    frontmatter: Mapping[str, object],
) -> tuple[str, ...]:
    aliases: list[str] = []
    doc_id = str(frontmatter.get("doc_id", "")).strip()
    if doc_id:
        aliases.append(doc_id)
    path = Path(rel_path)
    if path.parent.as_posix() == "in" and path.name.startswith("in-") and path.suffix == ".md":
        stem = path.stem
        aliases.append(stem)
        if stem.startswith("in-"):
            suffix = stem.split("-", 1)[1]
            aliases.append(f"in_{suffix}")
    return tuple(_sorted(list({value for value in aliases if value})))


def _governance_doc_title(
    *,
    rel_path: str,
    frontmatter: Mapping[str, object],
) -> str:
    for key in ("title", "doc_id"):
        value = str(frontmatter.get(key, "")).strip()
        if value:
            return value
    return Path(rel_path).name


def _ensure_governance_doc_node(
    state: _InvariantGraphBuildState,
    *,
    rel_path: str,
    default_status_hint: str = "",
    reasoning_summary: str = "",
) -> str:
    node_id = f"governance_doc:{rel_path}"
    existing = state.nodes_by_id.get(node_id)
    if existing is not None:
        return existing.node_id
    frontmatter, _body = _markdown_frontmatter(root=state.root, rel_path=rel_path)
    doc_ids = _governance_doc_aliases(rel_path=rel_path, frontmatter=frontmatter)
    title = _governance_doc_title(rel_path=rel_path, frontmatter=frontmatter)
    authority = str(frontmatter.get("doc_authority", "")).strip()
    role = str(frontmatter.get("doc_role", "")).strip()
    summary = reasoning_summary or (
        f"governance_doc authority={authority or '<unset>'} role={role or '<unset>'}"
    )
    node = _synthetic_node(
        node_id=node_id,
        title=title,
        ref_kind="governance_doc",
        value=rel_path,
        doc_ids=doc_ids,
        reasoning_summary=summary,
        reasoning_control="invariant_graph.governance_doc",
        rel_path=rel_path,
        node_kind="governance_doc",
        status_hint=default_status_hint or authority or role,
    )
    _add_node(state, node, replace=True)
    _link_node_refs(state, node)
    return node.node_id


def _influence_index_status_rows(root: Path) -> dict[str, tuple[str, str]]:
    path = root / _INFLUENCE_INDEX_DOC
    if not path.exists():
        return {}
    rows: dict[str, tuple[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("- in/"):
            continue
        rel_path, separator, tail = line[2:].partition(" — ")
        if not separator:
            continue
        start = tail.find("**")
        end = tail.find("**", start + 2) if start >= 0 else -1
        if start < 0 or end <= start + 2:
            continue
        status = tail[start + 2 : end].strip().lower()
        if not status:
            continue
        rows[rel_path.strip()] = (status, tail.strip())
    return rows


def _section_category(title: str) -> str:
    normalized = title.casefold()
    if "goal" in normalized or "objective" in normalized:
        return "goal"
    if "next step" in normalized:
        return "next_steps"
    if "acceptance" in normalized:
        return "acceptance"
    if "proposed approach" in normalized or "implementation" in normalized:
        return "approach"
    if "open question" in normalized:
        return "open_questions"
    if "dependenc" in normalized:
        return "dependencies"
    if "problem" in normalized:
        return "problem"
    return "section"


def _join_sppf_dependency_graph(
    state: _InvariantGraphBuildState,
    *,
    checklist_node_id: str,
) -> Mapping[str, tuple[str, ...]]:
    artifact_path = state.root / _SPPF_DEPENDENCY_GRAPH_ARTIFACT
    if not artifact_path.exists():
        return {}
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {}
    docs = payload.get("docs")
    issues = payload.get("issues")
    edges = payload.get("edges")
    if not isinstance(docs, Mapping) or not isinstance(issues, Mapping) or not isinstance(edges, list):
        return {}
    docs_without_issue = {
        str(value)
        for value in payload.get("docs_without_issue", [])
        if isinstance(value, str)
    }
    issues_without_doc_ref = {
        str(value)
        for value in payload.get("issues_without_doc_ref", [])
        if isinstance(value, str)
    }
    doc_node_ids_by_doc_id: defaultdict[str, list[str]] = defaultdict(list)
    doc_entry_node_ids: dict[str, str] = {}
    issue_node_ids: dict[str, str] = {}
    for entry_id, raw_doc in _sorted(list(docs.items()), key=lambda item: item[0]):
        if not isinstance(raw_doc, Mapping):
            continue
        doc_id = str(raw_doc.get("doc_id", "")).strip() or str(entry_id)
        revision = int(raw_doc.get("revision", 0) or 0)
        issue_ids = tuple(
            _sorted(
                [str(value) for value in raw_doc.get("issues", []) if isinstance(value, str)]
            )
        )
        node_id = f"sppf_doc_ref:{entry_id}"
        node = _synthetic_node(
            node_id=node_id,
            title=str(entry_id),
            ref_kind="sppf_doc_ref",
            value=str(entry_id),
            doc_ids=(doc_id,),
            reasoning_summary=(
                f"sppf checklist doc_ref revision={revision} issues={len(issue_ids)}"
            ),
            reasoning_control="invariant_graph.sppf_dependency.doc_ref",
            rel_path=_SPPF_CHECKLIST_DOC.as_posix(),
            node_kind="sppf_doc_ref",
            status_hint="issue_missing" if str(entry_id) in docs_without_issue else "",
        )
        _add_node(state, node, replace=True)
        _link_node_refs(state, node)
        _add_edge(state, "contains", checklist_node_id, node_id)
        doc_entry_node_ids[str(entry_id)] = node_id
        doc_node_ids_by_doc_id[doc_id].append(node_id)
    for issue_id, raw_issue in _sorted(list(issues.items()), key=lambda item: item[0]):
        if not isinstance(raw_issue, Mapping):
            continue
        doc_refs = tuple(
            _sorted(
                [str(value) for value in raw_issue.get("doc_refs", []) if isinstance(value, str)]
            )
        )
        checklist_state = str(raw_issue.get("checklist_state", "")).strip()
        doc_status = str(raw_issue.get("doc_status", "")).strip()
        impl_status = str(raw_issue.get("impl_status", "")).strip()
        node_id = f"sppf_issue:{issue_id}"
        node = _synthetic_node(
            node_id=node_id,
            title=str(issue_id),
            ref_kind="sppf_issue",
            value=str(issue_id),
            object_ids=(str(issue_id),),
            reasoning_summary=(
                "sppf issue checklist={checklist} doc={doc} impl={impl} refs={refs}".format(
                    checklist=checklist_state or "<none>",
                    doc=doc_status or "<none>",
                    impl=impl_status or "<none>",
                    refs=len(doc_refs),
                )
            ),
            reasoning_control="invariant_graph.sppf_dependency.issue",
            rel_path=_SPPF_CHECKLIST_DOC.as_posix(),
            line=int(raw_issue.get("line_no", 0) or 0),
            node_kind="sppf_issue",
            status_hint="doc_ref_missing" if str(issue_id) in issues_without_doc_ref else "",
        )
        _add_node(state, node, replace=True)
        _link_node_refs(state, node)
        _add_edge(state, "contains", checklist_node_id, node_id)
        issue_node_ids[str(issue_id)] = node_id
    for raw_edge in edges:
        if not isinstance(raw_edge, Mapping):
            continue
        source_id = doc_entry_node_ids.get(str(raw_edge.get("from", "")))
        target_id = issue_node_ids.get(str(raw_edge.get("to", "")))
        edge_kind = str(raw_edge.get("kind", "")).strip() or "sppf_link"
        if source_id and target_id:
            _add_edge(state, edge_kind, source_id, target_id)
            _add_edge(state, "tracks", source_id, target_id)
    return {
        doc_id: tuple(_sorted(node_ids))
        for doc_id, node_ids in doc_node_ids_by_doc_id.items()
    }


def _join_inbox_governance_docs(
    state: _InvariantGraphBuildState,
    *,
    influence_index_node_id: str,
    sppf_doc_node_ids: Mapping[str, tuple[str, ...]],
) -> None:
    status_rows = _influence_index_status_rows(state.root)
    for path in _sorted(
        [
            item
            for item in state.root.joinpath("in").glob("*.md")
            if item.is_file()
        ],
        key=lambda item: str(item.relative_to(state.root)).replace("\\", "/"),
    ):
        rel_path = str(path.relative_to(state.root)).replace("\\", "/")
        frontmatter, body = _markdown_frontmatter(root=state.root, rel_path=rel_path)
        doc_ids = _governance_doc_aliases(rel_path=rel_path, frontmatter=frontmatter)
        influence_status, influence_summary = status_rows.get(rel_path, ("", ""))
        doc_node_id = _ensure_governance_doc_node(
            state,
            rel_path=rel_path,
            default_status_hint=influence_status,
            reasoning_summary=(
                influence_summary
                or f"governance_doc role={frontmatter.get('doc_role', '<unset>')}"
            ),
        )
        _add_edge(state, "indexes", influence_index_node_id, doc_node_id)
        for doc_id in doc_ids:
            for sppf_node_id in sppf_doc_node_ids.get(doc_id, ()):
                _add_edge(state, "tracks", doc_node_id, sppf_node_id)
        inside_fence = False
        current_section_node_id = ""
        current_section_slug = ""
        for line_number, raw_line in enumerate(body.splitlines(), start=1):
            stripped = raw_line.strip()
            if stripped.startswith("```"):
                inside_fence = not inside_fence
                continue
            if inside_fence or not stripped:
                continue
            heading_match = _HEADING_RE.match(raw_line)
            if heading_match is not None:
                title = heading_match.group("title").strip()
                category = _section_category(title)
                section_node_id = f"governance_section:{stable_hash(rel_path, line_number, title)}"
                section_node = _synthetic_node(
                    node_id=section_node_id,
                    title=title,
                    ref_kind="governance_section",
                    value=f"{rel_path}:{line_number}:{title}",
                    doc_ids=doc_ids,
                    reasoning_summary=f"inbox governance section category={category}",
                    reasoning_control="invariant_graph.inbox_governance.section",
                    rel_path=rel_path,
                    line=line_number,
                    node_kind="governance_section",
                    status_hint=category,
                )
                _add_node(state, section_node, replace=True)
                _link_node_refs(state, section_node)
                _add_edge(state, "contains", doc_node_id, section_node_id)
                current_section_node_id = section_node_id
                current_section_slug = category
                continue
            ordered_match = _ORDERED_LIST_RE.match(raw_line)
            checklist_match = _CHECKLIST_ITEM_RE.match(raw_line)
            item_text = ""
            item_kind = ""
            if ordered_match is not None:
                item_text = ordered_match.group("text").strip()
                item_kind = "ordered"
            elif checklist_match is not None:
                item_text = checklist_match.group("text").strip()
                item_kind = checklist_match.group("marker").strip() or "checklist"
            if not item_text:
                continue
            item_node_id = f"governance_action_item:{stable_hash(rel_path, line_number, item_text)}"
            item_node = _synthetic_node(
                node_id=item_node_id,
                title=item_text,
                ref_kind="governance_action_item",
                value=f"{rel_path}:{line_number}:{item_text}",
                doc_ids=doc_ids,
                reasoning_summary=(
                    f"inbox governance action kind={item_kind} section={current_section_slug or 'none'}"
                ),
                reasoning_control="invariant_graph.inbox_governance.action_item",
                rel_path=rel_path,
                line=line_number,
                node_kind="governance_action_item",
                status_hint=current_section_slug or item_kind,
            )
            _add_node(state, item_node, replace=True)
            _link_node_refs(state, item_node)
            parent_node_id = current_section_node_id or doc_node_id
            _add_edge(state, "contains", parent_node_id, item_node_id)
            for issue_id in _sorted(list(set(_GH_ISSUE_RE.findall(item_text)))):
                issue_node_id = f"sppf_issue:{issue_id}"
                if issue_node_id in state.nodes_by_id:
                    _add_edge(state, "tracks", item_node_id, issue_node_id)


def _join_governance_convergence_sources(state: _InvariantGraphBuildState) -> None:
    checklist_node_id = _ensure_governance_doc_node(
        state,
        rel_path=_SPPF_CHECKLIST_DOC.as_posix(),
    )
    influence_index_node_id = _ensure_governance_doc_node(
        state,
        rel_path=_INFLUENCE_INDEX_DOC.as_posix(),
    )
    sppf_doc_node_ids = _join_sppf_dependency_graph(
        state,
        checklist_node_id=checklist_node_id,
    )
    _join_inbox_governance_docs(
        state,
        influence_index_node_id=influence_index_node_id,
        sppf_doc_node_ids=sppf_doc_node_ids,
    )


def _governance_doc_ids_for_path(
    state: _InvariantGraphBuildState,
    *,
    rel_path: str,
) -> tuple[str, ...]:
    if not rel_path.endswith(".md"):
        return ()
    frontmatter, _body = _markdown_frontmatter(root=state.root, rel_path=rel_path)
    return _governance_doc_aliases(rel_path=rel_path, frontmatter=frontmatter)


def _link_to_governance_doc_if_present(
    state: _InvariantGraphBuildState,
    *,
    source_node_id: str,
    rel_path: str,
) -> None:
    normalized = _normalize_rel_path(state.root, rel_path)
    if not normalized.endswith(".md"):
        return
    target_node_id = _ensure_governance_doc_node(
        state,
        rel_path=normalized,
    )
    _add_edge(state, "tracks", source_node_id, target_node_id)


def _link_to_git_state_entries_for_path(
    state: _InvariantGraphBuildState,
    *,
    source_node_id: str,
    rel_path: str,
) -> None:
    normalized = _normalize_rel_path(state.root, rel_path)
    if not normalized:
        return
    for node in state.nodes_by_id.values():
        if node.node_kind != "git_state_entry":
            continue
        if node.rel_path != normalized:
            continue
        _add_edge(state, "touches", source_node_id, node.node_id)


def _link_to_existing_nodes_for_path(
    state: _InvariantGraphBuildState,
    *,
    source_node_id: str,
    rel_path: str,
    edge_kind: str = "tracks",
) -> None:
    normalized = _normalize_rel_path(state.root, rel_path)
    if not normalized:
        return
    for node in state.nodes_by_id.values():
        if node.node_id == source_node_id:
            continue
        if node.rel_path != normalized:
            continue
        _add_edge(state, edge_kind, source_node_id, node.node_id)


def _append_diagnostic(
    state: _InvariantGraphBuildState,
    *,
    severity: str,
    code: str,
    node_id: str,
    raw_dependency: str,
    message: str,
) -> None:
    state.diagnostics.append(
        InvariantGraphDiagnostic(
            diagnostic_id=stable_hash(
                "invariant_graph_diagnostic",
                severity,
                code,
                node_id,
                raw_dependency,
                message,
            ),
            severity=severity,
            code=code,
            node_id=node_id,
            raw_dependency=raw_dependency,
            message=message,
        )
    )


def _append_ranking_signal(
    state: _InvariantGraphBuildState,
    *,
    node_id: str,
    touchpoint_object_id: str,
    touchsite_object_id: str,
    code: str,
    score: int,
    message: str,
    raw_dependency: str = "",
) -> None:
    signal_id = stable_hash(
        "invariant_graph_ranking_signal",
        node_id,
        touchpoint_object_id,
        touchsite_object_id,
        code,
        str(score),
        raw_dependency,
        message,
    )
    if any(existing.signal_id == signal_id for existing in state.ranking_signals):
        return
    state.ranking_signals.append(
        InvariantGraphRankingSignal(
            signal_id=signal_id,
            code=code,
            node_id=node_id,
            touchpoint_object_id=touchpoint_object_id,
            touchsite_object_id=touchsite_object_id,
            raw_dependency=raw_dependency,
            score=score,
            message=message,
        )
    )


def _join_docflow_compliance_artifact(state: _InvariantGraphBuildState) -> None:
    artifact = load_docflow_compliance_artifact(
        root=state.root,
        rel_path=_DOCFLOW_COMPLIANCE_ARTIFACT.as_posix(),
        identities=state.structured_artifact_identities,
    )
    if artifact is None:
        return
    report_node_id = "docflow_compliance_report:artifact"
    report_node = _synthetic_node(
        node_id=report_node_id,
        title="docflow compliance",
        ref_kind="docflow_compliance_report",
        value=_DOCFLOW_COMPLIANCE_ARTIFACT.as_posix(),
        object_ids=(artifact.identity.wire(),),
        reasoning_summary=(
            "compliant={compliant} contradicts={contradicts} excess={excess} "
            "proposed={proposed} unmet_fail={unmet_fail}".format(
                compliant=artifact.compliant_count,
                contradicts=artifact.contradiction_count,
                excess=artifact.excess_count,
                proposed=artifact.proposed_count,
                unmet_fail=artifact.unmet_fail_count,
            )
        ),
        reasoning_control="invariant_graph.docflow_compliance",
        rel_path=_DOCFLOW_COMPLIANCE_ARTIFACT.as_posix(),
        node_kind="docflow_compliance_report",
        status_hint=(
            "contradicts"
            if artifact.contradiction_count or artifact.unmet_fail_count
            else "ready"
        ),
    )
    _add_node(state, report_node, replace=True)
    _link_node_refs(state, report_node)
    for touchpoint_object_id in ("CSA-RGC-TP-006", "CSA-RGC-TP-007"):
        touchpoint_node_id = state.object_node_ids.get(touchpoint_object_id, "")
        if touchpoint_node_id:
            _add_edge(state, "contains", touchpoint_node_id, report_node_id)
    for rel_path in artifact.changed_paths:
        _link_to_governance_doc_if_present(
            state,
            source_node_id=report_node_id,
            rel_path=rel_path,
        )
        _link_to_git_state_entries_for_path(
            state,
            source_node_id=report_node_id,
            rel_path=rel_path,
        )
    for row in artifact.rows:
        row_node_id = f"docflow_compliance_row:{stable_hash(row.identity.wire())}"
        row_title = row.invariant or row.evidence_id or row.row_id
        row_node = _synthetic_node(
            node_id=row_node_id,
            title=row_title,
            ref_kind="docflow_compliance_row",
            value=row.identity.wire(),
            object_ids=tuple(
                item
                for item in (
                    row.identity.wire(),
                    row.row_id,
                    row.invariant,
                    row.evidence_id,
                )
                if item
            ),
            doc_ids=_governance_doc_ids_for_path(state, rel_path=row.rel_path),
            reasoning_summary=(
                "docflow compliance status={status} source_row_kind={source_row_kind} detail={detail}".format(
                    status=row.status or "<unset>",
                    source_row_kind=row.source_row_kind or "<unset>",
                    detail=row.detail or "<none>",
                )
            ),
            reasoning_control="invariant_graph.docflow_compliance.row",
            rel_path=row.rel_path,
            node_kind="docflow_compliance_row",
            status_hint=row.status,
        )
        _add_node(state, row_node, replace=True)
        _link_node_refs(state, row_node)
        _add_edge(state, "contains", report_node_id, row_node_id)
        if row.rel_path:
            _link_to_governance_doc_if_present(
                state,
                source_node_id=row_node_id,
                rel_path=row.rel_path,
            )
            _link_to_git_state_entries_for_path(
                state,
                source_node_id=row_node_id,
                rel_path=row.rel_path,
            )
    for obligation in artifact.obligations:
        obligation_node_id = (
            f"docflow_obligation:{stable_hash(obligation.identity.wire())}"
        )
        status_hint = obligation.status or "unknown"
        if obligation.triggered and obligation.status != "met":
            status_hint = f"unmet_{obligation.enforcement or 'unknown'}"
        obligation_node = _synthetic_node(
            node_id=obligation_node_id,
            title=obligation.obligation_id,
            ref_kind="docflow_obligation",
            value=obligation.identity.wire(),
            object_ids=tuple(
                item
                for item in (obligation.identity.wire(), obligation.obligation_id)
                if item
            ),
            reasoning_summary=obligation.description or obligation.obligation_id,
            reasoning_control="invariant_graph.docflow_compliance.obligation",
            rel_path=_DOCFLOW_COMPLIANCE_ARTIFACT.as_posix(),
            node_kind="docflow_obligation",
            status_hint=status_hint,
        )
        _add_node(state, obligation_node, replace=True)
        _link_node_refs(state, obligation_node)
        _add_edge(state, "contains", report_node_id, obligation_node_id)
        for rel_path in artifact.changed_paths:
            _link_to_governance_doc_if_present(
                state,
                source_node_id=obligation_node_id,
                rel_path=rel_path,
            )
            _link_to_git_state_entries_for_path(
                state,
                source_node_id=obligation_node_id,
                rel_path=rel_path,
            )


def _resolve_sppf_issue_node_id(
    state: _InvariantGraphBuildState,
    *,
    issue_id: str,
) -> str:
    normalized = str(issue_id).strip()
    if not normalized:
        return ""
    candidates = [normalized]
    if normalized.startswith("GH-"):
        candidates.append(normalized[3:])
    else:
        candidates.append(f"GH-{normalized}")
    for candidate in candidates:
        node_id = f"sppf_issue:{candidate}"
        if node_id in state.nodes_by_id:
            return node_id
    return ""


_DOCFLOW_PROVENANCE_RANKING_RULES = (
    RankingSignalRule(
        rule_id="docflow_issue_lifecycle_fetch_incomplete",
        entry_path=(),
        diagnostic_code="sppf_issue_lifecycle_fetch_incomplete",
        severity="warning",
        score=5,
        message_template=(
            "issue lifecycle fetch status {fetch_status} for strict docflow provenance "
            "reported {error_count} error(s)"
        ),
        captures=(
            RankingSignalCapture(
                name="fetch_status",
                path=("issue_lifecycle_fetch_status",),
            ),
            RankingSignalCapture(
                name="error_count",
                path=("issue_lifecycle_errors",),
                render_as="count",
            ),
        ),
        predicates=(
            RankingSignalPredicate(
                path=("issue_lifecycle_fetch_status",),
                op="in",
                expected=("error", "partial_error"),
            ),
        ),
    ),
    RankingSignalRule(
        rule_id="docflow_issue_lifecycle_state_mismatch",
        entry_path=("issue_lifecycles", "*"),
        diagnostic_code="sppf_issue_lifecycle_state_mismatch",
        severity="warning",
        score=4,
        message_template=(
            "GH-{issue_id} is {state}; active correction-unit linkage expects open lifecycle"
        ),
        captures=(
            RankingSignalCapture(name="issue_id", path=("issue_id",)),
            RankingSignalCapture(name="state", path=("state",)),
        ),
        predicates=(
            RankingSignalPredicate(
                path=("state",),
                op="not_in",
                expected=("open",),
            ),
        ),
    ),
    RankingSignalRule(
        rule_id="docflow_issue_lifecycle_missing_required_labels",
        entry_path=("issue_lifecycles", "*"),
        diagnostic_code="sppf_issue_lifecycle_missing_required_labels",
        severity="warning",
        score=3,
        message_template=(
            "GH-{issue_id} is missing required lifecycle label(s): {missing_labels}"
        ),
        captures=(RankingSignalCapture(name="issue_id", path=("issue_id",)),),
        predicates=(
            RankingSignalPredicate(
                path=("labels",),
                op="missing_any",
                expected=_DOCFLOW_REQUIRED_ISSUE_LIFECYCLE_LABELS,
                bind_name="missing_labels",
            ),
        ),
    ),
)


def _join_docflow_provenance_artifact(state: _InvariantGraphBuildState) -> None:
    artifact = load_docflow_compliance_artifact(
        root=state.root,
        rel_path=_DOCFLOW_COMPLIANCE_ARTIFACT.as_posix(),
        identities=state.structured_artifact_identities,
    )
    if artifact is None:
        return
    if not (
        artifact.rev_range
        or artifact.commits
        or artifact.issue_references
        or artifact.sppf_relevant_paths_changed
    ):
        return
    report_node_id = "docflow_provenance_report:artifact"
    report_node = _synthetic_node(
        node_id=report_node_id,
        title="docflow provenance",
        ref_kind="docflow_provenance_report",
        value=_DOCFLOW_COMPLIANCE_ARTIFACT.as_posix(),
        object_ids=tuple(
            item
            for item in (artifact.identity.wire(), artifact.rev_range)
            if item
        ),
        reasoning_summary=(
            "rev_range={rev_range} commits={commits} issue_refs={issue_refs} "
            "issue_lifecycles={issue_lifecycles} lifecycle_fetch={fetch_status} "
            "gh_reference_validated={validated}".format(
                rev_range=artifact.rev_range or "<unset>",
                commits=len(artifact.commits),
                issue_refs=len(artifact.issue_references),
                issue_lifecycles=len(artifact.issue_lifecycles),
                fetch_status=artifact.issue_lifecycle_fetch_status or "<unset>",
                validated=artifact.gh_reference_validated,
            )
        ),
        reasoning_control="invariant_graph.docflow_provenance",
        rel_path=_DOCFLOW_COMPLIANCE_ARTIFACT.as_posix(),
        node_kind="docflow_provenance_report",
        status_hint=(
            "unmet_fail"
            if artifact.sppf_relevant_paths_changed and not artifact.gh_reference_validated
            else "ready"
        ),
    )
    _add_node(state, report_node, replace=True)
    _link_node_refs(state, report_node)
    touchpoint_node_id = state.object_node_ids.get("CSA-RGC-TP-007", "")
    if touchpoint_node_id:
        _add_edge(state, "contains", touchpoint_node_id, report_node_id)
    for rel_path in artifact.changed_paths:
        _link_to_governance_doc_if_present(
            state,
            source_node_id=report_node_id,
            rel_path=rel_path,
        )
        _link_to_git_state_entries_for_path(
            state,
            source_node_id=report_node_id,
            rel_path=rel_path,
        )
    parent_node_id = report_node_id
    if artifact.rev_range:
        rev_range_node_id = f"docflow_commit_range:{stable_hash(artifact.rev_range)}"
        rev_range_node = _synthetic_node(
            node_id=rev_range_node_id,
            title=artifact.rev_range,
            ref_kind="docflow_commit_range",
            value=artifact.rev_range,
            object_ids=(artifact.rev_range,),
            reasoning_summary=(
                f"strict docflow commit range with {len(artifact.commits)} commit(s)"
            ),
            reasoning_control="invariant_graph.docflow_provenance.commit_range",
            rel_path=_DOCFLOW_COMPLIANCE_ARTIFACT.as_posix(),
            node_kind="docflow_commit_range",
            status_hint=(
                "issue_refs_missing"
                if artifact.sppf_relevant_paths_changed and not artifact.gh_reference_validated
                else ""
            ),
        )
        _add_node(state, rev_range_node, replace=True)
        _link_node_refs(state, rev_range_node)
        _add_edge(state, "contains", report_node_id, rev_range_node_id)
        parent_node_id = rev_range_node_id
    for commit in artifact.commits:
        commit_node_id = f"docflow_commit:{stable_hash(commit.identity.wire())}"
        commit_node = _synthetic_node(
            node_id=commit_node_id,
            title=str(commit),
            ref_kind="docflow_commit",
            value=commit.identity.wire(),
            object_ids=tuple(
                item for item in (commit.identity.wire(), commit.sha) if item
            ),
            reasoning_summary=commit.subject or commit.sha,
            reasoning_control="invariant_graph.docflow_provenance.commit",
            rel_path=_DOCFLOW_COMPLIANCE_ARTIFACT.as_posix(),
            node_kind="docflow_commit",
            status_hint=(
                "issue_refs_missing"
                if artifact.sppf_relevant_paths_changed and not artifact.gh_reference_validated
                else ""
            ),
        )
        _add_node(state, commit_node, replace=True)
        _link_node_refs(state, commit_node)
        _add_edge(state, "contains", parent_node_id, commit_node_id)
        git_head_node_id = f"git_head_commit:{commit.sha}"
        if git_head_node_id in state.nodes_by_id:
            _add_edge(state, "tracks", commit_node_id, git_head_node_id)
    issue_reference_node_ids: dict[str, str] = {}
    issue_lifecycle_node_ids: dict[str, str] = {}
    for issue_reference in artifact.issue_references:
        issue_node_id = (
            f"docflow_issue_reference:{stable_hash(issue_reference.identity.wire())}"
        )
        issue_node = _synthetic_node(
            node_id=issue_node_id,
            title=str(issue_reference),
            ref_kind="docflow_issue_reference",
            value=issue_reference.identity.wire(),
            object_ids=tuple(
                item
                for item in (
                    issue_reference.identity.wire(),
                    issue_reference.issue_id,
                    f"GH-{issue_reference.issue_id}",
                )
                if item
            ),
            reasoning_summary=(
                f"strict docflow issue reference commit_count={issue_reference.commit_count}"
            ),
            reasoning_control="invariant_graph.docflow_provenance.issue_reference",
            rel_path=_DOCFLOW_COMPLIANCE_ARTIFACT.as_posix(),
            node_kind="docflow_issue_reference",
            status_hint="ready" if artifact.gh_reference_validated else "",
        )
        _add_node(state, issue_node, replace=True)
        _link_node_refs(state, issue_node)
        _add_edge(state, "contains", report_node_id, issue_node_id)
        issue_reference_node_ids[issue_reference.issue_id] = issue_node_id
        checklist_issue_node_id = _resolve_sppf_issue_node_id(
            state,
            issue_id=issue_reference.issue_id,
        )
        if checklist_issue_node_id:
            _add_edge(state, "tracks", issue_node_id, checklist_issue_node_id)
    for issue_lifecycle in artifact.issue_lifecycles:
        lifecycle_node_id = (
            f"docflow_issue_lifecycle:{stable_hash(issue_lifecycle.identity.wire())}"
        )
        lifecycle_node = _synthetic_node(
            node_id=lifecycle_node_id,
            title=str(issue_lifecycle),
            ref_kind="docflow_issue_lifecycle",
            value=issue_lifecycle.identity.wire(),
            object_ids=tuple(
                item
                for item in (
                    issue_lifecycle.identity.wire(),
                    issue_lifecycle.issue_id,
                    f"GH-{issue_lifecycle.issue_id}",
                    *issue_lifecycle.labels,
                )
                if item
            ),
            reasoning_summary=(
                "issue lifecycle state={state} labels={labels}".format(
                    state=issue_lifecycle.state or "<unset>",
                    labels=",".join(issue_lifecycle.labels) or "<none>",
                )
            ),
            reasoning_control="invariant_graph.docflow_provenance.issue_lifecycle",
            rel_path=_DOCFLOW_COMPLIANCE_ARTIFACT.as_posix(),
            node_kind="docflow_issue_lifecycle",
            status_hint=issue_lifecycle.state,
        )
        _add_node(state, lifecycle_node, replace=True)
        _link_node_refs(state, lifecycle_node)
        issue_lifecycle_node_ids[issue_lifecycle.issue_id] = lifecycle_node_id
        _add_edge(
            state,
            "contains",
            issue_reference_node_ids.get(issue_lifecycle.issue_id, report_node_id),
            lifecycle_node_id,
        )
        checklist_issue_node_id = _resolve_sppf_issue_node_id(
            state,
            issue_id=issue_lifecycle.issue_id,
        )
        if checklist_issue_node_id:
            _add_edge(state, "tracks", lifecycle_node_id, checklist_issue_node_id)
    for match in evaluate_ranking_signal_rules(
        carrier=artifact,
        rules=_DOCFLOW_PROVENANCE_RANKING_RULES,
    ):
        capture_map = match.capture_map()
        target_node_id = report_node_id
        if "issue_id" in capture_map:
            target_node_id = issue_lifecycle_node_ids.get(
                str(capture_map["issue_id"]),
                issue_reference_node_ids.get(str(capture_map["issue_id"]), report_node_id),
            )
        _append_diagnostic(
            state,
            severity=match.severity,
            code=match.diagnostic_code,
            node_id=target_node_id,
            raw_dependency="CSA-RGC-TP-007",
            message=match.message,
        )
        _append_ranking_signal(
            state,
            node_id=target_node_id,
            touchpoint_object_id="CSA-RGC-TP-007",
            touchsite_object_id=(
                "CSA-RGC-TS-043"
                if match.rule_id == "docflow_issue_lifecycle_fetch_incomplete"
                else "CSA-RGC-TS-034"
            ),
            code=match.rule_id,
            score=match.score,
            raw_dependency="CSA-RGC-TP-007",
            message=match.message,
        )
    if artifact.sppf_relevant_paths_changed and not artifact.gh_reference_validated:
        sample_paths = ", ".join(_sorted(list(artifact.changed_paths))[:5])
        suffix = f" touching {sample_paths}" if sample_paths else ""
        diagnostic_message = (
            f"sppf_gh_reference_validation unmet for {artifact.rev_range or '<unknown-range>'}"
            f"{suffix}; no GH references were found for the current strict-docflow provenance carrier."
        )
        diagnostic_node_id = state.object_node_ids.get("CSA-RGC-TS-031", report_node_id)
        _append_diagnostic(
            state,
            severity="warning",
            code="sppf_gh_reference_validation",
            node_id=diagnostic_node_id,
            raw_dependency=artifact.rev_range or "docflow_provenance",
            message=diagnostic_message,
        )


def _join_docflow_packet_enforcement(state: _InvariantGraphBuildState) -> None:
    artifact = load_docflow_packet_enforcement_artifact(
        root=state.root,
        rel_path=_DOCFLOW_PACKET_ENFORCEMENT_ARTIFACT.as_posix(),
        identities=state.structured_artifact_identities,
    )
    if artifact is None:
        return
    report_node_id = "docflow_packet_enforcement:artifact"
    report_node = _synthetic_node(
        node_id=report_node_id,
        title="docflow packet enforcement",
        ref_kind="docflow_packet_enforcement",
        value=_DOCFLOW_PACKET_ENFORCEMENT_ARTIFACT.as_posix(),
        object_ids=(artifact.identity.wire(),),
        reasoning_summary=(
            "docflow packet enforcement blocked={blocked} drifted={drifted} new_rows={new_rows}".format(
                blocked=artifact.blocked,
                drifted=artifact.drifted,
                new_rows=artifact.new_row_count,
            )
        ),
        reasoning_control="invariant_graph.docflow_packet_enforcement",
        rel_path=_DOCFLOW_PACKET_ENFORCEMENT_ARTIFACT.as_posix(),
        node_kind="docflow_packet_enforcement",
        status_hint="blocked"
        if artifact.blocked or artifact.drifted or artifact.new_row_count
        else "ready",
    )
    _add_node(state, report_node, replace=True)
    for packet in artifact.packets:
        packet_node_id = f"docflow_packet:{stable_hash(packet.identity.wire())}"
        packet_node = _synthetic_node(
            node_id=packet_node_id,
            title=packet.packet_path or packet.classification or "docflow packet",
            ref_kind="docflow_packet",
            value=packet.identity.wire(),
            object_ids=(packet.identity.wire(),),
            doc_ids=_governance_doc_ids_for_path(state, rel_path=packet.packet_path),
            reasoning_summary=(
                "docflow packet status={status} classification={classification} rows={rows}".format(
                    status=packet.status or "<unset>",
                    classification=packet.classification or "<unset>",
                    rows=len(packet.row_ids),
                )
            ),
            reasoning_control="invariant_graph.docflow_packet_enforcement.packet",
            rel_path=packet.packet_path,
            node_kind="docflow_packet",
            status_hint=packet.status or packet.classification,
        )
        _add_node(state, packet_node, replace=True)
        _link_node_refs(state, packet_node)
        _add_edge(state, "contains", report_node_id, packet_node_id)
        _link_to_governance_doc_if_present(
            state,
            source_node_id=packet_node_id,
            rel_path=packet.packet_path,
        )
        for row_item in packet.rows:
            row_node_id = f"docflow_packet_row:{stable_hash(row_item.identity.wire())}"
            row_node = _synthetic_node(
                node_id=row_node_id,
                title=row_item.row_id,
                ref_kind="docflow_packet_row",
                value=row_item.identity.wire(),
                object_ids=(row_item.identity.wire(), row_item.row_id),
                doc_ids=packet_node.doc_ids,
                reasoning_summary=(
                    "docflow row status={status} packet={path}".format(
                        status=row_item.status or "<unset>",
                        path=row_item.packet_path or "<unset>",
                    )
                ),
                reasoning_control="invariant_graph.docflow_packet_enforcement.row",
                rel_path=row_item.packet_path,
                node_kind="docflow_packet_row",
                status_hint=row_item.status,
            )
            _add_node(state, row_node, replace=True)
            _link_node_refs(state, row_node)
            _add_edge(state, "contains", packet_node_id, row_node_id)
            _link_to_governance_doc_if_present(
                state,
                source_node_id=row_node_id,
                rel_path=row_item.packet_path,
            )
    for rel_path_list in (
        artifact.changed_paths,
        artifact.out_of_scope_touches,
        artifact.unresolved_touched_packets,
    ):
        for rel_path in rel_path_list:
            _link_to_governance_doc_if_present(
                state,
                source_node_id=report_node_id,
                rel_path=rel_path,
            )


def _join_controller_drift_artifact(state: _InvariantGraphBuildState) -> None:
    artifact = load_controller_drift_artifact(
        root=state.root,
        rel_path=_CONTROLLER_DRIFT_ARTIFACT.as_posix(),
        identities=state.structured_artifact_identities,
    )
    if artifact is None:
        return
    report_node_id = "controller_drift_report:artifact"
    report_node = _synthetic_node(
        node_id=report_node_id,
        title="controller drift",
        ref_kind="controller_drift_report",
        value=_CONTROLLER_DRIFT_ARTIFACT.as_posix(),
        object_ids=(artifact.identity.wire(),),
        reasoning_summary=(
            "controller drift findings={findings} highest_severity={severity}".format(
                findings=artifact.total_findings,
                severity=artifact.highest_severity or "<unset>",
            )
        ),
        reasoning_control="invariant_graph.controller_drift",
        rel_path=_CONTROLLER_DRIFT_ARTIFACT.as_posix(),
        node_kind="controller_drift_report",
        status_hint=artifact.highest_severity,
    )
    _add_node(state, report_node, replace=True)
    for finding in artifact.findings:
        finding_node_id = f"controller_drift_finding:{stable_hash(finding.identity.wire())}"
        finding_node = _synthetic_node(
            node_id=finding_node_id,
            title=finding.detail or finding.sensor or "controller drift finding",
            ref_kind="controller_drift_finding",
            value=finding.identity.wire(),
            object_ids=tuple(
                item
                for item in (finding.identity.wire(), finding.anchor)
                if item
            ),
            reasoning_summary=finding.detail or finding.sensor or finding.anchor,
            reasoning_control="invariant_graph.controller_drift.finding",
            rel_path=_CONTROLLER_DRIFT_ARTIFACT.as_posix(),
            node_kind="controller_drift_finding",
            status_hint=finding.severity or finding.sensor,
        )
        _add_node(state, finding_node, replace=True)
        _link_node_refs(state, finding_node)
        _add_edge(state, "contains", report_node_id, finding_node_id)
        for rel_path in finding.doc_paths:
            _link_to_governance_doc_if_present(
                state,
                source_node_id=finding_node_id,
                rel_path=rel_path,
            )


def _join_local_repro_closure_ledger(state: _InvariantGraphBuildState) -> None:
    artifact = load_local_repro_closure_ledger_artifact(
        root=state.root,
        rel_path=_LOCAL_REPRO_CLOSURE_LEDGER_ARTIFACT.as_posix(),
        identities=state.structured_artifact_identities,
    )
    if artifact is None:
        return
    report_node_id = "local_repro_closure_ledger:artifact"
    report_node = _synthetic_node(
        node_id=report_node_id,
        title="local repro closure ledger",
        ref_kind="local_repro_closure_ledger",
        value=_LOCAL_REPRO_CLOSURE_LEDGER_ARTIFACT.as_posix(),
        object_ids=(artifact.identity.wire(),),
        reasoning_summary=(
            "local repro closure entries={entries} schema_version={schema_version} workstream={workstream}".format(
                entries=len(artifact.entries),
                schema_version=artifact.source.schema_version,
                workstream=artifact.workstream or "<unset>",
            )
        ),
        reasoning_control="invariant_graph.local_repro_closure_ledger",
        rel_path=_LOCAL_REPRO_CLOSURE_LEDGER_ARTIFACT.as_posix(),
        node_kind="local_repro_closure_ledger",
        status_hint=artifact.workstream,
    )
    _add_node(state, report_node, replace=True)
    for entry in artifact.entries:
        status_hint = (
            "pass"
            if entry.validation_statuses
            and all(value == "pass" for value in entry.validation_statuses)
            else ""
        )
        entry_node_id = f"local_repro_entry:{stable_hash(entry.identity.wire())}"
        entry_node = _synthetic_node(
            node_id=entry_node_id,
            title=entry.cu_id or entry.summary or "local repro entry",
            ref_kind="local_repro_entry",
            value=entry.identity.wire(),
            object_ids=tuple(
                item
                for item in (entry.identity.wire(), entry.cu_id)
                if item
            ),
            reasoning_summary=entry.summary or entry.cu_id,
            reasoning_control="invariant_graph.local_repro_closure_ledger.entry",
            rel_path=_LOCAL_REPRO_CLOSURE_LEDGER_ARTIFACT.as_posix(),
            node_kind="local_repro_entry",
            status_hint=status_hint,
        )
        _add_node(state, entry_node, replace=True)
        _link_node_refs(state, entry_node)
        _add_edge(state, "contains", report_node_id, entry_node_id)


def _join_local_ci_repro_contract_artifact(state: _InvariantGraphBuildState) -> None:
    artifact = load_local_ci_repro_contract_artifact(
        root=state.root,
        rel_path=_LOCAL_CI_REPRO_CONTRACT_ARTIFACT.as_posix(),
        identities=state.structured_artifact_identities,
    )
    if artifact is None:
        return
    report_node_id = "local_ci_repro_contract:artifact"
    report_node = _synthetic_node(
        node_id=report_node_id,
        title="local CI reproduction contract",
        ref_kind="local_ci_repro_contract",
        value=_LOCAL_CI_REPRO_CONTRACT_ARTIFACT.as_posix(),
        object_ids=(artifact.identity.wire(),),
        reasoning_summary=(
            "local CI reproduction surfaces={surfaces} relations={relations}".format(
                surfaces=len(artifact.surfaces),
                relations=len(artifact.relations),
            )
        ),
        reasoning_control="invariant_graph.local_ci_repro_contract",
        rel_path=_LOCAL_CI_REPRO_CONTRACT_ARTIFACT.as_posix(),
        node_kind="local_ci_repro_contract",
        status_hint="pass"
        if all(item.status == "pass" for item in artifact.surfaces)
        and all(item.status == "pass" for item in artifact.relations)
        else "fail",
    )
    _add_node(state, report_node, replace=True)
    _link_node_refs(state, report_node)
    touchpoint_node_id = state.object_node_ids.get("CSA-RGC-TP-006", "")
    if touchpoint_node_id:
        _add_edge(state, "contains", touchpoint_node_id, report_node_id)
    surface_node_ids: dict[str, str] = {}
    for surface in artifact.surfaces:
        surface_node_id = f"local_ci_repro_surface:{stable_hash(surface.identity.wire())}"
        surface_node = _synthetic_node(
            node_id=surface_node_id,
            title=surface.title,
            ref_kind="local_ci_repro_surface",
            value=surface.identity.wire(),
            object_ids=tuple(
                item
                for item in (
                    surface.identity.wire(),
                    surface.surface_id,
                    surface.source_ref,
                    *surface.artifacts,
                )
                if item
            ),
            reasoning_summary=surface.summary,
            reasoning_control="invariant_graph.local_ci_repro_contract.surface",
            rel_path=_LOCAL_CI_REPRO_CONTRACT_ARTIFACT.as_posix(),
            node_kind="local_ci_repro_surface",
            status_hint=surface.status,
        )
        _add_node(state, surface_node, replace=True)
        _link_node_refs(state, surface_node)
        _add_edge(state, "contains", report_node_id, surface_node_id)
        surface_node_ids[surface.surface_id] = surface_node_id
        for artifact_path in surface.artifacts:
            _link_to_existing_nodes_for_path(
                state,
                source_node_id=surface_node_id,
                rel_path=artifact_path,
            )
        if surface.required_capabilities:
            for capability in surface.required_capabilities:
                capability_node_id = (
                    f"local_ci_repro_capability:{stable_hash(capability.identity.wire())}"
                )
                capability_node = _synthetic_node(
                    node_id=capability_node_id,
                    title=capability.capability_id,
                    ref_kind="local_ci_repro_capability",
                    value=capability.identity.wire(),
                    object_ids=tuple(
                        item
                        for item in (
                            capability.identity.wire(),
                            capability.capability_id,
                            surface.surface_id,
                            surface.source_ref,
                        )
                        if item
                    ),
                    reasoning_summary=capability.summary,
                    reasoning_control=(
                        "invariant_graph.local_ci_repro_contract.capability"
                    ),
                    rel_path=_LOCAL_CI_REPRO_CONTRACT_ARTIFACT.as_posix(),
                    node_kind="local_ci_repro_capability",
                    status_hint=capability.status,
                )
                _add_node(state, capability_node, replace=True)
                _link_node_refs(state, capability_node)
                _add_edge(state, "contains", surface_node_id, capability_node_id)
                if capability.status != "pass":
                    source_groups_text = "; ".join(
                        ", ".join(group)
                        for group in capability.source_alternative_token_groups
                    )
                    command_groups_text = "; ".join(
                        ", ".join(group)
                        for group in capability.command_alternative_token_groups
                    )
                    detail_parts = []
                    if source_groups_text:
                        detail_parts.append(
                            f"source alternatives: {source_groups_text}"
                        )
                    if command_groups_text:
                        detail_parts.append(
                            f"command alternatives: {command_groups_text}"
                        )
                    _append_diagnostic(
                        state,
                        severity="warning",
                        code="ci_repro_capability_contract_violation",
                        node_id=capability_node_id,
                        raw_dependency="CSA-RGC-TP-006",
                        message=(
                            f"{surface.surface_id} missing capability "
                            f"{capability.capability_id}"
                            + (
                                f" ({'; '.join(detail_parts)})"
                                if detail_parts
                                else ""
                            )
                        ),
                    )
        elif surface.status != "pass":
            missing_groups_text = "; ".join(
                ", ".join(group) for group in surface.missing_token_groups
            )
            _append_diagnostic(
                state,
                severity="warning",
                code="ci_repro_surface_contract_violation",
                node_id=surface_node_id,
                raw_dependency="CSA-RGC-TP-006",
                message=(
                    f"{surface.surface_id} missing required token groups: "
                    f"{missing_groups_text or '<none>'}"
                ),
            )
    for relation in artifact.relations:
        relation_node_id = f"local_ci_repro_relation:{stable_hash(relation.identity.wire())}"
        relation_node = _synthetic_node(
            node_id=relation_node_id,
            title=relation.relation_id,
            ref_kind="local_ci_repro_relation",
            value=relation.identity.wire(),
            object_ids=tuple(
                item
                for item in (
                    relation.identity.wire(),
                    relation.relation_id,
                    relation.source_surface_id,
                    relation.target_surface_id,
                )
                if item
            ),
            reasoning_summary=relation.summary,
            reasoning_control="invariant_graph.local_ci_repro_contract.relation",
            rel_path=_LOCAL_CI_REPRO_CONTRACT_ARTIFACT.as_posix(),
            node_kind="local_ci_repro_relation",
            status_hint=relation.status,
        )
        _add_node(state, relation_node, replace=True)
        _link_node_refs(state, relation_node)
        _add_edge(state, "contains", report_node_id, relation_node_id)
        source_node_id = surface_node_ids.get(relation.source_surface_id, "")
        target_node_id = surface_node_ids.get(relation.target_surface_id, "")
        if source_node_id:
            _add_edge(state, "tracks", relation_node_id, source_node_id)
        if target_node_id:
            _add_edge(state, "tracks", relation_node_id, target_node_id)
        if relation.status != "pass":
            missing_capabilities = tuple(
                item
                for item in (
                    *relation.source_missing_capability_ids,
                    *relation.target_missing_capability_ids,
                )
                if item
            )
            _append_diagnostic(
                state,
                severity="warning",
                code="ci_repro_relation_contract_violation",
                node_id=relation_node_id,
                raw_dependency="CSA-RGC-TP-006",
                    message=(
                        f"{relation.relation_id} is not currently satisfiable from the "
                        "declared local/remote CI reproduction surfaces."
                        + (
                        f" Missing capabilities: {', '.join(sorted(missing_capabilities))}."
                            if missing_capabilities
                            else ""
                        )
                    ),
                )


def _join_kernel_vm_alignment_artifact(state: _InvariantGraphBuildState) -> None:
    artifact = load_kernel_vm_alignment_artifact(
        root=state.root,
        rel_path=_KERNEL_VM_ALIGNMENT_ARTIFACT.as_posix(),
        identities=state.structured_artifact_identities,
    )
    if artifact is None:
        return
    report_node_id = "kernel_vm_alignment_report:artifact"
    report_node = _synthetic_node(
        node_id=report_node_id,
        title="kernel VM alignment",
        ref_kind="kernel_vm_alignment_report",
        value=_KERNEL_VM_ALIGNMENT_ARTIFACT.as_posix(),
        object_ids=tuple(
            item for item in (artifact.identity.wire(), artifact.fragment_id) if item
        ),
        reasoning_summary=(
            "kernel VM alignment bindings={bindings} pass={passed} partial={partial} "
            "fail={failed} residues={residues}".format(
                bindings=artifact.binding_count,
                passed=artifact.pass_count,
                partial=artifact.partial_count,
                failed=artifact.fail_count,
                residues=artifact.residue_count,
            )
        ),
        reasoning_control="invariant_graph.kernel_vm_alignment",
        rel_path=_KERNEL_VM_ALIGNMENT_ARTIFACT.as_posix(),
        node_kind="kernel_vm_alignment_report",
        status_hint="pass" if artifact.residue_count == 0 else "partial",
    )
    _add_node(state, report_node, replace=True)
    _link_node_refs(state, report_node)
    touchpoint_node_id = state.object_node_ids.get("CSA-RGC-TP-008", "")
    if touchpoint_node_id:
        _add_edge(state, "contains", touchpoint_node_id, report_node_id)
    binding_node_ids: dict[str, str] = {}
    for binding in artifact.bindings:
        binding_node_id = (
            f"kernel_vm_alignment_binding:{stable_hash(binding.identity.wire())}"
        )
        binding_node = _synthetic_node(
            node_id=binding_node_id,
            title=str(binding),
            ref_kind="kernel_vm_alignment_binding",
            value=binding.identity.wire(),
            object_ids=tuple(
                item
                for item in (
                    binding.identity.wire(),
                    binding.binding_id,
                    binding.fragment_id,
                    *binding.kernel_terms,
                    *binding.runtime_surface_symbols,
                    *binding.realizer_symbols,
                    *binding.runtime_object_symbols,
                )
                if item
            ),
            reasoning_summary=binding.summary,
            reasoning_control="invariant_graph.kernel_vm_alignment.binding",
            rel_path=_KERNEL_VM_ALIGNMENT_ARTIFACT.as_posix(),
            node_kind="kernel_vm_alignment_binding",
            status_hint=binding.status,
        )
        _add_node(state, binding_node, replace=True)
        _link_node_refs(state, binding_node)
        _add_edge(state, "contains", report_node_id, binding_node_id)
        binding_node_ids[binding.binding_id] = binding_node_id
        for rel_path in binding.evidence_paths:
            _link_to_existing_nodes_for_path(
                state,
                source_node_id=binding_node_id,
                rel_path=rel_path,
            )
    for residue in artifact.residues:
        residue_node_id = (
            f"kernel_vm_alignment_residue:{stable_hash(residue.identity.wire())}"
        )
        residue_node = _synthetic_node(
            node_id=residue_node_id,
            title=str(residue),
            ref_kind="kernel_vm_alignment_residue",
            value=residue.identity.wire(),
            object_ids=tuple(
                item
                for item in (
                    residue.identity.wire(),
                    residue.residue_id,
                    residue.binding_id,
                    residue.fragment_id,
                    residue.residue_kind,
                    *residue.kernel_terms,
                    *residue.runtime_surface_symbols,
                    *residue.realizer_symbols,
                    *residue.runtime_object_symbols,
                )
                if item
            ),
            reasoning_summary=residue.message,
            reasoning_control="invariant_graph.kernel_vm_alignment.residue",
            rel_path=_KERNEL_VM_ALIGNMENT_ARTIFACT.as_posix(),
            node_kind="kernel_vm_alignment_residue",
            status_hint=residue.residue_kind,
        )
        _add_node(state, residue_node, replace=True)
        _link_node_refs(state, residue_node)
        _add_edge(
            state,
            "contains",
            binding_node_ids.get(residue.binding_id, report_node_id),
            residue_node_id,
        )
        for rel_path in residue.evidence_paths:
            _link_to_existing_nodes_for_path(
                state,
                source_node_id=residue_node_id,
                rel_path=rel_path,
            )
        _append_diagnostic(
            state,
            severity=residue.severity or "warning",
            code=f"kernel_vm_alignment_{residue.residue_kind or 'residue'}",
            node_id=residue_node_id,
            raw_dependency=residue.binding_id or "CSA-RGC-TP-008",
            message=residue.message,
        )
        _append_ranking_signal(
            state,
            node_id=residue_node_id,
            touchpoint_object_id="CSA-RGC-TP-008",
            touchsite_object_id="CSA-RGC-TS-063",
            code=residue.residue_kind or "kernel_vm_alignment_residue",
            score=residue.score,
            raw_dependency=residue.binding_id,
            message=residue.message,
        )


def _join_identity_grammar_completion_artifact(state: _InvariantGraphBuildState) -> None:
    artifact = load_identity_grammar_completion_artifact(
        root=state.root,
        rel_path=_IDENTITY_GRAMMAR_COMPLETION_ARTIFACT.as_posix(),
        identities=state.structured_artifact_identities,
    )
    if artifact is None:
        return
    report_node_id = "identity_grammar_completion_report:artifact"
    report_node = _synthetic_node(
        node_id=report_node_id,
        title="identity grammar completion",
        ref_kind="identity_grammar_completion_report",
        value=_IDENTITY_GRAMMAR_COMPLETION_ARTIFACT.as_posix(),
        object_ids=(artifact.identity.wire(),),
        reasoning_summary=(
            "identity grammar surfaces={surfaces} pass={passed} fail={failed} "
            "residues={residues}".format(
                surfaces=artifact.surface_count,
                passed=artifact.pass_count,
                failed=artifact.fail_count,
                residues=artifact.residue_count,
            )
        ),
        reasoning_control="invariant_graph.identity_grammar_completion",
        rel_path=_IDENTITY_GRAMMAR_COMPLETION_ARTIFACT.as_posix(),
        node_kind="identity_grammar_completion_report",
        status_hint="pass" if artifact.residue_count == 0 else "partial",
    )
    _add_node(state, report_node, replace=True)
    _link_node_refs(state, report_node)
    touchpoint_node_id = state.object_node_ids.get("CSA-IDR-TP-004", "")
    if touchpoint_node_id:
        _add_edge(state, "contains", touchpoint_node_id, report_node_id)
    surface_node_ids: dict[str, str] = {}
    for surface in artifact.surfaces:
        surface_node_id = (
            f"identity_grammar_completion_surface:{stable_hash(surface.identity.wire())}"
        )
        surface_node = _synthetic_node(
            node_id=surface_node_id,
            title=str(surface),
            ref_kind="identity_grammar_completion_surface",
            value=surface.identity.wire(),
            object_ids=tuple(
                item
                for item in (
                    surface.identity.wire(),
                    surface.surface_id,
                    *surface.residue_ids,
                )
                if item
            ),
            reasoning_summary=surface.summary,
            reasoning_control="invariant_graph.identity_grammar_completion.surface",
            rel_path=_IDENTITY_GRAMMAR_COMPLETION_ARTIFACT.as_posix(),
            node_kind="identity_grammar_completion_surface",
            status_hint=surface.status,
        )
        _add_node(state, surface_node, replace=True)
        _link_node_refs(state, surface_node)
        _add_edge(state, "contains", report_node_id, surface_node_id)
        surface_node_ids[surface.surface_id] = surface_node_id
        for rel_path in surface.evidence_paths:
            _link_to_existing_nodes_for_path(
                state,
                source_node_id=surface_node_id,
                rel_path=rel_path,
            )
    for residue in artifact.residues:
        residue_node_id = (
            f"identity_grammar_completion_residue:{stable_hash(residue.identity.wire())}"
        )
        residue_node = _synthetic_node(
            node_id=residue_node_id,
            title=str(residue),
            ref_kind="identity_grammar_completion_residue",
            value=residue.identity.wire(),
            object_ids=tuple(
                item
                for item in (
                    residue.identity.wire(),
                    residue.residue_id,
                    residue.surface_id,
                    residue.residue_kind,
                )
                if item
            ),
            reasoning_summary=residue.message,
            reasoning_control="invariant_graph.identity_grammar_completion.residue",
            rel_path=_IDENTITY_GRAMMAR_COMPLETION_ARTIFACT.as_posix(),
            node_kind="identity_grammar_completion_residue",
            status_hint=residue.residue_kind,
        )
        _add_node(state, residue_node, replace=True)
        _link_node_refs(state, residue_node)
        _add_edge(
            state,
            "contains",
            surface_node_ids.get(residue.surface_id, report_node_id),
            residue_node_id,
        )
        for rel_path in residue.evidence_paths:
            _link_to_existing_nodes_for_path(
                state,
                source_node_id=residue_node_id,
                rel_path=rel_path,
            )
        _append_diagnostic(
            state,
            severity=residue.severity or "warning",
            code=f"identity_grammar_{residue.residue_kind or 'residue'}",
            node_id=residue_node_id,
            raw_dependency=residue.surface_id or "CSA-IDR-TP-004",
            message=residue.message,
        )
        _append_ranking_signal(
            state,
            node_id=residue_node_id,
            touchpoint_object_id="CSA-IDR-TP-004",
            touchsite_object_id=_IDENTITY_GRAMMAR_TOUCHSITE_IDS.get(
                residue.residue_kind,
                "CSA-IDR-TS-017",
            ),
            code=residue.residue_kind or "identity_grammar_residue",
            score=residue.score,
            raw_dependency=residue.surface_id,
            message=residue.message,
        )


def _join_cross_origin_witness_contract_artifact(state: _InvariantGraphBuildState) -> None:
    artifact = load_cross_origin_witness_contract_artifact(
        root=state.root,
        rel_path=_CROSS_ORIGIN_WITNESS_CONTRACT_ARTIFACT.as_posix(),
        identities=state.structured_artifact_identities,
    )
    if artifact is None:
        return
    report_node_id = "cross_origin_witness_report:artifact"
    report_node = _synthetic_node(
        node_id=report_node_id,
        title="cross-origin witness contract",
        ref_kind="cross_origin_witness_report",
        value=_CROSS_ORIGIN_WITNESS_CONTRACT_ARTIFACT.as_posix(),
        object_ids=(artifact.identity.wire(),),
        reasoning_summary=(
            "cross-origin witness cases={cases} witness_rows={rows} passing={passing}".format(
                cases=len(artifact.cases),
                rows=len(artifact.witness_rows),
                passing=sum(1 for item in artifact.cases if item.status == "pass"),
            )
        ),
        reasoning_control="invariant_graph.cross_origin_witness",
        rel_path=_CROSS_ORIGIN_WITNESS_CONTRACT_ARTIFACT.as_posix(),
        node_kind="cross_origin_witness_report",
        status_hint="pass"
        if all(item.status == "pass" for item in artifact.cases)
        else "fail",
    )
    _add_node(state, report_node, replace=True)
    _link_node_refs(state, report_node)
    touchpoint_node_id = state.object_node_ids.get("CSA-IGM-TP-001", "")
    if touchpoint_node_id:
        _add_edge(state, "contains", touchpoint_node_id, report_node_id)
    row_node_ids: dict[str, str] = {}
    for row in artifact.witness_rows:
        row_node_id = f"cross_origin_witness_row:{stable_hash(row.identity.wire())}"
        row_node = _synthetic_node(
            node_id=row_node_id,
            title=str(row),
            ref_kind="cross_origin_witness_row",
            value=row.identity.wire(),
            object_ids=tuple(
                item
                for item in (
                    row.identity.wire(),
                    row.row_key,
                    row.remap_key,
                    row.left_origin_key,
                    row.right_origin_key,
                )
                if item
            ),
            reasoning_summary=row.summary,
            reasoning_control="invariant_graph.cross_origin_witness.row",
            rel_path=_CROSS_ORIGIN_WITNESS_CONTRACT_ARTIFACT.as_posix(),
            node_kind="cross_origin_witness_row",
            status_hint=row.row_kind,
        )
        _add_node(state, row_node, replace=True)
        _link_node_refs(state, row_node)
        row_node_ids[row.row_key] = row_node_id
    for case in artifact.cases:
        case_node_id = f"cross_origin_witness_case:{stable_hash(case.identity.wire())}"
        case_node = _synthetic_node(
            node_id=case_node_id,
            title=case.title,
            ref_kind="cross_origin_witness_case",
            value=case.identity.wire(),
            object_ids=(case.identity.wire(), case.case_key),
            reasoning_summary=case.summary,
            reasoning_control="invariant_graph.cross_origin_witness.case",
            rel_path=_CROSS_ORIGIN_WITNESS_CONTRACT_ARTIFACT.as_posix(),
            node_kind="cross_origin_witness_case",
            status_hint=case.status,
        )
        _add_node(state, case_node, replace=True)
        _link_node_refs(state, case_node)
        _add_edge(state, "contains", report_node_id, case_node_id)
        for row_key in case.row_keys:
            row_node_id = row_node_ids.get(row_key, "")
            if row_node_id:
                _add_edge(state, "contains", case_node_id, row_node_id)


def _git_state_line_spans_overlap(
    *,
    line: int,
    start_line: int,
    line_count: int,
) -> bool:
    return start_line <= line <= (start_line + line_count - 1)


def _git_state_candidate_nodes(
    state: _InvariantGraphBuildState,
    *,
    entry,
) -> tuple[InvariantGraphNode, ...]:
    return tuple(
        node
        for node in state.nodes_by_id.values()
        if node.rel_path == entry.rel_path and node.node_kind not in _GIT_STATE_NODE_KINDS
    )


def _git_state_participation_parent_node_ids(
    *,
    candidate_nodes: tuple[InvariantGraphNode, ...],
    entry,
) -> tuple[str, ...]:
    if not candidate_nodes:
        return ()
    if not entry.current_line_spans:
        return tuple(_sorted([node.node_id for node in candidate_nodes]))
    overlapping_node_ids = [
        node.node_id
        for node in candidate_nodes
        if node.line > 0
        and any(
            _git_state_line_spans_overlap(
                line=node.line,
                start_line=line_span.start_line,
                line_count=line_span.line_count,
            )
            for line_span in entry.current_line_spans
        )
    ]
    line_agnostic_node_ids = [node.node_id for node in candidate_nodes if node.line <= 0]
    if overlapping_node_ids:
        return tuple(_sorted([*overlapping_node_ids, *line_agnostic_node_ids]))
    if line_agnostic_node_ids:
        return tuple(_sorted(line_agnostic_node_ids))
    return ()


def _join_git_state_artifact(state: _InvariantGraphBuildState) -> None:
    artifact = load_git_state_artifact(
        root=state.root,
        rel_path=_GIT_STATE_ARTIFACT.as_posix(),
        identities=state.structured_artifact_identities,
        prefer_live_repo_state=True,
    )
    if artifact is None:
        return
    counts = defaultdict(int)
    for entry in artifact.entries:
        counts[entry.state_class] += 1
    report_node_id = "git_state_report:artifact"
    report_node = _synthetic_node(
        node_id=report_node_id,
        title="git state",
        ref_kind="git_state_report",
        value=_GIT_STATE_ARTIFACT.as_posix(),
        object_ids=tuple(
            item
            for item in (artifact.identity.wire(), artifact.head_sha)
            if item
        ),
        reasoning_summary=(
            "head={head} branch={branch} committed={committed} staged={staged} "
            "unstaged={unstaged} untracked={untracked}".format(
                head=artifact.head_sha[:12] if artifact.head_sha else "<none>",
                branch=artifact.branch or "<detached>",
                committed=counts["committed"],
                staged=counts["staged"],
                unstaged=counts["unstaged"],
                untracked=counts["untracked"],
            )
        ),
        reasoning_control="invariant_graph.git_state",
        rel_path=_GIT_STATE_ARTIFACT.as_posix(),
        node_kind="git_state_report",
        status_hint=artifact.branch or ("detached" if artifact.is_detached else "unknown"),
    )
    _add_node(state, report_node, replace=True)
    _link_node_refs(state, report_node)
    for touchpoint_object_id in ("CSA-IGM-TP-004", "CSA-RGC-TP-006"):
        touchpoint_node_id = state.object_node_ids.get(touchpoint_object_id, "")
        if touchpoint_node_id:
            _add_edge(state, "contains", touchpoint_node_id, report_node_id)
    if artifact.head_sha:
        commit_node_id = f"git_head_commit:{artifact.head_sha}"
        commit_node = _synthetic_node(
            node_id=commit_node_id,
            title=artifact.head_sha[:12],
            ref_kind="git_head_commit",
            value=artifact.head_sha,
            object_ids=(artifact.head_sha,),
            reasoning_summary=(
                "branch={branch} upstream={upstream}".format(
                    branch=artifact.branch or "<detached>",
                    upstream=artifact.upstream or "<none>",
                )
            ),
            reasoning_control="invariant_graph.git_state.head_commit",
            rel_path=_GIT_STATE_ARTIFACT.as_posix(),
            node_kind="git_head_commit",
            status_hint=artifact.branch or "detached",
        )
        _add_node(state, commit_node, replace=True)
        _link_node_refs(state, commit_node)
        _add_edge(state, "contains", report_node_id, commit_node_id)
    for entry in artifact.entries:
        entry_node_id = f"git_state_entry:{stable_hash(entry.identity.wire())}"
        entry_node = _synthetic_node(
            node_id=entry_node_id,
            title=str(entry),
            ref_kind="git_state_entry",
            value=entry.identity.wire(),
            object_ids=tuple(
                item
                for item in (
                    entry.identity.wire(),
                    entry.rel_path,
                    entry.previous_path,
                )
                if item
            ),
            reasoning_summary=(
                f"state_class={entry.state_class} change_code={entry.change_code or '<unset>'}"
            ),
            reasoning_control="invariant_graph.git_state.entry",
            rel_path=entry.rel_path,
            node_kind="git_state_entry",
            status_hint=entry.state_class,
        )
        _add_node(state, entry_node, replace=True)
        _link_node_refs(state, entry_node)
        _add_edge(state, "contains", report_node_id, entry_node_id)
        candidate_nodes = _git_state_candidate_nodes(
            state,
            entry=entry,
        )
        participation_parent_node_ids = _git_state_participation_parent_node_ids(
            candidate_nodes=candidate_nodes,
            entry=entry,
        )
        for parent_node_id in participation_parent_node_ids:
            _add_edge(state, "touches", entry_node_id, parent_node_id)
        if not participation_parent_node_ids:
            for candidate_node in candidate_nodes:
                _add_edge(state, "shares_path_with", entry_node_id, candidate_node.node_id)
        if (
            entry.state_class in _DIRTY_GIT_STATE_CLASSES
            and not _is_mechanically_reconstructable_git_path(entry.rel_path)
        ):
            if participation_parent_node_ids:
                state.diagnostics.append(
                    InvariantGraphDiagnostic(
                        diagnostic_id=stable_hash(
                            "invariant_graph_workspace_preservation",
                            entry_node_id,
                            entry.state_class,
                            entry.rel_path,
                        ),
                        severity="warning",
                        code="workspace_preservation_needed",
                        node_id=entry_node_id,
                        raw_dependency=entry.state_class,
                        message=(
                            f"{entry.state_class} graph-participating change at {entry.rel_path} "
                            f"should be {_workspace_preservation_action_for_state(entry.state_class).replace('_', ' ')}."
                        ),
                    )
                )
            else:
                orphan_reason = (
                    "does not overlap current touchsite lines"
                    if candidate_nodes
                    else "does not match any current workstream-owned path"
                )
                state.diagnostics.append(
                    InvariantGraphDiagnostic(
                        diagnostic_id=stable_hash(
                            "invariant_graph_orphaned_workspace_change",
                            entry_node_id,
                            entry.state_class,
                            entry.rel_path,
                        ),
                        severity="warning",
                        code="orphaned_workspace_change",
                        node_id=entry_node_id,
                        raw_dependency=entry.state_class,
                        message=(
                            f"{entry.state_class} non-ephemeral change at {entry.rel_path} "
                            f"{orphan_reason}; attribute it before preservation."
                        ),
                    )
                )


def _join_ingress_merge_parity_artifact(state: _InvariantGraphBuildState) -> None:
    artifact = load_ingress_merge_parity_artifact(
        root=state.root,
        rel_path=_INGRESS_MERGE_PARITY_ARTIFACT.as_posix(),
        identities=state.structured_artifact_identities,
    )
    if artifact is None:
        return
    report_node_id = "ingress_merge_parity_report:artifact"
    report_node = _synthetic_node(
        node_id=report_node_id,
        title="ingress merge parity",
        ref_kind="ingress_merge_parity_report",
        value=_INGRESS_MERGE_PARITY_ARTIFACT.as_posix(),
        object_ids=(artifact.identity.wire(),),
        reasoning_summary=(
            "ingress merge parity cases={cases} passing={passing}".format(
                cases=len(artifact.cases),
                passing=sum(1 for item in artifact.cases if item.status == "pass"),
            )
        ),
        reasoning_control="invariant_graph.ingress_merge_parity",
        rel_path=_INGRESS_MERGE_PARITY_ARTIFACT.as_posix(),
        node_kind="ingress_merge_parity_report",
        status_hint="pass"
        if all(item.status == "pass" for item in artifact.cases)
        else "fail",
    )
    _add_node(state, report_node, replace=True)
    touchpoint_node_id = state.object_node_ids.get("CSA-IGM-TP-003", "")
    if touchpoint_node_id:
        _add_edge(state, "contains", touchpoint_node_id, report_node_id)
    for case in artifact.cases:
        case_node_id = f"ingress_merge_parity_case:{stable_hash(case.identity.wire())}"
        case_node = _synthetic_node(
            node_id=case_node_id,
            title=case.title,
            ref_kind="ingress_merge_parity_case",
            value=case.identity.wire(),
            object_ids=(case.identity.wire(), case.case_key),
            reasoning_summary=case.summary,
            reasoning_control="invariant_graph.ingress_merge_parity.case",
            rel_path=_INGRESS_MERGE_PARITY_ARTIFACT.as_posix(),
            node_kind="ingress_merge_parity_case",
            status_hint=case.status,
        )
        _add_node(state, case_node, replace=True)
        _link_node_refs(state, case_node)
        _add_edge(state, "contains", report_node_id, case_node_id)


def _join_control_loop_artifacts(state: _InvariantGraphBuildState) -> None:
    _join_docflow_packet_enforcement(state)
    _join_controller_drift_artifact(state)
    _join_local_repro_closure_ledger(state)
    _join_kernel_vm_alignment_artifact(state)
    _join_identity_grammar_completion_artifact(state)
    _join_cross_origin_witness_contract_artifact(state)
    _join_git_state_artifact(state)
    _join_docflow_compliance_artifact(state)
    _join_docflow_provenance_artifact(state)
    _join_ingress_merge_parity_artifact(state)
    _join_local_ci_repro_contract_artifact(state)


def _is_declared_workstream_dependency(
    state: _InvariantGraphBuildState,
    dependency: str,
) -> bool:
    if dependency in state.declared_workstream_ids:
        return True
    if dependency.startswith(("PSF-", "PRF-", "CSA-")):
        return True
    return any(
        dependency == root_id or dependency.startswith(f"{root_id}-")
        for root_id in state.workstream_root_ids
    )


def _resolve_blocking_dependencies(state: _InvariantGraphBuildState) -> None:
    for node in list(state.nodes_by_id.values()):
        for dependency in node.blocking_dependencies:
            target_id = state.object_node_ids.get(dependency)
            if target_id is None:
                if _is_declared_workstream_dependency(state, dependency):
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


def _graph_from_build_state(state: _InvariantGraphBuildState) -> InvariantGraph:
    return InvariantGraph(
        root=str(state.root),
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
        ranking_signals=tuple(
            _sorted(
                state.ranking_signals,
                key=lambda item: (
                    item.code,
                    item.touchpoint_object_id,
                    item.touchsite_object_id,
                    item.node_id,
                    item.signal_id,
                ),
            )
        ),
    )


def _build_base_invariant_graph(
    root: Path,
    *,
    declared_registries: tuple[WorkstreamRegistry, ...] | None = None,
) -> InvariantGraph:
    root = root.resolve()
    state = _new_build_state(root)
    marker_nodes = scan_invariant_markers(root)
    marker_node_id_by_marker_id: dict[str, str] = {}
    for marker_node in marker_nodes:
        graph_node = _node_from_scan_entry(marker_node)
        _add_node(state, graph_node)
        marker_node_id_by_marker_id[graph_node.marker_id] = graph_node.node_id
        _link_node_refs(state, graph_node)
    registries = (
        _iter_declared_workstream_registries()
        if declared_registries is None
        else declared_registries
    )
    for registry in registries:
        _enrich_workstream_registry(
            state,
            registry=registry,
            marker_node_id_by_marker_id=marker_node_id_by_marker_id,
        )
    _resolve_blocking_dependencies(state)
    _join_policy_signals(state)
    _join_test_coverage(state)
    _join_test_failures(state)
    _join_governance_convergence_sources(state)
    _join_control_loop_artifacts(state)
    return _graph_from_build_state(state)


def _planning_chart_item_node_id(item_id: str) -> str:
    return f"planning_chart_item:{stable_hash(item_id)}"


def _graph_with_planning_chart(
    graph: InvariantGraph,
    summary: PlanningChartSummary,
) -> InvariantGraph:
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    edges = list(graph.edges)
    edge_keys = {(edge.edge_kind, edge.source_id, edge.target_id) for edge in edges}

    def _add_node(node: InvariantGraphNode) -> None:
        nodes_by_id[node.node_id] = node

    def _add_edge(edge_kind: str, source_id: str, target_id: str) -> None:
        key = (edge_kind, source_id, target_id)
        if source_id == target_id or key in edge_keys:
            return
        edge_keys.add(key)
        edges.append(
            InvariantGraphEdge(
                edge_id=_edge_id(edge_kind, source_id, target_id),
                edge_kind=edge_kind,
                source_id=source_id,
                target_id=target_id,
            )
        )

    report_node_id = "planning_chart_report:artifact"
    phase_count_summary = ", ".join(
        f"{phase.phase_kind}={phase.item_count}" for phase in summary.phases
    )
    report_node = _synthetic_node(
        node_id=report_node_id,
        title="planning chart",
        ref_kind="planning_chart_report",
        value="artifacts/out/invariant_graph.json#planning_chart_summary",
        reasoning_summary=(
            f"planning chart items={summary.item_count}; {phase_count_summary}"
        ),
        reasoning_control="invariant_graph.planning_chart",
        rel_path="artifacts/out/invariant_graph.json",
        qualname="planning_chart_summary",
        node_kind="planning_chart_report",
        status_hint="populated" if summary.item_count > 0 else "empty",
    )
    _add_node(report_node)
    for phase in summary.phases:
        phase_node_id = f"planning_phase:{phase.phase_kind}"
        phase_node = _synthetic_node(
            node_id=phase_node_id,
            title=phase.phase_kind,
            ref_kind="planning_phase",
            value=phase.phase_kind,
            reasoning_summary=(
                f"{phase.phase_kind} items={phase.item_count} "
                f"selected={len(phase.selected_item_ids)}"
            ),
            reasoning_control="invariant_graph.planning_chart.phase",
            rel_path="artifacts/out/invariant_graph.json",
            qualname=f"planning_chart_summary.{phase.phase_kind}",
            node_kind="planning_phase",
            status_hint="active" if phase.item_count > 0 else "empty",
            phase_kind=phase.phase_kind,
        )
        _add_node(phase_node)
        _add_edge("contains", report_node_id, phase_node_id)
        for item in phase.items:
            item_node_id = _planning_chart_item_node_id(item.item_id)
            item_node = _synthetic_node(
                node_id=item_node_id,
                title=item.title,
                ref_kind="planning_chart_item",
                value=item.item_id,
                object_ids=tuple(
                    _sorted(list({item.item_id, *item.tracked_object_ids}))
                ),
                reasoning_summary=item.reasoning_summary,
                reasoning_control="invariant_graph.planning_chart.item",
                rel_path="artifacts/out/invariant_graph.json",
                qualname=item.item_id,
                node_kind="planning_chart_item",
                status_hint=item.status_hint,
                phase_kind=item.phase_kind,
                item_kind=item.item_kind,
                source_kind=item.source_kind,
                selection_rank=item.selection_rank,
                tracked_node_ids=item.tracked_node_ids,
                tracked_object_ids=item.tracked_object_ids,
            )
            _add_node(item_node)
            _add_edge("contains", phase_node_id, item_node_id)
            for tracked_node_id in item.tracked_node_ids:
                if tracked_node_id in nodes_by_id:
                    _add_edge("tracks", item_node_id, tracked_node_id)
            for tracked_object_id in item.tracked_object_ids:
                ref_node_id = _synthetic_ref_node_id("object_id", tracked_object_id)
                if ref_node_id in nodes_by_id:
                    _add_edge("tracks", item_node_id, ref_node_id)

    return replace(
        graph,
        nodes=tuple(
            _sorted(
                list(nodes_by_id.values()),
                key=lambda item: (item.node_kind, item.rel_path, item.line, item.node_id),
            )
        ),
        edges=tuple(
            _sorted(
                edges,
                key=lambda item: (item.edge_kind, item.source_id, item.target_id),
            )
        ),
        planning_chart_summary=summary,
    )


@dataclass(frozen=True)
class InvariantPlanningBundle:
    graph: InvariantGraph
    workstreams: InvariantWorkstreamsProjection


def build_invariant_planning_bundle(
    root: Path,
    *,
    declared_registries: tuple[WorkstreamRegistry, ...] | None = None,
    planning_chart_rules: tuple[PlanningChartRule, ...] | None = None,
) -> InvariantPlanningBundle:
    base_graph = _build_base_invariant_graph(
        root,
        declared_registries=declared_registries,
    )
    base_workstreams = _build_invariant_workstreams_projection(
        base_graph,
        root=root,
    )
    planning_chart_summary = build_planning_chart_summary(
        graph=base_graph,
        workstreams=base_workstreams,
        rules=planning_chart_rules,
    )
    graph = _graph_with_planning_chart(base_graph, planning_chart_summary)
    workstreams = replace(
        base_workstreams,
        node_lookup=graph.node_by_id(),
        planning_chart_summary=planning_chart_summary,
    )
    return InvariantPlanningBundle(graph=graph, workstreams=workstreams)


def build_invariant_graph(
    root: Path,
    *,
    declared_registries: tuple[WorkstreamRegistry, ...] | None = None,
    planning_chart_rules: tuple[PlanningChartRule, ...] | None = None,
) -> InvariantGraph:
    return build_invariant_planning_bundle(
        root,
        declared_registries=declared_registries,
        planning_chart_rules=planning_chart_rules,
    ).graph


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


def _build_invariant_workstreams_projection(
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
    ranking_signals_by_node_id: dict[str, tuple[InvariantGraphRankingSignal, ...]] = {}
    for signal in graph.ranking_signals:
        ranking_signals_by_node_id.setdefault(signal.node_id, tuple())
    if graph.ranking_signals:
        grouped_ranking_signals: defaultdict[str, list[InvariantGraphRankingSignal]] = (
            defaultdict(list)
        )
        for signal in graph.ranking_signals:
            grouped_ranking_signals[signal.node_id].append(signal)
        ranking_signals_by_node_id = {
            node_id: tuple(items) for node_id, items in grouped_ranking_signals.items()
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

    def _failing_test_case_ids(node_ids: Iterable[str]) -> tuple[str, ...]:
        failing_test_case_ids = {
            test_case_id
            for test_case_id in _test_case_ids(node_ids)
            if any(
                edge.edge_kind == "fails_with"
                and node_by_id.get(edge.target_id, None) is not None
                and node_by_id[edge.target_id].node_kind == "test_failure"
                for edge in edges_from.get(test_case_id, ())
            )
        }
        return tuple(_sorted(list(failing_test_case_ids)))

    def _test_failure_ids(node_ids: Iterable[str]) -> tuple[str, ...]:
        direct_failure_ids = {
            edge.source_id
            for node_id in node_ids
            for edge in edges_to.get(node_id, ())
            if edge.edge_kind == "fails_on"
            and node_by_id.get(edge.source_id, None) is not None
            and node_by_id[edge.source_id].node_kind == "test_failure"
        }
        covered_failure_ids = {
            edge.target_id
            for test_case_id in _failing_test_case_ids(node_ids)
            for edge in edges_from.get(test_case_id, ())
            if edge.edge_kind == "fails_with"
            and node_by_id.get(edge.target_id, None) is not None
            and node_by_id[edge.target_id].node_kind == "test_failure"
        }
        return tuple(_sorted(list(direct_failure_ids | covered_failure_ids)))

    def _diagnostic_count(node_ids: Iterable[str]) -> int:
        return sum(len(diagnostics_by_node_id.get(node_id, ())) for node_id in node_ids)

    def _ranking_signal_count(node_ids: Iterable[str]) -> int:
        return sum(len(ranking_signals_by_node_id.get(node_id, ())) for node_id in node_ids)

    def _ranking_signal_score(node_ids: Iterable[str]) -> int:
        return sum(
            signal.score
            for node_id in node_ids
            for signal in ranking_signals_by_node_id.get(node_id, ())
        )

    def _counterfactual_action_nodes(
        node_ids: Iterable[str],
    ) -> tuple[InvariantGraphNode, ...]:
        return tuple(
            node_by_id[node_id]
            for node_id in node_ids
            if node_id in node_by_id and node_by_id[node_id].node_kind == "counterfactual_action"
        )

    def _counterfactual_action_count(node_ids: Iterable[str]) -> int:
        return len(_counterfactual_action_nodes(node_ids))

    def _viable_counterfactual_action_count(node_ids: Iterable[str]) -> int:
        return sum(
            1
            for node in _counterfactual_action_nodes(node_ids)
            if node.status_hint not in _COUNTERFACTUAL_BLOCKED_READINESS_CLASSES
        )

    def _blocked_counterfactual_action_count(node_ids: Iterable[str]) -> int:
        return sum(
            1
            for node in _counterfactual_action_nodes(node_ids)
            if node.status_hint in _COUNTERFACTUAL_BLOCKED_READINESS_CLASSES
        )

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
            failing_test_case_count=len(_failing_test_case_ids((node.node_id,))),
            test_failure_count=len(_test_failure_ids((node.node_id,))),
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
            ranking_signal_count=_ranking_signal_count(touchpoint_descendants),
            ranking_signal_score=_ranking_signal_score(touchpoint_descendants),
            failing_test_case_count=len(_failing_test_case_ids(touchpoint_descendants)),
            test_failure_count=len(_test_failure_ids(touchpoint_descendants)),
            counterfactual_action_count=_counterfactual_action_count(
                touchpoint_descendants
            ),
            viable_counterfactual_action_count=_viable_counterfactual_action_count(
                touchpoint_descendants
            ),
            blocked_counterfactual_action_count=_blocked_counterfactual_action_count(
                touchpoint_descendants
            ),
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
            ranking_signal_count=_ranking_signal_count(subqueue_descendants),
            ranking_signal_score=_ranking_signal_score(subqueue_descendants),
            failing_test_case_count=len(_failing_test_case_ids(subqueue_descendants)),
            test_failure_count=len(_test_failure_ids(subqueue_descendants)),
            counterfactual_action_count=_counterfactual_action_count(
                subqueue_descendants
            ),
            viable_counterfactual_action_count=_viable_counterfactual_action_count(
                subqueue_descendants
            ),
            blocked_counterfactual_action_count=_blocked_counterfactual_action_count(
                subqueue_descendants
            ),
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
            ranking_signal_count=_ranking_signal_count(root_descendants),
            ranking_signal_score=_ranking_signal_score(root_descendants),
            doc_alignment_summary=None,
            subqueues=_stream_from_iterable(_iter_subqueues),
            touchpoints=_stream_from_iterable(_iter_touchpoints),
            failing_test_case_count=len(_failing_test_case_ids(root_descendants)),
            test_failure_count=len(_test_failure_ids(root_descendants)),
            counterfactual_action_count=_counterfactual_action_count(root_descendants),
            viable_counterfactual_action_count=_viable_counterfactual_action_count(
                root_descendants
            ),
            blocked_counterfactual_action_count=_blocked_counterfactual_action_count(
                root_descendants
            ),
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
        planning_chart_summary=graph.planning_chart_summary,
        node_lookup=node_by_id,
        workstreams=_stream_from_iterable(
            lambda: (
                _workstream_projection(root_object_id)
                for root_object_id in _sorted(list(graph.workstream_root_ids))
            )
        ),
    )


def build_invariant_workstreams(
    graph: InvariantGraph,
    *,
    root: Path | None = None,
    planning_chart_rules: tuple[PlanningChartRule, ...] | None = None,
) -> InvariantWorkstreamsProjection:
    projection = _build_invariant_workstreams_projection(graph, root=root)
    if projection.planning_chart_summary is not None:
        return projection
    planning_chart_summary = build_planning_chart_summary(
        graph=graph,
        workstreams=projection,
        rules=planning_chart_rules,
    )
    return replace(projection, planning_chart_summary=planning_chart_summary)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(workstreams.as_payload(), indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


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
    "InvariantPlanningBundle",
    "InvariantWorkstreamDrift",
    "blocker_chains",
    "build_invariant_graph",
    "build_invariant_planning_bundle",
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
