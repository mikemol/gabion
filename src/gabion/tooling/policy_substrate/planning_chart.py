from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, cast

from gabion.order_contract import ordered_or_sorted
from gabion.tooling.policy_substrate.planning_chart_identity import (
    build_planning_chart_identity_grammar,
)

if TYPE_CHECKING:
    from gabion.tooling.policy_substrate.invariant_graph import (
        InvariantGraph,
        InvariantGraphNode,
        InvariantWorkstreamsProjection,
    )


def _sorted[T](values: list[T], *, key=None) -> list[T]:
    return ordered_or_sorted(
        values,
        source="gabion.tooling.policy_substrate.planning_chart",
        key=key,
    )


class PlanningPhaseKind(StrEnum):
    SCAN = "scan"
    PREDICT = "predict"
    COMPLETE = "complete"


@dataclass(frozen=True)
class PlanningChartRule:
    rule_id: str
    phase_kind: PlanningPhaseKind
    item_kind: str
    source_kind: str
    target_kind: str
    node_kinds: tuple[str, ...] = ()
    node_kind_prefixes: tuple[str, ...] = ()
    edge_kind: str = ""
    source_node_kinds: tuple[str, ...] = ()
    source_node_kind_prefixes: tuple[str, ...] = ()
    target_node_kinds: tuple[str, ...] = ()
    target_node_kind_prefixes: tuple[str, ...] = ()
    selector_name: str = ""
    selection_rank: int = 0
    selected: bool = False


@dataclass(frozen=True)
class PlanningChartItem:
    item_id: str
    phase_kind: str
    item_kind: str
    source_kind: str
    title: str
    status_hint: str
    selection_rank: int
    tracked_node_ids: tuple[str, ...] = ()
    tracked_object_ids: tuple[str, ...] = ()
    rule_id: str = ""
    reasoning_summary: str = ""
    selected: bool = False

    def as_payload(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "phase_kind": self.phase_kind,
            "item_kind": self.item_kind,
            "source_kind": self.source_kind,
            "title": self.title,
            "status_hint": self.status_hint,
            "selection_rank": self.selection_rank,
            "tracked_node_ids": list(self.tracked_node_ids),
            "tracked_object_ids": list(self.tracked_object_ids),
            "rule_id": self.rule_id,
            "reasoning_summary": self.reasoning_summary,
            "selected": self.selected,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PlanningChartItem:
        return cls(
            item_id=str(payload.get("item_id", "")),
            phase_kind=str(payload.get("phase_kind", "")),
            item_kind=str(payload.get("item_kind", "")),
            source_kind=str(payload.get("source_kind", "")),
            title=str(payload.get("title", "")),
            status_hint=str(payload.get("status_hint", "")),
            selection_rank=int(payload.get("selection_rank", 0) or 0),
            tracked_node_ids=tuple(
                str(value) for value in payload.get("tracked_node_ids", [])
            ),
            tracked_object_ids=tuple(
                str(value) for value in payload.get("tracked_object_ids", [])
            ),
            rule_id=str(payload.get("rule_id", "")),
            reasoning_summary=str(payload.get("reasoning_summary", "")),
            selected=bool(payload.get("selected", False)),
        )


@dataclass(frozen=True)
class PlanningPhaseSummary:
    phase_kind: str
    item_count: int
    status_counts: Mapping[str, int]
    blocker_counts: Mapping[str, int]
    selected_item_ids: tuple[str, ...]
    items: tuple[PlanningChartItem, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "phase_kind": self.phase_kind,
            "item_count": self.item_count,
            "status_counts": dict(self.status_counts),
            "blocker_counts": dict(self.blocker_counts),
            "selected_item_ids": list(self.selected_item_ids),
            "items": [item.as_payload() for item in self.items],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PlanningPhaseSummary:
        return cls(
            phase_kind=str(payload.get("phase_kind", "")),
            item_count=int(payload.get("item_count", 0) or 0),
            status_counts={
                str(key): int(value or 0)
                for key, value in cast(
                    Mapping[str, object],
                    payload.get("status_counts", {}),
                ).items()
            },
            blocker_counts={
                str(key): int(value or 0)
                for key, value in cast(
                    Mapping[str, object],
                    payload.get("blocker_counts", {}),
                ).items()
            },
            selected_item_ids=tuple(
                str(value) for value in payload.get("selected_item_ids", [])
            ),
            items=tuple(
                PlanningChartItem.from_payload(item)
                for item in cast(Iterable[Mapping[str, object]], payload.get("items", []))
            ),
        )


@dataclass(frozen=True)
class PlanningChartSummary:
    item_count: int
    selected_completion_item_ids: tuple[str, ...]
    phases: tuple[PlanningPhaseSummary, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "item_count": self.item_count,
            "selected_completion_item_ids": list(self.selected_completion_item_ids),
            "phases": [phase.as_payload() for phase in self.phases],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PlanningChartSummary:
        return cls(
            item_count=int(payload.get("item_count", 0) or 0),
            selected_completion_item_ids=tuple(
                str(value)
                for value in payload.get("selected_completion_item_ids", [])
            ),
            phases=tuple(
                PlanningPhaseSummary.from_payload(item)
                for item in cast(Iterable[Mapping[str, object]], payload.get("phases", []))
            ),
        )


_BLOCKER_STATUS_HINTS = frozenset(
    {
        "ready_structural",
        "coverage_gap",
        "counterfactual_blocked",
        "policy_blocked",
        "diagnostic_blocked",
    }
)


def _node_kind_matches(rule: PlanningChartRule, node_kind: str) -> bool:
    if rule.node_kinds and node_kind in rule.node_kinds:
        return True
    return any(node_kind.startswith(prefix) for prefix in rule.node_kind_prefixes)


def _edge_node_kind_matches(
    *,
    node_kind: str,
    allowed_kinds: tuple[str, ...],
    allowed_prefixes: tuple[str, ...],
) -> bool:
    if allowed_kinds and node_kind in allowed_kinds:
        return True
    if allowed_prefixes and any(node_kind.startswith(prefix) for prefix in allowed_prefixes):
        return True
    return not allowed_kinds and not allowed_prefixes


def _tracked_node_id_for_object_id(graph: InvariantGraph, object_id: str) -> str:
    if not object_id:
        return ""
    node_id = f"object_id:{object_id}"
    if node_id in graph.node_by_id():
        return node_id
    return ""


def _default_rules() -> tuple[PlanningChartRule, ...]:
    return (
        PlanningChartRule(
            rule_id="planning_chart.scan.touchsites",
            phase_kind=PlanningPhaseKind.SCAN,
            item_kind="planning_surface",
            source_kind="declared_touchsite",
            target_kind="node",
            node_kinds=("synthetic_touchsite",),
        ),
        PlanningChartRule(
            rule_id="planning_chart.scan.control_loop",
            phase_kind=PlanningPhaseKind.SCAN,
            item_kind="control_loop_evidence",
            source_kind="control_loop_artifact",
            target_kind="node",
            node_kind_prefixes=(
                "docflow_",
                "controller_drift",
                "local_repro_closure",
                "local_ci_repro",
                "cross_origin_witness",
                "git_state",
                "ingress_merge_parity",
            ),
        ),
        PlanningChartRule(
            rule_id="planning_chart.scan.kernel_vm",
            phase_kind=PlanningPhaseKind.SCAN,
            item_kind="kernel_alignment_evidence",
            source_kind="kernel_vm_alignment",
            target_kind="node",
            node_kind_prefixes=("kernel_vm_alignment_",),
        ),
        PlanningChartRule(
            rule_id="planning_chart.predict.dependencies",
            phase_kind=PlanningPhaseKind.PREDICT,
            item_kind="blocking_dependency",
            source_kind="declared_dependency",
            target_kind="edge",
            edge_kind="depends_on",
        ),
        PlanningChartRule(
            rule_id="planning_chart.predict.counterfactual",
            phase_kind=PlanningPhaseKind.PREDICT,
            item_kind="counterfactual_action",
            source_kind="declared_counterfactual",
            target_kind="node",
            node_kinds=("counterfactual_action",),
        ),
        PlanningChartRule(
            rule_id="planning_chart.complete.workstream_frontier",
            phase_kind=PlanningPhaseKind.COMPLETE,
            item_kind="workstream_recommendation",
            source_kind="recommended_cut",
            target_kind="workstream_recommendation",
            selector_name="recommended_cut",
            selection_rank=0,
            selected=True,
        ),
        PlanningChartRule(
            rule_id="planning_chart.complete.workstream_ready",
            phase_kind=PlanningPhaseKind.COMPLETE,
            item_kind="workstream_recommendation",
            source_kind="recommended_ready_cut",
            target_kind="workstream_recommendation",
            selector_name="recommended_ready_cut",
            selection_rank=10,
        ),
        PlanningChartRule(
            rule_id="planning_chart.complete.workstream_coverage",
            phase_kind=PlanningPhaseKind.COMPLETE,
            item_kind="workstream_recommendation",
            source_kind="recommended_coverage_gap_cut",
            target_kind="workstream_recommendation",
            selector_name="recommended_coverage_gap_cut",
            selection_rank=20,
        ),
        PlanningChartRule(
            rule_id="planning_chart.complete.workstream_counterfactual",
            phase_kind=PlanningPhaseKind.COMPLETE,
            item_kind="workstream_recommendation",
            source_kind="recommended_counterfactual_blocked_cut",
            target_kind="workstream_recommendation",
            selector_name="recommended_counterfactual_blocked_cut",
            selection_rank=30,
        ),
        PlanningChartRule(
            rule_id="planning_chart.complete.workstream_policy",
            phase_kind=PlanningPhaseKind.COMPLETE,
            item_kind="workstream_recommendation",
            source_kind="recommended_policy_blocked_cut",
            target_kind="workstream_recommendation",
            selector_name="recommended_policy_blocked_cut",
            selection_rank=40,
        ),
        PlanningChartRule(
            rule_id="planning_chart.complete.workstream_diagnostic",
            phase_kind=PlanningPhaseKind.COMPLETE,
            item_kind="workstream_recommendation",
            source_kind="recommended_diagnostic_blocked_cut",
            target_kind="workstream_recommendation",
            selector_name="recommended_diagnostic_blocked_cut",
            selection_rank=50,
        ),
        PlanningChartRule(
            rule_id="planning_chart.complete.repo_followup",
            phase_kind=PlanningPhaseKind.COMPLETE,
            item_kind="repo_recommendation",
            source_kind="recommended_followup",
            target_kind="repo_followup",
            selector_name="recommended_repo_followup",
            selection_rank=0,
            selected=True,
        ),
        PlanningChartRule(
            rule_id="planning_chart.complete.repo_code_followup",
            phase_kind=PlanningPhaseKind.COMPLETE,
            item_kind="repo_recommendation",
            source_kind="recommended_code_followup",
            target_kind="repo_followup",
            selector_name="recommended_repo_code_followup",
            selection_rank=1,
            selected=True,
        ),
        PlanningChartRule(
            rule_id="planning_chart.complete.repo_human_followup",
            phase_kind=PlanningPhaseKind.COMPLETE,
            item_kind="repo_recommendation",
            source_kind="recommended_human_followup",
            target_kind="repo_followup",
            selector_name="recommended_repo_human_followup",
            selection_rank=2,
            selected=True,
        ),
        PlanningChartRule(
            rule_id="planning_chart.complete.repo_followup_lane",
            phase_kind=PlanningPhaseKind.COMPLETE,
            item_kind="repo_recommendation_lane",
            source_kind="recommended_followup_lane",
            target_kind="repo_followup_lane",
            selector_name="recommended_repo_followup_lane",
            selection_rank=10,
            selected=True,
        ),
        PlanningChartRule(
            rule_id="planning_chart.complete.repo_code_followup_lane",
            phase_kind=PlanningPhaseKind.COMPLETE,
            item_kind="repo_recommendation_lane",
            source_kind="recommended_code_followup_lane",
            target_kind="repo_followup_lane",
            selector_name="recommended_repo_code_followup_lane",
            selection_rank=11,
            selected=True,
        ),
        PlanningChartRule(
            rule_id="planning_chart.complete.repo_human_followup_lane",
            phase_kind=PlanningPhaseKind.COMPLETE,
            item_kind="repo_recommendation_lane",
            source_kind="recommended_human_followup_lane",
            target_kind="repo_followup_lane",
            selector_name="recommended_repo_human_followup_lane",
            selection_rank=12,
            selected=True,
        ),
    )


def default_planning_chart_rules() -> tuple[PlanningChartRule, ...]:
    return _default_rules()


def build_planning_chart_summary(
    *,
    graph: InvariantGraph,
    workstreams: InvariantWorkstreamsProjection,
    rules: tuple[PlanningChartRule, ...] | None = None,
) -> PlanningChartSummary:
    active_rules = _default_rules() if rules is None else rules
    node_by_id = graph.node_by_id()
    edges = tuple(
        _sorted(
            list(graph.edges),
            key=lambda item: (item.edge_kind, item.source_id, item.target_id),
        )
    )
    items: list[PlanningChartItem] = []

    def _make_node_item(
        *,
        rule: PlanningChartRule,
        node: InvariantGraphNode,
        ordinal: int,
    ) -> PlanningChartItem:
        return PlanningChartItem(
            item_id=f"{rule.rule_id}:{node.node_id}",
            phase_kind=rule.phase_kind.value,
            item_kind=rule.item_kind,
            source_kind=rule.source_kind,
            title=node.title,
            status_hint=node.status_hint,
            selection_rank=rule.selection_rank + ordinal,
            tracked_node_ids=(node.node_id,),
            tracked_object_ids=node.object_ids,
            rule_id=rule.rule_id,
            reasoning_summary=node.reasoning_summary,
            selected=rule.selected,
        )

    def _make_edge_item(
        *,
        rule: PlanningChartRule,
        edge,
        source_node: InvariantGraphNode,
        target_node: InvariantGraphNode,
        ordinal: int,
    ) -> PlanningChartItem:
        tracked_object_ids = tuple(
            _sorted(
                list(
                    {
                        *source_node.object_ids,
                        *target_node.object_ids,
                    }
                )
            )
        )
        return PlanningChartItem(
            item_id=f"{rule.rule_id}:{edge.source_id}:{edge.target_id}",
            phase_kind=rule.phase_kind.value,
            item_kind=rule.item_kind,
            source_kind=rule.source_kind,
            title=f"{source_node.title} depends on {target_node.title}",
            status_hint=source_node.status_hint or target_node.status_hint,
            selection_rank=rule.selection_rank + ordinal,
            tracked_node_ids=(edge.source_id, edge.target_id),
            tracked_object_ids=tracked_object_ids,
            rule_id=rule.rule_id,
            reasoning_summary=(
                f"{source_node.title} declares dependency on {target_node.title}."
            ),
            selected=rule.selected,
        )

    def _make_workstream_completion_item(
        *,
        rule: PlanningChartRule,
        workstream,
        candidate,
        ordinal: int,
    ) -> PlanningChartItem:
        tracked_object_ids = tuple(
            item
            for item in (
                workstream.object_id.wire(),
                candidate.object_id.wire(),
                candidate.owner_object_id.wire(),
            )
            if item
        )
        tracked_node_ids = tuple(
            item
            for item in (
                _tracked_node_id_for_object_id(graph, workstream.object_id.wire()),
                _tracked_node_id_for_object_id(graph, candidate.object_id.wire()),
                _tracked_node_id_for_object_id(graph, candidate.owner_object_id.wire()),
            )
            if item
        )
        return PlanningChartItem(
            item_id=f"{rule.rule_id}:{workstream.object_id.wire()}",
            phase_kind=rule.phase_kind.value,
            item_kind=rule.item_kind,
            source_kind=rule.source_kind,
            title=f"{workstream.object_id.wire()} -> {candidate.title}",
            status_hint=candidate.readiness_class,
            selection_rank=rule.selection_rank + ordinal,
            tracked_node_ids=tracked_node_ids,
            tracked_object_ids=tracked_object_ids,
            rule_id=rule.rule_id,
            reasoning_summary=(
                f"{workstream.object_id.wire()} emits {rule.source_kind} at "
                f"{candidate.object_id.wire()}."
            ),
            selected=rule.selected,
        )

    def _make_repo_completion_item(
        *,
        rule: PlanningChartRule,
        followup,
    ) -> PlanningChartItem:
        tracked_object_ids = tuple(
            item
            for item in (
                getattr(followup, "object_id", None),
                getattr(followup, "owner_object_id", None),
                getattr(followup, "owner_root_object_id", None),
            )
            if item
        )
        tracked_node_ids = tuple(
            item
            for item in (
                *(
                    _tracked_node_id_for_object_id(graph, object_id)
                    for object_id in tracked_object_ids
                ),
            )
            if item
        )
        title = getattr(followup, "title", "") or getattr(
            followup,
            "followup_family",
            "",
        )
        status_hint = getattr(
            followup,
            "blocker_class",
            "",
        ) or getattr(followup, "followup_class", "")
        item_key = (
            getattr(followup, "object_id", None)
            or getattr(followup, "followup_family", "")
            or title
        )
        return PlanningChartItem(
            item_id=f"{rule.rule_id}:{item_key}",
            phase_kind=rule.phase_kind.value,
            item_kind=rule.item_kind,
            source_kind=rule.source_kind,
            title=title,
            status_hint=status_hint,
            selection_rank=rule.selection_rank,
            tracked_node_ids=tracked_node_ids,
            tracked_object_ids=tracked_object_ids,
            rule_id=rule.rule_id,
            reasoning_summary=f"repo {rule.source_kind} emits {title}.",
            selected=rule.selected,
        )

    for rule in active_rules:
        if rule.target_kind == "node":
            matched_nodes = [
                node
                for node in graph.nodes
                if _node_kind_matches(rule, node.node_kind)
            ]
            for ordinal, node in enumerate(
                _sorted(
                    matched_nodes,
                    key=lambda item: (item.node_kind, item.rel_path, item.line, item.node_id),
                )
            ):
                items.append(_make_node_item(rule=rule, node=node, ordinal=ordinal))
            continue
        if rule.target_kind == "edge":
            matched_edges = []
            for edge in edges:
                if edge.edge_kind != rule.edge_kind:
                    continue
                source_node = node_by_id.get(edge.source_id)
                target_node = node_by_id.get(edge.target_id)
                if source_node is None or target_node is None:
                    continue
                if not _edge_node_kind_matches(
                    node_kind=source_node.node_kind,
                    allowed_kinds=rule.source_node_kinds,
                    allowed_prefixes=rule.source_node_kind_prefixes,
                ):
                    continue
                if not _edge_node_kind_matches(
                    node_kind=target_node.node_kind,
                    allowed_kinds=rule.target_node_kinds,
                    allowed_prefixes=rule.target_node_kind_prefixes,
                ):
                    continue
                matched_edges.append((edge, source_node, target_node))
            for ordinal, (edge, source_node, target_node) in enumerate(matched_edges):
                items.append(
                    _make_edge_item(
                        rule=rule,
                        edge=edge,
                        source_node=source_node,
                        target_node=target_node,
                        ordinal=ordinal,
                    )
                )
            continue
        if rule.target_kind == "workstream_recommendation":
            for ordinal, workstream in enumerate(
                _sorted(
                    list(workstreams.iter_workstreams()),
                    key=lambda item: item.object_id.wire(),
                )
            ):
                selector = getattr(workstream, rule.selector_name, None)
                if selector is None or not callable(selector):
                    continue
                candidate = selector()
                if candidate is None:
                    continue
                items.append(
                    _make_workstream_completion_item(
                        rule=rule,
                        workstream=workstream,
                        candidate=candidate,
                        ordinal=ordinal,
                    )
                )
            continue
        if rule.target_kind in {"repo_followup", "repo_followup_lane"}:
            selector = getattr(workstreams, rule.selector_name, None)
            if selector is None or not callable(selector):
                continue
            followup = selector()
            if followup is None:
                continue
            items.append(_make_repo_completion_item(rule=rule, followup=followup))

    grouped_by_phase: defaultdict[str, list[PlanningChartItem]] = defaultdict(list)
    for item in items:
        grouped_by_phase[item.phase_kind].append(item)

    phases: list[PlanningPhaseSummary] = []
    selected_completion_item_ids: list[str] = []
    for phase_kind in (
        PlanningPhaseKind.SCAN.value,
        PlanningPhaseKind.PREDICT.value,
        PlanningPhaseKind.COMPLETE.value,
    ):
        phase_items = tuple(
            _sorted(
                grouped_by_phase.get(phase_kind, []),
                key=lambda item: (
                    item.selection_rank,
                    item.item_kind,
                    item.source_kind,
                    item.title,
                    item.item_id,
                ),
            )
        )
        status_counts: defaultdict[str, int] = defaultdict(int)
        blocker_counts: defaultdict[str, int] = defaultdict(int)
        phase_selected_item_ids: list[str] = []
        for item in phase_items:
            status_key = item.status_hint or "unspecified"
            status_counts[status_key] += 1
            if item.status_hint in _BLOCKER_STATUS_HINTS:
                blocker_counts[item.status_hint] += 1
            if item.selected:
                phase_selected_item_ids.append(item.item_id)
                if phase_kind == PlanningPhaseKind.COMPLETE.value:
                    selected_completion_item_ids.append(item.item_id)
        phases.append(
            PlanningPhaseSummary(
                phase_kind=phase_kind,
                item_count=len(phase_items),
                status_counts=dict(
                    _sorted(list(status_counts.items()), key=lambda item: item[0])
                ),
                blocker_counts=dict(
                    _sorted(list(blocker_counts.items()), key=lambda item: item[0])
                ),
                selected_item_ids=tuple(phase_selected_item_ids),
                items=phase_items,
            )
        )

    return PlanningChartSummary(
        item_count=len(items),
        selected_completion_item_ids=tuple(selected_completion_item_ids),
        phases=tuple(phases),
    )


__all__ = [
    "PlanningChartItem",
    "PlanningChartRule",
    "PlanningChartSummary",
    "PlanningPhaseKind",
    "PlanningPhaseSummary",
    "build_planning_chart_identity_grammar",
    "build_planning_chart_summary",
    "default_planning_chart_rules",
]
