from __future__ import annotations

from dataclasses import dataclass

from gabion.tooling.policy_substrate.identity_zone import (
    IdentityAtom,
    IdentityCarrier,
    IdentityZoneName,
)
from gabion.tooling.policy_substrate.invariant_graph import (
    InvariantGraph,
    InvariantGraphDiagnostic,
    InvariantGraphEdge,
    InvariantGraphNode,
)
from gabion.tooling.policy_substrate.planning_chart import (
    PlanningChartRule,
    PlanningPhaseKind,
    PlanningChartSummary,
    PlanningPhaseSummary,
    PlanningChartItem,
    build_planning_chart_identity_grammar,
    build_planning_chart_summary,
)


@dataclass(frozen=True)
class _FakeId:
    value: str

    def wire(self) -> str:
        return self.value


@dataclass(frozen=True)
class _FakeCut:
    object_id: _FakeId
    owner_object_id: _FakeId
    title: str
    readiness_class: str


@dataclass(frozen=True)
class _FakeFollowup:
    followup_family: str
    title: str
    object_id: str
    owner_object_id: str
    owner_root_object_id: str
    blocker_class: str
    queue_id: str = ""


@dataclass(frozen=True)
class _FakeWorkstream:
    object_id: _FakeId
    cut: _FakeCut

    def recommended_cut(self) -> _FakeCut:
        return self.cut


@dataclass(frozen=True)
class _FakeWorkstreams:
    workstreams: tuple[_FakeWorkstream, ...]
    repo_followup: _FakeFollowup

    def iter_workstreams(self):
        return iter(self.workstreams)

    def recommended_repo_followup(self) -> _FakeFollowup:
        return self.repo_followup


def _node(
    node_id: str,
    node_kind: str,
    *,
    title: str,
    status_hint: str = "",
    object_ids: tuple[str, ...] = (),
    reasoning_summary: str = "",
) -> InvariantGraphNode:
    return InvariantGraphNode(
        node_id=node_id,
        node_kind=node_kind,
        title=title,
        marker_name="",
        marker_kind="",
        marker_id="",
        site_identity=f"site:{node_id}",
        structural_identity=f"struct:{node_id}",
        object_ids=object_ids,
        doc_ids=(),
        policy_ids=(),
        invariant_ids=(),
        reasoning_summary=reasoning_summary,
        reasoning_control="test",
        blocking_dependencies=(),
        rel_path="src/gabion/test_surface.py",
        qualname=node_id.replace(":", "."),
        line=1,
        column=1,
        ast_node_kind="synthetic",
        seam_class="",
        source_marker_node_id="",
        status_hint=status_hint,
    )


def test_build_planning_chart_summary_supports_injected_rules() -> None:
    graph = InvariantGraph(
        root="/repo",
        workstream_root_ids=("WS-001",),
        nodes=(
            _node(
                "touchsite:1",
                "synthetic_touchsite",
                title="semantic_fragment.reflect_projection_fiber_witness",
                object_ids=("WS-001-TP-001",),
                reasoning_summary="scan touchsite",
            ),
            _node(
                "kernel_vm_alignment_residue:1",
                "kernel_vm_alignment_residue",
                title="missing runtime object image",
                status_hint="missing_runtime_object_image",
                object_ids=("CSA-RGC-TP-008",),
                reasoning_summary="kernel residue",
            ),
            _node(
                "counterfactual_action:1",
                "counterfactual_action",
                title="Retire canonical value materialization",
                status_hint="policy_blocked",
                object_ids=("PSF-007-TP-001-ACT-001",),
                reasoning_summary="counterfactual action",
            ),
            _node(
                "work_item:source",
                "synthetic_work_item",
                title="source work item",
                object_ids=("WS-001-TP-001",),
            ),
            _node(
                "object_id:WS-001",
                "object_id",
                title="WS-001",
                object_ids=("WS-001",),
            ),
            _node(
                "object_id:WS-001-TP-001",
                "object_id",
                title="WS-001-TP-001",
                object_ids=("WS-001-TP-001",),
            ),
        ),
        edges=(
            InvariantGraphEdge(
                edge_id="depends_on:1",
                edge_kind="depends_on",
                source_id="work_item:source",
                target_id="object_id:WS-001-TP-001",
            ),
        ),
        diagnostics=(),
        ranking_signals=(),
    )
    workstreams = _FakeWorkstreams(
        workstreams=(
            _FakeWorkstream(
                object_id=_FakeId("WS-001"),
                cut=_FakeCut(
                    object_id=_FakeId("WS-001-TP-001"),
                    owner_object_id=_FakeId("WS-001"),
                    title="coverage gap cut",
                    readiness_class="coverage_gap",
                ),
            ),
        ),
        repo_followup=_FakeFollowup(
            followup_family="structural_cut",
            title="repo frontier",
            object_id="WS-001-TP-001",
            owner_object_id="WS-001",
            owner_root_object_id="WS-001",
            blocker_class="coverage_gap",
            queue_id=(
                "planner_queue|followup_family=structural_cut|followup_class=code|"
                "selection_scope_kind=singleton|selection_scope_id=|root_object_ids=WS-001"
            ),
        ),
    )
    rules = (
        PlanningChartRule(
            rule_id="scan.touchsite",
            phase_kind=PlanningPhaseKind.SCAN,
            item_kind="surface",
            source_kind="declared_touchsite",
            target_kind="node",
            node_kinds=("synthetic_touchsite",),
        ),
        PlanningChartRule(
            rule_id="scan.kernel",
            phase_kind=PlanningPhaseKind.SCAN,
            item_kind="kernel",
            source_kind="kernel_vm_alignment",
            target_kind="node",
            node_kinds=("kernel_vm_alignment_residue",),
        ),
        PlanningChartRule(
            rule_id="predict.counterfactual",
            phase_kind=PlanningPhaseKind.PREDICT,
            item_kind="counterfactual",
            source_kind="declared_counterfactual",
            target_kind="node",
            node_kinds=("counterfactual_action",),
        ),
        PlanningChartRule(
            rule_id="predict.dependency",
            phase_kind=PlanningPhaseKind.PREDICT,
            item_kind="dependency",
            source_kind="declared_dependency",
            target_kind="edge",
            edge_kind="depends_on",
        ),
        PlanningChartRule(
            rule_id="complete.workstream",
            phase_kind=PlanningPhaseKind.COMPLETE,
            item_kind="workstream_recommendation",
            source_kind="recommended_cut",
            target_kind="workstream_recommendation",
            selector_name="recommended_cut",
            selected=True,
        ),
        PlanningChartRule(
            rule_id="complete.repo",
            phase_kind=PlanningPhaseKind.COMPLETE,
            item_kind="repo_recommendation",
            source_kind="recommended_followup",
            target_kind="repo_followup",
            selector_name="recommended_repo_followup",
            selected=True,
            selection_rank=1,
        ),
    )

    summary = build_planning_chart_summary(
        graph=graph,
        workstreams=workstreams,
        rules=rules,
    )
    payload = summary.as_payload()
    phases = {phase["phase_kind"]: phase for phase in payload["phases"]}

    assert payload["item_count"] == 6
    assert set(phases) == {"scan", "predict", "complete"}
    assert {item["source_kind"] for item in phases["scan"]["items"]} == {
        "declared_touchsite",
        "kernel_vm_alignment",
    }
    assert {item["source_kind"] for item in phases["predict"]["items"]} == {
        "declared_counterfactual",
        "declared_dependency",
    }
    assert {item["source_kind"] for item in phases["complete"]["items"]} == {
        "recommended_cut",
        "recommended_followup",
    }
    repo_item = next(
        item
        for item in phases["complete"]["items"]
        if item["source_kind"] == "recommended_followup"
    )
    assert repo_item["queue_id"].startswith("planner_queue|followup_family=structural_cut|")
    assert payload["selected_completion_item_ids"] == [
        "complete.workstream:WS-001",
        "complete.repo:WS-001-TP-001",
    ]


def test_build_planning_chart_identity_grammar_anchors_unresolved_refs() -> None:
    summary = PlanningChartSummary(
        item_count=1,
        selected_completion_item_ids=("complete.repo:WS-001-TP-001",),
        phases=(
            PlanningPhaseSummary(
                phase_kind="complete",
                item_count=1,
                status_counts={"coverage_gap": 1},
                blocker_counts={"coverage_gap": 1},
                selected_item_ids=("complete.repo:WS-001-TP-001",),
                items=(
                    PlanningChartItem(
                        item_id="complete.repo:WS-001-TP-001",
                        phase_kind="complete",
                        item_kind="repo_recommendation",
                        source_kind="recommended_followup",
                        title="repo frontier",
                        status_hint="coverage_gap",
                        selection_rank=0,
                        tracked_node_ids=("node:1",),
                        tracked_object_ids=("WS-001-TP-001",),
                    ),
                ),
            ),
        ),
    )

    bundle = build_planning_chart_identity_grammar(summary=summary)

    assert "planning_chart" in bundle.zones
    assert "planning_external_anchor" in bundle.zones
    assert any(item.zone_name.value == "planning_chart" for item in bundle.carriers)
    assert any(item.zone_name.value == "planning_external_anchor" for item in bundle.carriers)
    assert {item.morphism_kind for item in bundle.morphisms} == {"derived_from"}


def test_build_planning_chart_identity_grammar_tracks_resolved_zone_carriers() -> None:
    summary = PlanningChartSummary(
        item_count=1,
        selected_completion_item_ids=("complete.repo:WS-001-TP-001",),
        phases=(
            PlanningPhaseSummary(
                phase_kind="complete",
                item_count=1,
                status_counts={"coverage_gap": 1},
                blocker_counts={"coverage_gap": 1},
                selected_item_ids=("complete.repo:WS-001-TP-001",),
                items=(
                    PlanningChartItem(
                        item_id="complete.repo:WS-001-TP-001",
                        phase_kind="complete",
                        item_kind="repo_recommendation",
                        source_kind="recommended_followup",
                        title="repo frontier",
                        status_hint="coverage_gap",
                        selection_rank=0,
                        tracked_node_ids=(),
                        tracked_object_ids=("WS-001-TP-001",),
                    ),
                ),
            ),
        ),
    )
    resolved_carrier = IdentityCarrier(
        canonical=IdentityAtom(atom_id=0, namespace="hotspot_queue.item", token="file:src/gabion/a.py"),
        zone_name=IdentityZoneName("hotspot_queue"),
        carrier_kind="file",
        label="src/gabion/a.py",
    )

    bundle = build_planning_chart_identity_grammar(
        summary=summary,
        resolved_carriers={"WS-001-TP-001": resolved_carrier},
    )

    assert any(item.zone_name.value == "hotspot_queue" for item in bundle.carriers)
    assert any(
        item.morphism_kind == "tracks" and item.target_zone == "hotspot_queue"
        for item in bundle.morphisms
    )
