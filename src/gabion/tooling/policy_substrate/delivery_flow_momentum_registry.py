from __future__ import annotations

from gabion.invariants import todo_decorator
from gabion.tooling.policy_substrate.workstream_registry import (
    RegisteredRootDefinition,
    RegisteredSubqueueDefinition,
    RegisteredTouchpointDefinition,
    WorkstreamRegistry,
    declared_touchsite_definition,
    registry_marker_metadata,
)


@todo_decorator(
    reason="DFM remains active while delivery-flow trend health still requires a planner-visible momentum root over historical CI and correction telemetry.",
    reasoning={
        "summary": "Delivery-flow momentum remains a synthetic trend root until historical runtime, recurrence, and closure telemetry converge on a stable low-drag baseline.",
        "control": "delivery_flow_momentum.root",
        "blocking_dependencies": (
            "DFM-SQ-001",
            "DFM-SQ-002",
        ),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="DFM closure",
    links=[{"kind": "object_id", "value": "DFM"}],
)
def _dfm_root() -> None:
    return None


@todo_decorator(
    reason="DFM-SQ-001 remains active while historical full-lane runtime trend still imposes delivery drag.",
    reasoning={
        "summary": "Historical full-lane runtime trend remains planner-visible momentum debt until the lane stops regressing over recent runs.",
        "control": "delivery_flow_momentum.runtime_trend",
        "blocking_dependencies": ("DFM-TP-001",),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="DFM closure",
    links=[
        {"kind": "object_id", "value": "DFM"},
        {"kind": "object_id", "value": "DFM-SQ-001"},
    ],
)
def _dfm_sq_runtime_trend() -> None:
    return None


@todo_decorator(
    reason="DFM-SQ-002 remains active while recurrence and closure telemetry still indicate momentum drag across recent runs.",
    reasoning={
        "summary": "Historical recurrence and correction-loop closure cadence remain synthetic momentum debt until they stop degrading across the recent window.",
        "control": "delivery_flow_momentum.recurrence_trend",
        "blocking_dependencies": ("DFM-TP-002",),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="DFM closure",
    links=[
        {"kind": "object_id", "value": "DFM"},
        {"kind": "object_id", "value": "DFM-SQ-002"},
    ],
)
def _dfm_sq_recurrence_trend() -> None:
    return None


@todo_decorator(
    reason="DFM-TP-001 remains active while recent full-lane runtime trend can still erode delivery momentum.",
    reasoning={
        "summary": "Historical lane runtime trend remains a delivery-momentum signal rather than a current-indicator blocker unless it crosses the current-severity band.",
        "control": "delivery_flow_momentum.runtime_trend.touchpoint",
    },
    owner="gabion.tooling.runtime_policy",
    expiry="DFM closure",
    links=[
        {"kind": "object_id", "value": "DFM"},
        {"kind": "object_id", "value": "DFM-SQ-001"},
        {"kind": "object_id", "value": "DFM-TP-001"},
    ],
)
def _dfm_tp_runtime_trend() -> None:
    return None


@todo_decorator(
    reason="DFM-TP-002 remains active while recurrence-rate and correction-lag trend still indicate flow drag across recent runs.",
    reasoning={
        "summary": "Recent recurrence and closure telemetry remain trend-only momentum signals and should stay planner-visible as a synthetic sibling root to DFR.",
        "control": "delivery_flow_momentum.recurrence_trend.touchpoint",
    },
    owner="gabion.tooling.runtime_policy",
    expiry="DFM closure",
    links=[
        {"kind": "object_id", "value": "DFM"},
        {"kind": "object_id", "value": "DFM-SQ-002"},
        {"kind": "object_id", "value": "DFM-TP-002"},
    ],
)
def _dfm_tp_recurrence_trend() -> None:
    return None


def _root_definition(
    *,
    root_id: str,
    title: str,
    subqueue_ids: tuple[str, ...],
    symbol,
) -> RegisteredRootDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="delivery_flow_momentum_root",
        structural_path=f"delivery_flow_momentum.root::{root_id}",
    )
    return RegisteredRootDefinition(
        root_id=root_id,
        title=title,
        rel_path=metadata.rel_path,
        qualname=metadata.qualname,
        line=metadata.line,
        site_identity=metadata.site_identity,
        structural_identity=metadata.structural_identity,
        marker_identity=metadata.marker_identity,
        marker_payload=metadata.marker_payload,
        subqueue_ids=subqueue_ids,
        status_hint="in_progress",
    )


def _subqueue_definition(
    *,
    root_id: str,
    subqueue_id: str,
    title: str,
    touchpoint_ids: tuple[str, ...],
    symbol,
) -> RegisteredSubqueueDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="delivery_flow_momentum_subqueue",
        structural_path=f"delivery_flow_momentum.subqueue::{subqueue_id}",
    )
    return RegisteredSubqueueDefinition(
        root_id=root_id,
        subqueue_id=subqueue_id,
        title=title,
        rel_path=metadata.rel_path,
        qualname=metadata.qualname,
        line=metadata.line,
        site_identity=metadata.site_identity,
        structural_identity=metadata.structural_identity,
        marker_identity=metadata.marker_identity,
        marker_payload=metadata.marker_payload,
        touchpoint_ids=touchpoint_ids,
        status_hint="in_progress",
    )


def _touchpoint_definition(
    *,
    root_id: str,
    subqueue_id: str,
    touchpoint_id: str,
    title: str,
    symbol,
    declared_touchsites: tuple = (),
) -> RegisteredTouchpointDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="delivery_flow_momentum_touchpoint",
        structural_path=f"delivery_flow_momentum.touchpoint::{touchpoint_id}",
    )
    return RegisteredTouchpointDefinition(
        root_id=root_id,
        touchpoint_id=touchpoint_id,
        subqueue_id=subqueue_id,
        title=title,
        rel_path=metadata.rel_path,
        qualname=metadata.qualname,
        line=metadata.line,
        site_identity=metadata.site_identity,
        structural_identity=metadata.structural_identity,
        marker_identity=metadata.marker_identity,
        marker_payload=metadata.marker_payload,
        status_hint="queued",
        declared_touchsites=declared_touchsites,
    )


def delivery_flow_momentum_workstream_registry() -> WorkstreamRegistry:
    root_id = "DFM"
    return WorkstreamRegistry(
        root=_root_definition(
            root_id=root_id,
            title="Delivery-Flow Momentum / Historical Trend Indicators",
            subqueue_ids=("DFM-SQ-001", "DFM-SQ-002"),
            symbol=_dfm_root,
        ),
        subqueues=(
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="DFM-SQ-001",
                title="Historical full-lane runtime trend",
                touchpoint_ids=("DFM-TP-001",),
                symbol=_dfm_sq_runtime_trend,
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="DFM-SQ-002",
                title="Historical recurrence and correction-lag trend",
                touchpoint_ids=("DFM-TP-002",),
                symbol=_dfm_sq_recurrence_trend,
            ),
        ),
        touchpoints=(
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="DFM-SQ-001",
                touchpoint_id="DFM-TP-001",
                title="Full-lane runtime trend drift",
                symbol=_dfm_tp_runtime_trend,
                declared_touchsites=(
                    declared_touchsite_definition(
                        touchsite_id="DFM-TS-001",
                        rel_path="artifacts/out/governance_telemetry_history.json",
                        qualname="governance_telemetry_history.json",
                        boundary_name="governance_telemetry_history.json",
                        line=1,
                        node_kind="artifact_file",
                        surface="delivery_flow_momentum",
                        structural_path="artifact::artifacts/out/governance_telemetry_history.json",
                    ),
                    declared_touchsite_definition(
                        touchsite_id="DFM-TS-002",
                        rel_path="artifacts/audit_reports/ci_step_timings.json",
                        qualname="ci_step_timings.json",
                        boundary_name="ci_step_timings.json",
                        line=1,
                        node_kind="artifact_file",
                        surface="delivery_flow_momentum",
                        structural_path="artifact::artifacts/audit_reports/ci_step_timings.json",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="DFM-SQ-002",
                touchpoint_id="DFM-TP-002",
                title="Recurrence-rate and closure-cadence drift",
                symbol=_dfm_tp_recurrence_trend,
                declared_touchsites=(
                    declared_touchsite_definition(
                        touchsite_id="DFM-TS-003",
                        rel_path="artifacts/out/governance_telemetry_history.json",
                        qualname="governance_telemetry_history.json",
                        boundary_name="governance_telemetry_history.json",
                        line=1,
                        node_kind="artifact_file",
                        surface="delivery_flow_momentum",
                        structural_path="artifact::artifacts/out/governance_telemetry_history.json",
                    ),
                    declared_touchsite_definition(
                        touchsite_id="DFM-TS-004",
                        rel_path="artifacts/out/governance_telemetry.json",
                        qualname="governance_telemetry.json",
                        boundary_name="governance_telemetry.json",
                        line=1,
                        node_kind="artifact_file",
                        surface="delivery_flow_momentum",
                        structural_path="artifact::artifacts/out/governance_telemetry.json",
                    ),
                ),
            ),
        ),
        tags=("delivery_flow_momentum",),
    )


__all__ = ["delivery_flow_momentum_workstream_registry"]
