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
    reason="DFM remains active while delivery-flow trend health still requires a planner-visible momentum root over runtime, recurrence, dwell, and closure telemetry.",
    reasoning={
        "summary": "Delivery-flow momentum remains a synthetic trend root until historical runtime, recurrence, red-state dwell, and closure telemetry converge on a stable low-drag baseline.",
        "control": "delivery_flow_momentum.root",
        "blocking_dependencies": ("DFM-SQ-001", "DFM-SQ-002", "DFM-SQ-003"),
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
    reason="DFM-SQ-002 remains active while recurrence telemetry still indicates momentum drag across recent runs.",
    reasoning={
        "summary": "Historical recurrence trend remains synthetic momentum debt until recurring loop degradation stops across the recent window.",
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
    reason="DFM-SQ-003 remains active while red-state dwell and closure-lag telemetry still indicate momentum drag across recent runs.",
    reasoning={
        "summary": "Historical red-state dwell and closure-lag telemetry remain planner-visible synthetic momentum debt until their recent-run drag collapses.",
        "control": "delivery_flow_momentum.dwell_and_closure_lag",
        "blocking_dependencies": ("DFM-TP-003",),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="DFM closure",
    links=[
        {"kind": "object_id", "value": "DFM"},
        {"kind": "object_id", "value": "DFM-SQ-003"},
    ],
)
def _dfm_sq_dwell_and_closure_lag() -> None:
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
    reason="DFM-TP-002 remains active while recurrence-rate trend still indicates flow drag across recent runs.",
    reasoning={
        "summary": "Recent recurrence telemetry remains trend-only momentum debt and should stay planner-visible as a synthetic sibling root to DFR.",
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


@todo_decorator(
    reason="DFM-TP-003 remains active while red-state dwell and closure lag still indicate delivery-flow momentum drag.",
    reasoning={
        "summary": "Red-state dwell and closure lag remain trend-only momentum signals and should stay planner-visible as a separate touchpoint from current reliability blockers.",
        "control": "delivery_flow_momentum.dwell_and_closure_lag.touchpoint",
    },
    owner="gabion.tooling.runtime_policy",
    expiry="DFM closure",
    links=[
        {"kind": "object_id", "value": "DFM"},
        {"kind": "object_id", "value": "DFM-SQ-003"},
        {"kind": "object_id", "value": "DFM-TP-003"},
    ],
)
def _dfm_tp_dwell_and_closure_lag() -> None:
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


def _summary_touchsites(
    *,
    artifact_touchsite_id: str,
    producer_touchsite_id: str,
) -> tuple:
    return (
        declared_touchsite_definition(
            touchsite_id=artifact_touchsite_id,
            rel_path="artifacts/out/delivery_flow_summary.json",
            qualname="delivery_flow_summary.json",
            boundary_name="delivery_flow_summary.json",
            line=1,
            node_kind="artifact_file",
            surface="delivery_flow_momentum",
            structural_path="artifact::artifacts/out/delivery_flow_summary.json",
        ),
        declared_touchsite_definition(
            touchsite_id=producer_touchsite_id,
            rel_path="scripts/governance/delivery_flow_emit.py",
            qualname="scripts.governance.delivery_flow_emit.main",
            boundary_name="gabion governance delivery-flow-emit",
            line=1,
            node_kind="script",
            surface="delivery_flow_momentum",
            structural_path="script::scripts/governance/delivery_flow_emit.py",
        ),
    )


def delivery_flow_momentum_workstream_registry() -> WorkstreamRegistry:
    root_id = "DFM"
    return WorkstreamRegistry(
        root=_root_definition(
            root_id=root_id,
            title="Delivery-Flow Momentum / Historical Trend Indicators",
            subqueue_ids=("DFM-SQ-001", "DFM-SQ-002", "DFM-SQ-003"),
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
                title="Historical recurrence trend",
                touchpoint_ids=("DFM-TP-002",),
                symbol=_dfm_sq_recurrence_trend,
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="DFM-SQ-003",
                title="Historical red-state dwell and closure lag",
                touchpoint_ids=("DFM-TP-003",),
                symbol=_dfm_sq_dwell_and_closure_lag,
            ),
        ),
        touchpoints=(
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="DFM-SQ-001",
                touchpoint_id="DFM-TP-001",
                title="Full-lane runtime trend drift",
                symbol=_dfm_tp_runtime_trend,
                declared_touchsites=_summary_touchsites(
                    artifact_touchsite_id="DFM-TS-001",
                    producer_touchsite_id="DFM-TS-002",
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="DFM-SQ-002",
                touchpoint_id="DFM-TP-002",
                title="Recurrence-rate trend drift",
                symbol=_dfm_tp_recurrence_trend,
                declared_touchsites=_summary_touchsites(
                    artifact_touchsite_id="DFM-TS-003",
                    producer_touchsite_id="DFM-TS-004",
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="DFM-SQ-003",
                touchpoint_id="DFM-TP-003",
                title="Red-state dwell and closure-lag drag",
                symbol=_dfm_tp_dwell_and_closure_lag,
                declared_touchsites=_summary_touchsites(
                    artifact_touchsite_id="DFM-TS-005",
                    producer_touchsite_id="DFM-TS-006",
                ),
            ),
        ),
        tags=("delivery_flow_momentum",),
    )


__all__ = ["delivery_flow_momentum_workstream_registry"]
