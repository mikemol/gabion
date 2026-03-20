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
    reason="DFR remains active while repo-local delivery flow still needs a planner-visible current-indicator root for red-state, parity, execution-health, and blocker-pattern pressure.",
    reasoning={
        "summary": "Delivery-flow reliability remains synthetic current-state work until failing tests, local parity gaps, execution-health blockers, and repeated blocker patterns converge into a stable green lane.",
        "control": "delivery_flow_reliability.root",
        "blocking_dependencies": ("DFR-SQ-001", "DFR-SQ-002", "DFR-SQ-003"),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="DFR closure",
    links=[{"kind": "object_id", "value": "DFR"}],
)
def _dfr_root() -> None:
    return None


@todo_decorator(
    reason="DFR-SQ-001 remains active while current red-state and local parity blockers still degrade the delivery loop.",
    reasoning={
        "summary": "Current test red-state and local-vs-CI reproduction drift remain active delivery blockers that must stay planner-visible.",
        "control": "delivery_flow_reliability.current_blockers",
        "blocking_dependencies": ("DFR-TP-001", "DFR-TP-002"),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="DFR closure",
    links=[
        {"kind": "object_id", "value": "DFR"},
        {"kind": "object_id", "value": "DFR-SQ-001"},
    ],
)
def _dfr_sq_current_blockers() -> None:
    return None


@todo_decorator(
    reason="DFR-SQ-002 remains active while observability gaps and severe current-band runtime regressions can still break the active delivery loop.",
    reasoning={
        "summary": "Execution observability and severe current-band runtime regressions remain active current-state blockers rather than trend-only telemetry.",
        "control": "delivery_flow_reliability.execution_health",
        "blocking_dependencies": ("DFR-TP-003",),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="DFR closure",
    links=[
        {"kind": "object_id", "value": "DFR"},
        {"kind": "object_id", "value": "DFR-SQ-002"},
    ],
)
def _dfr_sq_execution_health() -> None:
    return None


@todo_decorator(
    reason="DFR-SQ-003 remains active while repeated, stalled, or unstable current blockers still degrade the delivery loop.",
    reasoning={
        "summary": "Repeated, stalled, and unstable blocker patterns remain active reliability debt and need their own planner-visible current-indicator touchpoint.",
        "control": "delivery_flow_reliability.blocker_patterns",
        "blocking_dependencies": ("DFR-TP-004",),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="DFR closure",
    links=[
        {"kind": "object_id", "value": "DFR"},
        {"kind": "object_id", "value": "DFR-SQ-003"},
    ],
)
def _dfr_sq_blocker_patterns() -> None:
    return None


@todo_decorator(
    reason="DFR-TP-001 remains active while failing junit cases still indicate current full-lane red-state.",
    reasoning={
        "summary": "The synthetic delivery-flow reliability root must surface current junit red-state directly rather than forcing operators to infer it from broader workstreams.",
        "control": "delivery_flow_reliability.current_red_state.touchpoint",
    },
    owner="gabion.tooling.runtime_policy",
    expiry="DFR closure",
    links=[
        {"kind": "object_id", "value": "DFR"},
        {"kind": "object_id", "value": "DFR-SQ-001"},
        {"kind": "object_id", "value": "DFR-TP-001"},
    ],
)
def _dfr_tp_current_red_state() -> None:
    return None


@todo_decorator(
    reason="DFR-TP-002 remains active while local-vs-CI reproduction contracts can still fail the current delivery loop.",
    reasoning={
        "summary": "Local parity and CI reproduction capability drift remain active current blockers and need direct planner ownership.",
        "control": "delivery_flow_reliability.local_ci_parity.touchpoint",
    },
    owner="gabion.tooling.runtime_policy",
    expiry="DFR closure",
    links=[
        {"kind": "object_id", "value": "DFR"},
        {"kind": "object_id", "value": "DFR-SQ-001"},
        {"kind": "object_id", "value": "DFR-TP-002"},
    ],
)
def _dfr_tp_local_ci_parity() -> None:
    return None


@todo_decorator(
    reason="DFR-TP-003 remains active while observability gaps or severe current-band runtime regressions can still block the live delivery loop.",
    reasoning={
        "summary": "Current execution-health blockers such as observability violations and severe runtime current-band regressions remain current-indicator signals for delivery reliability.",
        "control": "delivery_flow_reliability.execution_health.touchpoint",
    },
    owner="gabion.tooling.runtime_policy",
    expiry="DFR closure",
    links=[
        {"kind": "object_id", "value": "DFR"},
        {"kind": "object_id", "value": "DFR-SQ-002"},
        {"kind": "object_id", "value": "DFR-TP-003"},
    ],
)
def _dfr_tp_execution_health() -> None:
    return None


@todo_decorator(
    reason="DFR-TP-004 remains active while repeated, stalled, or unstable blocker patterns still indicate current delivery fragility.",
    reasoning={
        "summary": "Repeated, stalled, and unstable blockers remain current-indicator signals and should not be smuggled into trend debt only.",
        "control": "delivery_flow_reliability.blocker_patterns.touchpoint",
    },
    owner="gabion.tooling.runtime_policy",
    expiry="DFR closure",
    links=[
        {"kind": "object_id", "value": "DFR"},
        {"kind": "object_id", "value": "DFR-SQ-003"},
        {"kind": "object_id", "value": "DFR-TP-004"},
    ],
)
def _dfr_tp_blocker_patterns() -> None:
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
        surface="delivery_flow_reliability_root",
        structural_path=f"delivery_flow_reliability.root::{root_id}",
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
        surface="delivery_flow_reliability_subqueue",
        structural_path=f"delivery_flow_reliability.subqueue::{subqueue_id}",
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
    test_path_prefixes: tuple[str, ...] = (),
) -> RegisteredTouchpointDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="delivery_flow_reliability_touchpoint",
        structural_path=f"delivery_flow_reliability.touchpoint::{touchpoint_id}",
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
        test_path_prefixes=test_path_prefixes,
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
            surface="delivery_flow_reliability",
            structural_path="artifact::artifacts/out/delivery_flow_summary.json",
        ),
        declared_touchsite_definition(
            touchsite_id=producer_touchsite_id,
            rel_path="scripts/governance/delivery_flow_emit.py",
            qualname="scripts.governance.delivery_flow_emit.main",
            boundary_name="gabion governance delivery-flow-emit",
            line=1,
            node_kind="script",
            surface="delivery_flow_reliability",
            structural_path="script::scripts/governance/delivery_flow_emit.py",
        ),
    )


def delivery_flow_reliability_workstream_registry() -> WorkstreamRegistry:
    root_id = "DFR"
    return WorkstreamRegistry(
        root=_root_definition(
            root_id=root_id,
            title="Delivery-Flow Reliability / Current Indicators",
            subqueue_ids=("DFR-SQ-001", "DFR-SQ-002", "DFR-SQ-003"),
            symbol=_dfr_root,
        ),
        subqueues=(
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="DFR-SQ-001",
                title="Current red-state and local parity blockers",
                touchpoint_ids=("DFR-TP-001", "DFR-TP-002"),
                symbol=_dfr_sq_current_blockers,
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="DFR-SQ-002",
                title="Execution observability and current-band runtime blockers",
                touchpoint_ids=("DFR-TP-003",),
                symbol=_dfr_sq_execution_health,
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="DFR-SQ-003",
                title="Repeated, stalled, and unstable current blockers",
                touchpoint_ids=("DFR-TP-004",),
                symbol=_dfr_sq_blocker_patterns,
            ),
        ),
        touchpoints=(
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="DFR-SQ-001",
                touchpoint_id="DFR-TP-001",
                title="Current full-suite red-state indicators",
                symbol=_dfr_tp_current_red_state,
                declared_touchsites=_summary_touchsites(
                    artifact_touchsite_id="DFR-TS-001",
                    producer_touchsite_id="DFR-TS-002",
                ),
                test_path_prefixes=("tests/",),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="DFR-SQ-001",
                touchpoint_id="DFR-TP-002",
                title="Current local-vs-CI parity blockers",
                symbol=_dfr_tp_local_ci_parity,
                declared_touchsites=_summary_touchsites(
                    artifact_touchsite_id="DFR-TS-003",
                    producer_touchsite_id="DFR-TS-004",
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="DFR-SQ-002",
                touchpoint_id="DFR-TP-003",
                title="Execution observability and severe current-band runtime blockers",
                symbol=_dfr_tp_execution_health,
                declared_touchsites=_summary_touchsites(
                    artifact_touchsite_id="DFR-TS-005",
                    producer_touchsite_id="DFR-TS-006",
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="DFR-SQ-003",
                touchpoint_id="DFR-TP-004",
                title="Repeated, stalled, and unstable current blocker patterns",
                symbol=_dfr_tp_blocker_patterns,
                declared_touchsites=_summary_touchsites(
                    artifact_touchsite_id="DFR-TS-007",
                    producer_touchsite_id="DFR-TS-008",
                ),
            ),
        ),
        tags=("delivery_flow_reliability",),
    )


__all__ = ["delivery_flow_reliability_workstream_registry"]
