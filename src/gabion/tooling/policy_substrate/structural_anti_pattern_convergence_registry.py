from __future__ import annotations

from gabion.invariants import todo_decorator
from gabion.tooling.policy_substrate.workstream_registry import (
    RegisteredRootDefinition,
    RegisteredSubqueueDefinition,
    RegisteredTouchpointDefinition,
    RegisteredTouchsiteDefinition,
    WorkstreamRegistry,
    registry_marker_metadata,
)


@todo_decorator(
    reason="SAC remains active until structural anti-pattern cleanup converges and the root-owned anti-pattern contract is green across repo scope.",
    reasoning={
        "summary": "The structural anti-pattern convergence root tracks repo-wide cleanup of upstream narrowing, filter-first traversal, streaming collection, and dispatch normalization until the owned contract guard closes green.",
        "control": "structural_anti_pattern_convergence.root",
        "blocking_dependencies": (
            "SAC-SQ-001",
            "SAC-SQ-002",
            "SAC-SQ-003",
            "SAC-SQ-004",
        ),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="SAC closure",
    links=[{"kind": "object_id", "value": "SAC"}],
)
def _sac_root() -> None:
    return None


@todo_decorator(
    reason="SAC-SQ-001 remains active until real runtime shapes stop leaking into wildcard sinks that should be unreachable by construction.",
    reasoning={
        "summary": "Helper and normalization surfaces still need upstream narrowing so wildcard invariant sinks stay dead after ingress discharge.",
        "control": "structural_anti_pattern_convergence.upstream_narrowing",
        "blocking_dependencies": ("SAC-TP-001",),
    },
    owner="gabion.analysis",
    expiry="SAC closure",
    links=[
        {"kind": "object_id", "value": "SAC"},
        {"kind": "object_id", "value": "SAC-SQ-001"},
    ],
)
def _sac_sq_upstream_narrowing() -> None:
    return None


@todo_decorator(
    reason="SAC-SQ-002 remains active until loop-body prefilters are hoisted into typed iterators, filter pipelines, or explicit ingress helpers.",
    reasoning={
        "summary": "AST and similar traversals still carry branchy in-loop type, name, and shape filters that should be pushed into the iterable boundary.",
        "control": "structural_anti_pattern_convergence.filter_first_traversal",
        "blocking_dependencies": ("SAC-TP-002",),
    },
    owner="gabion.analysis",
    expiry="SAC closure",
    links=[
        {"kind": "object_id", "value": "SAC"},
        {"kind": "object_id", "value": "SAC-SQ-002"},
    ],
)
def _sac_sq_filter_first_traversal() -> None:
    return None


@todo_decorator(
    reason="SAC-SQ-003 remains active until eager append/materialize helper paths are collapsed to streaming iteration where behavior is unchanged.",
    reasoning={
        "summary": "Collector helpers still materialize rows, targets, and witnesses earlier than necessary instead of projecting them as streams.",
        "control": "structural_anti_pattern_convergence.streaming_iteration",
        "blocking_dependencies": ("SAC-TP-003",),
    },
    owner="gabion.analysis",
    expiry="SAC closure",
    links=[
        {"kind": "object_id", "value": "SAC"},
        {"kind": "object_id", "value": "SAC-SQ-003"},
    ],
)
def _sac_sq_streaming_iteration() -> None:
    return None


@todo_decorator(
    reason="SAC-SQ-004 remains active until dispatch cleanup converges and the root-owned anti-pattern contract guard closes green.",
    reasoning={
        "summary": "Single-axis runtime dispatch still needs normalization, and the structural anti-pattern contract must become the mechanical closure condition for this root.",
        "control": "structural_anti_pattern_convergence.dispatch_and_guard",
        "blocking_dependencies": ("SAC-TP-004", "SAC-TP-005"),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="SAC closure",
    links=[
        {"kind": "object_id", "value": "SAC"},
        {"kind": "object_id", "value": "SAC-SQ-004"},
    ],
)
def _sac_sq_dispatch_and_guard() -> None:
    return None


@todo_decorator(
    reason="SAC-TP-001 remains queued until the current upstream narrowing tranche restores unreachable wildcard sinks by fixing ingress instead of weakening invariants.",
    reasoning={
        "summary": "Dataflow, indexed-scan, and helper normalization surfaces still need explicit upstream narrowing so semantic wildcards remain post-normalization sinks only.",
        "control": "structural_anti_pattern_convergence.upstream_narrowing.touchpoint",
    },
    owner="gabion.analysis",
    expiry="SAC closure",
    links=[
        {"kind": "object_id", "value": "SAC"},
        {"kind": "object_id", "value": "SAC-SQ-001"},
        {"kind": "object_id", "value": "SAC-TP-001"},
    ],
)
def _sac_tp_upstream_narrowing() -> None:
    return None


@todo_decorator(
    reason="SAC-TP-002 remains queued until loop-body prefilters in AST and similar traversals are hoisted into filter-first iterator boundaries.",
    reasoning={
        "summary": "Traversal helpers still carry pure branch filters in loop bodies instead of exposing already-classified inputs to the loop body.",
        "control": "structural_anti_pattern_convergence.filter_first_traversal.touchpoint",
    },
    owner="gabion.analysis",
    expiry="SAC closure",
    links=[
        {"kind": "object_id", "value": "SAC"},
        {"kind": "object_id", "value": "SAC-SQ-002"},
        {"kind": "object_id", "value": "SAC-TP-002"},
    ],
)
def _sac_tp_filter_first_traversal() -> None:
    return None


@todo_decorator(
    reason="SAC-TP-003 remains queued until append/materialize collector helpers converge on streaming iteration where semantics permit.",
    reasoning={
        "summary": "Witness, violation, and extraction helper paths still use eager list or tuple materialization where a stream pipeline is the correct shape.",
        "control": "structural_anti_pattern_convergence.streaming_iteration.touchpoint",
    },
    owner="gabion.analysis",
    expiry="SAC closure",
    links=[
        {"kind": "object_id", "value": "SAC"},
        {"kind": "object_id", "value": "SAC-SQ-003"},
        {"kind": "object_id", "value": "SAC-TP-003"},
    ],
)
def _sac_tp_streaming_iteration() -> None:
    return None


@todo_decorator(
    reason="SAC-TP-004 remains queued until remaining single-axis branch dispatch sites are normalized to typed helpers or singledispatch families.",
    reasoning={
        "summary": "Local runtime type dispatch still carries nested branch ladders where one-axis typed dispatch is the repo-preferred shape.",
        "control": "structural_anti_pattern_convergence.dispatch_normalization.touchpoint",
    },
    owner="gabion.analysis",
    expiry="SAC closure",
    links=[
        {"kind": "object_id", "value": "SAC"},
        {"kind": "object_id", "value": "SAC-SQ-004"},
        {"kind": "object_id", "value": "SAC-TP-004"},
    ],
)
def _sac_tp_dispatch_normalization() -> None:
    return None


@todo_decorator(
    reason="SAC-TP-005 remains queued until the structural anti-pattern contract audit is green and can serve as the closure guard for this root.",
    reasoning={
        "summary": "The root-owned contract audit must inventory and then block regressions for wildcard soft fallthroughs, loop-body prefilters, and eager helper materialization.",
        "control": "structural_anti_pattern_convergence.contract_guard.touchpoint",
    },
    owner="gabion.tooling.policy_substrate",
    expiry="SAC closure",
    links=[
        {"kind": "object_id", "value": "SAC"},
        {"kind": "object_id", "value": "SAC-SQ-004"},
        {"kind": "object_id", "value": "SAC-TP-005"},
    ],
)
def _sac_tp_contract_guard() -> None:
    return None


def _root_definition(
    *,
    root_id: str,
    title: str,
    subqueue_ids: tuple[str, ...],
    symbol,
    status_hint: str,
) -> RegisteredRootDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="structural_anti_pattern_convergence_root",
        structural_path=f"structural_anti_pattern_convergence.root::{root_id}",
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
        status_hint=status_hint,
    )


def _subqueue_definition(
    *,
    root_id: str,
    subqueue_id: str,
    title: str,
    touchpoint_ids: tuple[str, ...],
    symbol,
    status_hint: str,
) -> RegisteredSubqueueDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="structural_anti_pattern_convergence_subqueue",
        structural_path=f"structural_anti_pattern_convergence.subqueue::{subqueue_id}",
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
        status_hint=status_hint,
    )


def _touchpoint_definition(
    *,
    root_id: str,
    subqueue_id: str,
    touchpoint_id: str,
    title: str,
    symbol,
    status_hint: str,
    declared_touchsites: tuple[RegisteredTouchsiteDefinition, ...] = (),
) -> RegisteredTouchpointDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="structural_anti_pattern_convergence_touchpoint",
        structural_path=f"structural_anti_pattern_convergence.touchpoint::{touchpoint_id}",
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
        status_hint=status_hint,
        declared_touchsites=declared_touchsites,
    )


def _touchsite_definition(
    *,
    touchsite_id: str,
    rel_path: str,
    qualname: str,
    boundary_name: str,
    line: int = 1,
    column: int = 1,
    node_kind: str = "function_def",
    seam_class: str = "surviving_carrier_seam",
    status_hint: str = "open",
    object_ids: tuple[str, ...] = (),
) -> RegisteredTouchsiteDefinition:
    return RegisteredTouchsiteDefinition(
        touchsite_id=touchsite_id,
        rel_path=rel_path,
        qualname=qualname,
        boundary_name=boundary_name,
        line=line,
        column=column,
        node_kind=node_kind,
        site_identity=f"{rel_path}::{qualname}:{line}:{column}",
        structural_identity=f"{rel_path}::{boundary_name}",
        seam_class=seam_class,
        status_hint=status_hint,
        object_ids=object_ids,
    )


def structural_anti_pattern_convergence_workstream_registry() -> WorkstreamRegistry:
    root_id = "SAC"
    return WorkstreamRegistry(
        root=_root_definition(
            root_id=root_id,
            title="Structural Anti-Pattern Convergence",
            subqueue_ids=(
                "SAC-SQ-001",
                "SAC-SQ-002",
                "SAC-SQ-003",
                "SAC-SQ-004",
            ),
            symbol=_sac_root,
            status_hint="in_progress",
        ),
        subqueues=(
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="SAC-SQ-001",
                title="Upstream narrowing and invariant-sink closure",
                touchpoint_ids=("SAC-TP-001",),
                symbol=_sac_sq_upstream_narrowing,
                status_hint="in_progress",
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="SAC-SQ-002",
                title="Filter-first traversal and guard hoisting",
                touchpoint_ids=("SAC-TP-002",),
                symbol=_sac_sq_filter_first_traversal,
                status_hint="in_progress",
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="SAC-SQ-003",
                title="Streaming collection and non-materializing iteration",
                touchpoint_ids=("SAC-TP-003",),
                symbol=_sac_sq_streaming_iteration,
                status_hint="in_progress",
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="SAC-SQ-004",
                title="Dispatch normalization and regression guard closure",
                touchpoint_ids=("SAC-TP-004", "SAC-TP-005"),
                symbol=_sac_sq_dispatch_and_guard,
                status_hint="in_progress",
            ),
        ),
        touchpoints=(
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="SAC-SQ-001",
                touchpoint_id="SAC-TP-001",
                title="Upstream narrowing and wildcard sink closure",
                symbol=_sac_tp_upstream_narrowing,
                status_hint="queued",
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="SAC-SQ-002",
                touchpoint_id="SAC-TP-002",
                title="Filter-first traversal and guard hoisting",
                symbol=_sac_tp_filter_first_traversal,
                status_hint="queued",
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="SAC-SQ-003",
                touchpoint_id="SAC-TP-003",
                title="Streaming helper iteration and materialization collapse",
                symbol=_sac_tp_streaming_iteration,
                status_hint="queued",
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="SAC-SQ-004",
                touchpoint_id="SAC-TP-004",
                title="Dispatch normalization",
                symbol=_sac_tp_dispatch_normalization,
                status_hint="queued",
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="SAC-SQ-004",
                touchpoint_id="SAC-TP-005",
                title="Structural anti-pattern contract guard",
                symbol=_sac_tp_contract_guard,
                status_hint="queued",
                declared_touchsites=(
                    _touchsite_definition(
                        touchsite_id="SAC-TS-001",
                        rel_path="src/gabion/tooling/policy_substrate/structural_anti_pattern_contract.py",
                        qualname="collect_findings",
                        boundary_name="structural_anti_pattern_contract.collect_findings",
                    ),
                    _touchsite_definition(
                        touchsite_id="SAC-TS-002",
                        rel_path="scripts/policy/structural_anti_pattern_contract.py",
                        qualname="main",
                        boundary_name="structural_anti_pattern_contract.main",
                    ),
                    _touchsite_definition(
                        touchsite_id="SAC-TS-003",
                        rel_path="scripts/policy/policy_check.py",
                        qualname="check_structural_anti_pattern_contract",
                        boundary_name="policy_check.check_structural_anti_pattern_contract",
                    ),
                    _touchsite_definition(
                        touchsite_id="SAC-TS-004",
                        rel_path="tests/gabion/tooling/runtime_policy/test_structural_anti_pattern_contract.py",
                        qualname="test_collect_findings",
                        boundary_name="test_structural_anti_pattern_contract",
                    ),
                ),
            ),
        ),
        tags=("structural_convergence",),
    )


__all__ = ["structural_anti_pattern_convergence_workstream_registry"]
