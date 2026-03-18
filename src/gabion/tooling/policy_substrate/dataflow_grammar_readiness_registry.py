from __future__ import annotations

from gabion.invariants import todo_decorator
from gabion.tooling.policy_substrate.workstream_registry import (
    RegisteredDataflowSignalSelector,
    RegisteredRootDefinition,
    RegisteredSubqueueDefinition,
    RegisteredTouchpointDefinition,
    WorkstreamRegistry,
    registry_marker_metadata,
)


@todo_decorator(
    reason="DGR remains active until local dataflow-grammar runs no longer emit terminal failures, resumable timeout residue, or blocking obligation-trace signals.",
    reasoning={
        "summary": "The synthetic dataflow-grammar readiness root tracks structured local execution indicators from run-dataflow-stage artifacts until local dataflow readiness converges.",
        "control": "dataflow_grammar_readiness.root",
        "blocking_dependencies": ("DGR-SQ-001", "DGR-SQ-002"),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="DGR closure",
    links=[{"kind": "object_id", "value": "DGR"}],
)
def _dgr_root() -> None:
    return None


@todo_decorator(
    reason="DGR-SQ-001 remains active until local dataflow-grammar execution no longer terminates in hard-failure or resumable-timeout states.",
    reasoning={
        "summary": "Terminal execution outcomes from the local dataflow-stage runtime still expose blocking hard-failure and incomplete timeout states.",
        "control": "dataflow_grammar_readiness.terminal_execution",
        "blocking_dependencies": ("DGR-TP-001", "DGR-TP-002"),
    },
    owner="gabion.analysis.dataflow",
    expiry="DGR closure",
    links=[
        {"kind": "object_id", "value": "DGR"},
        {"kind": "object_id", "value": "DGR-SQ-001"},
    ],
)
def _dgr_sq_terminal_execution() -> None:
    return None


@todo_decorator(
    reason="DGR-SQ-002 remains active until local obligation-trace artifacts no longer report unsatisfied or skipped-by-policy contract rows.",
    reasoning={
        "summary": "Structured obligation-trace rows from local dataflow-grammar runs still expose unsatisfied and skipped-by-policy contract surfaces.",
        "control": "dataflow_grammar_readiness.obligation_trace",
        "blocking_dependencies": ("DGR-TP-003", "DGR-TP-004"),
    },
    owner="gabion.analysis.dataflow",
    expiry="DGR closure",
    links=[
        {"kind": "object_id", "value": "DGR"},
        {"kind": "object_id", "value": "DGR-SQ-002"},
    ],
)
def _dgr_sq_obligation_trace() -> None:
    return None


@todo_decorator(
    reason="DGR-TP-001 remains queued until local dataflow-grammar runs stop ending in hard-failure terminal status.",
    reasoning={
        "summary": "Hard-failure terminal outcomes remain an actionable local execution blocker for dataflow grammar readiness.",
        "control": "dataflow_grammar_readiness.terminal_hard_failure.touchpoint",
    },
    owner="gabion.analysis.dataflow",
    expiry="DGR closure",
    links=[
        {"kind": "object_id", "value": "DGR"},
        {"kind": "object_id", "value": "DGR-SQ-001"},
        {"kind": "object_id", "value": "DGR-TP-001"},
    ],
)
def _dgr_tp_terminal_hard_failure() -> None:
    return None


@todo_decorator(
    reason="DGR-TP-002 remains queued until local dataflow-grammar runs stop ending in resumable timeout states with incomplete timeout markers.",
    reasoning={
        "summary": "Timeout-resume terminal outcomes with local incompleteness markers still represent incomplete local dataflow-grammar execution.",
        "control": "dataflow_grammar_readiness.timeout_resume.touchpoint",
    },
    owner="gabion.analysis.dataflow",
    expiry="DGR closure",
    links=[
        {"kind": "object_id", "value": "DGR"},
        {"kind": "object_id", "value": "DGR-SQ-001"},
        {"kind": "object_id", "value": "DGR-TP-002"},
    ],
)
def _dgr_tp_timeout_resume() -> None:
    return None


@todo_decorator(
    reason="DGR-TP-003 remains queued until local obligation traces stop reporting unsatisfied contract rows.",
    reasoning={
        "summary": "Unsatisfied obligation-trace rows remain actionable local dataflow-grammar contract failures.",
        "control": "dataflow_grammar_readiness.unsatisfied_obligation.touchpoint",
    },
    owner="gabion.analysis.dataflow",
    expiry="DGR closure",
    links=[
        {"kind": "object_id", "value": "DGR"},
        {"kind": "object_id", "value": "DGR-SQ-002"},
        {"kind": "object_id", "value": "DGR-TP-003"},
    ],
)
def _dgr_tp_unsatisfied_obligations() -> None:
    return None


@todo_decorator(
    reason="DGR-TP-004 remains queued until local obligation traces stop reporting skipped-by-policy contract rows.",
    reasoning={
        "summary": "Skipped-by-policy obligation-trace rows remain local readiness residue until the local dataflow contract surface is complete.",
        "control": "dataflow_grammar_readiness.skipped_obligation.touchpoint",
    },
    owner="gabion.analysis.dataflow",
    expiry="DGR closure",
    links=[
        {"kind": "object_id", "value": "DGR"},
        {"kind": "object_id", "value": "DGR-SQ-002"},
        {"kind": "object_id", "value": "DGR-TP-004"},
    ],
)
def _dgr_tp_skipped_obligations() -> None:
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
        surface="dataflow_grammar_readiness_root",
        structural_path=f"dataflow_grammar_readiness.root::{root_id}",
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
        surface="dataflow_grammar_readiness_subqueue",
        structural_path=f"dataflow_grammar_readiness.subqueue::{subqueue_id}",
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
    selector: RegisteredDataflowSignalSelector,
) -> RegisteredTouchpointDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="dataflow_grammar_readiness_touchpoint",
        structural_path=f"dataflow_grammar_readiness.touchpoint::{touchpoint_id}",
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
        dataflow_signal_selector=selector,
    )


def dataflow_grammar_readiness_workstream_registry() -> WorkstreamRegistry:
    root_id = "DGR"
    return WorkstreamRegistry(
        root=_root_definition(
            root_id=root_id,
            title="Dataflow-Grammar Readiness / Local Execution Indicators",
            subqueue_ids=("DGR-SQ-001", "DGR-SQ-002"),
            symbol=_dgr_root,
            status_hint="in_progress",
        ),
        subqueues=(
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="DGR-SQ-001",
                title="Terminal execution outcome indicators",
                touchpoint_ids=("DGR-TP-001", "DGR-TP-002"),
                symbol=_dgr_sq_terminal_execution,
                status_hint="in_progress",
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="DGR-SQ-002",
                title="Obligation-trace contract indicators",
                touchpoint_ids=("DGR-TP-003", "DGR-TP-004"),
                symbol=_dgr_sq_obligation_trace,
                status_hint="in_progress",
            ),
        ),
        touchpoints=(
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="DGR-SQ-001",
                touchpoint_id="DGR-TP-001",
                title="Local dataflow-grammar hard-failure terminal outcomes",
                symbol=_dgr_tp_terminal_hard_failure,
                status_hint="queued",
                selector=RegisteredDataflowSignalSelector(
                    terminal_statuses=("hard_failure",),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="DGR-SQ-001",
                touchpoint_id="DGR-TP-002",
                title="Local dataflow-grammar timeout-resume and incomplete execution outcomes",
                symbol=_dgr_tp_timeout_resume,
                status_hint="queued",
                selector=RegisteredDataflowSignalSelector(
                    terminal_statuses=("timeout_resume",),
                    incompleteness_markers=("terminal_non_success", "timeout_or_partial_run"),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="DGR-SQ-002",
                touchpoint_id="DGR-TP-003",
                title="Local dataflow-grammar unsatisfied obligation rows",
                symbol=_dgr_tp_unsatisfied_obligations,
                status_hint="queued",
                selector=RegisteredDataflowSignalSelector(
                    obligation_statuses=("unsatisfied",),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="DGR-SQ-002",
                touchpoint_id="DGR-TP-004",
                title="Local dataflow-grammar skipped-by-policy obligation rows",
                symbol=_dgr_tp_skipped_obligations,
                status_hint="queued",
                selector=RegisteredDataflowSignalSelector(
                    obligation_statuses=("skipped_by_policy",),
                ),
            ),
        ),
        tags=("dataflow_grammar_readiness",),
    )


__all__ = ["dataflow_grammar_readiness_workstream_registry"]
