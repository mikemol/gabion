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
    reason="BIC remains active until CLI/dataflow ingress wiring, server-core coercion carriers, and CLI live-repo sentinel boundaries converge onto explicit shared surfaces.",
    reasoning={
        "summary": "CLI/dataflow transport helper wiring, server-core coercion logic, and CLI repo-state smoke tests still drift across parallel ingress and test surfaces.",
        "control": "boundary_ingress_convergence.root",
        "blocking_dependencies": (
            "BIC-SQ-001",
            "BIC-SQ-002",
            "BIC-SQ-003",
        ),
    },
    owner="gabion.tooling.policy_substrate",
    expiry="BIC closure",
    links=[{"kind": "object_id", "value": "BIC"}],
)
def _bic_root() -> None:
    return None


@todo_decorator(
    reason="BIC-SQ-001 remains active until CLI/dataflow transport ingress wiring converges on one shared carrier.",
    reasoning={
        "summary": "cli.py and tooling.runtime.dataflow_invocation_runner still thread parallel timeout, report-path, payload, dispatch, and run-check helpers instead of sharing one ingress carrier.",
        "control": "boundary_ingress_convergence.dataflow_transport_ingress",
        "blocking_dependencies": ("BIC-TP-001",),
    },
    owner="gabion.cli_support",
    expiry="BIC closure",
    links=[
        {"kind": "object_id", "value": "BIC"},
        {"kind": "object_id", "value": "BIC-SQ-001"},
    ],
)
def _bic_sq_dataflow_transport_ingress() -> None:
    return None


@todo_decorator(
    reason="BIC-SQ-002 remains active until server-core coercion helpers converge onto one shared carrier.",
    reasoning={
        "summary": "command_orchestrator, command_orchestrator_progress, and their downstream consumers still split overlapping runtime coercion logic across local singledispatch surfaces.",
        "control": "boundary_ingress_convergence.server_core_coercion",
        "blocking_dependencies": ("BIC-TP-002", "BIC-TP-003"),
    },
    owner="gabion.server_core",
    expiry="BIC closure",
    links=[
        {"kind": "object_id", "value": "BIC"},
        {"kind": "object_id", "value": "BIC-SQ-002"},
    ],
)
def _bic_sq_server_core_coercion() -> None:
    return None


@todo_decorator(
    reason="BIC-SQ-003 remains active until CLI live-repo smoke tests are separated from deterministic command-behavior tests.",
    reasoning={
        "summary": "REPO_ROOT-bound CLI smoke tests still live in the deterministic CLI case module instead of an explicit repo-state sentinel surface.",
        "control": "boundary_ingress_convergence.cli_live_repo_sentinels",
        "blocking_dependencies": ("BIC-TP-004",),
    },
    owner="gabion.cli",
    expiry="BIC closure",
    links=[
        {"kind": "object_id", "value": "BIC"},
        {"kind": "object_id", "value": "BIC-SQ-003"},
    ],
)
def _bic_sq_cli_live_repo_sentinels() -> None:
    return None


@todo_decorator(
    reason="BIC-TP-001 tracks the shared CLI/dataflow transport ingress carrier.",
    reasoning={
        "summary": "CLI and runtime dataflow entry surfaces should share one transport ingress carrier for timeout ticks, report-path resolution, payload helpers, dispatch, and run-check wiring.",
        "control": "boundary_ingress_convergence.dataflow_transport_touchpoint",
        "blocking_dependencies": ("BIC-SQ-001",),
    },
    owner="gabion.cli_support",
    expiry="BIC closure",
    links=[
        {"kind": "object_id", "value": "BIC"},
        {"kind": "object_id", "value": "BIC-SQ-001"},
        {"kind": "object_id", "value": "BIC-TP-001"},
    ],
)
def _bic_tp_dataflow_transport_ingress() -> None:
    return None


@todo_decorator(
    reason="BIC-TP-002 tracks extraction of the canonical server-core coercion carrier.",
    reasoning={
        "summary": "Server-core orchestrator and progress coercion helpers should move to one shared carrier backed by the runtime coercion substrate.",
        "control": "boundary_ingress_convergence.server_core_coercion_extract",
        "blocking_dependencies": ("BIC-SQ-002",),
    },
    owner="gabion.server_core",
    expiry="BIC closure",
    links=[
        {"kind": "object_id", "value": "BIC"},
        {"kind": "object_id", "value": "BIC-SQ-002"},
        {"kind": "object_id", "value": "BIC-TP-002"},
    ],
)
def _bic_tp_server_core_coercion_extract() -> None:
    return None


@todo_decorator(
    reason="BIC-TP-003 tracks downstream migration onto the shared server-core coercion carrier.",
    reasoning={
        "summary": "command_orchestrator_primitives and server_payload_dispatch should import the new server-core coercion carrier instead of reusing progress-local helper surfaces.",
        "control": "boundary_ingress_convergence.server_core_coercion_migrate",
        "blocking_dependencies": ("BIC-SQ-002",),
    },
    owner="gabion.server_core",
    expiry="BIC closure",
    links=[
        {"kind": "object_id", "value": "BIC"},
        {"kind": "object_id", "value": "BIC-SQ-002"},
        {"kind": "object_id", "value": "BIC-TP-003"},
    ],
)
def _bic_tp_server_core_coercion_migrate() -> None:
    return None


@todo_decorator(
    reason="BIC-TP-004 tracks the dedicated CLI live-repo sentinel split.",
    reasoning={
        "summary": "CLI REPO_ROOT smoke tests should move to a dedicated live-repo sentinel module so deterministic command-behavior cases remain synthetic and stable.",
        "control": "boundary_ingress_convergence.cli_live_repo_split",
        "blocking_dependencies": ("BIC-SQ-003",),
    },
    owner="gabion.cli",
    expiry="BIC closure",
    links=[
        {"kind": "object_id", "value": "BIC"},
        {"kind": "object_id", "value": "BIC-SQ-003"},
        {"kind": "object_id", "value": "BIC-TP-004"},
    ],
)
def _bic_tp_cli_live_repo_split() -> None:
    return None


def _root_definition(
    *,
    root_id: str,
    title: str,
    subqueue_ids: tuple[str, ...],
    symbol,
    status_hint: str = "",
) -> RegisteredRootDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="boundary_ingress_convergence_root",
        structural_path=f"boundary_ingress_convergence.root::{root_id}",
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
    status_hint: str = "",
) -> RegisteredSubqueueDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="boundary_ingress_convergence_subqueue",
        structural_path=f"boundary_ingress_convergence.subqueue::{subqueue_id}",
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
    declared_touchsites: tuple = (),
    status_hint: str = "",
) -> RegisteredTouchpointDefinition:
    metadata = registry_marker_metadata(
        symbol,
        surface="boundary_ingress_convergence_touchpoint",
        structural_path=f"boundary_ingress_convergence.touchpoint::{touchpoint_id}",
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
        declared_touchsites=declared_touchsites,
        status_hint=status_hint,
    )


def _module_touchsite(
    *,
    touchsite_id: str,
    rel_path: str,
    qualname: str,
) -> object:
    return declared_touchsite_definition(
        touchsite_id=touchsite_id,
        rel_path=rel_path,
        qualname=qualname,
        boundary_name=qualname,
        line=1,
        node_kind="module",
        surface="boundary_ingress_convergence_touchsite",
        structural_path=f"boundary_ingress_convergence.touchsite::{touchsite_id}",
        seam_class="surviving_carrier_seam",
    )


def boundary_ingress_convergence_workstream_registry() -> WorkstreamRegistry:
    root_id = "BIC"
    return WorkstreamRegistry(
        root=_root_definition(
            root_id=root_id,
            title="Boundary ingress convergence for CLI/dataflow and CLI repo-state surfaces",
            symbol=_bic_root,
            subqueue_ids=(
                "BIC-SQ-001",
                "BIC-SQ-002",
                "BIC-SQ-003",
            ),
            status_hint="in_progress",
        ),
        subqueues=(
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="BIC-SQ-001",
                title="CLI and runtime dataflow transport ingress convergence",
                symbol=_bic_sq_dataflow_transport_ingress,
                touchpoint_ids=("BIC-TP-001",),
                status_hint="in_progress",
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="BIC-SQ-002",
                title="Server-core coercion carrier convergence",
                symbol=_bic_sq_server_core_coercion,
                touchpoint_ids=("BIC-TP-002", "BIC-TP-003"),
                status_hint="in_progress",
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="BIC-SQ-003",
                title="CLI live-repo sentinel separation",
                symbol=_bic_sq_cli_live_repo_sentinels,
                touchpoint_ids=("BIC-TP-004",),
                status_hint="in_progress",
            ),
        ),
        touchpoints=(
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="BIC-SQ-001",
                touchpoint_id="BIC-TP-001",
                title="Shared CLI/dataflow transport ingress carrier",
                symbol=_bic_tp_dataflow_transport_ingress,
                status_hint="landed",
                declared_touchsites=(
                    _module_touchsite(
                        touchsite_id="BIC-TS-001-A",
                        rel_path="src/gabion/cli_support/shared/dataflow_transport_ingress.py",
                        qualname="dataflow_transport_ingress",
                    ),
                    _module_touchsite(
                        touchsite_id="BIC-TS-001-B",
                        rel_path="src/gabion/cli.py",
                        qualname="cli",
                    ),
                    _module_touchsite(
                        touchsite_id="BIC-TS-001-C",
                        rel_path="src/gabion/tooling/runtime/dataflow_invocation_runner.py",
                        qualname="dataflow_invocation_runner",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="BIC-SQ-002",
                touchpoint_id="BIC-TP-002",
                title="Canonical server-core coercion carrier",
                symbol=_bic_tp_server_core_coercion_extract,
                status_hint="landed",
                declared_touchsites=(
                    _module_touchsite(
                        touchsite_id="BIC-TS-002-A",
                        rel_path="src/gabion/server_core/coercion_contract.py",
                        qualname="coercion_contract",
                    ),
                    _module_touchsite(
                        touchsite_id="BIC-TS-002-B",
                        rel_path="src/gabion/server_core/command_orchestrator.py",
                        qualname="command_orchestrator",
                    ),
                    _module_touchsite(
                        touchsite_id="BIC-TS-002-C",
                        rel_path="src/gabion/server_core/command_orchestrator_progress.py",
                        qualname="command_orchestrator_progress",
                    ),
                    _module_touchsite(
                        touchsite_id="BIC-TS-002-D",
                        rel_path="tests/gabion/runtime/test_coercion_contract.py",
                        qualname="test_coercion_contract",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="BIC-SQ-002",
                touchpoint_id="BIC-TP-003",
                title="Downstream server-core migration to the shared coercion carrier",
                symbol=_bic_tp_server_core_coercion_migrate,
                status_hint="landed",
                declared_touchsites=(
                    _module_touchsite(
                        touchsite_id="BIC-TS-003-A",
                        rel_path="src/gabion/server_core/command_orchestrator_primitives.py",
                        qualname="command_orchestrator_primitives",
                    ),
                    _module_touchsite(
                        touchsite_id="BIC-TS-003-B",
                        rel_path="src/gabion/server_core/server_payload_dispatch.py",
                        qualname="server_payload_dispatch",
                    ),
                    _module_touchsite(
                        touchsite_id="BIC-TS-003-C",
                        rel_path="tests/gabion/runtime/test_coercion_contract.py",
                        qualname="test_coercion_contract",
                    ),
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="BIC-SQ-003",
                touchpoint_id="BIC-TP-004",
                title="Dedicated CLI live-repo sentinel surface",
                symbol=_bic_tp_cli_live_repo_split,
                status_hint="queued",
                declared_touchsites=(
                    _module_touchsite(
                        touchsite_id="BIC-TS-004-A",
                        rel_path="tests/gabion/cli/cli_commands_cases.py",
                        qualname="cli_commands_cases",
                    ),
                    _module_touchsite(
                        touchsite_id="BIC-TS-004-B",
                        rel_path="tests/gabion/cli/cli_live_repo_cases.py",
                        qualname="cli_live_repo_cases",
                    ),
                    _module_touchsite(
                        touchsite_id="BIC-TS-004-C",
                        rel_path="tests/gabion/cli/test_cli.py",
                        qualname="test_cli",
                    ),
                    _module_touchsite(
                        touchsite_id="BIC-TS-004-D",
                        rel_path="tests/gabion/cli/test_cli_live_repo.py",
                        qualname="test_cli_live_repo",
                    ),
                ),
            ),
        ),
        tags=("boundary_ingress_convergence",),
    )


__all__ = ["boundary_ingress_convergence_workstream_registry"]
