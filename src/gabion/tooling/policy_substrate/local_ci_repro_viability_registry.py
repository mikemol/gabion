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
    reason="LCR remains active while local CI reproduction can still fail due to unsatisfied surface, capability, or relation contracts.",
    reasoning={
        "summary": "Local CI reproduction viability remains synthetic current-state work until planner-visible local reproduction failures stop leaking through surfaces, missing capabilities, and unsatisfied relations.",
        "control": "local_ci_repro_viability.root",
        "blocking_dependencies": ("LCR-SQ-001", "LCR-SQ-002", "LCR-SQ-003"),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="LCR closure",
    links=[{"kind": "object_id", "value": "LCR"}],
)
def _lcr_root() -> None:
    return None


@todo_decorator(
    reason="LCR-SQ-001 remains active while local reproduction surfaces can still fail before parity is even satisfiable.",
    reasoning={
        "summary": "Failing local reproduction surfaces remain the first planner-visible signal that a local CI attempt will not satisfy the declared repro contract.",
        "control": "local_ci_repro_viability.surfaces",
        "blocking_dependencies": ("LCR-TP-001",),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="LCR closure",
    links=[
        {"kind": "object_id", "value": "LCR"},
        {"kind": "object_id", "value": "LCR-SQ-001"},
    ],
)
def _lcr_sq_surfaces() -> None:
    return None


@todo_decorator(
    reason="LCR-SQ-002 remains active while local reproduction capabilities can still be missing or unmatched.",
    reasoning={
        "summary": "Missing capability contracts remain the direct explanation for why the local CI repro lane cannot satisfy the declared surface obligations.",
        "control": "local_ci_repro_viability.capabilities",
        "blocking_dependencies": ("LCR-TP-002",),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="LCR closure",
    links=[
        {"kind": "object_id", "value": "LCR"},
        {"kind": "object_id", "value": "LCR-SQ-002"},
    ],
)
def _lcr_sq_capabilities() -> None:
    return None


@todo_decorator(
    reason="LCR-SQ-003 remains active while local-to-workflow reproduction relations can still be unsatisfied.",
    reasoning={
        "summary": "Unsatisfied local reproduction relations remain the direct planner-visible statement that the local lane still cannot reproduce the workflow lane.",
        "control": "local_ci_repro_viability.relations",
        "blocking_dependencies": ("LCR-TP-003",),
    },
    owner="gabion.tooling.runtime_policy",
    expiry="LCR closure",
    links=[
        {"kind": "object_id", "value": "LCR"},
        {"kind": "object_id", "value": "LCR-SQ-003"},
    ],
)
def _lcr_sq_relations() -> None:
    return None


@todo_decorator(
    reason="LCR-TP-001 remains active while local reproduction surfaces can still fail the declared viability contract.",
    reasoning={
        "summary": "Failing local reproduction surfaces should be directly planner-visible instead of remaining buried under the broader connectivity-synergy family only.",
        "control": "local_ci_repro_viability.surfaces.touchpoint",
    },
    owner="gabion.tooling.runtime_policy",
    expiry="LCR closure",
    links=[
        {"kind": "object_id", "value": "LCR"},
        {"kind": "object_id", "value": "LCR-SQ-001"},
        {"kind": "object_id", "value": "LCR-TP-001"},
    ],
)
def _lcr_tp_surfaces() -> None:
    return None


@todo_decorator(
    reason="LCR-TP-002 remains active while required local reproduction capabilities can still be missing or unmatched.",
    reasoning={
        "summary": "Missing local reproduction capabilities remain the direct explanation for why a local CI attempt will fail to satisfy the declared repro contract.",
        "control": "local_ci_repro_viability.capabilities.touchpoint",
    },
    owner="gabion.tooling.runtime_policy",
    expiry="LCR closure",
    links=[
        {"kind": "object_id", "value": "LCR"},
        {"kind": "object_id", "value": "LCR-SQ-002"},
        {"kind": "object_id", "value": "LCR-TP-002"},
    ],
)
def _lcr_tp_capabilities() -> None:
    return None


@todo_decorator(
    reason="LCR-TP-003 remains active while local reproduction relations can still be unsatisfied.",
    reasoning={
        "summary": "Unsatisfied local-to-workflow reproduction relations remain the direct statement that local reproduction viability has not converged.",
        "control": "local_ci_repro_viability.relations.touchpoint",
    },
    owner="gabion.tooling.runtime_policy",
    expiry="LCR closure",
    links=[
        {"kind": "object_id", "value": "LCR"},
        {"kind": "object_id", "value": "LCR-SQ-003"},
        {"kind": "object_id", "value": "LCR-TP-003"},
    ],
)
def _lcr_tp_relations() -> None:
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
        surface="local_ci_repro_viability_root",
        structural_path=f"local_ci_repro_viability.root::{root_id}",
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
        surface="local_ci_repro_viability_subqueue",
        structural_path=f"local_ci_repro_viability.subqueue::{subqueue_id}",
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
        surface="local_ci_repro_viability_touchpoint",
        structural_path=f"local_ci_repro_viability.touchpoint::{touchpoint_id}",
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


def _contract_touchsites(
    *,
    artifact_touchsite_id: str,
    producer_touchsite_id: str,
) -> tuple:
    return (
        declared_touchsite_definition(
            touchsite_id=artifact_touchsite_id,
            rel_path="artifacts/out/local_ci_repro_contract.json",
            qualname="local_ci_repro_contract.json",
            boundary_name="local_ci_repro_contract.json",
            line=1,
            node_kind="artifact_file",
            surface="local_ci_repro_viability",
            structural_path="artifact::artifacts/out/local_ci_repro_contract.json",
        ),
        declared_touchsite_definition(
            touchsite_id=producer_touchsite_id,
            rel_path="scripts/policy/policy_check.py",
            qualname="_write_local_ci_repro_contract_artifact",
            boundary_name="gabion policy check --workflows",
            line=621,
            node_kind="function_def",
            surface="local_ci_repro_viability",
            structural_path="policy_check::_write_local_ci_repro_contract_artifact",
        ),
    )


def local_ci_repro_viability_workstream_registry() -> WorkstreamRegistry:
    root_id = "LCR"
    return WorkstreamRegistry(
        root=_root_definition(
            root_id=root_id,
            title="Local CI Reproduction / Current Viability",
            subqueue_ids=("LCR-SQ-001", "LCR-SQ-002", "LCR-SQ-003"),
            symbol=_lcr_root,
        ),
        subqueues=(
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="LCR-SQ-001",
                title="Local reproduction surface viability",
                touchpoint_ids=("LCR-TP-001",),
                symbol=_lcr_sq_surfaces,
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="LCR-SQ-002",
                title="Local reproduction capability contracts",
                touchpoint_ids=("LCR-TP-002",),
                symbol=_lcr_sq_capabilities,
            ),
            _subqueue_definition(
                root_id=root_id,
                subqueue_id="LCR-SQ-003",
                title="Local-to-workflow reproduction relations",
                touchpoint_ids=("LCR-TP-003",),
                symbol=_lcr_sq_relations,
            ),
        ),
        touchpoints=(
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="LCR-SQ-001",
                touchpoint_id="LCR-TP-001",
                title="Failing local reproduction surfaces",
                symbol=_lcr_tp_surfaces,
                declared_touchsites=_contract_touchsites(
                    artifact_touchsite_id="LCR-TS-001",
                    producer_touchsite_id="LCR-TS-002",
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="LCR-SQ-002",
                touchpoint_id="LCR-TP-002",
                title="Missing or unmatched local reproduction capabilities",
                symbol=_lcr_tp_capabilities,
                declared_touchsites=_contract_touchsites(
                    artifact_touchsite_id="LCR-TS-003",
                    producer_touchsite_id="LCR-TS-004",
                ),
            ),
            _touchpoint_definition(
                root_id=root_id,
                subqueue_id="LCR-SQ-003",
                touchpoint_id="LCR-TP-003",
                title="Unsatisfied local-to-workflow reproduction relations",
                symbol=_lcr_tp_relations,
                declared_touchsites=_contract_touchsites(
                    artifact_touchsite_id="LCR-TS-005",
                    producer_touchsite_id="LCR-TS-006",
                ),
            ),
        ),
        tags=("local_ci_repro_viability",),
    )


__all__ = ["local_ci_repro_viability_workstream_registry"]
