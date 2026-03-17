from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from pathlib import Path
from typing import Callable

from gabion.analysis.aspf.aspf_lattice_algebra import canonical_structural_identity
from gabion.analysis.foundation.marker_protocol import (
    MarkerLifecycleState,
    MarkerPayload,
    marker_identity,
)
from gabion.tooling.policy_substrate.site_identity import canonical_site_identity
from gabion.invariants import invariant_decorations


_REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class WorkstreamRegistryMarkerMetadata:
    marker_payload: MarkerPayload
    marker_identity: str
    site_identity: str
    structural_identity: str
    rel_path: str
    qualname: str
    line: int


@dataclass(frozen=True)
class RegisteredTouchsiteDefinition:
    touchsite_id: str
    rel_path: str
    qualname: str
    boundary_name: str
    line: int
    column: int
    node_kind: str
    site_identity: str
    structural_identity: str
    seam_class: str
    status_hint: str = ""
    object_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegisteredCounterfactualActionDefinition:
    action_id: str
    title: str
    action_kind: str
    target_boundary_name: str = ""
    predicted_readiness_class: str = ""
    predicted_touchsite_delta: int = 0
    predicted_surviving_touchsite_delta: int = 0
    predicted_policy_signal_delta: int = 0
    predicted_diagnostic_delta: int = 0
    predicted_coverage_delta: int = 0
    score: int = 0
    rationale: str = ""
    object_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegisteredTouchpointDefinition:
    root_id: str
    touchpoint_id: str
    subqueue_id: str
    title: str
    rel_path: str
    qualname: str
    line: int
    site_identity: str
    structural_identity: str
    marker_identity: str
    marker_payload: MarkerPayload
    status_hint: str = ""
    collapse_private_helpers: bool = False
    surviving_boundary_names: tuple[str, ...] = ()
    declared_counterfactual_actions: tuple[
        RegisteredCounterfactualActionDefinition, ...
    ] = ()
    declared_touchsites: tuple[RegisteredTouchsiteDefinition, ...] = ()
    test_path_prefixes: tuple[str, ...] = ()
    scan_touchsites: bool = False


@dataclass(frozen=True)
class RegisteredSubqueueDefinition:
    root_id: str
    subqueue_id: str
    title: str
    rel_path: str
    qualname: str
    line: int
    site_identity: str
    structural_identity: str
    marker_identity: str
    marker_payload: MarkerPayload
    touchpoint_ids: tuple[str, ...]
    status_hint: str = ""


@dataclass(frozen=True)
class RegisteredRootDefinition:
    root_id: str
    title: str
    rel_path: str
    qualname: str
    line: int
    site_identity: str
    structural_identity: str
    marker_identity: str
    marker_payload: MarkerPayload
    subqueue_ids: tuple[str, ...]
    status_hint: str = ""


@dataclass(frozen=True)
class WorkstreamRegistry:
    # Registry packets remain single-root declarations; planner queues are a
    # separate overlay runtime distinction constructed in the planning substrate.
    root: RegisteredRootDefinition
    subqueues: tuple[RegisteredSubqueueDefinition, ...]
    touchpoints: tuple[RegisteredTouchpointDefinition, ...] = ()
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class WorkstreamClosureViolation:
    object_id: str
    node_kind: str
    code: str
    message: str
    rel_path: str
    qualname: str


_LANDED_REASON_REQUIRED_TOKENS = (
    "landed",
    "recorded",
    "closed",
    "converged",
    "completed",
)
_LANDED_REASON_FORBIDDEN_TOKENS = (
    "remains active until",
    "still drift",
    "still drifts",
    "still leak",
    "still leaks",
    "still split",
    "still carries",
    "still carry",
    "still depends",
    "still depend",
    "still reaches",
    "still reach",
    "still binds",
    "still dominate",
    "still dominates",
    "still treats",
    "still emits",
    "still resolve",
    "still resolves",
)


def registry_marker_metadata(
    symbol: Callable[..., object],
    *,
    surface: str,
    structural_path: str,
) -> WorkstreamRegistryMarkerMetadata:
    decorations = invariant_decorations(symbol)
    if len(decorations) != 1:
        raise ValueError(
            "Workstream registry symbols must carry exactly one invariant decoration."
        )
    payload = decorations[0]
    if payload.marker_kind.value != "todo":
        raise ValueError("Workstream registry symbols must use todo_decorator.")
    source_path = Path(inspect.getsourcefile(symbol) or __file__).resolve()
    rel_path = str(source_path.relative_to(_REPO_ROOT))
    qualname = str(symbol.__qualname__)
    line = int(inspect.getsourcelines(symbol)[1])
    return WorkstreamRegistryMarkerMetadata(
        marker_payload=payload,
        marker_identity=marker_identity(payload),
        site_identity=canonical_site_identity(
            rel_path=rel_path,
            qualname=qualname,
            line=line,
            column=1,
            node_kind="function_def",
            surface=surface,
        ),
        structural_identity=canonical_structural_identity(
            rel_path=rel_path,
            qualname=qualname,
            structural_path=structural_path,
            node_kind="function_def",
            surface=surface,
        ),
        rel_path=rel_path,
        qualname=qualname,
        line=line,
    )


def _node_text_for_closure_check(payload: MarkerPayload) -> str:
    return " ".join(
        part.strip().lower()
        for part in (payload.reason, payload.reasoning.summary)
        if part.strip()
    )


def _append_closure_payload_violations(
    *,
    object_id: str,
    node_kind: str,
    status_hint: str,
    marker_payload: MarkerPayload,
    rel_path: str,
    qualname: str,
    violations: list[WorkstreamClosureViolation],
) -> None:
    if status_hint == "landed":
        if marker_payload.lifecycle_state is not MarkerLifecycleState.LANDED:
            violations.append(
                WorkstreamClosureViolation(
                    object_id=object_id,
                    node_kind=node_kind,
                    code="landed_requires_landed_lifecycle",
                    message=(
                        f"{object_id}: landed {node_kind} must use landed marker lifecycle"
                    ),
                    rel_path=rel_path,
                    qualname=qualname,
                )
            )
        if marker_payload.reasoning.blocking_dependencies:
            violations.append(
                WorkstreamClosureViolation(
                    object_id=object_id,
                    node_kind=node_kind,
                    code="landed_forbids_blocking_dependencies",
                    message=(
                        f"{object_id}: landed {node_kind} must not retain blocking dependencies"
                    ),
                    rel_path=rel_path,
                    qualname=qualname,
                )
            )
        closure_text = _node_text_for_closure_check(marker_payload)
        if any(token in closure_text for token in _LANDED_REASON_FORBIDDEN_TOKENS) or not any(
            token in closure_text for token in _LANDED_REASON_REQUIRED_TOKENS
        ):
            violations.append(
                WorkstreamClosureViolation(
                    object_id=object_id,
                    node_kind=node_kind,
                    code="landed_requires_closed_language",
                    message=(
                        f"{object_id}: landed {node_kind} must use recorded/closed language, "
                        "not active-work phrasing"
                    ),
                    rel_path=rel_path,
                    qualname=qualname,
                )
            )
        return
    if marker_payload.lifecycle_state is MarkerLifecycleState.LANDED:
        violations.append(
            WorkstreamClosureViolation(
                object_id=object_id,
                node_kind=node_kind,
                code="nonlanded_forbids_landed_lifecycle",
                message=(
                    f"{object_id}: non-landed {node_kind} must not use landed marker lifecycle"
                ),
                rel_path=rel_path,
                qualname=qualname,
            )
        )


def validate_workstream_closure_consistency(
    registries: tuple[WorkstreamRegistry, ...],
) -> tuple[WorkstreamClosureViolation, ...]:
    violations: list[WorkstreamClosureViolation] = []
    for registry in registries:
        subqueues_by_id = {item.subqueue_id: item for item in registry.subqueues}
        touchpoints_by_subqueue: dict[str, list[RegisteredTouchpointDefinition]] = {}
        for touchpoint in registry.touchpoints:
            touchpoints_by_subqueue.setdefault(touchpoint.subqueue_id, []).append(touchpoint)
            _append_closure_payload_violations(
                object_id=touchpoint.touchpoint_id,
                node_kind="touchpoint",
                status_hint=touchpoint.status_hint,
                marker_payload=touchpoint.marker_payload,
                rel_path=touchpoint.rel_path,
                qualname=touchpoint.qualname,
                violations=violations,
            )
        for subqueue in registry.subqueues:
            _append_closure_payload_violations(
                object_id=subqueue.subqueue_id,
                node_kind="subqueue",
                status_hint=subqueue.status_hint,
                marker_payload=subqueue.marker_payload,
                rel_path=subqueue.rel_path,
                qualname=subqueue.qualname,
                violations=violations,
            )
            if subqueue.status_hint == "landed":
                nonlanded_touchpoints = tuple(
                    touchpoint.touchpoint_id
                    for touchpoint in touchpoints_by_subqueue.get(subqueue.subqueue_id, [])
                    if touchpoint.status_hint != "landed"
                )
                if nonlanded_touchpoints:
                    violations.append(
                        WorkstreamClosureViolation(
                            object_id=subqueue.subqueue_id,
                            node_kind="subqueue",
                            code="landed_parent_has_nonlanded_descendant",
                            message=(
                                f"{subqueue.subqueue_id}: landed subqueue has non-landed touchpoints "
                                f"{nonlanded_touchpoints}"
                            ),
                            rel_path=subqueue.rel_path,
                            qualname=subqueue.qualname,
                        )
                    )
        _append_closure_payload_violations(
            object_id=registry.root.root_id,
            node_kind="root",
            status_hint=registry.root.status_hint,
            marker_payload=registry.root.marker_payload,
            rel_path=registry.root.rel_path,
            qualname=registry.root.qualname,
            violations=violations,
        )
        if registry.root.status_hint == "landed":
            nonlanded_subqueues = tuple(
                subqueue.subqueue_id
                for subqueue in registry.subqueues
                if subqueue.status_hint != "landed"
            )
            if nonlanded_subqueues:
                violations.append(
                    WorkstreamClosureViolation(
                        object_id=registry.root.root_id,
                        node_kind="root",
                        code="landed_parent_has_nonlanded_descendant",
                        message=(
                            f"{registry.root.root_id}: landed root has non-landed subqueues "
                            f"{nonlanded_subqueues}"
                        ),
                        rel_path=registry.root.rel_path,
                        qualname=registry.root.qualname,
                    )
                )
            for subqueue_id in registry.root.subqueue_ids:
                if subqueue_id not in subqueues_by_id:
                    violations.append(
                        WorkstreamClosureViolation(
                            object_id=registry.root.root_id,
                            node_kind="root",
                            code="landed_parent_has_missing_descendant",
                            message=(
                                f"{registry.root.root_id}: landed root references missing subqueue "
                                f"{subqueue_id}"
                            ),
                            rel_path=registry.root.rel_path,
                            qualname=registry.root.qualname,
                        )
                    )
    return tuple(violations)


def declared_touchsite_definition(
    *,
    touchsite_id: str,
    rel_path: str,
    qualname: str,
    boundary_name: str,
    line: int,
    column: int = 1,
    node_kind: str = "function_def",
    surface: str,
    structural_path: str,
    seam_class: str = "surviving_carrier_seam",
    status_hint: str = "",
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
        site_identity=canonical_site_identity(
            rel_path=rel_path,
            qualname=qualname,
            line=line,
            column=column,
            node_kind=node_kind,
            surface=surface,
        ),
        structural_identity=canonical_structural_identity(
            rel_path=rel_path,
            qualname=qualname,
            structural_path=structural_path,
            node_kind=node_kind,
            surface=surface,
        ),
        seam_class=seam_class,
        status_hint=status_hint,
        object_ids=object_ids,
    )


def declared_touchsite_definition_from_symbol(
    symbol: Callable[..., object],
    *,
    touchsite_id: str,
    boundary_name: str,
    surface: str,
    structural_path: str,
    column: int = 1,
    node_kind: str = "function_def",
    seam_class: str = "surviving_carrier_seam",
    status_hint: str = "",
    object_ids: tuple[str, ...] = (),
) -> RegisteredTouchsiteDefinition:
    source_path = Path(inspect.getsourcefile(symbol) or __file__).resolve()
    rel_path = str(source_path.relative_to(_REPO_ROOT))
    qualname = str(symbol.__qualname__)
    line = int(inspect.getsourcelines(symbol)[1])
    return declared_touchsite_definition(
        touchsite_id=touchsite_id,
        rel_path=rel_path,
        qualname=qualname,
        boundary_name=boundary_name,
        line=line,
        column=column,
        node_kind=node_kind,
        surface=surface,
        structural_path=structural_path,
        seam_class=seam_class,
        status_hint=status_hint,
        object_ids=object_ids,
    )


__all__ = [
    "RegisteredCounterfactualActionDefinition",
    "RegisteredRootDefinition",
    "RegisteredSubqueueDefinition",
    "RegisteredTouchpointDefinition",
    "RegisteredTouchsiteDefinition",
    "WorkstreamClosureViolation",
    "WorkstreamRegistry",
    "WorkstreamRegistryMarkerMetadata",
    "declared_touchsite_definition",
    "declared_touchsite_definition_from_symbol",
    "registry_marker_metadata",
    "validate_workstream_closure_consistency",
]
