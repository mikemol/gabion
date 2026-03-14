from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from pathlib import Path
from typing import Callable

from gabion.analysis.aspf.aspf_lattice_algebra import canonical_structural_identity
from gabion.analysis.foundation.marker_protocol import MarkerPayload, marker_identity
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
    root: RegisteredRootDefinition
    subqueues: tuple[RegisteredSubqueueDefinition, ...]
    touchpoints: tuple[RegisteredTouchpointDefinition, ...] = ()
    tags: tuple[str, ...] = field(default_factory=tuple)


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
    "WorkstreamRegistry",
    "WorkstreamRegistryMarkerMetadata",
    "declared_touchsite_definition",
    "declared_touchsite_definition_from_symbol",
    "registry_marker_metadata",
]
