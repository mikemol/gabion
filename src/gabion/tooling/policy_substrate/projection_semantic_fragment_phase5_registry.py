from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Callable

from gabion.analysis.aspf.aspf_lattice_algebra import canonical_structural_identity
from gabion.analysis.foundation.marker_protocol import MarkerPayload, marker_identity
from gabion.tooling.policy_substrate.site_identity import canonical_site_identity
from gabion.invariants import invariant_decorations, todo_decorator

_REPO_ROOT = Path(__file__).resolve().parents[4]



@dataclass(frozen=True)
class ProjectionSemanticFragmentPhase5QueueDefinition:
    queue_id: str
    title: str
    rel_path: str
    qualname: str
    line: int
    site_identity: str
    structural_identity: str
    marker_identity: str
    marker_payload: MarkerPayload
    subqueue_ids: tuple[str, ...]


@dataclass(frozen=True)
class ProjectionSemanticFragmentPhase5SubqueueDefinition:
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


@dataclass(frozen=True)
class ProjectionSemanticFragmentPhase5TouchpointDefinition:
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
    collapse_private_helpers: bool
    surviving_boundary_names: tuple[str, ...]


def _registry_site_metadata(symbol: Callable[..., object]) -> tuple[str, str, int]:
    source_path = Path(inspect.getsourcefile(symbol) or __file__).resolve()
    rel_path = str(source_path.relative_to(_REPO_ROOT))
    start_line = int(inspect.getsourcelines(symbol)[1])
    return (rel_path, str(symbol.__qualname__), start_line)


def _todo_metadata(
    symbol: Callable[..., object],
    *,
    surface: str,
    structural_path: str,
) -> tuple[MarkerPayload, str, str, str, str, str, int]:
    decorations = invariant_decorations(symbol)
    if len(decorations) != 1:
        raise ValueError(
            "Phase-5 queue registry symbols must carry exactly one invariant decoration."
        )
    payload = decorations[0]
    if payload.marker_kind.value != "todo":
        raise ValueError("Phase-5 queue registry symbols must use todo_decorator.")
    rel_path, qualname, start_line = _registry_site_metadata(symbol)
    site_id = canonical_site_identity(
        rel_path=rel_path,
        qualname=qualname,
        line=start_line,
        column=1,
        node_kind="function_def",
        surface=surface,
    )
    structural_id = canonical_structural_identity(
        rel_path=rel_path,
        qualname=qualname,
        structural_path=structural_path,
        node_kind="function_def",
        surface=surface,
    )
    return (
        payload,
        marker_identity(payload),
        site_id,
        structural_id,
        rel_path,
        qualname,
        start_line,
    )


@todo_decorator(
    reason="PSF-007 queue remains active until all Phase-5 adapter seams are retired.",
    reasoning={
        "summary": "PSF-007 remains the active Phase-5 cutover queue until all subqueues land.",
        "control": "psf007.queue.phase5_cutover",
        "blocking_dependencies": (
            "PSF-007-SQ-001",
            "PSF-007-SQ-002",
            "PSF-007-SQ-003",
            "PSF-007-SQ-004",
            "PSF-007-SQ-005",
        ),
    },
    owner="gabion.analysis.projection",
    expiry="PSF-007 closure",
    links=[
        {"kind": "object_id", "value": "PSF-007"},
        {"kind": "doc_id", "value": "projection_semantic_fragment_ledger"},
        {"kind": "doc_id", "value": "projection_semantic_fragment_rfc"},
    ],
)
def _psf_007_queue() -> None:
    return None


@todo_decorator(
    reason="PSF-007 semantic row closure and canonicalization subqueue remains active.",
    reasoning={
        "summary": "Semantic row closure and canonicalization still retain temporary Phase-5 adapter seams.",
        "control": "psf007.subqueue.semantic_row_canonicalization",
        "blocking_dependencies": ("PSF-007-TP-001",),
    },
    owner="gabion.analysis.projection",
    expiry="PSF-007 closure",
    links=[
        {"kind": "object_id", "value": "PSF-007"},
        {"kind": "object_id", "value": "PSF-007-SQ-001"},
        {"kind": "doc_id", "value": "projection_semantic_fragment_rfc"},
    ],
)
def _psf_007_sq_semantic_row_canonicalization() -> None:
    return None


@todo_decorator(
    reason="PSF-007 lowering normalization subqueue remains active.",
    reasoning={
        "summary": "Authoring-to-semantic lowering still retains temporary normalization and payload-shaping seams.",
        "control": "psf007.subqueue.lowering_normalization",
        "blocking_dependencies": ("PSF-007-TP-002",),
    },
    owner="gabion.analysis.projection",
    expiry="PSF-007 closure",
    links=[
        {"kind": "object_id", "value": "PSF-007"},
        {"kind": "object_id", "value": "PSF-007-SQ-002"},
        {"kind": "doc_id", "value": "projection_semantic_fragment_rfc"},
    ],
)
def _psf_007_sq_lowering_normalization() -> None:
    return None


@todo_decorator(
    reason="PSF-007 lowering-to-compile dispatch subqueue remains active.",
    reasoning={
        "summary": "Semantic-op compilation dispatch still retains temporary adapter seams around surface and field selection.",
        "control": "psf007.subqueue.lowering_compile_dispatch",
        "blocking_dependencies": ("PSF-007-TP-003",),
    },
    owner="gabion.analysis.projection",
    expiry="PSF-007 closure",
    links=[
        {"kind": "object_id", "value": "PSF-007"},
        {"kind": "object_id", "value": "PSF-007-SQ-003"},
        {"kind": "doc_id", "value": "projection_semantic_fragment_rfc"},
    ],
)
def _psf_007_sq_lowering_compile_dispatch() -> None:
    return None


@todo_decorator(
    reason="PSF-007 semantic-row presentation compilation subqueue remains active.",
    reasoning={
        "summary": "Semantic-row compilation into SHACL/SPARQL plans still retains temporary adapter seams on the compile path.",
        "control": "psf007.subqueue.semantic_row_compile",
        "blocking_dependencies": ("PSF-007-TP-004",),
    },
    owner="gabion.analysis.projection",
    expiry="PSF-007 closure",
    links=[
        {"kind": "object_id", "value": "PSF-007"},
        {"kind": "object_id", "value": "PSF-007-SQ-004"},
        {"kind": "doc_id", "value": "projection_semantic_fragment_rfc"},
    ],
)
def _psf_007_sq_semantic_row_compile() -> None:
    return None


@todo_decorator(
    reason="PSF-007 typed execution planning/runtime subqueue remains active.",
    reasoning={
        "summary": "Typed execution planning and runtime application still retain temporary adapter seams on the final compatibility path.",
        "control": "psf007.subqueue.typed_execution",
        "blocking_dependencies": (
            "PSF-007-TP-005",
            "PSF-007-TP-006",
        ),
    },
    owner="gabion.analysis.projection",
    expiry="PSF-007 closure",
    links=[
        {"kind": "object_id", "value": "PSF-007"},
        {"kind": "object_id", "value": "PSF-007-SQ-005"},
        {"kind": "doc_id", "value": "projection_semantic_fragment_rfc"},
    ],
)
def _psf_007_sq_typed_execution() -> None:
    return None


@todo_decorator(
    reason="PSF-007 touchpoint for semantic_fragment.py remains active.",
    reasoning={
        "summary": "semantic_fragment.py still carries the canonicalization and semantic-row closure touchsites for PSF-007.",
        "control": "psf007.touchpoint.semantic_fragment",
        "blocking_dependencies": ("PSF-007-SQ-001",),
    },
    owner="gabion.analysis.projection",
    expiry="PSF-007 closure",
    links=[
        {"kind": "object_id", "value": "PSF-007"},
        {"kind": "object_id", "value": "PSF-007-SQ-001"},
        {"kind": "object_id", "value": "PSF-007-TP-001"},
        {"kind": "doc_id", "value": "projection_semantic_fragment_rfc"},
    ],
)
def _psf_007_tp_semantic_fragment() -> None:
    return None


@todo_decorator(
    reason="PSF-007 touchpoint for projection_semantic_lowering.py remains active.",
    reasoning={
        "summary": "projection_semantic_lowering.py still carries the authoring-to-semantic normalization touchsites for PSF-007.",
        "control": "psf007.touchpoint.projection_semantic_lowering",
        "blocking_dependencies": ("PSF-007-SQ-002",),
    },
    owner="gabion.analysis.projection",
    expiry="PSF-007 closure",
    links=[
        {"kind": "object_id", "value": "PSF-007"},
        {"kind": "object_id", "value": "PSF-007-SQ-002"},
        {"kind": "object_id", "value": "PSF-007-TP-002"},
        {"kind": "doc_id", "value": "projection_semantic_fragment_rfc"},
    ],
)
def _psf_007_tp_projection_semantic_lowering() -> None:
    return None


@todo_decorator(
    reason="PSF-007 touchpoint for projection_semantic_lowering_compile.py remains active.",
    reasoning={
        "summary": "projection_semantic_lowering_compile.py still carries the lowering-to-compile dispatch touchsites for PSF-007.",
        "control": "psf007.touchpoint.projection_semantic_lowering_compile",
        "blocking_dependencies": ("PSF-007-SQ-003",),
    },
    owner="gabion.analysis.projection",
    expiry="PSF-007 closure",
    links=[
        {"kind": "object_id", "value": "PSF-007"},
        {"kind": "object_id", "value": "PSF-007-SQ-003"},
        {"kind": "object_id", "value": "PSF-007-TP-003"},
        {"kind": "doc_id", "value": "projection_semantic_fragment_rfc"},
    ],
)
def _psf_007_tp_projection_semantic_lowering_compile() -> None:
    return None


@todo_decorator(
    reason="PSF-007 touchpoint for semantic_fragment_compile.py remains active.",
    reasoning={
        "summary": "semantic_fragment_compile.py still carries the semantic-row presentation compile touchsites for PSF-007.",
        "control": "psf007.touchpoint.semantic_fragment_compile",
        "blocking_dependencies": ("PSF-007-SQ-004",),
    },
    owner="gabion.analysis.projection",
    expiry="PSF-007 closure",
    links=[
        {"kind": "object_id", "value": "PSF-007"},
        {"kind": "object_id", "value": "PSF-007-SQ-004"},
        {"kind": "object_id", "value": "PSF-007-TP-004"},
        {"kind": "doc_id", "value": "projection_semantic_fragment_rfc"},
    ],
)
def _psf_007_tp_semantic_fragment_compile() -> None:
    return None


@todo_decorator(
    reason="PSF-007 touchpoint for projection_exec_plan.py remains active.",
    reasoning={
        "summary": "projection_exec_plan.py still carries the typed execution planning touchsites for PSF-007.",
        "control": "psf007.touchpoint.projection_exec_plan",
        "blocking_dependencies": ("PSF-007-SQ-005",),
    },
    owner="gabion.analysis.projection",
    expiry="PSF-007 closure",
    links=[
        {"kind": "object_id", "value": "PSF-007"},
        {"kind": "object_id", "value": "PSF-007-SQ-005"},
        {"kind": "object_id", "value": "PSF-007-TP-005"},
        {"kind": "doc_id", "value": "projection_semantic_fragment_rfc"},
    ],
)
def _psf_007_tp_projection_exec_plan() -> None:
    return None


@todo_decorator(
    reason="PSF-007 touchpoint for projection_exec.py remains active.",
    reasoning={
        "summary": "projection_exec.py still carries the typed execution runtime touchsites for PSF-007.",
        "control": "psf007.touchpoint.projection_exec",
        "blocking_dependencies": ("PSF-007-SQ-005",),
    },
    owner="gabion.analysis.projection",
    expiry="PSF-007 closure",
    links=[
        {"kind": "object_id", "value": "PSF-007"},
        {"kind": "object_id", "value": "PSF-007-SQ-005"},
        {"kind": "object_id", "value": "PSF-007-TP-006"},
        {"kind": "doc_id", "value": "projection_semantic_fragment_rfc"},
    ],
)
def _psf_007_tp_projection_exec() -> None:
    return None


def iter_phase5_queues() -> tuple[ProjectionSemanticFragmentPhase5QueueDefinition, ...]:
    payload, marker_id, site_id, structural_id, rel_path, qualname, line = _todo_metadata(
        _psf_007_queue,
        surface="projection_semantic_fragment_phase5_queue",
        structural_path="psf007.queue::PSF-007",
    )
    return (
        ProjectionSemanticFragmentPhase5QueueDefinition(
            queue_id="PSF-007",
            title="Cut over legacy adapters and retire semantic_carrier_adapter boundaries",
            rel_path=rel_path,
            qualname=qualname,
            line=line,
            site_identity=site_id,
            structural_identity=structural_id,
            marker_identity=marker_id,
            marker_payload=payload,
            subqueue_ids=(
                "PSF-007-SQ-001",
                "PSF-007-SQ-002",
                "PSF-007-SQ-003",
                "PSF-007-SQ-004",
                "PSF-007-SQ-005",
            ),
        ),
    )


def iter_phase5_subqueues() -> tuple[ProjectionSemanticFragmentPhase5SubqueueDefinition, ...]:
    definitions: list[ProjectionSemanticFragmentPhase5SubqueueDefinition] = []
    for subqueue_id, title, symbol, touchpoint_ids in (
        (
            "PSF-007-SQ-001",
            "Semantic row closure and canonicalization",
            _psf_007_sq_semantic_row_canonicalization,
            ("PSF-007-TP-001",),
        ),
        (
            "PSF-007-SQ-002",
            "Authoring-to-semantic lowering normalization",
            _psf_007_sq_lowering_normalization,
            ("PSF-007-TP-002",),
        ),
        (
            "PSF-007-SQ-003",
            "Semantic-op to compiled-plan dispatch",
            _psf_007_sq_lowering_compile_dispatch,
            ("PSF-007-TP-003",),
        ),
        (
            "PSF-007-SQ-004",
            "Semantic-row to presentation-plan compilation",
            _psf_007_sq_semantic_row_compile,
            ("PSF-007-TP-004",),
        ),
        (
            "PSF-007-SQ-005",
            "Typed execution planning/runtime",
            _psf_007_sq_typed_execution,
            ("PSF-007-TP-005", "PSF-007-TP-006"),
        ),
    ):
        (
            payload,
            marker_id,
            site_id,
            structural_id,
            rel_path,
            qualname,
            line,
        ) = _todo_metadata(
            symbol,
            surface="projection_semantic_fragment_phase5_subqueue",
            structural_path=f"psf007.subqueue::{subqueue_id}",
        )
        definitions.append(
            ProjectionSemanticFragmentPhase5SubqueueDefinition(
                subqueue_id=subqueue_id,
                title=title,
                rel_path=rel_path,
                qualname=qualname,
                line=line,
                site_identity=site_id,
                structural_identity=structural_id,
                marker_identity=marker_id,
                marker_payload=payload,
                touchpoint_ids=touchpoint_ids,
            )
        )
    return tuple(definitions)


def iter_phase5_touchpoints() -> tuple[ProjectionSemanticFragmentPhase5TouchpointDefinition, ...]:
    definitions: list[ProjectionSemanticFragmentPhase5TouchpointDefinition] = []
    for touchpoint_id, subqueue_id, title, rel_path, symbol in (
        (
            "PSF-007-TP-001",
            "PSF-007-SQ-001",
            "semantic_fragment.py canonicalization surfaces",
            "src/gabion/analysis/projection/semantic_fragment.py",
            _psf_007_tp_semantic_fragment,
        ),
        (
            "PSF-007-TP-002",
            "PSF-007-SQ-002",
            "projection_semantic_lowering.py normalization surfaces",
            "src/gabion/analysis/projection/projection_semantic_lowering.py",
            _psf_007_tp_projection_semantic_lowering,
        ),
        (
            "PSF-007-TP-003",
            "PSF-007-SQ-003",
            "projection_semantic_lowering_compile.py dispatch surfaces",
            "src/gabion/analysis/projection/projection_semantic_lowering_compile.py",
            _psf_007_tp_projection_semantic_lowering_compile,
        ),
        (
            "PSF-007-TP-004",
            "PSF-007-SQ-004",
            "semantic_fragment_compile.py presentation compile surfaces",
            "src/gabion/analysis/projection/semantic_fragment_compile.py",
            _psf_007_tp_semantic_fragment_compile,
        ),
        (
            "PSF-007-TP-005",
            "PSF-007-SQ-005",
            "projection_exec_plan.py planning surfaces",
            "src/gabion/analysis/projection/projection_exec_plan.py",
            _psf_007_tp_projection_exec_plan,
        ),
        (
            "PSF-007-TP-006",
            "PSF-007-SQ-005",
            "projection_exec.py runtime surfaces",
            "src/gabion/analysis/projection/projection_exec.py",
            _psf_007_tp_projection_exec,
        ),
    ):
        surviving_boundary_names: tuple[str, ...]
        if touchpoint_id == "PSF-007-TP-001":
            surviving_boundary_names = (
                "semantic_fragment.normalize_value",
                "semantic_fragment.stable_json_key",
            )
        elif touchpoint_id == "PSF-007-TP-002":
            surviving_boundary_names = (
                "projection_semantic_lowering.normalize_projection_op",
                "projection_semantic_lowering.lower_projection_op",
            )
        elif touchpoint_id == "PSF-007-TP-003":
            surviving_boundary_names = (
                "projection_semantic_lowering_compile.compile_semantic_projection_op",
                "projection_semantic_lowering_compile.semantic_rows_for_quotient_face",
                "projection_semantic_lowering_compile.semantic_rows_for_surface",
            )
        elif touchpoint_id == "PSF-007-TP-006":
            surviving_boundary_names = (
                "projection_exec.apply_execution_op",
                "projection_exec.sort_value",
                "projection_exec.canonical_group_reference",
            )
        else:
            surviving_boundary_names = ()
        (
            payload,
            marker_id,
            site_id,
            structural_id,
            _registry_rel_path,
            qualname,
            line,
        ) = _todo_metadata(
            symbol,
            surface="projection_semantic_fragment_phase5_touchpoint",
            structural_path=f"psf007.touchpoint::{touchpoint_id}",
        )
        definitions.append(
            ProjectionSemanticFragmentPhase5TouchpointDefinition(
                touchpoint_id=touchpoint_id,
                subqueue_id=subqueue_id,
                title=title,
                rel_path=rel_path,
                qualname=qualname,
                line=line,
                site_identity=site_id,
                structural_identity=structural_id,
                marker_identity=marker_id,
                marker_payload=payload,
                collapse_private_helpers=True,
                surviving_boundary_names=surviving_boundary_names,
            )
        )
    return tuple(definitions)


__all__ = [
    "ProjectionSemanticFragmentPhase5QueueDefinition",
    "ProjectionSemanticFragmentPhase5SubqueueDefinition",
    "ProjectionSemanticFragmentPhase5TouchpointDefinition",
    "iter_phase5_queues",
    "iter_phase5_subqueues",
    "iter_phase5_touchpoints",
]
