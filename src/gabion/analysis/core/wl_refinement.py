from __future__ import annotations

from gabion.analysis.core.canonical import canon
from gabion.analysis.projection.projection_spec import ProjectionSpec
from gabion.analysis.aspf.aspf import Forest, NodeId
from gabion.json_types import JSONValue


# gabion:grade_boundary kind=semantic_carrier_adapter name=wl_refinement.emit_wl_refinement_facets
def emit_wl_refinement_facets(
    *,
    forest: Forest,
    spec: ProjectionSpec,
    scope: tuple[NodeId, ...] = (),
) -> None:
    from gabion.analysis.core.wl_refinement_ingress import (
        _EmitWLExecution,
        _SkipWLExecution,
        wl_execution_decision,
        emit_wl_execution_bundle,
    )

    decision = wl_execution_decision(forest=forest, spec=spec, scope=scope)
    match decision:
        case _SkipWLExecution():
            return
        case _EmitWLExecution(bundle=bundle):
            emit_wl_execution_bundle(forest=forest, bundle=bundle)


def initial_wl_label(*, label_namespace: str, seed_struct: JSONValue) -> JSONValue:
    return canon([label_namespace, seed_struct, ["ms", []]])


def refined_wl_label(
    *,
    label_namespace: str,
    seed_struct: JSONValue,
    multiset_payload: JSONValue,
) -> JSONValue:
    return canon([label_namespace, seed_struct, multiset_payload])


def emitted_wl_label(
    *,
    label_namespace: str,
    step_index: int,
    seed_struct: JSONValue,
    multiset_payload: JSONValue,
) -> JSONValue:
    return canon([label_namespace, {"step": step_index}, seed_struct, multiset_payload])
