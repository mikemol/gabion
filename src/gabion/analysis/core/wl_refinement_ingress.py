# gabion:ambiguity_boundary_module
# gabion:boundary_normalization_module
# gabion:grade_boundary kind=semantic_carrier_adapter name=wl_refinement_ingress
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Literal

from gabion.analysis.aspf.aspf import Forest, NodeId
from gabion.analysis.core.canonical import digest_index, encode_canon
from gabion.analysis.core.determinism_invariants_ingress import (
    CanonicalMultisetDuplicateKeyViolation,
    CanonicalMultisetInvalidCountViolation,
    CanonicalMultisetInvariantSatisfied,
    CanonicalMultisetOrderViolation,
    NoDupesInvariantSatisfied,
    NoDupesInvariantViolation,
    SortedInvariantSatisfied,
    SortedInvariantViolation,
    canonical_multiset_invariant_outcome,
    no_dupes_invariant_outcome,
    sorted_invariant_outcome,
)
from gabion.analysis.core.wl_refinement import emitted_wl_label, initial_wl_label, refined_wl_label
from gabion.analysis.projection.projection_normalize import spec_hash as projection_spec_hash
from gabion.analysis.projection.projection_spec import ProjectionSpec
from gabion.analysis.foundation.resume_codec import sequence_optional
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.json_types import JSONObject, JSONValue
from gabion.order_contract import sort_once
from gabion.invariants import never


@dataclass(frozen=True)
class _WLRefinementParams:
    target_kind: str
    edge_alt_kinds: tuple[str, ...]
    direction: Literal["undirected", "directed"]
    seed_fields: tuple[str, ...]
    steps: int
    stabilize_early: bool
    emit_steps: Literal["final", "all"]
    label_namespace: str
    require_injective: bool


@dataclass(frozen=True)
class _WLExecutionBundle:
    spec_name: str
    spec_domain: str
    spec_version: int
    spec_identity: str
    spec_site: NodeId
    params: _WLRefinementParams
    target_nodes: tuple[NodeId, ...]
    adjacency: tuple[tuple[NodeId, tuple[NodeId, ...]], ...]
    seed_structs: tuple[tuple[NodeId, JSONValue], ...]


@dataclass(frozen=True)
class _SkipWLExecution:
    pass


@dataclass(frozen=True)
class _EmitWLExecution:
    bundle: _WLExecutionBundle


def wl_execution_decision(
    *,
    forest: Forest,
    spec: ProjectionSpec,
    scope: tuple[NodeId, ...],
) -> _SkipWLExecution | _EmitWLExecution:
    check_deadline()
    params = _normalize_wl_refinement_params(spec.params)
    target_nodes = _target_nodes_for_wl_refinement(
        forest=forest,
        target_kind=params.target_kind,
        scope=scope,
    )
    if not target_nodes:
        return _SkipWLExecution()
    return _EmitWLExecution(
        bundle=_build_wl_execution_bundle(
            forest=forest,
            spec=spec,
            params=params,
            target_nodes=target_nodes,
        )
    )


def emit_wl_execution_bundle(
    *,
    forest: Forest,
    bundle: _WLExecutionBundle,
) -> None:
    check_deadline()
    adjacency = {node_id: neighbors for node_id, neighbors in bundle.adjacency}
    seed_structs = {node_id: seed_struct for node_id, seed_struct in bundle.seed_structs}
    labels_by_step: list[dict[NodeId, JSONValue]] = []
    labels_prev = {
        node_id: initial_wl_label(
            label_namespace=bundle.params.label_namespace,
            seed_struct=seed_structs[node_id],
        )
        for node_id in bundle.target_nodes
    }
    last_step_index = 0
    for step_index in range(bundle.params.steps):
        check_deadline()
        labels_next: dict[NodeId, JSONValue] = {}
        emitted_labels: dict[NodeId, JSONValue] = {}
        for node_id in bundle.target_nodes:
            check_deadline()
            ordered_multiset_pairs = _ordered_multiset_pairs(
                neighbor_labels=tuple(labels_prev[neighbor] for neighbor in adjacency[node_id]),
            )
            _require_canonical_multiset(
                forest=forest,
                bundle=bundle,
                name="wl.multiset",
                pairs=tuple((entry[0], entry[1][1]) for entry in ordered_multiset_pairs),
                node=node_id,
            )
            multiset_payload = [
                "ms",
                [[entry[1][0], int(entry[1][1])] for entry in ordered_multiset_pairs],
            ]
            seed_struct = seed_structs[node_id]
            labels_next[node_id] = refined_wl_label(
                label_namespace=bundle.params.label_namespace,
                seed_struct=seed_struct,
                multiset_payload=multiset_payload,
            )
            emitted_labels[node_id] = emitted_wl_label(
                label_namespace=bundle.params.label_namespace,
                step_index=step_index,
                seed_struct=seed_struct,
                multiset_payload=multiset_payload,
            )
        labels_by_step.append(emitted_labels)
        last_step_index = step_index
        if bundle.params.stabilize_early and labels_next == labels_prev:
            break
        labels_prev = labels_next
    emit_indices = (
        tuple(range(len(labels_by_step)))
        if bundle.params.emit_steps == "all"
        else (last_step_index,)
    )
    for step_index in emit_indices:
        check_deadline()
        step_labels = labels_by_step[step_index]
        class_members = _label_class_members(bundle.target_nodes, step_labels)
        ordered_class_indices = tuple(sort_once(class_members, source="emit_wl_execution_bundle.class_indices"))
        if bundle.params.require_injective:
            _require_no_dupes(
                forest=forest,
                bundle=bundle,
                name="wl.injective_scope",
                values=ordered_class_indices,
                step=step_index,
            )
        class_rank = {
            label_index: rank for rank, label_index in enumerate(ordered_class_indices)
        }
        for node_id in bundle.target_nodes:
            check_deadline()
            label_struct = step_labels[node_id]
            label_key = encode_canon(label_struct)
            label_node = forest.add_node(
                "WLLabel",
                (label_key,),
                meta={
                    "wl_label": label_struct,
                    "wl_step": step_index,
                    "label_index": digest_index(label_struct),
                },
            )
            evidence = {
                "spec_name": bundle.spec_name,
                "spec_hash": bundle.spec_identity,
                "wl_step": step_index,
                "wl_label_rank": class_rank[label_key],
                "wl_class_size": len(class_members[label_key]),
                "wl_degree": len(adjacency[node_id]),
                "wl_label_index": digest_index(label_struct),
            }
            forest.add_alt(
                "SpecFacet",
                (bundle.spec_site, node_id, label_node),
                evidence=evidence,
            )


def _ordered_multiset_pairs(
    *,
    neighbor_labels: tuple[JSONValue, ...],
) -> tuple[tuple[str, tuple[JSONValue, int]], ...]:
    counts: dict[str, tuple[JSONValue, int]] = {}
    for label in neighbor_labels:
        check_deadline()
        label_index = encode_canon(label)
        previous = counts.get(label_index)
        if previous is None:
            counts[label_index] = (label, 1)
        else:
            counts[label_index] = (previous[0], previous[1] + 1)
    return tuple(
        sort_once(
            counts.items(),
            source="emit_wl_execution_bundle.multiset_pairs",
            key=lambda item: item[0],
        )
    )


def _label_class_members(
    target_nodes: tuple[NodeId, ...],
    step_labels: Mapping[NodeId, JSONValue],
) -> dict[str, list[NodeId]]:
    class_members: dict[str, list[NodeId]] = {}
    for node_id in target_nodes:
        check_deadline()
        label_index = encode_canon(step_labels[node_id])
        class_members.setdefault(label_index, []).append(node_id)
    return class_members


def _build_wl_execution_bundle(
    *,
    forest: Forest,
    spec: ProjectionSpec,
    params: _WLRefinementParams,
    target_nodes: tuple[NodeId, ...],
) -> _WLExecutionBundle:
    check_deadline()
    spec_identity = projection_spec_hash(spec)
    spec_site = forest.add_versioned_spec_site(
        spec_hash=spec_identity,
        spec_name=spec.name,
        spec_domain=spec.domain,
        spec_version=spec.spec_version,
    )
    _require_sorted(
        forest=forest,
        spec=spec,
        spec_identity=spec_identity,
        spec_site=spec_site,
        name="wl.target_nodes",
        values=target_nodes,
        key=lambda node_id: node_id.sort_key(),
    )
    _require_no_dupes(
        forest=forest,
        bundle=None,
        spec=spec,
        spec_identity=spec_identity,
        spec_site=spec_site,
        name="wl.target_nodes",
        values=target_nodes,
        key=lambda node_id: node_id.sort_key(),
    )
    adjacency_entries: list[tuple[NodeId, tuple[NodeId, ...]]] = []
    seed_structs: list[tuple[NodeId, JSONValue]] = []
    adjacency_lookup = {node_id: [] for node_id in target_nodes}
    for alt in forest.alts:
        check_deadline()
        if alt.kind not in params.edge_alt_kinds or len(alt.inputs) < 2:
            continue
        parent, child = alt.inputs[0], alt.inputs[1]
        if parent not in adjacency_lookup or child not in adjacency_lookup:
            continue
        adjacency_lookup[parent].append(child)
        if params.direction == "undirected":
            adjacency_lookup[child].append(parent)
    for node_id in target_nodes:
        check_deadline()
        raw_neighbors = tuple(adjacency_lookup[node_id])
        _require_sorted(
            forest=forest,
            spec=spec,
            spec_identity=spec_identity,
            spec_site=spec_site,
            name="wl.neighbors",
            values=raw_neighbors,
            key=lambda item: item.sort_key(),
            node=node_id,
        )
        _require_no_dupes(
            forest=forest,
            bundle=None,
            spec=spec,
            spec_identity=spec_identity,
            spec_site=spec_site,
            name="wl.neighbors",
            values=raw_neighbors,
            key=lambda item: item.sort_key(),
            node=node_id,
        )
        neighbors = tuple(
            sort_once(
                raw_neighbors,
                source="wl_refinement_ingress.neighbors",
                key=lambda item: item.sort_key(),
            )
        )
        adjacency_entries.append((node_id, neighbors))
        seed_structs.append(
            (
                node_id,
                _seed_struct(
                    seed_values=_normalized_seed_values(
                        node_id=node_id,
                        forest=forest,
                        seed_fields=params.seed_fields,
                        degree=len(neighbors),
                    ),
                ),
            )
        )
    return _WLExecutionBundle(
        spec_name=spec.name,
        spec_domain=spec.domain,
        spec_version=spec.spec_version,
        spec_identity=spec_identity,
        spec_site=spec_site,
        params=params,
        target_nodes=target_nodes,
        adjacency=tuple(adjacency_entries),
        seed_structs=tuple(seed_structs),
    )


def _target_nodes_for_wl_refinement(
    *,
    forest: Forest,
    target_kind: str,
    scope: tuple[NodeId, ...],
) -> tuple[NodeId, ...]:
    scope_lookup = set(scope)
    include_all_scope = not scope_lookup
    return tuple(
        sort_once(
            (
                node_id
                for node_id in forest.nodes
                if node_id.kind == target_kind
                and (include_all_scope or node_id in scope_lookup)
            ),
            source="wl_refinement_ingress.target_nodes",
            key=lambda node_id: node_id.sort_key(),
        )
    )


def _sorted_payload(violation: SortedInvariantViolation) -> JSONObject:
    return {
        "constraint": "sorted",
        "name": violation.name,
        "previous_key": violation.previous_key,
        "current_key": violation.current_key,
        "reverse": violation.reverse,
    }


def _no_dupes_payload(violation: NoDupesInvariantViolation) -> JSONObject:
    return {
        "constraint": "no_dupes",
        "name": violation.name,
        "duplicate_key": violation.duplicate_key,
    }


def _canonical_multiset_payload(
    violation: (
        CanonicalMultisetInvalidCountViolation
        | CanonicalMultisetDuplicateKeyViolation
        | CanonicalMultisetOrderViolation
    ),
) -> JSONObject:
    match violation:
        case CanonicalMultisetInvalidCountViolation():
            return {
                "constraint": "canonical_multiset",
                "name": violation.name,
                "invalid_count": violation.invalid_count,
                "key": violation.key,
            }
        case CanonicalMultisetDuplicateKeyViolation():
            return {
                "constraint": "canonical_multiset",
                "name": violation.name,
                "duplicate_key": violation.duplicate_key,
            }
        case CanonicalMultisetOrderViolation():
            return {
                "constraint": "canonical_multiset",
                "name": violation.name,
                "previous_key": violation.previous_key,
                "current_key": violation.current_key,
            }
        case _:
            never("canonical multiset payload builder must receive a violation")


def _emit_wl_determinism_violation(
    *,
    forest: Forest,
    spec_name: str,
    spec_domain: str,
    spec_version: int,
    spec_identity: str,
    spec_site: NodeId,
    payload: JSONObject,
) -> None:
    check_deadline()
    sink_suite = forest.add_suite_site("projection_spec", spec_name, "wl_determinism")
    constraint = payload.get("constraint")
    paramset = forest.add_paramset([str(constraint) if isinstance(constraint, str) else "determinism"])
    evidence: dict[str, object] = {
        "spec_name": spec_name,
        "spec_hash": spec_identity,
        "spec_domain": spec_domain,
        "spec_version": spec_version,
    }
    evidence.update(payload)
    forest.add_alt("NeverInvariantSink", (sink_suite, paramset), evidence=evidence)
    forest.add_alt("SpecFacet", (spec_site, sink_suite), evidence=evidence)


def _require_sorted(
    *,
    forest: Forest,
    spec: ProjectionSpec,
    spec_identity: str,
    spec_site: NodeId,
    name: str,
    values,
    key,
    node: NodeId | None = None,
) -> None:
    outcome = sorted_invariant_outcome(name, values, key=key)
    match outcome:
        case SortedInvariantSatisfied():
            return
        case SortedInvariantViolation() as violation:
            payload = _sorted_payload(violation)
            if node is not None:
                payload["node"] = node.sort_key()
            _emit_wl_determinism_violation(
                forest=forest,
                spec_name=spec.name,
                spec_domain=spec.domain,
                spec_version=spec.spec_version,
                spec_identity=spec_identity,
                spec_site=spec_site,
                payload=payload,
            )
            never("wl refinement determinism violation", **payload)


def _require_no_dupes(
    *,
    forest: Forest,
    name: str,
    values,
    key=lambda item: item,
    bundle: _WLExecutionBundle | None,
    spec: ProjectionSpec | None = None,
    spec_identity: str | None = None,
    spec_site: NodeId | None = None,
    node: NodeId | None = None,
    step: int | None = None,
) -> None:
    outcome = no_dupes_invariant_outcome(name, values, key=key)
    match outcome:
        case NoDupesInvariantSatisfied():
            return
        case NoDupesInvariantViolation() as violation:
            payload = _no_dupes_payload(violation)
            if node is not None:
                payload["node"] = node.sort_key()
            if step is not None:
                payload["step"] = step
            effective_spec_name = bundle.spec_name if bundle is not None else spec.name
            effective_spec_domain = bundle.spec_domain if bundle is not None else spec.domain
            effective_spec_version = bundle.spec_version if bundle is not None else spec.spec_version
            effective_spec_identity = bundle.spec_identity if bundle is not None else spec_identity
            effective_spec_site = bundle.spec_site if bundle is not None else spec_site
            _emit_wl_determinism_violation(
                forest=forest,
                spec_name=effective_spec_name,
                spec_domain=effective_spec_domain,
                spec_version=effective_spec_version,
                spec_identity=effective_spec_identity,
                spec_site=effective_spec_site,
                payload=payload,
            )
            never("wl refinement determinism violation", **payload)


def _require_canonical_multiset(
    *,
    forest: Forest,
    bundle: _WLExecutionBundle,
    name: str,
    pairs: tuple[tuple[str, int], ...],
    node: NodeId,
) -> None:
    outcome = canonical_multiset_invariant_outcome(name, pairs)
    match outcome:
        case CanonicalMultisetInvariantSatisfied():
            return
        case (
            CanonicalMultisetInvalidCountViolation()
            | CanonicalMultisetDuplicateKeyViolation()
            | CanonicalMultisetOrderViolation()
        ) as violation:
            payload = _canonical_multiset_payload(violation)
            payload["node"] = node.sort_key()
            _emit_wl_determinism_violation(
                forest=forest,
                spec_name=bundle.spec_name,
                spec_domain=bundle.spec_domain,
                spec_version=bundle.spec_version,
                spec_identity=bundle.spec_identity,
                spec_site=bundle.spec_site,
                payload=payload,
            )
            never("wl refinement determinism violation", **payload)


def _normalized_seed_values(
    *,
    node_id: NodeId,
    forest: Forest,
    seed_fields: tuple[str, ...],
    degree: int,
) -> tuple[tuple[str, JSONValue], ...]:
    node = forest.nodes.get(node_id)
    if node is None:
        never("wl seed nodes must be normalized to existing forest nodes before seeding")
    seed: list[tuple[str, JSONValue]] = []
    for field in seed_fields:
        check_deadline()
        if field == "degree":
            seed.append((field, degree))
            continue
        seed.append((field, _normalize_seed_json_value(node.meta.get(field))))
    return tuple(seed)


def _seed_struct(
    *,
    seed_values: tuple[tuple[str, JSONValue], ...],
) -> JSONValue:
    from gabion.analysis.core.canonical import canon

    return canon(dict(seed_values))


def _normalize_seed_json_value(value: object) -> JSONValue:
    match value:
        case None | str() | int() | float() | bool():
            return value
        case list() as values:
            return [_normalize_seed_json_value(item) for item in values]
        case Mapping() as mapping:
            return {
                _normalize_seed_key(key): _normalize_seed_json_value(item)
                for key, item in mapping.items()
            }
        case _:
            never("wl seed values must be normalized to JSONValue before seeding")


def _normalize_seed_key(key: object) -> str:
    match key:
        case str() as text:
            return text
        case _:
            never("wl seed mapping keys must be normalized to str before seeding")


def _normalize_wl_refinement_params(params: Mapping[str, JSONValue]) -> _WLRefinementParams:
    target_kind_value = params.get("target_kind", "SuiteSite")
    target_kind = str(target_kind_value or "SuiteSite").strip() or "SuiteSite"

    edge_alt_kinds_payload = sequence_optional(params.get("edge_alt_kinds"))
    edge_alt_kinds = (
        tuple(str(item).strip() for item in edge_alt_kinds_payload if str(item).strip())
        if edge_alt_kinds_payload is not None
        else ()
    )
    if not edge_alt_kinds:
        edge_alt_kinds = ("SuiteContains",)

    direction = (
        "directed"
        if str(params.get("direction", "undirected") or "undirected").strip() == "directed"
        else "undirected"
    )

    seed_fields_payload = sequence_optional(params.get("seed_fields"))
    seed_fields = (
        tuple(str(item).strip() for item in seed_fields_payload if str(item).strip())
        if seed_fields_payload is not None
        else ()
    )
    if not seed_fields:
        seed_fields = ("suite_kind",)

    steps_value = params.get("steps", 2)
    try:
        steps = int(steps_value) if steps_value is not None else 2
    except (TypeError, ValueError):
        steps = 2
    steps = max(1, steps)

    stabilize_early_text = str(params.get("stabilize_early", True)).strip().lower()
    stabilize_early = stabilize_early_text not in {"0", "false", "no", "off"}

    emit_steps = "all" if str(params.get("emit_steps", "final")).strip() == "all" else "final"
    label_namespace = str(params.get("label_namespace", "wl") or "wl").strip() or "wl"
    require_injective_text = str(params.get("require_injective_on_scope", False)).strip().lower()
    require_injective = require_injective_text in {"1", "true", "yes", "on"}

    return _WLRefinementParams(
        target_kind=target_kind,
        edge_alt_kinds=edge_alt_kinds,
        direction=direction,
        seed_fields=seed_fields,
        steps=steps,
        stabilize_early=stabilize_early,
        emit_steps=emit_steps,
        label_namespace=label_namespace,
        require_injective=require_injective,
    )
