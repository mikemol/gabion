from __future__ import annotations

from collections.abc import Callable, Mapping

from gabion.analysis.aspf import Forest, NodeId
from gabion.analysis.canonical import canon, digest_index, encode_canon
from gabion.analysis.determinism_invariants import (
    require_canonical_multiset,
    require_no_dupes,
    require_sorted,
)
from gabion.analysis.projection_normalize import spec_hash as projection_spec_hash
from gabion.analysis.projection_spec import ProjectionSpec
from gabion.analysis.timeout_context import check_deadline
from gabion.json_types import JSONValue
from gabion.order_contract import ordered_or_sorted


def emit_wl_refinement_facets(
    *,
    forest: Forest,
    spec: ProjectionSpec,
    scope: set[NodeId] | None = None,
    canon_fn: Callable[[object], JSONValue] = canon,
) -> None:
    check_deadline()
    target_kind = _string_param(spec.params, "target_kind", "SuiteSite")
    edge_alt_kinds = _string_list_param(
        spec.params,
        "edge_alt_kinds",
        default=("SuiteContains",),
    )
    direction = _string_param(spec.params, "direction", "undirected")
    seed_fields = _string_list_param(spec.params, "seed_fields", default=("suite_kind",))
    steps = max(1, _int_param(spec.params, "steps", 2))
    stabilize_early = _bool_param(spec.params, "stabilize_early", True)
    emit_steps = _string_param(spec.params, "emit_steps", "final")
    label_namespace = _string_param(spec.params, "label_namespace", "wl")
    require_injective = _bool_param(spec.params, "require_injective_on_scope", False)

    raw_targets = [
        node_id
        for node_id in forest.nodes
        if node_id.kind == target_kind and (scope is None or node_id in scope)
    ]
    if not raw_targets:
        return
    spec_identity = projection_spec_hash(spec)
    spec_site = forest.add_spec_site(
        spec_hash=spec_identity,
        spec_name=spec.name,
        spec_domain=spec.domain,
        spec_version=spec.spec_version,
    )

    def _on_determinism_violation(payload: dict[str, object]) -> None:
        check_deadline()
        sink_suite = forest.add_suite_site(
            "projection_spec",
            spec.name,
            "wl_determinism",
        )
        paramset = forest.add_paramset([str(payload.get("constraint", "determinism"))])
        evidence: dict[str, object] = {
            "spec_name": spec.name,
            "spec_hash": spec_identity,
            "spec_domain": spec.domain,
            "spec_version": spec.spec_version,
        }
        evidence.update(payload)
        forest.add_alt("NeverInvariantSink", (sink_suite, paramset), evidence=evidence)
        forest.add_alt("SpecFacet", (spec_site, sink_suite), evidence=evidence)

    require_sorted(
        "wl.target_nodes",
        raw_targets,
        key=lambda node_id: node_id.sort_key(),
        on_violation=_on_determinism_violation,
        spec_name=spec.name,
    )
    require_no_dupes(
        "wl.target_nodes",
        raw_targets,
        key=lambda node_id: node_id.sort_key(),
        on_violation=_on_determinism_violation,
        spec_name=spec.name,
    )
    target_nodes = ordered_or_sorted(
        raw_targets,
        source="emit_wl_refinement_facets.target_nodes",
        key=lambda node_id: node_id.sort_key(),
    )

    adjacency: dict[NodeId, list[NodeId]] = {node_id: [] for node_id in target_nodes}
    for alt in forest.alts:
        check_deadline()
        if alt.kind not in edge_alt_kinds or len(alt.inputs) < 2:
            continue
        parent, child = alt.inputs[0], alt.inputs[1]
        if parent not in adjacency or child not in adjacency:
            continue
        adjacency[parent].append(child)
        if direction == "undirected":
            adjacency[child].append(parent)

    labels_by_step: list[dict[NodeId, JSONValue]] = []
    labels_prev: dict[NodeId, JSONValue] = {}
    for node_id in target_nodes:
        check_deadline()
        seed_struct = _seed_struct(
            node_id=node_id,
            forest=forest,
            seed_fields=seed_fields,
            degree=len(adjacency[node_id]),
            canon_fn=canon_fn,
        )
        labels_prev[node_id] = canon_fn(
            [
                label_namespace,
                {"step": -1},
                seed_struct,
                ["ms", []],
            ]
        )

    last_step_index = 0
    for step_index in range(steps):
        check_deadline()
        labels_next: dict[NodeId, JSONValue] = {}
        for node_id in target_nodes:
            check_deadline()
            raw_neighbors = list(adjacency[node_id])
            require_sorted(
                "wl.neighbors",
                raw_neighbors,
                key=lambda item: item.sort_key(),
                on_violation=_on_determinism_violation,
                spec_name=spec.name,
                node=node_id.sort_key(),
            )
            require_no_dupes(
                "wl.neighbors",
                raw_neighbors,
                key=lambda item: item.sort_key(),
                on_violation=_on_determinism_violation,
                spec_name=spec.name,
                node=node_id.sort_key(),
            )
            neighbors = ordered_or_sorted(
                raw_neighbors,
                source="emit_wl_refinement_facets.neighbors",
                key=lambda item: item.sort_key(),
            )
            seed_struct = _seed_struct(
                node_id=node_id,
                forest=forest,
                seed_fields=seed_fields,
                degree=len(neighbors),
                canon_fn=canon_fn,
            )
            counts: dict[str, tuple[JSONValue, int]] = {}
            for neighbor in neighbors:
                check_deadline()
                label = labels_prev[neighbor]
                label_index = encode_canon(label)
                previous = counts.get(label_index)
                if previous is None:
                    counts[label_index] = (label, 1)
                else:
                    counts[label_index] = (previous[0], previous[1] + 1)
            ordered_multiset_pairs = ordered_or_sorted(
                counts.items(),
                source="emit_wl_refinement_facets.multiset_pairs",
                key=lambda item: item[0],
            )
            multiset_pairs = [(entry[0], entry[1][1]) for entry in ordered_multiset_pairs]
            require_canonical_multiset(
                "wl.multiset",
                multiset_pairs,
                on_violation=_on_determinism_violation,
                spec_name=spec.name,
                node=node_id.sort_key(),
            )
            label_struct = canon_fn(
                [
                    label_namespace,
                    {"step": step_index},
                    seed_struct,
                    [
                        "ms",
                        [[entry[1][0], int(entry[1][1])] for entry in ordered_multiset_pairs],
                    ],
                ]
            )
            labels_next[node_id] = label_struct
        labels_by_step.append(labels_next)
        last_step_index = step_index
        if stabilize_early and labels_next == labels_prev:
            break
        labels_prev = labels_next

    if emit_steps == "all":
        emit_indices = list(range(len(labels_by_step)))
    else:
        emit_indices = [last_step_index]

    for step_index in emit_indices:
        check_deadline()
        step_labels = labels_by_step[step_index]
        class_members: dict[str, list[NodeId]] = {}
        for node_id in target_nodes:
            check_deadline()
            label_index = encode_canon(step_labels[node_id])
            class_members.setdefault(label_index, []).append(node_id)
        ordered_class_indices = ordered_or_sorted(
            class_members,
            source="emit_wl_refinement_facets.class_indices",
        )
        if require_injective:
            require_no_dupes(
                "wl.injective_scope",
                ordered_class_indices,
                on_violation=_on_determinism_violation,
                spec_name=spec.name,
                step=step_index,
            )
        class_rank = {label_index: rank for rank, label_index in enumerate(ordered_class_indices)}
        for node_id in target_nodes:
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
                "spec_name": spec.name,
                "spec_hash": spec_identity,
                "wl_step": step_index,
                "wl_label_rank": class_rank[label_key],
                "wl_class_size": len(class_members[label_key]),
                "wl_degree": len(adjacency[node_id]),
                "wl_label_index": digest_index(label_struct),
            }
            forest.add_alt("SpecFacet", (spec_site, node_id, label_node), evidence=evidence)


def _seed_struct(
    *,
    node_id: NodeId,
    forest: Forest,
    seed_fields: tuple[str, ...],
    degree: int,
    canon_fn: Callable[[object], JSONValue] = canon,
) -> JSONValue:
    node = forest.nodes.get(node_id)
    if node is None:
        return {}
    seed: dict[str, JSONValue] = {}
    for field in seed_fields:
        check_deadline()
        if field == "degree":
            seed[field] = int(degree)
            continue
        value = node.meta.get(field)
        if value is None or isinstance(value, (str, int, float, bool)):
            seed[field] = value
            continue
        if isinstance(value, Mapping):
            seed[field] = canon_fn(value)
            continue
        if isinstance(value, list):
            seed[field] = canon_fn(value)
            continue
        seed[field] = str(value)
    return canon_fn(seed)


def _string_param(params: Mapping[str, JSONValue], name: str, default: str) -> str:
    value = params.get(name, default)
    text = str(value or default).strip()
    return text or default


def _int_param(params: Mapping[str, JSONValue], name: str, default: int) -> int:
    value = params.get(name, default)
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _bool_param(params: Mapping[str, JSONValue], name: str, default: bool) -> bool:
    value = params.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _string_list_param(
    params: Mapping[str, JSONValue],
    name: str,
    *,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    raw = params.get(name)
    if isinstance(raw, list):
        values = [str(item).strip() for item in raw if str(item).strip()]
        if values:
            return tuple(values)
    return default
