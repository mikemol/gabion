from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from gabion.analysis.aspf import Alt, Forest, NodeId
from gabion.json_types import JSONValue
from gabion.analysis.timeout_context import check_deadline
from gabion.invariants import never


@dataclass(frozen=True)
class ForestSignature:
    version: int
    nodes: dict[str, JSONValue]
    alts: dict[str, JSONValue]

    def as_dict(self) -> dict[str, JSONValue]:
        return {
            "version": self.version,
            "nodes": self.nodes,
            "alts": self.alts,
        }


def build_forest_signature(forest: Forest) -> dict[str, JSONValue]:
    return build_forest_signature_payload(forest)


def build_forest_signature_payload(
    forest: Forest,
    *,
    include_legacy_intern: bool = True,
    include_fingerprint_intern: bool = False,
) -> dict[str, JSONValue]:
    check_deadline()
    nodes = sorted(forest.nodes.keys(), key=lambda node_id: node_id.sort_key())
    node_intern: list[list[JSONValue]] = []
    fingerprint_intern: list[list[JSONValue]] = []
    node_index: dict[NodeId, int] = {}
    for idx, node_id in enumerate(nodes):
        check_deadline()
        node_index[node_id] = idx
        if include_legacy_intern:
            node_intern.append([node_id.kind, _normalize_key(node_id.key)])
        if include_fingerprint_intern:
            fingerprint_kind, fingerprint_key = node_id.fingerprint()
            fingerprint_intern.append([fingerprint_kind, list(fingerprint_key)])

    alt_kinds = sorted({alt.kind for alt in forest.alts})
    alt_kind_index = {kind: idx for idx, kind in enumerate(alt_kinds)}
    alt_edges: list[list[JSONValue]] = []
    alts_sorted = sorted(forest.alts, key=lambda alt: _alt_sort_key(alt, node_index))
    for alt in alts_sorted:
        check_deadline()
        kind_idx = alt_kind_index[alt.kind]
        inputs = [node_index[node_id] for node_id in alt.inputs]
        alt_edges.append([kind_idx, inputs])

    nodes_payload: dict[str, JSONValue] = {
        "count": len(nodes),
    }
    if include_legacy_intern:
        nodes_payload["intern"] = node_intern
    if include_fingerprint_intern:
        nodes_payload["intern_fingerprint"] = fingerprint_intern

    signature = ForestSignature(
        version=1,
        nodes=nodes_payload,
        alts={
            "kinds": alt_kinds,
            "edges": alt_edges,
            "count": len(alt_edges),
        },
    )
    return signature.as_dict()


def build_forest_signature_from_groups(
    groups_by_path: dict[object, dict[str, list[set[str]]]]
) -> dict[str, JSONValue]:
    check_deadline()
    forest = Forest()
    previous_path_key: str | None = None
    for path in groups_by_path:
        check_deadline()
        path_key = str(path)
        if previous_path_key is not None and previous_path_key > path_key:
            never(
                "groups_by_path path order regression",
                previous_path=previous_path_key,
                current_path=path_key,
            )
        previous_path_key = path_key
        groups = groups_by_path[path]
        path_name = _path_name(path)
        for fn_name in sorted(groups):
            check_deadline()
            site_id = forest.add_site(path_name, fn_name)
            for bundle in groups[fn_name]:
                check_deadline()
                paramset_id = forest.add_paramset(bundle)
                forest.add_alt("SignatureBundle", (site_id, paramset_id))
    return build_forest_signature(forest)


def _alt_sort_key(alt: Alt, node_index: dict[NodeId, int]) -> tuple:
    return (
        alt.kind,
        tuple(node_index[node_id] for node_id in alt.inputs),
    )


def _normalize_key(parts: Iterable[object]) -> list[JSONValue]:
    check_deadline()
    normalized: list[JSONValue] = []
    for part in parts:
        check_deadline()
        if isinstance(part, (str, int, float, bool)) or part is None:
            normalized.append(part)
        else:
            normalized.append(str(part))
    return normalized


def _path_name(path: object) -> str:
    name = getattr(path, "name", None)
    if isinstance(name, str):
        return name
    return str(path)
