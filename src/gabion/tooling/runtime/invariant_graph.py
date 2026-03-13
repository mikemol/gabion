from __future__ import annotations

import argparse
from pathlib import Path

from gabion.order_contract import ordered_or_sorted
from gabion.tooling.policy_substrate.invariant_graph import (
    InvariantGraph,
    blocker_chains,
    build_invariant_graph,
    load_invariant_graph,
    trace_nodes,
    write_invariant_graph,
)

_DEFAULT_ARTIFACT = Path("artifacts/out/invariant_graph.json")


def _sorted[T](values: list[T], *, key=None) -> list[T]:
    return ordered_or_sorted(
        values,
        source="tooling.runtime.invariant_graph",
        key=key,
    )


def _load_or_build_graph(*, root: Path, artifact: Path) -> InvariantGraph:
    if artifact.exists():
        return load_invariant_graph(artifact)
    return build_invariant_graph(root)


def _print_summary(*, graph: InvariantGraph) -> None:
    payload = graph.as_payload()
    counts = payload.get("counts", {})
    print(f"root: {graph.root}")
    print(f"nodes: {counts.get('node_count', 0)}")
    print(f"edges: {counts.get('edge_count', 0)}")
    print(f"diagnostics: {counts.get('diagnostic_count', 0)}")
    node_kind_counts = counts.get("node_kind_counts", {})
    if isinstance(node_kind_counts, dict):
        print("node kinds:")
        for key, value in _sorted(
            [(str(key), int(value)) for key, value in node_kind_counts.items()],
            key=lambda item: item[0],
        ):
            print(f"- {key}: {value}")
    edge_kind_counts = counts.get("edge_kind_counts", {})
    if isinstance(edge_kind_counts, dict):
        print("edge kinds:")
        for key, value in _sorted(
            [(str(key), int(value)) for key, value in edge_kind_counts.items()],
            key=lambda item: item[0],
        ):
            print(f"- {key}: {value}")


def _ownership_chain(graph: InvariantGraph, node_id: str) -> tuple[str, ...]:
    node_by_id = graph.node_by_id()
    edges_to = graph.edges_to()
    chain: list[str] = []
    current = node_id
    seen: set[str] = set()
    while current not in seen:
        seen.add(current)
        chain.append(current)
        parents = [
            edge.source_id
            for edge in edges_to.get(current, ())
            if edge.edge_kind == "contains" and edge.source_id in node_by_id
        ]
        if not parents:
            break
        current = _sorted(parents)[0]
    return tuple(reversed(chain))


def _print_trace(*, graph: InvariantGraph, raw_id: str) -> int:
    nodes = trace_nodes(graph, raw_id)
    if not nodes:
        print(f"no invariant-graph nodes matched: {raw_id}")
        return 1
    node_by_id = graph.node_by_id()
    edges_from = graph.edges_from()
    edges_to = graph.edges_to()
    for index, node in enumerate(nodes, start=1):
        if index > 1:
            print("")
        print(f"node: {node.node_id}")
        print(f"kind: {node.node_kind}")
        print(f"title: {node.title}")
        if node.marker_id:
            print(f"marker_id: {node.marker_id}")
        if node.object_ids:
            print(f"object_ids: {', '.join(node.object_ids)}")
        if node.doc_ids:
            print(f"doc_ids: {', '.join(node.doc_ids)}")
        if node.policy_ids:
            print(f"policy_ids: {', '.join(node.policy_ids)}")
        if node.invariant_ids:
            print(f"invariant_ids: {', '.join(node.invariant_ids)}")
        print(f"site_identity: {node.site_identity}")
        print(f"structural_identity: {node.structural_identity}")
        if node.rel_path:
            print(
                f"location: {node.rel_path}:{node.line}:{node.column} "
                f"({node.qualname or '<none>'})"
            )
        if node.reasoning_control:
            print(f"reasoning.control: {node.reasoning_control}")
        if node.blocking_dependencies:
            print(
                "blocking_dependencies: "
                + ", ".join(node.blocking_dependencies)
            )
        chain = _ownership_chain(graph, node.node_id)
        if chain:
            print("ownership_chain:")
            for value in chain:
                owner = node_by_id.get(value)
                print(f"- {value} :: {owner.title if owner is not None else '<missing>'}")
        outbound = [
            edge for edge in edges_from.get(node.node_id, ()) if edge.target_id in node_by_id
        ]
        inbound = [
            edge for edge in edges_to.get(node.node_id, ()) if edge.source_id in node_by_id
        ]
        print("outbound_edges:")
        for edge in outbound:
            target = node_by_id[edge.target_id]
            print(f"- {edge.edge_kind}: {target.node_id} :: {target.title}")
        if not outbound:
            print("- <none>")
        print("inbound_edges:")
        for edge in inbound:
            source = node_by_id[edge.source_id]
            print(f"- {edge.edge_kind}: {source.node_id} :: {source.title}")
        if not inbound:
            print("- <none>")
    return 0


def _print_blockers(*, graph: InvariantGraph, object_id: str) -> int:
    grouped = blocker_chains(graph, object_id=object_id)
    if not grouped:
        print(f"no blocker chains for object_id: {object_id}")
        return 1
    node_by_id = graph.node_by_id()
    for seam_class, chains in _sorted(list(grouped.items()), key=lambda item: item[0]):
        print(f"{seam_class}:")
        for chain in chains:
            rendered = " -> ".join(
                f"{node_id} ({node_by_id[node_id].title})"
                for node_id in chain
                if node_id in node_by_id
            )
            print(f"- {rendered}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--artifact", default=str(_DEFAULT_ARTIFACT))
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build")
    subparsers.add_parser("summary")

    trace_parser = subparsers.add_parser("trace")
    trace_parser.add_argument("--id", required=True)

    blockers_parser = subparsers.add_parser("blockers")
    blockers_parser.add_argument("--object-id", required=True)

    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    artifact = Path(args.artifact).resolve()

    if args.command == "build":
        graph = build_invariant_graph(root)
        write_invariant_graph(artifact, graph)
        print(str(artifact))
        return 0
    if args.command == "summary":
        _print_summary(graph=_load_or_build_graph(root=root, artifact=artifact))
        return 0
    if args.command == "trace":
        return _print_trace(
            graph=_load_or_build_graph(root=root, artifact=artifact),
            raw_id=str(args.id),
        )
    if args.command == "blockers":
        return _print_blockers(
            graph=_load_or_build_graph(root=root, artifact=artifact),
            object_id=str(args.object_id),
        )
    return 1


__all__ = ["main"]
