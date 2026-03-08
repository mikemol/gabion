from __future__ import annotations

"""Canonical call-graph algorithm owners for indexed dataflow analysis."""

from collections import deque
from collections.abc import Hashable, Iterable, Mapping
from typing import TypeVar

from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.order_contract import sort_once


_GraphNode = TypeVar("_GraphNode", bound=Hashable)


def _sorted_graph_nodes(
    nodes: Iterable[_GraphNode],
) -> list[_GraphNode]:
    try:
        return sort_once(
            nodes,
            source="dataflow_call_graph_algorithms._sorted_graph_nodes.site_1",
        )
    except TypeError:
        return sort_once(
            nodes,
            key=lambda item: repr(item),
            source="dataflow_call_graph_algorithms._sorted_graph_nodes.site_2",
        )


def _collect_recursive_nodes(
    edges: Mapping[_GraphNode, set[_GraphNode]],
) -> set[_GraphNode]:
    check_deadline()
    index = 0
    stack: list[_GraphNode] = []
    on_stack: set[_GraphNode] = set()
    indices: dict[_GraphNode, int] = {}
    lowlink: dict[_GraphNode, int] = {}
    recursive: set[_GraphNode] = set()

    def _strongconnect(node: _GraphNode) -> None:
        check_deadline()
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for succ in edges.get(node, set()):
            check_deadline()
            if succ not in indices:
                _strongconnect(succ)
                lowlink[node] = min(lowlink[node], lowlink.get(succ, lowlink[node]))
            elif succ in on_stack:
                lowlink[node] = min(lowlink[node], indices.get(succ, lowlink[node]))
        if lowlink.get(node) == indices.get(node):
            scc: list[_GraphNode] = []
            while True:
                check_deadline()
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w == node:
                    break
            if len(scc) > 1:
                recursive.update(scc)
            else:
                if node in edges.get(node, set()):
                    recursive.add(node)

    for node in edges:
        check_deadline()
        if node not in indices:
            _strongconnect(node)
    return recursive


def _collect_recursive_functions(edges: Mapping[str, set[str]]) -> set[str]:
    return _collect_recursive_nodes(edges)


def _reachable_from_roots(
    edges: Mapping[_GraphNode, set[_GraphNode]],
    roots: set[_GraphNode],
) -> set[_GraphNode]:
    check_deadline()
    reachable: set[_GraphNode] = set()
    queue: deque[_GraphNode] = deque(_sorted_graph_nodes(roots))
    while queue:
        check_deadline()
        node = queue.popleft()
        if node in reachable:
            continue
        reachable.add(node)
        for succ in _sorted_graph_nodes(edges.get(node, set())):
            check_deadline()
            if succ not in reachable:
                queue.append(succ)
    return reachable


__all__ = [
    "_collect_recursive_functions",
    "_collect_recursive_nodes",
    "_reachable_from_roots",
]
