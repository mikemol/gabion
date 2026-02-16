from __future__ import annotations
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import ordered_or_sorted

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass(frozen=True)
class ScheduleResult:
    order: List[str] = field(default_factory=list)
    cycles: List[Set[str]] = field(default_factory=list)


def topological_schedule(graph: Dict[str, Set[str]]) -> ScheduleResult:
    check_deadline()
    nodes: Set[str] = set(graph.keys())
    for deps in graph.values():
        check_deadline()
        nodes.update(deps)

    incoming: Dict[str, Set[str]] = {node: set() for node in nodes}
    outgoing: Dict[str, Set[str]] = {node: set() for node in nodes}

    for node, deps in graph.items():
        check_deadline()
        for dep in deps:
            check_deadline()
            outgoing[dep].add(node)
            incoming[node].add(dep)

    ready = ordered_or_sorted(
        (node for node, deps in incoming.items() if not deps),
        source="topological_schedule.ready",
    )
    order: List[str] = []

    while ready:
        check_deadline()
        node = ready.pop(0)
        order.append(node)
        for follower in ordered_or_sorted(
            outgoing[node],
            source="topological_schedule.followers",
        ):
            check_deadline()
            incoming[follower].discard(node)
            if not incoming[follower]:
                # A follower can only become ready when its final unresolved
                # dependency is removed in this loop body, so it should not
                # already be present in `ready`/`order` here.
                if follower not in ready and follower not in order:  # pragma: no branch
                    ready.append(follower)
        ready = ordered_or_sorted(
            ready,
            source="topological_schedule.ready_reorder",
        )

    remaining = {node for node, deps in incoming.items() if deps}
    cycles: List[Set[str]] = []
    if remaining:
        subgraph = {node: {dep for dep in graph.get(node, set()) if dep in remaining} for node in remaining}
        cycles = _strongly_connected_components(subgraph)
        cycles = [
            comp
            for comp in cycles
            if len(comp) > 1 or any(node in subgraph.get(node, set()) for node in comp)
        ]
    return ScheduleResult(order=order, cycles=cycles)


def _strongly_connected_components(graph: Dict[str, Set[str]]) -> List[Set[str]]:
    check_deadline()
    index = 0
    indices: Dict[str, int] = {}
    lowlinks: Dict[str, int] = {}
    stack: List[str] = []
    on_stack: Set[str] = set()
    components: List[Set[str]] = []

    def visit(node: str) -> None:
        check_deadline()
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for neighbor in graph.get(node, set()):
            check_deadline()
            if neighbor not in indices:
                visit(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:  # pragma: no branch
                lowlinks[node] = min(lowlinks[node], indices[neighbor])
        if lowlinks[node] == indices[node]:
            component: Set[str] = set()
            while True:
                check_deadline()
                popped = stack.pop()
                on_stack.discard(popped)
                component.add(popped)
                if popped == node:
                    break
            components.append(component)

    for node in graph:
        check_deadline()
        if node not in indices:
            visit(node)

    return components
