from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass(frozen=True)
class ScheduleResult:
    order: List[str] = field(default_factory=list)
    cycles: List[Set[str]] = field(default_factory=list)


def topological_schedule(graph: Dict[str, Set[str]]) -> ScheduleResult:
    nodes: Set[str] = set(graph.keys())
    for deps in graph.values():
        nodes.update(deps)

    incoming: Dict[str, Set[str]] = {node: set() for node in nodes}
    outgoing: Dict[str, Set[str]] = {node: set() for node in nodes}

    for node, deps in graph.items():
        for dep in deps:
            outgoing[dep].add(node)
            incoming[node].add(dep)

    ready = sorted(node for node, deps in incoming.items() if not deps)
    order: List[str] = []

    while ready:
        node = ready.pop(0)
        order.append(node)
        for follower in sorted(outgoing[node]):
            incoming[follower].discard(node)
            if not incoming[follower]:
                if follower not in ready and follower not in order:
                    ready.append(follower)
        ready.sort()

    remaining = {node for node, deps in incoming.items() if deps}
    cycles: List[Set[str]] = []
    if remaining:
        cycles.append(remaining)
    return ScheduleResult(order=order, cycles=cycles)
