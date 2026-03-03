# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CollectCallEdgesDeps:
    check_deadline_fn: Callable[[], None]
    is_test_path_fn: Callable[..., bool]


def collect_call_edges(
    *,
    by_name: dict[str, list[object]],
    by_qual: dict[str, object],
    symbol_table,
    project_root,
    class_index: dict[str, object],
    resolve_callee_outcome_fn: Callable[..., object],
    deps: CollectCallEdgesDeps,
) -> dict[str, set[str]]:
    deps.check_deadline_fn()
    edges: dict[str, set[str]] = defaultdict(set)
    for infos in by_name.values():
        deps.check_deadline_fn()
        for info in infos:
            deps.check_deadline_fn()
            if deps.is_test_path_fn(info.path):
                continue
            for call in info.calls:
                deps.check_deadline_fn()
                if call.is_test:
                    continue
                resolution = resolve_callee_outcome_fn(
                    call.callee,
                    info,
                    by_name,
                    by_qual,
                    symbol_table=symbol_table,
                    project_root=project_root,
                    class_index=class_index,
                    call=call,
                )
                if not resolution.candidates:
                    continue
                for candidate in resolution.candidates:
                    deps.check_deadline_fn()
                    edges[info.qual].add(candidate.qual)
    return edges
