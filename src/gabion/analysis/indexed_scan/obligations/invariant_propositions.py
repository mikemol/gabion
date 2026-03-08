from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True)
class CollectInvariantPropositionsDeps:
    check_deadline_fn: Callable[[], None]
    parse_module_source_fn: Callable[[Path], object]
    collect_functions_fn: Callable[[object], Iterable[object]]
    param_names_fn: Callable[..., list[str]]
    scope_path_fn: Callable[[Path, object], str]
    invariant_collector_ctor: Callable[..., object]
    invariant_proposition_type: type
    normalize_invariant_proposition_fn: Callable[..., object]


def collect_invariant_propositions(
    path: Path,
    *,
    ignore_params: set[str],
    project_root,
    emitters: Iterable[Callable[[object], Iterable[object]]] = (),
    deps: CollectInvariantPropositionsDeps,
) -> list[object]:
    deps.check_deadline_fn()
    tree = deps.parse_module_source_fn(path)
    propositions: list[object] = []
    for fn in deps.collect_functions_fn(tree):
        deps.check_deadline_fn()
        params = set(deps.param_names_fn(fn, ignore_params))
        if not params:
            continue
        scope = f"{deps.scope_path_fn(path, project_root)}:{fn.name}"
        collector = deps.invariant_collector_ctor(params, scope)
        for stmt in fn.body:
            deps.check_deadline_fn()
            collector.visit(stmt)
        propositions.extend(collector.propositions)
        for emitter in emitters:
            deps.check_deadline_fn()
            emitted = emitter(fn)
            for prop in emitted:
                deps.check_deadline_fn()
                if type(prop) is not deps.invariant_proposition_type:
                    raise TypeError(
                        "Invariant emitters must yield InvariantProposition instances."
                    )
                propositions.append(
                    deps.normalize_invariant_proposition_fn(
                        prop,
                        default_scope=scope,
                        default_source="emitter",
                    )
                )
    return propositions
