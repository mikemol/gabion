# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field


@dataclass
class KnobFlowFoldAccumulator:
    const_values: dict[tuple[str, str], set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    non_const: dict[tuple[str, str], bool] = field(
        default_factory=lambda: defaultdict(bool)
    )
    explicit_passed: dict[tuple[str, str], bool] = field(
        default_factory=lambda: defaultdict(bool)
    )
    call_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass(frozen=True)
class ComputeKnobParamNamesDeps:
    check_deadline_fn: Callable[[], None]
    analysis_index_ctor: Callable[..., object]
    iter_resolved_edge_param_events_fn: Callable[..., object]
    reduce_resolved_call_edges_fn: Callable[..., object]
    resolved_edge_reducer_spec_ctor: Callable[..., object]
    knob_flow_fold_acc_ctor: Callable[[], KnobFlowFoldAccumulator]


def compute_knob_param_names(
    *,
    by_name: Mapping[str, object],
    by_qual: Mapping[str, object],
    symbol_table,
    project_root,
    class_index: Mapping[str, object],
    strictness: str,
    analysis_index = None,
    deps: ComputeKnobParamNamesDeps,
) -> set[str]:
    deps.check_deadline_fn()
    index = analysis_index
    if index is None:
        index = deps.analysis_index_ctor(
            by_name=by_name,
            by_qual=by_qual,
            symbol_table=symbol_table,
            class_index=class_index,
        )

    def _fold(acc: KnobFlowFoldAccumulator, edge) -> None:
        acc.call_counts[edge.callee.qual] += 1
        for event in deps.iter_resolved_edge_param_events_fn(
            edge,
            strictness=strictness,
            include_variadics_in_low_star=True,
        ):
            deps.check_deadline_fn()
            key = (edge.callee.qual, event.param)
            if event.kind == "const":
                if event.value is not None:
                    acc.const_values[key].add(event.value)
            else:
                acc.non_const[key] = True
            acc.explicit_passed[key] = True

    folded = deps.reduce_resolved_call_edges_fn(
        index,
        project_root=project_root,
        require_transparent=True,
        spec=deps.resolved_edge_reducer_spec_ctor(
            reducer_id="knob_flow",
            init=deps.knob_flow_fold_acc_ctor,
            fold=_fold,
            finish=lambda acc: acc,
        ),
    )
    knob_names: set[str] = set()
    for key, values in folded.const_values.items():
        deps.check_deadline_fn()
        if folded.non_const.get(key):
            continue
        if len(values) == 1:
            knob_names.add(key[1])
    for qual, info in by_qual.items():
        deps.check_deadline_fn()
        if folded.call_counts.get(qual, 0) == 0:
            continue
        for param in info.defaults:
            deps.check_deadline_fn()
            if not folded.explicit_passed.get((qual, param), False):
                knob_names.add(param)
    return knob_names
