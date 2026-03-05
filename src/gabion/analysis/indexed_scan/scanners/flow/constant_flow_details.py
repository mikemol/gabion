# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True)
class CollectConstantFlowDetailsDeps:
    check_deadline_fn: Callable[[], None]
    require_not_none_fn: Callable[..., object]
    resolved_edge_reducer_spec_ctor: Callable[..., object]
    constant_flow_fold_accumulator_ctor: Callable[..., object]
    format_call_site_fn: Callable[..., str]
    function_key_fn: Callable[..., str]
    sort_once_fn: Callable[..., list[object]]
    constant_flow_detail_ctor: Callable[..., object]


def collect_constant_flow_details(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators=None,
    parse_failure_witnesses: list[object],
    analysis_index=None,
    iter_resolved_edge_param_events_fn: Callable[..., Iterable[object]],
    reduce_resolved_call_edges_fn: Callable[..., object],
    deps: CollectConstantFlowDetailsDeps,
) -> list[object]:
    deps.check_deadline_fn()
    index = deps.require_not_none_fn(
        analysis_index,
        reason="_collect_constant_flow_details requires prebuilt analysis_index",
        strict=True,
    )
    by_qual = index.by_qual

    def _fold(acc, edge) -> None:
        for event in iter_resolved_edge_param_events_fn(
            edge,
            strictness=strictness,
            include_variadics_in_low_star=False,
        ):
            deps.check_deadline_fn()
            key = (edge.callee.qual, event.param)
            if event.kind == "const":
                if event.value is not None:
                    acc.const_values[key].add(event.value)
                    if event.countable:
                        acc.call_counts[key] += 1
                        acc.call_sites[key].add(
                            deps.format_call_site_fn(edge.caller, edge.call)
                        )
            else:
                acc.non_const[key] = True
                if event.countable:
                    acc.call_counts[key] += 1

    folded = reduce_resolved_call_edges_fn(
        index,
        project_root=project_root,
        require_transparent=True,
        spec=deps.resolved_edge_reducer_spec_ctor(
            reducer_id="constant_flow",
            init=deps.constant_flow_fold_accumulator_ctor,
            fold=_fold,
            finish=lambda acc: acc,
        ),
    )

    details: list[object] = []
    for key, values in folded.const_values.items():
        deps.check_deadline_fn()
        if folded.non_const.get(key):
            continue
        if len(values) != 1:
            continue
        qual, param = key
        info = by_qual.get(qual)
        path = info.path if info is not None else Path(qual)
        name = (
            deps.function_key_fn(info.scope, info.name)
            if info is not None
            else qual.split(".")[-1]
        )
        count = folded.call_counts.get(key, 0)
        details.append(
            deps.constant_flow_detail_ctor(
                path=path,
                qual=qual,
                name=name,
                param=param,
                value=next(iter(values)),
                count=count,
                sites=tuple(
                    deps.sort_once_fn(
                        folded.call_sites.get(key, set()),
                        source=(
                            "gabion.analysis.dataflow_indexed_file_scan."
                            "_collect_constant_flow_details.site_1"
                        ),
                    )
                ),
            )
        )
    return deps.sort_once_fn(
        details,
        key=lambda entry: (str(entry.path), entry.name, entry.param),
        source=(
            "gabion.analysis.dataflow_indexed_file_scan."
            "_collect_constant_flow_details.site_2"
        ),
    )

