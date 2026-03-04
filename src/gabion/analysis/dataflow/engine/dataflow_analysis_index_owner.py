# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

"""Analysis-index owner surface during WS-5 migration."""

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.indexed_scan.scanners.edge_param_events import (
    iter_resolved_edge_param_events as _iter_resolved_edge_param_events_impl,
)


def _runtime_module():
    from gabion.analysis.dataflow.engine import dataflow_indexed_file_scan as _runtime

    return _runtime


def _collect_transitive_callers(
    callers_by_qual: dict[str, set[str]],
    by_qual: dict[str, object],
) -> dict[str, set[str]]:
    check_deadline()
    transitive: dict[str, set[str]] = {}
    for qual in by_qual:
        check_deadline()
        seen: set[str] = set()
        stack = list(callers_by_qual.get(qual, set()))
        while stack:
            check_deadline()
            caller = stack.pop()
            if caller in seen:
                continue
            seen.add(caller)
            stack.extend(callers_by_qual.get(caller, set()))
        transitive[qual] = seen
    return transitive


def _analysis_index_transitive_callers(
    analysis_index,
    *,
    project_root,
) -> dict[str, set[str]]:
    check_deadline()
    if analysis_index.transitive_callers is not None:
        return analysis_index.transitive_callers
    callers_by_qual: dict[str, set[str]] = defaultdict(set)
    for edge in _analysis_index_resolved_call_edges(
        analysis_index,
        project_root=project_root,
        require_transparent=False,
    ):
        check_deadline()
        callers_by_qual[edge.callee.qual].add(edge.caller.qual)
    analysis_index.transitive_callers = _collect_transitive_callers(
        callers_by_qual,
        analysis_index.by_qual,
    )
    return analysis_index.transitive_callers


def _analysis_index_resolved_call_edges(
    analysis_index,
    *,
    project_root,
    require_transparent: bool,
) -> tuple[object, ...]:
    runtime = _runtime_module()
    check_deadline()
    if require_transparent:
        cached_edges = analysis_index.resolved_transparent_call_edges
    else:
        cached_edges = analysis_index.resolved_call_edges
    if cached_edges is not None:
        return cached_edges
    edges: list[object] = []
    for infos in analysis_index.by_name.values():
        check_deadline()
        for info in infos:
            check_deadline()
            for call in info.calls:
                check_deadline()
                if not call.is_test:
                    callee = runtime._resolve_callee(
                        call.callee,
                        info,
                        analysis_index.by_name,
                        analysis_index.by_qual,
                        analysis_index.symbol_table,
                        project_root,
                        analysis_index.class_index,
                    )
                    if callee is not None and (not require_transparent or callee.transparent):
                        edges.append(runtime._ResolvedCallEdge(caller=info, call=call, callee=callee))
    frozen_edges = tuple(edges)
    if require_transparent:
        analysis_index.resolved_transparent_call_edges = frozen_edges
    else:
        analysis_index.resolved_call_edges = frozen_edges
    return frozen_edges


def _analysis_index_resolved_call_edges_by_caller(
    analysis_index,
    *,
    project_root,
    require_transparent: bool,
) -> dict[str, tuple[object, ...]]:
    check_deadline()
    if require_transparent and analysis_index.resolved_transparent_edges_by_caller is not None:
        return analysis_index.resolved_transparent_edges_by_caller
    grouped: dict[str, list[object]] = defaultdict(list)
    for edge in _analysis_index_resolved_call_edges(
        analysis_index,
        project_root=project_root,
        require_transparent=require_transparent,
    ):
        check_deadline()
        grouped[edge.caller.qual].append(edge)
    frozen_grouped = {qual: tuple(edges) for qual, edges in grouped.items()}
    if require_transparent:
        analysis_index.resolved_transparent_edges_by_caller = frozen_grouped
    return frozen_grouped


def _reduce_resolved_call_edges(
    analysis_index,
    *,
    project_root,
    require_transparent: bool,
    spec,
):
    check_deadline()
    acc = spec.init()
    for edge in _analysis_index_resolved_call_edges(
        analysis_index,
        project_root=project_root,
        require_transparent=require_transparent,
    ):
        check_deadline()
        spec.fold(acc, edge)
    return spec.finish(acc)


def _iter_resolved_edge_param_events(
    edge,
    *,
    strictness: str,
    include_variadics_in_low_star: bool,
):
    runtime = _runtime_module()
    yield from _iter_resolved_edge_param_events_impl(
        edge=edge,
        strictness=strictness,
        include_variadics_in_low_star=include_variadics_in_low_star,
        check_deadline_fn=check_deadline,
        event_ctor=runtime._ResolvedEdgeParamEvent,
    )


def _build_call_graph(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators=None,
    parse_failure_witnesses: list[dict[str, object]],
    analysis_index=None,
) -> tuple[dict[str, list[object]], dict[str, object], dict[str, set[str]]]:
    runtime = _runtime_module()
    check_deadline()
    index = analysis_index
    if index is None:
        index = runtime._build_analysis_index(
            list(paths),
            project_root=project_root,
            ignore_params=set(ignore_params),
            strictness=strictness,
            external_filter=external_filter,
            transparent_decorators=transparent_decorators,
            parse_failure_witnesses=list(parse_failure_witnesses),
        )
    transitive_callers = _analysis_index_transitive_callers(
        index,
        project_root=project_root,
    )
    return index.by_name, index.by_qual, transitive_callers


def _analyze_file_internal(path, *, recursive, config, resume_state, on_progress, on_profile):
    runtime = _runtime_module()
    return runtime._analyze_file_internal(
        path,
        recursive=recursive,
        config=config,
        resume_state=resume_state,
        on_progress=on_progress,
        on_profile=on_profile,
    )


def _build_analysis_collection_resume_payload(
    *,
    groups_by_path,
    param_spans_by_path,
    bundle_sites_by_path,
    invariant_propositions,
    completed_paths,
    in_progress_scan_by_path,
    analysis_index_resume,
    file_stage_timings_v1_by_path,
):
    runtime = _runtime_module()
    return runtime._build_analysis_collection_resume_payload(
        groups_by_path=groups_by_path,
        param_spans_by_path=param_spans_by_path,
        bundle_sites_by_path=bundle_sites_by_path,
        invariant_propositions=invariant_propositions,
        completed_paths=completed_paths,
        in_progress_scan_by_path=in_progress_scan_by_path,
        analysis_index_resume=analysis_index_resume,
        file_stage_timings_v1_by_path=file_stage_timings_v1_by_path,
    )


def _build_analysis_index(
    paths,
    *,
    project_root,
    ignore_params,
    strictness,
    external_filter,
    transparent_decorators,
    parse_failure_witnesses,
    resume_payload=None,
    on_progress=None,
    forest_spec_id=None,
    fingerprint_seed_revision=None,
    decision_ignore_params=None,
    decision_require_tiers=False,
):
    runtime = _runtime_module()
    return runtime._build_analysis_index(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        resume_payload=resume_payload,
        on_progress=on_progress,
        forest_spec_id=forest_spec_id,
        fingerprint_seed_revision=fingerprint_seed_revision,
        decision_ignore_params=decision_ignore_params,
        decision_require_tiers=decision_require_tiers,
    )


def _load_analysis_collection_resume_payload(
    *,
    payload,
    file_paths,
    include_invariant_propositions,
):
    runtime = _runtime_module()
    return runtime._load_analysis_collection_resume_payload(
        payload=payload,
        file_paths=file_paths,
        include_invariant_propositions=include_invariant_propositions,
    )


def _run_indexed_pass(*args, **kwargs):
    runtime = _runtime_module()
    return runtime._run_indexed_pass(*args, **kwargs)


__all__ = [
    "_analysis_index_resolved_call_edges",
    "_analysis_index_resolved_call_edges_by_caller",
    "_analysis_index_transitive_callers",
    "_analyze_file_internal",
    "_build_analysis_collection_resume_payload",
    "_build_analysis_index",
    "_build_call_graph",
    "_collect_transitive_callers",
    "_iter_resolved_edge_param_events",
    "_load_analysis_collection_resume_payload",
    "_reduce_resolved_call_edges",
    "_run_indexed_pass",
]
