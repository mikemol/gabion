# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

"""Analysis-index owner surface during WS-5 migration."""

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from gabion.analysis.aspf.aspf import NodeId, structural_key_atom, structural_key_json
from gabion.analysis.core.type_fingerprints import fingerprint_stage_cache_identity
from gabion.analysis.dataflow.engine.dataflow_resume_serialization import (
    _CACHE_IDENTITY_DIGEST_HEX,
    _CACHE_IDENTITY_PREFIX,
)
from gabion.analysis.derivation.derivation_cache import get_global_derivation_cache
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.indexed_scan.index.analysis_index_builder import (
    AnalysisIndexBuildDeps as _AnalysisIndexBuildDeps,
    build_analysis_index as _build_analysis_index_impl,
)
from gabion.analysis.indexed_scan.index.analysis_index_module_trees import (
    AnalysisIndexModuleTreesDeps as _AnalysisIndexModuleTreesDeps,
    analysis_index_module_trees as _analysis_index_module_trees_impl,
)
from gabion.analysis.indexed_scan.index.analysis_index_stage_cache import (
    AnalysisIndexStageCacheDeps as _AnalysisIndexStageCacheDeps,
    analysis_index_stage_cache as _analysis_index_stage_cache_impl,
)
from gabion.analysis.indexed_scan.scanners.edge_param_events import (
    iter_resolved_edge_param_events as _iter_resolved_edge_param_events_impl,
)
from gabion.analysis.indexed_scan.scanners.key_aliases import (
    stage_cache_key_aliases as _stage_cache_key_aliases_impl,
)
from gabion.order_contract import sort_once


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


def _sorted_text(values=None) -> tuple[str, ...]:
    if values is None:
        return ()
    cleaned = {str(value).strip() for value in values if str(value).strip()}
    return tuple(sort_once(cleaned, source="gabion.analysis.dataflow_indexed_file_scan._sorted_text.site_1"))


def _normalize_cache_config(value):
    if type(value) is dict:
        mapping = cast(dict[object, object], value)
        normalized = {
            str(key): _normalize_cache_config(mapping[key])
            for key in sort_once(mapping, source="_normalize_cache_config.mapping")
        }
        return normalized
    if type(value) is list:
        return [_normalize_cache_config(item) for item in value]
    return value


def _canonical_stage_cache_detail(detail) -> str:
    structural_detail = structural_key_atom(
        detail,
        source="gabion.analysis.dataflow_indexed_file_scan._canonical_stage_cache_detail",
    )
    canonical_json = structural_key_json(structural_detail)
    return json.dumps(canonical_json, sort_keys=False, separators=(",", ":"))


def _build_stage_cache_identity_spec(
    *,
    stage: str,
    cache_context,
    config_subset: Mapping[str, object],
):
    runtime = _runtime_module()
    normalized_config = _normalize_cache_config(config_subset)
    return runtime._StageCacheIdentitySpec(
        stage=stage,
        forest_spec_id=str(cache_context.forest_spec_id or ""),
        fingerprint_seed_revision=fingerprint_stage_cache_identity(cache_context.fingerprint_seed_revision),
        normalized_config=normalized_config,
    )


def _canonical_stage_cache_identity(spec) -> str:
    payload: dict[str, object] = {
        "stage": spec.stage,
        "forest_spec_id": spec.forest_spec_id,
        "fingerprint_seed_revision": spec.fingerprint_seed_revision,
        "config_subset": spec.normalized_config,
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"aspf:sha1:{digest}"


def _canonical_cache_identity(
    *,
    stage: str,
    cache_context,
    config_subset: Mapping[str, object],
):
    runtime = _runtime_module()
    spec = _build_stage_cache_identity_spec(
        stage=stage,
        cache_context=cache_context,
        config_subset=config_subset,
    )
    canonical = runtime._CacheIdentity.from_boundary(_canonical_stage_cache_identity(spec))
    if canonical is None:
        runtime.never("failed to construct canonical cache identity", stage=stage)  # pragma: no cover - invariant sink
    return canonical


def _cache_identity_aliases(identity: str) -> tuple[str, ...]:
    runtime = _runtime_module()
    canonical = runtime._CacheIdentity.from_boundary(identity)
    if canonical is None:
        return ("",)
    return (canonical.value,)


def _resume_variant_for_identity(
    variants: Mapping[str, dict[str, object]],
    expected_identity,
):
    direct = variants.get(expected_identity.value)
    if direct is not None:
        return direct
    return None


def _parse_stage_cache_key(
    *,
    stage,
    cache_context,
    config_subset: Mapping[str, object],
    detail,
):
    identity = _canonical_cache_identity(
        stage="parse",
        cache_context=cache_context,
        config_subset=config_subset,
    )
    return NodeId(
        kind="ParseStageCacheIdentity",
        key=(
            stage.value,
            identity.value,
            _canonical_stage_cache_detail(detail),
        ),
    )


def _index_stage_cache_identity(
    *,
    cache_context,
    config_subset: Mapping[str, object],
):
    return _canonical_cache_identity(
        stage="index",
        cache_context=cache_context,
        config_subset=config_subset,
    )


def _projection_stage_cache_identity(
    *,
    cache_context,
    config_subset: Mapping[str, object],
):
    return _canonical_cache_identity(
        stage="projection",
        cache_context=cache_context,
        config_subset=config_subset,
    )


def _stage_cache_key_aliases(key) -> tuple[object, ...]:
    return _stage_cache_key_aliases_impl(
        key,
        cache_identity_aliases_fn=_cache_identity_aliases,
        cache_identity_prefix=_CACHE_IDENTITY_PREFIX,
        cache_identity_digest_hex=_CACHE_IDENTITY_DIGEST_HEX,
        node_id_type=NodeId,
    )


def _get_stage_cache_bucket(
    analysis_index,
    *,
    scoped_cache_key,
) -> dict[Path, object]:
    stage_cache_by_key = analysis_index.stage_cache_by_key
    bucket = stage_cache_by_key.get(scoped_cache_key)
    if bucket is not None:
        return bucket
    for candidate_key in _stage_cache_key_aliases(scoped_cache_key):
        check_deadline()
        if candidate_key == scoped_cache_key:
            continue
        legacy_bucket = stage_cache_by_key.get(candidate_key)
        if legacy_bucket is not None:
            stage_cache_by_key[scoped_cache_key] = legacy_bucket
            return legacy_bucket
    return stage_cache_by_key.setdefault(scoped_cache_key, {})


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
    return cast(
        object,
        _build_analysis_index_impl(
            paths,
            project_root=project_root,
            ignore_params=ignore_params,
            strictness=strictness,
            external_filter=external_filter,
            transparent_decorators=transparent_decorators,
            parse_failure_witnesses=parse_failure_witnesses,
            resume_payload=resume_payload,
            on_progress=on_progress,
            accumulate_function_index_for_tree_fn=None,
            forest_spec_id=forest_spec_id,
            fingerprint_seed_revision=fingerprint_seed_revision,
            decision_ignore_params=decision_ignore_params,
            decision_require_tiers=decision_require_tiers,
            deps=_AnalysisIndexBuildDeps(
                check_deadline_fn=check_deadline,
                accumulate_function_index_for_tree_default_fn=runtime._accumulate_function_index_for_tree,
                sorted_text_fn=_sorted_text,
                cache_context_ctor=runtime._CacheSemanticContext,
                index_stage_cache_identity_fn=_index_stage_cache_identity,
                projection_stage_cache_identity_fn=_projection_stage_cache_identity,
                iter_monotonic_paths_fn=runtime._iter_monotonic_paths,
                load_analysis_index_resume_payload_fn=runtime._load_analysis_index_resume_payload,
                function_index_acc_ctor=runtime._FunctionIndexAccumulator,
                sort_once_fn=sort_once,
                profiling_payload_fn=runtime._profiling_v1_payload,
                serialize_resume_payload_fn=runtime._serialize_analysis_index_resume_payload,
                parse_module_source_fn=runtime._parse_module_source,
                parse_module_error_types=cast(
                    tuple[type[BaseException], ...],
                    runtime._PARSE_MODULE_ERROR_TYPES,
                ),
                record_parse_failure_witness_fn=runtime._record_parse_failure_witness,
                parse_module_stage_function_index=runtime._ParseModuleStage.FUNCTION_INDEX,
                parse_module_stage_symbol_table=runtime._ParseModuleStage.SYMBOL_TABLE,
                parse_module_stage_class_index=runtime._ParseModuleStage.CLASS_INDEX,
                accumulate_symbol_table_for_tree_fn=runtime._accumulate_symbol_table_for_tree,
                accumulate_class_index_for_tree_fn=runtime._accumulate_class_index_for_tree,
                timeout_exceeded_type=runtime.TimeoutExceeded,
                analysis_index_ctor=runtime.AnalysisIndex,
                progress_emit_min_interval_seconds=runtime._PROGRESS_EMIT_MIN_INTERVAL_SECONDS,
            ),
        ),
    )


def _analysis_index_module_trees(
    analysis_index,
    paths,
    *,
    stage,
    parse_failure_witnesses,
):
    runtime = _runtime_module()
    return cast(
        dict[Path, object | None],
        _analysis_index_module_trees_impl(
            analysis_index,
            paths,
            stage=stage,
            parse_failure_witnesses=parse_failure_witnesses,
            deps=_AnalysisIndexModuleTreesDeps(
                check_deadline_fn=check_deadline,
                parse_module_source_fn=runtime._parse_module_source,
                parse_module_error_types=runtime._PARSE_MODULE_ERROR_TYPES,
                record_parse_failure_witness_fn=runtime._record_parse_failure_witness,
            ),
        ),
    )


def _analysis_index_stage_cache(
    analysis_index,
    paths,
    *,
    spec,
    parse_failure_witnesses,
    module_trees_fn=None,
):
    runtime = _runtime_module()
    return cast(
        dict[Path, object | None],
        _analysis_index_stage_cache_impl(
            analysis_index,
            paths,
            spec=spec,
            parse_failure_witnesses=parse_failure_witnesses,
            module_trees_fn=module_trees_fn,
            deps=_AnalysisIndexStageCacheDeps(
                check_deadline_fn=check_deadline,
                get_global_derivation_cache_fn=get_global_derivation_cache,
                analysis_index_module_trees_fn=_analysis_index_module_trees,
                get_stage_cache_bucket_fn=_get_stage_cache_bucket,
                path_dependency_payload_fn=runtime._path_dependency_payload,
                analysis_index_stage_cache_op=runtime._ANALYSIS_INDEX_STAGE_CACHE_OP,
            ),
        ),
    )


def _run_indexed_pass(
    paths,
    *,
    project_root,
    ignore_params,
    strictness,
    external_filter,
    transparent_decorators=None,
    parse_failure_witnesses=None,
    analysis_index=None,
    spec,
    build_index=_build_analysis_index,
):
    runtime = _runtime_module()
    check_deadline()
    sink = runtime._parse_failure_sink(parse_failure_witnesses)
    index = analysis_index
    if index is None:
        index = build_index(
            paths,
            project_root=project_root,
            ignore_params=ignore_params,
            strictness=strictness,
            external_filter=external_filter,
            transparent_decorators=transparent_decorators,
            parse_failure_witnesses=sink,
        )
    context = runtime._IndexedPassContext(
        paths=paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=sink,
        analysis_index=index,
    )
    return spec.run(context)


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


__all__ = [
    "_analysis_index_module_trees",
    "_analysis_index_resolved_call_edges",
    "_analysis_index_resolved_call_edges_by_caller",
    "_analysis_index_stage_cache",
    "_analysis_index_transitive_callers",
    "_analyze_file_internal",
    "_build_stage_cache_identity_spec",
    "_cache_identity_aliases",
    "_canonical_cache_identity",
    "_canonical_stage_cache_detail",
    "_canonical_stage_cache_identity",
    "_build_analysis_collection_resume_payload",
    "_build_analysis_index",
    "_build_call_graph",
    "_collect_transitive_callers",
    "_get_stage_cache_bucket",
    "_index_stage_cache_identity",
    "_iter_resolved_edge_param_events",
    "_load_analysis_collection_resume_payload",
    "_normalize_cache_config",
    "_parse_stage_cache_key",
    "_projection_stage_cache_identity",
    "_reduce_resolved_call_edges",
    "_resume_variant_for_identity",
    "_run_indexed_pass",
    "_sorted_text",
    "_stage_cache_key_aliases",
]
