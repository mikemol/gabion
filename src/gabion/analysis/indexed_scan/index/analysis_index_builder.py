from __future__ import annotations

import ast
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, cast

from gabion.analysis.foundation.json_types import JSONObject, JSONValue


@dataclass(frozen=True)
class AnalysisIndexBuildDeps:
    check_deadline_fn: Callable[[], None]
    accumulate_function_index_for_tree_default_fn: Callable[..., None]
    sorted_text_fn: Callable[[object], tuple[str, ...]]
    cache_context_ctor: Callable[..., object]
    index_stage_cache_identity_fn: Callable[..., object]
    projection_stage_cache_identity_fn: Callable[..., object]
    iter_monotonic_paths_fn: Callable[..., list[Path]]
    load_analysis_index_resume_payload_fn: Callable[..., tuple[set[Path], dict[str, object], object, object]]
    function_index_acc_ctor: Callable[..., object]
    sort_once_fn: Callable[..., list[str]]
    profiling_payload_fn: Callable[..., JSONObject]
    serialize_resume_payload_fn: Callable[..., JSONObject]
    parse_module_source_fn: Callable[[Path], ast.Module]
    parse_module_error_types: tuple[type[BaseException], ...]
    record_parse_failure_witness_fn: Callable[..., None]
    parse_module_stage_function_index: object
    parse_module_stage_symbol_table: object
    parse_module_stage_class_index: object
    accumulate_symbol_table_for_tree_fn: Callable[..., None]
    accumulate_class_index_for_tree_fn: Callable[..., None]
    timeout_exceeded_type: type[BaseException]
    analysis_index_ctor: Callable[..., object]
    progress_emit_min_interval_seconds: float


def build_analysis_index(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators=None,
    parse_failure_witnesses: list[JSONObject],
    resume_payload=None,
    on_progress=None,
    accumulate_function_index_for_tree_fn=None,
    forest_spec_id=None,
    fingerprint_seed_revision=None,
    decision_ignore_params=None,
    decision_require_tiers: bool = False,
    deps: AnalysisIndexBuildDeps,
) -> object:
    deps.check_deadline_fn()
    accumulate_function_index_for_tree = accumulate_function_index_for_tree_fn
    if accumulate_function_index_for_tree is None:
        accumulate_function_index_for_tree = (
            deps.accumulate_function_index_for_tree_default_fn
        )
    normalized_ignore = deps.sorted_text_fn(ignore_params)
    normalized_transparent = deps.sorted_text_fn(transparent_decorators)
    normalized_decision_ignore = deps.sorted_text_fn(decision_ignore_params)
    cache_context = deps.cache_context_ctor(
        forest_spec_id=forest_spec_id,
        fingerprint_seed_revision=fingerprint_seed_revision,
    )
    # dataflow-bundle: decision_require_tiers, external_filter
    index_config_subset: dict[str, JSONValue] = {
        "ignore_params": list(normalized_ignore),
        "strictness": str(strictness),
        "transparent_decorators": list(normalized_transparent),
        "external_filter": external_filter,
        "decision_ignore_params": list(normalized_decision_ignore),
        "decision_require_tiers": decision_require_tiers,
    }
    index_cache_identity = deps.index_stage_cache_identity_fn(
        cache_context=cache_context,
        config_subset=index_config_subset,
    )
    projection_cache_identity = deps.projection_stage_cache_identity_fn(
        cache_context=cache_context,
        config_subset={
            "strictness": str(strictness),
            "external_filter": external_filter,
            "decision_require_tiers": decision_require_tiers,
        },
    )
    index_cache_identity_value = cast(str, getattr(index_cache_identity, "value"))
    projection_cache_identity_value = cast(
        str, getattr(projection_cache_identity, "value")
    )
    ordered_paths = deps.iter_monotonic_paths_fn(
        paths,
        source="_build_analysis_index.paths",
    )
    (
        hydrated_paths,
        by_qual,
        symbol_table,
        class_index,
    ) = deps.load_analysis_index_resume_payload_fn(
        payload=resume_payload,
        file_paths=ordered_paths,
        expected_index_cache_identity=index_cache_identity_value,
        expected_projection_cache_identity=projection_cache_identity_value,
    )
    symbol_table.external_filter = external_filter
    function_index_acc = deps.function_index_acc_ctor(
        by_name=defaultdict(list),
        by_qual={},
    )
    for qual in deps.sort_once_fn(
        by_qual,
        source="_build_analysis_index.resume.by_qual",
    ):
        deps.check_deadline_fn()
        info = by_qual[qual]
        function_index_acc.by_qual[qual] = info
        function_index_acc.by_name[info.name].append(info)
    progress_since_emit = 0
    last_progress_emit_monotonic = None
    profile_stage_ns: dict[str, int] = {
        "analysis_index.parse_module": 0,
        "analysis_index.function_index": 0,
        "analysis_index.symbol_table": 0,
        "analysis_index.class_index": 0,
    }
    profile_counters: Counter[str] = Counter(
        {
            "analysis_index.paths_total": len(ordered_paths),
            "analysis_index.paths_hydrated": len(hydrated_paths),
            "analysis_index.paths_parsed": 0,
            "analysis_index.parse_errors": 0,
        }
    )

    def _index_profile_payload() -> JSONObject:
        return deps.profiling_payload_fn(
            stage_ns=profile_stage_ns,
            counters=profile_counters,
        )

    def _emit_index_progress(*, force: bool = False) -> None:
        nonlocal progress_since_emit
        nonlocal last_progress_emit_monotonic
        progress_callback = on_progress
        if progress_callback is not None:
            progress_since_emit += 1
            now = time.monotonic()
            emit_allowed = (
                force
                or last_progress_emit_monotonic is None
                or (
                    now - last_progress_emit_monotonic
                    >= deps.progress_emit_min_interval_seconds
                )
            )
            if emit_allowed:
                progress_since_emit = 0
                last_progress_emit_monotonic = now
                progress_callback(
                    deps.serialize_resume_payload_fn(
                        hydrated_paths=hydrated_paths,
                        by_qual=function_index_acc.by_qual,
                        symbol_table=symbol_table,
                        class_index=class_index,
                        index_cache_identity=index_cache_identity_value,
                        projection_cache_identity=projection_cache_identity_value,
                        profiling_v1=_index_profile_payload(),
                        previous_payload=resume_payload,
                    )
                )

    try:
        for path in ordered_paths:
            deps.check_deadline_fn()
            if path in hydrated_paths:
                continue
            parse_started_ns = time.monotonic_ns()
            try:
                tree = deps.parse_module_source_fn(path)
            except deps.parse_module_error_types as exc:
                profile_stage_ns["analysis_index.parse_module"] += (
                    time.monotonic_ns() - parse_started_ns
                )
                profile_counters["analysis_index.parse_errors"] += 1
                deps.record_parse_failure_witness_fn(
                    sink=parse_failure_witnesses,
                    path=path,
                    stage=deps.parse_module_stage_function_index,
                    error=exc,
                )
                deps.record_parse_failure_witness_fn(
                    sink=parse_failure_witnesses,
                    path=path,
                    stage=deps.parse_module_stage_symbol_table,
                    error=exc,
                )
                deps.record_parse_failure_witness_fn(
                    sink=parse_failure_witnesses,
                    path=path,
                    stage=deps.parse_module_stage_class_index,
                    error=exc,
                )
                continue
            profile_stage_ns["analysis_index.parse_module"] += (
                time.monotonic_ns() - parse_started_ns
            )
            profile_counters["analysis_index.paths_parsed"] += 1
            function_started_ns = time.monotonic_ns()
            accumulate_function_index_for_tree(
                function_index_acc,
                path,
                tree,
                project_root=project_root,
                ignore_params=ignore_params,
                strictness=strictness,
                transparent_decorators=transparent_decorators,
            )
            profile_stage_ns["analysis_index.function_index"] += (
                time.monotonic_ns() - function_started_ns
            )
            symbol_started_ns = time.monotonic_ns()
            deps.accumulate_symbol_table_for_tree_fn(
                symbol_table,
                path,
                tree,
                project_root=project_root,
            )
            profile_stage_ns["analysis_index.symbol_table"] += (
                time.monotonic_ns() - symbol_started_ns
            )
            class_started_ns = time.monotonic_ns()
            deps.accumulate_class_index_for_tree_fn(
                class_index,
                path,
                tree,
                project_root=project_root,
            )
            profile_stage_ns["analysis_index.class_index"] += (
                time.monotonic_ns() - class_started_ns
            )
            hydrated_paths.add(path)
            profile_counters["analysis_index.paths_hydrated"] = len(hydrated_paths)
            _emit_index_progress()
    except deps.timeout_exceeded_type:
        _emit_index_progress(force=True)
        raise
    _emit_index_progress(force=True)
    return deps.analysis_index_ctor(
        by_name=dict(function_index_acc.by_name),
        by_qual=function_index_acc.by_qual,
        symbol_table=symbol_table,
        class_index=class_index,
        index_cache_identity=index_cache_identity_value,
        projection_cache_identity=projection_cache_identity_value,
    )
