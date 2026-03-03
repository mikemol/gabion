# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from gabion.analysis.foundation.json_types import JSONObject


@dataclass(frozen=True)
class AnalysisIndexStageCacheDeps:
    check_deadline_fn: Callable[[], None]
    get_global_derivation_cache_fn: Callable[[], object]
    analysis_index_module_trees_fn: Callable[..., object]
    get_stage_cache_bucket_fn: Callable[..., dict[Path, object]]
    path_dependency_payload_fn: Callable[[Path], dict[str, object]]
    analysis_index_stage_cache_op: object


def analysis_index_stage_cache(
    analysis_index,
    paths: list[Path],
    *,
    spec,
    parse_failure_witnesses: list[JSONObject],
    module_trees_fn=None,
    deps: AnalysisIndexStageCacheDeps,
) -> dict[Path, object]:
    deps.check_deadline_fn()
    module_trees = module_trees_fn
    if module_trees is None:
        module_trees = deps.analysis_index_module_trees_fn
    derivation_runtime = deps.get_global_derivation_cache_fn()
    trees = module_trees(
        analysis_index,
        paths,
        stage=spec.stage,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    scoped_cache_key = (analysis_index.index_cache_identity, spec.cache_key)
    cache = deps.get_stage_cache_bucket_fn(
        analysis_index,
        scoped_cache_key=scoped_cache_key,
    )
    results: dict[Path, object] = {}
    for path in paths:
        deps.check_deadline_fn()
        tree = trees.get(path)
        if tree is None:
            results[path] = None
            continue
        if path not in cache:
            try:
                dependencies = deps.path_dependency_payload_fn(path)
            except OSError:
                dependencies = {
                    "path": str(path),
                    "mtime_ns": 0,
                    "size": 0,
                }

            def _compute_stage_value():
                return spec.build(tree, path)

            cache[path] = derivation_runtime.derive(
                op=deps.analysis_index_stage_cache_op,
                structural_inputs={
                    "index_cache_identity": analysis_index.index_cache_identity,
                    "projection_cache_identity": analysis_index.projection_cache_identity,
                    "stage": spec.stage.value,
                    "cache_key": spec.cache_key,
                    "path": str(path.resolve()),
                },
                dependencies=dependencies,
                params={"cache_scope": "analysis_index_stage_cache"},
                compute_fn=_compute_stage_value,
                source="gabion.analysis.indexed_scan.analysis_index_stage_cache.analysis_index_stage_cache",
            )
        results[path] = cache[path]
    return results
