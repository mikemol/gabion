# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from gabion.analysis.foundation.json_types import JSONObject


@dataclass(frozen=True)
class AnalysisIndexModuleTreesDeps:
    check_deadline_fn: Callable[[], None]
    parse_module_source_fn: Callable[[Path], object]
    parse_module_error_types: tuple[type[BaseException], ...]
    record_parse_failure_witness_fn: Callable[..., None]


def analysis_index_module_trees(
    analysis_index,
    paths: list[Path],
    *,
    stage,
    parse_failure_witnesses: list[JSONObject],
    deps: AnalysisIndexModuleTreesDeps,
) -> dict[Path, object]:
    deps.check_deadline_fn()
    trees: dict[Path, object] = {}
    for path in paths:
        deps.check_deadline_fn()
        cached_tree = analysis_index.parsed_modules_by_path.get(path)
        if cached_tree is not None:
            trees[path] = cached_tree
            continue
        cached_error = analysis_index.module_parse_errors_by_path.get(path)
        if cached_error is not None:
            deps.record_parse_failure_witness_fn(
                sink=parse_failure_witnesses,
                path=path,
                stage=stage,
                error=cached_error,
            )
            trees[path] = None
            continue
        try:
            tree = deps.parse_module_source_fn(path)
        except deps.parse_module_error_types as exc:
            analysis_index.module_parse_errors_by_path[path] = exc
            deps.record_parse_failure_witness_fn(
                sink=parse_failure_witnesses,
                path=path,
                stage=stage,
                error=exc,
            )
            trees[path] = None
            continue
        analysis_index.parsed_modules_by_path[path] = tree
        trees[path] = tree
    return trees
