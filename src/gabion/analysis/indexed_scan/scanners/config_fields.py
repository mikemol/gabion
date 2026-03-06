# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast

from gabion.analysis.foundation.json_types import JSONObject
from gabion.analysis.indexed_scan.index.analysis_index_stage_cache import (
    AnalysisIndexStageCacheFn,
)


@dataclass(frozen=True)
class IterConfigFieldsDeps:
    check_deadline_fn: Callable[[], None]
    parse_module_tree_fn: Callable[..., object]
    parse_module_stage_config_fields: object
    simple_store_name_fn: Callable[..., object]


@dataclass(frozen=True)
class CollectConfigBundlesDeps:
    check_deadline_fn: Callable[[], None]
    forbid_adhoc_bundle_discovery_fn: Callable[[str], None]
    analysis_index_stage_cache_fn: AnalysisIndexStageCacheFn[object]
    stage_cache_spec_ctor: Callable[..., object]
    parse_module_stage_config_fields: object
    parse_stage_cache_key_fn: Callable[..., object]
    empty_cache_semantic_context: object
    iter_config_fields_fn: Callable[..., object]


def iter_config_fields(
    path: Path,
    *,
    tree = None,
    parse_failure_witnesses: list[JSONObject],
    deps: IterConfigFieldsDeps,
) -> dict[str, set[str]]:
    """Best-effort extraction of config bundles from dataclasses."""
    deps.check_deadline_fn()
    module_tree = tree
    if module_tree is None:
        module_tree = deps.parse_module_tree_fn(
            path,
            stage=deps.parse_module_stage_config_fields,
            parse_failure_witnesses=parse_failure_witnesses,
        )
    if module_tree is None:
        return {}
    bundles: dict[str, set[str]] = {}
    for node in ast.walk(module_tree):
        deps.check_deadline_fn()
        if type(node) is not ast.ClassDef:
            continue
        class_node = cast(ast.ClassDef, node)
        decorators = {getattr(d, "id", None) for d in class_node.decorator_list}
        is_dataclass = "dataclass" in decorators
        is_config = class_node.name.endswith("Config")
        if not is_dataclass and not is_config:
            continue
        fields: set[str] = set()
        for stmt in class_node.body:
            deps.check_deadline_fn()
            stmt_type = type(stmt)
            if stmt_type is ast.AnnAssign:
                ann_stmt = cast(ast.AnnAssign, stmt)
                raw_name = deps.simple_store_name_fn(ann_stmt.target)
                if type(raw_name) is str and (is_config or raw_name.endswith("_fn")):
                    fields.add(raw_name)
            elif stmt_type is ast.Assign:
                assign_stmt = cast(ast.Assign, stmt)
                for target in assign_stmt.targets:
                    deps.check_deadline_fn()
                    raw_name = deps.simple_store_name_fn(target)
                    if type(raw_name) is str and (is_config or raw_name.endswith("_fn")):
                        fields.add(raw_name)
        if fields:
            bundles[class_node.name] = fields
    return bundles


def collect_config_bundles(
    paths: list[Path],
    *,
    parse_failure_witnesses: list[JSONObject],
    analysis_index = None,
    deps: CollectConfigBundlesDeps,
) -> dict[Path, dict[str, set[str]]]:
    deps.check_deadline_fn()
    deps.forbid_adhoc_bundle_discovery_fn("_collect_config_bundles")
    bundles_by_path: dict[Path, dict[str, set[str]]] = {}
    if analysis_index is not None:
        config_fields_by_path = deps.analysis_index_stage_cache_fn(
            analysis_index,
            paths,
            spec=deps.stage_cache_spec_ctor(
                stage=deps.parse_module_stage_config_fields,
                cache_key=deps.parse_stage_cache_key_fn(
                    stage=deps.parse_module_stage_config_fields,
                    cache_context=deps.empty_cache_semantic_context,
                    config_subset={},
                    detail="config_fields",
                ),
                build=lambda tree, path: deps.iter_config_fields_fn(
                    path,
                    tree=tree,
                    parse_failure_witnesses=parse_failure_witnesses,
                ),
            ),
            parse_failure_witnesses=parse_failure_witnesses,
        )
        for path, bundles in config_fields_by_path.items():
            deps.check_deadline_fn()
            if bundles:
                bundles_by_path[path] = bundles
        return bundles_by_path
    for path in paths:
        deps.check_deadline_fn()
        bundles = deps.iter_config_fields_fn(
            path,
            parse_failure_witnesses=parse_failure_witnesses,
        )
        if bundles:
            bundles_by_path[path] = bundles
    return bundles_by_path
