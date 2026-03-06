# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast

from gabion.analysis.foundation.json_types import JSONObject
from gabion.analysis.indexed_scan.index.analysis_index_stage_cache import (
    AnalysisIndexStageCacheFn,
)


@dataclass(frozen=True)
class CallNodesForTreeDeps:
    check_deadline_fn: Callable[[], None]
    node_span_fn: Callable[..., object]


@dataclass(frozen=True)
class CollectCallNodesByPathDeps:
    check_deadline_fn: Callable[[], None]
    analysis_index_stage_cache_fn: AnalysisIndexStageCacheFn[object]
    stage_cache_spec_ctor: Callable[..., object]
    parse_module_stage_call_nodes: object
    parse_stage_cache_key_fn: Callable[..., object]
    empty_cache_semantic_context: object
    call_nodes_for_tree_fn: Callable[..., object]
    parse_module_tree_fn: Callable[..., object]


def call_nodes_for_tree(
    tree: ast.AST,
    *,
    deps: CallNodesForTreeDeps,
) -> dict[tuple[int, int, int, int], list[ast.Call]]:
    deps.check_deadline_fn()
    span_map: dict[tuple[int, int, int, int], list[ast.Call]] = defaultdict(list)
    for node in ast.walk(tree):
        deps.check_deadline_fn()
        if type(node) is ast.Call:
            call_node = cast(ast.Call, node)
            span = deps.node_span_fn(call_node)
            if span is not None:
                span_map[span].append(call_node)
    return span_map


def collect_call_nodes_by_path(
    paths: list[Path],
    *,
    trees = None,
    parse_failure_witnesses: list[JSONObject],
    analysis_index = None,
    deps: CollectCallNodesByPathDeps,
) -> dict[Path, dict[tuple[int, int, int, int], list[ast.Call]]]:
    deps.check_deadline_fn()
    if analysis_index is not None and trees is None:
        cached_by_path = deps.analysis_index_stage_cache_fn(
            analysis_index,
            paths,
            spec=deps.stage_cache_spec_ctor(
                stage=deps.parse_module_stage_call_nodes,
                cache_key=deps.parse_stage_cache_key_fn(
                    stage=deps.parse_module_stage_call_nodes,
                    cache_context=deps.empty_cache_semantic_context,
                    config_subset={},
                    detail="call_nodes",
                ),
                build=lambda tree, _path: deps.call_nodes_for_tree_fn(tree),
            ),
            parse_failure_witnesses=parse_failure_witnesses,
        )
        return {
            path: nodes
            for path, nodes in cached_by_path.items()
            if nodes is not None
        }
    call_nodes: dict[Path, dict[tuple[int, int, int, int], list[ast.Call]]] = {}
    for path in paths:
        deps.check_deadline_fn()
        if trees is not None and path in trees:
            tree = trees[path]
        else:
            tree = deps.parse_module_tree_fn(
                path,
                stage=deps.parse_module_stage_call_nodes,
                parse_failure_witnesses=parse_failure_witnesses,
            )
        if tree is not None:
            call_nodes[path] = deps.call_nodes_for_tree_fn(tree)
    return call_nodes
