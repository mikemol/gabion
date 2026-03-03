# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from gabion.analysis.foundation.json_types import JSONObject


@dataclass(frozen=True)
class CollectDeadlineFunctionFactsDeps:
    check_deadline_fn: Callable[[], None]
    analysis_index_stage_cache_fn: Callable[..., dict[Path, object]]
    stage_cache_spec_ctor: Callable[..., object]
    parse_stage_cache_key_fn: Callable[..., str]
    deadline_function_facts_stage: object
    empty_cache_semantic_context: object
    sorted_text_fn: Callable[[set[str]], list[str]]
    deadline_function_facts_for_tree_fn: Callable[..., dict[str, object]]
    parse_module_tree_fn: Callable[..., object]


def collect_deadline_function_facts(
    paths: list[Path],
    *,
    project_root=None,
    ignore_params: set[str],
    parse_failure_witnesses: list[JSONObject],
    trees=None,
    analysis_index=None,
    stage_cache_fn=None,
    deps: CollectDeadlineFunctionFactsDeps,
) -> dict[str, object]:
    deps.check_deadline_fn()
    ignore_param_names = set(ignore_params or ())
    if stage_cache_fn is None:
        stage_cache_fn = deps.analysis_index_stage_cache_fn

    if analysis_index is not None and trees is None:
        facts_by_path = stage_cache_fn(
            analysis_index,
            paths,
            spec=deps.stage_cache_spec_ctor(
                stage=deps.deadline_function_facts_stage,
                cache_key=deps.parse_stage_cache_key_fn(
                    stage=deps.deadline_function_facts_stage,
                    cache_context=deps.empty_cache_semantic_context,
                    config_subset={
                        "project_root": str(project_root) if project_root is not None else "",
                        "ignore_params": list(deps.sorted_text_fn(ignore_param_names)),
                    },
                    detail="deadline_function_facts",
                ),
                build=lambda tree, path: deps.deadline_function_facts_for_tree_fn(
                    path,
                    tree,
                    project_root=project_root,
                    ignore_params=ignore_param_names,
                ),
            ),
            parse_failure_witnesses=parse_failure_witnesses,
        )
        facts: dict[str, object] = {}
        for entry in facts_by_path.values():
            deps.check_deadline_fn()
            if entry is not None:
                facts.update(entry)
        return facts

    facts: dict[str, object] = {}
    for path in paths:
        deps.check_deadline_fn()
        if trees is not None and path in trees:
            tree = trees[path]
        else:
            tree = deps.parse_module_tree_fn(
                path,
                stage=deps.deadline_function_facts_stage,
                parse_failure_witnesses=parse_failure_witnesses,
            )
        if tree is not None:
            facts.update(
                deps.deadline_function_facts_for_tree_fn(
                    path,
                    tree,
                    project_root=project_root,
                    ignore_params=ignore_param_names,
                )
            )
    return facts


def collect_deadline_function_facts_from_runtime_module(
    paths: list[Path],
    *,
    project_root=None,
    ignore_params: set[str],
    parse_failure_witnesses: list[JSONObject],
    trees=None,
    analysis_index=None,
    stage_cache_fn=None,
    runtime_module,
) -> dict[str, object]:
    return collect_deadline_function_facts(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        parse_failure_witnesses=parse_failure_witnesses,
        trees=trees,
        analysis_index=analysis_index,
        stage_cache_fn=stage_cache_fn,
        deps=CollectDeadlineFunctionFactsDeps(
            check_deadline_fn=runtime_module.check_deadline,
            analysis_index_stage_cache_fn=runtime_module._analysis_index_stage_cache,
            stage_cache_spec_ctor=runtime_module._StageCacheSpec,
            parse_stage_cache_key_fn=runtime_module._parse_stage_cache_key,
            deadline_function_facts_stage=runtime_module._ParseModuleStage.DEADLINE_FUNCTION_FACTS,
            empty_cache_semantic_context=runtime_module._EMPTY_CACHE_SEMANTIC_CONTEXT,
            sorted_text_fn=runtime_module._sorted_text,
            deadline_function_facts_for_tree_fn=runtime_module._deadline_function_facts_for_tree,
            parse_module_tree_fn=runtime_module._parse_module_tree,
        ),
    )
