# gabion:boundary_normalization_module
from __future__ import annotations

"""Facade compatibility module for legacy indexed-dataflow symbols."""

from gabion.analysis.dataflow.engine.dataflow_deadline_contracts import (
    _CalleeResolutionOutcome,
    _DeadlineFunctionFacts,
    _DeadlineLoopFacts,
)
from gabion.analysis.dataflow.engine import dataflow_indexed_file_scan as _runtime
from gabion.analysis.dataflow.engine.dataflow_indexed_file_scan import (
    _DeadlineFunctionCollector,
    _accumulate_function_index_for_tree,
    _analyze_file_internal,
    _collect_call_edges,
    _collect_call_nodes_by_path,
    _collect_deadline_function_facts,
    _collect_deadline_local_info,
    _normalize_snapshot_path,
    _populate_bundle_forest,
    _resolve_callee_outcome,
)

from gabion.analysis.dataflow.engine.dataflow_analysis_index import (
    _build_analysis_index as _build_analysis_index_owner,
)
from gabion.analysis.dataflow.engine.dataflow_callee_resolution_support import (
    _resolve_method_in_hierarchy_outcome as _resolve_method_in_hierarchy_outcome_impl,
)
from gabion.analysis.dataflow.engine.dataflow_lint_helpers import (
    _internal_broad_type_lint_lines as _internal_broad_type_lint_lines_impl,
)

def _parse_lint_location(*args, **kwargs):
    from gabion.analysis.dataflow.engine.dataflow_lint_helpers import (
        _parse_lint_location as _parse_lint_location_impl,
    )

    return _parse_lint_location_impl(*args, **kwargs)


def _resolve_method_in_hierarchy(*args, **kwargs):
    outcome = _resolve_method_in_hierarchy_outcome_impl(*args, **kwargs)
    resolved = getattr(outcome, "resolved", None)
    if resolved is not None:
        return resolved
    return outcome


def _internal_broad_type_lint_lines(
    paths,
    *,
    project_root,
    ignore_params,
    strictness,
    external_filter,
    transparent_decorators=None,
    parse_failure_witnesses,
    analysis_index=None,
):
    if analysis_index is None:
        analysis_index = _build_analysis_index_owner(
            list(paths),
            project_root=project_root,
            ignore_params=set(ignore_params),
            strictness=strictness,
            external_filter=external_filter,
            transparent_decorators=transparent_decorators,
            parse_failure_witnesses=list(parse_failure_witnesses),
        )
    return _internal_broad_type_lint_lines_impl(
        list(paths),
        project_root=project_root,
        ignore_params=set(ignore_params),
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=list(parse_failure_witnesses),
        analysis_index=analysis_index,
    )


def __getattr__(name: str):
    return getattr(_runtime, name)


def __dir__() -> list[str]:
    return sorted(set(dir(_runtime)))
