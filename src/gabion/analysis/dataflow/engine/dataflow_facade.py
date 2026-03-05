# gabion:boundary_normalization_module
from __future__ import annotations

"""Facade compatibility module for legacy indexed-dataflow symbols."""

from gabion.analysis.dataflow.engine.dataflow_deadline_contracts import (
    _CalleeResolutionOutcome,
    _DeadlineFunctionFacts,
    _DeadlineLoopFacts,
)
from gabion.analysis.dataflow.engine.dataflow_deadline_runtime_owner import (
    _DeadlineFunctionCollector,
    _collect_call_edges,
    _collect_call_nodes_by_path,
    _collect_deadline_function_facts,
    _collect_deadline_local_info,
    _normalize_snapshot_path,
    _resolve_callee_outcome,
)
from gabion.analysis.dataflow.engine.dataflow_analysis_index_owner import (
    _accumulate_function_index_for_tree_runtime as _accumulate_function_index_for_tree,
    _analyze_file_internal,
)
from gabion.analysis.dataflow.engine.dataflow_projection_materialization import (
    _populate_bundle_forest,
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
from gabion.analysis.dataflow.engine import dataflow_indexed_file_scan as _runtime


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


# Explicit static compatibility exports for high-use helper surfaces.
_annotation_exception_candidates = _runtime._annotation_exception_candidates
_bind_call_args = _runtime._bind_call_args
_collect_dataclass_registry = _runtime._collect_dataclass_registry
_collect_recursive_functions = _runtime._collect_recursive_functions
_deadline_loop_forwarded_params = _runtime._deadline_loop_forwarded_params
_decision_surface_params = _runtime._decision_surface_params
_decorator_name = _runtime._decorator_name
_emit_report = _runtime._emit_report
_is_deadline_origin_call = _runtime._is_deadline_origin_call
_iter_config_fields = _runtime._iter_config_fields
_iter_dataclass_call_bundles = _runtime._iter_dataclass_call_bundles
_iter_documented_bundles = _runtime._iter_documented_bundles
_keyword_string_literal = _runtime._keyword_string_literal
_materialize_projection_spec_rows = _runtime._materialize_projection_spec_rows
_merge_counts_by_knobs = _runtime._merge_counts_by_knobs
_phase_work_progress = _runtime._phase_work_progress
_render_mermaid_component = _runtime._render_mermaid_component
_report_section_spec = _runtime._report_section_spec
_stage_cache_key_aliases = _runtime._stage_cache_key_aliases
_suite_site_label = _runtime._suite_site_label
_summarize_deadline_obligations = _runtime._summarize_deadline_obligations
_type_from_const_repr = _runtime._type_from_const_repr
_value_encoded_decision_params = _runtime._value_encoded_decision_params
_DeadlineLocalInfo = _runtime._DeadlineLocalInfo


def __getattr__(name: str):
    return getattr(_runtime, name)


def __dir__() -> list[str]:
    return sorted(set(dir(_runtime)))
