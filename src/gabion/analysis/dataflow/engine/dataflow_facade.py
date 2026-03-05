# gabion:boundary_normalization_module
from __future__ import annotations

"""Facade compatibility module for legacy indexed-dataflow symbols."""

from gabion.analysis.dataflow.engine.dataflow_deadline_contracts import (
    _CalleeResolutionOutcome,
    _DeadlineFunctionFacts,
    _DeadlineLocalInfo,
    _DeadlineLoopFacts,
)
from gabion.analysis.dataflow.engine.dataflow_deadline_runtime_owner import (
    _DeadlineFunctionCollector,
    _bind_call_args,
    _collect_call_edges,
    _collect_call_nodes_by_path,
    _collect_deadline_function_facts,
    _collect_deadline_local_info,
    _deadline_loop_forwarded_params,
    _is_deadline_origin_call,
    _normalize_snapshot_path,
    _resolve_callee_outcome,
)
from gabion.analysis.dataflow.engine.dataflow_analysis_index_owner import (
    _accumulate_function_index_for_tree_runtime as _accumulate_function_index_for_tree,
    _analyze_file_internal,
    _stage_cache_key_aliases,
)
from gabion.analysis.dataflow.engine.dataflow_projection_materialization import (
    _materialize_projection_spec_rows,
    _populate_bundle_forest,
    _suite_site_label,
)
from gabion.analysis.dataflow.engine.dataflow_documented_bundles import (
    _iter_documented_bundles,
)
from gabion.analysis.dataflow.engine.dataflow_function_index_decision_support import (
    _decision_surface_params,
    _value_encoded_decision_params,
)
from gabion.analysis.dataflow.engine.dataflow_call_graph_algorithms import (
    _collect_recursive_functions,
)
from gabion.analysis.dataflow.engine.dataflow_post_phase_analyses import (
    _annotation_exception_candidates,
    _collect_dataclass_registry,
    _iter_config_fields,
    _iter_dataclass_call_bundles,
    _keyword_string_literal,
    _type_from_const_repr,
)
from gabion.analysis.dataflow.engine.dataflow_deadline_summary_owner import (
    _summarize_deadline_obligations,
)
from gabion.analysis.dataflow.engine.dataflow_runtime_reporting_owner import (
    _report_section_spec,
)
from gabion.analysis.dataflow.engine.dataflow_bundle_merge import (
    _merge_counts_by_knobs,
)
from gabion.analysis.dataflow.engine.dataflow_analysis_index import (
    _phase_work_progress,
)

from gabion.analysis.dataflow.engine.dataflow_analysis_index import (
    _build_analysis_index as _build_analysis_index_owner,
)
from gabion.analysis.dataflow.engine.dataflow_callee_resolution_support import (
    _resolve_method_in_hierarchy_outcome as _resolve_method_in_hierarchy_outcome_impl,
)
from gabion.analysis.dataflow.engine.dataflow_lint_helpers import (
    _parse_lint_location as _parse_lint_location,
    _internal_broad_type_lint_lines as _internal_broad_type_lint_lines_impl,
)
from gabion.analysis.dataflow.io.dataflow_reporting import emit_report as _emit_report
from gabion.analysis.dataflow.io.dataflow_reporting_helpers import (
    render_mermaid_component as _render_mermaid_component,
)
from gabion.analysis.dataflow.engine import dataflow_indexed_file_scan as _runtime


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
_decorator_name = _runtime._decorator_name

_STATIC_FACADE_EXPORTS = (
    "AuditConfig",
    "CallAmbiguity",
    "CallArgs",
    "ClassInfo",
    "DEADLINE_OBLIGATIONS_SUMMARY_SPEC",
    "Forest",
    "FunctionInfo",
    "InvariantProposition",
    "NodeId",
    "ParamUse",
    "ParentAnnotator",
    "ReportCarrier",
    "SymbolTable",
    "TimeoutExceeded",
    "_DeadlineArgInfo",
    "_FILE_SCAN_PROGRESS_EMIT_INTERVAL",
    "_PROGRESS_EMIT_MIN_INTERVAL_SECONDS",
    "_analyze_function",
    "_branch_reachability_under_env",
    "_build_analysis_collection_resume_payload",
    "_build_function_index",
    "_build_parser",
    "_build_symbol_table",
    "_build_synth_registry_payload",
    "_classify_deadline_expr",
    "_collect_call_ambiguities",
    "_collect_call_edges_from_forest",
    "_collect_config_bundles",
    "_collect_exception_obligations",
    "_collect_fingerprint_atom_keys",
    "_collect_functions",
    "_collect_lambda_bindings_by_caller",
    "_collect_lambda_function_infos",
    "_collect_local_class_bases",
    "_collect_module_exports",
    "_collect_return_aliases",
    "_compute_fingerprint_coherence",
    "_compute_fingerprint_matches",
    "_compute_fingerprint_provenance",
    "_compute_fingerprint_rewrite_plans",
    "_compute_fingerprint_synth",
    "_compute_fingerprint_warnings",
    "_deadline_arg_info_map",
    "_deadline_lint_lines",
    "_decorators_transparent",
    "_dedupe_call_ambiguities",
    "_deserialize_function_info_for_resume",
    "_deserialize_symbol_table_for_resume",
    "_direct_lambda_callee_by_call_span",
    "_emit_call_ambiguities",
    "_enclosing_class",
    "_enclosing_function_scopes",
    "_enclosing_scopes",
    "_eval_bool_expr",
    "_eval_value_expr",
    "_exception_protocol_lint_lines",
    "_fallback_deadline_arg_info",
    "_find_provenance_entry_for_site",
    "_fingerprint_soundness_issues",
    "_function_key",
    "_glossary_match_strata",
    "_group_by_signature",
    "_is_broad_internal_type",
    "_is_deadline_param",
    "_is_dynamic_dispatch_callee_key",
    "_is_test_path",
    "_keyword_links_literal",
    "_lint_lines_from_bundle_evidence",
    "_lint_lines_from_call_ambiguities",
    "_lint_lines_from_constant_smells",
    "_lint_lines_from_type_evidence",
    "_lint_lines_from_unused_arg_smells",
    "_load_analysis_collection_resume_payload",
    "_load_analysis_index_resume_payload",
    "_load_file_scan_resume_state",
    "_materialize_ambiguity_suite_agg_spec",
    "_materialize_ambiguity_virtual_set_spec",
    "_materialize_suite_order_spec",
    "_names_in_expr",
    "_node_in_block",
    "_normalize_type_name",
    "_param_names",
    "_param_spans",
    "_parse_exception_path_id",
    "_parse_module_source",
    "_profiling_v1_payload",
    "_propagate_groups",
    "_refine_exception_name_from_annotations",
    "_resolve_callee",
    "_resolve_class_candidates",
    "_resolve_local_method_in_hierarchy",
    "_resolve_synth_registry_path",
    "_serialize_analysis_index_resume_payload",
    "_serialize_file_scan_resume_state",
    "_spec_row_span",
    "_split_top_level",
    "_suite_order_relation",
    "_suite_order_row_to_site",
    "_summarize_call_ambiguities",
    "_summarize_fingerprint_provenance",
    "_topologically_order_report_projection_specs",
    "_union_groups",
    "analyze_deadness_flow_repo",
    "analyze_decision_surfaces_repo",
    "analyze_file",
    "analyze_ingested_file",
    "analyze_value_encoded_decisions_repo",
    "check_deadline",
    "parse_adapter_capabilities",
    "render_report",
    "verify_rewrite_plan",
    "verify_rewrite_plans",
)

for _name in _STATIC_FACADE_EXPORTS:
    globals().setdefault(_name, getattr(_runtime, _name))
