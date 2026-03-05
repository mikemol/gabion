# gabion:boundary_normalization_module
from __future__ import annotations

"""Facade compatibility module for legacy indexed-dataflow symbols."""

# Temporary boundary adapter retained for external import compatibility.
_BOUNDARY_ADAPTER_LIFECYCLE: dict[str, object] = {
    "actor": "codex",
    "rationale": "WS-5 hard-cut completed; retain facade alias surface while external importers migrate",
    "scope": "dataflow_facade.alias_surface",
    "start": "2026-03-05",
    "expiry": "WS-5 compatibility-shim retirement",
    "rollback_condition": "no external consumers require facade path aliases",
    "evidence_links": ["docs/ws5_decomposition_ledger.md"],
}

from gabion.analysis.dataflow.engine.dataflow_deadline_contracts import (
    _DeadlineFunctionFacts,
    _DeadlineLocalInfo,
    _DeadlineLoopFacts,
)
from gabion.analysis.dataflow.engine.dataflow_deadline_helpers import (
    _DeadlineFunctionCollector,
    _DeadlineArgInfo,
    _bind_call_args,
    _classify_deadline_expr,
    _collect_call_edges_from_forest,
    _collect_call_edges,
    _collect_call_nodes_by_path,
    _collect_deadline_function_facts,
    _collect_deadline_local_info,
    _deadline_arg_info_map,
    _deadline_loop_forwarded_params,
    _fallback_deadline_arg_info,
    _is_dynamic_dispatch_callee_key,
    _is_deadline_origin_call,
    _is_deadline_param,
    _resolve_callee,
    _resolve_callee_outcome,
)
from gabion.analysis.dataflow.engine.dataflow_analysis_index import (
    _FILE_SCAN_PROGRESS_EMIT_INTERVAL,
    _PROGRESS_EMIT_MIN_INTERVAL_SECONDS,
    _analyze_file_internal,
    _build_analysis_index,
    _build_function_index,
    _build_symbol_table,
    _phase_work_progress,
    _profiling_v1_payload,
    _stage_cache_key_aliases,
    analyze_file,
)
from gabion.analysis.dataflow.engine.dataflow_projection_materialization import (
    CallAmbiguity,
    _ambiguity_suite_relation,
    _ambiguity_suite_row_to_suite,
    _ambiguity_virtual_count_gt_1,
    _collect_call_ambiguities,
    _collect_call_ambiguities_indexed,
    _dedupe_call_ambiguities,
    _emit_call_ambiguities,
    _format_span_fields,
    _lint_lines_from_call_ambiguities,
    _materialize_ambiguity_suite_agg_spec,
    _materialize_ambiguity_virtual_set_spec,
    _materialize_projection_spec_rows,
    _materialize_suite_order_spec,
    _populate_bundle_forest,
    _spec_row_span,
    _suite_order_relation,
    _suite_order_row_to_site,
    _suite_site_label,
    _summarize_call_ambiguities,
)
from gabion.analysis.dataflow.engine.dataflow_documented_bundles import (
    _iter_documented_bundles,
)
from gabion.analysis.dataflow.engine.dataflow_function_index_decision_support import (
    _decision_surface_params,
    _decorator_name,
    _decorators_transparent,
    _value_encoded_decision_params,
)
from gabion.analysis.dataflow.engine.dataflow_call_graph_algorithms import (
    _collect_recursive_functions,
)
from gabion.analysis.dataflow.engine.dataflow_function_index_helpers import (
    _enclosing_class,
    _enclosing_function_scopes,
    _enclosing_scopes,
    _is_test_path,
    _param_names,
    _param_spans,
)
from gabion.analysis.dataflow.engine.dataflow_ingest_helpers import (
    _collect_functions,
)
from gabion.analysis.dataflow.engine.dataflow_ingested_analysis_support import (
    _group_by_signature,
    _propagate_groups,
    _union_groups,
    analyze_ingested_file,
)
from gabion.analysis.dataflow.engine.dataflow_post_phase_analyses import (
    _StageCacheSpec,
    _annotation_exception_candidates,
    _build_property_hook_callable_index,
    _branch_reachability_under_env,
    _callsite_evidence_for_bundle,
    _collect_config_bundles,
    _collect_constant_flow_details,
    _collect_dataclass_registry,
    _collect_exception_obligations,
    _collect_handledness_witnesses,
    _collect_invariant_propositions,
    _collect_never_invariants,
    _combine_type_hints,
    _compute_knob_param_names,
    _eval_bool_expr,
    _eval_value_expr,
    _format_call_site,
    _format_type_flow_site,
    _iter_config_fields,
    _iter_dataclass_call_bundles,
    _keyword_links_literal,
    _keyword_string_literal,
    _names_in_expr,
    _node_in_block,
    _param_annotations_by_path,
    _parse_module_source,
    _refine_exception_name_from_annotations,
    _split_top_level,
    _type_from_const_repr,
    analyze_constant_flow_repo,
    analyze_deadness_flow_repo,
    analyze_decision_surfaces_repo,
    analyze_unused_arg_flow_repo,
    analyze_value_encoded_decisions_repo,
    generate_property_hook_manifest,
)
from gabion.analysis.dataflow.engine.dataflow_deadline_summary import (
    _summarize_deadline_obligations,
)
from gabion.analysis.dataflow.engine.dataflow_runtime_reporting import (
    _report_section_spec,
)
from gabion.analysis.dataflow.engine.dataflow_bundle_merge import (
    _merge_counts_by_knobs,
)
from gabion.analysis.dataflow.engine.dataflow_lambda_runtime_support import (
    _collect_lambda_bindings_by_caller,
    _collect_lambda_function_infos,
    _function_key,
)
from gabion.analysis.dataflow.engine.dataflow_function_index_runtime_support import (
    _direct_lambda_callee_by_call_span,
)
from gabion.analysis.dataflow.engine.dataflow_function_semantics import (
    _analyze_function,
    _collect_return_aliases,
)
from gabion.analysis.dataflow.engine.dataflow_resume_serialization import (
    _CACHE_IDENTITY_DIGEST_HEX,
    _CACHE_IDENTITY_PREFIX,
    _CacheIdentity,
    _build_analysis_collection_resume_payload,
    _deserialize_function_info_for_resume,
    _deserialize_invariants_for_resume,
    _deserialize_symbol_table_for_resume,
    _invariant_confidence,
    _invariant_digest,
    _load_analysis_collection_resume_payload,
    _load_analysis_index_resume_payload,
    _load_file_scan_resume_state,
    _normalize_invariant_proposition,
    _serialize_analysis_index_resume_payload,
    _serialize_file_scan_resume_state,
)
from gabion.analysis.dataflow.engine.dataflow_contracts import (
    AuditConfig,
    CallArgs,
    ClassInfo,
    FunctionInfo,
    InvariantProposition,
    ParamUse,
    ReportCarrier,
    SymbolTable,
)
from gabion.analysis.dataflow.engine.dataflow_adapter_contract import (
    parse_adapter_capabilities,
)
from gabion.analysis.dataflow.engine.dataflow_fingerprint_helpers import (
    _build_synth_registry_payload,
    _collect_fingerprint_atom_keys,
    _compute_fingerprint_coherence,
    _compute_fingerprint_matches,
    _compute_fingerprint_provenance,
    _compute_fingerprint_rewrite_plans,
    _compute_fingerprint_synth,
    _compute_fingerprint_warnings,
    _find_provenance_entry_for_site,
    _fingerprint_soundness_issues,
    _glossary_match_strata,
    _summarize_fingerprint_provenance,
    verify_rewrite_plan,
    verify_rewrite_plans,
)
from gabion.analysis.dataflow.engine.dataflow_evidence_helpers import (
    _collect_module_exports,
)
from gabion.analysis.dataflow.engine.dataflow_raw_runtime import (
    _resolve_synth_registry_path,
)

from gabion.analysis.dataflow.engine.dataflow_callee_resolution_support import (
    _resolve_method_in_hierarchy,
    _resolve_class_candidates,
)
from gabion.analysis.dataflow.engine.dataflow_local_class_hierarchy import (
    _collect_local_class_bases,
    _resolve_local_method_in_hierarchy,
)
from gabion.analysis.aspf.aspf import Forest, NodeId
from gabion.analysis.core.visitors import ParentAnnotator
from gabion.analysis.dataflow.engine.dataflow_lint_helpers import (
    _deadline_lint_lines,
    _exception_protocol_lint_lines,
    _internal_broad_type_lint_lines,
    _is_broad_internal_type,
    _lint_lines_from_bundle_evidence,
    _lint_lines_from_constant_smells,
    _lint_lines_from_type_evidence,
    _lint_lines_from_unused_arg_smells,
    _normalize_type_name,
    _parse_exception_path_id,
    _parse_lint_location,
)
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.indexed_scan.ast.expression_eval import EvalDecision as _EvalDecision
from gabion.analysis.projection.projection_registry import (
    DEADLINE_OBLIGATIONS_SUMMARY_SPEC,
)
from gabion.analysis.dataflow.io.dataflow_projection_helpers import (
    _topologically_order_report_projection_specs,
)
from gabion.analysis.dataflow.io.dataflow_reporting import emit_report as _emit_report
from gabion.analysis.dataflow.io.dataflow_reporting_helpers import (
    render_mermaid_component as _render_mermaid_component,
)
from gabion.analysis.dataflow.io.dataflow_reporting import render_report
from gabion.order_contract import sort_once
