# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

"""Owned indexed file-scan and helper surfaces extracted from runtime.

This module is the canonical owner for the indexed/file-scan helper graph
used by analysis, deadline, ambiguity, and reporting surfaces during
final runtime retirement.
"""


import argparse

import ast

import sys


from contextlib import ExitStack

from pathlib import Path

from typing import Iterable, Iterator, Literal, Mapping, Sequence

from gabion.ingest.python_ingest import ingest_python_file, iter_python_paths

from gabion.analysis.core.visitors import ImportVisitor, ParentAnnotator, UseVisitor

from gabion.analysis.foundation.json_types import JSONObject, JSONValue

from gabion.analysis.aspf.aspf import Alt, Forest, Node, NodeId

from gabion.analysis.semantics import evidence_keys

from gabion.analysis.core.type_fingerprints import (
    Fingerprint, FingerprintDimension, PrimeRegistry, TypeConstructorRegistry, _collect_base_atoms, _collect_constructors, SynthRegistry, build_synth_registry, build_fingerprint_registry, build_synth_registry_from_payload, bundle_fingerprint_dimensional, format_fingerprint, fingerprint_carrier_soundness, fingerprint_identity_payload, synth_registry_payload)

from gabion.analysis.core.forest_spec import (
    ForestSpec, build_forest_spec, default_forest_spec, forest_spec_metadata)

from gabion.analysis.foundation.timeout_context import (
    Deadline, GasMeter, TimeoutExceeded, TimeoutTickCarrier, build_timeout_context_from_stack, check_deadline, deadline_loop_iter, deadline_clock_scope, deadline_scope, forest_scope, reset_forest, set_forest)

from gabion.analysis.foundation.resume_codec import (
    allowed_path_lookup, int_str_pairs_from_sequence, int_tuple4_or_none, iter_valid_key_entries, load_resume_map, load_allowed_paths_from_sequence, mapping_payload, mapping_sections, mapping_or_empty, mapping_or_none, payload_with_format, payload_with_phase, sequence_or_none, str_list_from_sequence, str_map_from_mapping, str_pair_set_from_sequence, str_set_from_sequence, str_tuple_from_sequence)

from gabion.analysis.indexed_scan.index.analysis_carriers import AnalysisResult, ReportCarrier

from gabion.analysis.projection.projection_registry import (
    DEADLINE_OBLIGATIONS_SUMMARY_SPEC, LINT_FINDINGS_SPEC, NEVER_INVARIANTS_SPEC, REPORT_SECTION_LINES_SPEC, WL_REFINEMENT_SPEC)

from gabion.analysis.core.deprecated_substrate import (
    DeprecatedExtractionArtifacts, DeprecatedFiber, detect_report_section_extinction)

from gabion.analysis.dataflow.engine.dataflow_decision_surfaces import (
    compute_fingerprint_coherence as _ds_compute_fingerprint_coherence, compute_fingerprint_rewrite_plans as _ds_compute_fingerprint_rewrite_plans, extract_smell_sample as _ds_extract_smell_sample, lint_lines_from_bundle_evidence as _ds_lint_lines_from_bundle_evidence, lint_lines_from_constant_smells as _ds_lint_lines_from_constant_smells, lint_lines_from_type_evidence as _ds_lint_lines_from_type_evidence, lint_lines_from_unused_arg_smells as _ds_lint_lines_from_unused_arg_smells, parse_lint_location as _ds_parse_lint_location, summarize_coherence_witnesses as _ds_summarize_coherence_witnesses, summarize_deadness_witnesses as _ds_summarize_deadness_witnesses, summarize_rewrite_plans as _ds_summarize_rewrite_plans)
from gabion.analysis.dataflow.engine.dataflow_deadline_contracts import (
    _CalleeResolutionOutcome as _CalleeResolutionOutcome_owner,
    _DeadlineFunctionFacts as _DeadlineFunctionFacts_owner,
    _DeadlineLocalInfo as _DeadlineLocalInfo_owner,
    _DeadlineLoopFacts as _DeadlineLoopFacts_owner,
)
from gabion.analysis.dataflow.engine.dataflow_deadline_collector import (
    make_deadline_function_collector,
)
from gabion.analysis.dataflow.engine.dataflow_bundle_merge import (
    _merge_counts_by_knobs,
)
from gabion.analysis.dataflow.engine.dataflow_callee_resolution_support import (
    _callee_key as _callee_key_owner,
    _resolve_class_candidates,
    _resolve_method_in_hierarchy,
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
from gabion.analysis.dataflow.engine.dataflow_adapter_contract import (
    AdapterCapabilities,
    normalize_adapter_contract,
    parse_adapter_capabilities,
)
from gabion.analysis.dataflow.engine.dataflow_evidence_helpers import (
    _base_identifier as _base_identifier_owner,
    _collect_module_exports as _collect_module_exports_owner,
    _is_test_path as _is_test_path_owner,
    _module_name as _module_name_owner,
    _string_list as _string_list_owner,
    _target_names as _target_names_owner,
)
from gabion.analysis.dataflow.engine.dataflow_function_semantics import (
    _analyze_function,
    _callee_name as _callee_name_owner,
    _call_context,
    _collect_return_aliases,
    _const_repr,
    _normalize_callee as _normalize_callee_owner,
    _normalize_key_expr,
    _return_aliases,
)
from gabion.analysis.dataflow.engine.dataflow_function_index_runtime_support import (
    _direct_lambda_callee_by_call_span as _direct_lambda_callee_by_call_span_owner,
    _materialize_direct_lambda_callees as _materialize_direct_lambda_callees_owner,
    _unused_params as _unused_params_owner,
)
from gabion.analysis.dataflow.engine.dataflow_lambda_runtime_support import (
    _collect_closure_lambda_factories as _collect_closure_lambda_factories_owner,
    _collect_lambda_bindings_by_caller as _collect_lambda_bindings_by_caller_owner,
    _collect_lambda_function_infos as _collect_lambda_function_infos_owner,
    _function_key as _function_key_owner,
    _synthetic_lambda_name as _synthetic_lambda_name_owner,
)
from gabion.analysis.dataflow.engine.dataflow_function_index_decision_support import (
    _collect_param_roots as _collect_param_roots_owner,
    _contains_boolish as _contains_boolish_owner,
    _decorator_name as _decorator_name_owner,
    _decision_surface_form_entries as _decision_surface_form_entries_owner,
    _decision_surface_params as _decision_surface_params_owner,
    _decision_surface_reason_map as _decision_surface_reason_map_owner,
    _decision_root_name as _decision_root_name_owner,
    _decorators_transparent as _decorators_transparent_owner,
    _mark_param_roots as _mark_param_roots_owner,
    _value_encoded_decision_params as _value_encoded_decision_params_owner,
    is_decision_surface as _is_decision_surface_owner,
)
from gabion.analysis.dataflow.engine.dataflow_function_index_helpers import (
    _enclosing_class_runtime as _enclosing_class_owner,
    _enclosing_class_scopes_runtime as _enclosing_class_scopes_owner,
    _enclosing_function_scopes_runtime as _enclosing_function_scopes_owner,
    _enclosing_scopes_runtime as _enclosing_scopes_owner,
    _node_span_runtime as _node_span_owner,
    _param_annotations_runtime as _param_annotations_owner,
    _param_defaults_runtime as _param_defaults_owner,
    _param_names_runtime as _param_names_owner,
    _param_spans_runtime as _param_spans_owner,
)
from gabion.analysis.dataflow.engine.dataflow_call_graph_algorithms import (
    _collect_recursive_functions,
    _collect_recursive_nodes,
    _reachable_from_roots,
    _sorted_graph_nodes,
)
from gabion.analysis.dataflow.engine.dataflow_lint_helpers import (
    _constant_smells_from_details as _constant_smells_from_details_owner,
    _deadness_witnesses_from_constant_details as _deadness_witnesses_from_constant_details_owner,
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
)
from gabion.analysis.dataflow.engine.dataflow_deadline_helpers import (
    _is_deadline_annot as _is_deadline_annot_owner,
    _is_deadline_param as _is_deadline_param_owner,
)
from gabion.analysis.dataflow.engine.dataflow_local_class_hierarchy import (
    _collect_local_class_bases as _collect_local_class_bases_owner,
    _local_class_name as _local_class_name_owner,
    _resolve_local_method_in_hierarchy as _resolve_local_method_in_hierarchy_owner,
)
from gabion.analysis.dataflow.engine.dataflow_resume_paths import (
    normalize_snapshot_path as _normalize_snapshot_path_impl,
)
from gabion.analysis.dataflow.engine.dataflow_resume_serialization import (
    _ANALYSIS_INDEX_RESUME_MAX_VARIANTS,
    _ANALYSIS_INDEX_RESUME_VARIANTS_KEY,
    _CACHE_IDENTITY_DIGEST_HEX,
    _CACHE_IDENTITY_PREFIX,
    _CacheIdentity,
    _ResumeCacheIdentityPair,
    _analysis_index_resume_variant_payload,
    _analysis_index_resume_variants,
    _build_analysis_collection_resume_payload,
    _compute_invariant_evidence_key,
    _compute_invariant_id,
    _deserialize_bundle_sites_for_resume,
    _deserialize_call_args,
    _deserialize_call_args_list,
    _deserialize_class_info_for_resume,
    _deserialize_function_info_for_resume,
    _deserialize_groups_for_resume,
    _deserialize_invariants_for_resume,
    _deserialize_param_spans_for_resume,
    _deserialize_param_use,
    _deserialize_param_use_map,
    _deserialize_symbol_table_for_resume,
    _empty_analysis_collection_resume_payload,
    _empty_file_scan_resume_state,
    _invariant_confidence,
    _invariant_digest,
    _load_analysis_collection_resume_payload,
    _load_analysis_index_resume_payload as _load_analysis_index_resume_payload_owner,
    _load_file_scan_resume_state,
    _serialize_analysis_index_resume_payload,
    _serialize_bundle_sites_for_resume,
    _serialize_call_args,
    _serialize_call_args_list,
    _serialize_class_info_for_resume,
    _serialize_file_scan_resume_state,
    _serialize_function_info_for_resume,
    _serialize_groups_for_resume,
    _serialize_invariants_for_resume,
    _serialize_param_spans_for_resume,
    _serialize_param_use,
    _serialize_param_use_map,
    _serialize_symbol_table_for_resume,
    _normalize_invariant_proposition,
    _with_analysis_index_resume_variants,
)
from gabion.analysis.dataflow.engine.dataflow_post_phase_analyses import (
    ConstantFlowDetail as _ConstantFlowDetail_owner,
    _ConstantFlowFoldAccumulator,
    _analyze_decision_surface_indexed as _analyze_decision_surface_indexed_owner,
    _analyze_decision_surfaces_indexed as _analyze_decision_surfaces_indexed_owner,
    _analyze_value_encoded_decisions_indexed as _analyze_value_encoded_decisions_indexed_owner,
    _annotation_exception_candidates,
    _boundary_tier_obligation as _boundary_tier_obligation_owner,
    _dataclass_registry_for_tree as _dataclass_registry_for_tree_owner,
    _exception_handler_compatibility,
    _exception_param_names,
    _exception_path_id,
    _exception_type_name,
    _find_handling_try,
    _handler_label,
    _handler_type_names,
    _is_marker_call as _is_marker_call_owner,
    _is_never_marker_raise as _is_never_marker_raise_owner,
    _keyword_links_literal,
    _keyword_string_literal,
    _never_reason,
    _node_in_try_body,
    _refine_exception_name_from_annotations,
    _decorator_matches as _decorator_matches_owner,
    _build_property_hook_callable_index,
    _callsite_evidence_for_bundle,
    _collect_config_bundles,
    _collect_constant_flow_details,
    _collect_dataclass_registry,
    _dead_env_map,
    _decision_predicate_evidence as _decision_predicate_evidence_owner,
    _decision_reason_summary as _decision_reason_summary_owner,
    _decision_surface_alt_evidence as _decision_surface_alt_evidence_owner,
    _infer_type_flow as _infer_type_flow_owner,
    _decision_param_lint_line as _decision_param_lint_line_owner,
    _decision_tier_for as _decision_tier_for_owner,
    _branch_reachability_under_env,
    _analyze_unused_arg_flow_indexed as _analyze_unused_arg_flow_indexed_owner,
    _collect_exception_obligations,
    _collect_handledness_witnesses,
    _collect_invariant_propositions,
    _collect_never_invariants,
    _combine_type_hints,
    _DecisionSurfaceSpec as _DecisionSurfaceSpec_owner,
    _DIRECT_DECISION_SURFACE_SPEC as _DIRECT_DECISION_SURFACE_SPEC_owner,
    _enclosing_function_node,
    _eval_bool_expr,
    _eval_value_expr,
    _compute_knob_param_names,
    _expand_type_hint,
    _format_call_site,
    _format_type_flow_site,
    _is_reachability_false,
    _is_reachability_true,
    _iter_config_fields,
    _iter_dataclass_call_bundles,
    _lint_line as _lint_line_owner,
    _names_in_expr,
    _node_in_block,
    _param_annotations_by_path,
    _parse_module_source as _parse_module_source_owner,
    _simple_store_name as _simple_store_name_owner,
    _span_line_col as _span_line_col_owner,
    _split_top_level,
    _StageCacheSpec as _StageCacheSpec_owner,
    _suite_site_label as _suite_site_label_owner,
    _type_from_const_repr,
    _VALUE_DECISION_SURFACE_SPEC as _VALUE_DECISION_SURFACE_SPEC_owner,
    _ResolvedEdgeReducerSpec as _ResolvedEdgeReducerSpec_owner,
    analyze_decision_surfaces_repo as _analyze_decision_surfaces_repo_owner,
    analyze_constant_flow_repo,
    analyze_deadness_flow_repo,
    analyze_type_flow_repo as _analyze_type_flow_repo_owner,
    analyze_unused_arg_flow_repo,
    analyze_value_encoded_decisions_repo as _analyze_value_encoded_decisions_repo_owner,
    generate_property_hook_manifest,
    run_scan_domain_orchestrator as _run_scan_domain_orchestrator_owner,
)
from gabion.analysis.dataflow.engine.dataflow_projection_materialization import (
    _AmbiguitySuiteRow,
    CallAmbiguity as _CallAmbiguity_owner,
    _ProjectionSpan,
    _add_interned_alt as _add_interned_alt_owner,
    _ambiguity_suite_relation,
    _ambiguity_suite_row_to_suite,
    _ambiguity_virtual_count_gt_1,
    _collect_call_ambiguities,
    _collect_call_ambiguities_indexed,
    _decode_ambiguity_suite_row,
    _decode_projection_span,
    _dedupe_call_ambiguities,
    _emit_call_ambiguities,
    _format_span_fields as _format_span_fields_owner,
    _lint_lines_from_call_ambiguities,
    _materialize_ambiguity_suite_agg_spec,
    _materialize_ambiguity_virtual_set_spec,
    _materialize_projection_spec_rows,
    _materialize_statement_suite_contains as _materialize_statement_suite_contains_owner,
    _materialize_structured_suite_sites as _materialize_structured_suite_sites_owner,
    _materialize_structured_suite_sites_for_tree as _materialize_structured_suite_sites_for_tree_owner,
    _materialize_suite_order_spec,
    _populate_bundle_forest as _populate_bundle_forest_owner,
    _spec_row_span,
    _summarize_call_ambiguities,
    _suite_order_depth,
    _suite_order_relation,
    _suite_order_row_to_site,
)
from gabion.analysis.dataflow.engine.dataflow_documented_bundles import (
    _iter_documented_bundles as _iter_documented_bundles_owner,
)
from gabion.analysis.dataflow.engine.dataflow_ingest_helpers import (
    _collect_functions as _collect_functions_owner,
    _iter_paths as _iter_paths_owner,
)
from gabion.analysis.dataflow.engine.dataflow_ingested_analysis_support import (
    _adapt_ingest_carrier_to_analysis_maps as _adapt_ingest_carrier_to_analysis_maps_owner,
    _group_by_signature as _group_by_signature_owner,
    _propagate_groups as _propagate_groups_owner,
    _union_groups as _union_groups_owner,
    analyze_ingested_file as _analyze_ingested_file_owner,
)
from gabion.analysis.dataflow.engine.dataflow_analysis_index_owner import (
    _ANALYSIS_INDEX_STAGE_CACHE_OP as _ANALYSIS_INDEX_STAGE_CACHE_OP_owner,
    _AnalysisIndexCarrier as _AnalysisIndex_owner,
    _PhaseWorkProgress as _PhaseWorkProgress_owner,
    OptionalAnalysisIndex,
    _CacheSemanticContext,
    _EMPTY_CACHE_SEMANTIC_CONTEXT,
    _FunctionIndexAccumulator as _FunctionIndexAccumulator_owner,
    _IndexedPassContext,
    _IndexedPassSpec,
    _ModuleArtifactSpec,
    _ResolvedCallEdge as _ResolvedCallEdge_owner,
    _ResolvedEdgeParamEvent as _ResolvedEdgeParamEvent_owner,
    _StageCacheIdentitySpec,
    _analysis_index_module_trees,
    _analysis_index_resolved_call_edges,
    _analysis_index_resolved_call_edges_by_caller,
    _analysis_index_stage_cache,
    _analysis_index_transitive_callers,
    analyze_file as _analyze_file_owner,
    _analyze_file_internal as _analyze_file_internal_owner,
    _accumulate_class_index_for_tree_runtime as _accumulate_class_index_for_tree_owner,
    _accumulate_function_index_for_tree_runtime as _accumulate_function_index_for_tree_owner,
    _accumulate_symbol_table_for_tree_runtime as _accumulate_symbol_table_for_tree_owner,
    _build_analysis_index,
    _build_function_index_runtime as _build_function_index_owner,
    _build_symbol_table_runtime as _build_symbol_table_owner,
    _build_module_artifacts,
    _build_call_graph,
    _build_stage_cache_identity_spec,
    _cache_identity_aliases,
    _canonical_cache_identity,
    _canonical_stage_cache_detail,
    _canonical_stage_cache_identity,
    _collect_transitive_callers,
    _get_stage_cache_bucket,
    _index_stage_cache_identity,
    _iter_monotonic_paths_owner,
    _iter_resolved_edge_param_events,
    _function_index_module_artifact_spec_runtime as _function_index_module_artifact_spec_owner,
    _symbol_table_module_artifact_spec_runtime as _symbol_table_module_artifact_spec_owner,
    _normalize_cache_config,
    _parse_stage_cache_key,
    _path_dependency_payload as _path_dependency_payload_owner,
    _projection_stage_cache_identity,
    _reduce_resolved_call_edges,
    _resume_variant_for_identity,
    _run_indexed_pass,
    _phase_work_progress_owner,
    _profiling_v1_payload_owner,
    _sorted_text,
    _stage_cache_key_aliases,
)
from gabion.analysis.dataflow.engine.dataflow_deadline_runtime_owner import (
    _DeadlineArgInfo as _DeadlineArgInfo_owner,
    _FunctionSuiteKey as _FunctionSuiteKey_owner,
    _FunctionSuiteLookupOutcome as _FunctionSuiteLookupOutcome_owner,
    _FunctionSuiteLookupStatus as _FunctionSuiteLookupStatus_owner,
    _bind_call_args as _bind_call_args_owner,
    _call_candidate_target_site as _call_candidate_target_site_owner,
    _caller_param_bindings_for_call as _caller_param_bindings_for_call_owner,
    _classify_deadline_expr as _classify_deadline_expr_owner,
    _call_nodes_for_tree as _call_nodes_for_tree_owner,
    _collect_call_edges as _collect_call_edges_owner,
    _collect_call_edges_from_forest as _collect_call_edges_from_forest_owner,
    _collect_call_nodes_by_path as _collect_call_nodes_by_path_owner,
    _collect_deadline_function_facts as _collect_deadline_function_facts_owner,
    _collect_deadline_local_info as _collect_deadline_local_info_owner,
    _collect_call_resolution_obligation_details_from_forest as _collect_call_resolution_obligation_details_from_forest_owner,
    _collect_call_resolution_obligations_from_forest as _collect_call_resolution_obligations_from_forest_owner,
    _deadline_arg_info_map as _deadline_arg_info_map_owner,
    _deadline_loop_forwarded_params as _deadline_loop_forwarded_params_owner,
    _deadline_function_facts_for_tree as _deadline_function_facts_for_tree_owner,
    _dedupe_resolution_candidates as _dedupe_resolution_candidates_owner,
    _fallback_deadline_arg_info as _fallback_deadline_arg_info_owner,
    _function_suite_id as _function_suite_id_owner,
    _function_suite_key as _function_suite_key_owner,
    _is_dynamic_dispatch_callee_key as _is_dynamic_dispatch_callee_key_owner,
    _materialize_call_candidates as _materialize_call_candidates_owner,
    _node_to_function_suite_id as _node_to_function_suite_id_owner,
    _node_to_function_suite_lookup_outcome as _node_to_function_suite_lookup_outcome_owner,
    _obligation_candidate_suite_ids as _obligation_candidate_suite_ids_owner,
    _resolve_callee as _resolve_callee_owner,
    _resolve_callee_outcome as _resolve_callee_outcome_owner,
    _suite_caller_function_id as _suite_caller_function_id_owner,
)
from gabion.analysis.dataflow.engine.dataflow_deadline_summary_owner import (
    _summarize_deadline_obligations as _summarize_deadline_obligations_owner,
)
from gabion.analysis.dataflow.io.dataflow_parse_helpers import (
    _parse_module_tree_or_none as _parse_module_tree_owner,
)
from gabion.analysis.dataflow.engine.dataflow_runtime_reporting_owner import (
    ReportProjectionSpec as _ReportProjectionSpec_owner,
    _compute_violations as _compute_violations_owner,
    _report_section_identity_render as _report_section_identity_render_owner,
    _report_section_no_violations as _report_section_no_violations_owner,
    _report_section_spec as _report_section_spec_owner,
    _report_section_text as _report_section_text_owner,
)
from gabion.analysis.dataflow.io.dataflow_projection_helpers import (
    _topologically_order_report_projection_specs,
)
from gabion.analysis.dataflow.engine.dataflow_contracts import (
    AuditConfig as _ContractAuditConfig,
    CallArgs as _ContractCallArgs,
    ClassInfo as _ContractClassInfo,
    FunctionInfo as _ContractFunctionInfo,
    InvariantProposition,
    ParamUse as _ContractParamUse,
    SymbolTable as _ContractSymbolTable,
)

from gabion.analysis.dataflow.io.dataflow_reporting import (
    emit_report as _emit_report_owner,
    render_report,
)
from gabion.analysis.dataflow.io.dataflow_reporting_helpers import (
    render_mermaid_component as _render_mermaid_component,
)
from gabion.analysis.dataflow.io.dataflow_parse_helpers import (
    _ParseModuleStage,
    _forbid_adhoc_bundle_discovery as _forbid_adhoc_bundle_discovery_owner,
)
from gabion.analysis.indexed_scan.deadline.deadline_runtime import (
    is_deadline_origin_call as _is_deadline_origin_call_impl,
)
from gabion.analysis.indexed_scan.scanners.report_sections import (
    extract_report_sections as _extract_report_sections_impl, parse_report_section_marker as _parse_report_section_marker_impl)
from gabion.analysis.indexed_scan.scanners.parser_builder import (
    build_parser as _build_parser_impl)
from gabion.analysis.indexed_scan.scanners.run_entry import (
    analysis_deadline_scope as _analysis_deadline_scope_impl, normalize_transparent_decorators as _normalize_transparent_decorators_impl, resolve_baseline_path as _resolve_baseline_path_impl, resolve_synth_registry_path as _resolve_synth_registry_path_impl)

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef

OptionalIgnoredParams = set[str] | None

ParamAnnotationMap = dict[str, str | None]

ReturnAliasMap = dict[str, tuple[list[str], list[str]]]

OptionalReturnAliasMap = ReturnAliasMap | None

OptionalClassName = str | None

Span4 = tuple[int, int, int, int]

OptionalSpan4 = Span4 | None

OptionalString = str | None

OptionalFloat = float | None

OptionalPath = Path | None

OptionalStringSet = set[str] | None

OptionalPrimeRegistry = PrimeRegistry | None

OptionalTypeConstructorRegistry = TypeConstructorRegistry | None

OptionalSynthRegistry = SynthRegistry | None

OptionalJsonObject = JSONObject | None

OptionalForestSpec = ForestSpec | None

OptionalDeprecatedExtractionArtifacts = DeprecatedExtractionArtifacts | None

OptionalAstCall = ast.Call | None

NodeIdOrNone = NodeId | None

ParseCacheValue = ast.Module | BaseException

ReportProjectionPhase = Literal["collection", "forest", "edge", "post"]

_PhaseWorkProgress = _PhaseWorkProgress_owner

_phase_work_progress = _phase_work_progress_owner

_FunctionSuiteKey = _FunctionSuiteKey_owner

ParamUse = _ContractParamUse

CallArgs = _ContractCallArgs

# Canonical owner contract class (WS-5 hard-cut compatibility).
SymbolTable = _ContractSymbolTable

# Canonical owner contract class (WS-5 hard-cut compatibility).
AuditConfig = _ContractAuditConfig

_summarize_deadline_obligations = _summarize_deadline_obligations_owner


_profiling_v1_payload = _profiling_v1_payload_owner

ReportProjectionSpec = _ReportProjectionSpec_owner

_report_section_identity_render = _report_section_identity_render_owner

_report_section_no_violations = _report_section_no_violations_owner

_report_section_text = _report_section_text_owner

_report_section_spec = _report_section_spec_owner

CallAmbiguity = _CallAmbiguity_owner

_callee_name = _callee_name_owner

_normalize_callee = _normalize_callee_owner

_iter_paths = _iter_paths_owner

_collect_functions = _collect_functions_owner

_decorator_name = _decorator_name_owner

_decorator_matches = _decorator_matches_owner

_is_marker_call = _is_marker_call_owner

_is_never_marker_raise = _is_never_marker_raise_owner

_decorators_transparent = _decorators_transparent_owner

_collect_local_class_bases = _collect_local_class_bases_owner
_local_class_name = _local_class_name_owner
_resolve_local_method_in_hierarchy = _resolve_local_method_in_hierarchy_owner

_param_names = _param_names_owner

_decision_root_name = _decision_root_name_owner

is_decision_surface = _is_decision_surface_owner

_decision_surface_form_entries = _decision_surface_form_entries_owner

_decision_surface_reason_map = _decision_surface_reason_map_owner

_decision_surface_params = _decision_surface_params_owner

_mark_param_roots = _mark_param_roots_owner

_collect_param_roots = _collect_param_roots_owner

_contains_boolish = _contains_boolish_owner

_value_encoded_decision_params = _value_encoded_decision_params_owner

_DecisionSurfaceSpec = _DecisionSurfaceSpec_owner

_decision_predicate_evidence = _decision_predicate_evidence_owner

_decision_reason_summary = _decision_reason_summary_owner

_boundary_tier_obligation = _boundary_tier_obligation_owner

_decision_surface_alt_evidence = _decision_surface_alt_evidence_owner

_suite_site_label = _suite_site_label_owner

_DIRECT_DECISION_SURFACE_SPEC = _DIRECT_DECISION_SURFACE_SPEC_owner

_VALUE_DECISION_SURFACE_SPEC = _VALUE_DECISION_SURFACE_SPEC_owner

_analyze_decision_surface_indexed = _analyze_decision_surface_indexed_owner

_analyze_decision_surfaces_indexed = _analyze_decision_surfaces_indexed_owner

_analyze_value_encoded_decisions_indexed = _analyze_value_encoded_decisions_indexed_owner

_node_span = _node_span_owner

_param_spans = _param_spans_owner

_function_key = _function_key_owner

_enclosing_class = _enclosing_class_owner

_enclosing_scopes = _enclosing_scopes_owner

_enclosing_class_scopes = _enclosing_class_scopes_owner

_enclosing_function_scopes = _enclosing_function_scopes_owner

_param_annotations = _param_annotations_owner

_param_defaults = _param_defaults_owner

_ANALYSIS_INDEX_STAGE_CACHE_OP = _ANALYSIS_INDEX_STAGE_CACHE_OP_owner

_path_dependency_payload = _path_dependency_payload_owner

_parse_module_tree = _parse_module_tree_owner

_is_deadline_annot = _is_deadline_annot_owner

_is_deadline_param = _is_deadline_param_owner

_is_deadline_origin_call = _is_deadline_origin_call_impl

_target_names = _target_names_owner

_simple_store_name = _simple_store_name_owner

_DeadlineLoopFacts = _DeadlineLoopFacts_owner
_DeadlineLocalInfo = _DeadlineLocalInfo_owner
_DeadlineFunctionFacts = _DeadlineFunctionFacts_owner
_DeadlineFunctionCollector = make_deadline_function_collector(
    node_span_fn=_node_span,
    check_deadline_fn=check_deadline,
    deadline_loop_facts_ctor=_DeadlineLoopFacts,
)
_collect_deadline_local_info = _collect_deadline_local_info_owner
_collect_deadline_function_facts = _collect_deadline_function_facts_owner
_deadline_function_facts_for_tree = _deadline_function_facts_for_tree_owner
_collect_call_nodes_by_path = _collect_call_nodes_by_path_owner
_call_nodes_for_tree = _call_nodes_for_tree_owner
_collect_call_edges = _collect_call_edges_owner
_FunctionSuiteKey = _FunctionSuiteKey_owner
_FunctionSuiteLookupStatus = _FunctionSuiteLookupStatus_owner
_FunctionSuiteLookupOutcome = _FunctionSuiteLookupOutcome_owner
_function_suite_key = _function_suite_key_owner
_function_suite_id = _function_suite_id_owner
_node_to_function_suite_lookup_outcome = _node_to_function_suite_lookup_outcome_owner
_suite_caller_function_id = _suite_caller_function_id_owner
_node_to_function_suite_id = _node_to_function_suite_id_owner
_obligation_candidate_suite_ids = _obligation_candidate_suite_ids_owner
_collect_call_edges_from_forest = _collect_call_edges_from_forest_owner
_collect_call_resolution_obligations_from_forest = _collect_call_resolution_obligations_from_forest_owner
_collect_call_resolution_obligation_details_from_forest = _collect_call_resolution_obligation_details_from_forest_owner
_call_candidate_target_site = _call_candidate_target_site_owner
_materialize_call_candidates = _materialize_call_candidates_owner

_DeadlineArgInfo = _DeadlineArgInfo_owner
_bind_call_args = _bind_call_args_owner
_caller_param_bindings_for_call = _caller_param_bindings_for_call_owner
_classify_deadline_expr = _classify_deadline_expr_owner
_fallback_deadline_arg_info = _fallback_deadline_arg_info_owner
_deadline_arg_info_map = _deadline_arg_info_map_owner
_deadline_loop_forwarded_params = _deadline_loop_forwarded_params_owner

run_scan_domain_orchestrator = _run_scan_domain_orchestrator_owner

analyze_decision_surfaces_repo = _analyze_decision_surfaces_repo_owner

analyze_value_encoded_decisions_repo = _analyze_value_encoded_decisions_repo_owner

_span_line_col = _span_line_col_owner

_infer_type_flow = _infer_type_flow_owner

_analyze_unused_arg_flow_indexed = _analyze_unused_arg_flow_indexed_owner

_format_span_fields = _format_span_fields_owner

_lint_line = _lint_line_owner

_add_interned_alt = _add_interned_alt_owner

_decision_param_lint_line = _decision_param_lint_line_owner

_decision_tier_for = _decision_tier_for_owner

AnalysisIndex = _AnalysisIndex_owner

_ResolvedCallEdge = _ResolvedCallEdge_owner

_ResolvedEdgeReducerSpec = _ResolvedEdgeReducerSpec_owner

_ResolvedEdgeParamEvent = _ResolvedEdgeParamEvent_owner

_StageCacheSpec = _StageCacheSpec_owner

_parse_module_source = _parse_module_source_owner


_forbid_adhoc_bundle_discovery = _forbid_adhoc_bundle_discovery_owner

_materialize_statement_suite_contains = _materialize_statement_suite_contains_owner
_materialize_structured_suite_sites_for_tree = _materialize_structured_suite_sites_for_tree_owner
_materialize_structured_suite_sites = _materialize_structured_suite_sites_owner
_populate_bundle_forest = _populate_bundle_forest_owner

_is_test_path = _is_test_path_owner

_unused_params = _unused_params_owner

_group_by_signature = _group_by_signature_owner
_union_groups = _union_groups_owner
_propagate_groups = _propagate_groups_owner
_adapt_ingest_carrier_to_analysis_maps = _adapt_ingest_carrier_to_analysis_maps_owner
analyze_ingested_file = _analyze_ingested_file_owner

_analyze_file_internal = _analyze_file_internal_owner

analyze_file = _analyze_file_owner

_callee_key = _callee_key_owner

# Canonical owner contract class (WS-5 hard-cut compatibility).
FunctionInfo = _ContractFunctionInfo

# Canonical owner contract class (WS-5 hard-cut compatibility).
ClassInfo = _ContractClassInfo

_module_name = _module_name_owner

_string_list = _string_list_owner

_base_identifier = _base_identifier_owner

_collect_module_exports = _collect_module_exports_owner

_accumulate_symbol_table_for_tree = _accumulate_symbol_table_for_tree_owner

_symbol_table_module_artifact_spec = _symbol_table_module_artifact_spec_owner

_build_symbol_table = _build_symbol_table_owner

_accumulate_class_index_for_tree = _accumulate_class_index_for_tree_owner

_FunctionIndexAccumulator = _FunctionIndexAccumulator_owner

_accumulate_function_index_for_tree = _accumulate_function_index_for_tree_owner

_synthetic_lambda_name = _synthetic_lambda_name_owner
_collect_lambda_function_infos = _collect_lambda_function_infos_owner
_collect_lambda_bindings_by_caller = _collect_lambda_bindings_by_caller_owner
_collect_closure_lambda_factories = _collect_closure_lambda_factories_owner
_direct_lambda_callee_by_call_span = _direct_lambda_callee_by_call_span_owner
_materialize_direct_lambda_callees = _materialize_direct_lambda_callees_owner

_function_index_module_artifact_spec = _function_index_module_artifact_spec_owner

_build_function_index = _build_function_index_owner

_resolve_callee = _resolve_callee_owner

_is_dynamic_dispatch_callee_key = _is_dynamic_dispatch_callee_key_owner

_CalleeResolutionOutcome = _CalleeResolutionOutcome_owner

_dedupe_resolution_candidates = _dedupe_resolution_candidates_owner

_resolve_callee_outcome = _resolve_callee_outcome_owner

analyze_type_flow_repo = _analyze_type_flow_repo_owner

ConstantFlowDetail = _ConstantFlowDetail_owner

_constant_smells_from_details = _constant_smells_from_details_owner

_deadness_witnesses_from_constant_details = (
    _deadness_witnesses_from_constant_details_owner
)


_iter_documented_bundles = _iter_documented_bundles_owner

_dataclass_registry_for_tree = _dataclass_registry_for_tree_owner

_parse_report_section_marker = _parse_report_section_marker_impl

_emit_report = _emit_report_owner

extract_report_sections = _extract_report_sections_impl

_normalize_snapshot_path = _normalize_snapshot_path_impl

_FILE_SCAN_PROGRESS_EMIT_INTERVAL = 1

_PROGRESS_EMIT_MIN_INTERVAL_SECONDS = 1.0

_iter_monotonic_paths = _iter_monotonic_paths_owner

_load_analysis_index_resume_payload = _load_analysis_index_resume_payload_owner

_compute_violations = _compute_violations_owner

_resolve_baseline_path = _resolve_baseline_path_impl

_resolve_synth_registry_path = _resolve_synth_registry_path_impl

_build_parser = _build_parser_impl

_normalize_transparent_decorators = _normalize_transparent_decorators_impl

_analysis_deadline_scope = _analysis_deadline_scope_impl
