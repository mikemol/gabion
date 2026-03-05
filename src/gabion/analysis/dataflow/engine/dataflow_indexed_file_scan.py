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
    compute_fingerprint_coherence as _ds_compute_fingerprint_coherence, compute_fingerprint_rewrite_plans as _ds_compute_fingerprint_rewrite_plans, lint_lines_from_bundle_evidence as _ds_lint_lines_from_bundle_evidence, lint_lines_from_constant_smells as _ds_lint_lines_from_constant_smells, lint_lines_from_type_evidence as _ds_lint_lines_from_type_evidence, lint_lines_from_unused_arg_smells as _ds_lint_lines_from_unused_arg_smells, parse_lint_location as _ds_parse_lint_location, summarize_coherence_witnesses as _ds_summarize_coherence_witnesses, summarize_deadness_witnesses as _ds_summarize_deadness_witnesses, summarize_rewrite_plans as _ds_summarize_rewrite_plans)
from gabion.analysis.dataflow.engine.dataflow_deadline_contracts import (
    _CalleeResolutionOutcome,
    _DeadlineFunctionFacts,
    _DeadlineLocalInfo,
    _DeadlineLoopFacts,
)
from gabion.analysis.dataflow.engine.dataflow_deadline_collector import (
    make_deadline_function_collector,
)
from gabion.analysis.dataflow.engine.dataflow_bundle_merge import (
    _merge_counts_by_knobs,
)
from gabion.analysis.dataflow.engine.dataflow_callee_resolution_support import (
    _callee_key,
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
    _base_identifier,
    _collect_module_exports,
    _is_test_path,
    _module_name,
    _string_list,
    _target_names,
)
from gabion.analysis.dataflow.engine.dataflow_function_semantics import (
    _analyze_function,
    _callee_name,
    _call_context,
    _collect_return_aliases,
    _const_repr,
    _normalize_callee,
    _normalize_key_expr,
)
from gabion.analysis.dataflow.engine.dataflow_function_index_runtime_support import (
    _direct_lambda_callee_by_call_span,
    _materialize_direct_lambda_callees,
    _unused_params,
)
from gabion.analysis.dataflow.engine.dataflow_lambda_runtime_support import (
    _collect_lambda_bindings_by_caller,
    _collect_lambda_function_infos,
    _function_key,
)
from gabion.analysis.dataflow.engine.dataflow_function_index_decision_support import (
    _collect_param_roots,
    _contains_boolish,
    _decorator_name,
    _decision_surface_form_entries,
    _decision_surface_params,
    _decision_surface_reason_map,
    _decision_root_name,
    _decorators_transparent,
    _mark_param_roots,
    _value_encoded_decision_params,
    is_decision_surface,
)
from gabion.analysis.dataflow.engine.dataflow_function_index_helpers import (
    _enclosing_class_runtime as _enclosing_class,
    _enclosing_class_scopes_runtime as _enclosing_class_scopes,
    _enclosing_function_scopes_runtime as _enclosing_function_scopes,
    _enclosing_scopes_runtime as _enclosing_scopes,
    _node_span_runtime as _node_span,
    _param_annotations_runtime as _param_annotations,
    _param_defaults_runtime as _param_defaults,
    _param_names_runtime as _param_names,
    _param_spans_runtime as _param_spans,
)
from gabion.analysis.dataflow.engine.dataflow_call_graph_algorithms import (
    _collect_recursive_functions,
    _collect_recursive_nodes,
    _reachable_from_roots,
)
from gabion.analysis.dataflow.engine.dataflow_lint_helpers import (
    _constant_smells_from_details,
    _deadness_witnesses_from_constant_details,
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
    _is_deadline_annot,
    _is_deadline_param,
)
from gabion.analysis.dataflow.engine.dataflow_local_class_hierarchy import (
    _collect_local_class_bases,
    _local_class_name,
    _resolve_local_method_in_hierarchy,
)
from gabion.analysis.dataflow.engine.dataflow_resume_paths import (
    normalize_snapshot_path as _normalize_snapshot_path,
)
from gabion.analysis.dataflow.engine.dataflow_resume_serialization import (
    _ANALYSIS_INDEX_RESUME_MAX_VARIANTS,
    _ANALYSIS_INDEX_RESUME_VARIANTS_KEY,
    _CACHE_IDENTITY_DIGEST_HEX,
    _CACHE_IDENTITY_PREFIX,
    _CacheIdentity,
    _analysis_index_resume_variants,
    _build_analysis_collection_resume_payload,
    _compute_invariant_evidence_key,
    _compute_invariant_id,
    _deserialize_function_info_for_resume,
    _deserialize_invariants_for_resume,
    _deserialize_symbol_table_for_resume,
    _invariant_confidence,
    _invariant_digest,
    _load_analysis_collection_resume_payload,
    _load_analysis_index_resume_payload,
    _load_file_scan_resume_state,
    _serialize_analysis_index_resume_payload,
    _serialize_file_scan_resume_state,
    _serialize_symbol_table_for_resume,
    _normalize_invariant_proposition,
)
from gabion.analysis.dataflow.engine.dataflow_post_phase_analyses import (
    ConstantFlowDetail,
    _ConstantFlowFoldAccumulator,
    _analyze_decision_surface_indexed,
    _analyze_decision_surfaces_indexed,
    _analyze_value_encoded_decisions_indexed,
    _annotation_exception_candidates,
    _boundary_tier_obligation,
    _dataclass_registry_for_tree,
    _exception_param_names,
    _exception_type_name,
    _handler_label,
    _handler_type_names,
    _is_marker_call,
    _is_never_marker_raise,
    _keyword_links_literal,
    _keyword_string_literal,
    _refine_exception_name_from_annotations,
    _decorator_matches,
    _build_property_hook_callable_index,
    _callsite_evidence_for_bundle,
    _collect_config_bundles,
    _collect_constant_flow_details,
    _collect_dataclass_registry,
    _dead_env_map,
    _decision_predicate_evidence,
    _decision_reason_summary,
    _decision_surface_alt_evidence,
    _infer_type_flow,
    _decision_param_lint_line,
    _decision_tier_for,
    _branch_reachability_under_env,
    _collect_exception_obligations,
    _collect_handledness_witnesses,
    _collect_invariant_propositions,
    _collect_never_invariants,
    _combine_type_hints,
    _DecisionSurfaceSpec,
    _DIRECT_DECISION_SURFACE_SPEC,
    _enclosing_function_node,
    _eval_bool_expr,
    _eval_value_expr,
    _compute_knob_param_names,
    _expand_type_hint,
    _format_call_site,
    _format_type_flow_site,
    _iter_config_fields,
    _iter_dataclass_call_bundles,
    _lint_line,
    _names_in_expr,
    _node_in_block,
    _param_annotations_by_path,
    _parse_module_source,
    _simple_store_name,
    _span_line_col,
    _split_top_level,
    _StageCacheSpec,
    _suite_site_label,
    _type_from_const_repr,
    _VALUE_DECISION_SURFACE_SPEC,
    _ResolvedEdgeReducerSpec,
    analyze_decision_surfaces_repo,
    analyze_constant_flow_repo,
    analyze_deadness_flow_repo,
    analyze_type_flow_repo,
    analyze_unused_arg_flow_repo,
    analyze_value_encoded_decisions_repo,
    generate_property_hook_manifest,
)
from gabion.analysis.dataflow.engine.dataflow_projection_materialization import (
    CallAmbiguity,
    _add_interned_alt,
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
    _materialize_statement_suite_contains,
    _materialize_structured_suite_sites,
    _materialize_structured_suite_sites_for_tree,
    _materialize_suite_order_spec,
    _populate_bundle_forest,
    _spec_row_span,
    _summarize_call_ambiguities,
    _suite_order_relation,
    _suite_order_row_to_site,
)
from gabion.analysis.dataflow.engine.dataflow_documented_bundles import (
    _iter_documented_bundles,
)
from gabion.analysis.dataflow.engine.dataflow_ingest_helpers import (
    _collect_functions,
    _iter_paths,
)
from gabion.analysis.dataflow.engine.dataflow_ingested_analysis_support import (
    _group_by_signature,
    _propagate_groups,
    _union_groups,
    analyze_ingested_file,
)
from gabion.analysis.dataflow.engine.dataflow_analysis_index_owner import (
    _EMPTY_CACHE_SEMANTIC_CONTEXT,
    _IndexedPassContext,
    _IndexedPassSpec,
    _analysis_index_module_trees,
    _analysis_index_resolved_call_edges,
    _analysis_index_resolved_call_edges_by_caller,
    _analysis_index_stage_cache,
    _analysis_index_transitive_callers,
    analyze_file,
    _analyze_file_internal,
    _accumulate_function_index_for_tree,
    _build_analysis_index,
    _build_function_index,
    _build_symbol_table,
    _build_call_graph,
    _iter_monotonic_paths,
    _iter_resolved_edge_param_events,
    _parse_stage_cache_key,
    _reduce_resolved_call_edges,
    _run_indexed_pass,
    _phase_work_progress,
    _profiling_v1_payload,
    _sorted_text,
    _stage_cache_key_aliases,
)
from gabion.analysis.dataflow.engine.dataflow_deadline_runtime_owner import (
    _DeadlineArgInfo,
    _bind_call_args,
    _classify_deadline_expr,
    _collect_call_edges,
    _collect_call_edges_from_forest,
    _collect_call_nodes_by_path,
    _collect_deadline_function_facts,
    _collect_deadline_local_info,
    _deadline_arg_info_map,
    _deadline_loop_forwarded_params,
    _fallback_deadline_arg_info,
    _is_dynamic_dispatch_callee_key,
    _materialize_call_candidates,
    _resolve_callee,
    _resolve_callee_outcome,
)
from gabion.analysis.dataflow.engine.dataflow_deadline_summary_owner import (
    _summarize_deadline_obligations,
)
from gabion.analysis.dataflow.io.dataflow_parse_helpers import (
    _parse_module_tree_or_none as _parse_module_tree,
)
from gabion.analysis.dataflow.engine.dataflow_runtime_reporting_owner import (
    ReportProjectionSpec,
    _compute_violations,
    _report_section_spec,
)
from gabion.analysis.dataflow.io.dataflow_projection_helpers import (
    _topologically_order_report_projection_specs,
)
from gabion.analysis.dataflow.engine.dataflow_contracts import (
    AuditConfig,
    CallArgs,
    ClassInfo,
    FunctionInfo,
    InvariantProposition,
    ParamUse,
    SymbolTable,
)

from gabion.analysis.dataflow.io.dataflow_reporting import (
    emit_report as _emit_report,
    render_report,
)
from gabion.analysis.dataflow.io.dataflow_reporting_helpers import (
    render_mermaid_component as _render_mermaid_component,
)
from gabion.analysis.dataflow.io.dataflow_parse_helpers import (
    _ParseModuleStage,
    _forbid_adhoc_bundle_discovery,
)
from gabion.analysis.indexed_scan.deadline.deadline_runtime import (
    is_deadline_origin_call as _is_deadline_origin_call,
)
from gabion.analysis.indexed_scan.scanners.report_sections import (
    extract_report_sections,
)
from gabion.analysis.indexed_scan.scanners.parser_builder import (
    build_parser as _build_parser,
)
from gabion.analysis.indexed_scan.scanners.run_entry import (
    analysis_deadline_scope as _analysis_deadline_scope,
    normalize_transparent_decorators as _normalize_transparent_decorators,
    resolve_baseline_path as _resolve_baseline_path,
    resolve_synth_registry_path as _resolve_synth_registry_path,
)

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

_DeadlineFunctionCollector = make_deadline_function_collector(
    node_span_fn=_node_span,
    check_deadline_fn=check_deadline,
    deadline_loop_facts_ctor=_DeadlineLoopFacts,
)

_FILE_SCAN_PROGRESS_EMIT_INTERVAL = 1

_PROGRESS_EMIT_MIN_INTERVAL_SECONDS = 1.0
