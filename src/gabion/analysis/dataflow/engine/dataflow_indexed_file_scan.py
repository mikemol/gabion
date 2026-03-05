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

import json

import hashlib

import os

import sys


from collections import Counter, defaultdict

from contextlib import ExitStack, contextmanager

from dataclasses import dataclass, field

from enum import StrEnum

from pathlib import Path

from typing import Callable, Generic, Hashable, Iterable, Iterator, Literal, Mapping, Sequence, TypeVar, cast

import re

from gabion.ingest.python_ingest import ingest_python_file, iter_python_paths

from gabion.analysis.core.visitors import ImportVisitor, ParentAnnotator, UseVisitor

from gabion.analysis.foundation.json_types import JSONObject, JSONValue

from gabion.analysis.aspf.aspf import Alt, Forest, Node, NodeId

from gabion.analysis.semantics import evidence_keys

from gabion.invariants import never, require_not_none

from gabion.order_contract import OrderPolicy, sort_once

from gabion.analysis.core.type_fingerprints import (
    Fingerprint, FingerprintDimension, PrimeRegistry, TypeConstructorRegistry, _collect_base_atoms, _collect_constructors, SynthRegistry, build_synth_registry, build_fingerprint_registry, build_synth_registry_from_payload, bundle_fingerprint_dimensional, format_fingerprint, fingerprint_carrier_soundness, fingerprint_identity_payload, synth_registry_payload)

from gabion.analysis.core.forest_spec import (
    ForestSpec, build_forest_spec, default_forest_spec, forest_spec_metadata)

from gabion.analysis.foundation.timeout_context import (
    Deadline, GasMeter, TimeoutExceeded, TimeoutTickCarrier, build_timeout_context_from_stack, check_deadline, deadline_loop_iter, deadline_clock_scope, deadline_scope, forest_scope, reset_forest, set_forest)

from gabion.analysis.projection.projection_normalize import spec_hash as projection_spec_hash

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
from gabion.analysis.dataflow.engine.dataflow_function_semantics import (
    _analyze_function,
    _call_context,
    _collect_return_aliases,
    _const_repr,
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
    _synthetic_lambda_name as _synthetic_lambda_name_owner,
)
from gabion.analysis.dataflow.engine.dataflow_function_index_decision_support import (
    _collect_param_roots as _collect_param_roots_owner,
    _contains_boolish as _contains_boolish_owner,
    _decision_surface_form_entries as _decision_surface_form_entries_owner,
    _decision_surface_params as _decision_surface_params_owner,
    _decision_surface_reason_map as _decision_surface_reason_map_owner,
    _decorators_transparent as _decorators_transparent_owner,
    _mark_param_roots as _mark_param_roots_owner,
    _value_encoded_decision_params as _value_encoded_decision_params_owner,
    is_decision_surface as _is_decision_surface_owner,
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
from gabion.analysis.dataflow.engine.dataflow_local_class_hierarchy import (
    _collect_local_class_bases as _collect_local_class_bases_owner,
    _local_class_name as _local_class_name_owner,
    _resolve_local_method_in_hierarchy as _resolve_local_method_in_hierarchy_owner,
)
from gabion.analysis.dataflow.engine.dataflow_parse_failures import (
    _PARSE_MODULE_ERROR_TYPES,
    _parse_failure_sink,
    _parse_failure_witness,
    _record_parse_failure_witness,
)
from gabion.analysis.dataflow.engine.dataflow_resume_paths import (
    iter_monotonic_paths as _iter_monotonic_paths_impl,
    normalize_snapshot_path as _normalize_snapshot_path_impl,
)
from gabion.analysis.dataflow.engine.dataflow_resume_serialization import (
    _ANALYSIS_INDEX_RESUME_MAX_VARIANTS,
    _ANALYSIS_INDEX_RESUME_VARIANTS_KEY,
    _CACHE_IDENTITY_DIGEST_HEX,
    _CACHE_IDENTITY_PREFIX,
    _CacheIdentity,
    _ResumeCacheIdentityPair,
    _analysis_collection_resume_path_key,
    _analysis_index_resume_variant_payload,
    _analysis_index_resume_variants,
    _build_analysis_collection_resume_payload,
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
    _with_analysis_index_resume_variants,
)
from gabion.analysis.dataflow.engine.dataflow_post_phase_analyses import (
    _ConstantFlowFoldAccumulator,
    _annotation_exception_candidates,
    _exception_handler_compatibility,
    _exception_param_names,
    _exception_path_id,
    _exception_type_name,
    _find_handling_try,
    _handler_label,
    _handler_type_names,
    _keyword_links_literal,
    _keyword_string_literal,
    _never_reason,
    _node_in_try_body,
    _refine_exception_name_from_annotations,
    _build_property_hook_callable_index,
    _callsite_evidence_for_bundle,
    _collect_config_bundles,
    _collect_constant_flow_details,
    _collect_dataclass_registry,
    _dead_env_map,
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
    _span_line_col as _span_line_col_owner,
    _split_top_level,
    _type_from_const_repr,
    analyze_decision_surfaces_repo as _analyze_decision_surfaces_repo_owner,
    analyze_constant_flow_repo,
    analyze_deadness_flow_repo,
    analyze_type_flow_repo_with_evidence,
    analyze_type_flow_repo_with_map,
    analyze_unused_arg_flow_repo,
    analyze_value_encoded_decisions_repo as _analyze_value_encoded_decisions_repo_owner,
    generate_property_hook_manifest,
    run_scan_domain_orchestrator as _run_scan_domain_orchestrator_owner,
)
from gabion.analysis.dataflow.engine.dataflow_projection_materialization import (
    _AmbiguitySuiteRow,
    _ProjectionSpan,
    _ambiguity_suite_relation,
    _ambiguity_suite_row_to_suite,
    _ambiguity_virtual_count_gt_1,
    _collect_call_ambiguities,
    _collect_call_ambiguities_indexed,
    _decode_ambiguity_suite_row,
    _decode_projection_span,
    _dedupe_call_ambiguities,
    _emit_call_ambiguities,
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
from gabion.analysis.dataflow.engine.dataflow_ingested_analysis_support import (
    _adapt_ingest_carrier_to_analysis_maps as _adapt_ingest_carrier_to_analysis_maps_owner,
    _group_by_signature as _group_by_signature_owner,
    _propagate_groups as _propagate_groups_owner,
    _union_groups as _union_groups_owner,
    analyze_ingested_file as _analyze_ingested_file_owner,
)
from gabion.analysis.dataflow.engine.dataflow_analysis_index_owner import (
    _ANALYSIS_INDEX_STAGE_CACHE_OP as _ANALYSIS_INDEX_STAGE_CACHE_OP_owner,
    OptionalAnalysisIndex,
    OptionalDecorators,
    OptionalParseFailures,
    OptionalProjectRoot,
    _CacheSemanticContext,
    _EMPTY_CACHE_SEMANTIC_CONTEXT,
    _IndexedPassContext,
    _IndexedPassSpec,
    _ModuleArtifactSpec,
    _StageCacheIdentitySpec,
    _analysis_index_module_trees,
    _analysis_index_resolved_call_edges,
    _analysis_index_resolved_call_edges_by_caller,
    _analysis_index_stage_cache,
    _analysis_index_transitive_callers,
    _analyze_file_internal as _analyze_file_internal_owner,
    _accumulate_class_index_for_tree_runtime as _accumulate_class_index_for_tree_owner,
    _accumulate_function_index_for_tree_runtime as _accumulate_function_index_for_tree_owner,
    _accumulate_symbol_table_for_tree_runtime as _accumulate_symbol_table_for_tree_owner,
    _build_analysis_index,
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
    _iter_resolved_edge_param_events,
    _normalize_cache_config,
    _parse_stage_cache_key,
    _path_dependency_payload as _path_dependency_payload_owner,
    _projection_stage_cache_identity,
    _reduce_resolved_call_edges,
    _resume_variant_for_identity,
    _run_indexed_pass,
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
from gabion.analysis.dataflow.io.dataflow_projection_helpers import (
    _topologically_order_report_projection_specs,
)
from gabion.analysis.semantics.semantic_primitives import (
    CallArgumentMapping, CallableId, DecisionPredicateEvidence, ParameterId, SpanIdentity)
from gabion.analysis.dataflow.engine.dataflow_contracts import InvariantProposition, ReportCarrier as _DataflowReportCarrier, SymbolTable as _ContractSymbolTable

from gabion.analysis.dataflow.io.dataflow_reporting import (
    emit_report as _emit_report,
    render_report,
)
from gabion.analysis.dataflow.io.dataflow_reporting_helpers import (
    render_mermaid_component as _render_mermaid_component,
)
from gabion.analysis.indexed_scan.deadline.deadline_runtime import (
    is_deadline_origin_call as _is_deadline_origin_call_impl,
)
from gabion.analysis.indexed_scan.deadline.deadline_obligation_summary import (
    SummarizeDeadlineObligationsDeps as _SummarizeDeadlineObligationsDeps, summarize_deadline_obligations as _summarize_deadline_obligations_impl)
from gabion.analysis.indexed_scan.scanners.report_sections import (
    extract_report_sections as _extract_report_sections_impl, parse_report_section_marker as _parse_report_section_marker_impl)
from gabion.analysis.indexed_scan.ast.expression_eval import (
    BoolEvalOutcome as _BoolEvalOutcome, EvalDecision as _EvalDecision, ValueEvalOutcome as _ValueEvalOutcome)
from gabion.analysis.indexed_scan.scanners.parser_builder import (
    build_parser as _build_parser_impl)
from gabion.analysis.indexed_scan.scanners.run_entry import (
    analysis_deadline_scope as _analysis_deadline_scope_impl, normalize_transparent_decorators as _normalize_transparent_decorators_impl, resolve_baseline_path as _resolve_baseline_path_impl, resolve_synth_registry_path as _resolve_synth_registry_path_impl)
from gabion.analysis.indexed_scan.calls.callee_resolution_helpers import (
    decorator_name as _decorator_name_impl)
from gabion.analysis.indexed_scan.scanners.materialization.dataclass_registry import (
    DataclassRegistryForTreeDeps as _DataclassRegistryForTreeDeps, dataclass_registry_for_tree as _dataclass_registry_for_tree_impl)
from gabion.analysis.indexed_scan.obligations.decision_surface_runtime import (
    DecisionSurfaceAnalyzeDeps as _DecisionSurfaceAnalyzeDeps, analyze_decision_surface_indexed as _analyze_decision_surface_indexed_impl)
from gabion.analysis.indexed_scan.state.module_exports import (
    ModuleExportsCollectDeps as _ModuleExportsCollectDeps, collect_module_exports as _collect_module_exports_impl)
_AST_UNPARSE_ERROR_TYPES = (
    AttributeError,
    TypeError,
    ValueError,
    RecursionError,
)

_LITERAL_EVAL_ERROR_TYPES = (
    SyntaxError,
    ValueError,
    TypeError,
    MemoryError,
    RecursionError,
)

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef

OptionalIgnoredParams = set[str] | None

ParamAnnotationMap = dict[str, str | None]

@dataclass(frozen=True)
class AnnotationValue:
    text: str
    parse_status: Literal["present", "missing", "unparse_failure"]

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

OptionalAstNode = ast.AST | None

OptionalAstCall = ast.Call | None

NodeIdOrNone = NodeId | None

ParseCacheValue = ast.Module | BaseException

class _ParseModuleStage(StrEnum):
    PARAM_ANNOTATIONS = "param_annotations"
    DEADLINE_FUNCTION_FACTS = "deadline_function_facts"
    CALL_NODES = "call_nodes"
    SUITE_CONTAINMENT = "suite_containment"
    SYMBOL_TABLE = "symbol_table"
    CLASS_INDEX = "class_index"
    FUNCTION_INDEX = "function_index"
    CONFIG_FIELDS = "config_fields"
    DATACLASS_REGISTRY = "dataclass_registry"
    DATACLASS_CALL_BUNDLES = "dataclass_call_bundles"
    RAW_SORTED_AUDIT = "raw_sorted_audit"

ReportProjectionPhase = Literal["collection", "forest", "edge", "post"]

@dataclass(frozen=True)
class _PhaseWorkProgress:
    work_done: int
    work_total: int

def _phase_work_progress(*, work_done: int, work_total: int) -> _PhaseWorkProgress:
    check_deadline()
    normalized_total = max(int(work_total), 0)
    normalized_done = max(int(work_done), 0)
    if normalized_total:
        normalized_done = min(normalized_done, normalized_total)
    return _PhaseWorkProgress(work_done=normalized_done, work_total=normalized_total)

_FunctionSuiteKey = _FunctionSuiteKey_owner

@dataclass
class ParamUse:
    direct_forward: set[tuple[str, str]]
    non_forward: bool
    current_aliases: set[str]
    forward_sites: dict[tuple[str, str], set[tuple[int, int, int, int]]] = field(
        default_factory=dict
    )
    unknown_key_carrier: bool = False
    unknown_key_sites: set[tuple[int, int, int, int]] = field(default_factory=set)

@dataclass(frozen=True)
class CallArgs:
    callee: str
    pos_map: dict[str, str]
    kw_map: dict[str, str]
    const_pos: dict[str, str]
    const_kw: dict[str, str]
    non_const_pos: set[str]
    non_const_kw: set[str]
    star_pos: list[tuple[int, str]]
    star_kw: list[str]
    is_test: bool
    span: OptionalSpan4 = None
    callable_kind: str = "function"
    callable_source: str = "symbol"

    def __post_init__(self) -> None:
        if set(self.pos_map) & set(self.const_pos):
            never("positional slot cannot be both param and constant")  # pragma: no cover - invariant sink
        if set(self.pos_map) & set(self.non_const_pos):
            never("positional slot cannot be both param and non-const")  # pragma: no cover - invariant sink
        if set(self.const_pos) & set(self.non_const_pos):
            never("positional slot cannot be both const and non-const")  # pragma: no cover - invariant sink
        if set(self.kw_map) & set(self.const_kw):
            never("keyword slot cannot be both param and constant")  # pragma: no cover - invariant sink
        if set(self.kw_map) & set(self.non_const_kw):
            never("keyword slot cannot be both param and non-const")  # pragma: no cover - invariant sink
        if set(self.const_kw) & set(self.non_const_kw):
            never("keyword slot cannot be both const and non-const")  # pragma: no cover - invariant sink

    def callable_id(self) -> CallableId:
        return CallableId.from_raw(self.callee)

    def argument_mapping(self) -> CallArgumentMapping:
        positional = {
            int(idx): ParameterId.from_raw(param)
            for idx, param in self.pos_map.items()
        }
        keywords = {
            key: ParameterId.from_raw(param)
            for key, param in self.kw_map.items()
        }
        return CallArgumentMapping(
            positional=positional,
            keywords=keywords,
            star_positional=tuple(
                (idx, ParameterId.from_raw(param)) for idx, param in self.star_pos
            ),
            star_keywords=tuple(ParameterId.from_raw(param) for param in self.star_kw),
        )

def _invariant_digest(payload: Mapping[str, object], *, prefix: str) -> str:
    encoded = json.dumps(payload, sort_keys=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.blake2s(encoded, digest_size=12).hexdigest()
    return f"{prefix}:{digest}"

def _invariant_confidence(value: OptionalFloat) -> float:
    if value is None:
        return 1.0
    return max(0.0, min(1.0, float(value)))

def _compute_invariant_id(
    *,
    form: str,
    terms: tuple[str, ...],
    scope: str,
    source: str,
) -> str:
    payload = {
        "form": form,
        "terms": list(terms),
        "scope": scope,
        "source": source,
    }
    return _invariant_digest(payload, prefix="inv")

def _compute_invariant_evidence_key(
    *,
    invariant_id: str,
    form: str,
    terms: tuple[str, ...],
    scope: str,
) -> str:
    term_display = ",".join(terms)
    return f"E:invariant::{scope}::{form}::{term_display}::{invariant_id}"

def _normalize_invariant_proposition(
    proposition: InvariantProposition,
    *,
    default_scope: str,
    default_source: str,
) -> InvariantProposition:
    scope = proposition.scope or default_scope
    source = proposition.source or default_source
    invariant_id = proposition.invariant_id or _compute_invariant_id(
        form=proposition.form,
        terms=proposition.terms,
        scope=scope,
        source=source,
    )
    evidence_keys = proposition.evidence_keys or (
        _compute_invariant_evidence_key(
            invariant_id=invariant_id,
            form=proposition.form,
            terms=proposition.terms,
            scope=scope,
        ),
    )
    return InvariantProposition(
        form=proposition.form,
        terms=proposition.terms,
        scope=scope,
        source=source,
        invariant_id=invariant_id,
        confidence=_invariant_confidence(proposition.confidence),
        evidence_keys=tuple(str(key) for key in evidence_keys),
    )

@dataclass
class SymbolTable:
    imports: dict[tuple[str, str], str] = field(default_factory=dict)
    internal_roots: set[str] = field(default_factory=set)
    external_filter: bool = True
    star_imports: dict[str, set[str]] = field(default_factory=dict)
    module_exports: dict[str, set[str]] = field(default_factory=dict)
    module_export_map: dict[str, dict[str, str]] = field(default_factory=dict)

    def resolve(self, current_module: str, name: str) -> OptionalString:
        if (current_module, name) in self.imports:
            fqn = self.imports[(current_module, name)]
            if self.external_filter:
                root = fqn.split(".")[0]
                if root not in self.internal_roots:
                    return None
            return fqn
        return f"{current_module}.{name}"

    def resolve_star(self, current_module: str, name: str) -> OptionalString:
        check_deadline()
        candidates = self.star_imports.get(current_module, set())
        if not candidates:
            return None
        for module in sort_once(
            candidates,
            source="SymbolTable.resolve_star.candidates",
        ):
            check_deadline()
            exports = self.module_exports.get(module)
            if exports is not None and name in exports:
                export_map = self.module_export_map.get(module, {})
                mapped = export_map.get(name)
                if mapped:
                    if self.external_filter:
                        root = mapped.split(".")[0]
                        if root in self.internal_roots:
                            return mapped
                    else:
                        return mapped
                resolved = f"{module}.{name}".strip(".")
                if not module:
                    return resolved
                if self.external_filter:
                    root = module.split(".")[0]
                    if root in self.internal_roots:
                        return resolved
                    continue
                return resolved
        return None

# Canonical owner contract class (WS-5 hard-cut compatibility).
SymbolTable = _ContractSymbolTable

@dataclass
class AuditConfig:
    project_root: OptionalPath = None
    exclude_dirs: set[str] = field(default_factory=set)
    ignore_params: set[str] = field(default_factory=set)
    decision_ignore_params: set[str] = field(default_factory=set)
    external_filter: bool = True
    strictness: str = "high"
    transparent_decorators: OptionalStringSet = None
    decision_tiers: dict[str, int] = field(default_factory=dict)
    decision_require_tiers: bool = False
    never_exceptions: set[str] = field(default_factory=set)
    deadline_roots: set[str] = field(default_factory=set)
    fingerprint_registry: OptionalPrimeRegistry = None
    fingerprint_index: dict[Fingerprint, set[str]] = field(default_factory=dict)
    constructor_registry: OptionalTypeConstructorRegistry = None
    fingerprint_seed_revision: OptionalString = None
    fingerprint_synth_min_occurrences: int = 0
    fingerprint_synth_version: str = "synth@1"
    fingerprint_synth_registry: OptionalSynthRegistry = None
    invariant_emitters: tuple[
        Callable[[ast.FunctionDef], Iterable[InvariantProposition]],
        ...,
    ] = field(default_factory=tuple)
    adapter_contract: OptionalJsonObject = None
    required_analysis_surfaces: set[str] = field(default_factory=set)

    def is_ignored_path(self, path: Path) -> bool:
        parts = set(path.parts)
        return bool(self.exclude_dirs & parts)

_ANALYSIS_PROFILING_FORMAT_VERSION = 1


def _summarize_deadline_obligations(entries, *, max_entries=20, forest):
    return _summarize_deadline_obligations_impl(
        entries,
        max_entries=max_entries,
        forest=forest,
        deps=_SummarizeDeadlineObligationsDeps(
            check_deadline_fn=check_deadline,
            projection_spec_hash_fn=projection_spec_hash,
            deadline_obligations_summary_spec=DEADLINE_OBLIGATIONS_SUMMARY_SPEC,
            require_not_none_fn=require_not_none,
            int_tuple4_or_none_fn=int_tuple4_or_none,
            format_span_fields_fn=_format_span_fields,
        ),
    )


def _profiling_v1_payload(*, stage_ns: Mapping[str, int], counters: Mapping[str, int]) -> JSONObject:
    return {
        "format_version": _ANALYSIS_PROFILING_FORMAT_VERSION,
        "stage_ns": {str(key): int(stage_ns[key]) for key in stage_ns},
        "counters": {str(key): int(counters[key]) for key in counters},
    }

_ReportSectionValue = TypeVar("_ReportSectionValue")

@dataclass(frozen=True)
class ReportProjectionSpec(Generic[_ReportSectionValue]):
    section_id: str
    phase: ReportProjectionPhase
    deps: tuple[str, ...]
    build: Callable[
        [ReportCarrier, dict[Path, dict[str, list[set[str]]]]],
        _ReportSectionValue,
    ]
    render: Callable[[_ReportSectionValue], list[str]]
    violation_extract: Callable[[_ReportSectionValue], list[str]]
    preview_build: object = None

def _report_section_identity_render(lines: list[str]) -> list[str]:
    return lines

def _report_section_no_violations(_lines: list[str]) -> list[str]:
    return []

def _report_section_text(
    report: ReportCarrier,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    *,
    section_id: str,
) -> list[str]:
    rendered, _ = _emit_report(
        groups_by_path,
        max_components=10,
        report=cast(_DataflowReportCarrier, report),
    )
    return extract_report_sections(rendered).get(section_id, [])

def _report_section_spec(
    *,
    section_id: str,
    phase: ReportProjectionPhase,
    deps: tuple[str, ...] = (),
    preview_build = None,
) -> ReportProjectionSpec[list[str]]:
    return ReportProjectionSpec[list[str]](
        section_id=section_id,
        phase=phase,
        deps=deps,
        build=lambda report, groups_by_path, _section_id=section_id: _report_section_text(
            report,
            groups_by_path,
            section_id=_section_id,
        ),
        render=_report_section_identity_render,
        violation_extract=_report_section_no_violations,
        preview_build=preview_build,
    )

@dataclass(frozen=True)
class CallAmbiguity:
    kind: str
    caller: FunctionInfo
    call: "CallArgs | None"
    callee_key: str
    candidates: tuple[FunctionInfo, ...]
    phase: str

def _callee_name(call: ast.Call) -> str:
    try:
        return ast.unparse(call.func)
    except _AST_UNPARSE_ERROR_TYPES:
        return "<call>"

def _normalize_callee(name: str, class_name) -> str:
    if not class_name:
        return name
    if name.startswith("self.") or name.startswith("cls."):
        parts = name.split(".")
        if len(parts) == 2:
            return f"{class_name}.{parts[1]}"
    return name

def _iter_paths(paths: Iterable[str], config: AuditConfig) -> list[Path]:
    return iter_python_paths(
        paths,
        config=config,
        check_deadline=check_deadline,
        sort_once=sort_once,
    )

def _collect_functions(tree: ast.AST) -> list[FunctionNode]:
    check_deadline()
    funcs: list[FunctionNode] = []
    for idx, node in enumerate(ast.walk(tree), start=1):
        if (idx & 63) == 0:
            check_deadline()
        node_type = type(node)
        if node_type is ast.FunctionDef or node_type is ast.AsyncFunctionDef:
            funcs.append(cast(FunctionNode, node))
    return funcs

def _decorator_name(node: ast.AST):
    return _decorator_name_impl(node, check_deadline_fn=check_deadline)

def _decorator_matches(name: str, allowlist: set[str]) -> bool:
    if name in allowlist:
        return True
    if "." in name and name.split(".")[-1] in allowlist:
        return True
    return False

def _is_marker_call(call: ast.Call, aliases: set[str]) -> bool:
    name = _decorator_name(call.func)
    if not name:
        return False
    return _decorator_matches(name, aliases)

def _is_never_marker_raise(
    function: str,
    exception_name,
    never_exceptions: set[str],
) -> bool:
    if not exception_name or not never_exceptions:
        return False
    if not _decorator_matches(exception_name, never_exceptions):
        return False
    return function == "never" or function.endswith(".never")

def _decorators_transparent(
    fn: FunctionNode,
    transparent_decorators,
) -> bool:
    return _decorators_transparent_owner(fn, transparent_decorators)

_collect_local_class_bases = _collect_local_class_bases_owner
_local_class_name = _local_class_name_owner
_resolve_local_method_in_hierarchy = _resolve_local_method_in_hierarchy_owner

def _param_names(
    fn: FunctionNode,
    ignore_params: OptionalIgnoredParams = None,
) -> list[str]:
    args = (
        fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
    )
    names = [a.arg for a in args]
    if fn.args.vararg:
        names.append(fn.args.vararg.arg)
    if fn.args.kwarg:
        names.append(fn.args.kwarg.arg)
    if names and names[0] in {"self", "cls"}:
        names = names[1:]
    if ignore_params:
        names = [name for name in names if name not in ignore_params]
    return names

def _decision_root_name(node: ast.AST):
    check_deadline()
    current = node
    while True:
        check_deadline()
        current_type = type(current)
        if current_type is ast.Attribute:
            current = cast(ast.Attribute, current).value
        elif current_type is ast.Subscript:
            current = cast(ast.Subscript, current).value
        else:
            break
    if type(current) is ast.Name:
        return cast(ast.Name, current).id
    return None

def is_decision_surface(node: ast.AST) -> bool:
    return _is_decision_surface_owner(node)

def _decision_surface_form_entries(
    fn: ast.AST,
) -> list[tuple[str, ast.AST]]:
    return _decision_surface_form_entries_owner(fn)

def _decision_surface_reason_map(
    fn: FunctionNode,
    ignore_params: OptionalIgnoredParams = None,
) -> dict[str, set[str]]:
    return _decision_surface_reason_map_owner(fn, ignore_params)

def _decision_surface_params(
    fn: FunctionNode,
    ignore_params: OptionalIgnoredParams = None,
) -> set[str]:
    return _decision_surface_params_owner(fn, ignore_params)

def _mark_param_roots(expr: ast.AST, params: set[str], out: set[str]) -> None:
    _mark_param_roots_owner(expr, params, out)

def _collect_param_roots(expr: ast.AST, params: set[str]) -> set[str]:
    return _collect_param_roots_owner(expr, params)

def _contains_boolish(expr: ast.AST) -> bool:
    return _contains_boolish_owner(expr)

def _value_encoded_decision_params(
    fn: ast.AST,
    ignore_params = None,
) -> tuple[set[str], set[str]]:
    return _value_encoded_decision_params_owner(fn, ignore_params)

@dataclass(frozen=True)
class _DecisionSurfaceSpec:
    pass_id: str
    alt_kind: str
    surface_label: str
    params: Callable[[FunctionInfo], set[str]]
    descriptor: Callable[[FunctionInfo, str], str]
    alt_evidence: Callable[[str, str], JSONObject]
    surface_lint_code: str
    surface_lint_message: Callable[[str, str, str], str]
    emit_surface_lint: Callable[[int, object], bool]
    tier_lint_code: str
    tier_missing_message: Callable[[str, str], str]
    tier_internal_message: Callable[[str, int, str, str], str]
    rewrite_line: object = None

def _decision_predicate_evidence(
    info: FunctionInfo,
    param: str,
) -> DecisionPredicateEvidence:
    reasons = tuple(
        sort_once(
            info.decision_surface_reasons.get(param, set()),
            source="_decision_predicate_evidence.reasons",
        )
    )
    span = info.param_spans.get(param)
    return DecisionPredicateEvidence(
        parameter=ParameterId.from_raw(param),
        reasons=reasons,
        spans=(SpanIdentity.from_tuple(span),) if span is not None else (),
    )

def _decision_reason_summary(info: FunctionInfo, params: Iterable[str]) -> str:
    labels: set[str] = set()
    for param in params:
        check_deadline()
        evidence = _decision_predicate_evidence(info, param)
        labels.update(evidence.reasons)
    if not labels:
        return "heuristic"
    return ", ".join(
        sort_once(labels, source="_decision_reason_summary.labels")
    )

def _boundary_tier_obligation(caller_count: int) -> str:
    if caller_count > 0:
        return "tier-2:decision-bundle-elevation"
    return "tier-3:decision-table-boundary"

def _decision_surface_alt_evidence(
    *,
    spec: _DecisionSurfaceSpec,
    boundary: str,
    descriptor: str,
    params: Iterable[str],
    caller_count: int,
    reason_summary: str,
) -> JSONObject:
    base_evidence = dict(spec.alt_evidence(boundary, descriptor))
    payload: JSONObject = {
        "boundary": base_evidence.get("boundary", boundary),
        "classification_descriptor": descriptor,
        "classification_reason": reason_summary,
        "decision_params": sort_once(
            set(params),
            source="_decision_surface_alt_evidence.params",
        ),
    }
    if "meta" in base_evidence:
        payload["meta"] = base_evidence["meta"]
    for key in sort_once(
        (str(k) for k in base_evidence if str(k) not in {"boundary", "meta"}),
        source="_decision_surface_alt_evidence.base_evidence",
    ):
        payload[key] = base_evidence[key]
    payload["tier_obligation"] = _boundary_tier_obligation(caller_count)
    payload["tier_pathway"] = "internal" if caller_count > 0 else "boundary"
    return payload

def _suite_site_label(*, forest: Forest, suite_id: NodeId) -> str:
    suite_node = forest.nodes.get(suite_id)
    if suite_node is None:
        never("suite site missing during label projection", suite_id=str(suite_id))  # pragma: no cover - invariant sink
    path = str(suite_node.meta.get("path", "") or "")
    qual = str(suite_node.meta.get("qual", "") or "")
    suite_kind = str(suite_node.meta.get("suite_kind", "") or "")
    span = int_tuple4_or_none(suite_node.meta.get("span"))
    if not path or not qual or not suite_kind or span is None:
        never(  # pragma: no cover - invariant sink
            "suite site label projection missing identity",
            path=path,
            qual=qual,
            suite_kind=suite_kind,
            span=suite_node.meta.get("span"),
        )
    span_text = _format_span_fields(*span)
    return f"{path}:{qual}[{suite_kind}]@{span_text}" if span_text else f"{path}:{qual}[{suite_kind}]"

_DIRECT_DECISION_SURFACE_SPEC = _DecisionSurfaceSpec(
    pass_id="decision_surfaces",
    alt_kind="DecisionSurface",
    surface_label="decision surface params",
    params=lambda info: info.decision_params,
    descriptor=lambda info, boundary: (
        f"{boundary}; reason={_decision_reason_summary(info, info.decision_params)}"
    ),
    alt_evidence=lambda boundary, _descriptor: {
        "meta": boundary,
        "boundary": boundary,
    },
    surface_lint_code="GABION_DECISION_SURFACE",
    surface_lint_message=lambda param, boundary, _descriptor: (
        f"decision surface param '{param}' ({boundary})"
    ),
    emit_surface_lint=lambda caller_count, tier: caller_count == 0 and tier is None,
    tier_lint_code="GABION_DECISION_TIER",
    tier_missing_message=lambda param, _descriptor: (
        f"decision param '{param}' missing decision tier metadata"
    ),
    tier_internal_message=lambda param, tier, boundary, _descriptor: (
        f"tier-{tier} decision param '{param}' used below boundary ({boundary})"
    ),
)

_VALUE_DECISION_SURFACE_SPEC = _DecisionSurfaceSpec(
    pass_id="value_encoded_decisions",
    alt_kind="ValueDecisionSurface",
    surface_label="value-encoded decision params",
    params=lambda info: info.value_decision_params,
    descriptor=lambda info, _boundary: ", ".join(
        sort_once(
            info.value_decision_reasons,
            source="_VALUE_DECISION_SURFACE_SPEC.descriptor",
        )
    )
    or "heuristic",
    alt_evidence=lambda boundary, descriptor: {
        "meta": descriptor,
        "boundary": boundary,
        "reasons": descriptor,
    },
    surface_lint_code="GABION_VALUE_DECISION_SURFACE",
    surface_lint_message=lambda param, boundary, descriptor: (
        f"value-encoded decision param '{param}' ({boundary}; {descriptor})"
    ),
    emit_surface_lint=lambda _caller_count, tier: tier is None,
    tier_lint_code="GABION_VALUE_DECISION_TIER",
    tier_missing_message=lambda param, descriptor: (
        f"value-encoded decision param '{param}' missing decision tier metadata ({descriptor})"
    ),
    tier_internal_message=lambda param, tier, boundary, descriptor: (
        f"tier-{tier} value-encoded decision param '{param}' used below boundary ({boundary}; {descriptor})"
    ),
    rewrite_line=lambda info, params, descriptor: (
        f"{info.path.name}:{info.qual} consider rebranching value-encoded decision params: "
        + ", ".join(params)
        + f" ({descriptor})"
    ),
)

def _analyze_decision_surface_indexed(
    context: _IndexedPassContext,
    *,
    spec: _DecisionSurfaceSpec,
    decision_tiers,
    require_tiers: bool,
    forest: Forest,
) -> tuple[list[str], list[str], list[str], list[str]]:
    return _analyze_decision_surface_indexed_impl(
        context,
        spec=spec,
        decision_tiers=decision_tiers,
        require_tiers=require_tiers,
        forest=forest,
        deps=_DecisionSurfaceAnalyzeDeps(
            build_call_graph_fn=_build_call_graph,
            check_deadline_fn=check_deadline,
            is_test_path_fn=_is_test_path,
            sort_once_fn=sort_once,
            decision_reason_summary_fn=_decision_reason_summary,
            decision_surface_alt_evidence_fn=_decision_surface_alt_evidence,
            suite_site_label_fn=_suite_site_label,
            decision_tier_for_fn=_decision_tier_for,
            decision_param_lint_line_fn=_decision_param_lint_line,
        ),
    )

def _analyze_decision_surfaces_indexed(
    context: _IndexedPassContext,
    *,
    decision_tiers,
    require_tiers: bool,
    forest: Forest,
    run_fn: Callable[..., tuple[list[str], list[str], list[str], list[str]]] = _analyze_decision_surface_indexed,
) -> tuple[list[str], list[str], list[str]]:
    surfaces, warnings, rewrites, lint_lines = run_fn(
        context,
        spec=_DIRECT_DECISION_SURFACE_SPEC,
        decision_tiers=decision_tiers,
        require_tiers=require_tiers,
        forest=forest,
    )
    if rewrites:
        never(
            "decision_surfaces rewrites must be empty",
            pass_id=_DIRECT_DECISION_SURFACE_SPEC.pass_id,
        )
    return surfaces, warnings, lint_lines

def _analyze_value_encoded_decisions_indexed(
    context: _IndexedPassContext,
    *,
    decision_tiers,
    require_tiers: bool,
    forest: Forest,
) -> tuple[list[str], list[str], list[str], list[str]]:
    return _analyze_decision_surface_indexed(
        context,
        spec=_VALUE_DECISION_SURFACE_SPEC,
        decision_tiers=decision_tiers,
        require_tiers=require_tiers,
        forest=forest,
    )

def _node_span(node: ast.AST):
    if not hasattr(node, "lineno") or not hasattr(node, "col_offset"):
        return None
    start_line = max(getattr(node, "lineno", 1) - 1, 0)
    start_col = max(getattr(node, "col_offset", 0), 0)
    end_line = max(getattr(node, "end_lineno", getattr(node, "lineno", 1)) - 1, 0)
    end_col = getattr(node, "end_col_offset", start_col + 1)
    if end_line == start_line and end_col <= start_col:
        end_col = start_col + 1
    return (start_line, start_col, end_line, end_col)

def _param_spans(
    fn: FunctionNode,
    ignore_params: OptionalIgnoredParams = None,
) -> dict[str, tuple[int, int, int, int]]:
    check_deadline()
    spans: dict[str, tuple[int, int, int, int]] = {}
    args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
    names = [a.arg for a in args]
    if names and names[0] in {"self", "cls"}:
        args = args[1:]
        names = names[1:]
    for arg in args:
        check_deadline()
        if ignore_params and arg.arg in ignore_params:
            continue
        span = _node_span(arg)
        if span is not None:
            spans[arg.arg] = span
    if fn.args.vararg:
        name = fn.args.vararg.arg
        if not ignore_params or name not in ignore_params:
            span = _node_span(fn.args.vararg)
            if span is not None:
                spans[name] = span
    if fn.args.kwarg:
        name = fn.args.kwarg.arg
        if not ignore_params or name not in ignore_params:
            span = _node_span(fn.args.kwarg)
            if span is not None:
                spans[name] = span
    return spans

def _function_key(scope: Iterable[str], name: str) -> str:
    parts = list(scope)
    parts.append(name)
    return ".".join(parts)

def _enclosing_class(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
):
    check_deadline()
    current = parents.get(node)
    while current is not None:
        check_deadline()
        if type(current) is ast.ClassDef:
            return cast(ast.ClassDef, current).name
        current = parents.get(current)
    return None

def _enclosing_scopes(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> list[str]:
    check_deadline()
    scopes: list[str] = []
    current = parents.get(node)
    while current is not None:
        check_deadline()
        current_type = type(current)
        if current_type is ast.ClassDef:
            scopes.append(cast(ast.ClassDef, current).name)
        elif current_type is ast.FunctionDef or current_type is ast.AsyncFunctionDef:
            scopes.append(cast(FunctionNode, current).name)
        current = parents.get(current)
    return list(reversed(scopes))

def _enclosing_class_scopes(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> list[str]:
    check_deadline()
    scopes: list[str] = []
    current = parents.get(node)
    while current is not None:
        check_deadline()
        if type(current) is ast.ClassDef:
            scopes.append(cast(ast.ClassDef, current).name)
        current = parents.get(current)
    return list(reversed(scopes))

def _enclosing_function_scopes(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> list[str]:
    check_deadline()
    scopes: list[str] = []
    current = parents.get(node)
    while current is not None:
        check_deadline()
        current_type = type(current)
        if current_type is ast.FunctionDef or current_type is ast.AsyncFunctionDef:
            scopes.append(cast(FunctionNode, current).name)
        current = parents.get(current)
    return list(reversed(scopes))

def _param_annotations(
    fn: FunctionNode,
    ignore_params: OptionalIgnoredParams = None,
) -> ParamAnnotationMap:
    check_deadline()
    args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
    names = [a.arg for a in args]
    annots: ParamAnnotationMap = {}

    for name, arg in zip(names, args):
        check_deadline()
        annotation_value = _extract_annotation_value(arg.annotation)
        annots[name] = annotation_value.text if annotation_value.parse_status == "present" else None
    if fn.args.vararg:
        vararg = fn.args.vararg
        annotation_value = _extract_annotation_value(vararg.annotation)
        annots[vararg.arg] = annotation_value.text if annotation_value.parse_status == "present" else None
    if fn.args.kwarg:
        kwarg = fn.args.kwarg
        annotation_value = _extract_annotation_value(kwarg.annotation)
        annots[kwarg.arg] = annotation_value.text if annotation_value.parse_status == "present" else None
    if names and names[0] in {"self", "cls"}:
        annots.pop(names[0], None)
    if ignore_params:
        for name in list(annots.keys()):
            check_deadline()
            if name in ignore_params:
                annots.pop(name, None)
    return annots

def _extract_annotation_value(annotation: OptionalAstNode) -> AnnotationValue:
    check_deadline()
    if annotation is None:
        return AnnotationValue(text="", parse_status="missing")
    try:
        return AnnotationValue(
            text=ast.unparse(annotation),
            parse_status="present",
        )
    except _AST_UNPARSE_ERROR_TYPES:
        return AnnotationValue(text="", parse_status="unparse_failure")

def _param_defaults(
    fn: FunctionNode,
    ignore_params: OptionalIgnoredParams = None,
) -> set[str]:
    check_deadline()
    defaults: set[str] = set()
    args = fn.args.posonlyargs + fn.args.args
    names = [a.arg for a in args]
    if fn.args.defaults:
        defaulted = names[-len(fn.args.defaults) :]
        defaults.update(defaulted)
    for kw_arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        check_deadline()
        if default is not None:
            defaults.add(kw_arg.arg)
    if names and names[0] in {"self", "cls"}:
        defaults.discard(names[0])
    if ignore_params:
        defaults = {name for name in defaults if name not in ignore_params}
    return defaults

_ANALYSIS_INDEX_STAGE_CACHE_OP = _ANALYSIS_INDEX_STAGE_CACHE_OP_owner

_path_dependency_payload = _path_dependency_payload_owner

def _parse_module_tree(
    path: Path,
    *,
    stage: _ParseModuleStage,
    parse_failure_witnesses: list[JSONObject],
):
    try:
        return ast.parse(path.read_text())
    except _PARSE_MODULE_ERROR_TYPES as exc:
        _record_parse_failure_witness(
            sink=parse_failure_witnesses,
            path=path,
            stage=stage,
            error=exc,
        )
        return None

def _is_deadline_annot(annot) -> bool:
    if not annot:
        return False
    return bool(re.search(r"\bDeadline\b", annot))

def _is_deadline_param(name: str, annot) -> bool:
    if _is_deadline_annot(annot):
        return True
    if annot is None and name.lower() == "deadline":
        return True
    return False

def _is_deadline_origin_call(expr: ast.AST) -> bool:
    return _is_deadline_origin_call_impl(expr)

def _target_names(target: ast.AST) -> set[str]:
    check_deadline()
    names: set[str] = set()
    for node in ast.walk(target):
        check_deadline()
        if type(node) is ast.Name:
            name_node = cast(ast.Name, node)
            if type(name_node.ctx) is ast.Store:
                names.add(name_node.id)
    return names

def _simple_store_name(target: ast.AST) -> OptionalString:
    if type(target) is ast.Name:
        return cast(ast.Name, target).id
    return None

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

def _format_span_fields(
    line: object,
    col: object,
    end_line: object,
    end_col: object,
) -> str:
    from gabion.analysis.dataflow.io.dataflow_reporting_helpers import _format_span_fields as _impl

    return _impl(line, col, end_line, end_col)

_lint_line = _lint_line_owner

def _add_interned_alt(
    *,
    forest: Forest,
    kind: str,
    inputs: Iterable[NodeId],
    evidence = None,
) -> Alt:
    return forest.add_alt(kind, inputs, evidence=evidence)

_decision_param_lint_line = _decision_param_lint_line_owner

_decision_tier_for = _decision_tier_for_owner

@dataclass
class AnalysisIndex:
    by_name: dict[str, list[FunctionInfo]]
    by_qual: dict[str, FunctionInfo]
    symbol_table: SymbolTable
    class_index: dict[str, ClassInfo]
    parsed_modules_by_path: dict[Path, ast.Module] = field(default_factory=dict)
    module_parse_errors_by_path: dict[Path, Exception] = field(default_factory=dict)
    stage_cache_by_key: dict[Hashable, dict[Path, object]] = field(default_factory=dict)
    index_cache_identity: str = ""
    projection_cache_identity: str = ""
    transitive_callers: "dict[str, set[str]] | None" = None
    resolved_call_edges: 'tuple["_ResolvedCallEdge", ...] | None' = None
    resolved_transparent_call_edges: 'tuple["_ResolvedCallEdge", ...] | None' = None
    resolved_transparent_edges_by_caller: 'dict[str, tuple["_ResolvedCallEdge", ...]] | None' = None

@dataclass(frozen=True)
class _ResolvedCallEdge:
    caller: FunctionInfo
    call: CallArgs
    callee: FunctionInfo

_StageCacheValue = TypeVar("_StageCacheValue")

_ResolvedEdgeAcc = TypeVar("_ResolvedEdgeAcc")

_ResolvedEdgeOut = TypeVar("_ResolvedEdgeOut")

@dataclass(frozen=True)
class _ResolvedEdgeReducerSpec(Generic[_ResolvedEdgeAcc, _ResolvedEdgeOut]):
    reducer_id: str
    init: Callable[[], _ResolvedEdgeAcc]
    fold: Callable[[_ResolvedEdgeAcc, _ResolvedCallEdge], None]
    finish: Callable[[_ResolvedEdgeAcc], _ResolvedEdgeOut]

@dataclass(frozen=True)
class _ResolvedEdgeParamEvent:
    kind: str
    param: str
    value: OptionalString
    countable: bool

@dataclass(frozen=True)
class _StageCacheSpec(Generic[_StageCacheValue]):
    stage: _ParseModuleStage
    cache_key: Hashable
    build: Callable[[ast.Module, Path], _StageCacheValue]

def _parse_module_source(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _forbid_adhoc_bundle_discovery(reason: str) -> None:
    if os.environ.get("GABION_FORBID_ADHOC_BUNDLES") == "1":
        raise AssertionError(
            f"Ad-hoc bundle discovery invoked while forest-only invariant active: {reason}"
        )

_materialize_statement_suite_contains = _materialize_statement_suite_contains_owner
_materialize_structured_suite_sites_for_tree = _materialize_structured_suite_sites_for_tree_owner
_materialize_structured_suite_sites = _materialize_structured_suite_sites_owner
_populate_bundle_forest = _populate_bundle_forest_owner

def _is_test_path(path: Path) -> bool:
    if "tests" in path.parts:
        return True
    return path.name.startswith("test_")

_unused_params = _unused_params_owner

_group_by_signature = _group_by_signature_owner
_union_groups = _union_groups_owner
_propagate_groups = _propagate_groups_owner
_adapt_ingest_carrier_to_analysis_maps = _adapt_ingest_carrier_to_analysis_maps_owner
analyze_ingested_file = _analyze_ingested_file_owner

_analyze_file_internal = _analyze_file_internal_owner

def analyze_file(
    path: Path,
    recursive: bool = True,
    *,
    config = None,
) -> tuple[dict[str, list[set[str]]], dict[str, dict[str, tuple[int, int, int, int]]]]:
    groups, spans, _ = _analyze_file_internal(path, recursive=recursive, config=config)
    return groups, spans

def _callee_key(name: str) -> str:
    if not name:
        return name
    return name.split(".")[-1]

@dataclass
class FunctionInfo:
    name: str
    qual: str
    path: Path
    params: list[str]
    annots: ParamAnnotationMap
    calls: list[CallArgs]
    unused_params: set[str]
    unknown_key_carriers: set[str] = field(default_factory=set)
    defaults: set[str] = field(default_factory=set)
    transparent: bool = True
    class_name: OptionalString = None
    scope: tuple[str, ...] = ()
    lexical_scope: tuple[str, ...] = ()
    decision_params: set[str] = field(default_factory=set)
    decision_surface_reasons: dict[str, set[str]] = field(default_factory=dict)
    value_decision_params: set[str] = field(default_factory=set)
    value_decision_reasons: set[str] = field(default_factory=set)
    positional_params: tuple[str, ...] = ()
    kwonly_params: tuple[str, ...] = ()
    vararg: OptionalString = None
    kwarg: OptionalString = None
    param_spans: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    function_span: OptionalSpan4 = None
    local_lambda_bindings: dict[str, tuple[str, ...]] = field(default_factory=dict)

@dataclass
class ClassInfo:
    qual: str
    module: str
    bases: list[str]
    methods: set[str]

def _module_name(path: Path, project_root = None) -> str:
    rel = path.with_suffix("")
    if project_root is not None:
        try:
            rel = rel.relative_to(project_root)
        except ValueError:
            pass
    parts = list(rel.parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    return ".".join(parts)

def _string_list(node: ast.AST):
    check_deadline()
    node_type = type(node)
    if node_type is ast.List or node_type is ast.Tuple:
        container = cast(ast.List | ast.Tuple, node)
        values: list[str] = []
        for elt in container.elts:
            check_deadline()
            if type(elt) is ast.Constant and type(cast(ast.Constant, elt).value) is str:
                values.append(cast(str, cast(ast.Constant, elt).value))
            else:
                return None
        return values
    return None

def _base_identifier(node: ast.AST):
    check_deadline()
    node_type = type(node)
    if node_type is ast.Name:
        return cast(ast.Name, node).id
    if node_type is ast.Attribute:
        try:
            return ast.unparse(node)
        except _AST_UNPARSE_ERROR_TYPES:
            return None
    if node_type is ast.Subscript:
        return _base_identifier(cast(ast.Subscript, node).value)
    if node_type is ast.Call:
        return _base_identifier(cast(ast.Call, node).func)
    return None

def _collect_module_exports(
    tree: ast.AST,
    *,
    module_name: str,
    import_map: dict[str, str],
) -> tuple[set[str], dict[str, str]]:
    return _collect_module_exports_impl(
        tree,
        module_name=module_name,
        import_map=import_map,
        deps=_ModuleExportsCollectDeps(
            check_deadline_fn=check_deadline,
            string_list_fn=_string_list,
            target_names_fn=_target_names,
        ),
    )

_accumulate_symbol_table_for_tree = _accumulate_symbol_table_for_tree_owner

def _symbol_table_module_artifact_spec(
    *,
    project_root,
    external_filter: bool,
) -> _ModuleArtifactSpec[SymbolTable, SymbolTable]:
    return _ModuleArtifactSpec[SymbolTable, SymbolTable](
        artifact_id="symbol_table",
        stage=_ParseModuleStage.SYMBOL_TABLE,
        init=lambda: SymbolTable(external_filter=external_filter),
        fold=lambda table, path, tree: _accumulate_symbol_table_for_tree(
            table,
            path,
            tree,
            project_root=project_root,
        ),
        finish=lambda table: table,
    )

def _build_symbol_table(
    paths: list[Path],
    project_root,
    *,
    external_filter: bool,
    parse_failure_witnesses: list[JSONObject],
) -> SymbolTable:
    check_deadline()
    raw_table, = _build_module_artifacts(
        paths,
        specs=(
            cast(
                _ModuleArtifactSpec[object, object],
                _symbol_table_module_artifact_spec(
                    project_root=project_root,
                    external_filter=external_filter,
                ),
            ),
        ),
        parse_failure_witnesses=parse_failure_witnesses,
    )
    return cast(SymbolTable, raw_table)

_accumulate_class_index_for_tree = _accumulate_class_index_for_tree_owner

@dataclass
class _FunctionIndexAccumulator:
    by_name: dict[str, list[FunctionInfo]] = field(
        default_factory=lambda: defaultdict(list)
    )
    by_qual: dict[str, FunctionInfo] = field(default_factory=dict)

_accumulate_function_index_for_tree = _accumulate_function_index_for_tree_owner

_synthetic_lambda_name = _synthetic_lambda_name_owner
_collect_lambda_function_infos = _collect_lambda_function_infos_owner
_collect_lambda_bindings_by_caller = _collect_lambda_bindings_by_caller_owner
_collect_closure_lambda_factories = _collect_closure_lambda_factories_owner
_direct_lambda_callee_by_call_span = _direct_lambda_callee_by_call_span_owner
_materialize_direct_lambda_callees = _materialize_direct_lambda_callees_owner

def _function_index_module_artifact_spec(
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    transparent_decorators,
) -> _ModuleArtifactSpec[
    _FunctionIndexAccumulator,
    tuple[dict[str, list[FunctionInfo]], dict[str, FunctionInfo]],
]:
    return _ModuleArtifactSpec[
        _FunctionIndexAccumulator,
        tuple[dict[str, list[FunctionInfo]], dict[str, FunctionInfo]],
    ](
        artifact_id="function_index",
        stage=_ParseModuleStage.FUNCTION_INDEX,
        init=_FunctionIndexAccumulator,
        fold=lambda acc, path, tree: _accumulate_function_index_for_tree(
            acc,
            path,
            tree,
            project_root=project_root,
            ignore_params=ignore_params,
            strictness=strictness,
            transparent_decorators=transparent_decorators,
        ),
        finish=lambda acc: (acc.by_name, acc.by_qual),
    )

def _build_function_index(
    paths: list[Path],
    project_root,
    ignore_params: set[str],
    strictness: str,
    transparent_decorators = None,
    *,
    parse_failure_witnesses: list[JSONObject],
) -> tuple[dict[str, list[FunctionInfo]], dict[str, FunctionInfo]]:
    check_deadline()
    raw_index, = _build_module_artifacts(
        paths,
        specs=(
            cast(
                _ModuleArtifactSpec[object, object],
                _function_index_module_artifact_spec(
                    project_root=project_root,
                    ignore_params=ignore_params,
                    strictness=strictness,
                    transparent_decorators=transparent_decorators,
                ),
            ),
        ),
        parse_failure_witnesses=parse_failure_witnesses,
    )
    return cast(
        tuple[dict[str, list[FunctionInfo]], dict[str, FunctionInfo]],
        raw_index,
    )

_resolve_callee = _resolve_callee_owner

_is_dynamic_dispatch_callee_key = _is_dynamic_dispatch_callee_key_owner

_CalleeResolutionOutcome = _CalleeResolutionOutcome_owner

_dedupe_resolution_candidates = _dedupe_resolution_candidates_owner

def _resolve_callee_outcome(
    callee_key: str,
    caller: FunctionInfo,
    by_name: dict[str, list[FunctionInfo]],
    by_qual: dict[str, FunctionInfo],
    *,
    symbol_table = None,
    project_root = None,
    class_index = None,
    call = None,
    local_lambda_bindings = None,
    resolve_callee_fn = _resolve_callee,
) -> _CalleeResolutionOutcome:
    return _resolve_callee_outcome_owner(
        callee_key,
        caller,
        by_name,
        by_qual,
        symbol_table=symbol_table,
        project_root=project_root,
        class_index=class_index,
        call=call,
        local_lambda_bindings=local_lambda_bindings,
        resolve_callee_fn=resolve_callee_fn,
    )


def analyze_type_flow_repo(
    paths: list[Path],
    *,
    project_root: OptionalProjectRoot,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators: OptionalDecorators = None,
    parse_failure_witnesses: OptionalParseFailures = None,
    analysis_index: OptionalAnalysisIndex = None,
) -> tuple[list[str], list[str]]:
    inferred, suggestions, ambiguities = analyze_type_flow_repo_with_map(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
    )
    return suggestions, ambiguities


@dataclass(frozen=True)
class ConstantFlowDetail:
    path: Path
    qual: str
    name: str
    param: str
    value: str
    count: int
    sites: tuple[str, ...] = ()

_constant_smells_from_details = _constant_smells_from_details_owner

_deadness_witnesses_from_constant_details = (
    _deadness_witnesses_from_constant_details_owner
)


_iter_documented_bundles = _iter_documented_bundles_owner

def _dataclass_registry_for_tree(
    path: Path,
    tree: ast.AST,
    *,
    project_root = None,
) -> dict[str, list[str]]:
    return cast(
        dict[str, list[str]],
        _dataclass_registry_for_tree_impl(
            path,
            tree,
            project_root=project_root,
            deps=_DataclassRegistryForTreeDeps(
                check_deadline_fn=check_deadline,
                module_name_fn=_module_name,
                simple_store_name_fn=_simple_store_name,
                decorator_text_fn=lambda dec: (
                    ast.unparse(dec) if hasattr(ast, "unparse") else ""
                ),
            ),
        ),
    )

def _parse_report_section_marker(line: str):
    return _parse_report_section_marker_impl(line)

def extract_report_sections(markdown: str) -> dict[str, list[str]]:
    return _extract_report_sections_impl(markdown, check_deadline_fn=check_deadline)

def _normalize_snapshot_path(path: Path, root) -> str:
    return _normalize_snapshot_path_impl(path, root)

_FILE_SCAN_PROGRESS_EMIT_INTERVAL = 1

_PROGRESS_EMIT_MIN_INTERVAL_SECONDS = 1.0

def _iter_monotonic_paths(
    paths: Iterable[Path],
    *,
    source: str,
) -> list[Path]:
    return _iter_monotonic_paths_impl(
        paths,
        source=source,
        analysis_collection_resume_path_key_fn=_analysis_collection_resume_path_key,
        check_deadline_fn=check_deadline,
        never_fn=never,
    )

def _load_analysis_index_resume_payload(
    *,
    payload,
    file_paths: Sequence[Path],
    expected_index_cache_identity: str = "",
    expected_projection_cache_identity: str = "",
) -> tuple[set[Path], dict[str, FunctionInfo], SymbolTable, dict[str, ClassInfo]]:
    hydrated_paths, by_qual_raw, symbol_table_raw, class_index_raw = (
        _load_analysis_index_resume_payload_owner(
            payload=payload,
            file_paths=file_paths,
            expected_index_cache_identity=expected_index_cache_identity,
            expected_projection_cache_identity=expected_projection_cache_identity,
        )
    )
    return (
        hydrated_paths,
        cast(dict[str, FunctionInfo], by_qual_raw),
        cast(SymbolTable, symbol_table_raw),
        cast(dict[str, ClassInfo], class_index_raw),
    )

def _compute_violations(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    max_components: int,
    *,
    report: ReportCarrier,
) -> list[str]:
    _, violations = _emit_report(
        groups_by_path,
        max_components,
        report=cast(_DataflowReportCarrier, report),
    )
    return sort_once(
        set(violations),
        source="_compute_violations.violations",
    )

def _resolve_baseline_path(path, root: Path):
    return _resolve_baseline_path_impl(path, root)

def _resolve_synth_registry_path(path, root: Path):
    return _resolve_synth_registry_path_impl(path, root)

def _build_parser() -> argparse.ArgumentParser:
    return _build_parser_impl()

def _normalize_transparent_decorators(
    value: object,
) -> object:
    return _normalize_transparent_decorators_impl(value, check_deadline_fn=check_deadline)

@contextmanager
def _analysis_deadline_scope(args: argparse.Namespace):
    with _analysis_deadline_scope_impl(args):
        yield
