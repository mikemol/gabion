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

import time

from collections import Counter, defaultdict

from contextlib import ExitStack, contextmanager

from dataclasses import dataclass, field, replace

from enum import StrEnum

from pathlib import Path

from typing import Callable, Generic, Hashable, Iterable, Iterator, Literal, Mapping, Sequence, TypeVar, cast

import re

from gabion.analysis.projection.pattern_schema import (
    PatternAxis, PatternInstance, PatternResidue, PatternSchema, execution_signature, mismatch_residue_payload)

from gabion.ingest.python_ingest import ingest_python_file, iter_python_paths

from gabion.analysis.core.visitors import ImportVisitor, ParentAnnotator, UseVisitor

from gabion.analysis.semantics.evidence import (
    Site, exception_obligation_summary_for_site, normalize_bundle_key)

from gabion.analysis.foundation.json_types import JSONObject, JSONValue

from gabion.analysis.semantics.schema_audit import find_anonymous_schema_surfaces

from gabion.analysis.aspf.aspf import Alt, Forest, Node, NodeId, structural_key_atom, structural_key_json

from gabion.analysis.derivation.derivation_cache import get_global_derivation_cache

from gabion.analysis.derivation.derivation_contract import DerivationOp

from gabion.analysis.semantics import evidence_keys

from gabion.exceptions import NeverThrown

from gabion.invariants import never, require_not_none

from gabion.order_contract import OrderPolicy, sort_once

from gabion.config import (
    dataflow_defaults, dataflow_adapter_payload, dataflow_deadline_roots, dataflow_required_surfaces, decision_defaults, decision_ignore_list, decision_require_tiers, decision_tier_map, exception_defaults, exception_marker_family, exception_never_list, fingerprint_defaults, merge_payload, synthesis_defaults)

from gabion.analysis.foundation.marker_protocol import (
    DEFAULT_MARKER_ALIASES)

from gabion.analysis.core.type_fingerprints import (
    Fingerprint, FingerprintDimension, PrimeRegistry, TypeConstructorRegistry, _collect_base_atoms, _collect_constructors, SynthRegistry, build_synth_registry, build_fingerprint_registry, build_synth_registry_from_payload, bundle_fingerprint_dimensional, format_fingerprint, fingerprint_carrier_soundness, fingerprint_identity_payload, fingerprint_stage_cache_identity, synth_registry_payload)

from gabion.analysis.core.forest_signature import (
    build_forest_signature, build_forest_signature_from_groups)

from gabion.analysis.core.forest_spec import (
    ForestSpec, build_forest_spec, default_forest_spec, forest_spec_metadata)

from gabion.analysis.foundation.timeout_context import (
    Deadline, GasMeter, TimeoutExceeded, TimeoutTickCarrier, build_timeout_context_from_stack, check_deadline, deadline_loop_iter, deadline_clock_scope, deadline_scope, forest_scope, reset_forest, set_forest)

from gabion.analysis.projection.projection_exec import apply_spec

from gabion.analysis.projection.projection_normalize import spec_hash as projection_spec_hash

from gabion.analysis.foundation.baseline_io import load_json

from gabion.analysis.projection.decision_flow import (
    build_decision_tables, detect_repeated_guard_bundles, enforce_decision_protocol_contracts)

from gabion.analysis.foundation.resume_codec import (
    allowed_path_lookup, int_str_pairs_from_sequence, int_tuple4_or_none, iter_valid_key_entries, load_resume_map, load_allowed_paths_from_sequence, mapping_payload, mapping_sections, mapping_or_empty, mapping_or_none, payload_with_format, payload_with_phase, sequence_or_none, str_list_from_sequence, str_map_from_mapping, str_pair_set_from_sequence, str_set_from_sequence, str_tuple_from_sequence)

from gabion.analysis.indexed_scan.index.analysis_carriers import AnalysisResult, ReportCarrier

from gabion.analysis.projection.projection_registry import (
    AMBIGUITY_SUMMARY_SPEC, AMBIGUITY_SUITE_AGG_SPEC, AMBIGUITY_VIRTUAL_SET_SPEC, DEADLINE_OBLIGATIONS_SUMMARY_SPEC, LINT_FINDINGS_SPEC, NEVER_INVARIANTS_SPEC, REPORT_SECTION_LINES_SPEC, SUITE_ORDER_SPEC, WL_REFINEMENT_SPEC, spec_metadata_lines_from_payload, spec_metadata_payload)

from gabion.analysis.core.wl_refinement import emit_wl_refinement_facets

from gabion.analysis.aspf.aspf_core import parse_2cell_witness

from gabion.analysis.core.deprecated_substrate import (
    DeprecatedExtractionArtifacts, DeprecatedFiber, detect_report_section_extinction)

from gabion.analysis.core.structure_reuse_classes import build_structure_class, structure_class_payload

from gabion.analysis.aspf.aspf_decision_surface import classify_drift_by_homotopy

from gabion.analysis.dataflow.engine.dataflow_decision_surfaces import (
    compute_fingerprint_coherence as _ds_compute_fingerprint_coherence, compute_fingerprint_rewrite_plans as _ds_compute_fingerprint_rewrite_plans, extract_smell_sample as _ds_extract_smell_sample, lint_lines_from_bundle_evidence as _ds_lint_lines_from_bundle_evidence, lint_lines_from_constant_smells as _ds_lint_lines_from_constant_smells, lint_lines_from_type_evidence as _ds_lint_lines_from_type_evidence, lint_lines_from_unused_arg_smells as _ds_lint_lines_from_unused_arg_smells, parse_lint_location as _ds_parse_lint_location, summarize_coherence_witnesses as _ds_summarize_coherence_witnesses, summarize_deadness_witnesses as _ds_summarize_deadness_witnesses, summarize_rewrite_plans as _ds_summarize_rewrite_plans)
from gabion.analysis.dataflow.engine.dataflow_bundle_merge import (
    _merge_counts_by_knobs,
)
from gabion.analysis.dataflow.engine.dataflow_callee_resolution import (
    CalleeResolutionContext as _CalleeResolutionContextCore,
    collect_callee_resolution_effects as _collect_callee_resolution_effects_impl,
    resolve_callee_with_effects as _resolve_callee_with_effects_impl,
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
from gabion.analysis.dataflow.engine.dataflow_call_graph_algorithms import (
    _collect_recursive_functions,
    _collect_recursive_nodes,
    _reachable_from_roots,
    _sorted_graph_nodes,
)
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
)
from gabion.analysis.dataflow.engine.dataflow_resume_paths import (
    iter_monotonic_paths as _iter_monotonic_paths_impl,
    normalize_snapshot_path as _normalize_snapshot_path_impl,
)
from gabion.analysis.dataflow.engine.dataflow_resume_serialization import (
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
    _build_property_hook_callable_index,
    _callsite_evidence_for_bundle,
    _collect_config_bundles,
    _collect_constant_flow_details,
    _collect_dataclass_registry,
    _combine_type_hints,
    _compute_knob_param_names,
    _expand_type_hint,
    _format_call_site,
    _format_type_flow_site,
    _iter_config_fields,
    _iter_dataclass_call_bundles,
    _split_top_level,
    _type_from_const_repr,
    analyze_constant_flow_repo,
    analyze_deadness_flow_repo,
    analyze_type_flow_repo_with_evidence,
    analyze_type_flow_repo_with_map,
    analyze_unused_arg_flow_repo,
    generate_property_hook_manifest,
)
from gabion.analysis.dataflow.io.dataflow_projection_helpers import (
    _topologically_order_report_projection_specs,
)

from gabion.analysis.dataflow.engine.dataflow_exception_obligations import (
    exception_handler_compatibility as _exc_exception_handler_compatibility, exception_param_names as _exc_exception_param_names, handler_type_names as _exc_handler_type_names, exception_type_name as _exc_exception_type_name, handler_is_broad as _exc_handler_is_broad, handler_label as _exc_handler_label, node_in_try_body as _exc_node_in_try_body, _builtin_exception_class as _exc_builtin_exception_class)

from gabion.analysis.semantics.semantic_primitives import (
    AnalysisPassPrerequisites, CallArgumentMapping, CallableId, DecisionPredicateEvidence, ParameterId, SpanIdentity)
from gabion.analysis.dataflow.engine.dataflow_contracts import InvariantProposition, ReportCarrier as _DataflowReportCarrier, SymbolTable as _ContractSymbolTable

from gabion.analysis.dataflow.io.dataflow_report_rendering import (
    render_unsupported_by_adapter_section as _report_render_unsupported_section, render_synthesis_section as _report_render_synthesis_section)
from gabion.analysis.dataflow.io.dataflow_reporting import (
    emit_report as _emit_report,
    render_report,
)
from gabion.analysis.dataflow.io.dataflow_reporting_helpers import (
    render_mermaid_component as _render_mermaid_component,
)

from gabion.analysis.dataflow.io.dataflow_snapshot_contracts import (
    DecisionSnapshotSurfaces, StructureSnapshotDiffRequest)

from gabion.analysis.projection.pattern_schema_projection import (
    bundle_pattern_instances as _bundle_pattern_instances_impl, detect_execution_pattern_matches as _detect_execution_pattern_matches_impl, execution_pattern_instances as _execution_pattern_instances_impl, execution_pattern_suggestions as _execution_pattern_suggestions_impl, pattern_schema_matches as _pattern_schema_matches_impl, pattern_schema_residue_entries as _pattern_schema_residue_entries_impl, pattern_schema_residue_lines as _pattern_schema_residue_lines_impl, pattern_schema_snapshot_entries as _pattern_schema_snapshot_entries_impl, pattern_schema_suggestions as _pattern_schema_suggestions_impl, pattern_schema_suggestions_from_instances as _pattern_schema_suggestions_from_instances_impl, tier2_unreified_residue_entries as _tier2_unreified_residue_entries_impl)
from gabion.analysis.indexed_scan.deadline.deadline_runtime import (
    DeadlineArgInfo as _DeadlineArgInfoRuntime, FunctionSuiteKey as _FunctionSuiteKeyRuntime, FunctionSuiteLookupOutcome as _FunctionSuiteLookupOutcomeRuntime, FunctionSuiteLookupStatus as _FunctionSuiteLookupStatusRuntime, bind_call_args as _bind_call_args_impl, call_candidate_target_site as _call_candidate_target_site_impl, caller_param_bindings_for_call as _caller_param_bindings_for_call_impl, classify_deadline_expr as _classify_deadline_expr_impl, collect_call_edges_from_forest as _collect_call_edges_from_forest_impl, collect_call_resolution_obligation_details_from_forest as _collect_call_resolution_obligation_details_from_forest_impl, collect_call_resolution_obligations_from_forest as _collect_call_resolution_obligations_from_forest_impl, deadline_arg_info_map as _deadline_arg_info_map_impl, deadline_loop_forwarded_params as _deadline_loop_forwarded_params_impl, fallback_deadline_arg_info as _fallback_deadline_arg_info_runtime_impl, function_suite_id as _function_suite_id_impl, function_suite_key as _function_suite_key_impl, is_deadline_origin_call as _is_deadline_origin_call_impl, materialize_call_candidates as _materialize_call_candidates_impl, node_to_function_suite_id as _node_to_function_suite_id_impl, node_to_function_suite_lookup_outcome as _node_to_function_suite_lookup_outcome_impl, obligation_candidate_suite_ids as _obligation_candidate_suite_ids_impl, suite_caller_function_id as _suite_caller_function_id_impl)
from gabion.analysis.indexed_scan.deadline.deadline_obligation_summary import (
    SummarizeDeadlineObligationsDeps as _SummarizeDeadlineObligationsDeps, summarize_deadline_obligations as _summarize_deadline_obligations_impl)
from gabion.analysis.indexed_scan.index.analysis_index_stage_cache import (
    AnalysisIndexStageCacheDeps as _AnalysisIndexStageCacheDeps, analysis_index_stage_cache as _analysis_index_stage_cache_impl)
from gabion.analysis.indexed_scan.index.analysis_index_module_trees import (
    AnalysisIndexModuleTreesDeps as _AnalysisIndexModuleTreesDeps, analysis_index_module_trees as _analysis_index_module_trees_impl)
from gabion.analysis.indexed_scan.obligations.exception_obligations import (
    collect_exception_obligations as _collect_exception_obligations_impl, collect_exception_obligations_from_runtime_module as _collect_exception_obligations_impl_runtime, dead_env_map as _dead_env_map_impl)
from gabion.analysis.indexed_scan.obligations.handledness import (
    collect_handledness_witnesses as _collect_handledness_witnesses_impl)
from gabion.analysis.indexed_scan.obligations.never_invariants import (
    collect_never_invariants as _collect_never_invariants_impl, keyword_links_literal as _keyword_links_literal_impl, keyword_string_literal as _keyword_string_literal_impl, never_reason as _never_reason_impl)
from gabion.analysis.indexed_scan.scanners.report_sections import (
    extract_report_sections as _extract_report_sections_impl, parse_report_section_marker as _parse_report_section_marker_impl, spec_row_span as _spec_row_span_impl)
from gabion.analysis.indexed_scan.scanners.flow.group_propagation import (
    PropagateGroupsDeps as _PropagateGroupsDeps, propagate_groups as _propagate_groups_impl)
from gabion.analysis.indexed_scan.scanners.materialization.structured_suite_sites import (
    MaterializeStructuredSuiteSitesDeps as _MaterializeStructuredSuiteSitesDeps, MaterializeStructuredSuiteSitesForTreeDeps as _MaterializeStructuredSuiteSitesForTreeDeps, materialize_structured_suite_sites as _materialize_structured_suite_sites_impl, materialize_structured_suite_sites_for_tree as _materialize_structured_suite_sites_for_tree_impl)
from gabion.analysis.indexed_scan.obligations.invariant_propositions import (
    CollectInvariantPropositionsDeps as _CollectInvariantPropositionsDeps, collect_invariant_propositions as _collect_invariant_propositions_impl)
from gabion.analysis.indexed_scan.ast.expression_eval import (
    BoolEvalOutcome as _BoolEvalOutcome, EvalDecision as _EvalDecision, ValueEvalOutcome as _ValueEvalOutcome, branch_reachability_under_env as _branch_reachability_under_env_impl, eval_bool_expr as _eval_bool_expr_impl, eval_value_expr as _eval_value_expr_impl, is_reachability_false as _is_reachability_false_impl, is_reachability_true as _is_reachability_true_impl)
from gabion.analysis.indexed_scan.scanners.materialization.statement_materialization import (
    materialize_statement_suite_contains as _materialize_statement_suite_contains_impl)
from gabion.analysis.indexed_scan.scanners.parser_builder import (
    build_parser as _build_parser_impl)
from gabion.analysis.indexed_scan.scanners.run_entry import (
    analysis_deadline_scope as _analysis_deadline_scope_impl, normalize_transparent_decorators as _normalize_transparent_decorators_impl, resolve_baseline_path as _resolve_baseline_path_impl, resolve_synth_registry_path as _resolve_synth_registry_path_impl)
from gabion.analysis.indexed_scan.scanners.key_aliases import (
    normalize_key_expr as _normalize_key_expr_impl, stage_cache_key_aliases as _stage_cache_key_aliases_impl)
from gabion.analysis.indexed_scan.scanners.edge_param_events import (
    iter_resolved_edge_param_events as _iter_resolved_edge_param_events_impl)
from gabion.analysis.indexed_scan.scanners.flow.unused_arg_flow import (
    analyze_unused_arg_flow_indexed as _analyze_unused_arg_flow_indexed_impl)
from gabion.analysis.indexed_scan.calls.callee_resolution_helpers import (
    decorator_name as _decorator_name_impl, resolve_local_method_in_hierarchy as _resolve_local_method_in_hierarchy_impl)
from gabion.analysis.indexed_scan.index.analysis_index_builder import (
    AnalysisIndexBuildDeps as _AnalysisIndexBuildDeps, build_analysis_index as _build_analysis_index_impl)
from gabion.analysis.indexed_scan.scanners.materialization.bundle_forest_builder import (
    populate_bundle_forest_from_runtime_module as _populate_bundle_forest_impl_runtime)
from gabion.analysis.indexed_scan.scanners.materialization.dataclass_registry import (
    DataclassRegistryForTreeDeps as _DataclassRegistryForTreeDeps, dataclass_registry_for_tree as _dataclass_registry_for_tree_impl)
from gabion.analysis.indexed_scan.calls.call_ambiguities import (
    CallAmbiguitiesEmitDeps as _CallAmbiguitiesEmitDeps, emit_call_ambiguities as _emit_call_ambiguities_impl)
from gabion.analysis.indexed_scan.obligations.decision_surface_runtime import (
    DecisionSurfaceAnalyzeDeps as _DecisionSurfaceAnalyzeDeps, analyze_decision_surface_indexed as _analyze_decision_surface_indexed_impl)
from gabion.analysis.indexed_scan.scanners.flow.type_flow import (
    TypeFlowInferDeps as _TypeFlowInferDeps, infer_type_flow as _infer_type_flow_impl)
from gabion.analysis.indexed_scan.state.function_index_accumulator import (
    FunctionIndexAccumulatorDeps as _FunctionIndexAccumulatorDeps, accumulate_function_index_for_tree as _accumulate_function_index_for_tree_impl)
from gabion.analysis.indexed_scan.calls.callee_outcome_runtime import (
    ResolveCalleeDeps as _ResolveCalleeDeps, resolve_callee as _resolve_callee_impl, resolve_callee_outcome as _resolve_callee_outcome_impl, resolve_callee_outcome_from_runtime_module as _resolve_callee_outcome_impl_runtime)
from gabion.analysis.indexed_scan.calls.call_nodes_by_path import (
    CallNodesForTreeDeps as _CallNodesForTreeDeps, CollectCallNodesByPathDeps as _CollectCallNodesByPathDeps, call_nodes_for_tree as _call_nodes_for_tree_impl, collect_call_nodes_by_path as _collect_call_nodes_by_path_impl)
from gabion.analysis.indexed_scan.scanners.materialization.suite_order_relation import (
    AmbiguitySuiteRelationDeps as _AmbiguitySuiteRelationDeps, SuiteOrderRelationDeps as _SuiteOrderRelationDeps, ambiguity_suite_relation as _ambiguity_suite_relation_impl, suite_order_relation as _suite_order_relation_impl)
from gabion.analysis.indexed_scan.state.module_exports import (
    ModuleExportsCollectDeps as _ModuleExportsCollectDeps, collect_module_exports as _collect_module_exports_impl)
from gabion.analysis.indexed_scan.calls.call_ambiguity_summary import (
    CallAmbiguitySummaryDeps as _CallAmbiguitySummaryDeps, summarize_call_ambiguities as _summarize_call_ambiguities_impl)
from gabion.analysis.indexed_scan.ast.lambda_bindings import (
    ClosureLambdaFactoriesDeps as _ClosureLambdaFactoriesDeps, LambdaBindingsByCallerDeps as _LambdaBindingsByCallerDeps, collect_closure_lambda_factories as _collect_closure_lambda_factories_impl, collect_lambda_bindings_by_caller as _collect_lambda_bindings_by_caller_impl)
from gabion.schema import SynthesisResponse

from gabion.refactor.rewrite_plan import rewrite_plan_schema, validate_rewrite_plan_payload

from gabion.synthesis import NamingContext, SynthesisConfig, Synthesizer

from gabion.synthesis.emission import render_protocol_stubs as _render_protocol_stubs

from gabion.synthesis.merge import merge_bundles

from gabion.synthesis.schedule import topological_schedule

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

_PARSE_MODULE_ERROR_TYPES = (
    OSError,
    UnicodeError,
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

_FunctionSuiteKey = _FunctionSuiteKeyRuntime

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

def _invariant_term(expr: ast.AST, params: set[str]):
    expr_type = type(expr)
    if expr_type is ast.Name:
        name_expr = cast(ast.Name, expr)
        return next(iter(params.intersection({name_expr.id})), None)
    if expr_type is ast.Call:
        call_expr = cast(ast.Call, expr)
        if type(call_expr.func) is ast.Name:
            func_name = cast(ast.Name, call_expr.func)
            if func_name.id == "len" and len(call_expr.args) == 1:
                arg = call_expr.args[0]
                if type(arg) is ast.Name:
                    arg_id = cast(ast.Name, arg).id
                    return next((f"{entry}.length" for entry in params.intersection({arg_id})), None)
    return None

def _extract_invariant_from_expr(
    expr: ast.AST,
    params: set[str],
    *,
    scope: str,
    source: str = "assert",
) -> object:
    if type(expr) is not ast.Compare:
        return None
    compare_expr = cast(ast.Compare, expr)
    if len(compare_expr.ops) != 1 or len(compare_expr.comparators) != 1:
        return None
    if type(compare_expr.ops[0]) is not ast.Eq:
        return None
    left = _invariant_term(compare_expr.left, params)
    right = _invariant_term(compare_expr.comparators[0], params)
    if left is not None and right is not None:
        return InvariantProposition(
            form="Equal",
            terms=(left, right),
            scope=scope,
            source=source,
        )
    return None

class _InvariantCollector(ast.NodeVisitor):
    # dataflow-bundle: params, scope
    def __init__(self, params: set[str], scope: str) -> None:
        self._params = params
        self._scope = scope
        self.propositions: list[InvariantProposition] = []
        self._seen: set[tuple[str, tuple[str, ...], str]] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Assert(self, node: ast.Assert) -> None:
        prop = _extract_invariant_from_expr(
            node.test,
            self._params,
            scope=self._scope,
        )
        if prop is not None:
            normalized = _normalize_invariant_proposition(
                prop,
                default_scope=self._scope,
                default_source="assert",
            )
            key = (normalized.form, normalized.terms, normalized.scope or "")
            if key not in self._seen:
                self._seen.add(key)
                self.propositions.append(normalized)
        self.generic_visit(node)

def _scope_path(path: Path, root) -> str:
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)

def _collect_invariant_propositions(
    path: Path,
    *,
    ignore_params: set[str],
    project_root,
    emitters: Iterable[
        Callable[[ast.FunctionDef], Iterable[InvariantProposition]]
    ] = (),
) -> list[InvariantProposition]:
    return cast(
        list[InvariantProposition],
        _collect_invariant_propositions_impl(
            path,
            ignore_params=ignore_params,
            project_root=project_root,
            emitters=cast(Iterable[Callable[[object], Iterable[object]]], emitters),
            deps=_CollectInvariantPropositionsDeps(
                check_deadline_fn=check_deadline,
                parse_module_source_fn=_parse_module_source,
                collect_functions_fn=cast(Callable[[object], Iterable[object]], _collect_functions),
                param_names_fn=_param_names,
                scope_path_fn=_scope_path,
                invariant_collector_ctor=cast(Callable[..., object], _InvariantCollector),
                invariant_proposition_type=InvariantProposition,
                normalize_invariant_proposition_fn=_normalize_invariant_proposition,
            ),
        ),
    )

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
    check_deadline()
    if not fn.decorator_list:
        return True
    if not transparent_decorators:
        return True
    for deco in fn.decorator_list:
        check_deadline()
        name = _decorator_name(deco)
        if not name:
            return False
        if not _decorator_matches(name, transparent_decorators):
            return False
    return True

def _collect_local_class_bases(
    tree: ast.AST, parents: dict[ast.AST, ast.AST]
) -> dict[str, list[str]]:
    check_deadline()
    class_bases: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        check_deadline()
        if type(node) is ast.ClassDef:
            class_node = cast(ast.ClassDef, node)
            scopes = _enclosing_class_scopes(class_node, parents)
            qual_parts = list(scopes)
            qual_parts.append(class_node.name)
            qual = ".".join(qual_parts)
            bases: list[str] = []
            for base in class_node.bases:
                check_deadline()
                base_name = _base_identifier(base)
                if base_name:
                    bases.append(base_name)
            class_bases[qual] = bases
    return class_bases

def _local_class_name(base: str, class_bases: dict[str, list[str]]):
    if base in class_bases:
        return base
    if "." in base:
        tail = base.split(".")[-1]
        if tail in class_bases:
            return tail
    return None

def _resolve_local_method_in_hierarchy(
    class_name: str,
    method: str,
    *,
    class_bases: dict[str, list[str]],
    local_functions: set[str],
    seen: set[str],
):
    return _resolve_local_method_in_hierarchy_impl(
        class_name,
        method,
        class_bases=class_bases,
        local_functions=local_functions,
        seen=seen,
        check_deadline_fn=check_deadline,
        local_class_name_fn=_local_class_name,
    )

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
    node_type = type(node)
    return (
        node_type is ast.If
        or node_type is ast.While
        or node_type is ast.Assert
        or node_type is ast.IfExp
        or node_type is ast.Match
        or node_type is ast.comprehension
    )

def _decision_surface_form_entries(
    fn: ast.AST,
) -> list[tuple[str, ast.AST]]:
    check_deadline()
    entries: list[tuple[str, ast.AST]] = []
    for node in ast.walk(fn):
        check_deadline()
        if not is_decision_surface(node):
            continue
        node_type = type(node)
        if node_type is ast.If:
            entries.append(("if", cast(ast.If, node).test))
            continue
        if node_type is ast.While:
            entries.append(("while", cast(ast.While, node).test))
            continue
        if node_type is ast.Assert:
            entries.append(("assert", cast(ast.Assert, node).test))
            continue
        if node_type is ast.IfExp:
            entries.append(("ifexp", cast(ast.IfExp, node).test))
            continue
        if node_type is ast.Match:
            match_node = cast(ast.Match, node)
            entries.append(("match_subject", match_node.subject))
            for case in match_node.cases:
                check_deadline()
                if case.guard is not None:
                    entries.append(("match_guard", case.guard))
            continue
        for guard in cast(ast.comprehension, node).ifs:
            check_deadline()
            entries.append(("comprehension_guard", guard))
    return entries

def _decision_surface_reason_map(
    fn: FunctionNode,
    ignore_params: OptionalIgnoredParams = None,
) -> dict[str, set[str]]:
    check_deadline()
    params = set(_param_names(fn, ignore_params))
    if not params:
        return {}
    reason_map: dict[str, set[str]] = defaultdict(set)
    for reason, expr in _decision_surface_form_entries(fn):
        check_deadline()
        found = _collect_param_roots(expr, params)
        for param in found:
            check_deadline()
            reason_map[param].add(reason)
    return reason_map

def _decision_surface_params(
    fn: FunctionNode,
    ignore_params: OptionalIgnoredParams = None,
) -> set[str]:
    check_deadline()
    reason_map = _decision_surface_reason_map(fn, ignore_params)
    return set(reason_map)

def _mark_param_roots(expr: ast.AST, params: set[str], out: set[str]) -> None:
    check_deadline()
    for node in ast.walk(expr):
        check_deadline()
        node_type = type(node)
        if node_type is ast.Name and cast(ast.Name, node).id in params:
            out.add(cast(ast.Name, node).id)
            continue
        if node_type is ast.Attribute or node_type is ast.Subscript:
            root = _decision_root_name(node)
            if root in params:
                out.add(root)

def _collect_param_roots(expr: ast.AST, params: set[str]) -> set[str]:
    found: set[str] = set()
    _mark_param_roots(expr, params, found)
    return found

def _contains_boolish(expr: ast.AST) -> bool:
    check_deadline()
    for node in ast.walk(expr):
        check_deadline()
        node_type = type(node)
        if node_type is ast.Compare or node_type is ast.BoolOp:
            return True
        if node_type is ast.UnaryOp and type(cast(ast.UnaryOp, node).op) is ast.Not:
            return True
    return False

def _value_encoded_decision_params(
    fn: ast.AST,
    ignore_params = None,
) -> tuple[set[str], set[str]]:
    from gabion.analysis.indexed_scan.scanners.flow.value_encoded_decision_params import (
        ValueEncodedDecisionParamsDeps as _ValueEncodedDecisionParamsDeps)
    from gabion.analysis.indexed_scan.scanners.flow.value_encoded_decision_params import (
        value_encoded_decision_params as _value_encoded_decision_params_impl)

    return _value_encoded_decision_params_impl(
        fn,
        ignore_params,
        deps=_ValueEncodedDecisionParamsDeps(
            check_deadline_fn=check_deadline,
            param_names_fn=_param_names,
            mark_param_roots_fn=_mark_param_roots,
            contains_boolish_fn=_contains_boolish,
            collect_param_roots_fn=_collect_param_roots,
        ),
    )

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

def analyze_decision_surfaces_repo(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators = None,
    decision_tiers = None,
    require_tiers: bool = False,
    forest: Forest,
    parse_failure_witnesses = None,
    analysis_index = None,
) -> tuple[list[str], list[str], list[str]]:
    from gabion.analysis.dataflow.engine.decision_surface_analyzer import (
        DecisionSurfaceAnalyzerInput,
        analyze_decision_surfaces,
    )
    from gabion.analysis.dataflow.engine.scan_kernel import (
        ScanKernelDeps,
        ScanKernelRequest,
    )

    check_deadline()
    analyzer_output = analyze_decision_surfaces(
        data=DecisionSurfaceAnalyzerInput(
            kernel_request=ScanKernelRequest(
                paths=paths,
                project_root=project_root,
                ignore_params=ignore_params,
                strictness=strictness,
                external_filter=external_filter,
                transparent_decorators=transparent_decorators,
                parse_failure_witnesses=parse_failure_witnesses,
                analysis_index=analysis_index,
            ),
            decision_tiers=decision_tiers,
            require_tiers=require_tiers,
            forest=forest,
        ),
        deps=ScanKernelDeps(run_indexed_pass_fn=_run_indexed_pass),
        runner=lambda context: _analyze_decision_surfaces_indexed(
            context,
            decision_tiers=decision_tiers,
            require_tiers=require_tiers,
            forest=forest,
        ),
    )
    return (
        analyzer_output.surfaces,
        analyzer_output.warnings,
        analyzer_output.lint_lines,
    )

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

def analyze_value_encoded_decisions_repo(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators = None,
    decision_tiers = None,
    require_tiers: bool = False,
    forest: Forest,
    parse_failure_witnesses = None,
    analysis_index = None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    from gabion.analysis.dataflow.engine.decision_surface_analyzer import (
        DecisionSurfaceAnalyzerInput,
        analyze_value_encoded_decisions,
    )
    from gabion.analysis.dataflow.engine.scan_kernel import (
        ScanKernelDeps,
        ScanKernelRequest,
    )

    check_deadline()
    analyzer_output = analyze_value_encoded_decisions(
        data=DecisionSurfaceAnalyzerInput(
            kernel_request=ScanKernelRequest(
                paths=paths,
                project_root=project_root,
                ignore_params=ignore_params,
                strictness=strictness,
                external_filter=external_filter,
                transparent_decorators=transparent_decorators,
                parse_failure_witnesses=parse_failure_witnesses,
                analysis_index=analysis_index,
            ),
            decision_tiers=decision_tiers,
            require_tiers=require_tiers,
            forest=forest,
        ),
        deps=ScanKernelDeps(run_indexed_pass_fn=_run_indexed_pass),
        runner=lambda context: _analyze_value_encoded_decisions_indexed(
            context,
            decision_tiers=decision_tiers,
            require_tiers=require_tiers,
            forest=forest,
        ),
    )
    return (
        analyzer_output.surfaces,
        analyzer_output.warnings,
        analyzer_output.rewrites,
        analyzer_output.lint_lines,
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

def _parse_failure_witness(
    *,
    path: Path,
    stage,
    error: Exception,
) -> JSONObject:
    stage_value = stage.value if type(stage) is _ParseModuleStage else stage
    return {
        "path": str(path),
        "stage": stage_value,
        "error_type": type(error).__name__,
        "error": str(error),
    }

def _record_parse_failure_witness(
    *,
    sink: list[JSONObject],
    path: Path,
    stage,
    error: Exception,
) -> None:
    sink.append(_parse_failure_witness(path=path, stage=stage, error=error))

def _parse_failure_sink(
    parse_failure_witnesses,
) -> list[JSONObject]:
    sink = parse_failure_witnesses
    if sink is None:
        sink = []
    return sink

_ANALYSIS_INDEX_STAGE_CACHE_OP = DerivationOp(
    name="analysis_index.stage_cache",
    version=1,
    scope="gabion.analysis.dataflow_indexed_file_scan",
)

def _path_dependency_payload(
    path: Path,
) -> dict[str, object]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
    }

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

def _param_annotations_by_path(
    paths: list[Path],
    *,
    ignore_params: set[str],
    parse_failure_witnesses: list[JSONObject],
) -> dict[Path, dict[str, ParamAnnotationMap]]:
    check_deadline()
    annotations: dict[Path, dict[str, ParamAnnotationMap]] = {}
    for path in paths:
        check_deadline()
        tree = _parse_module_tree(
            path,
            stage=_ParseModuleStage.PARAM_ANNOTATIONS,
            parse_failure_witnesses=parse_failure_witnesses,
        )
        if tree is not None:
            parent = ParentAnnotator()
            parent.visit(tree)
            parents = parent.parents
            by_fn: dict[str, ParamAnnotationMap] = {}
            for fn in _collect_functions(tree):
                check_deadline()
                scopes = _enclosing_scopes(fn, parents)
                fn_key = _function_key(scopes, fn.name)
                by_fn[fn_key] = _param_annotations(fn, ignore_params)
            annotations[path] = by_fn
    return annotations

def _enclosing_function_node(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
):
    check_deadline()
    current = parents.get(node)
    while current is not None:
        check_deadline()
        current_type = type(current)
        if current_type is ast.FunctionDef or current_type is ast.AsyncFunctionDef:
            return cast(ast.FunctionDef | ast.AsyncFunctionDef, current)
        current = parents.get(current)
    return None

def _exception_param_names(expr, params: set[str]) -> list[str]:
    return _exc_exception_param_names(expr, params, check_deadline=check_deadline)

def _exception_type_name(expr):
    return _exc_exception_type_name(expr, decorator_name=_decorator_name)

def _annotation_exception_candidates(annotation) -> tuple[str, ...]:
    check_deadline()
    if not annotation:
        return ()
    try:
        expr = ast.parse(annotation, mode="eval").body
    except SyntaxError:
        return ()
    candidates: set[str] = set()
    for node in ast.walk(expr):
        check_deadline()
        node_type = type(node)
        if node_type is ast.Name:
            node_name = cast(ast.Name, node)
            cls = _exc_builtin_exception_class(node_name.id)
            if cls is not None:
                candidates.add(node_name.id)
        elif node_type is ast.Attribute:
            node_attr = cast(ast.Attribute, node)
            cls = _exc_builtin_exception_class(node_attr.attr)
            if cls is not None:
                candidates.add(node_attr.attr)
    return tuple(
        sort_once(
            candidates,
            source="_annotation_exception_candidates.candidates",
            policy=OrderPolicy.SORT,
        )
    )

def _refine_exception_name_from_annotations(
    expr,
    *,
    param_annotations: ParamAnnotationMap,
):
    check_deadline()
    direct_name = _exception_type_name(expr)
    if type(expr) is not ast.Name:
        return direct_name, None, ()
    annotation = param_annotations.get(cast(ast.Name, expr).id)
    candidates = _annotation_exception_candidates(annotation)
    if not candidates:
        return direct_name, None, ()
    if len(candidates) == 1:
        return candidates[0], "PARAM_ANNOTATION", candidates
    return direct_name, "PARAM_ANNOTATION_AMBIGUOUS", candidates

def _handler_type_names(handler_type) -> tuple[str, ...]:
    return _exc_handler_type_names(
        handler_type,
        decorator_name=_decorator_name,
        check_deadline=check_deadline,
    )

def _exception_handler_compatibility(
    exception_name,
    handler_type,
) -> str:
    return _exc_exception_handler_compatibility(
        exception_name,
        handler_type,
        decorator_name=_decorator_name,
        check_deadline=check_deadline,
    )

def _exception_path_id(
    *,
    path: str,
    function: str,
    source_kind: str,
    lineno: int,
    col: int,
    kind: str,
) -> str:
    return f"{path}:{function}:{source_kind}:{lineno}:{col}:{kind}"

def _handler_label(handler: ast.ExceptHandler) -> str:
    return _exc_handler_label(handler)

def _node_in_try_body(node: ast.AST, try_node: ast.Try) -> bool:
    return _exc_node_in_try_body(node, try_node, check_deadline=check_deadline)

def _find_handling_try(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
):
    check_deadline()
    current = parents.get(node)
    try_ancestors: list[ast.Try] = []
    while current is not None:
        check_deadline()
        if type(current) is ast.Try:
            try_ancestors.append(cast(ast.Try, current))
        current = parents.get(current)
    return next(
        (try_node for try_node in try_ancestors if _node_in_try_body(node, try_node)),
        None,
    )

def _node_in_block(node: ast.AST, block: list[ast.stmt]) -> bool:
    check_deadline()
    for stmt in block:
        check_deadline()
        if node is stmt:
            return True
        for child in ast.walk(stmt):
            check_deadline()
            if node is child:
                return True
    return False

def _names_in_expr(expr: ast.AST) -> set[str]:
    check_deadline()
    names: set[str] = set()
    for node in ast.walk(expr):
        check_deadline()
        if type(node) is ast.Name:
            names.add(cast(ast.Name, node).id)
    return names

def _eval_value_expr(expr: ast.AST, env: dict[str, JSONValue]) -> _ValueEvalOutcome:
    return _eval_value_expr_impl(
        expr,
        env,
        check_deadline_fn=check_deadline,
    )


def _eval_bool_expr(expr: ast.AST, env: dict[str, JSONValue]) -> _BoolEvalOutcome:
    return _eval_bool_expr_impl(
        expr,
        env,
        check_deadline_fn=check_deadline,
    )


def _branch_reachability_under_env(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    env: dict[str, JSONValue],
) -> _EvalDecision:
    return _branch_reachability_under_env_impl(
        node,
        parents,
        env,
        check_deadline_fn=check_deadline,
        node_in_block_fn=_node_in_block,
    )


def _is_reachability_false(reachability: _EvalDecision) -> bool:
    return _is_reachability_false_impl(reachability)


def _is_reachability_true(reachability: _EvalDecision) -> bool:
    return _is_reachability_true_impl(reachability)

def _collect_handledness_witnesses(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
) -> list[JSONObject]:
    return _collect_handledness_witnesses_impl(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        check_deadline_fn=check_deadline,
        parent_annotator_factory=ParentAnnotator,
        collect_functions_fn=_collect_functions,
        param_names_fn=_param_names,
        param_annotations_fn=_param_annotations,
        normalize_snapshot_path_fn=_normalize_snapshot_path,
        find_handling_try_fn=_find_handling_try,
        enclosing_function_node_fn=_enclosing_function_node,
        enclosing_scopes_fn=_enclosing_scopes,
        function_key_fn=_function_key,
        refine_exception_name_from_annotations_fn=_refine_exception_name_from_annotations,
        exception_param_names_fn=_exception_param_names,
        exception_path_id_fn=_exception_path_id,
        exception_handler_compatibility_fn=_exception_handler_compatibility,
        handler_label_fn=_handler_label,
        handler_type_names_fn=_handler_type_names,
    )

def _dead_env_map(
    deadness_witnesses,
) -> dict[tuple[str, str], dict[str, tuple[JSONValue, JSONObject]]]:
    return _dead_env_map_impl(
        deadness_witnesses,
        check_deadline_fn=check_deadline,
        sequence_or_none_fn=sequence_or_none,
        mapping_or_none_fn=mapping_or_none,
        literal_eval_error_types=_LITERAL_EVAL_ERROR_TYPES,
    )

def _collect_exception_obligations(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    handledness_witnesses=None,
    deadness_witnesses=None,
    never_exceptions=None,
) -> list[JSONObject]:
    return _collect_exception_obligations_impl_runtime(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        handledness_witnesses=handledness_witnesses,
        deadness_witnesses=deadness_witnesses,
        never_exceptions=never_exceptions,
        runtime_module=sys.modules[__name__],
    )

def _keyword_string_literal(call: ast.Call, key: str) -> str:
    return _keyword_string_literal_impl(
        call,
        key,
        check_deadline_fn=check_deadline,
    )

def _keyword_links_literal(call: ast.Call) -> list[JSONObject]:
    return _keyword_links_literal_impl(
        call,
        check_deadline_fn=check_deadline,
        sort_once_fn=sort_once,
    )

def _never_reason(call: ast.Call):
    return _never_reason_impl(call, check_deadline_fn=check_deadline)

def _collect_never_invariants(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    forest: Forest,
    marker_aliases: Sequence[str] = (),
    deadness_witnesses=None,
) -> list[JSONObject]:
    return _collect_never_invariants_impl(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        forest=forest,
        marker_aliases=marker_aliases,
        deadness_witnesses=deadness_witnesses,
        check_deadline_fn=check_deadline,
        parent_annotator_factory=ParentAnnotator,
        collect_functions_fn=_collect_functions,
        param_names_fn=_param_names,
        normalize_snapshot_path_fn=_normalize_snapshot_path,
        enclosing_function_node_fn=_enclosing_function_node,
        enclosing_scopes_fn=_enclosing_scopes,
        function_key_fn=_function_key,
        exception_param_names_fn=_exception_param_names,
        node_span_fn=_node_span,
        dead_env_map_fn=_dead_env_map,
        branch_reachability_under_env_fn=_branch_reachability_under_env,
        is_reachability_false_fn=_is_reachability_false,
        is_reachability_true_fn=_is_reachability_true,
        names_in_expr_fn=_names_in_expr,
        sort_once_fn=sort_once,
        order_policy_sort=OrderPolicy.SORT,
        order_policy_enforce=OrderPolicy.ENFORCE,
        is_marker_call_fn=_is_marker_call,
        decorator_name_fn=_decorator_name,
        require_not_none_fn=require_not_none,
    )

_DEADLINE_CHECK_METHODS = {"check", "expired"}

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

class _DeadlineFunctionCollector(ast.NodeVisitor):
    def __init__(self, root: ast.AST, params: set[str]) -> None:
        self._root = root
        self._params = params
        self.loop = False
        self.check_params: set[str] = set()
        self.ambient_check = False
        self.loop_sites: list[_DeadlineLoopFacts] = []
        self._loop_stack: list[_DeadlineLoopFacts] = []
        self.assignments: list[tuple[list[ast.AST], OptionalAstNode, OptionalSpan4]] = []

    def _mark_param_check(self, name: str) -> None:
        if self._loop_stack:
            self._loop_stack[-1].check_params.add(name)
        else:
            self.check_params.add(name)

    def _mark_ambient_check(self) -> None:
        if self._loop_stack:
            self._loop_stack[-1].ambient_check = True
        else:
            self.ambient_check = True

    def _record_call_span(self, node: ast.AST) -> None:
        if self._loop_stack:
            span = _node_span(node)
            if span is not None:
                self._loop_stack[-1].call_spans.add(span)

    def _iter_marks_ambient(self, expr: ast.AST) -> bool:
        if type(expr) is ast.Call:
            func = cast(ast.Call, expr).func
            func_type = type(func)
            if func_type is ast.Name:
                return cast(ast.Name, func).id == "deadline_loop_iter"
            if func_type is ast.Attribute:
                return cast(ast.Attribute, func).attr == "deadline_loop_iter"
        return False

    def _visit_loop_body(
        self,
        node: ast.AST,
        kind: str,
        *,
        ambient_check: bool = False,
    ) -> None:
        self.loop = True
        loop_fact = _DeadlineLoopFacts(
            span=_node_span(node),
            kind=kind,
            depth=len(self._loop_stack) + 1,
            ambient_check=ambient_check,
        )
        self._loop_stack.append(loop_fact)
        for stmt in getattr(node, "body", []):
            check_deadline()
            self.visit(stmt)
        self._loop_stack.pop()
        self.loop_sites.append(loop_fact)
        for stmt in getattr(node, "orelse", []):
            check_deadline()
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node is not self._root:
            return
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node is not self._root:
            return
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_For(self, node: ast.For) -> None:
        self.loop = True
        ambient_check = self._iter_marks_ambient(node.iter)
        self.visit(node.target)
        self.visit(node.iter)
        self._visit_loop_body(node, "for", ambient_check=ambient_check)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.loop = True
        ambient_check = self._iter_marks_ambient(node.iter)
        self.visit(node.target)
        self.visit(node.iter)
        self._visit_loop_body(node, "async_for", ambient_check=ambient_check)

    def visit_While(self, node: ast.While) -> None:
        self.loop = True
        self.visit(node.test)
        self._visit_loop_body(node, "while")

    def visit_Call(self, node: ast.Call) -> None:
        self._record_call_span(node)
        func = node.func
        func_type = type(func)
        if func_type is ast.Attribute:
            attribute_func = cast(ast.Attribute, func)
            if attribute_func.attr == "deadline_loop_iter":
                self._mark_ambient_check()
            if (
                attribute_func.attr in _DEADLINE_CHECK_METHODS
                and type(attribute_func.value) is ast.Name
                and cast(ast.Name, attribute_func.value).id in self._params
            ):
                self._mark_param_check(cast(ast.Name, attribute_func.value).id)
            if attribute_func.attr == "check_deadline" and node.args:
                first = node.args[0]
                if type(first) is ast.Name and cast(ast.Name, first).id in self._params:
                    self._mark_param_check(cast(ast.Name, first).id)
            if attribute_func.attr in {"check_deadline", "require_deadline"} and not node.args:
                self._mark_ambient_check()
        elif func_type is ast.Name:
            name_func = cast(ast.Name, func)
            if name_func.id == "deadline_loop_iter":
                self._mark_ambient_check()
            if name_func.id == "check_deadline" and node.args:
                first = node.args[0]
                if type(first) is ast.Name and cast(ast.Name, first).id in self._params:
                    self._mark_param_check(cast(ast.Name, first).id)
            if name_func.id in {"check_deadline", "require_deadline"} and not node.args:
                self._mark_ambient_check()
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.assignments.append((node.targets, node.value, _node_span(node)))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.assignments.append(([node.target], node.value, _node_span(node)))
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.assignments.append(([node.target], node.value, _node_span(node)))
        self.generic_visit(node)

@dataclass
class _DeadlineLoopFacts:
    span: OptionalSpan4
    kind: str
    depth: int = 1
    check_params: set[str] = field(default_factory=set)
    ambient_check: bool = False
    call_spans: set[tuple[int, int, int, int]] = field(default_factory=set)

@dataclass(frozen=True)
class _DeadlineLocalInfo:
    origin_vars: set[str]
    origin_spans: dict[str, tuple[int, int, int, int]]
    alias_to_param: dict[str, str]

@dataclass(frozen=True)
class _DeadlineFunctionFacts:
    path: Path
    qual: str
    span: OptionalSpan4
    loop: bool
    check_params: set[str]
    ambient_check: bool
    loop_sites: list[_DeadlineLoopFacts]
    local_info: _DeadlineLocalInfo

def _collect_deadline_local_info(
    assignments: list[tuple[list[ast.AST], OptionalAstNode, OptionalSpan4]],
    params: set[str],
) -> _DeadlineLocalInfo:
    from gabion.analysis.indexed_scan.deadline.deadline_local_info import (
        CollectDeadlineLocalInfoDeps as _CollectDeadlineLocalInfoDeps)
    from gabion.analysis.indexed_scan.deadline.deadline_local_info import (
        collect_deadline_local_info as _collect_deadline_local_info_impl)

    return cast(
        _DeadlineLocalInfo,
        _collect_deadline_local_info_impl(
            assignments,
            params,
            deps=_CollectDeadlineLocalInfoDeps(
                check_deadline_fn=check_deadline,
                is_deadline_origin_call_fn=_is_deadline_origin_call,
                target_names_fn=_target_names,
                deadline_local_info_ctor=_DeadlineLocalInfo,
            ),
        ),
    )

def _collect_deadline_function_facts(
    paths: list[Path],
    *,
    project_root = None,
    ignore_params: set[str],
    parse_failure_witnesses: list[JSONObject],
    trees = None,
    analysis_index = None,
    stage_cache_fn = None,
) -> dict[str, _DeadlineFunctionFacts]:
    from gabion.analysis.indexed_scan.deadline.deadline_function_facts import (
        collect_deadline_function_facts_from_runtime_module as _collect_deadline_function_facts_impl)

    return cast(
        dict[str, _DeadlineFunctionFacts],
        _collect_deadline_function_facts_impl(
            paths,
            project_root=project_root,
            ignore_params=ignore_params,
            parse_failure_witnesses=parse_failure_witnesses,
            trees=trees,
            analysis_index=analysis_index,
            stage_cache_fn=stage_cache_fn,
            runtime_module=sys.modules[__name__],
        ),
    )

def _deadline_function_facts_for_tree(
    path: Path,
    tree: ast.AST,
    *,
    project_root,
    ignore_params: set[str],
) -> dict[str, _DeadlineFunctionFacts]:
    check_deadline()
    parents = ParentAnnotator()
    parents.visit(tree)
    module = _module_name(path, project_root)
    facts: dict[str, _DeadlineFunctionFacts] = {}
    for fn in _collect_functions(tree):
        check_deadline()
        scopes = _enclosing_scopes(fn, parents.parents)
        qual_parts = [module] if module else []
        if scopes:
            qual_parts.extend(scopes)
        qual_parts.append(fn.name)
        qual = ".".join(qual_parts)
        params = set(_param_names(fn, ignore_params))
        collector = _DeadlineFunctionCollector(fn, params)
        collector.visit(fn)
        local_info = _collect_deadline_local_info(collector.assignments, params)
        facts[qual] = _DeadlineFunctionFacts(
            path=path,
            qual=qual,
            span=_node_span(fn),
            loop=collector.loop,
            check_params=set(collector.check_params),
            ambient_check=collector.ambient_check,
            loop_sites=list(collector.loop_sites),
            local_info=local_info,
        )
    return facts

def _collect_call_nodes_by_path(
    paths: list[Path],
    *,
    trees = None,
    parse_failure_witnesses: list[JSONObject],
    analysis_index = None,
) -> dict[Path, dict[tuple[int, int, int, int], list[ast.Call]]]:
    return cast(
        dict[Path, dict[tuple[int, int, int, int], list[ast.Call]]],
        _collect_call_nodes_by_path_impl(
            paths,
            trees=trees,
            parse_failure_witnesses=parse_failure_witnesses,
            analysis_index=analysis_index,
            deps=_CollectCallNodesByPathDeps(
                check_deadline_fn=check_deadline,
                analysis_index_stage_cache_fn=_analysis_index_stage_cache,
                stage_cache_spec_ctor=_StageCacheSpec,
                parse_module_stage_call_nodes=_ParseModuleStage.CALL_NODES,
                parse_stage_cache_key_fn=_parse_stage_cache_key,
                empty_cache_semantic_context=_EMPTY_CACHE_SEMANTIC_CONTEXT,
                call_nodes_for_tree_fn=_call_nodes_for_tree,
                parse_module_tree_fn=_parse_module_tree,
            ),
        ),
    )

def _call_nodes_for_tree(
    tree: ast.AST,
) -> dict[tuple[int, int, int, int], list[ast.Call]]:
    return cast(
        dict[tuple[int, int, int, int], list[ast.Call]],
        _call_nodes_for_tree_impl(
            tree,
            deps=_CallNodesForTreeDeps(
                check_deadline_fn=check_deadline,
                node_span_fn=_node_span,
            ),
        ),
    )

def _collect_call_edges(
    *,
    by_name: dict[str, list[FunctionInfo]],
    by_qual: dict[str, FunctionInfo],
    symbol_table: SymbolTable,
    project_root,
    class_index: dict[str, ClassInfo],
    resolve_callee_outcome_fn = None,
) -> dict[str, set[str]]:
    from gabion.analysis.indexed_scan.calls.call_edges import CollectCallEdgesDeps as _CollectCallEdgesDeps
    from gabion.analysis.indexed_scan.calls.call_edges import collect_call_edges as _collect_call_edges_impl

    return cast(
        dict[str, set[str]],
        _collect_call_edges_impl(
            by_name=cast(dict[str, list[object]], by_name),
            by_qual=cast(dict[str, object], by_qual),
            symbol_table=symbol_table,
            project_root=project_root,
            class_index=cast(dict[str, object], class_index),
            resolve_callee_outcome_fn=resolve_callee_outcome_fn
            or _resolve_callee_outcome,
            deps=_CollectCallEdgesDeps(
                check_deadline_fn=check_deadline,
                is_test_path_fn=_is_test_path,
            ),
        ),
    )

def _function_suite_key(path: str, qual: str) -> _FunctionSuiteKey:
    return cast(_FunctionSuiteKey, _function_suite_key_impl(path, qual))

def _function_suite_id(key: _FunctionSuiteKey) -> NodeId:
    return _function_suite_id_impl(cast(_FunctionSuiteKeyRuntime, key))

_FunctionSuiteLookupStatus = _FunctionSuiteLookupStatusRuntime
_FunctionSuiteLookupOutcome = _FunctionSuiteLookupOutcomeRuntime

def _node_to_function_suite_lookup_outcome(
    forest: Forest,
    node_id: NodeId,
) -> _FunctionSuiteLookupOutcome:
    return cast(
        _FunctionSuiteLookupOutcome,
        _node_to_function_suite_lookup_outcome_impl(forest, node_id),
    )

def _suite_caller_function_id(
    suite_node: Node,
) -> NodeId:
    return _suite_caller_function_id_impl(suite_node)

def _node_to_function_suite_id(
    forest: Forest,
    node_id: NodeId,
):
    return _node_to_function_suite_id_impl(forest, node_id)

def _obligation_candidate_suite_ids(
    *,
    by_name: dict[str, list[FunctionInfo]],
    callee_key: str,
) -> set[NodeId]:
    return _obligation_candidate_suite_ids_impl(by_name=by_name, callee_key=callee_key)

def _collect_call_edges_from_forest(
    forest: Forest,
    *,
    by_name: dict[str, list[FunctionInfo]],
) -> dict[NodeId, set[NodeId]]:
    return _collect_call_edges_from_forest_impl(forest, by_name=by_name)

def _collect_call_resolution_obligations_from_forest(
    forest: Forest,
) -> list[tuple[NodeId, NodeId, tuple[int, int, int, int], str]]:
    return _collect_call_resolution_obligations_from_forest_impl(forest)

def _collect_call_resolution_obligation_details_from_forest(
    forest: Forest,
) -> list[tuple[NodeId, NodeId, tuple[int, int, int, int], str, str]]:
    return _collect_call_resolution_obligation_details_from_forest_impl(forest)

def _call_candidate_target_site(
    *,
    forest: Forest,
    candidate: FunctionInfo,
) -> NodeId:
    return _call_candidate_target_site_impl(forest=forest, candidate=candidate)

def _materialize_call_candidates(
    *,
    forest: Forest,
    by_name: dict[str, list[FunctionInfo]],
    by_qual: dict[str, FunctionInfo],
    symbol_table: SymbolTable,
    project_root,
    class_index: dict[str, ClassInfo],
    resolve_callee_outcome_fn = None,
) -> None:
    if resolve_callee_outcome_fn is None:
        resolve_callee_outcome_fn = _resolve_callee_outcome
    _materialize_call_candidates_impl(
        forest=forest,
        by_name=by_name,
        by_qual=by_qual,
        symbol_table=symbol_table,
        project_root=project_root,
        class_index=cast(dict[str, object], class_index),
        resolve_callee_outcome_fn=resolve_callee_outcome_fn,
        normalize_snapshot_path_fn=_normalize_snapshot_path,
    )

_DeadlineArgInfo = _DeadlineArgInfoRuntime

def _bind_call_args(
    call_node: ast.Call,
    callee: FunctionInfo,
    *,
    strictness: str,
) -> dict[str, ast.AST]:
    return _bind_call_args_impl(call_node, callee, strictness=strictness)

def _caller_param_bindings_for_call(
    call: CallArgs,
    callee: FunctionInfo,
    *,
    strictness: str,
) -> dict[str, set[str]]:
    return _caller_param_bindings_for_call_impl(call, callee, strictness=strictness)

def _classify_deadline_expr(
    expr: ast.AST,
    *,
    alias_to_param: Mapping[str, str],
    origin_vars: set[str],
) -> _DeadlineArgInfo:
    return cast(
        _DeadlineArgInfo,
        _classify_deadline_expr_impl(
            expr,
            alias_to_param=alias_to_param,
            origin_vars=origin_vars,
        ),
    )

def _fallback_deadline_arg_info(
    call: CallArgs,
    callee: FunctionInfo,
    *,
    strictness: str,
) -> dict[str, _DeadlineArgInfo]:
    return cast(
        dict[str, _DeadlineArgInfo],
        _fallback_deadline_arg_info_runtime_impl(call, callee, strictness=strictness),
    )

def _deadline_arg_info_map(
    call: CallArgs,
    callee: FunctionInfo,
    *,
    call_node: OptionalAstCall,
    alias_to_param: Mapping[str, str],
    origin_vars: set[str],
    strictness: str,
) -> dict[str, _DeadlineArgInfo]:
    return cast(
        dict[str, _DeadlineArgInfo],
        _deadline_arg_info_map_impl(
            call,
            callee,
            call_node=call_node,
            alias_to_param=alias_to_param,
            origin_vars=origin_vars,
            strictness=strictness,
        ),
    )

def _deadline_loop_forwarded_params(
    *,
    qual: str,
    loop_fact: _DeadlineLoopFacts,
    deadline_params: Mapping[str, set[str]],
    call_infos: Mapping[str, list[tuple[CallArgs, FunctionInfo, dict[str, "_DeadlineArgInfo"]]]],
) -> set[str]:
    return _deadline_loop_forwarded_params_impl(
        qual=qual,
        loop_fact=loop_fact,
        deadline_params=deadline_params,
        call_infos=cast(
            Mapping[str, list[tuple[CallArgs, FunctionInfo, dict[str, _DeadlineArgInfoRuntime]]]],
            call_infos,
        ),
    )

@dataclass(frozen=True)
class _ProjectionSpan:
    line: int
    col: int
    end_line: int
    end_col: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.line, self.col, self.end_line, self.end_col)

@dataclass(frozen=True)
class _AmbiguitySuiteRow:
    path: str
    qual: str
    suite_kind: str
    span: _ProjectionSpan

def _decode_projection_span(row: Mapping[str, JSONValue]) -> _ProjectionSpan:
    def _coerce(name: str, value: JSONValue) -> int:
        if value is None:
            never(
                f"projection spec missing {name}",
                field=name,
            )
        try:
            return int(value)
        except (TypeError, ValueError):
            never(
                f"projection spec {name} must be an int",
                field=name,
                value=value,
            )

    line = _coerce("span_line", row.get("span_line"))
    col = _coerce("span_col", row.get("span_col"))
    end_line = _coerce("span_end_line", row.get("span_end_line"))
    end_col = _coerce("span_end_col", row.get("span_end_col"))
    if line < 0 or col < 0 or end_line < 0 or end_col < 0:
        never(
            "projection spec span fields must be non-negative",
            span_line=line,
            span_col=col,
            span_end_line=end_line,
            span_end_col=end_col,
        )
    return _ProjectionSpan(line=line, col=col, end_line=end_line, end_col=end_col)

def _spec_row_span(row: Mapping[str, JSONValue]):
    return _spec_row_span_impl(row)

def _materialize_projection_spec_rows(
    *,
    spec: ProjectionSpec,
    projected: Iterable[Mapping[str, JSONValue]],
    forest: Forest,
    row_to_site: Callable[[Mapping[str, JSONValue]], NodeIdOrNone],
) -> None:
    from gabion.analysis.dataflow.io.dataflow_reporting_helpers import _materialize_projection_spec_rows as _impl

    _impl(spec=spec, projected=projected, forest=forest, row_to_site=row_to_site)

def _suite_order_depth(suite_kind: str) -> int:
    if suite_kind in {"function", "spec"}:
        return 0
    return 1

def _suite_order_relation(
    forest: Forest,
) -> tuple[list[dict[str, JSONValue]], dict[tuple[object, ...], NodeId]]:
    return _suite_order_relation_impl(
        forest,
        deps=_SuiteOrderRelationDeps(
            check_deadline_fn=check_deadline,
            never_fn=never,
            int_tuple4_or_none_fn=int_tuple4_or_none,
            suite_order_depth_fn=_suite_order_depth,
            sort_once_fn=sort_once,
        ),
    )

def _suite_order_row_to_site(
    row: Mapping[str, JSONValue],
    suite_index: Mapping[tuple[object, ...], NodeId],
):
    path = str(row.get("suite_path", "") or "")
    qual = str(row.get("suite_qual", "") or "")
    suite_kind = str(row.get("suite_kind", "") or "")
    if not path or not qual or not suite_kind:
        return None
    try:
        span_line = int(row.get("span_line", -1))
        span_col = int(row.get("span_col", -1))
        span_end_line = int(row.get("span_end_line", -1))
        span_end_col = int(row.get("span_end_col", -1))
    except (TypeError, ValueError):
        return None
    key = (
        path,
        qual,
        suite_kind,
        span_line,
        span_col,
        span_end_line,
        span_end_col,
    )
    return suite_index.get(key)

def _materialize_suite_order_spec(
    *,
    forest: Forest,
) -> None:
    from gabion.analysis.dataflow.io.dataflow_spec_materialization import (
        materialize_suite_order_spec,
    )

    materialize_suite_order_spec(
        forest=forest,
        suite_order_relation_runner=_suite_order_relation,
        row_to_site_runner=_suite_order_row_to_site,
        projection_spec=SUITE_ORDER_SPEC,
        projection_apply_runner=apply_spec,
        materialize_rows_runner=_materialize_projection_spec_rows,
    )

def _ambiguity_suite_relation(
    forest: Forest,
) -> list[dict[str, JSONValue]]:
    return _ambiguity_suite_relation_impl(
        forest,
        deps=_AmbiguitySuiteRelationDeps(
            check_deadline_fn=check_deadline,
            never_fn=never,
            int_tuple4_or_none_fn=int_tuple4_or_none,
        ),
    )

def _decode_ambiguity_suite_row(row: Mapping[str, JSONValue]) -> _AmbiguitySuiteRow:
    path = str(row.get("suite_path", "") or "")
    qual = str(row.get("suite_qual", "") or "")
    suite_kind = str(row.get("suite_kind", "") or "")
    if not path or not qual or not suite_kind:
        never(
            "ambiguity suite row missing suite identity",
            path=path,
            qual=qual,
            suite_kind=suite_kind,
        )
    return _AmbiguitySuiteRow(
        path=path,
        qual=qual,
        suite_kind=suite_kind,
        span=_decode_projection_span(row),
    )

def _ambiguity_suite_row_to_suite(
    row: Mapping[str, JSONValue],
    forest: Forest,
) -> NodeId:
    decoded = _decode_ambiguity_suite_row(row)
    return forest.add_suite_site(
        decoded.path,
        decoded.qual,
        decoded.suite_kind,
        span=decoded.span.as_tuple(),
    )

def _ambiguity_virtual_count_gt_1(
    row: Mapping[str, JSONValue],
    _params: Mapping[str, JSONValue],
) -> bool:
    try:
        return int(row.get("count", 0) or 0) > 1
    except (TypeError, ValueError):
        return False

def _materialize_ambiguity_suite_agg_spec(
    *,
    forest: Forest,
) -> None:
    from gabion.analysis.dataflow.io.dataflow_spec_materialization import (
        materialize_ambiguity_suite_agg_spec,
    )

    materialize_ambiguity_suite_agg_spec(
        forest=forest,
        ambiguity_relation_runner=_ambiguity_suite_relation,
        row_to_suite_runner=_ambiguity_suite_row_to_suite,
        projection_spec=AMBIGUITY_SUITE_AGG_SPEC,
        projection_apply_runner=apply_spec,
        materialize_rows_runner=_materialize_projection_spec_rows,
    )

def _materialize_ambiguity_virtual_set_spec(
    *,
    forest: Forest,
) -> None:
    from gabion.analysis.dataflow.io.dataflow_spec_materialization import (
        materialize_ambiguity_virtual_set_spec,
    )

    materialize_ambiguity_virtual_set_spec(
        forest=forest,
        ambiguity_relation_runner=_ambiguity_suite_relation,
        row_to_suite_runner=_ambiguity_suite_row_to_suite,
        projection_spec=AMBIGUITY_VIRTUAL_SET_SPEC,
        projection_apply_runner=apply_spec,
        materialize_rows_runner=_materialize_projection_spec_rows,
        count_gt_1_runner=_ambiguity_virtual_count_gt_1,
    )


def run_scan_domain_orchestrator(
    *,
    paths: list[Path],
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    forest: Forest,
    transparent_decorators = None,
    decision_tiers = None,
    require_tiers: bool = False,
    parse_failure_witnesses = None,
    analysis_index = None,
) -> dict[str, list[str]]:
    decision_surfaces, decision_warnings, decision_lint = analyze_decision_surfaces_repo(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        decision_tiers=decision_tiers,
        require_tiers=require_tiers,
        forest=forest,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
    )
    value_surfaces, value_warnings, value_rewrites, value_lint = analyze_value_encoded_decisions_repo(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        decision_tiers=decision_tiers,
        require_tiers=require_tiers,
        forest=forest,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
    )
    return {
        "decision_surfaces": decision_surfaces,
        "decision_warnings": decision_warnings,
        "decision_lint": decision_lint,
        "value_surfaces": value_surfaces,
        "value_warnings": value_warnings,
        "value_rewrites": value_rewrites,
        "value_lint": value_lint,
    }

def _span_line_col(span):
    from gabion.analysis.dataflow.engine.dataflow_lint_helpers import _span_line_col as _impl

    return _impl(span)

def _summarize_call_ambiguities(
    entries: list[JSONObject],
    *,
    max_entries: int = 20,
) -> list[str]:
    return _summarize_call_ambiguities_impl(
        entries,
        max_entries=max_entries,
        deps=_CallAmbiguitySummaryDeps(
            check_deadline_fn=check_deadline,
            apply_spec_fn=apply_spec,
            ambiguity_summary_spec=AMBIGUITY_SUMMARY_SPEC,
            spec_metadata_lines_from_payload_fn=spec_metadata_lines_from_payload,
            spec_metadata_payload_fn=spec_metadata_payload,
            sort_once_fn=sort_once,
            format_span_fields_fn=_format_span_fields,
        ),
    )

def _format_span_fields(
    line: object,
    col: object,
    end_line: object,
    end_col: object,
) -> str:
    from gabion.analysis.dataflow.io.dataflow_reporting_helpers import _format_span_fields as _impl

    return _impl(line, col, end_line, end_col)

def _lint_line(path: str, line: int, col: int, code: str, message: str) -> str:
    return f"{path}:{line}:{col}: {code} {message}".strip()

def _add_interned_alt(
    *,
    forest: Forest,
    kind: str,
    inputs: Iterable[NodeId],
    evidence = None,
) -> Alt:
    return forest.add_alt(kind, inputs, evidence=evidence)

def _decision_param_lint_line(
    info: "FunctionInfo",
    param: str,
    *,
    project_root,
    code: str,
    message: str,
):
    from gabion.analysis.dataflow.engine.dataflow_lint_helpers import _decision_param_lint_line as _impl

    return _impl(
        info,
        param,
        project_root=project_root,
        code=code,
        message=message,
    )

def _decision_tier_for(
    info: "FunctionInfo",
    param: str,
    *,
    tier_map: dict[str, int],
    project_root,
):
    check_deadline()
    if not tier_map:
        return None
    span = info.param_spans.get(param)
    if span is not None:
        path = _normalize_snapshot_path(info.path, project_root)
        line, col, _, _ = span
        location = f"{path}:{line + 1}:{col + 1}"
        for key in (location, f"{location}:{param}"):
            check_deadline()
            if key in tier_map:
                return tier_map[key]
    for key in (f"{info.qual}:{param}", f"{info.qual}.{param}", param):
        check_deadline()
        if key in tier_map:
            return tier_map[key]
    return None

def _collect_transitive_callers(
    callers_by_qual: dict[str, set[str]],
    by_qual: dict[str, FunctionInfo],
) -> dict[str, set[str]]:
    check_deadline()
    transitive: dict[str, set[str]] = {}
    for qual in by_qual:
        check_deadline()
        seen: set[str] = set()
        stack = list(callers_by_qual.get(qual, set()))
        while stack:
            check_deadline()
            caller = stack.pop()
            if caller in seen:
                continue
            seen.add(caller)
            stack.extend(callers_by_qual.get(caller, set()))
        transitive[qual] = seen
    return transitive

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

_IndexedPassResult = TypeVar("_IndexedPassResult")

_ResolvedEdgeAcc = TypeVar("_ResolvedEdgeAcc")

_ResolvedEdgeOut = TypeVar("_ResolvedEdgeOut")

_ModuleArtifactAcc = TypeVar("_ModuleArtifactAcc")

_ModuleArtifactOut = TypeVar("_ModuleArtifactOut")

OptionalProjectRoot = Path | None

OptionalDecorators = set[str] | None

OptionalParseFailures = list[JSONObject] | None

OptionalAnalysisIndex = AnalysisIndex | None

@dataclass(frozen=True)
class _IndexedPassContext:
    paths: list[Path]
    project_root: OptionalProjectRoot
    ignore_params: set[str]
    strictness: str
    external_filter: bool
    transparent_decorators: OptionalDecorators
    parse_failure_witnesses: list[JSONObject]
    analysis_index: AnalysisIndex

@dataclass(frozen=True)
class _IndexedPassSpec(Generic[_IndexedPassResult]):
    pass_id: str
    run: Callable[[_IndexedPassContext], _IndexedPassResult]

@dataclass(frozen=True)
class _ResolvedEdgeReducerSpec(Generic[_ResolvedEdgeAcc, _ResolvedEdgeOut]):
    reducer_id: str
    init: Callable[[], _ResolvedEdgeAcc]
    fold: Callable[[_ResolvedEdgeAcc, _ResolvedCallEdge], None]
    finish: Callable[[_ResolvedEdgeAcc], _ResolvedEdgeOut]

@dataclass(frozen=True)
class _ModuleArtifactSpec(Generic[_ModuleArtifactAcc, _ModuleArtifactOut]):
    artifact_id: str
    stage: _ParseModuleStage
    init: Callable[[], _ModuleArtifactAcc]
    fold: Callable[[_ModuleArtifactAcc, Path, ast.Module], None]
    finish: Callable[[_ModuleArtifactAcc], _ModuleArtifactOut]

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

@dataclass(frozen=True)
class _CacheSemanticContext:
    forest_spec_id: OptionalString = None
    fingerprint_seed_revision: OptionalString = None

@dataclass(frozen=True)
class _StageCacheIdentitySpec:
    stage: Literal["parse", "index", "projection"]
    forest_spec_id: str
    fingerprint_seed_revision: str
    normalized_config: JSONValue

_EMPTY_CACHE_SEMANTIC_CONTEXT = _CacheSemanticContext()

_ANALYSIS_INDEX_RESUME_VARIANTS_KEY = "resume_variants"

_ANALYSIS_INDEX_RESUME_MAX_VARIANTS = 4

_CACHE_IDENTITY_PREFIX = "aspf:sha1:"

_CACHE_IDENTITY_DIGEST_HEX = re.compile(r"^[0-9a-f]{40}$")

@dataclass(frozen=True)
class _CacheIdentity:
    value: str

    @classmethod
    def from_digest(cls, digest: str) -> "_CacheIdentity | None":
        cleaned = str(digest or "").strip().lower()
        if not _CACHE_IDENTITY_DIGEST_HEX.fullmatch(cleaned):
            return None
        return cls(f"{_CACHE_IDENTITY_PREFIX}{cleaned}")

    @classmethod
    def from_boundary(cls, raw_identity) -> "_CacheIdentity | None":
        identity = str(raw_identity or "").strip()
        if not identity:
            return None
        if identity.startswith(_CACHE_IDENTITY_PREFIX):
            digest = identity[len(_CACHE_IDENTITY_PREFIX) :]
            return cls.from_digest(digest)
        return cls.from_digest(identity)

    @classmethod
    def from_boundary_required(cls, raw_identity, *, field: str) -> "_CacheIdentity":
        identity = cls.from_boundary(raw_identity)
        if identity is None:
            never("invalid cache identity", field=field)
            return cls(value="")  # pragma: no cover - never() raises
        return identity

@dataclass(frozen=True)
class _ResumeCacheIdentityPair:
    canonical_index: _CacheIdentity
    canonical_projection: _CacheIdentity

    def encode(self) -> dict[str, str]:
        return {
            "index_cache_identity": self.canonical_index.value,
            "projection_cache_identity": self.canonical_projection.value,
        }

    @classmethod
    def decode_required(cls, payload: Mapping[str, JSONValue]) -> "_ResumeCacheIdentityPair":
        return cls(
            canonical_index=_CacheIdentity.from_boundary_required(
                payload.get("index_cache_identity"),
                field="index_cache_identity",
            ),
            canonical_projection=_CacheIdentity.from_boundary_required(
                payload.get("projection_cache_identity"),
                field="projection_cache_identity",
            ),
        )

def _sorted_text(values = None) -> tuple[str, ...]:
    if values is None:
        return ()
    cleaned = {str(value).strip() for value in values if str(value).strip()}
    return tuple(sort_once(cleaned, source = 'gabion.analysis.dataflow_indexed_file_scan._sorted_text.site_1'))

def _normalize_cache_config(value: JSONValue) -> JSONValue:
    if type(value) is dict:
        mapping = cast(dict[object, JSONValue], value)
        normalized = {
            str(key): _normalize_cache_config(mapping[key])
            for key in sort_once(mapping, source="_normalize_cache_config.mapping")
        }
        return cast(JSONValue, normalized)
    if type(value) is list:
        return cast(JSONValue, [_normalize_cache_config(item) for item in value])
    return value

def _canonical_stage_cache_detail(detail: Hashable) -> str:
    structural_detail = structural_key_atom(
        detail,
        source="gabion.analysis.dataflow_indexed_file_scan._canonical_stage_cache_detail",
    )
    canonical_json = structural_key_json(structural_detail)
    return json.dumps(canonical_json, sort_keys=False, separators=(",", ":"))

def _build_stage_cache_identity_spec(
    *,
    stage: Literal["parse", "index", "projection"],
    cache_context: _CacheSemanticContext,
    config_subset: Mapping[str, JSONValue],
) -> _StageCacheIdentitySpec:
    normalized_config = _normalize_cache_config(cast(JSONValue, config_subset))
    return _StageCacheIdentitySpec(
        stage=stage,
        forest_spec_id=str(cache_context.forest_spec_id or ""),
        fingerprint_seed_revision=fingerprint_stage_cache_identity(cache_context.fingerprint_seed_revision),
        normalized_config=normalized_config,
    )

def _canonical_stage_cache_identity(spec: _StageCacheIdentitySpec) -> str:
    payload: dict[str, JSONValue] = {
        "stage": spec.stage,
        "forest_spec_id": spec.forest_spec_id,
        "fingerprint_seed_revision": spec.fingerprint_seed_revision,
        "config_subset": spec.normalized_config,
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"aspf:sha1:{digest}"

def _canonical_cache_identity(
    *,
    stage: Literal["parse", "index", "projection"],
    cache_context: _CacheSemanticContext,
    config_subset: Mapping[str, JSONValue],
) -> _CacheIdentity:
    spec = _build_stage_cache_identity_spec(
        stage=stage,
        cache_context=cache_context,
        config_subset=config_subset,
    )
    canonical = _CacheIdentity.from_boundary(_canonical_stage_cache_identity(spec))
    if canonical is None:
        never("failed to construct canonical cache identity", stage=stage)  # pragma: no cover - invariant sink
    return canonical

def _cache_identity_aliases(identity: str) -> tuple[str, ...]:
    canonical = _CacheIdentity.from_boundary(identity)
    if canonical is None:
        return ("",)
    return (canonical.value,)

def _resume_variant_for_identity(
    variants: Mapping[str, JSONObject],
    expected_identity: _CacheIdentity,
):
    direct = variants.get(expected_identity.value)
    if direct is not None:
        return direct
    return None

def _parse_stage_cache_key(
    *,
    stage: _ParseModuleStage,
    cache_context: _CacheSemanticContext,
    config_subset: Mapping[str, JSONValue],
    detail: Hashable,
) -> NodeId:
    identity = _canonical_cache_identity(
        stage="parse",
        cache_context=cache_context,
        config_subset=config_subset,
    )
    return NodeId(
        kind="ParseStageCacheIdentity",
        key=(
            stage.value,
            identity.value,
            _canonical_stage_cache_detail(detail),
        ),
    )

def _index_stage_cache_identity(
    *,
    cache_context: _CacheSemanticContext,
    config_subset: Mapping[str, JSONValue],
) -> _CacheIdentity:
    return _canonical_cache_identity(
        stage="index",
        cache_context=cache_context,
        config_subset=config_subset,
    )

def _projection_stage_cache_identity(
    *,
    cache_context: _CacheSemanticContext,
    config_subset: Mapping[str, JSONValue],
) -> _CacheIdentity:
    return _canonical_cache_identity(
        stage="projection",
        cache_context=cache_context,
        config_subset=config_subset,
    )

def _stage_cache_key_aliases(key: Hashable) -> tuple[Hashable, ...]:
    return _stage_cache_key_aliases_impl(
        key,
        cache_identity_aliases_fn=_cache_identity_aliases,
        cache_identity_prefix=_CACHE_IDENTITY_PREFIX,
        cache_identity_digest_hex=_CACHE_IDENTITY_DIGEST_HEX,
        node_id_type=NodeId,
    )

def _get_stage_cache_bucket(
    analysis_index: AnalysisIndex,
    *,
    scoped_cache_key: Hashable,
) -> dict[Path, object]:
    stage_cache_by_key = analysis_index.stage_cache_by_key
    bucket = stage_cache_by_key.get(scoped_cache_key)
    if bucket is not None:
        return bucket
    for candidate_key in _stage_cache_key_aliases(scoped_cache_key):
        check_deadline()
        if candidate_key == scoped_cache_key:
            continue
        legacy_bucket = stage_cache_by_key.get(candidate_key)
        if legacy_bucket is not None:
            stage_cache_by_key[scoped_cache_key] = legacy_bucket
            return legacy_bucket
    return stage_cache_by_key.setdefault(scoped_cache_key, {})

def _parse_module_source(path: Path) -> ast.Module:
    return ast.parse(path.read_text())

def _build_module_artifacts(
    paths: list[Path],
    *,
    specs: tuple[_ModuleArtifactSpec[object, object], ...],
    parse_failure_witnesses: list[JSONObject],
    parse_module: Callable[[Path], ast.Module] = _parse_module_source,
) -> tuple[object, ...]:
    from gabion.analysis.indexed_scan.state.module_artifacts import (
        BuildModuleArtifactsDeps as _BuildModuleArtifactsDeps)
    from gabion.analysis.indexed_scan.state.module_artifacts import (
        build_module_artifacts as _build_module_artifacts_impl)

    return cast(
        tuple[object, ...],
        _build_module_artifacts_impl(
            paths,
            specs=cast(tuple[object, ...], specs),
            parse_failure_witnesses=cast(list[object], parse_failure_witnesses),
            parse_module=parse_module,
            deps=_BuildModuleArtifactsDeps(
                check_deadline_fn=check_deadline,
                parse_module_error_types=cast(
                    tuple[type[BaseException], ...],
                    _PARSE_MODULE_ERROR_TYPES,
                ),
                record_parse_failure_witness_fn=_record_parse_failure_witness,
            ),
        ),
    )

def _build_analysis_index(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators=None,
    parse_failure_witnesses: list[JSONObject],
    resume_payload=None,
    on_progress=None,
    accumulate_function_index_for_tree_fn=None,
    forest_spec_id=None,
    fingerprint_seed_revision=None,
    decision_ignore_params=None,
    decision_require_tiers: bool = False,
) -> AnalysisIndex:
    return cast(
        AnalysisIndex,
        _build_analysis_index_impl(
            paths,
            project_root=project_root,
            ignore_params=ignore_params,
            strictness=strictness,
            external_filter=external_filter,
            transparent_decorators=transparent_decorators,
            parse_failure_witnesses=parse_failure_witnesses,
            resume_payload=resume_payload,
            on_progress=on_progress,
            accumulate_function_index_for_tree_fn=accumulate_function_index_for_tree_fn,
            forest_spec_id=forest_spec_id,
            fingerprint_seed_revision=fingerprint_seed_revision,
            decision_ignore_params=decision_ignore_params,
            decision_require_tiers=decision_require_tiers,
            deps=_AnalysisIndexBuildDeps(
                check_deadline_fn=check_deadline,
                accumulate_function_index_for_tree_default_fn=_accumulate_function_index_for_tree,
                sorted_text_fn=_sorted_text,
                cache_context_ctor=_CacheSemanticContext,
                index_stage_cache_identity_fn=_index_stage_cache_identity,
                projection_stage_cache_identity_fn=_projection_stage_cache_identity,
                iter_monotonic_paths_fn=_iter_monotonic_paths,
                load_analysis_index_resume_payload_fn=_load_analysis_index_resume_payload,
                function_index_acc_ctor=_FunctionIndexAccumulator,
                sort_once_fn=sort_once,
                profiling_payload_fn=_profiling_v1_payload,
                serialize_resume_payload_fn=_serialize_analysis_index_resume_payload,
                parse_module_source_fn=_parse_module_source,
                parse_module_error_types=cast(
                    tuple[type[BaseException], ...],
                    _PARSE_MODULE_ERROR_TYPES,
                ),
                record_parse_failure_witness_fn=_record_parse_failure_witness,
                parse_module_stage_function_index=_ParseModuleStage.FUNCTION_INDEX,
                parse_module_stage_symbol_table=_ParseModuleStage.SYMBOL_TABLE,
                parse_module_stage_class_index=_ParseModuleStage.CLASS_INDEX,
                accumulate_symbol_table_for_tree_fn=_accumulate_symbol_table_for_tree,
                accumulate_class_index_for_tree_fn=_accumulate_class_index_for_tree,
                timeout_exceeded_type=TimeoutExceeded,
                analysis_index_ctor=AnalysisIndex,
                progress_emit_min_interval_seconds=_PROGRESS_EMIT_MIN_INTERVAL_SECONDS,
            ),
        ),
    )

def _run_indexed_pass(
    paths: list[Path],
    *,
    project_root: OptionalProjectRoot,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators: OptionalDecorators = None,
    parse_failure_witnesses: OptionalParseFailures = None,
    analysis_index: OptionalAnalysisIndex = None,
    spec: _IndexedPassSpec[_IndexedPassResult],
    build_index: Callable[..., AnalysisIndex] = _build_analysis_index,
) -> _IndexedPassResult:
    check_deadline()
    sink = _parse_failure_sink(parse_failure_witnesses)
    index = analysis_index
    if index is None:
        index = build_index(
            paths,
            project_root=project_root,
            ignore_params=ignore_params,
            strictness=strictness,
            external_filter=external_filter,
            transparent_decorators=transparent_decorators,
            parse_failure_witnesses=sink,
        )
    context = _IndexedPassContext(
        paths=paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=sink,
        analysis_index=index,
    )
    return spec.run(context)

def _analysis_index_module_trees(
    analysis_index: AnalysisIndex,
    paths: list[Path],
    *,
    stage: _ParseModuleStage,
    parse_failure_witnesses: list[JSONObject],
):
    return cast(
        dict[Path, ast.Module | None],
        _analysis_index_module_trees_impl(
            analysis_index,
            paths,
            stage=stage,
            parse_failure_witnesses=parse_failure_witnesses,
            deps=_AnalysisIndexModuleTreesDeps(
                check_deadline_fn=check_deadline,
                parse_module_source_fn=cast(Callable[[Path], object], _parse_module_source),
                parse_module_error_types=_PARSE_MODULE_ERROR_TYPES,
                record_parse_failure_witness_fn=_record_parse_failure_witness,
            ),
        ),
    )

def _analysis_index_stage_cache(
    analysis_index: AnalysisIndex,
    paths: list[Path],
    *,
    spec: _StageCacheSpec[_StageCacheValue],
    parse_failure_witnesses: list[JSONObject],
    module_trees_fn = None,
):
    return cast(
        dict[Path, _StageCacheValue | None],
        _analysis_index_stage_cache_impl(
            analysis_index,
            paths,
            spec=spec,
            parse_failure_witnesses=parse_failure_witnesses,
            module_trees_fn=module_trees_fn,
            deps=_AnalysisIndexStageCacheDeps(
                check_deadline_fn=check_deadline,
                get_global_derivation_cache_fn=get_global_derivation_cache,
                analysis_index_module_trees_fn=_analysis_index_module_trees,
                get_stage_cache_bucket_fn=_get_stage_cache_bucket,
                path_dependency_payload_fn=_path_dependency_payload,
                analysis_index_stage_cache_op=_ANALYSIS_INDEX_STAGE_CACHE_OP,
            ),
        ),
    )

def _analysis_index_transitive_callers(
    analysis_index: AnalysisIndex,
    *,
    project_root,
) -> dict[str, set[str]]:
    check_deadline()
    if analysis_index.transitive_callers is not None:
        return analysis_index.transitive_callers
    callers_by_qual: dict[str, set[str]] = defaultdict(set)
    for edge in _analysis_index_resolved_call_edges(
        analysis_index,
        project_root=project_root,
        require_transparent=False,
    ):
        check_deadline()
        callers_by_qual[edge.callee.qual].add(edge.caller.qual)
    analysis_index.transitive_callers = _collect_transitive_callers(
        callers_by_qual,
        analysis_index.by_qual,
    )
    return analysis_index.transitive_callers

def _analysis_index_resolved_call_edges(
    analysis_index: AnalysisIndex,
    *,
    project_root,
    require_transparent: bool,
) -> tuple[_ResolvedCallEdge, ...]:
    check_deadline()
    if require_transparent:
        cached_edges = analysis_index.resolved_transparent_call_edges
    else:
        cached_edges = analysis_index.resolved_call_edges
    if cached_edges is not None:
        return cached_edges
    edges: list[_ResolvedCallEdge] = []
    for infos in analysis_index.by_name.values():
        check_deadline()
        for info in infos:
            check_deadline()
            for call in info.calls:
                check_deadline()
                if not call.is_test:
                    callee = _resolve_callee(
                        call.callee,
                        info,
                        analysis_index.by_name,
                        analysis_index.by_qual,
                        analysis_index.symbol_table,
                        project_root,
                        analysis_index.class_index,
                    )
                    if callee is not None and (not require_transparent or callee.transparent):
                        edges.append(_ResolvedCallEdge(caller=info, call=call, callee=callee))
    frozen_edges = tuple(edges)
    if require_transparent:
        analysis_index.resolved_transparent_call_edges = frozen_edges
    else:
        analysis_index.resolved_call_edges = frozen_edges
    return frozen_edges

def _analysis_index_resolved_call_edges_by_caller(
    analysis_index: AnalysisIndex,
    *,
    project_root,
    require_transparent: bool,
) -> dict[str, tuple[_ResolvedCallEdge, ...]]:
    check_deadline()
    if require_transparent and analysis_index.resolved_transparent_edges_by_caller is not None:
        return analysis_index.resolved_transparent_edges_by_caller
    grouped: dict[str, list[_ResolvedCallEdge]] = defaultdict(list)
    for edge in _analysis_index_resolved_call_edges(
        analysis_index,
        project_root=project_root,
        require_transparent=require_transparent,
    ):
        check_deadline()
        grouped[edge.caller.qual].append(edge)
    frozen_grouped = {qual: tuple(edges) for qual, edges in grouped.items()}
    if require_transparent:
        analysis_index.resolved_transparent_edges_by_caller = frozen_grouped
    return frozen_grouped

def _reduce_resolved_call_edges(
    analysis_index: AnalysisIndex,
    *,
    project_root,
    require_transparent: bool,
    spec: _ResolvedEdgeReducerSpec[_ResolvedEdgeAcc, _ResolvedEdgeOut],
) -> _ResolvedEdgeOut:
    check_deadline()
    acc = spec.init()
    for edge in _analysis_index_resolved_call_edges(
        analysis_index,
        project_root=project_root,
        require_transparent=require_transparent,
    ):
        check_deadline()
        spec.fold(acc, edge)
    return spec.finish(acc)

def _iter_resolved_edge_param_events(
    edge: _ResolvedCallEdge,
    *,
    strictness: str,
    include_variadics_in_low_star: bool,
) -> Iterator[_ResolvedEdgeParamEvent]:
    yield from _iter_resolved_edge_param_events_impl(
        edge=edge,
        strictness=strictness,
        include_variadics_in_low_star=include_variadics_in_low_star,
        check_deadline_fn=check_deadline,
        event_ctor=_ResolvedEdgeParamEvent,
    )

def _build_call_graph(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators = None,
    parse_failure_witnesses: list[JSONObject],
    analysis_index = None,
) -> tuple[dict[str, list[FunctionInfo]], dict[str, FunctionInfo], dict[str, set[str]]]:
    check_deadline()
    index = analysis_index
    if index is None:
        index = _build_analysis_index(
            list(paths),
            project_root=project_root,
            ignore_params=set(ignore_params),
            strictness=strictness,
            external_filter=external_filter,
            transparent_decorators=transparent_decorators,
            parse_failure_witnesses=list(parse_failure_witnesses),
        )
    transitive_callers = _analysis_index_transitive_callers(
        index,
        project_root=project_root,
    )
    return index.by_name, index.by_qual, transitive_callers

def _collect_call_ambiguities_indexed(
    context: _IndexedPassContext,
    *,
    resolve_callee_fn = None,
) -> list[CallAmbiguity]:
    ambiguities: list[CallAmbiguity] = []
    resolve_callee = _resolve_callee if resolve_callee_fn is None else resolve_callee_fn

    def _sink(
        caller: FunctionInfo,
        call,
        candidates: list[FunctionInfo],
        phase: str,
        callee_key: str,
    ) -> None:
        ordered = tuple(sort_once(candidates, key=lambda info: info.qual, source = 'gabion.analysis.dataflow_indexed_file_scan._sink.site_1'))
        ambiguities.append(
            CallAmbiguity(
                kind="local_resolution_ambiguous",
                caller=caller,
                call=call,
                callee_key=callee_key,
                candidates=ordered,
                phase=phase,
            )
        )

    for infos in context.analysis_index.by_name.values():
        check_deadline()
        for info in infos:
            check_deadline()
            for call in info.calls:
                check_deadline()
                if call.is_test:
                    continue
                resolve_callee(
                    call.callee,
                    info,
                    context.analysis_index.by_name,
                    context.analysis_index.by_qual,
                    context.analysis_index.symbol_table,
                    context.project_root,
                    context.analysis_index.class_index,
                    call=call,
                    ambiguity_sink=_sink,
                )
    return ambiguities

def _collect_call_ambiguities(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators = None,
    parse_failure_witnesses: list[JSONObject],
    analysis_index = None,
) -> list[CallAmbiguity]:
    check_deadline()
    return _run_indexed_pass(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
        spec=_IndexedPassSpec(
            pass_id="collect_call_ambiguities",
            run=_collect_call_ambiguities_indexed,
        ),
    )

def _dedupe_call_ambiguities(
    ambiguities: Iterable[CallAmbiguity],
) -> list[CallAmbiguity]:
    check_deadline()
    seen: set[tuple[object, ...]] = set()
    ordered: list[CallAmbiguity] = []
    for entry in ambiguities:
        check_deadline()
        span = entry.call.span if entry.call is not None else None
        candidate_keys = tuple(
            (candidate.path, candidate.qual) for candidate in entry.candidates
        )
        key = (
            entry.kind,
            entry.caller.path,
            entry.caller.qual,
            span,
            entry.callee_key,
            candidate_keys,
        )
        if key in seen:
            continue
        seen.add(key)
        ordered.append(entry)
    return ordered

def _emit_call_ambiguities(
    ambiguities: Iterable[CallAmbiguity],
    *,
    project_root,
    forest: Forest,
) -> list[JSONObject]:
    return _emit_call_ambiguities_impl(
        ambiguities,
        project_root=project_root,
        forest=forest,
        deps=_CallAmbiguitiesEmitDeps(
            check_deadline_fn=check_deadline,
            normalize_snapshot_path_fn=_normalize_snapshot_path,
            normalize_targets_fn=evidence_keys.normalize_targets,
            never_fn=never,
            call_candidate_target_site_fn=_call_candidate_target_site,
            add_interned_alt_fn=_add_interned_alt,
            make_ambiguity_set_key_fn=evidence_keys.make_ambiguity_set_key,
            normalize_key_fn=evidence_keys.normalize_key,
            make_partition_witness_key_fn=evidence_keys.make_partition_witness_key,
            key_identity_fn=evidence_keys.key_identity,
        ),
    )

def _lint_lines_from_call_ambiguities(entries: Iterable[JSONObject]) -> list[str]:
    from gabion.analysis.dataflow.engine.dataflow_lint_helpers import _lint_lines_from_call_ambiguities as _impl

    return _impl(entries)

def _forbid_adhoc_bundle_discovery(reason: str) -> None:
    if os.environ.get("GABION_FORBID_ADHOC_BUNDLES") == "1":
        raise AssertionError(
            f"Ad-hoc bundle discovery invoked while forest-only invariant active: {reason}"
        )

def _materialize_statement_suite_contains(
    *,
    forest: Forest,
    path_name: str,
    qual: str,
    statements: Sequence[ast.stmt],
    parent_suite: NodeId,
) -> None:
    _materialize_statement_suite_contains_impl(
        forest=forest,
        path_name=path_name,
        qual=qual,
        statements=statements,
        parent_suite=parent_suite,
        node_span_fn=_node_span,
        check_deadline_fn=check_deadline,
    )

def _materialize_structured_suite_sites_for_tree(
    *,
    forest: Forest,
    path: Path,
    tree: ast.Module,
    project_root,
) -> None:
    _materialize_structured_suite_sites_for_tree_impl(
        forest=forest,
        path=path,
        tree=tree,
        project_root=project_root,
        deps=_MaterializeStructuredSuiteSitesForTreeDeps(
            check_deadline_fn=check_deadline,
            parent_annotator_factory=ParentAnnotator,
            module_name_fn=_module_name,
            collect_functions_fn=_collect_functions,
            enclosing_scopes_fn=_enclosing_scopes,
            node_span_fn=_node_span,
            materialize_statement_suite_contains_fn=_materialize_statement_suite_contains,
        ),
    )

def _materialize_structured_suite_sites(
    *,
    forest: Forest,
    file_paths: list[Path],
    project_root,
    parse_failure_witnesses: list[JSONObject],
    analysis_index = None,
) -> None:
    _materialize_structured_suite_sites_impl(
        forest=forest,
        file_paths=file_paths,
        project_root=project_root,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
        deps=_MaterializeStructuredSuiteSitesDeps(
            check_deadline_fn=check_deadline,
            iter_monotonic_paths_fn=_iter_monotonic_paths,
            analysis_index_module_trees_fn=_analysis_index_module_trees,
            parse_module_tree_fn=_parse_module_tree,
            parse_module_stage_suite_containment=_ParseModuleStage.SUITE_CONTAINMENT,
            materialize_structured_suite_sites_for_tree_fn=_materialize_structured_suite_sites_for_tree,
        ),
    )

def _populate_bundle_forest(
    forest: Forest,
    *,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    file_paths: list[Path],
    project_root = None,
    include_all_sites: bool = True,
    ignore_params = None,
    strictness: str = "high",
    transparent_decorators = None,
    parse_failure_witnesses: list[JSONObject],
    analysis_index = None,
    on_progress = None,
) -> None:
    _populate_bundle_forest_impl_runtime(
        forest,
        groups_by_path=groups_by_path,
        file_paths=file_paths,
        project_root=project_root,
        include_all_sites=include_all_sites,
        ignore_params=ignore_params,
        strictness=strictness,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
        on_progress=on_progress,
        runtime_module=sys.modules[__name__],
    )

def _is_test_path(path: Path) -> bool:
    if "tests" in path.parts:
        return True
    return path.name.startswith("test_")

def _unused_params(use_map: dict[str, ParamUse]) -> tuple[set[str], set[str]]:
    check_deadline()
    unused: set[str] = set()
    unknown_key_carriers: set[str] = set()
    for name, info in use_map.items():
        check_deadline()
        if info.non_forward:
            continue
        if info.direct_forward:
            continue
        if info.unknown_key_carrier:
            unknown_key_carriers.add(name)
            continue
        unused.add(name)
    return unused, unknown_key_carriers

def _group_by_signature(use_map: dict[str, ParamUse]) -> list[set[str]]:
    check_deadline()
    sig_map: dict[tuple[tuple[str, str], ...], list[str]] = defaultdict(list)
    for name, info in use_map.items():
        check_deadline()
        if info.non_forward:
            continue
        sig = tuple(sort_once(info.direct_forward, source = 'gabion.analysis.dataflow_indexed_file_scan._group_by_signature.site_1'))
        # Empty forwarding signatures are usually just unused params; treating them as
        # bundles creates noisy Tier-3 violations and unstable fingerprint baselines.
        if not sig:
            continue
        sig_map[sig].append(name)
    groups = [set(names) for names in sig_map.values() if len(names) > 1]
    return groups

def _union_groups(groups: list[set[str]]) -> list[set[str]]:
    check_deadline()
    changed = True
    while changed:
        check_deadline()
        changed = False
        out = []
        while groups:
            check_deadline()
            base = groups.pop()
            merged = True
            while merged:
                check_deadline()
                merged = False
                for i, other in enumerate(groups):
                    check_deadline()
                    if base & other:
                        base |= other
                        groups.pop(i)
                        merged = True
                        changed = True
                        break
            out.append(base)
        groups = out
    return groups

def _propagate_groups(
    call_args: list[CallArgs],
    callee_groups: dict[str, list[set[str]]],
    callee_param_orders: dict[str, list[str]],
    strictness: str,
    opaque_callees = None,
) -> list[set[str]]:
    return cast(
        list[set[str]],
        _propagate_groups_impl(
            cast(list[object], call_args),
            callee_groups,
            callee_param_orders,
            strictness,
            opaque_callees=opaque_callees,
            deps=_PropagateGroupsDeps(check_deadline_fn=check_deadline),
        ),
    )

def _adapt_ingest_carrier_to_analysis_maps(ingest_carrier):
    return (
        dict(ingest_carrier.function_use),
        dict(ingest_carrier.function_calls),
        dict(ingest_carrier.function_param_orders),
        dict(ingest_carrier.function_param_spans),
        set(ingest_carrier.opaque_callees),
    )

def analyze_ingested_file(
    ingest_carrier,
    *,
    recursive: bool,
    config: AuditConfig,
    on_profile = None,
) -> tuple[
    dict[str, list[set[str]]],
    dict[str, dict[str, tuple[int, int, int, int]]],
    dict[str, list[list[JSONObject]]],
]:
    from gabion.analysis.indexed_scan.scanners.analyze_ingested_file import (
        AnalyzeIngestedFileDeps as _AnalyzeIngestedFileDeps)
    from gabion.analysis.indexed_scan.scanners.analyze_ingested_file import (
        analyze_ingested_file as _analyze_ingested_file_impl)

    return _analyze_ingested_file_impl(
        ingest_carrier,
        recursive=recursive,
        config=config,
        on_profile=on_profile,
        deps=_AnalyzeIngestedFileDeps(
            adapt_ingest_carrier_to_analysis_maps_fn=_adapt_ingest_carrier_to_analysis_maps,
            profiling_v1_payload_fn=_profiling_v1_payload,
            monotonic_ns_fn=time.monotonic_ns,
            group_by_signature_fn=_group_by_signature,
            callsite_evidence_for_bundle_fn=_callsite_evidence_for_bundle,
            propagate_groups_fn=_propagate_groups,
            union_groups_fn=_union_groups,
            check_deadline_fn=check_deadline,
        ),
    )

def _analyze_file_internal(
    path: Path,
    recursive: bool = True,
    *,
    config = None,
    resume_state = None,
    on_progress = None,
    on_profile = None,
    analyze_function_fn = None,
) -> tuple[
    dict[str, list[set[str]]],
    dict[str, dict[str, tuple[int, int, int, int]]],
    dict[str, list[list[JSONObject]]],
]:
    from gabion.analysis.indexed_scan.state.file_internal_analysis import (
        analyze_file_internal_from_runtime_module_defaults as _analyze_file_internal_impl)

    return cast(
        tuple[
            dict[str, list[set[str]]],
            dict[str, dict[str, tuple[int, int, int, int]]],
            dict[str, list[list[JSONObject]]],
        ],
        _analyze_file_internal_impl(
            path,
            recursive=recursive,
            config=config,
            resume_state=resume_state,
            on_progress=on_progress,
            on_profile=on_profile,
            analyze_function_fn=analyze_function_fn,
            runtime_module=sys.modules[__name__],
        ),
    )

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

def _is_broad_type(annot) -> bool:
    if annot is None:
        return True
    base = annot.replace("typing.", "")
    return base in {"Any", "object"}

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

def _accumulate_symbol_table_for_tree(
    table: SymbolTable,
    path: Path,
    tree: ast.Module,
    *,
    project_root,
) -> None:
    check_deadline()
    module = _module_name(path, project_root)
    table.internal_roots.add(module.split(".")[0])
    visitor = ImportVisitor(module, table)
    visitor.visit(tree)
    import_map = {
        local: fqn for (mod, local), fqn in table.imports.items() if mod == module
    }
    exports, export_map = _collect_module_exports(
        tree,
        module_name=module,
        import_map=import_map,
    )
    table.module_exports[module] = exports
    table.module_export_map[module] = export_map

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

def _accumulate_class_index_for_tree(
    class_index: dict[str, ClassInfo],
    path: Path,
    tree: ast.Module,
    *,
    project_root,
) -> None:
    from gabion.analysis.indexed_scan.scanners.class_index_accumulator import (
        AccumulateClassIndexForTreeDeps as _AccumulateClassIndexForTreeDeps)
    from gabion.analysis.indexed_scan.scanners.class_index_accumulator import (
        accumulate_class_index_for_tree as _accumulate_class_index_for_tree_impl)

    _accumulate_class_index_for_tree_impl(
        cast(dict[str, object], class_index),
        path,
        tree,
        project_root=project_root,
        deps=_AccumulateClassIndexForTreeDeps(
            check_deadline_fn=check_deadline,
            parent_annotator_ctor=ParentAnnotator,
            module_name_fn=_module_name,
            enclosing_class_scopes_fn=_enclosing_class_scopes,
            base_identifier_fn=_base_identifier,
            class_info_ctor=ClassInfo,
        ),
    )

@dataclass
class _FunctionIndexAccumulator:
    by_name: dict[str, list[FunctionInfo]] = field(
        default_factory=lambda: defaultdict(list)
    )
    by_qual: dict[str, FunctionInfo] = field(default_factory=dict)

def _accumulate_function_index_for_tree(
    acc: _FunctionIndexAccumulator,
    path: Path,
    tree: ast.Module,
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    transparent_decorators,
) -> None:
    from gabion.analysis.indexed_scan.state.function_index_accumulator import (
        accumulate_function_index_for_tree_from_runtime_module as _accumulate_function_index_for_tree_impl_runtime)

    _accumulate_function_index_for_tree_impl_runtime(
        acc,
        path,
        tree,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        transparent_decorators=transparent_decorators,
        runtime_module=sys.modules[__name__],
    )

def _synthetic_lambda_name(
    *,
    module: str,
    lexical_scope: Sequence[str],
    span: tuple[int, int, int, int],
) -> str:
    check_deadline()
    lexical = ".".join(lexical_scope) if lexical_scope else "<module>"
    stable_payload = f"{module}|{lexical}|{span[0]}:{span[1]}:{span[2]}:{span[3]}"
    digest = hashlib.sha1(stable_payload.encode("utf-8")).hexdigest()[:12]
    return f"<lambda:{digest}>"

def _collect_lambda_function_infos(
    tree: ast.AST,
    *,
    path: Path,
    module: str,
    parent_map: Mapping[ast.AST, ast.AST],
    ignore_params,
) -> list[FunctionInfo]:
    from gabion.analysis.indexed_scan.ast.lambda_bindings import (
        CollectLambdaFunctionInfosDeps as _CollectLambdaFunctionInfosDeps)
    from gabion.analysis.indexed_scan.ast.lambda_bindings import (
        collect_lambda_function_infos as _collect_lambda_function_infos_impl)

    return cast(
        list[FunctionInfo],
        _collect_lambda_function_infos_impl(
            tree,
            path=path,
            module=module,
            parent_map=parent_map,
            ignore_params=ignore_params,
            deps=_CollectLambdaFunctionInfosDeps(
                check_deadline_fn=check_deadline,
                node_span_fn=_node_span,
                enclosing_function_scopes_fn=_enclosing_function_scopes,
                enclosing_scopes_fn=_enclosing_scopes,
                enclosing_class_fn=_enclosing_class,
                synthetic_lambda_name_fn=_synthetic_lambda_name,
                function_info_ctor=FunctionInfo,
            ),
        ),
    )

def _collect_lambda_bindings_by_caller(
    tree: ast.AST,
    *,
    module: str,
    parent_map: dict[ast.AST, ast.AST],
    lambda_infos: Sequence[FunctionInfo],
) -> dict[str, dict[str, tuple[str, ...]]]:
    return _collect_lambda_bindings_by_caller_impl(
        tree,
        module=module,
        parent_map=parent_map,
        lambda_infos=cast(Sequence[object], lambda_infos),
        deps=_LambdaBindingsByCallerDeps(
            check_deadline_fn=check_deadline,
            require_not_none_fn=require_not_none,
            collect_closure_lambda_factories_fn=_collect_closure_lambda_factories,
            node_span_fn=_node_span,
            enclosing_scopes_fn=_enclosing_scopes,
            target_names_fn=_target_names,
            sort_once_fn=sort_once,
        ),
    )

def _collect_closure_lambda_factories(
    tree: ast.AST,
    *,
    module: str,
    parent_map: dict[ast.AST, ast.AST],
    lambda_qual_by_span: Mapping[tuple[int, int, int, int], str],
) -> dict[str, set[str]]:
    return _collect_closure_lambda_factories_impl(
        tree,
        module=module,
        parent_map=parent_map,
        lambda_qual_by_span=lambda_qual_by_span,
        deps=_ClosureLambdaFactoriesDeps(
            check_deadline_fn=check_deadline,
            node_span_fn=_node_span,
            target_names_fn=_target_names,
            enclosing_scopes_fn=_enclosing_scopes,
            function_key_fn=_function_key,
        ),
    )

def _direct_lambda_callee_by_call_span(
    tree: ast.AST,
    *,
    lambda_infos: Sequence[FunctionInfo],
) -> dict[tuple[int, int, int, int], str]:
    check_deadline()
    lambda_qual_by_span = {
        info.function_span: info.qual
        for info in lambda_infos
        if info.function_span is not None
    }
    mapping: dict[tuple[int, int, int, int], str] = {}
    for node in ast.walk(tree):
        check_deadline()
        if type(node) is ast.Call:
            call_node = cast(ast.Call, node)
            if type(call_node.func) is ast.Lambda:
                call_span = _node_span(call_node)
                lambda_span = _node_span(call_node.func)
                if call_span is not None and lambda_span is not None:
                    callee = lambda_qual_by_span.get(lambda_span)
                    if callee is not None:
                        mapping[call_span] = callee
    return mapping

def _materialize_direct_lambda_callees(
    call_args: Sequence[CallArgs],
    *,
    direct_lambda_callee_by_call_span: Mapping[tuple[int, int, int, int], str],
) -> list[CallArgs]:
    out: list[CallArgs] = []
    for call in call_args:
        check_deadline()
        if call.span is not None and call.span in direct_lambda_callee_by_call_span:
            out.append(replace(call, callee=direct_lambda_callee_by_call_span[call.span]))
            continue
        out.append(call)
    return out

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

def _resolve_callee(
    callee_key: str,
    caller: FunctionInfo,
    by_name: dict[str, list[FunctionInfo]],
    by_qual: dict[str, FunctionInfo],
    symbol_table = None,
    project_root = None,
    class_index = None,
    call = None,
    ambiguity_sink = None,
    local_lambda_bindings = None,
):
    return cast(
        FunctionInfo | None,
        _resolve_callee_impl(
            callee_key,
            caller,
            cast(dict[str, list[object]], by_name),
            cast(dict[str, object], by_qual),
            symbol_table=symbol_table,
            project_root=project_root,
            class_index=class_index,
            call=call,
            ambiguity_sink=ambiguity_sink,
            local_lambda_bindings=local_lambda_bindings,
            deps=_ResolveCalleeDeps(
                check_deadline_fn=check_deadline,
                callee_resolution_context_core_ctor=_CalleeResolutionContextCore,
                resolve_callee_with_effects_fn=_resolve_callee_with_effects_impl,
                collect_callee_resolution_effects_fn=_collect_callee_resolution_effects_impl,
                module_name_fn=_module_name,
            ),
        ),
    )

def _is_dynamic_dispatch_callee_key(callee_key: str) -> bool:
    """Classify obvious syntax-level dynamic-dispatch call shapes."""
    check_deadline()
    text = callee_key.strip()
    if not text:
        return False
    if text.startswith("getattr("):
        return True
    if "." not in text:
        return False
    base, _, _ = text.partition(".")
    base = base.strip()
    if not base or base in {"self", "cls"}:
        return False
    if any(token in base for token in ("(", "[", "{")):
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", base) is None:
        return True
    return False

@dataclass(frozen=True)
class _CalleeResolutionOutcome:
    status: str
    phase: str
    callee_key: str
    candidates: tuple[FunctionInfo, ...] = ()

def _dedupe_resolution_candidates(
    candidates: Iterable[FunctionInfo],
) -> tuple[FunctionInfo, ...]:
    deduped: dict[str, FunctionInfo] = {}
    for candidate in candidates:
        check_deadline()
        if _is_test_path(candidate.path):
            continue
        deduped[candidate.qual] = candidate
    return tuple(sort_once(deduped.values(), key=lambda info: info.qual, source = 'gabion.analysis.dataflow_indexed_file_scan._dedupe_resolution_candidates.site_1'))

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
    return cast(
        _CalleeResolutionOutcome,
        _resolve_callee_outcome_impl_runtime(
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
            runtime_module=sys.modules[__name__],
        ),
    )

def _infer_type_flow(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    strictness: str,
    external_filter: bool,
    transparent_decorators = None,
    max_sites_per_param: int = 3,
    parse_failure_witnesses: list[JSONObject],
    analysis_index = None,
):
    return _infer_type_flow_impl(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        max_sites_per_param=max_sites_per_param,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
        deps=_TypeFlowInferDeps(
            check_deadline_fn=check_deadline,
            analysis_pass_prerequisites_ctor=AnalysisPassPrerequisites,
            require_not_none_fn=require_not_none,
            analysis_index_resolved_call_edges_by_caller_fn=_analysis_index_resolved_call_edges_by_caller,
            caller_param_bindings_for_call_fn=_caller_param_bindings_for_call,
            function_key_fn=_function_key,
            normalize_snapshot_path_fn=_normalize_snapshot_path,
            is_test_path_fn=_is_test_path,
            is_broad_type_fn=_is_broad_type,
            sort_once_fn=sort_once,
        ),
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

def _constant_smells_from_details(
    details: Iterable[ConstantFlowDetail],
) -> list[str]:
    from gabion.analysis.dataflow.engine.dataflow_lint_helpers import _constant_smells_from_details as _impl

    return _impl(details)

def _deadness_witnesses_from_constant_details(
    details: Iterable[ConstantFlowDetail],
    *,
    project_root,
) -> list[JSONObject]:
    from gabion.analysis.dataflow.engine.dataflow_lint_helpers import _deadness_witnesses_from_constant_details as _impl

    return _impl(details, project_root=project_root)

def _analyze_unused_arg_flow_indexed(
    context: _IndexedPassContext,
) -> list[str]:
    return _analyze_unused_arg_flow_indexed_impl(
        context,
        analysis_index_resolved_call_edges_fn=_analysis_index_resolved_call_edges,
        check_deadline_fn=check_deadline,
        sort_once_fn=sort_once,
    )


_BUNDLE_MARKER = re.compile(r"dataflow-bundle:\s*(.*)")

def _iter_documented_bundles(path: Path) -> set[tuple[str, ...]]:
    """Return bundles documented via '# dataflow-bundle: a, b' markers."""
    check_deadline()
    _forbid_adhoc_bundle_discovery("_iter_documented_bundles")
    bundles: set[tuple[str, ...]] = set()
    try:
        text = path.read_text()
    except (OSError, UnicodeError):
        return bundles
    for line in text.splitlines():
        check_deadline()
        match = _BUNDLE_MARKER.search(line)
        if not match:
            continue
        payload = match.group(1)
        if not payload:
            continue
        parts = [p.strip() for p in re.split(r"[,\s]+", payload) if p.strip()]
        if len(parts) < 2:
            continue
        bundles.add(tuple(sort_once(parts, source = 'gabion.analysis.dataflow_indexed_file_scan._iter_documented_bundles.site_1')))
    return bundles

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
