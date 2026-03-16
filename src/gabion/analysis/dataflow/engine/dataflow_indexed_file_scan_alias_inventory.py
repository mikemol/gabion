# gabion:ambiguity_boundary_module
# gabion:grade_boundary kind=semantic_carrier_adapter name=dataflow_indexed_file_scan_alias_surface
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Iterable, Iterator, Literal, Mapping, Sequence


BOUNDARY_ADAPTER_LIFECYCLE: dict[str, object] = {
    "actor": "codex",
    "rationale": "WS-5 hard-cut completed; retain monolith alias surface while external importers migrate",
    "scope": "dataflow_indexed_file_scan.alias_surface",
    "start": "2026-03-05",
    "expiry": "WS-5 compatibility-shim retirement",
    "rollback_condition": "no external consumers require monolith path aliases",
    "evidence_links": ["docs/ws5_decomposition_ledger.md"],
}


@dataclass(frozen=True)
class AliasBindingSpec:
    source_name: str
    export_name: str


@dataclass(frozen=True)
class ModuleAliasSpec:
    module_path: str
    bindings: tuple[AliasBindingSpec, ...]


@dataclass(frozen=True)
class AliasGroupSpec:
    group_id: str
    label: str
    module_specs: tuple[ModuleAliasSpec, ...]


@dataclass(frozen=True)
class AliasSurfaceMaterialization:
    exports: dict[str, object]
    inventory: dict[str, object]
    telemetry: dict[str, object]


_LOCAL_SUPPORT_BINDINGS: tuple[tuple[str, object], ...] = (
    ("argparse", argparse),
    ("ast", ast),
    ("sys", sys),
    ("ExitStack", ExitStack),
    ("Path", Path),
    ("Iterable", Iterable),
    ("Iterator", Iterator),
    ("Literal", Literal),
    ("Mapping", Mapping),
    ("Sequence", Sequence),
)

_LOCAL_TYPE_ALIAS_NAMES: tuple[str, ...] = (
    "FunctionNode",
    "OptionalIgnoredParams",
    "ParamAnnotationMap",
    "ReturnAliasMap",
    "OptionalReturnAliasMap",
    "OptionalClassName",
    "Span4",
    "OptionalSpan4",
    "OptionalString",
    "OptionalFloat",
    "OptionalPath",
    "OptionalStringSet",
    "OptionalPrimeRegistry",
    "OptionalTypeConstructorRegistry",
    "OptionalSynthRegistry",
    "OptionalJsonObject",
    "OptionalForestSpec",
    "OptionalDeprecatedExtractionArtifacts",
    "OptionalAstCall",
    "NodeIdOrNone",
    "ParseCacheValue",
    "ReportProjectionPhase",
)

ALIAS_GROUP_SPECS: tuple[AliasGroupSpec, ...] = (
    AliasGroupSpec(
        group_id="compatibility_support",
        label="Compatibility Support",
        module_specs=(
            ModuleAliasSpec(
                module_path="gabion.ingest.python_ingest",
                bindings=(
                    AliasBindingSpec(
                        source_name="ingest_python_file",
                        export_name="ingest_python_file",
                    ),
                    AliasBindingSpec(
                        source_name="iter_python_paths",
                        export_name="iter_python_paths",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.core.visitors",
                bindings=(
                    AliasBindingSpec(
                        source_name="ImportVisitor",
                        export_name="ImportVisitor",
                    ),
                    AliasBindingSpec(
                        source_name="ParentAnnotator",
                        export_name="ParentAnnotator",
                    ),
                    AliasBindingSpec(source_name="UseVisitor", export_name="UseVisitor"),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.foundation.json_types",
                bindings=(
                    AliasBindingSpec(source_name="JSONObject", export_name="JSONObject"),
                    AliasBindingSpec(source_name="JSONValue", export_name="JSONValue"),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.aspf.aspf",
                bindings=(
                    AliasBindingSpec(source_name="Alt", export_name="Alt"),
                    AliasBindingSpec(source_name="Forest", export_name="Forest"),
                    AliasBindingSpec(source_name="Node", export_name="Node"),
                    AliasBindingSpec(source_name="NodeId", export_name="NodeId"),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.semantics",
                bindings=(
                    AliasBindingSpec(
                        source_name="evidence_keys", export_name="evidence_keys"
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.core.type_fingerprints",
                bindings=(
                    AliasBindingSpec(
                        source_name="Fingerprint", export_name="Fingerprint"
                    ),
                    AliasBindingSpec(
                        source_name="FingerprintDimension",
                        export_name="FingerprintDimension",
                    ),
                    AliasBindingSpec(
                        source_name="PrimeRegistry", export_name="PrimeRegistry"
                    ),
                    AliasBindingSpec(
                        source_name="TypeConstructorRegistry",
                        export_name="TypeConstructorRegistry",
                    ),
                    AliasBindingSpec(
                        source_name="_collect_base_atoms",
                        export_name="_collect_base_atoms",
                    ),
                    AliasBindingSpec(
                        source_name="_collect_constructors",
                        export_name="_collect_constructors",
                    ),
                    AliasBindingSpec(
                        source_name="SynthRegistry", export_name="SynthRegistry"
                    ),
                    AliasBindingSpec(
                        source_name="build_synth_registry",
                        export_name="build_synth_registry",
                    ),
                    AliasBindingSpec(
                        source_name="build_fingerprint_registry",
                        export_name="build_fingerprint_registry",
                    ),
                    AliasBindingSpec(
                        source_name="build_synth_registry_from_payload",
                        export_name="build_synth_registry_from_payload",
                    ),
                    AliasBindingSpec(
                        source_name="bundle_fingerprint_dimensional",
                        export_name="bundle_fingerprint_dimensional",
                    ),
                    AliasBindingSpec(
                        source_name="format_fingerprint",
                        export_name="format_fingerprint",
                    ),
                    AliasBindingSpec(
                        source_name="fingerprint_carrier_soundness",
                        export_name="fingerprint_carrier_soundness",
                    ),
                    AliasBindingSpec(
                        source_name="fingerprint_identity_payload",
                        export_name="fingerprint_identity_payload",
                    ),
                    AliasBindingSpec(
                        source_name="synth_registry_payload",
                        export_name="synth_registry_payload",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.core.forest_spec",
                bindings=(
                    AliasBindingSpec(
                        source_name="ForestSpec", export_name="ForestSpec"
                    ),
                    AliasBindingSpec(
                        source_name="build_forest_spec",
                        export_name="build_forest_spec",
                    ),
                    AliasBindingSpec(
                        source_name="default_forest_spec",
                        export_name="default_forest_spec",
                    ),
                    AliasBindingSpec(
                        source_name="forest_spec_metadata",
                        export_name="forest_spec_metadata",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.foundation.timeout_context",
                bindings=(
                    AliasBindingSpec(source_name="Deadline", export_name="Deadline"),
                    AliasBindingSpec(source_name="GasMeter", export_name="GasMeter"),
                    AliasBindingSpec(
                        source_name="TimeoutExceeded",
                        export_name="TimeoutExceeded",
                    ),
                    AliasBindingSpec(
                        source_name="TimeoutTickCarrier",
                        export_name="TimeoutTickCarrier",
                    ),
                    AliasBindingSpec(
                        source_name="build_timeout_context_from_stack",
                        export_name="build_timeout_context_from_stack",
                    ),
                    AliasBindingSpec(
                        source_name="check_deadline", export_name="check_deadline"
                    ),
                    AliasBindingSpec(
                        source_name="deadline_loop_iter",
                        export_name="deadline_loop_iter",
                    ),
                    AliasBindingSpec(
                        source_name="deadline_clock_scope",
                        export_name="deadline_clock_scope",
                    ),
                    AliasBindingSpec(
                        source_name="deadline_scope", export_name="deadline_scope"
                    ),
                    AliasBindingSpec(
                        source_name="forest_scope", export_name="forest_scope"
                    ),
                    AliasBindingSpec(
                        source_name="reset_forest", export_name="reset_forest"
                    ),
                    AliasBindingSpec(source_name="set_forest", export_name="set_forest"),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.foundation.resume_codec",
                bindings=(
                    AliasBindingSpec(
                        source_name="allowed_path_lookup",
                        export_name="allowed_path_lookup",
                    ),
                    AliasBindingSpec(
                        source_name="int_str_pairs_from_sequence",
                        export_name="int_str_pairs_from_sequence",
                    ),
                    AliasBindingSpec(
                        source_name="iter_valid_key_entries",
                        export_name="iter_valid_key_entries",
                    ),
                    AliasBindingSpec(
                        source_name="load_resume_map", export_name="load_resume_map"
                    ),
                    AliasBindingSpec(
                        source_name="load_allowed_paths_from_sequence",
                        export_name="load_allowed_paths_from_sequence",
                    ),
                    AliasBindingSpec(
                        source_name="mapping_payload", export_name="mapping_payload"
                    ),
                    AliasBindingSpec(
                        source_name="mapping_sections", export_name="mapping_sections"
                    ),
                    AliasBindingSpec(
                        source_name="mapping_default_empty",
                        export_name="mapping_default_empty",
                    ),
                    AliasBindingSpec(
                        source_name="mapping_optional", export_name="mapping_optional"
                    ),
                    AliasBindingSpec(
                        source_name="payload_with_format",
                        export_name="payload_with_format",
                    ),
                    AliasBindingSpec(
                        source_name="payload_with_phase",
                        export_name="payload_with_phase",
                    ),
                    AliasBindingSpec(
                        source_name="sequence_optional",
                        export_name="sequence_optional",
                    ),
                    AliasBindingSpec(
                        source_name="str_list_from_sequence",
                        export_name="str_list_from_sequence",
                    ),
                    AliasBindingSpec(
                        source_name="str_map_from_mapping",
                        export_name="str_map_from_mapping",
                    ),
                    AliasBindingSpec(
                        source_name="str_pair_set_from_sequence",
                        export_name="str_pair_set_from_sequence",
                    ),
                    AliasBindingSpec(
                        source_name="str_set_from_sequence",
                        export_name="str_set_from_sequence",
                    ),
                    AliasBindingSpec(
                        source_name="str_tuple_from_sequence",
                        export_name="str_tuple_from_sequence",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.projection.projection_registry",
                bindings=(
                    AliasBindingSpec(
                        source_name="DEADLINE_OBLIGATIONS_SUMMARY_SPEC",
                        export_name="DEADLINE_OBLIGATIONS_SUMMARY_SPEC",
                    ),
                    AliasBindingSpec(
                        source_name="LINT_FINDINGS_SPEC",
                        export_name="LINT_FINDINGS_SPEC",
                    ),
                    AliasBindingSpec(
                        source_name="REPORT_SECTION_LINES_SPEC",
                        export_name="REPORT_SECTION_LINES_SPEC",
                    ),
                    AliasBindingSpec(
                        source_name="WL_REFINEMENT_SPEC",
                        export_name="WL_REFINEMENT_SPEC",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.core.deprecated_substrate",
                bindings=(
                    AliasBindingSpec(
                        source_name="DeprecatedExtractionArtifacts",
                        export_name="DeprecatedExtractionArtifacts",
                    ),
                    AliasBindingSpec(
                        source_name="DeprecatedFiber", export_name="DeprecatedFiber"
                    ),
                    AliasBindingSpec(
                        source_name="detect_report_section_extinction",
                        export_name="detect_report_section_extinction",
                    ),
                ),
            ),
        ),
    ),
    AliasGroupSpec(
        group_id="decision_support",
        label="Decision Support",
        module_specs=(
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_decision_surfaces",
                bindings=(
                    AliasBindingSpec(
                        source_name="compute_fingerprint_coherence",
                        export_name="_ds_compute_fingerprint_coherence",
                    ),
                    AliasBindingSpec(
                        source_name="compute_fingerprint_rewrite_plans",
                        export_name="_ds_compute_fingerprint_rewrite_plans",
                    ),
                    AliasBindingSpec(
                        source_name="lint_lines_from_bundle_evidence",
                        export_name="_ds_lint_lines_from_bundle_evidence",
                    ),
                    AliasBindingSpec(
                        source_name="lint_lines_from_constant_smells",
                        export_name="_ds_lint_lines_from_constant_smells",
                    ),
                    AliasBindingSpec(
                        source_name="lint_lines_from_type_evidence",
                        export_name="_ds_lint_lines_from_type_evidence",
                    ),
                    AliasBindingSpec(
                        source_name="lint_lines_from_unused_arg_smells",
                        export_name="_ds_lint_lines_from_unused_arg_smells",
                    ),
                    AliasBindingSpec(
                        source_name="parse_lint_location",
                        export_name="_ds_parse_lint_location",
                    ),
                    AliasBindingSpec(
                        source_name="summarize_coherence_witnesses",
                        export_name="_ds_summarize_coherence_witnesses",
                    ),
                    AliasBindingSpec(
                        source_name="summarize_deadness_witnesses",
                        export_name="_ds_summarize_deadness_witnesses",
                    ),
                    AliasBindingSpec(
                        source_name="summarize_rewrite_plans",
                        export_name="_ds_summarize_rewrite_plans",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_bundle_merge",
                bindings=(
                    AliasBindingSpec(
                        source_name="_merge_counts_by_knobs",
                        export_name="_merge_counts_by_knobs",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_callee_resolution_support",
                bindings=(
                    AliasBindingSpec(
                        source_name="_callee_key", export_name="_callee_key"
                    ),
                    AliasBindingSpec(
                        source_name="_resolve_class_candidates",
                        export_name="_resolve_class_candidates",
                    ),
                    AliasBindingSpec(
                        source_name="_resolve_method_in_hierarchy",
                        export_name="_resolve_method_in_hierarchy",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_fingerprint_helpers",
                bindings=(
                    AliasBindingSpec(
                        source_name="_build_synth_registry_payload",
                        export_name="_build_synth_registry_payload",
                    ),
                    AliasBindingSpec(
                        source_name="_collect_fingerprint_atom_keys",
                        export_name="_collect_fingerprint_atom_keys",
                    ),
                    AliasBindingSpec(
                        source_name="_compute_fingerprint_coherence",
                        export_name="_compute_fingerprint_coherence",
                    ),
                    AliasBindingSpec(
                        source_name="_compute_fingerprint_matches",
                        export_name="_compute_fingerprint_matches",
                    ),
                    AliasBindingSpec(
                        source_name="_compute_fingerprint_provenance",
                        export_name="_compute_fingerprint_provenance",
                    ),
                    AliasBindingSpec(
                        source_name="_compute_fingerprint_rewrite_plans",
                        export_name="_compute_fingerprint_rewrite_plans",
                    ),
                    AliasBindingSpec(
                        source_name="_compute_fingerprint_synth",
                        export_name="_compute_fingerprint_synth",
                    ),
                    AliasBindingSpec(
                        source_name="_compute_fingerprint_warnings",
                        export_name="_compute_fingerprint_warnings",
                    ),
                    AliasBindingSpec(
                        source_name="_find_provenance_entry_for_site",
                        export_name="_find_provenance_entry_for_site",
                    ),
                    AliasBindingSpec(
                        source_name="_fingerprint_soundness_issues",
                        export_name="_fingerprint_soundness_issues",
                    ),
                    AliasBindingSpec(
                        source_name="_glossary_match_strata",
                        export_name="_glossary_match_strata",
                    ),
                    AliasBindingSpec(
                        source_name="_summarize_fingerprint_provenance",
                        export_name="_summarize_fingerprint_provenance",
                    ),
                    AliasBindingSpec(
                        source_name="verify_rewrite_plan",
                        export_name="verify_rewrite_plan",
                    ),
                    AliasBindingSpec(
                        source_name="verify_rewrite_plans",
                        export_name="verify_rewrite_plans",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_adapter_contract",
                bindings=(
                    AliasBindingSpec(
                        source_name="parse_adapter_capabilities",
                        export_name="parse_adapter_capabilities",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_evidence_helpers",
                bindings=(
                    AliasBindingSpec(
                        source_name="_base_identifier",
                        export_name="_base_identifier",
                    ),
                    AliasBindingSpec(
                        source_name="_collect_module_exports",
                        export_name="_collect_module_exports",
                    ),
                    AliasBindingSpec(
                        source_name="_is_test_path", export_name="_is_test_path"
                    ),
                    AliasBindingSpec(
                        source_name="_module_name", export_name="_module_name"
                    ),
                    AliasBindingSpec(
                        source_name="_target_names", export_name="_target_names"
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_function_semantics",
                bindings=(
                    AliasBindingSpec(
                        source_name="_analyze_function",
                        export_name="_analyze_function",
                    ),
                    AliasBindingSpec(
                        source_name="_call_context", export_name="_call_context"
                    ),
                    AliasBindingSpec(
                        source_name="_collect_return_aliases",
                        export_name="_collect_return_aliases",
                    ),
                    AliasBindingSpec(
                        source_name="_const_repr", export_name="_const_repr"
                    ),
                    AliasBindingSpec(
                        source_name="_normalize_key_expr",
                        export_name="_normalize_key_expr",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_function_index_runtime_support",
                bindings=(
                    AliasBindingSpec(
                        source_name="_direct_lambda_callee_by_call_span",
                        export_name="_direct_lambda_callee_by_call_span",
                    ),
                    AliasBindingSpec(
                        source_name="_materialize_direct_lambda_callees",
                        export_name="_materialize_direct_lambda_callees",
                    ),
                    AliasBindingSpec(
                        source_name="_unused_params", export_name="_unused_params"
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_lambda_runtime_support",
                bindings=(
                    AliasBindingSpec(
                        source_name="_collect_lambda_bindings_by_caller",
                        export_name="_collect_lambda_bindings_by_caller",
                    ),
                    AliasBindingSpec(
                        source_name="_collect_lambda_function_infos",
                        export_name="_collect_lambda_function_infos",
                    ),
                    AliasBindingSpec(
                        source_name="_function_key", export_name="_function_key"
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_function_index_decision_support",
                bindings=(
                    AliasBindingSpec(
                        source_name="_collect_param_roots",
                        export_name="_collect_param_roots",
                    ),
                    AliasBindingSpec(
                        source_name="_contains_boolish",
                        export_name="_contains_boolish",
                    ),
                    AliasBindingSpec(
                        source_name="_decorator_name", export_name="_decorator_name"
                    ),
                    AliasBindingSpec(
                        source_name="_decision_surface_form_entries",
                        export_name="_decision_surface_form_entries",
                    ),
                    AliasBindingSpec(
                        source_name="_decision_surface_params",
                        export_name="_decision_surface_params",
                    ),
                    AliasBindingSpec(
                        source_name="_decision_surface_reason_map",
                        export_name="_decision_surface_reason_map",
                    ),
                    AliasBindingSpec(
                        source_name="_decorators_transparent",
                        export_name="_decorators_transparent",
                    ),
                    AliasBindingSpec(
                        source_name="_mark_param_roots",
                        export_name="_mark_param_roots",
                    ),
                    AliasBindingSpec(
                        source_name="_value_encoded_decision_params",
                        export_name="_value_encoded_decision_params",
                    ),
                    AliasBindingSpec(
                        source_name="is_decision_surface",
                        export_name="is_decision_surface",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_call_graph_algorithms",
                bindings=(
                    AliasBindingSpec(
                        source_name="_collect_recursive_functions",
                        export_name="_collect_recursive_functions",
                    ),
                    AliasBindingSpec(
                        source_name="_collect_recursive_nodes",
                        export_name="_collect_recursive_nodes",
                    ),
                    AliasBindingSpec(
                        source_name="_reachable_from_roots",
                        export_name="_reachable_from_roots",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_lint_helpers",
                bindings=(
                    AliasBindingSpec(
                        source_name="_constant_smells_from_details",
                        export_name="_constant_smells_from_details",
                    ),
                    AliasBindingSpec(
                        source_name="_deadness_witnesses_from_constant_details",
                        export_name="_deadness_witnesses_from_constant_details",
                    ),
                    AliasBindingSpec(
                        source_name="_deadline_lint_lines",
                        export_name="_deadline_lint_lines",
                    ),
                    AliasBindingSpec(
                        source_name="_exception_protocol_lint_lines",
                        export_name="_exception_protocol_lint_lines",
                    ),
                    AliasBindingSpec(
                        source_name="_internal_broad_type_lint_lines",
                        export_name="_internal_broad_type_lint_lines",
                    ),
                    AliasBindingSpec(
                        source_name="_is_broad_internal_type",
                        export_name="_is_broad_internal_type",
                    ),
                    AliasBindingSpec(
                        source_name="_lint_lines_from_bundle_evidence",
                        export_name="_lint_lines_from_bundle_evidence",
                    ),
                    AliasBindingSpec(
                        source_name="_lint_lines_from_constant_smells",
                        export_name="_lint_lines_from_constant_smells",
                    ),
                    AliasBindingSpec(
                        source_name="_lint_lines_from_type_evidence",
                        export_name="_lint_lines_from_type_evidence",
                    ),
                    AliasBindingSpec(
                        source_name="_lint_lines_from_unused_arg_smells",
                        export_name="_lint_lines_from_unused_arg_smells",
                    ),
                    AliasBindingSpec(
                        source_name="_normalize_type_name",
                        export_name="_normalize_type_name",
                    ),
                    AliasBindingSpec(
                        source_name="_parse_exception_path_id",
                        export_name="_parse_exception_path_id",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_local_class_hierarchy",
                bindings=(
                    AliasBindingSpec(
                        source_name="_collect_local_class_bases",
                        export_name="_collect_local_class_bases",
                    ),
                    AliasBindingSpec(
                        source_name="_resolve_local_method_in_hierarchy",
                        export_name="_resolve_local_method_in_hierarchy",
                    ),
                ),
            ),
        ),
    ),
    AliasGroupSpec(
        group_id="deadline_runtime",
        label="Deadline Runtime",
        module_specs=(
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_deadline_contracts",
                bindings=(
                    AliasBindingSpec(
                        source_name="_CalleeResolutionOutcome",
                        export_name="_CalleeResolutionOutcome",
                    ),
                    AliasBindingSpec(
                        source_name="_DeadlineFunctionFacts",
                        export_name="_DeadlineFunctionFacts",
                    ),
                    AliasBindingSpec(
                        source_name="_DeadlineLocalInfo",
                        export_name="_DeadlineLocalInfo",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_deadline_helpers",
                bindings=(
                    AliasBindingSpec(
                        source_name="_is_deadline_param",
                        export_name="_is_deadline_param",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_deadline_helpers",
                bindings=(
                    AliasBindingSpec(
                        source_name="_DeadlineFunctionCollector",
                        export_name="_DeadlineFunctionCollector",
                    ),
                    AliasBindingSpec(
                        source_name="_DeadlineArgInfo",
                        export_name="_DeadlineArgInfo",
                    ),
                    AliasBindingSpec(
                        source_name="_bind_call_args", export_name="_bind_call_args"
                    ),
                    AliasBindingSpec(
                        source_name="_classify_deadline_expr",
                        export_name="_classify_deadline_expr",
                    ),
                    AliasBindingSpec(
                        source_name="_collect_call_edges",
                        export_name="_collect_call_edges",
                    ),
                    AliasBindingSpec(
                        source_name="_collect_call_edges_from_forest",
                        export_name="_collect_call_edges_from_forest",
                    ),
                    AliasBindingSpec(
                        source_name="_collect_call_nodes_by_path",
                        export_name="_collect_call_nodes_by_path",
                    ),
                    AliasBindingSpec(
                        source_name="_collect_deadline_function_facts",
                        export_name="_collect_deadline_function_facts",
                    ),
                    AliasBindingSpec(
                        source_name="_collect_deadline_local_info",
                        export_name="_collect_deadline_local_info",
                    ),
                    AliasBindingSpec(
                        source_name="_deadline_arg_info_map",
                        export_name="_deadline_arg_info_map",
                    ),
                    AliasBindingSpec(
                        source_name="_deadline_loop_forwarded_params",
                        export_name="_deadline_loop_forwarded_params",
                    ),
                    AliasBindingSpec(
                        source_name="_fallback_deadline_arg_info",
                        export_name="_fallback_deadline_arg_info",
                    ),
                    AliasBindingSpec(
                        source_name="_is_dynamic_dispatch_callee_key",
                        export_name="_is_dynamic_dispatch_callee_key",
                    ),
                    AliasBindingSpec(
                        source_name="_materialize_call_candidates",
                        export_name="_materialize_call_candidates",
                    ),
                    AliasBindingSpec(
                        source_name="_resolve_callee", export_name="_resolve_callee"
                    ),
                    AliasBindingSpec(
                        source_name="_resolve_callee_outcome",
                        export_name="_resolve_callee_outcome",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.indexed_scan.deadline.deadline_runtime",
                bindings=(
                    AliasBindingSpec(
                        source_name="is_deadline_origin_call",
                        export_name="_is_deadline_origin_call",
                    ),
                ),
            ),
        ),
    ),
    AliasGroupSpec(
        group_id="resume_runtime",
        label="Resume Runtime",
        module_specs=(
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_resume_paths",
                bindings=(
                    AliasBindingSpec(
                        source_name="normalize_snapshot_path",
                        export_name="_normalize_snapshot_path",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_resume_serialization",
                bindings=(
                    AliasBindingSpec(
                        source_name="_CACHE_IDENTITY_DIGEST_HEX",
                        export_name="_CACHE_IDENTITY_DIGEST_HEX",
                    ),
                    AliasBindingSpec(
                        source_name="_CACHE_IDENTITY_PREFIX",
                        export_name="_CACHE_IDENTITY_PREFIX",
                    ),
                    AliasBindingSpec(
                        source_name="_CacheIdentity", export_name="_CacheIdentity"
                    ),
                    AliasBindingSpec(
                        source_name="_build_analysis_collection_resume_payload",
                        export_name="_build_analysis_collection_resume_payload",
                    ),
                    AliasBindingSpec(
                        source_name="_deserialize_function_info_for_resume",
                        export_name="_deserialize_function_info_for_resume",
                    ),
                    AliasBindingSpec(
                        source_name="_deserialize_invariants_for_resume",
                        export_name="_deserialize_invariants_for_resume",
                    ),
                    AliasBindingSpec(
                        source_name="_deserialize_symbol_table_for_resume",
                        export_name="_deserialize_symbol_table_for_resume",
                    ),
                    AliasBindingSpec(
                        source_name="_invariant_confidence",
                        export_name="_invariant_confidence",
                    ),
                    AliasBindingSpec(
                        source_name="_invariant_digest",
                        export_name="_invariant_digest",
                    ),
                    AliasBindingSpec(
                        source_name="_load_analysis_collection_resume_payload",
                        export_name="_load_analysis_collection_resume_payload",
                    ),
                    AliasBindingSpec(
                        source_name="_load_analysis_index_resume_payload",
                        export_name="_load_analysis_index_resume_payload",
                    ),
                    AliasBindingSpec(
                        source_name="_load_file_scan_resume_state",
                        export_name="_load_file_scan_resume_state",
                    ),
                    AliasBindingSpec(
                        source_name="_serialize_analysis_index_resume_payload",
                        export_name="_serialize_analysis_index_resume_payload",
                    ),
                    AliasBindingSpec(
                        source_name="_serialize_file_scan_resume_state",
                        export_name="_serialize_file_scan_resume_state",
                    ),
                    AliasBindingSpec(
                        source_name="_normalize_invariant_proposition",
                        export_name="_normalize_invariant_proposition",
                    ),
                ),
            ),
        ),
    ),
    AliasGroupSpec(
        group_id="post_phase_analysis",
        label="Post Phase Analysis",
        module_specs=(
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_post_phase_analyses",
                bindings=(
                    AliasBindingSpec(
                        source_name="_annotation_exception_candidates",
                        export_name="_annotation_exception_candidates",
                    ),
                    AliasBindingSpec(
                        source_name="_keyword_links_literal",
                        export_name="_keyword_links_literal",
                    ),
                    AliasBindingSpec(
                        source_name="_keyword_string_literal",
                        export_name="_keyword_string_literal",
                    ),
                    AliasBindingSpec(
                        source_name="_refine_exception_name_from_annotations",
                        export_name="_refine_exception_name_from_annotations",
                    ),
                    AliasBindingSpec(
                        source_name="_build_property_hook_callable_index",
                        export_name="_build_property_hook_callable_index",
                    ),
                    AliasBindingSpec(
                        source_name="_callsite_evidence_for_bundle",
                        export_name="_callsite_evidence_for_bundle",
                    ),
                    AliasBindingSpec(
                        source_name="_collect_config_bundles",
                        export_name="_collect_config_bundles",
                    ),
                    AliasBindingSpec(
                        source_name="_collect_constant_flow_details",
                        export_name="_collect_constant_flow_details",
                    ),
                    AliasBindingSpec(
                        source_name="_collect_dataclass_registry",
                        export_name="_collect_dataclass_registry",
                    ),
                    AliasBindingSpec(
                        source_name="_branch_reachability_under_env",
                        export_name="_branch_reachability_under_env",
                    ),
                    AliasBindingSpec(
                        source_name="_collect_exception_obligations",
                        export_name="_collect_exception_obligations",
                    ),
                    AliasBindingSpec(
                        source_name="_collect_handledness_witnesses",
                        export_name="_collect_handledness_witnesses",
                    ),
                    AliasBindingSpec(
                        source_name="_collect_invariant_propositions",
                        export_name="_collect_invariant_propositions",
                    ),
                    AliasBindingSpec(
                        source_name="_collect_never_invariants",
                        export_name="_collect_never_invariants",
                    ),
                    AliasBindingSpec(
                        source_name="_combine_type_hints",
                        export_name="_combine_type_hints",
                    ),
                    AliasBindingSpec(
                        source_name="_eval_bool_expr",
                        export_name="_eval_bool_expr",
                    ),
                    AliasBindingSpec(
                        source_name="_eval_value_expr",
                        export_name="_eval_value_expr",
                    ),
                    AliasBindingSpec(
                        source_name="_compute_knob_param_names",
                        export_name="_compute_knob_param_names",
                    ),
                    AliasBindingSpec(
                        source_name="_format_call_site",
                        export_name="_format_call_site",
                    ),
                    AliasBindingSpec(
                        source_name="_format_type_flow_site",
                        export_name="_format_type_flow_site",
                    ),
                    AliasBindingSpec(
                        source_name="_iter_config_fields",
                        export_name="_iter_config_fields",
                    ),
                    AliasBindingSpec(
                        source_name="_iter_dataclass_call_bundles",
                        export_name="_iter_dataclass_call_bundles",
                    ),
                    AliasBindingSpec(
                        source_name="_names_in_expr", export_name="_names_in_expr"
                    ),
                    AliasBindingSpec(
                        source_name="_node_in_block", export_name="_node_in_block"
                    ),
                    AliasBindingSpec(
                        source_name="_param_annotations_by_path",
                        export_name="_param_annotations_by_path",
                    ),
                    AliasBindingSpec(
                        source_name="_parse_module_source",
                        export_name="_parse_module_source",
                    ),
                    AliasBindingSpec(
                        source_name="_split_top_level", export_name="_split_top_level"
                    ),
                    AliasBindingSpec(
                        source_name="_StageCacheSpec", export_name="_StageCacheSpec"
                    ),
                    AliasBindingSpec(
                        source_name="_type_from_const_repr",
                        export_name="_type_from_const_repr",
                    ),
                    AliasBindingSpec(
                        source_name="analyze_decision_surfaces_repo",
                        export_name="analyze_decision_surfaces_repo",
                    ),
                    AliasBindingSpec(
                        source_name="analyze_constant_flow_repo",
                        export_name="analyze_constant_flow_repo",
                    ),
                    AliasBindingSpec(
                        source_name="analyze_deadness_flow_repo",
                        export_name="analyze_deadness_flow_repo",
                    ),
                    AliasBindingSpec(
                        source_name="analyze_unused_arg_flow_repo",
                        export_name="analyze_unused_arg_flow_repo",
                    ),
                    AliasBindingSpec(
                        source_name="analyze_value_encoded_decisions_repo",
                        export_name="analyze_value_encoded_decisions_repo",
                    ),
                    AliasBindingSpec(
                        source_name="generate_property_hook_manifest",
                        export_name="generate_property_hook_manifest",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_documented_bundles",
                bindings=(
                    AliasBindingSpec(
                        source_name="_iter_documented_bundles",
                        export_name="_iter_documented_bundles",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_ingest_helpers",
                bindings=(
                    AliasBindingSpec(
                        source_name="_collect_functions",
                        export_name="_collect_functions",
                    ),
                    AliasBindingSpec(
                        source_name="_iter_paths", export_name="_iter_paths"
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_ingested_analysis_support",
                bindings=(
                    AliasBindingSpec(
                        source_name="_group_by_signature",
                        export_name="_group_by_signature",
                    ),
                    AliasBindingSpec(
                        source_name="_propagate_groups",
                        export_name="_propagate_groups",
                    ),
                    AliasBindingSpec(
                        source_name="_union_groups", export_name="_union_groups"
                    ),
                    AliasBindingSpec(
                        source_name="analyze_ingested_file",
                        export_name="analyze_ingested_file",
                    ),
                ),
            ),
        ),
    ),
    AliasGroupSpec(
        group_id="analysis_index",
        label="Analysis Index",
        module_specs=(
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_analysis_index",
                bindings=(
                    AliasBindingSpec(
                        source_name="_FILE_SCAN_PROGRESS_EMIT_INTERVAL",
                        export_name="_FILE_SCAN_PROGRESS_EMIT_INTERVAL",
                    ),
                    AliasBindingSpec(
                        source_name="_PROGRESS_EMIT_MIN_INTERVAL_SECONDS",
                        export_name="_PROGRESS_EMIT_MIN_INTERVAL_SECONDS",
                    ),
                    AliasBindingSpec(
                        source_name="_EMPTY_CACHE_SEMANTIC_CONTEXT",
                        export_name="_EMPTY_CACHE_SEMANTIC_CONTEXT",
                    ),
                    AliasBindingSpec(
                        source_name="_IndexedPassContext",
                        export_name="_IndexedPassContext",
                    ),
                    AliasBindingSpec(
                        source_name="_IndexedPassSpec",
                        export_name="_IndexedPassSpec",
                    ),
                    AliasBindingSpec(
                        source_name="_analysis_index_module_trees",
                        export_name="_analysis_index_module_trees",
                    ),
                    AliasBindingSpec(
                        source_name="_analysis_index_resolved_call_edges",
                        export_name="_analysis_index_resolved_call_edges",
                    ),
                    AliasBindingSpec(
                        source_name="_analysis_index_resolved_call_edges_by_caller",
                        export_name="_analysis_index_resolved_call_edges_by_caller",
                    ),
                    AliasBindingSpec(
                        source_name="_analysis_index_stage_cache",
                        export_name="_analysis_index_stage_cache",
                    ),
                    AliasBindingSpec(
                        source_name="_analysis_index_transitive_callers",
                        export_name="_analysis_index_transitive_callers",
                    ),
                    AliasBindingSpec(source_name="analyze_file", export_name="analyze_file"),
                    AliasBindingSpec(
                        source_name="_analyze_file_internal",
                        export_name="_analyze_file_internal",
                    ),
                    AliasBindingSpec(
                        source_name="_accumulate_function_index_for_tree",
                        export_name="_accumulate_function_index_for_tree",
                    ),
                    AliasBindingSpec(
                        source_name="_build_analysis_index",
                        export_name="_build_analysis_index",
                    ),
                    AliasBindingSpec(
                        source_name="_build_function_index",
                        export_name="_build_function_index",
                    ),
                    AliasBindingSpec(
                        source_name="_build_symbol_table",
                        export_name="_build_symbol_table",
                    ),
                    AliasBindingSpec(
                        source_name="_build_call_graph",
                        export_name="_build_call_graph",
                    ),
                    AliasBindingSpec(
                        source_name="_iter_monotonic_paths",
                        export_name="_iter_monotonic_paths",
                    ),
                    AliasBindingSpec(
                        source_name="_iter_resolved_edge_param_events",
                        export_name="_iter_resolved_edge_param_events",
                    ),
                    AliasBindingSpec(
                        source_name="_parse_stage_cache_key",
                        export_name="_parse_stage_cache_key",
                    ),
                    AliasBindingSpec(
                        source_name="_reduce_resolved_call_edges",
                        export_name="_reduce_resolved_call_edges",
                    ),
                    AliasBindingSpec(
                        source_name="_run_indexed_pass",
                        export_name="_run_indexed_pass",
                    ),
                    AliasBindingSpec(
                        source_name="_phase_work_progress",
                        export_name="_phase_work_progress",
                    ),
                    AliasBindingSpec(
                        source_name="_profiling_v1_payload",
                        export_name="_profiling_v1_payload",
                    ),
                    AliasBindingSpec(
                        source_name="_sorted_text", export_name="_sorted_text"
                    ),
                    AliasBindingSpec(
                        source_name="_stage_cache_key_aliases",
                        export_name="_stage_cache_key_aliases",
                    ),
                ),
            ),
        ),
    ),
    AliasGroupSpec(
        group_id="projection_materialization",
        label="Projection Materialization",
        module_specs=(
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_projection_materialization",
                bindings=(
                    AliasBindingSpec(
                        source_name="CallAmbiguity", export_name="CallAmbiguity"
                    ),
                    AliasBindingSpec(
                        source_name="_ambiguity_suite_relation",
                        export_name="_ambiguity_suite_relation",
                    ),
                    AliasBindingSpec(
                        source_name="_ambiguity_suite_row_to_suite",
                        export_name="_ambiguity_suite_row_to_suite",
                    ),
                    AliasBindingSpec(
                        source_name="_ambiguity_virtual_count_gt_1",
                        export_name="_ambiguity_virtual_count_gt_1",
                    ),
                    AliasBindingSpec(
                        source_name="_collect_call_ambiguities",
                        export_name="_collect_call_ambiguities",
                    ),
                    AliasBindingSpec(
                        source_name="_collect_call_ambiguities_indexed",
                        export_name="_collect_call_ambiguities_indexed",
                    ),
                    AliasBindingSpec(
                        source_name="_dedupe_call_ambiguities",
                        export_name="_dedupe_call_ambiguities",
                    ),
                    AliasBindingSpec(
                        source_name="_emit_call_ambiguities",
                        export_name="_emit_call_ambiguities",
                    ),
                    AliasBindingSpec(
                        source_name="_format_span_fields",
                        export_name="_format_span_fields",
                    ),
                    AliasBindingSpec(
                        source_name="_lint_lines_from_call_ambiguities",
                        export_name="_lint_lines_from_call_ambiguities",
                    ),
                    AliasBindingSpec(
                        source_name="_materialize_ambiguity_suite_agg_spec",
                        export_name="_materialize_ambiguity_suite_agg_spec",
                    ),
                    AliasBindingSpec(
                        source_name="_materialize_ambiguity_virtual_set_spec",
                        export_name="_materialize_ambiguity_virtual_set_spec",
                    ),
                    AliasBindingSpec(
                        source_name="_materialize_projection_spec_rows",
                        export_name="_materialize_projection_spec_rows",
                    ),
                    AliasBindingSpec(
                        source_name="_materialize_suite_order_spec",
                        export_name="_materialize_suite_order_spec",
                    ),
                    AliasBindingSpec(
                        source_name="_populate_bundle_forest",
                        export_name="_populate_bundle_forest",
                    ),
                    AliasBindingSpec(
                        source_name="_spec_row_span", export_name="_spec_row_span"
                    ),
                    AliasBindingSpec(
                        source_name="_summarize_call_ambiguities",
                        export_name="_summarize_call_ambiguities",
                    ),
                    AliasBindingSpec(
                        source_name="_suite_order_relation",
                        export_name="_suite_order_relation",
                    ),
                    AliasBindingSpec(
                        source_name="_suite_order_row_to_site",
                        export_name="_suite_order_row_to_site",
                    ),
                ),
            ),
        ),
    ),
    AliasGroupSpec(
        group_id="reporting_io",
        label="Reporting And Projection IO",
        module_specs=(
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.io.dataflow_parse_helpers",
                bindings=(
                    AliasBindingSpec(
                        source_name="_parse_module_tree_optional",
                        export_name="_parse_module_tree",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_runtime_reporting",
                bindings=(
                    AliasBindingSpec(
                        source_name="_report_section_spec",
                        export_name="_report_section_spec",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.io.dataflow_projection_helpers",
                bindings=(
                    AliasBindingSpec(
                        source_name="_topologically_order_report_projection_specs",
                        export_name="_topologically_order_report_projection_specs",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.engine.dataflow_contracts",
                bindings=(
                    AliasBindingSpec(
                        source_name="AuditConfig", export_name="AuditConfig"
                    ),
                    AliasBindingSpec(source_name="CallArgs", export_name="CallArgs"),
                    AliasBindingSpec(source_name="ClassInfo", export_name="ClassInfo"),
                    AliasBindingSpec(
                        source_name="FunctionInfo", export_name="FunctionInfo"
                    ),
                    AliasBindingSpec(
                        source_name="InvariantProposition",
                        export_name="InvariantProposition",
                    ),
                    AliasBindingSpec(source_name="ParamUse", export_name="ParamUse"),
                    AliasBindingSpec(
                        source_name="SymbolTable", export_name="SymbolTable"
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.io.dataflow_reporting",
                bindings=(
                    AliasBindingSpec(
                        source_name="emit_report", export_name="_emit_report"
                    ),
                    AliasBindingSpec(
                        source_name="render_report", export_name="render_report"
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.io.dataflow_reporting_helpers",
                bindings=(
                    AliasBindingSpec(
                        source_name="render_mermaid_component",
                        export_name="_render_mermaid_component",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.dataflow.io.dataflow_parse_helpers",
                bindings=(
                    AliasBindingSpec(
                        source_name="_ParseModuleStage",
                        export_name="_ParseModuleStage",
                    ),
                    AliasBindingSpec(
                        source_name="_forbid_adhoc_bundle_discovery",
                        export_name="_forbid_adhoc_bundle_discovery",
                    ),
                ),
            ),
            ModuleAliasSpec(
                module_path="gabion.analysis.indexed_scan.scanners.report_sections",
                bindings=(
                    AliasBindingSpec(
                        source_name="extract_report_sections",
                        export_name="extract_report_sections",
                    ),
                ),
            ),
        ),
    ),
)


def _materialize_module_aliases() -> tuple[
    dict[str, object],
    tuple[dict[str, object], ...],
    Counter[str],
]:
    module_cache: dict[str, object] = {}
    exports: dict[str, object] = {name: value for name, value in _LOCAL_SUPPORT_BINDINGS}
    group_inventory: list[dict[str, object]] = []
    module_alias_counts: Counter[str] = Counter()

    for group_spec in ALIAS_GROUP_SPECS:
        group_export_names: list[str] = []
        module_inventory: list[dict[str, object]] = []
        for module_spec in group_spec.module_specs:
            module = module_cache.get(module_spec.module_path)
            if module is None:
                module = importlib.import_module(module_spec.module_path)
                module_cache[module_spec.module_path] = module
            module_export_names: list[str] = []
            for binding in module_spec.bindings:
                if binding.export_name in exports:
                    raise ValueError(
                        "duplicate dataflow indexed alias export "
                        f"{binding.export_name!r} from {module_spec.module_path}"
                    )
                exports[binding.export_name] = getattr(module, binding.source_name)
                group_export_names.append(binding.export_name)
                module_export_names.append(binding.export_name)
            module_alias_counts[module_spec.module_path] += len(module_spec.bindings)
            module_inventory.append(
                {
                    "module_path": module_spec.module_path,
                    "alias_count": len(module_export_names),
                    "export_names": tuple(module_export_names),
                }
            )
        group_inventory.append(
            {
                "group_id": group_spec.group_id,
                "label": group_spec.label,
                "owner_module_count": len(
                    {entry["module_path"] for entry in module_inventory}
                ),
                "alias_count": len(group_export_names),
                "export_names": tuple(group_export_names),
                "modules": tuple(module_inventory),
            }
        )

    return exports, tuple(group_inventory), module_alias_counts


def _add_local_type_aliases(surface: dict[str, object]) -> None:
    # Keep historically exposed boundary aliases available on the compatibility surface.
    surface["FunctionNode"] = ast.FunctionDef | ast.AsyncFunctionDef
    surface["OptionalIgnoredParams"] = set[str] | None
    surface["ParamAnnotationMap"] = dict[str, str | None]
    surface["ReturnAliasMap"] = dict[str, tuple[list[str], list[str]]]
    surface["OptionalReturnAliasMap"] = surface["ReturnAliasMap"] | None
    surface["OptionalClassName"] = str | None
    surface["Span4"] = tuple[int, int, int, int]
    surface["OptionalSpan4"] = surface["Span4"] | None
    surface["OptionalString"] = str | None
    surface["OptionalFloat"] = float | None
    surface["OptionalPath"] = Path | None
    surface["OptionalStringSet"] = set[str] | None
    surface["OptionalPrimeRegistry"] = surface["PrimeRegistry"] | None
    surface["OptionalTypeConstructorRegistry"] = (
        surface["TypeConstructorRegistry"] | None
    )
    surface["OptionalSynthRegistry"] = surface["SynthRegistry"] | None
    surface["OptionalJsonObject"] = surface["JSONObject"] | None
    surface["OptionalForestSpec"] = surface["ForestSpec"] | None
    surface["OptionalDeprecatedExtractionArtifacts"] = (
        surface["DeprecatedExtractionArtifacts"] | None
    )
    surface["OptionalAstCall"] = ast.Call | None
    surface["NodeIdOrNone"] = surface["NodeId"] | None
    surface["ParseCacheValue"] = ast.Module | BaseException
    surface["ReportProjectionPhase"] = Literal["collection", "forest", "edge", "post"]


def materialize_alias_boundary_surface() -> AliasSurfaceMaterialization:
    exports, group_inventory, module_alias_counts = _materialize_module_aliases()
    _add_local_type_aliases(exports)

    exported_names = tuple(exports)
    star_export_names = tuple(name for name in exported_names if not name.startswith("_"))
    remaining_hot_spots = tuple(
        {
            "module_path": module_path,
            "alias_count": alias_count,
        }
        for module_path, alias_count in module_alias_counts.most_common()
        if alias_count >= 10
    )

    inventory = {
        "surface_module": "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan",
        "compatibility_scope": BOUNDARY_ADAPTER_LIFECYCLE["scope"],
        "local_support_names": tuple(name for name, _ in _LOCAL_SUPPORT_BINDINGS),
        "local_type_alias_names": _LOCAL_TYPE_ALIAS_NAMES,
        "group_count": len(group_inventory),
        "module_groups": group_inventory,
        "exported_names": exported_names,
        "star_export_names": star_export_names,
    }
    telemetry = {
        "surface_module": "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan",
        "compatibility_scope": BOUNDARY_ADAPTER_LIFECYCLE["scope"],
        "exported_alias_count": len(exported_names),
        "star_export_count": len(star_export_names),
        "local_support_count": len(_LOCAL_SUPPORT_BINDINGS),
        "local_type_alias_count": len(_LOCAL_TYPE_ALIAS_NAMES),
        "owner_module_spread": len(module_alias_counts),
        "remaining_hot_spots": remaining_hot_spots,
        "compatibility_policy_surfaces": {
            "private_symbol_allowlist_path": "docs/policy/private_symbol_import_allowlist.txt",
            "debt_ledger_path": "docs/audits/dataflow_runtime_debt_ledger.md",
            "retirement_ledger_path": "docs/audits/dataflow_runtime_retirement_ledger.md",
            "decomposition_ledger_path": "docs/ws5_decomposition_ledger.md",
        },
    }
    return AliasSurfaceMaterialization(
        exports=exports,
        inventory=inventory,
        telemetry=telemetry,
    )
