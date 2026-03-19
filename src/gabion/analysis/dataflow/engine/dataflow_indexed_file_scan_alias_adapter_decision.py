# gabion:ambiguity_boundary_module
# gabion:grade_boundary kind=semantic_carrier_adapter name=dataflow_indexed_file_scan_alias_adapter_decision
from __future__ import annotations

"""Decision-surface and fingerprint alias groups for the legacy monolith path."""

from gabion.analysis.dataflow.engine.dataflow_indexed_file_scan_alias_contract import (
    AliasGroupSpec,
    alias_group,
    module_alias,
)

DECISION_ALIAS_GROUPS: tuple[AliasGroupSpec, ...] = (
    alias_group(
        'decision_support',
        'Decision Support',
        module_alias(
            'gabion.analysis.dataflow.engine.dataflow_decision_surfaces',
            ('compute_fingerprint_coherence', '_ds_compute_fingerprint_coherence'),
            ('compute_fingerprint_rewrite_plans', '_ds_compute_fingerprint_rewrite_plans'),
            ('lint_lines_from_bundle_evidence', '_ds_lint_lines_from_bundle_evidence'),
            ('lint_lines_from_constant_smells', '_ds_lint_lines_from_constant_smells'),
            ('lint_lines_from_type_evidence', '_ds_lint_lines_from_type_evidence'),
            ('lint_lines_from_unused_arg_smells', '_ds_lint_lines_from_unused_arg_smells'),
            ('parse_lint_location', '_ds_parse_lint_location'),
            ('summarize_coherence_witnesses', '_ds_summarize_coherence_witnesses'),
            ('summarize_deadness_witnesses', '_ds_summarize_deadness_witnesses'),
            ('summarize_rewrite_plans', '_ds_summarize_rewrite_plans'),
        ),
        module_alias(
            'gabion.analysis.dataflow.engine.dataflow_bundle_merge',
            'merge_counts_by_knobs',
        ),
        module_alias(
            'gabion.analysis.dataflow.engine.dataflow_callee_resolution_support',
            '_callee_key',
            '_resolve_class_candidates',
            '_resolve_method_in_hierarchy',
        ),
        module_alias(
            'gabion.analysis.dataflow.engine.dataflow_fingerprint_helpers',
            '_build_synth_registry_payload',
            '_collect_fingerprint_atom_keys',
            '_compute_fingerprint_coherence',
            '_compute_fingerprint_matches',
            '_compute_fingerprint_provenance',
            '_compute_fingerprint_rewrite_plans',
            '_compute_fingerprint_synth',
            '_compute_fingerprint_warnings',
            '_find_provenance_entry_for_site',
            '_fingerprint_soundness_issues',
            '_glossary_match_strata',
            '_summarize_fingerprint_provenance',
            'verify_rewrite_plan',
            'verify_rewrite_plans',
        ),
        module_alias(
            'gabion.analysis.dataflow.engine.dataflow_adapter_contract',
            'parse_adapter_capabilities',
        ),
        module_alias(
            'gabion.analysis.dataflow.engine.dataflow_evidence_helpers',
            '_base_identifier',
            '_collect_module_exports',
            '_is_test_path',
            '_module_name',
            '_target_names',
        ),
        module_alias(
            'gabion.analysis.dataflow.engine.dataflow_function_semantics',
            '_analyze_function',
            '_call_context',
            '_collect_return_aliases',
            '_const_repr',
            '_normalize_key_expr',
        ),
        module_alias(
            'gabion.analysis.dataflow.engine.dataflow_function_index_runtime_support',
            '_direct_lambda_callee_by_call_span',
            '_materialize_direct_lambda_callees',
            '_unused_params',
        ),
        module_alias(
            'gabion.analysis.dataflow.engine.dataflow_lambda_runtime_support',
            '_collect_lambda_bindings_by_caller',
            '_collect_lambda_function_infos',
            '_function_key',
        ),
        module_alias(
            'gabion.analysis.dataflow.engine.dataflow_function_index_decision_support',
            '_collect_param_roots',
            '_contains_boolish',
            '_decorator_name',
            '_decision_surface_form_entries',
            '_decision_surface_params',
            '_decision_surface_reason_map',
            '_decorators_transparent',
            '_mark_param_roots',
            '_value_encoded_decision_params',
            'is_decision_surface',
        ),
        module_alias(
            'gabion.analysis.dataflow.engine.dataflow_call_graph_algorithms',
            '_collect_recursive_functions',
            '_collect_recursive_nodes',
            '_reachable_from_roots',
        ),
        module_alias(
            'gabion.analysis.dataflow.engine.dataflow_lint_helpers',
            '_constant_smells_from_details',
            '_deadness_witnesses_from_constant_details',
            '_deadline_lint_lines',
            '_exception_protocol_lint_lines',
            '_internal_broad_type_lint_lines',
            '_is_broad_internal_type',
            '_lint_lines_from_bundle_evidence',
            '_lint_lines_from_constant_smells',
            '_lint_lines_from_type_evidence',
            '_lint_lines_from_unused_arg_smells',
            '_normalize_type_name',
            '_parse_exception_path_id',
        ),
        module_alias(
            'gabion.analysis.dataflow.engine.dataflow_local_class_hierarchy',
            '_collect_local_class_bases',
            '_resolve_local_method_in_hierarchy',
        ),
    ),
)
