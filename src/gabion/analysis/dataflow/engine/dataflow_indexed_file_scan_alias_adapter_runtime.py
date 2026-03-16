# gabion:ambiguity_boundary_module
# gabion:grade_boundary kind=semantic_carrier_adapter name=dataflow_indexed_file_scan_alias_adapter_runtime
from __future__ import annotations

"""Deadline and resume runtime alias groups for the legacy monolith path."""

from gabion.analysis.dataflow.engine.dataflow_indexed_file_scan_alias_contract import (
    AliasGroupSpec,
    alias_group,
    module_alias,
)

RUNTIME_ALIAS_GROUPS: tuple[AliasGroupSpec, ...] = (
    alias_group(
        'deadline_runtime',
        'Deadline Runtime',
        module_alias(
            'gabion.analysis.dataflow.engine.dataflow_deadline_contracts',
            '_CalleeResolutionOutcome',
            '_DeadlineFunctionFacts',
            '_DeadlineLocalInfo',
        ),
        module_alias(
            'gabion.analysis.dataflow.engine.dataflow_deadline_helpers',
            '_is_deadline_param',
        ),
        module_alias(
            'gabion.analysis.dataflow.engine.dataflow_deadline_helpers',
            '_DeadlineFunctionCollector',
            '_DeadlineArgInfo',
            '_bind_call_args',
            '_classify_deadline_expr',
            '_collect_call_edges',
            '_collect_call_edges_from_forest',
            '_collect_call_nodes_by_path',
            '_collect_deadline_function_facts',
            '_collect_deadline_local_info',
            '_deadline_arg_info_map',
            '_deadline_loop_forwarded_params',
            '_fallback_deadline_arg_info',
            '_is_dynamic_dispatch_callee_key',
            '_materialize_call_candidates',
            '_resolve_callee',
            '_resolve_callee_outcome',
        ),
        module_alias(
            'gabion.analysis.indexed_scan.deadline.deadline_runtime',
            ('is_deadline_origin_call', '_is_deadline_origin_call'),
        ),
    ),
    alias_group(
        'resume_runtime',
        'Resume Runtime',
        module_alias(
            'gabion.analysis.dataflow.engine.dataflow_resume_paths',
            ('normalize_snapshot_path', '_normalize_snapshot_path'),
        ),
        module_alias(
            'gabion.analysis.dataflow.engine.dataflow_resume_serialization',
            '_CACHE_IDENTITY_DIGEST_HEX',
            '_CACHE_IDENTITY_PREFIX',
            '_CacheIdentity',
            '_build_analysis_collection_resume_payload',
            '_deserialize_function_info_for_resume',
            '_deserialize_invariants_for_resume',
            '_deserialize_symbol_table_for_resume',
            '_invariant_confidence',
            '_invariant_digest',
            '_load_analysis_collection_resume_payload',
            '_load_analysis_index_resume_payload',
            '_load_file_scan_resume_state',
            '_serialize_analysis_index_resume_payload',
            '_serialize_file_scan_resume_state',
            '_normalize_invariant_proposition',
        ),
    ),
)
