# gabion:boundary_normalization_module
from __future__ import annotations

"""Resume serialization owner facade during WS-5 decomposition."""

import importlib

_RUNTIME_MODULE = "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan"
_runtime = importlib.import_module(_RUNTIME_MODULE)

_serialize_param_use = _runtime._serialize_param_use
_deserialize_param_use = _runtime._deserialize_param_use
_serialize_param_use_map = _runtime._serialize_param_use_map
_deserialize_param_use_map = _runtime._deserialize_param_use_map
_serialize_call_args = _runtime._serialize_call_args
_deserialize_call_args = _runtime._deserialize_call_args
_serialize_call_args_list = _runtime._serialize_call_args_list
_deserialize_call_args_list = _runtime._deserialize_call_args_list
_serialize_function_info_for_resume = _runtime._serialize_function_info_for_resume
_deserialize_function_info_for_resume = _runtime._deserialize_function_info_for_resume
_serialize_class_info_for_resume = _runtime._serialize_class_info_for_resume
_deserialize_class_info_for_resume = _runtime._deserialize_class_info_for_resume
_serialize_symbol_table_for_resume = _runtime._serialize_symbol_table_for_resume
_deserialize_symbol_table_for_resume = _runtime._deserialize_symbol_table_for_resume
_analysis_index_resume_variant_payload = _runtime._analysis_index_resume_variant_payload
_analysis_index_resume_variants = _runtime._analysis_index_resume_variants
_with_analysis_index_resume_variants = _runtime._with_analysis_index_resume_variants
_serialize_analysis_index_resume_payload = _runtime._serialize_analysis_index_resume_payload
_load_analysis_index_resume_payload = _runtime._load_analysis_index_resume_payload
_serialize_groups_for_resume = _runtime._serialize_groups_for_resume
_deserialize_groups_for_resume = _runtime._deserialize_groups_for_resume
_serialize_param_spans_for_resume = _runtime._serialize_param_spans_for_resume
_deserialize_param_spans_for_resume = _runtime._deserialize_param_spans_for_resume
_serialize_bundle_sites_for_resume = _runtime._serialize_bundle_sites_for_resume
_deserialize_bundle_sites_for_resume = _runtime._deserialize_bundle_sites_for_resume
_serialize_invariants_for_resume = _runtime._serialize_invariants_for_resume
_deserialize_invariants_for_resume = _runtime._deserialize_invariants_for_resume
_serialize_file_scan_resume_state = _runtime._serialize_file_scan_resume_state
_load_file_scan_resume_state = _runtime._load_file_scan_resume_state
_build_analysis_collection_resume_payload = _runtime._build_analysis_collection_resume_payload
_load_analysis_collection_resume_payload = _runtime._load_analysis_collection_resume_payload
_empty_analysis_collection_resume_payload = _runtime._empty_analysis_collection_resume_payload


__all__ = [
    "_analysis_index_resume_variant_payload",
    "_analysis_index_resume_variants",
    "_build_analysis_collection_resume_payload",
    "_deserialize_bundle_sites_for_resume",
    "_deserialize_call_args",
    "_deserialize_call_args_list",
    "_deserialize_class_info_for_resume",
    "_deserialize_function_info_for_resume",
    "_deserialize_groups_for_resume",
    "_deserialize_invariants_for_resume",
    "_deserialize_param_spans_for_resume",
    "_deserialize_param_use",
    "_deserialize_param_use_map",
    "_deserialize_symbol_table_for_resume",
    "_empty_analysis_collection_resume_payload",
    "_load_analysis_collection_resume_payload",
    "_load_analysis_index_resume_payload",
    "_load_file_scan_resume_state",
    "_serialize_analysis_index_resume_payload",
    "_serialize_bundle_sites_for_resume",
    "_serialize_call_args",
    "_serialize_call_args_list",
    "_serialize_class_info_for_resume",
    "_serialize_file_scan_resume_state",
    "_serialize_function_info_for_resume",
    "_serialize_groups_for_resume",
    "_serialize_invariants_for_resume",
    "_serialize_param_spans_for_resume",
    "_serialize_param_use",
    "_serialize_param_use_map",
    "_serialize_symbol_table_for_resume",
    "_with_analysis_index_resume_variants",
]
