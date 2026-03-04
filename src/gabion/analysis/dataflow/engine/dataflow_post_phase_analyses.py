# gabion:boundary_normalization_module
from __future__ import annotations

"""Post-phase analysis owner facade during WS-5 decomposition."""

import importlib

_RUNTIME_MODULE = "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan"
_runtime = importlib.import_module(_RUNTIME_MODULE)

analyze_type_flow_repo_with_map = _runtime.analyze_type_flow_repo_with_map
analyze_type_flow_repo_with_evidence = _runtime.analyze_type_flow_repo_with_evidence
analyze_constant_flow_repo = _runtime.analyze_constant_flow_repo
analyze_deadness_flow_repo = _runtime.analyze_deadness_flow_repo
analyze_unused_arg_flow_repo = _runtime.analyze_unused_arg_flow_repo
_collect_constant_flow_details = _runtime._collect_constant_flow_details
_collect_exception_obligations = _runtime._collect_exception_obligations
_collect_handledness_witnesses = _runtime._collect_handledness_witnesses
_collect_never_invariants = _runtime._collect_never_invariants
_collect_invariant_propositions = _runtime._collect_invariant_propositions
_param_annotations_by_path = _runtime._param_annotations_by_path
analyze_decision_surfaces_repo = _runtime.analyze_decision_surfaces_repo
analyze_value_encoded_decisions_repo = _runtime.analyze_value_encoded_decisions_repo
generate_property_hook_manifest = _runtime.generate_property_hook_manifest
_build_property_hook_callable_index = _runtime._build_property_hook_callable_index
_compute_knob_param_names = _runtime._compute_knob_param_names
_collect_config_bundles = _runtime._collect_config_bundles
_iter_config_fields = _runtime._iter_config_fields
_collect_dataclass_registry = _runtime._collect_dataclass_registry
_iter_dataclass_call_bundles = _runtime._iter_dataclass_call_bundles
_deserialize_invariants_for_resume = _runtime._deserialize_invariants_for_resume
_callsite_evidence_for_bundle = _runtime._callsite_evidence_for_bundle
_format_call_site = _runtime._format_call_site
_format_type_flow_site = _runtime._format_type_flow_site
_type_from_const_repr = _runtime._type_from_const_repr
_combine_type_hints = _runtime._combine_type_hints


__all__ = [
    "_build_property_hook_callable_index",
    "_callsite_evidence_for_bundle",
    "_collect_config_bundles",
    "_collect_constant_flow_details",
    "_collect_dataclass_registry",
    "_collect_exception_obligations",
    "_collect_handledness_witnesses",
    "_collect_invariant_propositions",
    "_collect_never_invariants",
    "_combine_type_hints",
    "_compute_knob_param_names",
    "_deserialize_invariants_for_resume",
    "_format_call_site",
    "_format_type_flow_site",
    "_iter_config_fields",
    "_iter_dataclass_call_bundles",
    "_param_annotations_by_path",
    "_type_from_const_repr",
    "analyze_constant_flow_repo",
    "analyze_deadness_flow_repo",
    "analyze_decision_surfaces_repo",
    "analyze_type_flow_repo_with_evidence",
    "analyze_type_flow_repo_with_map",
    "analyze_unused_arg_flow_repo",
    "analyze_value_encoded_decisions_repo",
    "generate_property_hook_manifest",
]
