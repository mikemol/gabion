# gabion:boundary_normalization_module
from __future__ import annotations

"""Temporary synthesis bridge for extracted owner surfaces."""

from gabion.analysis.dataflow.engine.dataflow_analysis_index import _build_call_graph
from gabion.analysis.dataflow.engine.dataflow_indexed_file_scan import (
    _collect_config_bundles, _combine_type_hints, _compute_knob_param_names, _type_from_const_repr, analyze_type_flow_repo_with_map, generate_property_hook_manifest)
_assert_exports = (
    _build_call_graph,
    _compute_knob_param_names,
    _collect_config_bundles,
    analyze_type_flow_repo_with_map,
    _type_from_const_repr,
    _combine_type_hints,
    generate_property_hook_manifest,
)


__all__ = [
    "_build_call_graph",
    "_collect_config_bundles",
    "_combine_type_hints",
    "_compute_knob_param_names",
    "_type_from_const_repr",
    "analyze_type_flow_repo_with_map",
    "generate_property_hook_manifest",
]
