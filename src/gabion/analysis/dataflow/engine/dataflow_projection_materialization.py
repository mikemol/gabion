# gabion:boundary_normalization_module
from __future__ import annotations

"""Projection/spec materialization owner facade during WS-5 decomposition."""

import importlib

_RUNTIME_MODULE = "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan"
_runtime = importlib.import_module(_RUNTIME_MODULE)

_ProjectionSpan = _runtime._ProjectionSpan
_AmbiguitySuiteRow = _runtime._AmbiguitySuiteRow
CallAmbiguity = _runtime.CallAmbiguity
_decode_projection_span = _runtime._decode_projection_span
_spec_row_span = _runtime._spec_row_span
_materialize_projection_spec_rows = _runtime._materialize_projection_spec_rows
_suite_order_depth = _runtime._suite_order_depth
_suite_order_relation = _runtime._suite_order_relation
_suite_order_row_to_site = _runtime._suite_order_row_to_site
_suite_site_label = _runtime._suite_site_label
_materialize_suite_order_spec = _runtime._materialize_suite_order_spec
_ambiguity_suite_relation = _runtime._ambiguity_suite_relation
_decode_ambiguity_suite_row = _runtime._decode_ambiguity_suite_row
_ambiguity_suite_row_to_suite = _runtime._ambiguity_suite_row_to_suite
_ambiguity_virtual_count_gt_1 = _runtime._ambiguity_virtual_count_gt_1
_materialize_ambiguity_suite_agg_spec = _runtime._materialize_ambiguity_suite_agg_spec
_materialize_ambiguity_virtual_set_spec = _runtime._materialize_ambiguity_virtual_set_spec
_collect_call_ambiguities_indexed = _runtime._collect_call_ambiguities_indexed
_collect_call_ambiguities = _runtime._collect_call_ambiguities
_dedupe_call_ambiguities = _runtime._dedupe_call_ambiguities
_emit_call_ambiguities = _runtime._emit_call_ambiguities
_summarize_call_ambiguities = _runtime._summarize_call_ambiguities
_lint_lines_from_call_ambiguities = _runtime._lint_lines_from_call_ambiguities
_populate_bundle_forest = _runtime._populate_bundle_forest


__all__ = [
    "CallAmbiguity",
    "_AmbiguitySuiteRow",
    "_ProjectionSpan",
    "_ambiguity_suite_relation",
    "_ambiguity_suite_row_to_suite",
    "_ambiguity_virtual_count_gt_1",
    "_collect_call_ambiguities",
    "_collect_call_ambiguities_indexed",
    "_decode_ambiguity_suite_row",
    "_decode_projection_span",
    "_dedupe_call_ambiguities",
    "_emit_call_ambiguities",
    "_lint_lines_from_call_ambiguities",
    "_materialize_ambiguity_suite_agg_spec",
    "_materialize_ambiguity_virtual_set_spec",
    "_materialize_projection_spec_rows",
    "_materialize_suite_order_spec",
    "_populate_bundle_forest",
    "_spec_row_span",
    "_suite_order_depth",
    "_suite_order_relation",
    "_suite_order_row_to_site",
    "_suite_site_label",
    "_summarize_call_ambiguities",
]
