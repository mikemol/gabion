from __future__ import annotations

"""Ambiguity and bundle-forest helper boundary during runtime retirement."""


from dataclasses import dataclass

from gabion.analysis.dataflow.engine.dataflow_contracts import FunctionInfo
from gabion.analysis.dataflow.engine.dataflow_projection_materialization import (
    _ambiguity_suite_relation as _ambiguity_suite_relation_runtime_adapter,
    _ambiguity_suite_row_to_suite as _ambiguity_suite_row_to_suite_runtime_adapter,
    _ambiguity_virtual_count_gt_1 as _ambiguity_virtual_count_gt_1_runtime_adapter,
    _collect_call_ambiguities as _collect_call_ambiguities_runtime_adapter,
    _collect_call_ambiguities_indexed as _collect_call_ambiguities_indexed_runtime_adapter,
    _dedupe_call_ambiguities as _dedupe_call_ambiguities_runtime_adapter,
    _emit_call_ambiguities as _emit_call_ambiguities_runtime_adapter,
    _lint_lines_from_call_ambiguities as _lint_lines_from_call_ambiguities_runtime_adapter,
    _materialize_ambiguity_suite_agg_spec as _materialize_ambiguity_suite_agg_spec_runtime_adapter,
    _materialize_ambiguity_virtual_set_spec as _materialize_ambiguity_virtual_set_spec_runtime_adapter,
    _materialize_suite_order_spec as _materialize_suite_order_spec_runtime_adapter,
    _populate_bundle_forest as _populate_bundle_forest_runtime_adapter,
    _suite_order_relation as _suite_order_relation_runtime_adapter,
    _suite_order_row_to_site as _suite_order_row_to_site_runtime_adapter,
    _suite_site_label as _suite_site_label_runtime_adapter,
    _summarize_call_ambiguities as _summarize_call_ambiguities_runtime_adapter,
)


@dataclass(frozen=True)
class CallAmbiguity:
    kind: str
    caller: FunctionInfo
    call: object
    callee_key: str
    candidates: tuple[FunctionInfo, ...]
    phase: str


_collect_call_ambiguities = _collect_call_ambiguities_runtime_adapter
_collect_call_ambiguities_indexed = _collect_call_ambiguities_indexed_runtime_adapter
_dedupe_call_ambiguities = _dedupe_call_ambiguities_runtime_adapter
_emit_call_ambiguities = _emit_call_ambiguities_runtime_adapter
_lint_lines_from_call_ambiguities = _lint_lines_from_call_ambiguities_runtime_adapter
_materialize_ambiguity_suite_agg_spec = _materialize_ambiguity_suite_agg_spec_runtime_adapter
_materialize_ambiguity_virtual_set_spec = _materialize_ambiguity_virtual_set_spec_runtime_adapter
_materialize_suite_order_spec = _materialize_suite_order_spec_runtime_adapter
_populate_bundle_forest = _populate_bundle_forest_runtime_adapter
_summarize_call_ambiguities = _summarize_call_ambiguities_runtime_adapter
_ambiguity_suite_relation = _ambiguity_suite_relation_runtime_adapter
_ambiguity_suite_row_to_suite = _ambiguity_suite_row_to_suite_runtime_adapter
_ambiguity_virtual_count_gt_1 = _ambiguity_virtual_count_gt_1_runtime_adapter
_suite_order_relation = _suite_order_relation_runtime_adapter
_suite_order_row_to_site = _suite_order_row_to_site_runtime_adapter
_suite_site_label = _suite_site_label_runtime_adapter


__all__ = [
    "CallAmbiguity",
    "_ambiguity_suite_relation",
    "_ambiguity_suite_row_to_suite",
    "_ambiguity_virtual_count_gt_1",
    "_collect_call_ambiguities",
    "_collect_call_ambiguities_indexed",
    "_dedupe_call_ambiguities",
    "_emit_call_ambiguities",
    "_lint_lines_from_call_ambiguities",
    "_materialize_ambiguity_suite_agg_spec",
    "_materialize_ambiguity_virtual_set_spec",
    "_materialize_suite_order_spec",
    "_populate_bundle_forest",
    "_suite_order_relation",
    "_suite_order_row_to_site",
    "_suite_site_label",
    "_summarize_call_ambiguities",
]
