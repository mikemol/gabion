# gabion:boundary_normalization_module
from __future__ import annotations

"""Ambiguity and bundle-forest helper boundary during runtime retirement."""


from dataclasses import dataclass

from gabion.analysis.dataflow_contracts import FunctionInfo
from gabion.analysis.dataflow_indexed_file_scan import (
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


def _collect_call_ambiguities(
    paths,
    *,
    project_root,
    ignore_params,
    strictness,
    external_filter,
    transparent_decorators=None,
    parse_failure_witnesses=None,
    analysis_index=None,
):
    return _collect_call_ambiguities_runtime_adapter(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
        analysis_index=analysis_index,
    )


def _collect_call_ambiguities_indexed(*args, **kwargs):
    return _collect_call_ambiguities_indexed_runtime_adapter(*args, **kwargs)


def _dedupe_call_ambiguities(*args, **kwargs):
    return _dedupe_call_ambiguities_runtime_adapter(*args, **kwargs)


def _emit_call_ambiguities(call_ambiguities, *, project_root, forest):
    return _emit_call_ambiguities_runtime_adapter(
        call_ambiguities,
        project_root=project_root,
        forest=forest,
    )


def _lint_lines_from_call_ambiguities(*args, **kwargs):
    return _lint_lines_from_call_ambiguities_runtime_adapter(*args, **kwargs)


def _materialize_ambiguity_suite_agg_spec(*, forest):
    return _materialize_ambiguity_suite_agg_spec_runtime_adapter(forest=forest)


def _materialize_ambiguity_virtual_set_spec(*, forest):
    return _materialize_ambiguity_virtual_set_spec_runtime_adapter(forest=forest)


def _materialize_suite_order_spec(*, forest):
    return _materialize_suite_order_spec_runtime_adapter(forest=forest)


def _populate_bundle_forest(
    forest,
    *,
    groups_by_path,
    file_paths,
    project_root,
    include_all_sites,
    ignore_params,
    strictness,
    transparent_decorators,
    parse_failure_witnesses,
    analysis_index=None,
    on_progress=None,
):
    return _populate_bundle_forest_runtime_adapter(
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
    )


def _summarize_call_ambiguities(*args, **kwargs):
    return _summarize_call_ambiguities_runtime_adapter(*args, **kwargs)


def _ambiguity_suite_relation(forest):
    return _ambiguity_suite_relation_runtime_adapter(forest)


def _ambiguity_suite_row_to_suite(row, forest):
    return _ambiguity_suite_row_to_suite_runtime_adapter(row, forest)


def _ambiguity_virtual_count_gt_1(row, forest):
    return _ambiguity_virtual_count_gt_1_runtime_adapter(row, forest)


def _suite_order_relation(*, forest):
    return _suite_order_relation_runtime_adapter(forest=forest)


def _suite_order_row_to_site(row, *, forest):
    return _suite_order_row_to_site_runtime_adapter(row, forest=forest)


def _suite_site_label(*, forest, suite_id):
    return _suite_site_label_runtime_adapter(forest=forest, suite_id=suite_id)


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
