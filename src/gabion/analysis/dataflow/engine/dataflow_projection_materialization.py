# gabion:boundary_normalization_module
from __future__ import annotations

"""Projection/spec materialization owners extracted from the legacy monolith."""

from dataclasses import dataclass
from collections.abc import Callable, Iterable, Mapping

from gabion.analysis.aspf.aspf import Alt, Forest, NodeId
from gabion.analysis.dataflow.engine.dataflow_analysis_index_owner import (
    _IndexedPassSpec,
    _run_indexed_pass,
)
from gabion.analysis.dataflow.engine.dataflow_evidence_helpers import _resolve_callee
from gabion.analysis.dataflow.engine.dataflow_resume_paths import (
    normalize_snapshot_path as _normalize_snapshot_path_impl,
)
from gabion.analysis.foundation.json_types import JSONObject, JSONValue
from gabion.analysis.foundation.resume_codec import int_tuple4_or_none
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.semantics import evidence_keys
from gabion.analysis.indexed_scan.calls.call_ambiguities import (
    CallAmbiguitiesEmitDeps as _CallAmbiguitiesEmitDeps,
    emit_call_ambiguities as _emit_call_ambiguities_impl,
)
from gabion.analysis.indexed_scan.calls.call_ambiguity_summary import (
    CallAmbiguitySummaryDeps as _CallAmbiguitySummaryDeps,
    summarize_call_ambiguities as _summarize_call_ambiguities_impl,
)
from gabion.analysis.indexed_scan.deadline.deadline_runtime import (
    call_candidate_target_site as _call_candidate_target_site_impl,
)
from gabion.analysis.indexed_scan.scanners.materialization.suite_order_relation import (
    AmbiguitySuiteRelationDeps as _AmbiguitySuiteRelationDeps,
    SuiteOrderRelationDeps as _SuiteOrderRelationDeps,
    ambiguity_suite_relation as _ambiguity_suite_relation_impl,
    suite_order_relation as _suite_order_relation_impl,
)
from gabion.analysis.indexed_scan.scanners.report_sections import (
    spec_row_span as _spec_row_span_impl,
)
from gabion.analysis.projection.projection_exec import apply_spec
from gabion.analysis.projection.projection_registry import (
    AMBIGUITY_SUMMARY_SPEC,
    AMBIGUITY_SUITE_AGG_SPEC,
    AMBIGUITY_VIRTUAL_SET_SPEC,
    SUITE_ORDER_SPEC,
    spec_metadata_lines_from_payload,
    spec_metadata_payload,
)
from gabion.invariants import never
from gabion.order_contract import sort_once


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


@dataclass(frozen=True)
class CallAmbiguity:
    kind: str
    caller: object
    call: object
    callee_key: str
    candidates: tuple[object, ...]
    phase: str


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
    spec: object,
    projected: Iterable[Mapping[str, JSONValue]],
    forest: Forest,
    row_to_site: Callable[[Mapping[str, JSONValue]], object],
) -> None:
    from gabion.analysis.dataflow.io.dataflow_reporting_helpers import (
        _materialize_projection_spec_rows as _impl,
    )

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
    if span_text:
        return f"{path}:{qual}[{suite_kind}]@{span_text}"
    return f"{path}:{qual}[{suite_kind}]"


def _format_span_fields(
    line: object,
    col: object,
    end_line: object,
    end_col: object,
) -> str:
    from gabion.analysis.dataflow.io.dataflow_reporting_helpers import (
        _format_span_fields as _impl,
    )

    return _impl(line, col, end_line, end_col)


def _normalize_snapshot_path(path, root) -> str:
    return _normalize_snapshot_path_impl(path, root)


def _add_interned_alt(
    *,
    forest: Forest,
    kind: str,
    inputs: Iterable[NodeId],
    evidence=None,
) -> Alt:
    return forest.add_alt(kind, inputs, evidence=evidence)


def _call_candidate_target_site(
    *,
    forest: Forest,
    candidate,
) -> NodeId:
    return _call_candidate_target_site_impl(forest=forest, candidate=candidate)


def _collect_call_ambiguities_indexed(
    context,
    *,
    resolve_callee_fn=None,
) -> list[CallAmbiguity]:
    ambiguities: list[CallAmbiguity] = []
    resolve_callee = _resolve_callee if resolve_callee_fn is None else resolve_callee_fn

    def _sink(
        caller,
        call,
        candidates,
        phase: str,
        callee_key: str,
    ) -> None:
        ordered = tuple(
            sort_once(
                candidates,
                key=lambda info: info.qual,
                source="gabion.analysis.dataflow_projection_materialization._sink.site_1",
            )
        )
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
    paths,
    *,
    project_root,
    ignore_params,
    strictness: str,
    external_filter: bool,
    transparent_decorators=None,
    parse_failure_witnesses,
    analysis_index=None,
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


def _lint_lines_from_call_ambiguities(entries: Iterable[JSONObject]) -> list[str]:
    from gabion.analysis.dataflow.engine.dataflow_lint_helpers import (
        _lint_lines_from_call_ambiguities as _impl,
    )

    return _impl(entries)


def _populate_bundle_forest(*args, **kwargs):
    from gabion.analysis.dataflow.engine.dataflow_facade import (
        _populate_bundle_forest as _populate_bundle_forest_runtime,
    )

    return _populate_bundle_forest_runtime(*args, **kwargs)


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
