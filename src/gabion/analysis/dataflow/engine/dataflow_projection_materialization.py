# gabion:boundary_normalization_module
from __future__ import annotations

"""Projection/spec materialization owners extracted from the legacy monolith."""

from dataclasses import dataclass
import importlib
from collections.abc import Callable, Iterable, Mapping

from gabion.analysis.aspf.aspf import Forest, NodeId
from gabion.analysis.foundation.json_types import JSONValue
from gabion.analysis.foundation.resume_codec import int_tuple4_or_none
from gabion.analysis.foundation.timeout_context import check_deadline
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
    AMBIGUITY_SUITE_AGG_SPEC,
    AMBIGUITY_VIRTUAL_SET_SPEC,
    SUITE_ORDER_SPEC,
)
from gabion.invariants import never
from gabion.order_contract import sort_once

_RUNTIME_MODULE = "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan"


def _runtime_module():
    return importlib.import_module(_RUNTIME_MODULE)


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
    return _runtime_module()._suite_site_label(forest=forest, suite_id=suite_id)


def _collect_call_ambiguities_indexed(*args, **kwargs):
    return _runtime_module()._collect_call_ambiguities_indexed(*args, **kwargs)


def _collect_call_ambiguities(*args, **kwargs):
    return _runtime_module()._collect_call_ambiguities(*args, **kwargs)


def _dedupe_call_ambiguities(*args, **kwargs):
    return _runtime_module()._dedupe_call_ambiguities(*args, **kwargs)


def _emit_call_ambiguities(*args, **kwargs):
    return _runtime_module()._emit_call_ambiguities(*args, **kwargs)


def _summarize_call_ambiguities(*args, **kwargs):
    return _runtime_module()._summarize_call_ambiguities(*args, **kwargs)


def _lint_lines_from_call_ambiguities(*args, **kwargs):
    return _runtime_module()._lint_lines_from_call_ambiguities(*args, **kwargs)


def _populate_bundle_forest(*args, **kwargs):
    return _runtime_module()._populate_bundle_forest(*args, **kwargs)


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
