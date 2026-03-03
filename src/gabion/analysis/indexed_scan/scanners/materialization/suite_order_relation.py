# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

from gabion.analysis.aspf.aspf import Forest, NodeId
from gabion.analysis.foundation.json_types import JSONValue


@dataclass(frozen=True)
class SuiteOrderRelationDeps:
    check_deadline_fn: Callable[[], None]
    never_fn: Callable[..., object]
    int_tuple4_or_none_fn: Callable[..., object]
    suite_order_depth_fn: Callable[[str], int]
    sort_once_fn: Callable[..., list[object]]


@dataclass(frozen=True)
class AmbiguitySuiteRelationDeps:
    check_deadline_fn: Callable[[], None]
    never_fn: Callable[..., object]
    int_tuple4_or_none_fn: Callable[..., object]


def suite_order_relation(
    forest: Forest,
    *,
    deps: SuiteOrderRelationDeps,
) -> tuple[list[dict[str, JSONValue]], dict[tuple[object, ...], NodeId]]:
    alt_degree: Counter[NodeId] = Counter()
    for alt in forest.alts:
        deps.check_deadline_fn()
        if alt.kind == "SpecFacet":
            continue
        for node_id in alt.inputs:
            deps.check_deadline_fn()
            alt_degree[node_id] += 1
    relation: list[dict[str, JSONValue]] = []
    suite_index: dict[tuple[object, ...], NodeId] = {}
    for node_id, node in forest.nodes.items():
        deps.check_deadline_fn()
        if node_id.kind != "SuiteSite":
            continue
        suite_kind = str(node.meta.get("suite_kind", "") or "")
        if suite_kind == "spec":
            continue
        path = str(node.meta.get("path", "") or "")
        qual = str(node.meta.get("qual", "") or "")
        if not path or not qual:
            deps.never_fn(
                "suite order requires path/qual",
                path=path,
                qual=qual,
                suite_kind=suite_kind,
            )
        span = node.meta.get("span")
        parsed_span = deps.int_tuple4_or_none_fn(span)
        if parsed_span is None:
            deps.never_fn(
                "suite order requires span",
                path=path,
                qual=qual,
                suite_kind=suite_kind,
                span=span,
            )
        span_line, span_col, span_end_line, span_end_col = parsed_span
        depth = deps.suite_order_depth_fn(suite_kind)
        complexity = int(alt_degree.get(node_id, 0))
        order_key: list[JSONValue] = [
            depth,
            complexity,
            path,
            qual,
            span_line,
            span_col,
            span_end_line,
            span_end_col,
        ]
        relation.append(
            {
                "suite_path": path,
                "suite_qual": qual,
                "suite_kind": suite_kind,
                "span_line": span_line,
                "span_col": span_col,
                "span_end_line": span_end_line,
                "span_end_col": span_end_col,
                "depth": depth,
                "complexity": complexity,
                "order_key": order_key,
            }
        )
        suite_index[
            (path, qual, suite_kind, span_line, span_col, span_end_line, span_end_col)
        ] = node_id
    relation = deps.sort_once_fn(
        relation,
        key=lambda row: (
            int(row.get("depth", 0) or 0),
            int(row.get("complexity", 0) or 0),
            str(row.get("suite_path", "") or ""),
            str(row.get("suite_qual", "") or ""),
            int(row.get("span_line", -1) or -1),
            int(row.get("span_col", -1) or -1),
            int(row.get("span_end_line", -1) or -1),
            int(row.get("span_end_col", -1) or -1),
        ),
        source="gabion.analysis.dataflow_indexed_file_scan._suite_order_relation.relation",
    )
    return relation, suite_index


def ambiguity_suite_relation(
    forest: Forest,
    *,
    deps: AmbiguitySuiteRelationDeps,
) -> list[dict[str, JSONValue]]:
    relation: list[dict[str, JSONValue]] = []
    for alt in forest.alts:
        deps.check_deadline_fn()
        if alt.kind != "CallCandidate":
            continue
        if len(alt.inputs) < 2:
            continue
        suite_id = alt.inputs[0]
        suite_node = forest.nodes.get(suite_id)
        if suite_node is not None and suite_node.kind == "SuiteSite":
            suite_kind = str(suite_node.meta.get("suite_kind", "") or "")
            if suite_kind == "call":
                path = str(suite_node.meta.get("path", "") or "")
                qual = str(suite_node.meta.get("qual", "") or "")
                if not path or not qual:
                    deps.never_fn(
                        "ambiguity suite requires path/qual",
                        path=path,
                        qual=qual,
                        suite_kind=suite_kind,
                    )
                span = suite_node.meta.get("span")
                parsed_span = deps.int_tuple4_or_none_fn(span)
                if parsed_span is None:
                    deps.never_fn(
                        "ambiguity suite requires span",
                        path=path,
                        qual=qual,
                        suite_kind=suite_kind,
                        span=span,
                    )
                span_line, span_col, span_end_line, span_end_col = parsed_span
                relation.append(
                    {
                        "suite_path": path,
                        "suite_qual": qual,
                        "suite_kind": suite_kind,
                        "span_line": span_line,
                        "span_col": span_col,
                        "span_end_line": span_end_line,
                        "span_end_col": span_end_col,
                        "kind": str(alt.evidence.get("kind", "") or ""),
                        "phase": str(alt.evidence.get("phase", "") or ""),
                    }
                )
    return relation
