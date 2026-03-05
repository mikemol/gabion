# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

"""Lint and deadness helper ownership module during runtime retirement."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

from gabion.analysis.dataflow.engine.dataflow_decision_surfaces import (
    lint_lines_from_bundle_evidence as _ds_lint_lines_from_bundle_evidence, lint_lines_from_constant_smells as _ds_lint_lines_from_constant_smells, lint_lines_from_type_evidence as _ds_lint_lines_from_type_evidence, lint_lines_from_unused_arg_smells as _ds_lint_lines_from_unused_arg_smells, parse_lint_location as _ds_parse_lint_location)
from gabion.analysis.dataflow.engine.dataflow_evidence_helpers import (
    _is_test_path,
)
from gabion.analysis.dataflow.engine.dataflow_analysis_index_owner import (
    _build_analysis_index,
    _analysis_index_transitive_callers,
    _iter_monotonic_paths,
)
from gabion.analysis.dataflow.io.dataflow_reporting_helpers import (
    _materialize_projection_spec_rows, bundle_projection_from_forest as _bundle_projection_from_forest, bundle_site_index as _bundle_site_index, connected_components as _connected_components, has_bundles as _has_bundles, render_component_callsite_evidence as _render_component_callsite_evidence)
from gabion.analysis.dataflow.io.dataflow_snapshot_io import _normalize_snapshot_path
from gabion.analysis.dataflow.engine.dataflow_resume_serialization import (
    _analysis_collection_resume_path_key as _resume_analysis_collection_resume_path_key,
)
from gabion.analysis.dataflow.engine.dataflow_bundle_merge import _merge_counts_by_knobs as _merge_counts_by_knobs_impl
from gabion.analysis.foundation.json_types import JSONObject, JSONValue
from gabion.analysis.projection.projection_exec import apply_spec
from gabion.analysis.projection.projection_registry import LINT_FINDINGS_SPEC
from gabion.analysis.projection.projection_spec import ProjectionSpec
from gabion.analysis.foundation.resume_codec import (
    int_tuple4_or_none, mapping_or_empty, mapping_or_none, sequence_or_none)
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import never
from gabion.order_contract import sort_once

_NEVER_STATUS_ORDER = {"VIOLATION": 0, "OBLIGATION": 1, "PROVEN_UNREACHABLE": 2}

_analysis_collection_resume_path_key = _resume_analysis_collection_resume_path_key

def _analysis_index_by_qual_and_transitive_callers(
    *,
    analysis_index: object,
    project_root,
) -> tuple[dict[str, object], dict[str, set[str]]]:
    if analysis_index is None:
        never(
            "analysis index required for broad-type lint",
            source="dataflow_lint_helpers._analysis_index_by_qual_and_transitive_callers",
        )
    by_qual = cast(dict[str, object], getattr(analysis_index, "by_qual", {}))
    cached_transitive = cast(
        dict[str, set[str]] | None,
        getattr(analysis_index, "transitive_callers", None),
    )
    if cached_transitive is not None:
        return by_qual, cached_transitive

    transitive = _analysis_index_transitive_callers(
        analysis_index,
        project_root=project_root,
    )
    return by_qual, transitive


@dataclass(frozen=True)
class ConstantFlowDetail:
    path: Path
    qual: str
    name: str
    param: str
    value: str
    count: int
    sites: tuple[str, ...] = ()


def _compute_lint_lines(
    *,
    forest,
    groups_by_path,
    bundle_sites_by_path,
    type_callsite_evidence,
    ambiguity_witnesses,
    exception_obligations,
    never_invariants,
    deadline_obligations,
    decision_lint_lines,
    broad_type_lint_lines,
    constant_smells,
    unused_arg_smells,
    project_lint_rows_from_forest_fn = None,
):
    if project_lint_rows_from_forest_fn is None:
        project_lint_rows_from_forest_fn = _project_lint_rows_from_forest

    bundle_evidence = _collect_bundle_evidence_lines(
        forest=forest,
        groups_by_path=groups_by_path,
        bundle_sites_by_path=bundle_sites_by_path,
    )
    bundle_lint_lines = _lint_lines_from_bundle_evidence(bundle_evidence)
    type_lint_lines = _lint_lines_from_type_evidence(type_callsite_evidence)
    ambiguity_lint_lines = _lint_lines_from_call_ambiguities(ambiguity_witnesses)
    exception_lint_lines = _exception_protocol_lint_lines(exception_obligations)
    never_lint_lines = _never_invariant_lint_lines(never_invariants)
    deadline_lint_lines = _deadline_lint_lines(deadline_obligations)
    constant_lint_lines = _lint_lines_from_constant_smells(constant_smells)
    unused_arg_lint_lines = _lint_lines_from_unused_arg_smells(unused_arg_smells)

    lint_rows: list[dict[str, JSONValue]] = []
    lint_rows.extend(_lint_rows_from_lines(bundle_lint_lines, source="bundle_evidence"))
    lint_rows.extend(_lint_rows_from_lines(type_lint_lines, source="type_evidence"))
    lint_rows.extend(
        _lint_rows_from_lines(ambiguity_lint_lines, source="ambiguity_witnesses")
    )
    lint_rows.extend(
        _lint_rows_from_lines(exception_lint_lines, source="exception_obligations")
    )
    lint_rows.extend(_lint_rows_from_lines(never_lint_lines, source="never_invariants"))
    lint_rows.extend(
        _lint_rows_from_lines(deadline_lint_lines, source="deadline_obligations")
    )
    lint_rows.extend(_lint_rows_from_lines(decision_lint_lines, source="decision_surfaces"))
    lint_rows.extend(_lint_rows_from_lines(broad_type_lint_lines, source="broad_type"))
    lint_rows.extend(_lint_rows_from_lines(constant_lint_lines, source="constant_smells"))
    lint_rows.extend(
        _lint_rows_from_lines(unused_arg_lint_lines, source="unused_arg_smells")
    )

    _materialize_lint_rows(forest=forest, rows=lint_rows)
    projected = project_lint_rows_from_forest_fn(forest=forest)
    if not projected:
        return []

    rendered: list[str] = []
    for row in projected:
        check_deadline()
        path = str(row.get("path", "") or "")
        code = str(row.get("code", "") or "")
        message = str(row.get("message", "") or "")
        if not path or not code:
            continue
        try:
            line = int(row.get("line", 1) or 1)
            col = int(row.get("col", 1) or 1)
        except (TypeError, ValueError):
            continue
        rendered.append(_lint_line(path, line, col, code, message))
    return rendered


def _deadline_lint_lines(entries: list[JSONObject]) -> list[str]:
    check_deadline()
    lines: list[str] = []
    for entry in entries:
        check_deadline()
        site = mapping_or_none(entry.get("site")) or {}
        path = str(site.get("path", "") or "")
        line, col = _span_line_col(entry.get("span"))
        if not path:
            continue
        status = entry.get("status", "UNKNOWN")
        kind = entry.get("kind", "?")
        detail = entry.get("detail", "")
        message = f"{status} {kind} {detail}".strip()
        lines.append(_lint_line(path, line or 1, col or 1, "GABION_DEADLINE", message))
    return lines


def _collect_bundle_evidence_lines(
    *,
    forest,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    bundle_sites_by_path: dict[Path, dict[str, list[list[JSONObject]]]],
) -> list[str]:
    check_deadline()
    if not groups_by_path or not _has_bundles(groups_by_path):
        return []
    file_paths = _iter_monotonic_paths(
        groups_by_path.keys(),
        source="_collect_bundle_evidence_lines.groups_by_path",
    )
    projection = _bundle_projection_from_forest(forest, file_paths=file_paths)
    components = _connected_components(projection.nodes, projection.adj)
    bundle_site_index = _bundle_site_index(groups_by_path, bundle_sites_by_path)
    evidence_lines: list[str] = []
    for component in components:
        check_deadline()
        evidence = _render_component_callsite_evidence(
            component=component,
            nodes=projection.nodes,
            bundle_map=projection.bundle_map,
            bundle_counts=projection.bundle_counts,
            adj=projection.adj,
            documented_by_path=projection.documented_by_path,
            declared_global=projection.declared_global,
            bundle_site_index=bundle_site_index,
            root=projection.root,
            path_lookup=projection.path_lookup,
        )
        evidence_lines.extend(evidence)
    return evidence_lines


def _parse_lint_remainder(remainder: str) -> tuple[str, str]:
    text = remainder.strip()
    if not text:
        return ("GABION_UNKNOWN", "")
    head, *tail = text.split(maxsplit=1)
    code = head.strip() or "GABION_UNKNOWN"
    message = tail[0].strip() if tail else ""
    return (code, message)


def _lint_rows_from_lines(
    lines: Iterable[str],
    *,
    source: str,
) -> list[dict[str, JSONValue]]:
    check_deadline()
    rows: list[dict[str, JSONValue]] = []
    for line in lines:
        check_deadline()
        parsed = _parse_lint_location(line)
        if parsed is not None:
            path, lineno, col, remainder = parsed
            code, message = _parse_lint_remainder(remainder)
            rows.append(
                {
                    "path": path,
                    "line": int(lineno),
                    "col": int(col),
                    "code": code,
                    "message": message,
                    "source": source,
                }
            )
    return rows


def _materialize_lint_rows(
    *,
    forest,
    rows: Iterable[Mapping[str, JSONValue]],
) -> None:
    check_deadline()
    for row in rows:
        check_deadline()
        path = str(row.get("path", "") or "")
        if not path:
            continue
        try:
            lineno = int(row.get("line", 1) or 1)
            col = int(row.get("col", 1) or 1)
        except (TypeError, ValueError):
            continue
        code = str(row.get("code", "") or "")
        if not code:
            continue
        message = str(row.get("message", "") or "")
        source = str(row.get("source", "") or "")
        lint_node = forest.add_node(
            "LintFinding",
            (
                path,
                lineno,
                col,
                code,
                message,
            ),
            meta={
                "path": path,
                "line": lineno,
                "col": col,
                "code": code,
                "message": message,
            },
        )
        file_node = forest.add_file_site(path)
        forest.add_alt(
            "LintFinding",
            (file_node, lint_node),
            evidence={"source": source},
        )


def _lint_relation_from_forest(forest) -> list[dict[str, JSONValue]]:
    check_deadline()
    by_identity: dict[tuple[str, int, int, str, str], set[str]] = {}
    for alt in forest.alts:
        check_deadline()
        if alt.kind != "LintFinding" or len(alt.inputs) < 2:
            continue
        lint_node_id = alt.inputs[1]
        if lint_node_id.kind != "LintFinding":
            continue
        lint_node = forest.nodes.get(lint_node_id)
        if lint_node is not None:
            path = str(lint_node.meta.get("path", "") or "")
            code = str(lint_node.meta.get("code", "") or "")
            message = str(lint_node.meta.get("message", "") or "")
            if not path or not code:
                continue
            try:
                line = int(lint_node.meta.get("line", 1) or 1)
                col = int(lint_node.meta.get("col", 1) or 1)
            except (TypeError, ValueError):
                continue
            key = (path, line, col, code, message)
            source = str(alt.evidence.get("source", "") or "")
            bucket = by_identity.setdefault(key, set())
            if source:
                bucket.add(source)
    relation: list[dict[str, JSONValue]] = []
    for key in sort_once(by_identity, source="dataflow_lint_helpers._lint_relation"):
        check_deadline()
        path, line, col, code, message = key
        relation.append(
            {
                "path": path,
                "line": line,
                "col": col,
                "code": code,
                "message": message,
                "sources": sort_once(
                    by_identity[key],
                    source="dataflow_lint_helpers._lint_relation.sources",
                ),
            }
        )
    return relation


def _project_lint_rows_from_forest(
    *,
    forest,
    relation_fn: Callable[[object], list[dict[str, JSONValue]]] = _lint_relation_from_forest,
    apply_spec_fn: Callable[
        [ProjectionSpec, list[dict[str, JSONValue]]],
        list[dict[str, JSONValue]],
    ] = apply_spec,
) -> list[dict[str, JSONValue]]:
    relation = relation_fn(forest)
    if not relation:
        return []
    projected = apply_spec_fn(LINT_FINDINGS_SPEC, relation)

    def _row_to_file_site(row: Mapping[str, JSONValue]):
        path = str(row.get("path", "") or "")
        if not path:
            return None
        return forest.add_file_site(path)

    _materialize_projection_spec_rows(
        spec=LINT_FINDINGS_SPEC,
        projected=projected,
        forest=forest,
        row_to_site=_row_to_file_site,
    )
    return projected


def _never_sort_key(entry: JSONObject) -> tuple:
    status = str(entry.get("status", "UNKNOWN"))
    order = _NEVER_STATUS_ORDER.get(status, 3)
    site = mapping_or_empty(entry.get("site"))
    path = str(site.get("path", ""))
    function = str(site.get("function", ""))
    span = entry.get("span")
    line = -1
    col = -1
    span_entries = sequence_or_none(span)
    if span_entries is not None and len(span_entries) == 4:
        try:
            line = int(span_entries[0])
            col = int(span_entries[1])
        except (TypeError, ValueError):
            line = -1
            col = -1
    return (order, path, function, line, col, str(entry.get("never_id", "")))


def _never_invariant_lint_lines(entries: list[JSONObject]) -> list[str]:
    check_deadline()
    lines: list[str] = []
    for entry in sort_once(
        entries,
        key=_never_sort_key,
        source="dataflow_lint_helpers._never_invariant_lint_lines",
    ):
        check_deadline()
        status = entry.get("status", "UNKNOWN")
        span_entries = sequence_or_none(entry.get("span"))
        if status != "PROVEN_UNREACHABLE" and span_entries is not None and len(span_entries) == 4:
            site = mapping_or_empty(entry.get("site"))
            path = str(site.get("path", "?"))
            reason = entry.get("reason") or ""
            witness_ref = entry.get("witness_ref")
            environment = entry.get("environment_ref")
            undecidable = entry.get("undecidable_reason") or ""
            line, col, _, _ = span_entries
            bits: list[str] = [f"status={status}"]
            if reason:
                bits.append(f"reason={reason}")
            if witness_ref:
                bits.append(f"witness={witness_ref}")
            if environment:
                bits.append(f"env={json.dumps(environment, sort_keys=False)}")
            if status == "OBLIGATION":
                if undecidable:
                    bits.append(f"why={undecidable}")
                else:
                    bits.append("why=no witness env available")
            message = f"never() invariant ({'; '.join(bits)})"
            lines.append(
                _lint_line(
                    path,
                    int(line) + 1,
                    int(col) + 1,
                    "GABION_NEVER_INVARIANT",
                    message,
                )
            )
    return lines


def _constant_smells_from_details(
    details: Iterable[ConstantFlowDetail],
) -> list[str]:
    check_deadline()
    smells: list[str] = []
    for detail in details:
        check_deadline()
        path_name = detail.path.name
        site_suffix = ""
        if detail.sites:
            sample = ", ".join(detail.sites[:3])
            site_suffix = f" (e.g. {sample})"
        smells.append(
            f"{path_name}:{detail.name}.{detail.param} only observed constant {detail.value} across {detail.count} non-test call(s){site_suffix}"
        )
    return sort_once(smells, source="dataflow_lint_helpers._constant_smells_from_details")


def _deadness_witnesses_from_constant_details(
    details: Iterable[ConstantFlowDetail],
    *,
    project_root,
) -> list[JSONObject]:
    check_deadline()
    witnesses: list[JSONObject] = []
    for detail in details:
        check_deadline()
        path_value = _normalize_snapshot_path(detail.path, project_root)
        predicate = f"{detail.param} != {detail.value}"
        core = [
            f"observed constant {detail.value} across {detail.count} non-test call(s)"
        ]
        deadness_id = f"deadness:{path_value}:{detail.name}:{detail.param}:{detail.value}"
        witnesses.append(
            {
                "deadness_id": deadness_id,
                "path": path_value,
                "function": detail.name,
                "bundle": [detail.param],
                "environment": {detail.param: detail.value},
                "predicate": predicate,
                "core": core,
                "result": "UNREACHABLE",
                "call_sites": list(detail.sites[:10]),
                "projection": (
                    f"{detail.name}.{detail.param} constant {detail.value} across "
                    f"{detail.count} non-test call(s)"
                ),
            }
        )
    return sort_once(
        witnesses,
        key=lambda entry: (
            str(entry.get("path", "")),
            str(entry.get("function", "")),
            ",".join(cast(list[str], entry.get("bundle", []))),
            str(entry.get("predicate", "")),
        ),
        source="dataflow_lint_helpers._deadness_witnesses_from_constant_details",
    )


def _exception_protocol_lint_lines(entries: list[JSONObject]) -> list[str]:
    check_deadline()
    lines: list[str] = []
    for entry in entries:
        check_deadline()
        if entry.get("protocol") != "never":
            continue
        if entry.get("status") == "DEAD":
            continue
        exception_id = str(entry.get("exception_path_id", ""))
        parsed = _parse_exception_path_id(exception_id)
        if not parsed:
            continue
        path, lineno, col = parsed
        exception_name = entry.get("exception_name") or "?"
        status = entry.get("status", "UNKNOWN")
        message = f"never-throw exception {exception_name} (status={status})"
        lines.append(_lint_line(path, lineno, col, "GABION_EXC_NEVER", message))
    return lines


@dataclass(frozen=True)
class _BroadTypeLintContext:
    paths: list[Path]
    project_root: object
    ignore_params: set[str]
    strictness: str
    external_filter: bool
    transparent_decorators: object
    parse_failure_witnesses: list[JSONObject]
    analysis_index: object


def _decision_param_lint_line(
    info,
    param: str,
    *,
    project_root,
    code: str,
    message: str,
):
    span = info.param_spans.get(param)
    if span is not None:
        path = _normalize_snapshot_path(info.path, project_root)
        line, col, _, _ = span
        return _lint_line(path, line + 1, col + 1, code, message)
    return None


def _internal_broad_type_lint_lines_indexed(context: _BroadTypeLintContext) -> list[str]:
    by_qual, transitive_callers = _analysis_index_by_qual_and_transitive_callers(
        analysis_index=context.analysis_index,
        project_root=context.project_root,
    )
    lines: list[str] = []
    for info in by_qual.values():
        check_deadline()
        if _is_test_path(info.path):
            continue
        caller_count = len(transitive_callers.get(info.qual, set()))
        if caller_count == 0:
            continue
        for param, annot in info.annots.items():
            check_deadline()
            if not _is_broad_internal_type(annot):
                continue
            message = (
                f"internal param '{param}' uses broad type '{annot}' "
                f"(internal callers: {caller_count})"
            )
            lint = _decision_param_lint_line(
                info,
                param,
                project_root=context.project_root,
                code="GABION_BROAD_TYPE",
                message=message,
            )
            if lint is not None:
                lines.append(lint)
    return sort_once(
        set(lines),
        source="dataflow_lint_helpers._internal_broad_type_lint_lines_indexed",
    )


def _internal_broad_type_lint_lines(
    paths,
    *,
    project_root,
    ignore_params,
    strictness,
    external_filter,
    transparent_decorators=None,
    parse_failure_witnesses,
    analysis_index=None,
):
    check_deadline()
    context = _BroadTypeLintContext(
        paths=list(paths),
        project_root=project_root,
        ignore_params=set(ignore_params),
        strictness=strictness,
        external_filter=external_filter,
        transparent_decorators=transparent_decorators,
        parse_failure_witnesses=list(parse_failure_witnesses),
        analysis_index=analysis_index,
    )
    return _internal_broad_type_lint_lines_indexed(context)


_BROAD_SCALAR_TYPES = {
    "str",
    "int",
    "float",
    "bool",
    "bytes",
    "bytearray",
    "complex",
}

_NONE_TYPES = {"None", "NoneType", "type(None)"}


def _is_node_id_type(value: str) -> bool:
    return value == "NodeId" or value.endswith(".NodeId")


def _is_literal_type(value: str) -> bool:
    return value.startswith("Literal[")


def _split_top_level(value: str, sep: str) -> list[str]:
    check_deadline()
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in value:
        check_deadline()
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth = max(depth - 1, 0)
        if ch == sep and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _strip_type(value: str) -> str:
    return value.strip()


def _expand_type_hint(hint: str) -> set[str]:
    hint = hint.strip()
    if not hint:
        return set()
    if hint.startswith("Optional[") and hint.endswith("]"):
        inner = hint[len("Optional[") : -1]
        return {_strip_type(t) for t in _split_top_level(inner, ",")} | {"None"}
    if hint.startswith("Union[") and hint.endswith("]"):
        inner = hint[len("Union[") : -1]
        return {_strip_type(t) for t in _split_top_level(inner, ",")}
    if "|" in hint:
        return {_strip_type(t) for t in _split_top_level(hint, "|")}
    return {hint}


def _normalize_type_name(value: str) -> str:
    value = value.strip()
    if value.startswith("typing."):
        value = value[len("typing.") :]
    if value.startswith("builtins."):
        value = value[len("builtins.") :]
    return value


def _is_broad_internal_type(annot) -> bool:
    if annot is None:
        return False
    normalized = str(annot).replace("typing.", "")
    expanded = {_normalize_type_name(t) for t in _expand_type_hint(normalized)}
    non_none = {t for t in expanded if t not in _NONE_TYPES}
    if not non_none:
        return False
    if all(_is_node_id_type(t) for t in non_none):
        return False
    if any(_is_literal_type(t) for t in non_none):
        return True
    if "Any" in non_none or "object" in non_none:
        return True
    if _BROAD_SCALAR_TYPES & non_none:
        return True
    return False


def _span_line_col(span):
    parsed = int_tuple4_or_none(span)
    if parsed is None:
        return None, None
    return parsed[0] + 1, parsed[1] + 1


def _lint_lines_from_bundle_evidence(evidence: Iterable[str]) -> list[str]:
    return _ds_lint_lines_from_bundle_evidence(
        evidence,
        check_deadline=check_deadline,
        lint_line=_lint_line,
    )


def _lint_lines_from_type_evidence(evidence: Iterable[str]) -> list[str]:
    return _ds_lint_lines_from_type_evidence(
        evidence,
        check_deadline=check_deadline,
        lint_line=_lint_line,
    )


def _lint_lines_from_call_ambiguities(entries: Iterable[JSONObject]) -> list[str]:
    check_deadline()
    lines: list[str] = []
    for entry in entries:
        check_deadline()
        if type(entry) is not dict:
            continue
        entry_payload = cast(Mapping[str, Any], entry)
        site = entry_payload.get("site", {})
        if type(site) is not dict:
            continue
        site_mapping = cast(Mapping[str, Any], site)
        path = str(site_mapping.get("path", "") or "")
        if not path:
            continue
        lineno, col = _span_line_col(site_mapping.get("span"))
        candidate_count = entry_payload.get("candidate_count")
        try:
            count_value = int(candidate_count) if candidate_count is not None else 0
        except (TypeError, ValueError):
            count_value = 0
        kind = str(entry_payload.get("kind", "") or "ambiguity")
        message = f"{kind} candidates={count_value}"
        lines.append(_lint_line(path, lineno or 1, col or 1, "GABION_AMBIGUITY", message))
    return lines


def _lint_lines_from_constant_smells(smells: Iterable[str]) -> list[str]:
    return _ds_lint_lines_from_constant_smells(
        smells,
        check_deadline=check_deadline,
        lint_line=_lint_line,
    )


def _lint_lines_from_unused_arg_smells(smells: Iterable[str]) -> list[str]:
    return _ds_lint_lines_from_unused_arg_smells(
        smells,
        check_deadline=check_deadline,
        lint_line=_lint_line,
    )


def _merge_counts_by_knobs(*args, **kwargs):
    return _merge_counts_by_knobs_impl(*args, **kwargs)


def _parse_exception_path_id(value: str):
    parts = value.split(":", 5)
    if len(parts) != 6:
        return None
    path = parts[0]
    try:
        lineno = int(parts[3])
        col = int(parts[4])
    except ValueError:
        return None
    return path, lineno, col


def _parse_lint_location(*args, **kwargs):
    return _ds_parse_lint_location(*args, **kwargs)


def _lint_line(path: str, line: int, col: int, code: str, message: str) -> str:
    return f"{path}:{line}:{col}: {code} {message}".strip()


__all__ = [
    "_compute_lint_lines",
    "_constant_smells_from_details",
    "_deadness_witnesses_from_constant_details",
    "_exception_protocol_lint_lines",
    "_internal_broad_type_lint_lines",
    "_internal_broad_type_lint_lines_indexed",
    "_is_broad_internal_type",
    "_lint_lines_from_bundle_evidence",
    "_lint_lines_from_call_ambiguities",
    "_lint_lines_from_constant_smells",
    "_lint_lines_from_type_evidence",
    "_lint_lines_from_unused_arg_smells",
    "_merge_counts_by_knobs",
    "_normalize_type_name",
    "_parse_exception_path_id",
    "_parse_lint_location",
]
