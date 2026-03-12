from __future__ import annotations

import ast
import os
from collections.abc import Callable, Iterable, Mapping
from functools import cache
from pathlib import Path
from typing import cast

from gabion.analysis.dataflow.engine.dataflow_contracts import InvariantProposition
from gabion.analysis.dataflow.engine.dataflow_decision_surfaces import (
    summarize_coherence_witnesses as _ds_summarize_coherence_witnesses, summarize_deadness_witnesses as _ds_summarize_deadness_witnesses, summarize_rewrite_plans as _ds_summarize_rewrite_plans)
from gabion.analysis.dataflow.io.dataflow_graph_rendering import (
    bundle_projection_from_forest as _bundle_projection_from_forest, bundle_site_index as _bundle_site_index, connected_components as _connected_components, has_bundles as _has_bundles, render_component_callsite_evidence as _render_component_callsite_evidence, render_mermaid_component as _render_mermaid_component)
from gabion.analysis.indexed_scan.scanners.report_sections import (
    spec_row_span as _spec_row_span_impl)
from gabion.analysis.dataflow.io.dataflow_parse_helpers import _ParseModuleStage, _parse_module_tree
from gabion.analysis.dataflow.io.dataflow_snapshot_io import (
    report_section_marker as _report_section_marker)
from gabion.analysis.foundation.json_types import JSONObject, JSONValue
from gabion.analysis.projection.projection_exec import apply_execution_ops
from gabion.analysis.projection.projection_exec_plan import execution_ops_from_spec
from gabion.analysis.projection.projection_normalize import spec_hash as projection_spec_hash
from gabion.analysis.projection.projection_registry import REPORT_SECTION_LINES_SPEC
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.order_contract import sort_once
from gabion.runtime_shape_dispatch import (
    json_list_optional as _json_list_optional,
    json_mapping_optional as _json_mapping_optional,
    str_optional as _str_optional,
)
from gabion.invariants import decision_protocol, never

_FORBID_RAW_SORTED_ENV = "GABION_FORBID_RAW_SORTED"


@decision_protocol
@cache
def _report_section_lines_execution_ops():
    return execution_ops_from_spec(REPORT_SECTION_LINES_SPEC)


def parse_witness_contract_violations() -> list[str]:
    # Contract checks are now handled in dedicated audit/test paths.
    return []


def _raw_sorted_baseline_key(path: Path) -> str:
    parts = path.parts
    if "src" in parts:
        start = parts.index("src")
        return str(Path(*parts[start:]))
    return str(path)


def _iter_paths(paths: Iterable[Path]) -> Iterable[Path]:
    return sort_once(paths, key=lambda path: str(path), source="_iter_paths")


def _raw_sorted_callsite_counts(
    paths: Iterable[Path],
    *,
    parse_failure_witnesses: list[JSONObject],
) -> dict[str, list[tuple[int, int]]]:
    counts: dict[str, list[tuple[int, int]]] = {}
    for path in _iter_paths(paths):
        check_deadline()
        if path.suffix == ".py":
            outcome = _parse_module_tree(
                path,
                stage=_ParseModuleStage.RAW_SORTED_AUDIT,
                parse_failure_witnesses=parse_failure_witnesses,
            )
            if outcome.kind == "parsed":
                locations: list[tuple[int, int]] = []
                for node in ast.walk(outcome.tree):
                    check_deadline()
                    match node:
                        case ast.Call(func=ast.Name(id="sorted")) as call_node:
                            line = int(getattr(call_node, "lineno", 1))
                            col = int(getattr(call_node, "col_offset", 0)) + 1
                            locations.append((line, col))
                if locations:
                    counts[_raw_sorted_baseline_key(path)] = locations
    return counts


def raw_sorted_contract_violations(
    file_paths,
    *,
    parse_failure_witnesses: list[JSONObject],
    strict_forbid = None,
    baseline_counts = None,
) -> list[str]:
    counts = _raw_sorted_callsite_counts(
        file_paths,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    strict_forbid_flag = (
        os.environ.get(_FORBID_RAW_SORTED_ENV) == "1"
        if strict_forbid is None
        else bool(strict_forbid)
    )
    baseline_counts_map: Mapping[str, int] = (
        {} if baseline_counts is None else baseline_counts
    )
    violations: list[str] = []
    for path in sort_once(
        counts,
        source="raw_sorted_contract_violations.counts",
    ):
        check_deadline()
        current = len(counts[path])
        if strict_forbid_flag:
            for line, col in counts[path]:
                check_deadline()
                violations.append(
                    f"{path}:{line}:{col} order_contract raw sorted() forbidden; use sort_once(...)"
                )
        else:
            baseline = baseline_counts_map.get(path)
            if baseline is None:
                violations.append(
                    f"{path} order_contract raw_sorted introduced count={current} baseline=0"
                )
            elif current > baseline:
                violations.append(
                    f"{path} order_contract raw_sorted exceeded baseline current={current} baseline={baseline}"
                )
    return violations


def report_section_marker(section_id: str) -> str:
    return _report_section_marker(section_id)


def _materialize_projection_spec_rows(
    *,
    spec: ProjectionSpec,
    projected: Iterable[Mapping[str, JSONValue]],
    forest,
    row_to_site: Callable[[Mapping[str, JSONValue]], object],
) -> None:
    spec_identity = projection_spec_hash(spec)
    spec_site = forest.add_spec_site(
        spec_hash=spec_identity,
        spec_name=str(spec.name),
        spec_domain=str(spec.domain),
        spec_version=int(spec.spec_version) if spec.spec_version else None,
    )
    for row in projected:
        check_deadline()
        site_id = row_to_site(row)
        if site_id is not None:
            evidence: dict[str, object] = {
                "spec_name": str(spec.name),
                "spec_hash": spec_identity,
            }
            for key, value in row.items():
                check_deadline()
                evidence[str(key)] = value
            forest.add_alt("SpecFacet", (spec_site, site_id), evidence=evidence)


def _materialize_report_section_lines(
    *,
    forest,
    section_key,
    lines: Iterable[str],
) -> None:
    check_deadline()
    report_file = forest.add_file_site("<report>")
    for idx, text in enumerate(lines):
        check_deadline()
        text_value = str(text)
        line_node = forest.add_node(
            "ReportSectionLine",
            (section_key.run_id, section_key.section, idx, text_value),
            meta={
                "run_id": section_key.run_id,
                "section": section_key.section,
                "line_index": idx,
                "text": text_value,
            },
        )
        forest.add_alt(
            "ReportSectionLine",
            (report_file, line_node),
            evidence={
                "run_id": section_key.run_id,
                "section": section_key.section,
            },
        )


def _report_section_line_relation(
    *,
    forest,
    section_key,
) -> list[dict[str, JSONValue]]:
    check_deadline()
    relation: list[dict[str, JSONValue]] = []
    for alt in forest.alts:
        check_deadline()
        if alt.kind == "ReportSectionLine" and len(alt.inputs) >= 2:
            if (
                str(alt.evidence.get("run_id", "") or "") == section_key.run_id
                and str(alt.evidence.get("section", "") or "") == section_key.section
            ):
                node_id = alt.inputs[1]
                if node_id.kind == "ReportSectionLine":
                    node = forest.nodes.get(node_id)
                    if node is not None:
                        node_run_id = str(node.meta.get("run_id", "") or "")
                        node_section = str(node.meta.get("section", "") or "")
                        if (
                            node_run_id == section_key.run_id
                            and node_section == section_key.section
                        ):
                            try:
                                line_index = int(node.meta.get("line_index", 0) or 0)
                            except (TypeError, ValueError):
                                line_index = -1
                            if line_index >= 0:
                                relation.append(
                                    {
                                        "section": section_key.section,
                                        "line_index": line_index,
                                        "text": str(node.meta.get("text", "") or ""),
                                    }
                                )
    return relation


def project_report_section_lines(
    *,
    forest,
    section_key,
    lines: Iterable[str],
) -> list[str]:
    check_deadline()
    _materialize_report_section_lines(
        forest=forest,
        section_key=section_key,
        lines=lines,
    )
    relation = _report_section_line_relation(
        forest=forest,
        section_key=section_key,
    )
    if not relation:
        return []
    projected = apply_execution_ops(_report_section_lines_execution_ops(), relation)
    _materialize_projection_spec_rows(
        spec=REPORT_SECTION_LINES_SPEC,
        projected=projected,
        forest=forest,
        row_to_site=lambda _row: forest.add_file_site("<report>"),
    )
    rendered: list[str] = []
    for row in projected:
        check_deadline()
        rendered.append(str(row.get("text", "") or ""))
    return rendered


def has_bundles(groups_by_path) -> bool:
    return _has_bundles(groups_by_path)


def bundle_projection_from_forest(forest, *, file_paths):
    return _bundle_projection_from_forest(
        forest,
        file_paths=file_paths,
    )


def connected_components(nodes, adj):
    return _connected_components(nodes, adj)


def bundle_site_index(groups_by_path, bundle_sites_by_path):
    return _bundle_site_index(groups_by_path, bundle_sites_by_path)


def render_mermaid_component(
    nodes,
    bundle_map,
    bundle_counts,
    adj,
    component,
    declared_global,
    declared_by_path,
    documented_by_path,
) -> tuple[str, str]:
    return _render_mermaid_component(
        nodes,
        bundle_map,
        bundle_counts,
        adj,
        component,
        declared_global,
        declared_by_path,
        documented_by_path,
    )


def render_component_callsite_evidence(
    *,
    component,
    nodes,
    bundle_map,
    bundle_counts,
    adj,
    documented_by_path,
    declared_global,
    bundle_site_index,
    root,
    path_lookup,
) -> list[str]:
    return _render_component_callsite_evidence(
        component=component,
        nodes=nodes,
        bundle_map=bundle_map,
        bundle_counts=bundle_counts,
        adj=adj,
        documented_by_path=documented_by_path,
        declared_global=declared_global,
        bundle_site_index=bundle_site_index,
        root=root,
        path_lookup=path_lookup,
    )


def summarize_deadness_witnesses(entries: list[JSONObject]) -> list[str]:
    return _ds_summarize_deadness_witnesses(entries, check_deadline=check_deadline)


def summarize_coherence_witnesses(entries: list[JSONObject]) -> list[str]:
    return _ds_summarize_coherence_witnesses(entries, check_deadline=check_deadline)


def summarize_rewrite_plans(entries: list[JSONObject]) -> list[str]:
    return _ds_summarize_rewrite_plans(entries, check_deadline=check_deadline)


def _format_span_fields(
    line: object,
    col: object,
    end_line: object,
    end_col: object,
) -> str:
    try:
        line_value = int(line)
        col_value = int(col)
        end_line_value = int(end_line)
        end_col_value = int(end_col)
    except (TypeError, ValueError):
        return ""
    if (
        line_value < 0
        or col_value < 0
        or end_line_value < 0
        or end_col_value < 0
    ):
        return ""
    return (
        f"{line_value + 1}:{col_value + 1}-"
        f"{end_line_value + 1}:{end_col_value + 1}"
    )


def _span4(value: object) -> tuple[object, object, object, object]:
    match value:
        case [a, b, c, d, *_]:
            return (a, b, c, d)
    return (-1, -1, -1, -1)
def summarize_never_invariants(entries: list[JSONObject]) -> list[str]:
    check_deadline()
    if not entries:
        return []
    lines: list[str] = []
    for entry in sort_once(
        entries,
        key=lambda row: (
            str(row.get("status", "")),
            str((row.get("site") or {}).get("path", "")),
            str((row.get("site") or {}).get("function", "")),
            str(row.get("never_id", "")),
        ),
        source="summarize_never_invariants",
    ):
        check_deadline()
        site = cast(Mapping[str, JSONValue], entry.get("site", {}) or {})
        path = str(site.get("path", "?") or "?")
        function = str(site.get("function", "?") or "?")
        suite_kind = str(site.get("suite_kind", "?") or "?")
        span = _format_span_fields(*_span4(entry.get("span")))
        status = str(entry.get("status", "UNKNOWN") or "UNKNOWN")
        reason = str(entry.get("reason", "") or "")
        marker_kind = str(entry.get("marker_kind", "never") or "never")
        suffix = f"@{span}" if span else ""
        bits = [f"status={status}"]
        if reason:
            bits.append(f"reason={reason}")
        lines.append(
            f"{path}:{function}[{suite_kind}]{suffix} "
            f"{marker_kind}() ({'; '.join(bits)})"
        )
    return lines


def summarize_call_ambiguities(entries: list[JSONObject]) -> list[str]:
    check_deadline()
    if not entries:
        return []
    counts: dict[str, int] = {}
    lines: list[str] = ["Counts by witness kind:"]
    normalized: list[JSONObject] = []
    for entry in entries:
        check_deadline()
        kind = str(entry.get("kind", "") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
        normalized.append(entry)
    for kind in sort_once(counts, source="summarize_call_ambiguities.counts"):
        check_deadline()
        lines.append(f"- {kind}: {counts[kind]}")
    lines.append("Top ambiguous sites:")
    for entry in sort_once(
        normalized,
        key=lambda row: (
            str((row.get("site") or {}).get("path", "")),
            str((row.get("site") or {}).get("function", "")),
            int(row.get("candidate_count", 0) or 0),
        ),
        source="summarize_call_ambiguities.entries",
    )[:20]:
        check_deadline()
        site = cast(Mapping[str, JSONValue], entry.get("site", {}) or {})
        path = str(site.get("path", "?") or "?")
        function = str(site.get("function", "?") or "?")
        span = _format_span_fields(*_span4(site.get("span")))
        suffix = f"@{span}" if span else ""
        count = int(entry.get("candidate_count", 0) or 0)
        lines.append(f"- {path}:{function}{suffix} candidates={count}")
    return lines


def summarize_exception_obligations(entries: list[JSONObject]) -> list[str]:
    check_deadline()
    if not entries:
        return []
    lines: list[str] = []
    for entry in entries[:10]:
        check_deadline()
        site = cast(Mapping[str, JSONValue], entry.get("site", {}) or {})
        path = site.get("path", "?")
        function = site.get("function", "?")
        bundle = site.get("bundle", [])
        status = entry.get("status", "UNKNOWN")
        source = entry.get("source_kind", "?")
        exception_name = entry.get("exception_name")
        protocol = entry.get("protocol")
        reason_code = entry.get("handledness_reason_code", "UNKNOWN_REASON")
        refinement = str(entry.get("type_refinement_opportunity", "") or "")
        suffix = f" reason={reason_code}"
        if exception_name:
            suffix += f" exception={exception_name}"
        if protocol:
            suffix += f" protocol={protocol}"
        if refinement:
            suffix += f" refine={refinement}"
        lines.append(
            f"{path}:{function} bundle={bundle} source={source} status={status}{suffix}"
        )
    if len(entries) > 10:
        lines.append(f"... {len(entries) - 10} more")
    return lines


def summarize_handledness_witnesses(entries: list[JSONObject]) -> list[str]:
    check_deadline()
    if not entries:
        return []
    lines: list[str] = []
    for entry in entries[:10]:
        check_deadline()
        site = cast(Mapping[str, JSONValue], entry.get("site", {}) or {})
        path = site.get("path", "?")
        function = site.get("function", "?")
        bundle = site.get("bundle", [])
        handler = entry.get("handler_boundary", "?")
        result = entry.get("result", "UNKNOWN")
        reason_code = entry.get("handledness_reason_code", "UNKNOWN_REASON")
        refinement = str(entry.get("type_refinement_opportunity", "") or "")
        suffix = f" reason={reason_code}"
        if refinement:
            suffix += f" refine={refinement}"
        lines.append(
            f"{path}:{function} bundle={bundle} handler={handler} result={result}{suffix}"
        )
    if len(entries) > 10:
        lines.append(f"... {len(entries) - 10} more")
    return lines


def summarize_deadline_obligations(entries: list[JSONObject], *, forest) -> list[str]:
    del forest
    check_deadline()
    if not entries:
        return []
    lines: list[str] = []
    for entry in entries[:20]:
        check_deadline()
        site = cast(Mapping[str, JSONValue], entry.get("site", {}) or {})
        path = site.get("path", "?")
        function = site.get("function", "?")
        span = _format_span_fields(*_span4(entry.get("span")))
        status = entry.get("status", "UNKNOWN")
        kind = entry.get("kind", "?")
        detail = entry.get("detail", "")
        suffix = f"@{span}" if span else ""
        lines.append(
            f"{path}:{function}{suffix} status={status} kind={kind} {detail}".strip()
        )
    if len(entries) > 20:
        lines.append(f"... {len(entries) - 20} more")
    return lines


def summarize_runtime_obligations(entries: list[JSONObject]) -> list[str]:
    check_deadline()
    if not entries:
        return []
    lines: list[str] = []
    ordered = sort_once(
        entries,
        key=lambda entry: (
            str(entry.get("status", "")),
            str(entry.get("contract", "")),
            str(entry.get("kind", "")),
            str(entry.get("section_id", "")),
            str(entry.get("detail", "")),
        ),
        source="summarize_runtime_obligations",
    )
    for entry in ordered[:50]:
        check_deadline()
        status = str(entry.get("status", "OBLIGATION"))
        contract = str(entry.get("contract", "runtime_contract"))
        kind = str(entry.get("kind", "unknown"))
        section_id = entry.get("section_id")
        phase = entry.get("phase")
        detail = str(entry.get("detail", "")).strip()
        section_part = ""
        section_text = _str_optional(section_id)
        if section_text:
            section_part = f" section={section_id}"
        phase_part = ""
        phase_text = _str_optional(phase)
        if phase_text:
            phase_part = f" phase={phase}"
        line = f"{status} {contract} {kind}{section_part}{phase_part}".strip()
        if detail:
            line = f"{line} detail={detail}"
        lines.append(line)
    if len(ordered) > 50:
        lines.append(f"... {len(ordered) - 50} more")
    return lines


def summarize_parse_failure_witnesses(entries: list[JSONObject]) -> list[str]:
    check_deadline()
    if not entries:
        return []
    lines: list[str] = []
    ordered = sort_once(
        entries,
        key=lambda entry: (
            str(entry.get("path", "")),
            str(entry.get("stage", "")),
            str(entry.get("error_type", "")),
            str(entry.get("error", "")),
        ),
        source="summarize_parse_failure_witnesses",
    )
    for entry in ordered[:25]:
        check_deadline()
        path = str(entry.get("path", "?"))
        stage = str(entry.get("stage", "?"))
        error_type = str(entry.get("error_type", "Error"))
        error = str(entry.get("error", "")).strip()
        if error:
            lines.append(f"{path} stage={stage} {error_type}: {error}")
        else:
            lines.append(f"{path} stage={stage} {error_type}")
    if len(ordered) > 25:
        lines.append(f"... {len(ordered) - 25} more")
    return lines


def _str_tuple_from_sequence(value: object) -> tuple[str, ...]:
    entries = _json_list_optional(value)
    if entries is None:
        return tuple()
    return tuple(str(item) for item in entries)


def summarize_fingerprint_provenance(entries: list[JSONObject]) -> list[str]:
    check_deadline()
    if not entries:
        return []
    grouped: dict[tuple[object, ...], list[JSONObject]] = {}
    for entry in entries:
        check_deadline()
        matches = _str_tuple_from_sequence(entry.get("glossary_matches")) or tuple()
        if matches:
            key = ("glossary", matches)
        else:
            base_keys = _str_tuple_from_sequence(entry.get("base_keys")) or tuple()
            ctor_keys = _str_tuple_from_sequence(entry.get("ctor_keys")) or tuple()
            key = ("types", base_keys, ctor_keys)
        grouped.setdefault(key, []).append(entry)
    lines: list[str] = []
    grouped_entries = sort_once(
        grouped.items(),
        source="summarize_fingerprint_provenance.grouped",
        key=lambda item: (-len(item[1]), item[0]),
    )
    for key, group in grouped_entries[:20]:
        check_deadline()
        if key and key[0] == "glossary":
            label = "glossary=" + ", ".join(cast(tuple[str, ...], key[1]))
        else:
            base_keys = list(cast(tuple[str, ...], key[1]))
            ctor_keys = list(cast(tuple[str, ...], key[2]))
            label = f"base={base_keys}"
            if ctor_keys:
                label += f" ctor={ctor_keys}"
        lines.append(f"- {label} occurrences={len(group)}")
        for entry in group[:3]:
            check_deadline()
            path = entry.get("path")
            fn_name = entry.get("function")
            bundle = entry.get("bundle")
            lines.append(f"  - {path}:{fn_name} bundle={bundle}")
        if len(group) > 3:
            lines.append(f"  - ... ({len(group) - 3} more)")
    return lines


def _format_invariant_proposition(prop: InvariantProposition) -> str:
    if prop.form == "Equal" and len(prop.terms) == 2:
        rendered = f"{prop.terms[0]} == {prop.terms[1]}"
    else:
        rendered = f"{prop.form}({', '.join(prop.terms)})"
    prefix = f"{prop.scope}: " if prop.scope else ""
    suffix = f" [{prop.source}]" if prop.source else ""
    return f"{prefix}{rendered}{suffix}"


def format_invariant_propositions(entries) -> list[str]:
    return [_format_invariant_proposition(prop) for prop in entries]


def exception_protocol_evidence(entries: list[JSONObject]) -> list[str]:
    check_deadline()
    lines: list[str] = []
    for entry in entries:
        check_deadline()
        if entry.get("protocol") != "never":
            continue
        exception_id = entry.get("exception_path_id", "?")
        exception_name = entry.get("exception_name") or "?"
        status = entry.get("status", "UNKNOWN")
        lines.append(
            f"{exception_id} exception={exception_name} protocol=never status={status}"
        )
    return lines


def exception_protocol_warnings(entries: list[JSONObject]) -> list[str]:
    check_deadline()
    warnings: list[str] = []
    for entry in entries:
        check_deadline()
        if entry.get("protocol") != "never":
            continue
        if entry.get("status") == "DEAD":
            continue
        site = cast(Mapping[str, JSONValue], entry.get("site", {}) or {})
        path = site.get("path", "?")
        function = site.get("function", "?")
        exception_name = entry.get("exception_name") or "?"
        status = entry.get("status", "UNKNOWN")
        warnings.append(
            f"{path}:{function} raises {exception_name} (protocol=never, status={status})"
        )
    return warnings


def parse_failure_violation_lines(entries: list[JSONObject]) -> list[str]:
    check_deadline()
    if not entries:
        return []
    lines: list[str] = []
    for entry in sort_once(
        entries,
        key=lambda item: (
            str(item.get("path", "")),
            str(item.get("stage", "")),
            str(item.get("error_type", "")),
            str(item.get("error", "")),
        ),
        source="parse_failure_violation_lines",
    ):
        check_deadline()
        path = str(entry.get("path", "?"))
        stage = str(entry.get("stage", "?"))
        error_type = str(entry.get("error_type", "Error"))
        error = str(entry.get("error", "")).strip()
        if error:
            lines.append(f"{path} parse_failure stage={stage} {error_type}: {error}")
        else:
            lines.append(f"{path} parse_failure stage={stage} {error_type}")
    return lines


def runtime_obligation_violation_lines(entries: list[JSONObject]) -> list[str]:
    check_deadline()
    violations: list[str] = []
    for entry in sort_once(
        entries,
        key=lambda item: (
            str(item.get("contract", "")),
            str(item.get("kind", "")),
            str(item.get("section_id", "")),
            str(item.get("phase", "")),
            str(item.get("detail", "")),
        ),
        source="runtime_obligation_violation_lines",
    ):
        check_deadline()
        if str(entry.get("status", "")).upper() != "VIOLATION":
            continue
        contract = str(entry.get("contract", "runtime_contract"))
        kind = str(entry.get("kind", "unknown"))
        section_id = entry.get("section_id")
        phase = entry.get("phase")
        detail = str(entry.get("detail", "")).strip()
        section_text = _str_optional(section_id)
        phase_text = _str_optional(phase)
        section_part = (
            f" section={section_text}"
            if section_text is not None and section_text
            else ""
        )
        phase_part = (
            f" phase={phase_text}"
            if phase_text is not None and phase_text
            else ""
        )
        text = f"{contract} {kind}{section_part}{phase_part}".strip()
        if detail:
            text = f"{text} detail={detail}"
        violations.append(text)
    return violations


def render_type_mermaid(
    suggestions: list[str],
    ambiguities: list[str],
) -> str:
    check_deadline()
    lines = ["```mermaid", "flowchart LR"]
    node_id = 0

    def _node(label: str) -> str:
        nonlocal node_id
        node_id += 1
        node = f"type_{node_id}"
        safe = label.replace('"', "'")
        lines.append(f'  {node}["{safe}"]')
        return node

    for entry in suggestions:
        check_deadline()
        if " can tighten to " not in entry:
            continue
        lhs, rhs = entry.split(" can tighten to ", 1)
        src = _node(lhs)
        dst = _node(rhs)
        lines.append(f"  {src} --> {dst}")
    for entry in ambiguities:
        check_deadline()
        if " downstream types conflict: " not in entry:
            continue
        lhs, rhs = entry.split(" downstream types conflict: ", 1)
        src = _node(lhs)
        rhs_value = rhs.strip()
        if rhs_value.startswith("[") and rhs_value.endswith("]"):
            rhs_value = rhs_value[1:-1]
        type_names: list[str] = []
        for item in rhs_value.split(","):
            check_deadline()
            item_value = item.strip()
            if not item_value:
                continue
            type_names.append(item_value.strip("'\""))
        for type_name in type_names:
            check_deadline()
            dst = _node(type_name)
            lines.append(f"  {src} -.-> {dst}")
    lines.append("```")
    return "\n".join(lines)

def _spec_row_span(row: Mapping[str, JSONValue]):
    return _spec_row_span_impl(row)
