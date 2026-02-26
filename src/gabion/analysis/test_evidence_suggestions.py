# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations
from gabion.analysis.timeout_context import check_deadline

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from gabion.analysis import evidence_keys
from gabion.analysis.baseline_io import load_json, parse_version
from gabion.analysis.aspf import Alt, Forest, NodeId
from gabion.analysis.json_types import JSONObject
from gabion.analysis.dataflow_audit import (
    AuditConfig,
    ClassInfo,
    FunctionInfo,
    ParentAnnotator,
    SymbolTable,
    _alt_input,
    _build_function_index,
    _build_symbol_table,
    _collect_class_index,
    _enclosing_scopes,
    _is_test_path,
    _iter_paths,
    _module_name,
    _paramset_key,
    _resolve_callee,
)
from gabion.invariants import require_not_none
from gabion.analysis.report_doc import ReportDoc
from gabion.order_contract import sort_once


GRAPH_SOURCE = "graph"
CALL_FOOTPRINT_FALLBACK_SOURCE = "graph.call_footprint_fallback"
HEURISTIC_SOURCE = "heuristic"
DEFAULT_MAX_DEPTH = 2

_ALT_EVIDENCE_PREFIX = {
    "DecisionSurface": "decision_surface/direct",
    "ValueDecisionSurface": "decision_surface/value_encoded",
    "NeverInvariantSink": "never/sink",
    "SignatureBundle": "paramset",
}
_NO_RESULT = None


@dataclass(frozen=True)
class TestEvidenceEntry:
    test_id: str
    file: str
    line: int
    evidence: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class EvidenceSuggestion:
    key: dict[str, object]
    display: str

    @property
    def identity(self) -> str:
        return evidence_keys.key_identity(self.key)


@dataclass(frozen=True)
class Suggestion:
    test_id: str
    file: str
    line: int
    suggested: tuple[EvidenceSuggestion, ...]
    matches: tuple[str, ...]
    source: str
    derived_from: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class SuggestionSummary:
    total: int
    suggested: int
    suggested_graph: int
    suggested_heuristic: int
    skipped_mapped: int
    skipped_no_match: int
    graph_unresolved: int
    unmapped_modules: tuple[tuple[str, int], ...]
    unmapped_prefixes: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _GraphSuggestion:
    suggested: tuple[EvidenceSuggestion, ...]
    source: str
    derived_from: tuple[dict[str, str], ...]


def load_test_evidence(path: str) -> list[TestEvidenceEntry]:
    check_deadline()
    payload = load_json(path)
    parse_version(
        payload,
        expected=(1, 2),
        field="schema_version",
        error_context="test evidence",
    )
    tests_payload = payload.get("tests", [])
    match tests_payload:
        case list() as tests:
            pass
        case _:
            tests = []
    if tests_payload is not tests:
        raise ValueError("test evidence payload is missing tests list")
    entries: list[TestEvidenceEntry] = []
    for entry in tests:
        match entry:
            case dict() as entry_data:
                test_id = str(entry_data.get("test_id", "") or "").strip()
                if test_id:
                    file_path = str(entry_data.get("file", "") or "").strip()
                    line = int(entry_data.get("line", 0) or 0)
                    evidence = _normalize_evidence_list(entry_data.get("evidence", []))
                    raw_status = entry_data.get("status")
                    status = str(raw_status).strip() if raw_status is not None else ""
                    if not status:
                        status = "mapped" if evidence else "unmapped"
                    entries.append(
                        TestEvidenceEntry(
                            test_id=test_id,
                            file=file_path,
                            line=line,
                            evidence=tuple(evidence),
                            status=status,
                        )
                    )
            case _:
                pass
    return sort_once(entries, key=lambda item: item.test_id, source = 'src/gabion/analysis/test_evidence_suggestions.py:145')


def suggest_evidence(
    entries: Iterable[TestEvidenceEntry],
    *,
    root=".",
    paths=None,
    forest: Forest,
    config=None,
    max_depth: int = DEFAULT_MAX_DEPTH,
    include_heuristics: bool = True,
    graph_suggestions_fn=None,
    suggest_for_entry_fn=None,
) -> tuple[list[Suggestion], SuggestionSummary]:
    check_deadline()
    # dataflow-bundle: entries, root, paths, forest, config
    suggestions: list[Suggestion] = []
    skipped_mapped = 0
    skipped_no_match = 0
    suggested_graph = 0
    suggested_heuristic = 0
    graph_unresolved = 0
    entry_list = sort_once(entries, key=lambda item: item.test_id, source = 'src/gabion/analysis/test_evidence_suggestions.py:337')
    total = len(entry_list)
    unmapped_entries = [entry for entry in entry_list if not entry.evidence]
    skipped_mapped = total - len(unmapped_entries)
    root_path = Path(root)
    graph_suggestions = graph_suggestions_fn or _graph_suggestions
    suggest_for_entry = suggest_for_entry_fn or _suggest_for_entry
    if unmapped_entries:
        graph_suggestions_map, graph_resolved = graph_suggestions(
            unmapped_entries,
            root=root_path,
            paths=paths,
            forest=forest,
            config=config,
            max_depth=max_depth,
        )
    else:
        graph_suggestions_map, graph_resolved = {}, set()
    for entry in unmapped_entries:
        if entry.test_id not in graph_resolved:
            graph_unresolved += 1
        graph_suggested = graph_suggestions_map.get(entry.test_id)
        if graph_suggested:
            suggestions.append(
                Suggestion(
                    test_id=entry.test_id,
                    file=entry.file,
                    line=entry.line,
                    suggested=graph_suggested.suggested,
                    matches=(),
                    source=graph_suggested.source,
                    derived_from=graph_suggested.derived_from,
                )
            )
            suggested_graph += 1
            continue
        if include_heuristics and entry.test_id not in graph_resolved:
            heuristic_suggested, matches = suggest_for_entry(entry)
            if heuristic_suggested:
                suggestions.append(
                    Suggestion(
                        test_id=entry.test_id,
                        file=entry.file,
                        line=entry.line,
                        suggested=tuple(heuristic_suggested),
                        matches=tuple(matches),
                        source=HEURISTIC_SOURCE,
                    )
                )
                suggested_heuristic += 1
                continue
        skipped_no_match += 1
    unmapped_modules, unmapped_prefixes = _summarize_unmapped(entry_list)
    summary = SuggestionSummary(
        total=total,
        suggested=len(suggestions),
        suggested_graph=suggested_graph,
        suggested_heuristic=suggested_heuristic,
        skipped_mapped=skipped_mapped,
        skipped_no_match=skipped_no_match,
        graph_unresolved=graph_unresolved,
        unmapped_modules=unmapped_modules,
        unmapped_prefixes=unmapped_prefixes,
    )
    return suggestions, summary


def render_markdown(
    suggestions: list[Suggestion],
    summary: SuggestionSummary,
) -> str:
    check_deadline()
    # dataflow-bundle: suggestions, summary
    doc = ReportDoc("out_test_evidence_suggestions")
    doc.section("Summary")
    doc.line(f"- total: {summary.total}")
    doc.line(f"- suggested: {summary.suggested}")
    doc.line(f"- suggested_graph: {summary.suggested_graph}")
    doc.line(f"- suggested_heuristic: {summary.suggested_heuristic}")
    doc.line(f"- skipped_mapped: {summary.skipped_mapped}")
    doc.line(f"- skipped_no_match: {summary.skipped_no_match}")
    doc.line(f"- graph_unresolved: {summary.graph_unresolved}")
    doc.line()
    doc.line("Top Unmapped Modules:")
    if summary.unmapped_modules:
        for module, count in summary.unmapped_modules:
            doc.line(f"- {module}: {count}")
    else:
        doc.line("- None")
    doc.line()
    doc.line("Top Unmapped Test Prefixes:")
    if summary.unmapped_prefixes:
        for prefix, count in summary.unmapped_prefixes:
            doc.line(f"- {prefix}: {count}")
    else:
        doc.line("- None")
    doc.line()
    doc.line("## Suggestions")
    if not suggestions:
        doc.line("- None")
        return doc.emit()

    for entry in sort_once(suggestions, key=lambda item: item.test_id, source = 'src/gabion/analysis/test_evidence_suggestions.py:262'):
        evidence_list = ", ".join(item.display for item in entry.suggested)
        details = [f"source: {entry.source}"]
        if entry.matches:
            details.append(f"matched: {', '.join(entry.matches)}")
        doc.line(
            f"- `{entry.test_id}` -> {evidence_list} ({'; '.join(details)})"
        )
    return doc.emit()


def render_json_payload(
    suggestions: list[Suggestion],
    summary: SuggestionSummary,
) -> dict[str, object]:
    # dataflow-bundle: suggestions, summary
    return {
        "version": 4,
        "summary": {
            "total": summary.total,
            "suggested": summary.suggested,
            "suggested_graph": summary.suggested_graph,
            "suggested_heuristic": summary.suggested_heuristic,
            "skipped_mapped": summary.skipped_mapped,
            "skipped_no_match": summary.skipped_no_match,
            "graph_unresolved": summary.graph_unresolved,
        },
        "unmapped_modules": [
            {"module": module, "count": count}
            for module, count in summary.unmapped_modules
        ],
        "unmapped_prefixes": [
            {"prefix": prefix, "count": count}
            for prefix, count in summary.unmapped_prefixes
        ],
        "suggestions": [
            {
                "test_id": entry.test_id,
                "file": entry.file,
                "line": entry.line,
                "suggested": [
                    {"key": item.key, "display": item.display}
                    for item in entry.suggested
                ],
                "matched": list(entry.matches),
                "source": entry.source,
                **(
                    {"derived_from": list(entry.derived_from)}
                    if entry.derived_from
                    else {}
                ),
            }
            for entry in sort_once(suggestions, key=lambda item: item.test_id, source = 'src/gabion/analysis/test_evidence_suggestions.py:314')
        ],
    }


def collect_call_footprints(
    entries: Iterable[TestEvidenceEntry],
    *,
    root=".",
    paths=None,
    config=None,
    iter_paths_fn=None,
    build_function_index_fn=None,
    build_symbol_table_fn=None,
    collect_class_index_fn=None,
    build_test_index_fn=None,
    resolve_callee_fn=None,
    collect_call_footprint_targets_fn=None,
) -> dict[str, tuple[dict[str, str], ...]]:
    check_deadline()
    # dataflow-bundle: entries, root, paths, config
    entry_list = sort_once(entries, key=lambda item: item.test_id, source = 'src/gabion/analysis/test_evidence_suggestions.py:337')
    if not entry_list:
        return {}
    root_path = Path(root)
    config = config or AuditConfig(project_root=root_path)
    project_root = config.project_root or root_path
    iter_paths = iter_paths_fn or _iter_paths
    build_function_index = build_function_index_fn or _build_function_index
    build_symbol_table = build_symbol_table_fn or _build_symbol_table
    collect_class_index = collect_class_index_fn or _collect_class_index
    build_test_index = build_test_index_fn or _build_test_index
    resolve_callee = resolve_callee_fn or _resolve_callee
    collect_call_footprint_targets = (
        collect_call_footprint_targets_fn or _collect_call_footprint_targets
    )
    path_list = iter_paths([str(p) for p in (paths or [root_path])], config)
    if not path_list:
        return {}
    parse_failure_witnesses: list[JSONObject] = []
    by_name, by_qual = build_function_index(
        path_list,
        project_root,
        config.ignore_params,
        config.strictness,
        config.transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    symbol_table = build_symbol_table(
        path_list,
        project_root,
        external_filter=config.external_filter,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    class_index = collect_class_index(
        path_list,
        project_root,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    test_index = build_test_index(by_qual, project_root)
    cache: dict[str, tuple[FunctionInfo, ...]] = {}
    node_cache: dict[Path, dict[tuple[tuple[str, ...], str], ast.AST]] = {}
    module_cache = {}

    def _resolved_callees(info: FunctionInfo) -> tuple[FunctionInfo, ...]:
        check_deadline()
        if info.qual in cache:
            return cache[info.qual]
        resolved_callees: dict[str, FunctionInfo] = {}
        for call in info.calls:
            callee = resolve_callee(
                call.callee,
                info,
                by_name,
                by_qual,
                symbol_table,
                project_root,
                class_index,
            )
            if callee is not None:
                resolved_callees[callee.qual] = callee
        ordered = tuple(resolved_callees[key] for key in sort_once(resolved_callees, source = 'src/gabion/analysis/test_evidence_suggestions.py:398'))
        cache[info.qual] = ordered
        return ordered

    footprints: dict[str, tuple[dict[str, str], ...]] = {}
    for entry in entry_list:
        info = test_index.get(entry.test_id)
        if info is not None:
            direct_callees = _resolved_callees(info)
            targets = collect_call_footprint_targets(
                info,
                entry=entry,
                direct_callees=direct_callees,
                node_cache=node_cache,
                module_cache=module_cache,
                symbol_table=symbol_table,
                by_name=by_name,
                by_qual=by_qual,
                class_index=class_index,
                project_root=project_root,
            )
            if targets:
                footprints[entry.test_id] = tuple(targets)
    return footprints


def _graph_suggestions(
    entries: Sequence[TestEvidenceEntry],
    *,
    root: Path,
    paths,
    forest: Forest,
    config,
    max_depth: int,
    iter_paths_fn=None,
    build_function_index_fn=None,
    build_symbol_table_fn=None,
    collect_class_index_fn=None,
    build_test_index_fn=None,
    build_forest_evidence_index_fn=None,
    resolve_callee_fn=None,
    collect_reachable_fn=None,
    collect_call_footprint_targets_fn=None,
) -> tuple[dict[str, _GraphSuggestion], set[str]]:
    check_deadline()
    # dataflow-bundle: entries, root, paths, forest, config
    if not entries:
        return {}, set()
    config = config or AuditConfig(project_root=root)
    project_root = config.project_root or root
    iter_paths = iter_paths_fn or _iter_paths
    build_function_index = build_function_index_fn or _build_function_index
    build_symbol_table = build_symbol_table_fn or _build_symbol_table
    collect_class_index = collect_class_index_fn or _collect_class_index
    build_test_index = build_test_index_fn or _build_test_index
    build_forest_evidence_index = build_forest_evidence_index_fn or _build_forest_evidence_index
    resolve_callee = resolve_callee_fn or _resolve_callee
    collect_reachable = collect_reachable_fn or _collect_reachable
    collect_call_footprint_targets = (
        collect_call_footprint_targets_fn or _collect_call_footprint_targets
    )
    path_list = iter_paths([str(p) for p in (paths or [root])], config)
    if not path_list:
        return {}, set()
    parse_failure_witnesses: list[JSONObject] = []
    by_name, by_qual = build_function_index(
        path_list,
        project_root,
        config.ignore_params,
        config.strictness,
        config.transparent_decorators,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    symbol_table = build_symbol_table(
        path_list,
        project_root,
        external_filter=config.external_filter,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    class_index = collect_class_index(
        path_list,
        project_root,
        parse_failure_witnesses=parse_failure_witnesses,
    )
    test_index = build_test_index(by_qual, project_root)
    site_index, evidence_by_site = build_forest_evidence_index(forest)
    resolved: set[str] = set()
    suggestions: dict[str, _GraphSuggestion] = {}
    cache: dict[str, tuple[FunctionInfo, ...]] = {}
    node_cache: dict[Path, dict[tuple[tuple[str, ...], str], ast.AST]] = {}
    module_cache = {}

    def _resolved_callees(info: FunctionInfo) -> tuple[FunctionInfo, ...]:
        check_deadline()
        if info.qual in cache:
            return cache[info.qual]
        resolved_callees: dict[str, FunctionInfo] = {}
        for call in info.calls:
            callee = resolve_callee(
                call.callee,
                info,
                by_name,
                by_qual,
                symbol_table,
                project_root,
                class_index,
            )
            if callee is not None:
                resolved_callees[callee.qual] = callee
        ordered = tuple(resolved_callees[key] for key in sort_once(resolved_callees, source = 'src/gabion/analysis/test_evidence_suggestions.py:514'))
        cache[info.qual] = ordered
        return ordered

    for entry in entries:
        info = test_index.get(entry.test_id)
        if info is not None:
            resolved.add(entry.test_id)
            direct_callees = _resolved_callees(info)
            reachable = collect_reachable(
                info,
                max_depth=max_depth,
                resolve_callees=_resolved_callees,
            )
            evidence_items: dict[str, EvidenceSuggestion] = {}
            for callee in reachable:
                if not _is_test_path(callee.path):
                    site_id = site_index.get((callee.path.name, callee.qual))
                    if site_id is not None:
                        for item in evidence_by_site.get(site_id, ()):
                            evidence_items[item.identity] = item
            if not evidence_items:
                targets = collect_call_footprint_targets(
                    info,
                    entry=entry,
                    direct_callees=direct_callees,
                    node_cache=node_cache,
                    module_cache=module_cache,
                    symbol_table=symbol_table,
                    by_name=by_name,
                    by_qual=by_qual,
                    class_index=class_index,
                    project_root=project_root,
                )
                if targets:
                    key = evidence_keys.make_call_footprint_key(
                        path=entry.file,
                        qual=_test_qual(entry.test_id),
                        targets=targets,
                    )
                    suggestion = EvidenceSuggestion(
                        key=key,
                        display=evidence_keys.render_display(key),
                    )
                    ordered = (suggestion,)
                    suggestions[entry.test_id] = _GraphSuggestion(
                        suggested=ordered,
                        source=CALL_FOOTPRINT_FALLBACK_SOURCE,
                        derived_from=tuple(targets),
                    )
            if evidence_items:
                ordered = tuple(evidence_items[key] for key in sort_once(evidence_items, source = 'src/gabion/analysis/test_evidence_suggestions.py:569'))
                suggestions[entry.test_id] = _GraphSuggestion(
                    suggested=ordered,
                    source=GRAPH_SOURCE,
                    derived_from=(),
                )
    return suggestions, resolved


def _collect_reachable(
    start: FunctionInfo,
    *,
    max_depth: int,
    resolve_callees: Callable[[FunctionInfo], Sequence[FunctionInfo]],
) -> list[FunctionInfo]:
    check_deadline()
    visited = {start.qual}
    frontier = [start]
    reachable: list[FunctionInfo] = []
    for _ in range(max_depth):
        if not frontier:
            break
        next_frontier: list[FunctionInfo] = []
        for info in sort_once(frontier, key=lambda item: item.qual, source = 'src/gabion/analysis/test_evidence_suggestions.py:592'):
            for callee in resolve_callees(info):
                if callee.qual in visited:
                    continue
                visited.add(callee.qual)
                reachable.append(callee)
                next_frontier.append(callee)
        frontier = next_frontier
    return reachable


def _collect_call_footprint_targets(
    info: FunctionInfo,
    *,
    entry: TestEvidenceEntry,
    direct_callees: Sequence[FunctionInfo],
    node_cache: dict[Path, dict[tuple[tuple[str, ...], str], ast.AST]],
    module_cache,
    symbol_table: SymbolTable,
    by_name: Mapping[str, Sequence[FunctionInfo]],
    by_qual: Mapping[str, FunctionInfo],
    class_index,
    project_root: Path,
) -> tuple[dict[str, str], ...]:
    targets = [
        {"path": str(callee.path.name), "qual": str(callee.qual)}
        for callee in direct_callees
    ]
    if targets:
        return tuple(targets)
    outer = _find_module_level_calls(
        info,
        entry=entry,
        node_cache=node_cache,
        module_cache=module_cache,
        symbol_table=symbol_table,
        by_name=by_name,
        by_qual=by_qual,
        class_index=class_index,
        project_root=project_root,
    )
    if not outer:
        return ()
    return tuple({"path": path, "qual": qual} for path, qual in outer)


def _find_module_level_calls(
    info: FunctionInfo,
    *,
    entry: TestEvidenceEntry,
    node_cache: dict[Path, dict[tuple[tuple[str, ...], str], ast.AST]],
    module_cache,
    symbol_table: SymbolTable,
    by_name: Mapping[str, Sequence[FunctionInfo]],
    by_qual: Mapping[str, FunctionInfo],
    class_index,
    project_root: Path,
) -> tuple[tuple[str, str], ...]:
    check_deadline()
    if not entry.file:
        return ()
    test_path = Path(entry.file)
    if not test_path.is_absolute():
        test_path = project_root / test_path
    if test_path not in node_cache:
        try:
            tree = ast.parse(test_path.read_text(encoding="utf-8"))
        except OSError:
            return ()
        parents = ParentAnnotator()
        parents.visit(tree)
        node_cache[test_path] = _index_nodes_by_scope(tree, parents.parents)
    nodes = node_cache.get(test_path, {})
    scopes = tuple(info.scope)
    node = nodes.get((scopes, info.name))
    if node is None:
        return ()
    module_name = _module_name(test_path, project_root)
    resolved: dict[str, tuple[str, str]] = {}
    for call in _iter_outer_calls(node):
        for callee_name in _call_symbol_refs(call):
            callee = _resolve_callee(
                callee_name,
                info,
                dict(by_name),
                by_qual,
                symbol_table,
                project_root,
                class_index,
            )
            if callee is None:
                resolved_module = _resolve_symbol_target(
                    callee_name,
                    module_name,
                    symbol_table,
                    module_cache,
                    project_root,
                )
                if resolved_module:
                    path, qual = resolved_module
                    resolved[f"{path}:{qual}"] = (path, qual)
            else:
                resolved[callee.qual] = (str(callee.path.name), str(callee.qual))
        for module_literal in _call_module_literals(call):
            resolved_module = _resolve_module_literal(module_literal, project_root, module_cache)
            if resolved_module:
                path, qual = resolved_module
                resolved[f"{path}:{qual}"] = (path, qual)
    ordered = [resolved[key] for key in sort_once(resolved, source = 'src/gabion/analysis/test_evidence_suggestions.py:700')]
    return tuple(ordered)


def _index_nodes_by_scope(
    tree: ast.AST, parents: dict[ast.AST, ast.AST]
) -> dict[tuple[tuple[str, ...], str], ast.AST]:
    check_deadline()
    index: dict[tuple[tuple[str, ...], str], ast.AST] = {}
    for node in ast.walk(tree):
        match node:
            case ast.FunctionDef() | ast.AsyncFunctionDef():
                scopes = tuple(_enclosing_scopes(node, parents))
                index[(scopes, node.name)] = node
            case _:
                pass
    return index


def _iter_outer_calls(node: ast.AST) -> list[ast.Call]:
    check_deadline()
    calls: list[ast.Call] = []
    stack = list(getattr(node, "body", ()))
    while stack:
        current = stack.pop()
        expand_children = True
        match current:
            case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.ClassDef():
                expand_children = False
            case ast.Call():
                calls.append(current)
            case _:
                pass
        if expand_children:
            stack.extend(ast.iter_child_nodes(current))
    return calls


def _call_symbol_refs(call: ast.Call) -> list[str]:
    check_deadline()
    refs: list[str] = []
    target = call.func
    match target:
        case ast.Name(id=name):
            refs.append(name)
        case ast.Attribute() as attr:
            name = _attribute_chain(attr)
            if name:
                refs.append(name)
        case _:
            pass
    for arg in call.args:
        name = _expr_symbol_ref(arg)
        if name:
            refs.append(name)
    for kw in call.keywords:
        match kw.value:
            case None:
                pass
            case value:
                name = _expr_symbol_ref(value)
                if name:
                    refs.append(name)
    return refs


def _call_module_literals(call: ast.Call) -> list[str]:
    check_deadline()
    values: list[str] = []
    for arg in call.args:
        literal = _module_literal(arg)
        if literal:
            values.append(literal)
    for kw in call.keywords:
        match kw.value:
            case None:
                pass
            case value:
                literal = _module_literal(value)
                if literal:
                    values.append(literal)
    return values


def _expr_symbol_ref(node: ast.AST):
    match node:
        case ast.Name(id=name):
            return name
        case ast.Attribute() as attr:
            return _attribute_chain(attr)
        case _:
            return _NO_RESULT


def _module_literal(node: ast.AST):
    match node:
        case ast.Constant(value=str() as value):
            return value.strip()
        case _:
            return _NO_RESULT


def _attribute_chain(node: ast.Attribute):
    check_deadline()
    parts: list[str] = []
    current: ast.AST = node
    while True:
        match current:
            case ast.Attribute(attr=attr, value=value):
                parts.append(attr)
                current = value
            case ast.Name(id=name):
                parts.append(name)
                return ".".join(reversed(parts))
            case _:
                return _NO_RESULT


def _resolve_symbol_target(
    callee_name,
    module_name,
    symbol_table,
    module_cache,
    project_root,
):
    if not callee_name:
        return _NO_RESULT
    if "." in callee_name:
        base, *rest = callee_name.split(".")
    else:
        base, rest = callee_name, []
    if (module_name, base) not in symbol_table.imports:
        return _NO_RESULT
    base_fqn = symbol_table.resolve(module_name, base)
    if base_fqn is None:
        return _NO_RESULT
    module_fqn = base_fqn
    module_path = _resolve_module_file(module_fqn, project_root, module_cache)
    if module_path is None and "." in base_fqn:
        module_fqn = base_fqn.rsplit(".", 1)[0]
        module_path = _resolve_module_file(module_fqn, project_root, module_cache)
    if module_path is None:
        return _NO_RESULT
    qual = base_fqn if not rest else base_fqn + "." + ".".join(rest)
    return (module_path.name, qual)


def _resolve_module_literal(
    value,
    project_root,
    module_cache,
):
    if not value or value.startswith("."):
        return _NO_RESULT
    if any(part.strip() == "" for part in value.split(".")):
        return _NO_RESULT
    module_path = _resolve_module_file(value, project_root, module_cache)
    if module_path is None:
        return _NO_RESULT
    return (module_path.name, value)


def _resolve_module_file(
    module_name,
    project_root,
    module_cache,
):
    check_deadline()
    module_path = module_cache.get(module_name)
    if module_name in module_cache:
        return module_path
    for base in (project_root, project_root / "src"):
        candidate = base / Path(*module_name.split("."))
        file_candidate = candidate.with_suffix(".py")
        if file_candidate.exists():
            module_cache[module_name] = file_candidate
            return file_candidate
        if candidate.is_dir():
            init_candidate = candidate / "__init__.py"
            if init_candidate.exists():
                module_cache[module_name] = init_candidate
                return init_candidate
    module_cache[module_name] = _NO_RESULT
    return _NO_RESULT


def _build_test_index(
    by_qual: Mapping[str, FunctionInfo],
    project_root,
) -> dict[str, FunctionInfo]:
    check_deadline()
    index: dict[str, FunctionInfo] = {}
    for info in by_qual.values():
        rel_path = _rel_path(info.path, project_root)
        scopes = list(info.scope)
        qualname = "::".join([*scopes, info.name]) if scopes else info.name
        test_id = f"{rel_path}::{qualname}"
        index[test_id] = info
    return index


def _rel_path(path: Path, project_root) -> str:
    if project_root is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except Exception:
        return str(path)


def _build_forest_evidence_index(
    forest: Forest,
) -> tuple[dict[tuple[str, str], NodeId], dict[NodeId, tuple[EvidenceSuggestion, ...]]]:
    check_deadline()
    site_index: dict[tuple[str, str], NodeId] = {}
    for node_id, node in forest.nodes.items():
        if node_id.kind != "FunctionSite":
            continue
        path, qual = _site_parts(node_id, forest)
        if not path or not qual:
            continue
        site_index[(path, qual)] = node_id

    evidence_by_site: dict[NodeId, dict[str, EvidenceSuggestion]] = defaultdict(dict)
    for alt in forest.alts:
        if alt.kind in _ALT_EVIDENCE_PREFIX:
            site_id = _alt_input(alt, "FunctionSite")
            if site_id is not None:
                suggestion = _evidence_for_alt(alt, forest)
                if suggestion is not None:
                    evidence_by_site[site_id][suggestion.identity] = suggestion
    ordered: dict[NodeId, tuple[EvidenceSuggestion, ...]] = {}
    for site_id, items in evidence_by_site.items():
        ordered[site_id] = tuple(items[key] for key in sort_once(items, source = 'src/gabion/analysis/test_evidence_suggestions.py:916'))
    return site_index, ordered


def _evidence_for_alt(
    alt,
    forest,
    *,
    prefix_map=None,
):
    prefix_map = prefix_map or _ALT_EVIDENCE_PREFIX
    prefix = prefix_map.get(alt.kind)
    if prefix is None:
        return _NO_RESULT
    paramset_id = _alt_input(alt, "ParamSet")
    if paramset_id is None:
        return _NO_RESULT
    paramset_key = _format_paramset(_paramset_key(forest, paramset_id))
    if not paramset_key:
        return _NO_RESULT
    if alt.kind == "SignatureBundle":
        key = evidence_keys.make_paramset_key(paramset_key.split(","))
        display = evidence_keys.render_display(key)
        return EvidenceSuggestion(key=key, display=display)
    site_id = _alt_input(alt, "FunctionSite")
    if site_id is None:
        return _NO_RESULT
    path, qual = _site_parts(site_id, forest)
    if not path or not qual:
        return _NO_RESULT
    if prefix == "decision_surface/direct":
        key = evidence_keys.make_decision_surface_key(
            mode="direct",
            path=path,
            qual=qual,
            param=paramset_key,
        )
    elif prefix == "decision_surface/value_encoded":
        key = evidence_keys.make_decision_surface_key(
            mode="value_encoded",
            path=path,
            qual=qual,
            param=paramset_key,
        )
    elif prefix == "never/sink":
        key = evidence_keys.make_never_sink_key(
            path=path,
            qual=qual,
            param=paramset_key,
        )
    else:
        return _NO_RESULT
    display = evidence_keys.render_display(key)
    return EvidenceSuggestion(key=key, display=display)


def _test_qual(test_id: str) -> str:
    if "::" in test_id:
        return test_id.split("::", 1)[1]
    return test_id


def _site_parts(node_id: NodeId, forest: Forest) -> tuple[str, str]:
    node = forest.nodes.get(node_id)
    path = ""
    qual = ""
    if node is not None:
        path = str(node.meta.get("path") or "")
        qual = str(node.meta.get("qual") or "")
    if not path and node_id.key:
        path = str(node_id.key[0])
    if not qual and len(node_id.key) > 1:
        qual = str(node_id.key[1])
    return path, qual


def _format_paramset(items: Sequence[str]) -> str:
    return ",".join(items)


def _suggest_for_entry(
    entry: TestEvidenceEntry,
    *,
    rules_fn=None,
    parse_display_fn=None,
    make_opaque_key_fn=None,
    normalize_key_fn=None,
    render_display_fn=None,
    is_opaque_fn=None,
) -> tuple[list[EvidenceSuggestion], list[str]]:
    check_deadline()
    file_haystack, name_haystack = _suggestion_haystack(entry)
    rules = (rules_fn or _suggestion_rules)()
    parse_display = parse_display_fn or evidence_keys.parse_display
    make_opaque_key = make_opaque_key_fn or evidence_keys.make_opaque_key
    normalize_key = normalize_key_fn or evidence_keys.normalize_key
    render_display = render_display_fn or evidence_keys.render_display
    is_opaque = is_opaque_fn or evidence_keys.is_opaque
    suggested: list[EvidenceSuggestion] = []
    matches: list[str] = []
    for rule in rules:
        if rule.matches(file=file_haystack, name=name_haystack):
            for display in rule.evidence:
                key = parse_display(display)
                if key is None:
                    key = make_opaque_key(display)
                key = normalize_key(key)
                rendered = render_display(key)
                if is_opaque(key):
                    rendered = display
                suggested.append(EvidenceSuggestion(key=key, display=rendered))
            matches.append(rule.rule_id)
    return _dedupe_suggestions(suggested), matches


def _dedupe_suggestions(items: list[EvidenceSuggestion]) -> list[EvidenceSuggestion]:
    check_deadline()
    seen: dict[str, EvidenceSuggestion] = {}
    for item in items:
        seen[item.identity] = item
    return [seen[key] for key in sort_once(seen, source = 'src/gabion/analysis/test_evidence_suggestions.py:1036')]


def _suggestion_haystack(entry: TestEvidenceEntry) -> tuple[str, str]:
    name = entry.test_id.split("::")[-1]
    file_haystack = entry.file.lower()
    name_haystack = name.lower()
    return file_haystack, name_haystack


@dataclass(frozen=True)
class _SuggestionRule:
    rule_id: str
    evidence: tuple[str, ...]
    needles: tuple[str, ...]
    scope: str = "both"
    exclude: tuple[str, ...] = ()

    def matches(self, *, file: str, name: str) -> bool:
        if self.scope == "file":
            haystack = file
        elif self.scope == "name":
            haystack = name
        else:
            haystack = f"{file} {name}"
        if self.exclude and any(needle in haystack for needle in self.exclude):
            return False
        return any(needle in haystack for needle in self.needles)


def _suggestion_rules() -> list[_SuggestionRule]:
    return [
        _SuggestionRule(
            rule_id="alias_invariance",
            evidence=("E:bundle/alias_invariance",),
            needles=("alias_", "rename_invariance"),
        ),
        _SuggestionRule(
            rule_id="baseline_ratchet",
            evidence=("E:baseline/ratchet_monotonicity",),
            needles=("baseline",),
        ),
        _SuggestionRule(
            rule_id="aspf_forest",
            evidence=("E:forest/packed_reuse", "E:forest/canonical_paramset"),
            needles=("aspf",),
        ),
        _SuggestionRule(
            rule_id="never_invariants",
            evidence=("E:never/sink_classification",),
            needles=("never_invariants",),
        ),
        _SuggestionRule(
            rule_id="fingerprint_registry",
            evidence=("E:fingerprint/registry_determinism",),
            needles=("type_fingerprints",),
            scope="file",
        ),
        _SuggestionRule(
            rule_id="fingerprint_provenance",
            evidence=("E:fingerprint/match_provenance",),
            needles=("provenance", "matches"),
            scope="name",
        ),
        _SuggestionRule(
            rule_id="fingerprint_rewrite_plan",
            evidence=("E:fingerprint/rewrite_plan_verification",),
            needles=("rewrite_plan", "rewrite_plans", "rewrite plan", "rewrite_plan_summary"),
            scope="name",
        ),
        _SuggestionRule(
            rule_id="schema_surfaces",
            evidence=("E:schema/anonymous_payload_surfaces",),
            needles=("schema_audit", "anonymous_schema", "schema_surfaces"),
            scope="both",
        ),
        _SuggestionRule(
            rule_id="cli_command_surface",
            evidence=("E:cli/command_surface_integrity",),
            needles=("test_cli_", "cli_helpers", "cli_commands", "cli_payloads"),
            scope="file",
        ),
        _SuggestionRule(
            rule_id="server_command_dispatch",
            evidence=("E:transport/server_command_dispatch",),
            needles=("server_execute", "execute_command"),
            scope="file",
        ),
        _SuggestionRule(
            rule_id="transport_payload_roundtrip",
            evidence=("E:transport/payload_roundtrip",),
            needles=("test_run",),
            scope="both",
            exclude=(
                "test_cli_",
                "cli_helpers",
                "cli_commands",
                "cli_payloads",
                "server_execute",
            ),
        ),
        _SuggestionRule(
            rule_id="decision_surface_direct",
            evidence=("E:decision_surface/direct",),
            needles=("decision_surface", "decision surfaces"),
            scope="name",
            exclude=("value_encoded", "value_decision", "value-encoded"),
        ),
        _SuggestionRule(
            rule_id="decision_surface_value_encoded",
            evidence=("E:decision_surface/value_encoded",),
            needles=("value_encoded", "value_decision", "value-encoded"),
            scope="name",
        ),
        _SuggestionRule(
            rule_id="decision_surface_rebranch",
            evidence=("E:decision_surface/rebranch_suggested",),
            needles=("rebranch", "value_rewrite", "value_rewrites", "decision_rewrite"),
            scope="name",
        ),
        _SuggestionRule(
            rule_id="contextvar_rewrite",
            evidence=("E:context/ambient_rewrite_suggested",),
            needles=("contextvar", "ambient"),
            scope="name",
        ),
        _SuggestionRule(
            rule_id="docflow_contract",
            evidence=("E:policy/docflow_contract",),
            needles=("docflow",),
            scope="name",
        ),
        _SuggestionRule(
            rule_id="policy_workflow_safety",
            evidence=("E:policy/workflow_safety",),
            needles=("policy", "policy_check", "workflow_safety"),
            scope="name",
        ),
    ]


def _normalize_evidence_list(value: object) -> list[str]:
    check_deadline()
    if value is None:
        return list()
    items: list[str] = []
    match value:
        case str() as item:
            items = [item]
        case list() | tuple() | set() as item_list:
            for item in item_list:
                match item:
                    case str() as display:
                        items.append(display)
                    case {"display": str() as display}:
                        items.append(display)
                    case _:
                        pass
        case _:
            return list()
    cleaned = [item.strip() for item in items if item.strip()]
    return sort_once(set(cleaned), source = 'src/gabion/analysis/test_evidence_suggestions.py:1195')
def _summarize_unmapped(
    entries: Iterable[TestEvidenceEntry],
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    check_deadline()
    # dataflow-bundle: entries
    module_counts: dict[str, int] = {}
    prefix_counts: dict[str, int] = {}
    for entry in entries:
        if entry.evidence:
            continue
        module_counts[entry.file] = module_counts.get(entry.file, 0) + 1
        prefix = _test_prefix(entry.test_id)
        if prefix:
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    modules = _top_counts(module_counts)
    prefixes = _top_counts(prefix_counts)
    return modules, prefixes


def _top_counts(source: dict[str, int], *, limit: int = 10) -> tuple[tuple[str, int], ...]:
    ordered = sort_once(source.items(), key=lambda item: (-item[1], item[0]), source = 'src/gabion/analysis/test_evidence_suggestions.py:1218')
    return tuple(ordered[:limit])


def _test_prefix(test_id: str) -> str:
    name = test_id.split("::")[-1]
    if not name.startswith("test_"):
        return ""
    parts = name.split("_")
    if len(parts) < 2 or not parts[1]:
        return "test"
    return f"test_{parts[1]}"
