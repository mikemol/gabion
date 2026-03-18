#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import singledispatch
import json
from pathlib import Path
import tomllib
from typing import Any, Iterator

from gabion.order_contract import ordered_or_sorted
from gabion.tooling.policy_substrate.invariant_marker_scan import (
    decorated_symbol_marker_index,
    scan_invariant_markers,
)


SRC_ROOT = Path("src") / "gabion"
SCRIPTS_ROOT = Path("scripts")

def _sorted[T](values: list[T], *, key=None) -> list[T]:
    return ordered_or_sorted(values, source="scripts.policy.symbol_activity_audit", key=key)
@dataclass(frozen=True)
class InvariantMarker:
    marker_kind: str
    valid: bool
    reason: str
    owner: str
    expiry: str
    reasoning_summary: str
    reasoning_control: str
    blocking_dependencies: tuple[str, ...]
    lifecycle_state: str
    missing_fields: tuple[str, ...]
    decorator: str


@dataclass(frozen=True)
class SymbolDef:
    module: str
    rel_path: str
    symbol: str
    kind: str
    lineno: int
    has_register_decorator: bool
    invariant_marker: InvariantMarker | None


@dataclass
class ModuleData:
    module: str
    rel_path: str
    imports: set[str]
    from_imports: list[tuple[str, str, str]]
    module_aliases: dict[str, str]
    defs: list[SymbolDef]
    load_names_value: set[str]
    load_names_type: set[str]
    call_names: set[str]
    call_arg_names: set[str]
    decorator_names: set[str]
    attr_refs_value: list[tuple[str, str]]
    attr_refs_type: list[tuple[str, str]]
    attr_calls: list[tuple[str, str]]
    attr_call_args: list[tuple[str, str]]
    attr_decorators: list[tuple[str, str]]


def _is_gabion_import_node(node: ast.AST) -> bool:
    return isinstance(node, (ast.Import, ast.ImportFrom))


def _has_duplicate_symbol_defs(item: tuple[tuple[str, str], list[SymbolDef]]) -> bool:
    return len(item[1]) > 1


def _is_cross_module_public_duplicate(item: tuple[str, list[SymbolDef]]) -> bool:
    return len({defn.module for defn in item[1]}) > 1


@singledispatch
def _symbol_defs_for_stmt(
    stmt: ast.stmt,
    *,
    module: str,
    rel_path: str,
    marker_index: dict[tuple[str, str, int], InvariantMarker],
) -> Iterator[SymbolDef]:
    return iter(())


def _symbol_def(
    *,
    module: str,
    rel_path: str,
    name: str,
    kind: str,
    lineno: int,
    decorator_list: list[ast.expr],
    marker_index: dict[tuple[str, str, int], InvariantMarker],
) -> SymbolDef:
    return SymbolDef(
        module=module,
        rel_path=rel_path,
        symbol=name,
        kind=kind,
        lineno=int(lineno),
        has_register_decorator=_has_register_decorator(decorator_list),
        invariant_marker=_invariant_marker_from_scan_entry(
            rel_path=rel_path,
            symbol=name,
            line=int(lineno),
            marker_index=marker_index,
        ),
    )


@_symbol_defs_for_stmt.register(ast.FunctionDef)
def _symbol_defs_for_function_stmt(
    stmt: ast.FunctionDef,
    *,
    module: str,
    rel_path: str,
    marker_index: dict[tuple[str, str, int], InvariantMarker],
) -> Iterator[SymbolDef]:
    yield _symbol_def(
        module=module,
        rel_path=rel_path,
        name=stmt.name,
        kind="function",
        lineno=stmt.lineno,
        decorator_list=stmt.decorator_list,
        marker_index=marker_index,
    )


@_symbol_defs_for_stmt.register(ast.AsyncFunctionDef)
def _symbol_defs_for_async_function_stmt(
    stmt: ast.AsyncFunctionDef,
    *,
    module: str,
    rel_path: str,
    marker_index: dict[tuple[str, str, int], InvariantMarker],
) -> Iterator[SymbolDef]:
    yield _symbol_def(
        module=module,
        rel_path=rel_path,
        name=stmt.name,
        kind="async_function",
        lineno=stmt.lineno,
        decorator_list=stmt.decorator_list,
        marker_index=marker_index,
    )


@_symbol_defs_for_stmt.register(ast.ClassDef)
def _symbol_defs_for_class_stmt(
    stmt: ast.ClassDef,
    *,
    module: str,
    rel_path: str,
    marker_index: dict[tuple[str, str, int], InvariantMarker],
) -> Iterator[SymbolDef]:
    yield _symbol_def(
        module=module,
        rel_path=rel_path,
        name=stmt.name,
        kind="class",
        lineno=stmt.lineno,
        decorator_list=stmt.decorator_list,
        marker_index=marker_index,
    )


class _SymbolUseAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.load_names_value: set[str] = set()
        self.load_names_type: set[str] = set()
        self.call_names: set[str] = set()
        self.call_arg_names: set[str] = set()
        self.decorator_names: set[str] = set()
        self.attr_refs_value: list[tuple[str, str]] = []
        self.attr_refs_type: list[tuple[str, str]] = []
        self.attr_calls: list[tuple[str, str]] = []
        self.attr_call_args: list[tuple[str, str]] = []
        self.attr_decorators: list[tuple[str, str]] = []
        self._annotation_depth = 0
        self._call_arg_depth = 0
        self._decorator_depth = 0

    def _visit_annotation(self, node: ast.AST | None) -> None:
        if node is None:
            return
        self._annotation_depth += 1
        self.visit(node)
        self._annotation_depth -= 1

    def _visit_decorator(self, node: ast.AST) -> None:
        self._decorator_depth += 1
        self.visit(node)
        self._decorator_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for decorator in node.decorator_list:
            self._visit_decorator(decorator)
        self._visit_annotation(node.returns)
        args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        for arg in args:
            self._visit_annotation(arg.annotation)
        if node.args.vararg is not None:
            self._visit_annotation(node.args.vararg.annotation)
        if node.args.kwarg is not None:
            self._visit_annotation(node.args.kwarg.annotation)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        for statement in node.body:
            self.visit(statement)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self._visit_decorator(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        for statement in node.body:
            self.visit(statement)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.target)
        self._visit_annotation(node.annotation)
        if node.value is not None:
            self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self.call_names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            self.attr_calls.append((node.func.value.id, node.func.attr))

        self.visit(node.func)
        self._call_arg_depth += 1
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self._call_arg_depth -= 1

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            return
        if self._annotation_depth > 0:
            self.load_names_type.add(node.id)
            return
        self.load_names_value.add(node.id)
        if self._call_arg_depth > 0:
            self.call_arg_names.add(node.id)
        if self._decorator_depth > 0:
            self.decorator_names.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name):
            pair = (node.value.id, node.attr)
            if self._annotation_depth > 0:
                self.attr_refs_type.append(pair)
            else:
                self.attr_refs_value.append(pair)
                if self._call_arg_depth > 0:
                    self.attr_call_args.append(pair)
                if self._decorator_depth > 0:
                    self.attr_decorators.append(pair)
        self.visit(node.value)
def _invariant_marker_from_scan_entry(
    *,
    rel_path: str,
    symbol: str,
    line: int,
    marker_index: dict[tuple[str, str, int], InvariantMarker],
) -> InvariantMarker | None:
    return marker_index.get((rel_path, symbol, line))


def _decorator_is_register(decorator: ast.expr) -> bool:
    match decorator:
        case ast.Call(func=ast.Attribute(attr="register")):
            return True
        case ast.Attribute(attr="register"):
            return True
        case _:
            return False


def _has_register_decorator(decorators: list[ast.expr]) -> bool:
    return any(_decorator_is_register(decorator) for decorator in decorators)


def _module_name(path: Path, root: Path) -> str:
    return ".".join(path.relative_to(root / "src").with_suffix("").parts)


def _resolve_from_import(*, module: str, level: int, imported_module: str | None) -> str:
    if level == 0:
        return imported_module or ""
    base = module.split(".")[:-level]
    if imported_module:
        base += imported_module.split(".")
    return ".".join(base)


def _iter_source_modules(root: Path) -> list[Path]:
    base = root / SRC_ROOT
    if not base.exists():
        return []
    return _sorted([path for path in base.rglob("*.py") if path.is_file()])


def _collect_module_data(*, root: Path) -> dict[str, ModuleData]:
    modules: dict[str, ModuleData] = {}
    marker_index = {
        key: InvariantMarker(
            marker_kind=node.marker_kind,
            valid=node.valid,
            reason=node.marker_reason,
            owner=node.owner,
            expiry=node.expiry,
            reasoning_summary=node.reasoning_summary,
            reasoning_control=node.reasoning_control,
            blocking_dependencies=node.blocking_dependencies,
            lifecycle_state=node.lifecycle_state,
            missing_fields=node.missing_fields,
            decorator=node.marker_name,
        )
        for key, node in decorated_symbol_marker_index(scan_invariant_markers(root)).items()
    }
    for path in _iter_source_modules(root):
        module = _module_name(path, root)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        rel_path = path.relative_to(root).as_posix()
        imports: set[str] = set()
        from_imports: list[tuple[str, str, str]] = []
        module_aliases: dict[str, str] = {}
        defs: list[SymbolDef] = []

        defs.extend(
            defn
            for node in tree.body
            for defn in _symbol_defs_for_stmt(
                node,
                module=module,
                rel_path=rel_path,
                marker_index=marker_index,
            )
        )

        for node in filter(_is_gabion_import_node, ast.walk(tree)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name.startswith("gabion"):
                        continue
                    imports.add(alias.name)
                    module_aliases[alias.asname or alias.name.split(".")[-1]] = alias.name
            elif isinstance(node, ast.ImportFrom):
                resolved = _resolve_from_import(
                    module=module,
                    level=node.level,
                    imported_module=node.module,
                )
                if not resolved.startswith("gabion"):
                    continue
                imports.add(resolved)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    from_imports.append((resolved, alias.name, alias.asname or alias.name))

        analyzer = _SymbolUseAnalyzer()
        analyzer.visit(tree)
        modules[module] = ModuleData(
            module=module,
            rel_path=rel_path,
            imports=imports,
            from_imports=from_imports,
            module_aliases=module_aliases,
            defs=defs,
            load_names_value=analyzer.load_names_value,
            load_names_type=analyzer.load_names_type,
            call_names=analyzer.call_names,
            call_arg_names=analyzer.call_arg_names,
            decorator_names=analyzer.decorator_names,
            attr_refs_value=analyzer.attr_refs_value,
            attr_refs_type=analyzer.attr_refs_type,
            attr_calls=analyzer.attr_calls,
            attr_call_args=analyzer.attr_call_args,
            attr_decorators=analyzer.attr_decorators,
        )
    return modules


def _known_module(raw: str, module_set: set[str]) -> str | None:
    if raw in module_set:
        return raw
    init_name = f"{raw}.__init__"
    if init_name in module_set:
        return init_name
    return None


def _roots_from_pyproject(*, root: Path, module_set: set[str]) -> set[str]:
    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.exists():
        return set()
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    scripts = payload.get("project", {}).get("scripts", {})
    if not isinstance(scripts, dict):
        return set()
    roots: set[str] = set()
    for target in scripts.values():
        if not isinstance(target, str):
            continue
        module_name = target.split(":", 1)[0]
        known = _known_module(module_name, module_set)
        if known is not None:
            roots.add(known)
    init_module = _known_module("gabion.__init__", module_set)
    if init_module is not None:
        roots.add(init_module)
    return roots


def _roots_from_scripts(*, root: Path, module_set: set[str]) -> set[str]:
    scripts_root = root / SCRIPTS_ROOT
    if not scripts_root.exists():
        return set()
    roots: set[str] = set()
    for path in _sorted([item for item in scripts_root.rglob("*.py") if item.is_file()]):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name.startswith("gabion"):
                        continue
                    known = _known_module(alias.name, module_set)
                    if known is not None:
                        roots.add(known)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                base = node.module or ""
                if not base.startswith("gabion"):
                    continue
                known_base = _known_module(base, module_set)
                if known_base is not None:
                    roots.add(known_base)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    known_symbol = _known_module(f"{base}.{alias.name}", module_set)
                    if known_symbol is not None:
                        roots.add(known_symbol)
    return roots


def _module_edges(*, modules: dict[str, ModuleData], module_set: set[str]) -> dict[str, set[str]]:
    edges: dict[str, set[str]] = {name: set() for name in module_set}
    for module_name, module_data in modules.items():
        for imported in module_data.imports:
            known = _known_module(imported, module_set)
            if known is not None:
                edges[module_name].add(known)
        for base, symbol, _alias in module_data.from_imports:
            known_symbol = _known_module(f"{base}.{symbol}", module_set)
            if known_symbol is not None:
                edges[module_name].add(known_symbol)
                continue
            known_base = _known_module(base, module_set)
            if known_base is not None:
                edges[module_name].add(known_base)
    return edges


def _reachable_modules(
    *,
    roots: set[str],
    edges: dict[str, set[str]],
) -> tuple[set[str], dict[str, str], dict[str, str | None]]:
    reachable: set[str] = set()
    root_owner: dict[str, str] = {}
    predecessor: dict[str, str | None] = {}
    queue: deque[tuple[str, str, str | None]] = deque((root, root, None) for root in _sorted(list(roots)))

    while queue:
        module_name, root_name, parent = queue.popleft()
        if module_name in reachable:
            continue
        reachable.add(module_name)
        root_owner[module_name] = root_name
        predecessor[module_name] = parent
        for target in _sorted(list(edges.get(module_name, set()))):
            if target not in reachable:
                queue.append((target, root_name, module_name))
    return reachable, root_owner, predecessor


def _trace_root_path(
    *,
    module_name: str,
    predecessor: dict[str, str | None],
) -> str:
    if module_name not in predecessor:
        return module_name
    parts: list[str] = []
    cursor: str | None = module_name
    while cursor is not None:
        parts.append(cursor)
        cursor = predecessor.get(cursor)
    parts.reverse()
    return " -> ".join(parts)


def _symbol_key(defn: SymbolDef) -> tuple[str, str]:
    return (defn.module, defn.symbol)


def _canonical_symbol_defs(modules: dict[str, ModuleData]) -> dict[tuple[str, str], SymbolDef]:
    canonical: dict[tuple[str, str], SymbolDef] = {}
    for module_data in modules.values():
        for defn in _sorted(module_data.defs, key=lambda item: item.lineno):
            canonical[_symbol_key(defn)] = defn
    return canonical


def _empty_finding(
    *,
    module: str,
    symbol: str,
    line: int,
    root_path: str,
    rel_path: str,
    ref_count_value: int = 0,
    ref_count_type: int = 0,
    call_count: int = 0,
) -> dict[str, Any]:
    return {
        "module": module,
        "symbol": symbol,
        "line": int(line),
        "path": rel_path,
        "root_path": root_path,
        "ref_count_value": int(ref_count_value),
        "ref_count_type": int(ref_count_type),
        "call_count": int(call_count),
    }


def _invariant_marker_payload(marker: InvariantMarker) -> dict[str, Any]:
    return {
        "marker_kind": marker.marker_kind,
        "decorator": marker.decorator,
        "reason": marker.reason,
        "owner": marker.owner,
        "expiry": marker.expiry,
        "reasoning": {
            "summary": marker.reasoning_summary,
            "control": marker.reasoning_control,
            "blocking_dependencies": list(marker.blocking_dependencies),
        },
        "lifecycle_state": marker.lifecycle_state,
        "valid": marker.valid,
        "missing_fields": list(marker.missing_fields),
    }


def _summarize_marked_finding(finding: dict[str, Any], marker: InvariantMarker) -> dict[str, Any]:
    payload = dict(finding)
    payload["invariant_marker"] = _invariant_marker_payload(marker)
    return payload


def _collect_symbol_activity(
    *,
    modules: dict[str, ModuleData],
    reachable: set[str],
    root_paths: dict[str, str],
    canonical_defs: dict[tuple[str, str], SymbolDef],
) -> dict[str, list[dict[str, Any]]]:
    value_refs: defaultdict[tuple[str, str], int] = defaultdict(int)
    type_refs: defaultdict[tuple[str, str], int] = defaultdict(int)
    call_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    dynamic_candidates: defaultdict[tuple[str, str], int] = defaultdict(int)

    for module_data in modules.values():
        for defn in module_data.defs:
            key = _symbol_key(defn)
            if key not in canonical_defs:
                continue
            if defn.symbol in module_data.load_names_value:
                value_refs[key] += 1
            if defn.symbol in module_data.load_names_type:
                type_refs[key] += 1
            if defn.symbol in module_data.call_names:
                call_counts[key] += 1
            if defn.symbol in module_data.call_arg_names or defn.symbol in module_data.decorator_names:
                dynamic_candidates[key] += 1

        for base, symbol, alias in module_data.from_imports:
            key = (base, symbol)
            if key not in canonical_defs:
                continue
            if alias in module_data.load_names_value:
                value_refs[key] += 1
            if alias in module_data.load_names_type:
                type_refs[key] += 1
            if alias in module_data.call_names:
                call_counts[key] += 1
            if alias in module_data.call_arg_names or alias in module_data.decorator_names:
                dynamic_candidates[key] += 1

        for alias, attr in module_data.attr_refs_value:
            source = module_data.module_aliases.get(alias)
            if source is None:
                continue
            key = (source, attr)
            if key in canonical_defs:
                value_refs[key] += 1
        for alias, attr in module_data.attr_refs_type:
            source = module_data.module_aliases.get(alias)
            if source is None:
                continue
            key = (source, attr)
            if key in canonical_defs:
                type_refs[key] += 1
        for alias, attr in module_data.attr_calls:
            source = module_data.module_aliases.get(alias)
            if source is None:
                continue
            key = (source, attr)
            if key in canonical_defs:
                call_counts[key] += 1
        for alias, attr in [*module_data.attr_call_args, *module_data.attr_decorators]:
            source = module_data.module_aliases.get(alias)
            if source is None:
                continue
            key = (source, attr)
            if key in canonical_defs:
                dynamic_candidates[key] += 1

    findings: dict[str, list[dict[str, Any]]] = {
        "never_referenced": [],
        "ref_not_invoked_value": [],
        "ref_not_invoked_type_only": [],
        "dynamic_dispatch_unresolved": [],
    }

    for key, defn in _sorted(
        list(canonical_defs.items()),
        key=lambda item: (item[0][0], item[1].lineno, item[0][1]),
    ):
        if defn.module not in reachable:
            continue
        value_count = int(value_refs[key])
        type_count = int(type_refs[key])
        call_count = int(call_counts[key])
        if defn.has_register_decorator:
            # Registration decorators are explicit runtime binding sites.
            call_count = max(1, call_count)
        dynamic_count = int(dynamic_candidates[key])
        finding = _empty_finding(
            module=defn.module,
            symbol=defn.symbol,
            line=defn.lineno,
            rel_path=defn.rel_path,
            root_path=root_paths.get(defn.module, defn.module),
            ref_count_value=value_count,
            ref_count_type=type_count,
            call_count=call_count,
        )
        if call_count > 0:
            continue
        if value_count == 0 and type_count == 0:
            findings["never_referenced"].append(finding)
            continue
        if value_count == 0 and type_count > 0:
            findings["ref_not_invoked_type_only"].append(finding)
            continue
        if dynamic_count > 0:
            finding["dynamic_candidate_count"] = dynamic_count
            findings["dynamic_dispatch_unresolved"].append(finding)
            continue
        findings["ref_not_invoked_value"].append(finding)

    for bucket in findings:
        findings[bucket] = _sorted(
            findings[bucket],
            key=lambda item: (
                str(item.get("module", "")),
                int(item.get("line", 0)),
                str(item.get("symbol", "")),
            ),
        )
    return findings


def _collect_duplicate_findings(
    *,
    modules: dict[str, ModuleData],
    root_paths: dict[str, str],
    canonical_defs: dict[tuple[str, str], SymbolDef],
) -> dict[str, list[dict[str, Any]]]:
    by_module_symbol: defaultdict[tuple[str, str], list[SymbolDef]] = defaultdict(list)
    public_symbol_defs: defaultdict[str, list[SymbolDef]] = defaultdict(list)
    for module_data in modules.values():
        for defn in module_data.defs:
            by_module_symbol[(defn.module, defn.symbol)].append(defn)
            if not defn.symbol.startswith("_"):
                public_symbol_defs[defn.symbol].append(defn)

    same_module: list[dict[str, Any]] = []
    duplicate_module_symbol_defs = filter(_has_duplicate_symbol_defs, by_module_symbol.items())
    for (module_name, symbol), defs in duplicate_module_symbol_defs:
        peer_lines = _sorted([int(defn.lineno) for defn in defs])
        for defn in _sorted(defs, key=lambda item: item.lineno):
            canonical = canonical_defs.get((defn.module, defn.symbol), defn)
            finding = _empty_finding(
                module=module_name,
                symbol=symbol,
                line=defn.lineno,
                rel_path=defn.rel_path,
                root_path=root_paths.get(module_name, module_name),
                ref_count_value=0,
                ref_count_type=0,
                call_count=0,
            )
            finding["duplicate_kind"] = "same_module"
            finding["peer_lines"] = peer_lines
            finding["canonical_line"] = int(canonical.lineno)
            same_module.append(finding)

    cross_module_public: list[dict[str, Any]] = []
    cross_module_public_defs = filter(_is_cross_module_public_duplicate, public_symbol_defs.items())
    for symbol, defs in cross_module_public_defs:
        peer_locations = _sorted(
            [f"{defn.module}:{defn.lineno}" for defn in defs]
        )
        for defn in _sorted(defs, key=lambda item: (item.module, item.lineno)):
            finding = _empty_finding(
                module=defn.module,
                symbol=symbol,
                line=defn.lineno,
                rel_path=defn.rel_path,
                root_path=root_paths.get(defn.module, defn.module),
            )
            finding["duplicate_kind"] = "cross_module_public"
            finding["peer_locations"] = peer_locations
            cross_module_public.append(finding)

    return {
        "same_module": _sorted(
            same_module,
            key=lambda item: (
                str(item.get("module", "")),
                str(item.get("symbol", "")),
                int(item.get("line", 0)),
            ),
        ),
        "cross_module_public": _sorted(
            cross_module_public,
            key=lambda item: (
                str(item.get("symbol", "")),
                str(item.get("module", "")),
                int(item.get("line", 0)),
            ),
        ),
    }


def _collect_unreachable_modules(
    *,
    modules: dict[str, ModuleData],
    reachable: set[str],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for module_name in _sorted([name for name in modules if name not in reachable]):
        module_data = modules[module_name]
        findings.append(
            {
                "module": module_name,
                "symbol": "",
                "line": 0,
                "path": module_data.rel_path,
                "root_path": "",
                "ref_count_value": 0,
                "ref_count_type": 0,
                "call_count": 0,
            }
        )
    return findings


def _index_defs_by_location(modules: dict[str, ModuleData]) -> dict[tuple[str, str, int], SymbolDef]:
    indexed: dict[tuple[str, str, int], SymbolDef] = {}
    for module_data in modules.values():
        for defn in module_data.defs:
            indexed[(defn.module, defn.symbol, defn.lineno)] = defn
    return indexed


def _marker_for_finding(
    *,
    finding: dict[str, Any],
    canonical_defs: dict[tuple[str, str], SymbolDef],
    indexed_defs: dict[tuple[str, str, int], SymbolDef],
) -> InvariantMarker | None:
    module_name = str(finding.get("module", ""))
    symbol = str(finding.get("symbol", ""))
    line = int(finding.get("line", 0) or 0)
    if not module_name or not symbol:
        return None
    direct = indexed_defs.get((module_name, symbol, line))
    if direct is not None and direct.invariant_marker is not None:
        return direct.invariant_marker
    canonical = canonical_defs.get((module_name, symbol))
    if canonical is not None:
        return canonical.invariant_marker
    return None


def _partition_suppressed(
    *,
    findings_by_bucket: dict[str, list[dict[str, Any]]],
    canonical_defs: dict[tuple[str, str], SymbolDef],
    indexed_defs: dict[tuple[str, str, int], SymbolDef],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    unsuppressed: dict[str, list[dict[str, Any]]] = {}
    blocked: dict[str, list[dict[str, Any]]] = {}
    for bucket, findings in findings_by_bucket.items():
        if bucket == "unreachable_modules":
            unsuppressed[bucket] = findings
            blocked[bucket] = []
            continue
        unsuppressed_bucket: list[dict[str, Any]] = []
        blocked_bucket: list[dict[str, Any]] = []
        for finding in findings:
            marker = _marker_for_finding(
                finding=finding,
                canonical_defs=canonical_defs,
                indexed_defs=indexed_defs,
            )
            if marker is not None and marker.valid and marker.marker_kind == "todo":
                blocked_bucket.append(_summarize_marked_finding(finding, marker))
            else:
                if marker is not None:
                    invalid = dict(finding)
                    invalid["invariant_marker"] = _invariant_marker_payload(marker)
                    unsuppressed_bucket.append(invalid)
                else:
                    unsuppressed_bucket.append(finding)
        unsuppressed[bucket] = _sorted(
            unsuppressed_bucket,
            key=lambda item: (
                str(item.get("module", "")),
                str(item.get("symbol", "")),
                int(item.get("line", 0)),
            ),
        )
        blocked[bucket] = _sorted(
            blocked_bucket,
            key=lambda item: (
                str(item.get("module", "")),
                str(item.get("symbol", "")),
                int(item.get("line", 0)),
            ),
        )
    return unsuppressed, blocked


def _counts_payload(
    *,
    findings: dict[str, list[dict[str, Any]]],
    blocked: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    bucket_names = _sorted(list(findings.keys()))
    by_bucket: dict[str, dict[str, int]] = {}
    unsuppressed_total = 0
    blocked_total = 0
    total = 0
    for bucket in bucket_names:
        unsuppressed_count = len(findings.get(bucket, []))
        blocked_count = len(blocked.get(bucket, []))
        bucket_total = unsuppressed_count + blocked_count
        by_bucket[bucket] = {
            "total": bucket_total,
            "unsuppressed": unsuppressed_count,
            "blocked_by_todo": blocked_count,
        }
        unsuppressed_total += unsuppressed_count
        blocked_total += blocked_count
        total += bucket_total
    return {
        "total_findings": total,
        "unsuppressed_total": unsuppressed_total,
        "blocked_by_todo_total": blocked_total,
        "by_bucket": by_bucket,
    }


def analyze(*, root: Path) -> dict[str, Any]:
    modules = _collect_module_data(root=root)
    module_names = set(modules.keys())
    runtime_roots = _roots_from_pyproject(root=root, module_set=module_names)
    script_roots = _roots_from_scripts(root=root, module_set=module_names)
    roots = runtime_roots | script_roots
    edges = _module_edges(modules=modules, module_set=module_names)
    reachable, root_owner, predecessor = _reachable_modules(roots=roots, edges=edges)

    root_paths = {
        module_name: _trace_root_path(module_name=module_name, predecessor=predecessor)
        for module_name in reachable
    }
    canonical_defs = _canonical_symbol_defs(modules)
    indexed_defs = _index_defs_by_location(modules)

    duplicate_findings = _collect_duplicate_findings(
        modules=modules,
        root_paths=root_paths,
        canonical_defs=canonical_defs,
    )
    activity_findings = _collect_symbol_activity(
        modules=modules,
        reachable=reachable,
        root_paths=root_paths,
        canonical_defs=canonical_defs,
    )
    unreachable_findings = _collect_unreachable_modules(modules=modules, reachable=reachable)

    combined_findings: dict[str, list[dict[str, Any]]] = {
        "unreachable_modules": unreachable_findings,
        "same_module": duplicate_findings["same_module"],
        "cross_module_public": duplicate_findings["cross_module_public"],
        "never_referenced": activity_findings["never_referenced"],
        "ref_not_invoked_value": activity_findings["ref_not_invoked_value"],
        "ref_not_invoked_type_only": activity_findings["ref_not_invoked_type_only"],
        "dynamic_dispatch_unresolved": activity_findings["dynamic_dispatch_unresolved"],
    }

    unsuppressed, blocked = _partition_suppressed(
        findings_by_bucket=combined_findings,
        canonical_defs=canonical_defs,
        indexed_defs=indexed_defs,
    )

    payload = {
        "format_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "root": str(root),
        "scope": {
            "active_roots": "runtime+scripts",
        },
        "roots": {
            "runtime": _sorted(list(runtime_roots)),
            "scripts": _sorted(list(script_roots)),
            "all": _sorted(list(roots)),
        },
        "module_reachability": {
            "reachable_count": len(reachable),
            "unreachable_count": len(module_names - reachable),
            "reachable_modules": _sorted(list(reachable)),
            "unreachable_modules": _sorted(list(module_names - reachable)),
            "root_owner": {module: root_owner[module] for module in _sorted(list(root_owner.keys()))},
            "root_path": {module: root_paths[module] for module in _sorted(list(root_paths.keys()))},
        },
        "counts": _counts_payload(findings=unsuppressed, blocked=blocked),
        "findings": unsuppressed,
        "blocked_by_todo": blocked,
    }
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Symbol Activity Audit",
        "",
        f"- root: `{payload.get('root', '')}`",
        f"- scope: `{payload.get('scope', {}).get('active_roots', '')}`",
        "",
    ]
    counts = payload.get("counts", {})
    by_bucket = counts.get("by_bucket", {})
    lines.append("## Counts")
    for bucket in _sorted(list(by_bucket.keys())):
        bucket_counts = by_bucket[bucket]
        lines.append(
            f"- {bucket}: total={bucket_counts.get('total', 0)} "
            f"unsuppressed={bucket_counts.get('unsuppressed', 0)} "
            f"blocked_by_todo={bucket_counts.get('blocked_by_todo', 0)}"
        )
    lines.append("")
    lines.append("## Unsuppressed Samples")
    for bucket in _sorted(list((payload.get("findings") or {}).keys())):
        items = (payload.get("findings") or {}).get(bucket, [])
        if not items:
            continue
        lines.append(f"### {bucket}")
        for item in items[:10]:
            module = item.get("module", "")
            symbol = item.get("symbol", "")
            line = item.get("line", 0)
            path = item.get("path", "")
            lines.append(f"- {path}:{line} {module}.{symbol}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run(
    *,
    root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
    check: bool = False,
) -> int:
    payload = analyze(root=root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_markdown(payload), encoding="utf-8")

    total = int((payload.get("counts") or {}).get("total_findings", 0) or 0)
    unsuppressed = int((payload.get("counts") or {}).get("unsuppressed_total", 0) or 0)
    blocked = int((payload.get("counts") or {}).get("blocked_by_todo_total", 0) or 0)
    print(
        "symbol-activity-audit: "
        f"total={total} unsuppressed={unsuppressed} blocked_by_todo={blocked} out={out_path}"
    )
    if check and unsuppressed > 0:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="artifacts/out/symbol_activity_audit.json")
    parser.add_argument("--markdown-out", default="")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    markdown_target = Path(args.markdown_out).resolve() if str(args.markdown_out).strip() else None
    return run(
        root=Path(args.root).resolve(),
        out_path=Path(args.out).resolve(),
        markdown_out=markdown_target,
        check=bool(args.check),
    )


if __name__ == "__main__":
    raise SystemExit(main())
