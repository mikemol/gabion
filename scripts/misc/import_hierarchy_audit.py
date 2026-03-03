#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SCAN_TOPS = ("src", "tests", "scripts", "in")
DEFAULT_TARGET_THRESHOLD = 20


@dataclass(frozen=True)
class FileRecord:
    path: str
    module: str
    is_package: bool
    source: str


@dataclass(frozen=True)
class DefinitionRecord:
    path: str
    module: str
    name: str
    qualname: str
    kind: str
    lineno: int


@dataclass(frozen=True)
class ImportBinding:
    alias: str
    kind: str
    target_module: str
    target_symbol: str | None
    imported_module: str | None


@dataclass(frozen=True)
class ImportLinkRecord:
    src: str
    dst: str | None
    src_scope: str
    import_kind: str
    imported_module: str | None
    imported_name: str | None
    binding_alias: str | None
    target_module: str | None
    target_symbol: str | None
    level: int
    lineno: int
    col_offset: int
    resolution: str


@dataclass(frozen=True)
class CallCandidate:
    src: str
    src_scope: str
    lineno: int
    col_offset: int
    callee_expr: str
    alias: str
    attr_chain: tuple[str, ...]
    binding: ImportBinding


@dataclass(frozen=True)
class CallLinkRecord:
    src: str
    dst: str | None
    src_scope: str
    lineno: int
    col_offset: int
    callee_expr: str
    alias: str
    attr_chain: list[str]
    binding_kind: str
    target_module: str | None
    target_symbol: str | None
    target_definition_kind: str | None
    resolution: str


@dataclass
class ScopeFrame:
    qualname: str
    frame_type: str
    bindings: dict[str, ImportBinding]


@dataclass(frozen=True)
class FileSymbolAnalysis:
    top_level_definitions: dict[str, str]
    definitions: list[DefinitionRecord]
    import_links: list[ImportLinkRecord]
    call_candidates: list[CallCandidate]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _iter_python_paths(root: Path, tops: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for top in tops:
        top_path = root / top
        if not top_path.exists():
            continue
        files.extend(top_path.rglob("*.py"))
    return sorted(
        (path for path in files if path.is_file()),
        key=lambda path: str(path.relative_to(root)).replace("\\", "/"),
    )


def _module_name_for_path(rel_path: str) -> tuple[str, bool]:
    is_package = rel_path.endswith("/__init__.py")
    stem = rel_path[:-3]
    parts = stem.split("/")
    if parts and parts[0] == "src":
        parts = parts[1:]
    if is_package:
        parts = parts[:-1]
    module = ".".join(part for part in parts if part)
    return module, is_package


def _load_workspace_records(root: Path, tops: Iterable[str]) -> list[FileRecord]:
    records: list[FileRecord] = []
    for path in _iter_python_paths(root, tops):
        rel_path = str(path.relative_to(root)).replace("\\", "/")
        module, is_package = _module_name_for_path(rel_path)
        source = path.read_text(encoding="utf-8")
        records.append(
            FileRecord(
                path=rel_path,
                module=module,
                is_package=is_package,
                source=source,
            )
        )
    return records


def _git_list_head_files(root: Path, tops: Iterable[str]) -> list[str]:
    cmd = ["git", "ls-tree", "-r", "--name-only", "HEAD", "--", *tops]
    proc = subprocess.run(
        cmd,
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return sorted(
        (
            line.strip()
            for line in proc.stdout.splitlines()
            if line.strip().endswith(".py")
        ),
    )


def _git_show_head_file(root: Path, rel_path: str) -> str:
    cmd = ["git", "show", f"HEAD:{rel_path}"]
    proc = subprocess.run(
        cmd,
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return proc.stdout


def _load_head_records(root: Path, tops: Iterable[str]) -> list[FileRecord]:
    records: list[FileRecord] = []
    for rel_path in _git_list_head_files(root, tops):
        module, is_package = _module_name_for_path(rel_path)
        source = _git_show_head_file(root, rel_path)
        records.append(
            FileRecord(
                path=rel_path,
                module=module,
                is_package=is_package,
                source=source,
            )
        )
    return records


def _build_module_to_path(records: list[FileRecord]) -> dict[str, str]:
    path_by_module: defaultdict[str, list[str]] = defaultdict(list)
    for record in records:
        path_by_module[record.module].append(record.path)
    module_to_path: dict[str, str] = {}
    for module, paths in path_by_module.items():
        module_to_path[module] = sorted(paths)[0]
    return module_to_path


def _resolve_module(module_to_path: dict[str, str], module_name: str) -> str | None:
    current = module_name
    while current:
        resolved = module_to_path.get(current)
        if resolved is not None:
            return resolved
        if "." not in current:
            break
        current = current.rsplit(".", 1)[0]
    return None


def _resolve_module_exact(module_to_path: dict[str, str], module_name: str) -> str | None:
    return module_to_path.get(module_name)


def _resolve_relative_module(
    *,
    current_module: str,
    current_is_package: bool,
    level: int,
    module: str | None,
) -> str | None:
    if level <= 0:
        return module
    package = current_module if current_is_package else current_module.rsplit(".", 1)[0]
    package_parts = [part for part in package.split(".") if part]
    ascend = max(level - 1, 0)
    if ascend > len(package_parts):
        return None
    base_parts = package_parts[: len(package_parts) - ascend]
    if module:
        base_parts.extend(module.split("."))
    if not base_parts:
        return None
    return ".".join(base_parts)


def _build_graph_payload(records: list[FileRecord]) -> dict[str, object]:
    module_to_path = _build_module_to_path(records)

    edges: set[tuple[str, str]] = set()
    module_none_unresolved: list[dict[str, object]] = []
    module_none_imports = 0
    module_none_resolved = 0

    for record in records:
        try:
            tree = ast.parse(record.source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported = alias.name
                    if not imported:
                        continue
                    resolved = _resolve_module(module_to_path, imported)
                    if resolved is not None and resolved != record.path:
                        edges.add((record.path, resolved))
                continue

            if not isinstance(node, ast.ImportFrom):
                continue

            module_name = _resolve_relative_module(
                current_module=record.module,
                current_is_package=record.is_package,
                level=int(node.level),
                module=node.module,
            )
            if module_name is None:
                if node.module is None:
                    module_none_imports += 1
                    module_none_unresolved.append(
                        {
                            "src": record.path,
                            "level": int(node.level),
                            "names": sorted(alias.name for alias in node.names),
                            "reason": "relative_base_unresolvable",
                        }
                    )
                continue

            statement_resolved = False
            base_resolved = _resolve_module(module_to_path, module_name)
            if base_resolved is not None and base_resolved != record.path:
                edges.add((record.path, base_resolved))
                statement_resolved = True

            unresolved_aliases: list[str] = []
            for alias in node.names:
                name = alias.name
                if name == "*":
                    continue
                candidate = f"{module_name}.{name}"
                candidate_resolved = _resolve_module(module_to_path, candidate)
                if candidate_resolved is not None and candidate_resolved != record.path:
                    edges.add((record.path, candidate_resolved))
                    statement_resolved = True
                elif node.module is None:
                    unresolved_aliases.append(name)

            if node.module is None:
                module_none_imports += 1
                if statement_resolved:
                    module_none_resolved += 1
                if unresolved_aliases and not statement_resolved:
                    module_none_unresolved.append(
                        {
                            "src": record.path,
                            "level": int(node.level),
                            "names": sorted(unresolved_aliases),
                            "reason": "relative_alias_unresolvable",
                        }
                    )

    nodes = sorted(record.path for record in records)
    ordered_edges = sorted(edges)
    sccs = _tarjan_scc(nodes, ordered_edges)
    nontrivial = [
        sorted(component)
        for component in sccs
        if len(component) > 1
    ]
    nontrivial_sorted = sorted(nontrivial, key=lambda members: (-len(members), members))

    node_to_scc: dict[str, int] = {}
    for idx, component in enumerate(sccs):
        for node in component:
            node_to_scc[node] = idx
    dag_edges: set[tuple[int, int]] = set()
    for src, dst in ordered_edges:
        src_scc = node_to_scc[src]
        dst_scc = node_to_scc[dst]
        if src_scc != dst_scc:
            dag_edges.add((src_scc, dst_scc))

    return {
        "dag_edge_count": len(dag_edges),
        "edge_count": len(ordered_edges),
        "edges": [list(edge) for edge in ordered_edges],
        "largest_scc_size": max((len(component) for component in sccs), default=0),
        "module_none_imports": module_none_imports,
        "module_none_resolved": module_none_resolved,
        "module_none_unresolved": module_none_unresolved,
        "module_none_unresolved_count": len(module_none_unresolved),
        "node_count": len(nodes),
        "nodes": nodes,
        "nontrivial_scc_count": len(nontrivial_sorted),
        "scc_count": len(sccs),
        "scc_samples": nontrivial_sorted,
    }


class SymbolAnalyzer(ast.NodeVisitor):
    def __init__(
        self,
        *,
        record: FileRecord,
        module_to_path: dict[str, str],
    ) -> None:
        self._record = record
        self._module_to_path = module_to_path
        self._scope_stack: list[ScopeFrame] = [
            ScopeFrame(qualname="<module>", frame_type="module", bindings={})
        ]
        self.top_level_definitions: dict[str, str] = {}
        self.definitions: list[DefinitionRecord] = []
        self.import_links: list[ImportLinkRecord] = []
        self.call_candidates: list[CallCandidate] = []

    def _current_scope(self) -> ScopeFrame:
        return self._scope_stack[-1]

    def _scope_qualname(self, name: str) -> str:
        parent = self._current_scope().qualname
        if parent == "<module>":
            return name
        return f"{parent}.{name}"

    def _lookup_binding(self, alias: str) -> ImportBinding | None:
        for frame in reversed(self._scope_stack):
            binding = frame.bindings.get(alias)
            if binding is not None:
                return binding
        return None

    def _push_scope(self, *, qualname: str, frame_type: str) -> None:
        self._scope_stack.append(
            ScopeFrame(qualname=qualname, frame_type=frame_type, bindings={})
        )

    def _pop_scope(self) -> None:
        self._scope_stack.pop()

    def _add_binding(self, binding: ImportBinding) -> None:
        self._current_scope().bindings[binding.alias] = binding

    def _add_import_link(
        self,
        *,
        dst: str | None,
        import_kind: str,
        imported_module: str | None,
        imported_name: str | None,
        binding_alias: str | None,
        target_module: str | None,
        target_symbol: str | None,
        level: int,
        lineno: int,
        col_offset: int,
        resolution: str,
    ) -> None:
        self.import_links.append(
            ImportLinkRecord(
                src=self._record.path,
                dst=dst,
                src_scope=self._current_scope().qualname,
                import_kind=import_kind,
                imported_module=imported_module,
                imported_name=imported_name,
                binding_alias=binding_alias,
                target_module=target_module,
                target_symbol=target_symbol,
                level=level,
                lineno=lineno,
                col_offset=col_offset,
                resolution=resolution,
            )
        )

    def _add_definition(self, *, node: ast.AST, name: str, kind: str) -> None:
        qualname = self._scope_qualname(name)
        lineno = int(getattr(node, "lineno", 0) or 0)
        if self._current_scope().frame_type == "module":
            self.top_level_definitions.setdefault(name, kind)
        self.definitions.append(
            DefinitionRecord(
                path=self._record.path,
                module=self._record.module,
                name=name,
                qualname=qualname,
                kind=kind,
                lineno=lineno,
            )
        )

    def _record_call(
        self,
        *,
        node: ast.Call,
        alias: str,
        attr_chain: tuple[str, ...],
        binding: ImportBinding,
    ) -> None:
        expr = alias
        if attr_chain:
            expr = f"{alias}.{".".join(attr_chain)}"
        self.call_candidates.append(
            CallCandidate(
                src=self._record.path,
                src_scope=self._current_scope().qualname,
                lineno=int(getattr(node, "lineno", 0) or 0),
                col_offset=int(getattr(node, "col_offset", 0) or 0),
                callee_expr=expr,
                alias=alias,
                attr_chain=attr_chain,
                binding=binding,
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add_definition(node=node, name=node.name, kind="class")
        qualname = self._scope_qualname(node.name)
        self._push_scope(qualname=qualname, frame_type="class")
        self.generic_visit(node)
        self._pop_scope()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        parent_type = self._current_scope().frame_type
        kind = "method" if parent_type == "class" else "function"
        self._add_definition(node=node, name=node.name, kind=kind)
        qualname = self._scope_qualname(node.name)
        self._push_scope(qualname=qualname, frame_type="function")
        self.generic_visit(node)
        self._pop_scope()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        parent_type = self._current_scope().frame_type
        kind = "async_method" if parent_type == "class" else "async_function"
        self._add_definition(node=node, name=node.name, kind=kind)
        qualname = self._scope_qualname(node.name)
        self._push_scope(qualname=qualname, frame_type="function")
        self.generic_visit(node)
        self._pop_scope()

    def visit_Import(self, node: ast.Import) -> None:
        lineno = int(getattr(node, "lineno", 0) or 0)
        col_offset = int(getattr(node, "col_offset", 0) or 0)
        for alias in node.names:
            imported = alias.name
            if not imported:
                continue
            bound_alias = alias.asname or imported.split(".", 1)[0]
            target_module = imported if alias.asname else imported.split(".", 1)[0]
            binding = ImportBinding(
                alias=bound_alias,
                kind="module",
                target_module=target_module,
                target_symbol=None,
                imported_module=imported,
            )
            self._add_binding(binding)
            resolved = _resolve_module(self._module_to_path, imported)
            resolution = "resolved" if resolved is not None else "unresolved_module"
            self._add_import_link(
                dst=resolved if resolved != self._record.path else None,
                import_kind="import",
                imported_module=imported,
                imported_name=None,
                binding_alias=bound_alias,
                target_module=imported,
                target_symbol=None,
                level=0,
                lineno=lineno,
                col_offset=col_offset,
                resolution=resolution,
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        lineno = int(getattr(node, "lineno", 0) or 0)
        col_offset = int(getattr(node, "col_offset", 0) or 0)
        level = int(node.level)
        module_name = _resolve_relative_module(
            current_module=self._record.module,
            current_is_package=self._record.is_package,
            level=level,
            module=node.module,
        )

        if module_name is None:
            self._add_import_link(
                dst=None,
                import_kind="from",
                imported_module=node.module,
                imported_name=None,
                binding_alias=None,
                target_module=None,
                target_symbol=None,
                level=level,
                lineno=lineno,
                col_offset=col_offset,
                resolution="relative_base_unresolvable",
            )
            return

        base_resolved = _resolve_module(self._module_to_path, module_name)
        base_dst = base_resolved if base_resolved != self._record.path else None
        self._add_import_link(
            dst=base_dst,
            import_kind="from_module",
            imported_module=module_name,
            imported_name=None,
            binding_alias=None,
            target_module=module_name,
            target_symbol=None,
            level=level,
            lineno=lineno,
            col_offset=col_offset,
            resolution="resolved" if base_dst is not None else "unresolved_module",
        )

        for alias in node.names:
            imported_name = alias.name
            if imported_name == "*":
                self._add_import_link(
                    dst=base_dst,
                    import_kind="from_star",
                    imported_module=module_name,
                    imported_name="*",
                    binding_alias="*",
                    target_module=module_name,
                    target_symbol=None,
                    level=level,
                    lineno=lineno,
                    col_offset=col_offset,
                    resolution="resolved" if base_dst is not None else "unresolved_module",
                )
                continue

            binding_alias = alias.asname or imported_name
            candidate_module = f"{module_name}.{imported_name}"
            candidate_resolved = _resolve_module_exact(
                self._module_to_path,
                candidate_module,
            )
            candidate_dst = (
                candidate_resolved
                if candidate_resolved is not None and candidate_resolved != self._record.path
                else None
            )

            if candidate_dst is not None:
                binding = ImportBinding(
                    alias=binding_alias,
                    kind="module",
                    target_module=candidate_module,
                    target_symbol=None,
                    imported_module=candidate_module,
                )
                self._add_binding(binding)
                self._add_import_link(
                    dst=candidate_dst,
                    import_kind="from_submodule",
                    imported_module=module_name,
                    imported_name=imported_name,
                    binding_alias=binding_alias,
                    target_module=candidate_module,
                    target_symbol=None,
                    level=level,
                    lineno=lineno,
                    col_offset=col_offset,
                    resolution="resolved",
                )
                continue

            binding = ImportBinding(
                alias=binding_alias,
                kind="symbol",
                target_module=module_name,
                target_symbol=imported_name,
                imported_module=module_name,
            )
            self._add_binding(binding)
            self._add_import_link(
                dst=base_dst,
                import_kind="from_symbol",
                imported_module=module_name,
                imported_name=imported_name,
                binding_alias=binding_alias,
                target_module=module_name,
                target_symbol=imported_name,
                level=level,
                lineno=lineno,
                col_offset=col_offset,
                resolution="resolved" if base_dst is not None else "unresolved_symbol_parent",
            )

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            binding = self._lookup_binding(func.id)
            if binding is not None:
                self._record_call(
                    node=node,
                    alias=func.id,
                    attr_chain=(),
                    binding=binding,
                )
        elif isinstance(func, ast.Attribute):
            base_name, attrs = _attribute_chain(func)
            if base_name is not None and attrs:
                binding = self._lookup_binding(base_name)
                if binding is not None:
                    self._record_call(
                        node=node,
                        alias=base_name,
                        attr_chain=attrs,
                        binding=binding,
                    )
        self.generic_visit(node)


def _attribute_chain(node: ast.Attribute) -> tuple[str | None, tuple[str, ...]]:
    attrs: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        attrs.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None, ()
    attrs.reverse()
    return current.id, tuple(attrs)


def _analyze_symbols(
    records: list[FileRecord],
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, list[DefinitionRecord]],
    list[ImportLinkRecord],
    list[CallCandidate],
]:
    module_to_path = _build_module_to_path(records)
    top_level_definitions: dict[str, dict[str, str]] = {}
    definition_records: dict[str, list[DefinitionRecord]] = {}
    import_links: list[ImportLinkRecord] = []
    call_candidates: list[CallCandidate] = []

    for record in records:
        try:
            tree = ast.parse(record.source)
        except SyntaxError:
            top_level_definitions[record.path] = {}
            definition_records[record.path] = []
            continue
        analyzer = SymbolAnalyzer(record=record, module_to_path=module_to_path)
        analyzer.visit(tree)
        top_level_definitions[record.path] = dict(sorted(analyzer.top_level_definitions.items()))
        definition_records[record.path] = sorted(
            analyzer.definitions,
            key=lambda definition: (
                definition.lineno,
                definition.qualname,
                definition.kind,
            ),
        )
        import_links.extend(analyzer.import_links)
        call_candidates.extend(analyzer.call_candidates)

    import_links.sort(
        key=lambda link: (
            link.src,
            link.dst or "",
            link.lineno,
            link.col_offset,
            link.import_kind,
            link.binding_alias or "",
            link.imported_module or "",
            link.imported_name or "",
        )
    )
    call_candidates.sort(
        key=lambda candidate: (
            candidate.src,
            candidate.src_scope,
            candidate.lineno,
            candidate.col_offset,
            candidate.callee_expr,
        )
    )
    return top_level_definitions, definition_records, import_links, call_candidates


def _resolve_module_binding_target(
    *,
    module_to_path: dict[str, str],
    binding: ImportBinding,
    attr_chain: tuple[str, ...],
) -> tuple[str | None, str | None, str | None, str]:
    candidate_roots: list[str] = [binding.target_module]
    if binding.imported_module is not None and binding.imported_module not in candidate_roots:
        candidate_roots.append(binding.imported_module)

    best: tuple[int, str, str, tuple[str, ...]] | None = None
    for root in candidate_roots:
        for prefix_len in range(len(attr_chain), -1, -1):
            if prefix_len == 0:
                candidate_module = root
            else:
                candidate_module = f"{root}.{".".join(attr_chain[:prefix_len])}"
            if prefix_len == 0:
                resolved = _resolve_module(module_to_path, candidate_module)
            else:
                resolved = _resolve_module_exact(module_to_path, candidate_module)
            if resolved is None:
                continue
            remaining = attr_chain[prefix_len:]
            score = (prefix_len, -len(remaining))
            if best is None or score > (best[0], -len(best[3])):
                best = (prefix_len, candidate_module, resolved, remaining)

    if best is None:
        return None, None, None, "unresolved_module"

    _, module_name, resolved_path, remaining = best
    if remaining:
        symbol = remaining[-1] if len(remaining) == 1 else ".".join(remaining)
        if len(remaining) == 1:
            return resolved_path, module_name, symbol, "resolved_module_symbol"
        return resolved_path, module_name, symbol, "resolved_module_symbol_chain"

    return resolved_path, module_name, None, "resolved_module"


def _resolve_call_links(
    *,
    call_candidates: list[CallCandidate],
    module_to_path: dict[str, str],
    top_level_definitions: dict[str, dict[str, str]],
) -> tuple[list[CallLinkRecord], list[CallLinkRecord]]:
    resolved_links: list[CallLinkRecord] = []
    unresolved_links: list[CallLinkRecord] = []

    for candidate in call_candidates:
        binding = candidate.binding
        dst_path: str | None = None
        target_module: str | None = None
        target_symbol: str | None = None
        resolution = "unresolved"

        if binding.kind == "symbol":
            target_module = binding.target_module
            target_symbol = binding.target_symbol
            resolved_module = _resolve_module(module_to_path, binding.target_module)
            if resolved_module is None:
                resolution = "unresolved_module"
            else:
                dst_path = resolved_module
                if candidate.attr_chain:
                    chain_text = ".".join(candidate.attr_chain)
                    base_symbol = target_symbol or ""
                    if base_symbol:
                        target_symbol = f"{base_symbol}.{chain_text}"
                    else:
                        target_symbol = chain_text
                    resolution = "symbol_attribute_chain"
                elif target_symbol is None:
                    resolution = "resolved_module"
                else:
                    resolution = "resolved_symbol_reference"
        else:
            (
                dst_path,
                target_module,
                target_symbol,
                resolution,
            ) = _resolve_module_binding_target(
                module_to_path=module_to_path,
                binding=binding,
                attr_chain=candidate.attr_chain,
            )

        target_definition_kind: str | None = None
        if (
            dst_path is not None
            and target_symbol is not None
            and "." not in target_symbol
        ):
            target_definition_kind = top_level_definitions.get(dst_path, {}).get(target_symbol)
            if target_definition_kind in {"function", "async_function"}:
                resolution = "resolved_function"
            elif target_definition_kind is not None:
                resolution = "resolved_symbol"

        link = CallLinkRecord(
            src=candidate.src,
            dst=dst_path,
            src_scope=candidate.src_scope,
            lineno=candidate.lineno,
            col_offset=candidate.col_offset,
            callee_expr=candidate.callee_expr,
            alias=candidate.alias,
            attr_chain=list(candidate.attr_chain),
            binding_kind=binding.kind,
            target_module=target_module,
            target_symbol=target_symbol,
            target_definition_kind=target_definition_kind,
            resolution=resolution,
        )

        if dst_path is None or dst_path == candidate.src:
            unresolved_links.append(link)
        else:
            resolved_links.append(link)

    resolved_links.sort(
        key=lambda link: (
            link.src,
            link.dst or "",
            link.lineno,
            link.col_offset,
            link.callee_expr,
            link.target_symbol or "",
        )
    )
    unresolved_links.sort(
        key=lambda link: (
            link.src,
            link.src_scope,
            link.lineno,
            link.col_offset,
            link.callee_expr,
            link.resolution,
        )
    )
    return resolved_links, unresolved_links


def _import_link_payload(link: ImportLinkRecord) -> dict[str, object]:
    return {
        "src": link.src,
        "dst": link.dst,
        "src_scope": link.src_scope,
        "import_kind": link.import_kind,
        "imported_module": link.imported_module,
        "imported_name": link.imported_name,
        "binding_alias": link.binding_alias,
        "target_module": link.target_module,
        "target_symbol": link.target_symbol,
        "level": link.level,
        "lineno": link.lineno,
        "col_offset": link.col_offset,
        "resolution": link.resolution,
    }


def _call_link_payload(link: CallLinkRecord) -> dict[str, object]:
    return {
        "src": link.src,
        "dst": link.dst,
        "src_scope": link.src_scope,
        "lineno": link.lineno,
        "col_offset": link.col_offset,
        "callee_expr": link.callee_expr,
        "alias": link.alias,
        "attr_chain": list(link.attr_chain),
        "binding_kind": link.binding_kind,
        "target_module": link.target_module,
        "target_symbol": link.target_symbol,
        "target_definition_kind": link.target_definition_kind,
        "resolution": link.resolution,
    }


def _component_topology(
    nodes: list[str],
    edges: list[tuple[str, str]],
) -> dict[str, object]:
    unique_nodes = sorted(set(nodes))
    node_set = set(unique_nodes)
    unique_edges = sorted(
        {
            (src, dst)
            for src, dst in edges
            if src != dst and src in node_set and dst in node_set
        }
    )
    components = _tarjan_scc(unique_nodes, unique_edges)
    component_by_node: dict[str, int] = {}
    for component_idx, members in enumerate(components):
        for member in members:
            component_by_node[member] = component_idx

    component_dag_edges: set[tuple[int, int]] = set()
    for src, dst in unique_edges:
        src_component = component_by_node[src]
        dst_component = component_by_node[dst]
        if src_component != dst_component:
            component_dag_edges.add((src_component, dst_component))
    ordered_component_dag_edges = sorted(component_dag_edges)

    successors: defaultdict[int, set[int]] = defaultdict(set)
    indegree_by_component: dict[int, int] = {idx: 0 for idx in range(len(components))}
    for src_component, dst_component in ordered_component_dag_edges:
        if dst_component not in successors[src_component]:
            successors[src_component].add(dst_component)
            indegree_by_component[dst_component] += 1

    component_key = {
        idx: tuple(components[idx])
        for idx in range(len(components))
    }
    ready = sorted(
        [idx for idx, indegree in indegree_by_component.items() if indegree == 0],
        key=lambda idx: component_key[idx],
    )
    topological_component_order: list[int] = []
    while ready:
        component_idx = ready.pop(0)
        topological_component_order.append(component_idx)
        for successor in sorted(
            successors.get(component_idx, set()),
            key=lambda idx: component_key[idx],
        ):
            indegree_by_component[successor] -= 1
            if indegree_by_component[successor] == 0:
                ready.append(successor)
                ready.sort(key=lambda idx: component_key[idx])

    if len(topological_component_order) != len(components):
        topological_component_order = sorted(
            range(len(components)),
            key=lambda idx: component_key[idx],
        )

    component_rank = {
        component_idx: rank
        for rank, component_idx in enumerate(topological_component_order)
    }
    return {
        "components": components,
        "component_by_node": component_by_node,
        "component_dag_edges": ordered_component_dag_edges,
        "topological_component_order": topological_component_order,
        "component_rank": component_rank,
    }


def _object_id(path: str, name: str) -> str:
    return f"{path}::{name}"


def _build_object_topology_for_scc(
    *,
    internal_import_links: list[ImportLinkRecord],
    internal_call_links: list[CallLinkRecord],
) -> dict[str, object]:
    object_path_by_id: dict[str, str] = {}
    object_name_by_id: dict[str, str] = {}
    object_roles_by_id: defaultdict[str, set[str]] = defaultdict(set)
    object_link_rows: list[dict[str, object]] = []

    def _register_object(*, object_id: str, path: str, name: str, role: str) -> None:
        object_path_by_id.setdefault(object_id, path)
        object_name_by_id.setdefault(object_id, name)
        object_roles_by_id[object_id].add(role)

    for link in internal_import_links:
        if link.dst is None:
            continue
        src_object = _object_id(link.src, link.src_scope)
        dst_name = link.target_symbol if link.target_symbol else "<module>"
        dst_object = _object_id(link.dst, dst_name)
        _register_object(
            object_id=src_object,
            path=link.src,
            name=link.src_scope,
            role="scope",
        )
        _register_object(
            object_id=dst_object,
            path=link.dst,
            name=dst_name,
            role="symbol" if link.target_symbol else "module",
        )
        object_link_rows.append(
            {
                "src_object": src_object,
                "dst_object": dst_object,
                "link_kind": f"import:{link.import_kind}",
                "src": link.src,
                "dst": link.dst,
                "src_scope": link.src_scope,
                "target_symbol": link.target_symbol,
                "lineno": link.lineno,
                "col_offset": link.col_offset,
                "resolution": link.resolution,
            }
        )

    for link in internal_call_links:
        if link.dst is None:
            continue
        src_object = _object_id(link.src, link.src_scope)
        dst_name = link.target_symbol if link.target_symbol else "<module>"
        dst_object = _object_id(link.dst, dst_name)
        _register_object(
            object_id=src_object,
            path=link.src,
            name=link.src_scope,
            role="scope",
        )
        _register_object(
            object_id=dst_object,
            path=link.dst,
            name=dst_name,
            role="symbol" if link.target_symbol else "module",
        )
        object_link_rows.append(
            {
                "src_object": src_object,
                "dst_object": dst_object,
                "link_kind": "call",
                "src": link.src,
                "dst": link.dst,
                "src_scope": link.src_scope,
                "target_symbol": link.target_symbol,
                "lineno": link.lineno,
                "col_offset": link.col_offset,
                "resolution": link.resolution,
            }
        )

    object_nodes = sorted(object_path_by_id.keys())
    object_edges = sorted(
        {
            (str(row["src_object"]), str(row["dst_object"]))
            for row in object_link_rows
            if str(row["src_object"]) != str(row["dst_object"])
        }
    )
    topology = _component_topology(object_nodes, object_edges)
    components: list[list[str]] = topology["components"]  # type: ignore[assignment]
    component_rank: dict[int, int] = topology["component_rank"]  # type: ignore[assignment]
    component_by_node: dict[str, int] = topology["component_by_node"]  # type: ignore[assignment]
    topological_component_order: list[int] = topology["topological_component_order"]  # type: ignore[assignment]
    component_dag_edges: list[tuple[int, int]] = topology["component_dag_edges"]  # type: ignore[assignment]

    component_id_by_index: dict[int, str] = {}
    component_rows: list[dict[str, object]] = []
    for position, component_idx in enumerate(topological_component_order, start=1):
        component_id = f"OC-{position:03d}"
        component_id_by_index[component_idx] = component_id
        members = sorted(components[component_idx])
        component_rows.append(
            {
                "component_id": component_id,
                "size": len(members),
                "is_cycle": len(members) > 1,
                "members": members,
            }
        )

    object_rows_topological: list[dict[str, object]] = []
    for component_idx in topological_component_order:
        for object_id in sorted(components[component_idx]):
            object_rows_topological.append(
                {
                    "object_id": object_id,
                    "path": object_path_by_id.get(object_id, ""),
                    "name": object_name_by_id.get(object_id, ""),
                    "roles": sorted(object_roles_by_id.get(object_id, set())),
                    "component_id": component_id_by_index[component_idx],
                }
            )

    object_link_rows_topological = sorted(
        object_link_rows,
        key=lambda row: (
            component_rank.get(
                component_by_node.get(str(row["src_object"]), -1),
                10**9,
            ),
            component_rank.get(
                component_by_node.get(str(row["dst_object"]), -1),
                10**9,
            ),
            str(row["src_object"]),
            str(row["dst_object"]),
            int(row["lineno"]),
            int(row["col_offset"]),
            str(row["link_kind"]),
        ),
    )

    component_edge_rows = sorted(
        [
            {
                "src_component_id": component_id_by_index[src_component],
                "dst_component_id": component_id_by_index[dst_component],
            }
            for src_component, dst_component in component_dag_edges
        ],
        key=lambda row: (
            int(str(row["src_component_id"]).split("-")[1]),
            int(str(row["dst_component_id"]).split("-")[1]),
        ),
    )

    return {
        "object_node_count": len(object_nodes),
        "object_link_count": len(object_link_rows),
        "component_count": len(components),
        "nontrivial_component_count": sum(1 for component in components if len(component) > 1),
        "objects_topological_order": object_rows_topological,
        "object_links_topological_order": object_link_rows_topological,
        "component_topological_order": component_rows,
        "component_edges_topological_order": component_edge_rows,
    }


def _build_scc_linkage_payload(
    records: list[FileRecord],
    graph_payload: dict[str, object],
    *,
    tops: tuple[str, ...],
) -> dict[str, object]:
    module_to_path = _build_module_to_path(records)
    (
        top_level_definitions,
        definition_records,
        import_links,
        call_candidates,
    ) = _analyze_symbols(records)
    call_links, unresolved_call_links = _resolve_call_links(
        call_candidates=call_candidates,
        module_to_path=module_to_path,
        top_level_definitions=top_level_definitions,
    )

    import_links_by_edge: defaultdict[tuple[str, str], list[ImportLinkRecord]] = defaultdict(list)
    symbol_import_links_by_edge: defaultdict[tuple[str, str], list[ImportLinkRecord]] = defaultdict(list)
    for link in import_links:
        if link.dst is None:
            continue
        edge = (link.src, link.dst)
        import_links_by_edge[edge].append(link)
        if link.target_symbol is not None:
            symbol_import_links_by_edge[edge].append(link)

    call_links_by_edge: defaultdict[tuple[str, str], list[CallLinkRecord]] = defaultdict(list)
    for link in call_links:
        if link.dst is None:
            continue
        call_links_by_edge[(link.src, link.dst)].append(link)

    edges = [tuple(edge) for edge in graph_payload.get("edges", [])]
    nodes = [str(node) for node in graph_payload.get("nodes", [])]
    file_topology = _component_topology(
        nodes,
        [(str(src), str(dst)) for src, dst in edges],
    )
    file_component_by_node: dict[str, int] = file_topology["component_by_node"]  # type: ignore[assignment]
    file_component_rank: dict[int, int] = file_topology["component_rank"]  # type: ignore[assignment]
    nontrivial_sccs = [
        [str(node) for node in component]
        for component in graph_payload.get("scc_samples", [])
    ]

    scc_payloads: list[dict[str, object]] = []
    for idx, members in enumerate(nontrivial_sccs, start=1):
        member_set = set(members)
        internal_edges = sorted(
            (src, dst)
            for src, dst in edges
            if src in member_set and dst in member_set
        )
        internal_import_links = [
            link
            for link in import_links
            if link.dst is not None and link.src in member_set and link.dst in member_set
        ]
        internal_call_links = [
            link
            for link in call_links
            if link.dst is not None and link.src in member_set and link.dst in member_set
        ]
        object_topology = _build_object_topology_for_scc(
            internal_import_links=internal_import_links,
            internal_call_links=internal_call_links,
        )

        internal_edge_rows: list[dict[str, object]] = []
        for src, dst in internal_edges:
            edge = (src, dst)
            edge_import_links = import_links_by_edge.get(edge, [])
            edge_symbol_links = symbol_import_links_by_edge.get(edge, [])
            edge_call_links = call_links_by_edge.get(edge, [])
            target_symbols = sorted(
                {
                    *(link.target_symbol for link in edge_symbol_links if link.target_symbol),
                    *(link.target_symbol for link in edge_call_links if link.target_symbol),
                }
            )
            source_scopes = sorted(
                {
                    *(link.src_scope for link in edge_import_links),
                    *(link.src_scope for link in edge_call_links),
                }
            )
            internal_edge_rows.append(
                {
                    "src": src,
                    "dst": dst,
                    "import_link_count": len(edge_import_links),
                    "symbol_import_link_count": len(edge_symbol_links),
                    "function_call_link_count": len(edge_call_links),
                    "source_scopes": source_scopes,
                    "target_symbols": target_symbols,
                }
            )

        cut_candidates = sorted(
            internal_edge_rows,
            key=lambda row: (
                int(row["function_call_link_count"]),
                int(row["symbol_import_link_count"]),
                int(row["import_link_count"]),
                str(row["src"]),
                str(row["dst"]),
            ),
        )

        member_definitions: list[dict[str, object]] = []
        for path in sorted(member_set):
            top_defs = [
                {"name": name, "kind": kind}
                for name, kind in top_level_definitions.get(path, {}).items()
            ]
            defs = definition_records.get(path, [])
            member_definitions.append(
                {
                    "path": path,
                    "top_level_definitions": top_defs,
                    "definition_count": len(defs),
                }
            )

        external_incoming = sorted(
            (src, dst)
            for src, dst in edges
            if src not in member_set and dst in member_set
        )
        external_outgoing = sorted(
            (src, dst)
            for src, dst in edges
            if src in member_set and dst not in member_set
        )

        scc_payloads.append(
            {
                "scc_id": f"SCC-{idx:03d}",
                "topological_rank": file_component_rank.get(
                    file_component_by_node.get(sorted(members)[0], -1),
                    10**9,
                ),
                "size": len(members),
                "members": sorted(members),
                "member_definitions": member_definitions,
                "internal_file_edges": internal_edge_rows,
                "internal_import_links": [
                    _import_link_payload(link) for link in internal_import_links
                ],
                "internal_function_links": [
                    _call_link_payload(link) for link in internal_call_links
                ],
                "object_topology": object_topology,
                "cut_candidates": cut_candidates,
                "external_incoming_file_edges": [list(edge) for edge in external_incoming],
                "external_outgoing_file_edges": [list(edge) for edge in external_outgoing],
            }
        )

    resolved_import_links = [link for link in import_links if link.dst is not None]
    sccs_topological_order = sorted(
        [
            {
                "scc_id": str(row["scc_id"]),
                "topological_rank": int(row["topological_rank"]),
                "size": int(row["size"]),
                "members": list(row["members"]),
            }
            for row in scc_payloads
        ],
        key=lambda row: (int(row["topological_rank"]), str(row["scc_id"])),
    )
    node_to_scc_id: dict[str, str] = {}
    for row in scc_payloads:
        members = row.get("members")
        if not isinstance(members, list):
            continue
        scc_id = str(row["scc_id"])
        for member in members:
            node_to_scc_id[str(member)] = scc_id
    rank_by_scc_id = {
        str(row["scc_id"]): int(row["topological_rank"])
        for row in sccs_topological_order
    }
    scc_dag_edges = sorted(
        {
            (node_to_scc_id[src], node_to_scc_id[dst])
            for src, dst in edges
            if src in node_to_scc_id
            and dst in node_to_scc_id
            and node_to_scc_id[src] != node_to_scc_id[dst]
        },
        key=lambda edge: (
            rank_by_scc_id.get(edge[0], 10**9),
            rank_by_scc_id.get(edge[1], 10**9),
            edge[0],
            edge[1],
        ),
    )
    payload = {
        "schema_version": 2,
        "scan_tops": list(tops),
        "nontrivial_scc_count": len(scc_payloads),
        "resolved_import_link_count": len(resolved_import_links),
        "resolved_function_or_symbol_call_link_count": len(call_links),
        "unresolved_call_link_count": len(unresolved_call_links),
        "sccs": scc_payloads,
        "sccs_topological_order": sccs_topological_order,
        "scc_dag_edges_topological_order": [list(edge) for edge in scc_dag_edges],
        "unresolved_call_links_sample": [
            _call_link_payload(link) for link in unresolved_call_links[:200]
        ],
    }
    return payload


def _tarjan_scc(
    nodes: list[str],
    edges: list[tuple[str, str]],
) -> list[list[str]]:
    adjacency: defaultdict[str, list[str]] = defaultdict(list)
    for src, dst in edges:
        adjacency[src].append(dst)
    for src in adjacency:
        adjacency[src].sort()

    index_counter = 0
    index_by_node: dict[str, int] = {}
    lowlink_by_node: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    sccs: list[list[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index_counter
        index_by_node[node] = index_counter
        lowlink_by_node[node] = index_counter
        index_counter += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in adjacency.get(node, []):
            if neighbor not in index_by_node:
                strongconnect(neighbor)
                lowlink_by_node[node] = min(
                    lowlink_by_node[node], lowlink_by_node[neighbor]
                )
            elif neighbor in on_stack:
                lowlink_by_node[node] = min(
                    lowlink_by_node[node], index_by_node[neighbor]
                )

        if lowlink_by_node[node] == index_by_node[node]:
            component: list[str] = []
            while stack:
                popped = stack.pop()
                on_stack.remove(popped)
                component.append(popped)
                if popped == node:
                    break
            sccs.append(sorted(component))

    for node in nodes:
        if node not in index_by_node:
            strongconnect(node)
    return sccs


def _build_folder_counts(
    records: list[FileRecord],
    *,
    threshold: int,
) -> dict[str, object]:
    counts: defaultdict[str, int] = defaultdict(int)
    for record in records:
        parent = str(Path(record.path).parent).replace("\\", "/")
        counts[parent] += 1
    rows = [
        {"path": path, "py_files": count}
        for path, count in sorted(counts.items())
    ]
    hotspots = [row for row in rows if int(row["py_files"]) > threshold]
    return {
        "hotspots_over_threshold": hotspots,
        "rows": rows,
        "target_threshold": threshold,
    }


def _summary_metrics(payload: dict[str, object]) -> dict[str, int]:
    return {
        "dag_edge_count": int(payload["dag_edge_count"]),
        "edge_count": int(payload["edge_count"]),
        "largest_scc_size": int(payload["largest_scc_size"]),
        "module_none_imports": int(payload["module_none_imports"]),
        "module_none_resolved": int(payload["module_none_resolved"]),
        "module_none_unresolved_count": int(payload["module_none_unresolved_count"]),
        "node_count": int(payload["node_count"]),
        "nontrivial_scc_count": int(payload["nontrivial_scc_count"]),
        "scc_count": int(payload["scc_count"]),
    }


def _summary_delta(current: dict[str, int], head: dict[str, int]) -> dict[str, int]:
    return {key: int(current[key]) - int(head[key]) for key in current}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a file-level import DAG and SCC symbol/function linkage artifacts for Python code.",
    )
    parser.add_argument(
        "--tops",
        nargs="+",
        default=list(DEFAULT_SCAN_TOPS),
        help="Top-level directories to scan (default: src tests scripts in).",
    )
    parser.add_argument(
        "--out-current",
        default="out/scout/import_hierarchy_current.json",
        help="Output path for workspace graph JSON.",
    )
    parser.add_argument(
        "--out-head",
        default="out/scout/import_hierarchy_head.json",
        help="Output path for HEAD graph JSON.",
    )
    parser.add_argument(
        "--out-summary",
        default="out/scout/import_hierarchy_summary.json",
        help="Output path for summary JSON.",
    )
    parser.add_argument(
        "--out-folder-counts",
        default="out/scout/folder_py_counts.json",
        help="Output path for per-folder Python file counts.",
    )
    parser.add_argument(
        "--out-scc-linkages",
        default="out/scout/import_hierarchy_scc_linkages.json",
        help="Output path for SCC-level symbol/function linkage JSON.",
    )
    parser.add_argument(
        "--target-threshold",
        type=int,
        default=DEFAULT_TARGET_THRESHOLD,
        help="Hotspot threshold for folder counts (default: 20).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = _repo_root()
    tops = tuple(args.tops)

    workspace_records = _load_workspace_records(root, tops)
    current_payload = _build_graph_payload(workspace_records)
    folder_counts_payload = _build_folder_counts(
        workspace_records,
        threshold=int(args.target_threshold),
    )
    scc_linkage_payload = _build_scc_linkage_payload(
        workspace_records,
        current_payload,
        tops=tops,
    )

    try:
        head_records = _load_head_records(root, tops)
        head_payload = _build_graph_payload(head_records)
    except (subprocess.CalledProcessError, FileNotFoundError):
        head_payload = current_payload

    current_summary = _summary_metrics(current_payload)
    head_summary = _summary_metrics(head_payload)
    summary_payload = {
        "current": current_summary,
        "delta": _summary_delta(current_summary, head_summary),
        "head": head_summary,
        "scan_tops": list(tops),
    }

    _write_json(root / args.out_current, current_payload)
    _write_json(root / args.out_head, head_payload)
    _write_json(root / args.out_summary, summary_payload)
    _write_json(root / args.out_folder_counts, folder_counts_payload)
    _write_json(root / args.out_scc_linkages, scc_linkage_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
