from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from gabion.analysis.dataflow_audit import report_projection_spec_rows
from gabion.analysis.projection_registry import REGISTERED_SPECS

_DEFAULT_ARTIFACT_PATH = Path("artifacts/audit_reports/impact_index.json")
_PYTHON_SOURCE_ROOTS = ("src", "tests")
_MARKDOWN_GLOBS = ("*.md", "docs/**/*.md", "in/**/*.md")
_COMMAND_PATTERN = re.compile(r"gabion\.[A-Za-z][A-Za-z0-9]+")
_ANCHOR_PATTERN = re.compile(r"<a\s+id=\"(?P<anchor>[A-Za-z0-9_\-:.]+)\"\s*></a>")
_INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_\.]*")


@dataclass(frozen=True)
class SymbolInfo:
    node_id: str
    module: str
    qualname: str
    path: str
    start_line: int
    end_line: int
    start_col: int
    end_col: int


@dataclass
class ImpactIndexGraph:
    nodes: dict[str, dict[str, dict[str, object]]] = field(default_factory=dict)
    forward: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    reverse: dict[str, list[dict[str, str]]] = field(default_factory=dict)

    def add_node(self, node_type: str, key: str, payload: dict[str, object]) -> str:
        node_id = f"{node_type}:{key}"
        self.nodes.setdefault(node_type, {})[node_id] = payload
        return node_id

    def add_edge(self, source: str, edge_type: str, target: str) -> None:
        self.forward.setdefault(source, []).append({"type": edge_type, "target": target})
        self.reverse.setdefault(target, []).append({"type": edge_type, "source": source})

    def to_payload(self) -> dict[str, object]:
        return {
            "nodes": {
                node_type: [self.nodes[node_type][node_id] for node_id in sorted(self.nodes[node_type])]
                for node_type in sorted(self.nodes)
            },
            "adjacency": {
                "forward": {node_id: edges for node_id, edges in sorted(self.forward.items())},
                "reverse": {node_id: edges for node_id, edges in sorted(self.reverse.items())},
            },
        }


class _SymbolCollector(ast.NodeVisitor):
    def __init__(self, path: str, module: str) -> None:
        self.path = path
        self.module = module
        self.stack: list[str] = []
        self.symbols: list[SymbolInfo] = []

    def _visit_symbol(self, node: ast.AST, name: str) -> None:
        start_line = int(getattr(node, "lineno", 1))
        end_line = int(getattr(node, "end_lineno", start_line))
        start_col = int(getattr(node, "col_offset", 0))
        end_col = int(getattr(node, "end_col_offset", start_col))
        qualname = ".".join((*self.stack, name)) if self.stack else name
        symbol_node_id = f"symbol:{self.module}:{qualname}"
        self.symbols.append(
            SymbolInfo(
                node_id=symbol_node_id,
                module=self.module,
                qualname=qualname,
                path=self.path,
                start_line=start_line,
                end_line=end_line,
                start_col=start_col,
                end_col=end_col,
            )
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_symbol(node, node.name)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_symbol(node, node.name)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_symbol(node, node.name)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


def _iter_python_files(root: Path) -> Iterable[Path]:
    for source_root in _PYTHON_SOURCE_ROOTS:
        target = root / source_root
        if not target.exists():
            continue
        yield from target.rglob("*.py")


def _module_from_path(path: Path) -> str:
    parts = list(path.with_suffix("").parts)
    if parts and parts[0] in _PYTHON_SOURCE_ROOTS:
        parts = parts[1:]
    return ".".join(parts)


def _parse_python(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _extract_call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _collect_import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{module}.{alias.name}".strip(".")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name
    return aliases




def _iter_local_calls(node: ast.AST) -> Iterable[ast.Call]:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(child, ast.Call):
            yield child
        yield from _iter_local_calls(child)


def _iter_local_names(node: ast.AST) -> Iterable[str]:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(child, ast.Name):
            yield child.id
        yield from _iter_local_names(child)

def _iter_markdown_files(root: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in _MARKDOWN_GLOBS:
        for path in root.glob(pattern):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            yield path


def _section_node_id(path: str, anchor: str) -> str:
    return f"doc_section:{path}#{anchor}"


def _emit_registry_sections(graph: ImpactIndexGraph) -> list[str]:
    section_ids: list[str] = []
    for row in report_projection_spec_rows():
        section_id = str(row.get("section_id") or "")
        if not section_id:
            continue
        key = f"{section_id}"
        node_id = graph.add_node(
            "report_section",
            key,
            {
                "id": f"report_section:{section_id}",
                "section_id": section_id,
                "phase": str(row.get("phase") or ""),
                "deps": list(row.get("deps") or []),
            },
        )
        section_ids.append(node_id)
    for spec in REGISTERED_SPECS.values():
        key = f"projection:{spec.name}"
        node_id = graph.add_node(
            "report_section",
            key,
            {
                "id": f"projection:{spec.name}",
                "section_id": spec.name,
                "phase": "projection_registry",
                "deps": [],
                "domain": spec.domain,
            },
        )
        section_ids.append(node_id)
    return section_ids


def _emit_command_nodes(graph: ImpactIndexGraph, root: Path) -> list[str]:
    command_ids: list[str] = []
    for rel_path in (Path("src/gabion/cli.py"), Path("src/gabion/server.py")):
        path = root / rel_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for match in sorted(set(_COMMAND_PATTERN.findall(text))):
            command_ids.append(
                graph.add_node(
                    "command",
                    match,
                    {
                        "id": f"command:{match}",
                        "surface": "lsp",
                        "name": match,
                    },
                )
            )
        for match in re.findall(r'@app\.command\("([a-z0-9\-]+)"\)', text):
            command = f"gabion {match}"
            command_ids.append(
                graph.add_node(
                    "command",
                    command,
                    {
                        "id": f"command:{command}",
                        "surface": "cli",
                        "name": command,
                    },
                )
            )
    return command_ids


def build_impact_index(repo_root: Path | None = None) -> dict[str, object]:
    root = (repo_root or Path.cwd()).resolve()
    graph = ImpactIndexGraph()

    symbols_by_id: dict[str, SymbolInfo] = {}
    symbol_lookup_by_name: dict[str, set[str]] = {}

    for path in _iter_python_files(root):
        rel_path = str(path.relative_to(root))
        module = _module_from_path(path.relative_to(root))
        tree = _parse_python(path)
        collector = _SymbolCollector(rel_path, module)
        collector.visit(tree)
        for symbol in collector.symbols:
            symbols_by_id[symbol.node_id] = symbol
            symbol_lookup_by_name.setdefault(symbol.qualname.split(".")[-1], set()).add(symbol.node_id)
            symbol_lookup_by_name.setdefault(symbol.qualname, set()).add(symbol.node_id)
            symbol_node = graph.add_node(
                "symbol",
                f"{symbol.module}:{symbol.qualname}",
                {
                    "id": symbol.node_id,
                    "module": symbol.module,
                    "qualname": symbol.qualname,
                },
            )
            span_key = (
                f"{symbol.path}:{symbol.start_line}:{symbol.end_line}:"
                f"{symbol.start_col}:{symbol.end_col}"
            )
            span_node = graph.add_node(
                "span",
                span_key,
                {
                    "id": f"span:{span_key}",
                    "path": symbol.path,
                    "start_line": symbol.start_line,
                    "end_line": symbol.end_line,
                    "start_col": symbol.start_col,
                    "end_col": symbol.end_col,
                },
            )
            graph.add_edge(span_node, "span_to_symbol", symbol_node)

    command_nodes = _emit_command_nodes(graph, root)
    report_section_nodes = _emit_registry_sections(graph)

    for path in _iter_python_files(root):
        rel_path = str(path.relative_to(root))
        module = _module_from_path(path.relative_to(root))
        tree = _parse_python(path)
        imports = _collect_import_aliases(tree)

        class _UsageCollector(ast.NodeVisitor):
            def __init__(self) -> None:
                self.stack: list[str] = []

            def _visit_symbol(self, node: ast.AST, name: str) -> None:
                self.stack.append(name)
                caller_qual = ".".join(self.stack)
                caller_id = f"symbol:{module}:{caller_qual}"

                for child in _iter_local_calls(node):
                    if isinstance(child, ast.Call):
                        call_name = _extract_call_name(child)
                        if not call_name:
                            continue
                        callsite_key = f"{rel_path}:{child.lineno}:{child.col_offset}:{caller_qual}:{call_name}"
                        callsite_node = graph.add_node(
                            "callsite",
                            callsite_key,
                            {
                                "id": f"callsite:{callsite_key}",
                                "path": rel_path,
                                "line": child.lineno,
                                "col": child.col_offset,
                                "caller_symbol": caller_id,
                                "call_expr": call_name,
                            },
                        )
                        graph.add_edge(callsite_node, "callsite_to_caller", caller_id)
                        for callee_symbol_id in sorted(symbol_lookup_by_name.get(call_name, set())):
                            graph.add_edge(callee_symbol_id, "symbol_to_caller_symbol", caller_id)
                            graph.add_edge(callsite_node, "callsite_to_symbol", callee_symbol_id)

                if rel_path.startswith("tests/") and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and name.startswith("test_"):
                    test_key = f"{rel_path}::{name}"
                    test_node = graph.add_node(
                        "test",
                        test_key,
                        {
                            "id": f"test:{test_key}",
                            "test_file": rel_path,
                            "test_function_id": name,
                        },
                    )
                    names = set(_iter_local_names(node))
                    for name_ref in names:
                        if name_ref in imports and imports[name_ref].startswith("gabion"):
                            target_name = imports[name_ref].split(".")[-1]
                            for symbol_id in sorted(symbol_lookup_by_name.get(target_name, set())):
                                graph.add_edge(test_node, "test_to_symbol", symbol_id)
                        for symbol_id in sorted(symbol_lookup_by_name.get(name_ref, set())):
                            graph.add_edge(test_node, "test_to_symbol", symbol_id)

                self.generic_visit(node)
                self.stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._visit_symbol(node, node.name)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._visit_symbol(node, node.name)

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self._visit_symbol(node, node.name)

        _UsageCollector().visit(tree)

    for path in _iter_markdown_files(root):
        rel_path = str(path.relative_to(root))
        text = path.read_text(encoding="utf-8")
        anchors = [match.group("anchor") for match in _ANCHOR_PATTERN.finditer(text)]
        if not anchors:
            continue
        for anchor in anchors:
            doc_node = graph.add_node(
                "doc_section",
                f"{rel_path}#{anchor}",
                {
                    "id": _section_node_id(rel_path, anchor),
                    "file": rel_path,
                    "anchor": anchor,
                },
            )
            window_start = text.find(f'id="{anchor}"')
            section_text = text[window_start : window_start + 1200] if window_start >= 0 else text
            identifiers = set(_IDENTIFIER_PATTERN.findall(section_text))
            identifiers.update(_INLINE_CODE_PATTERN.findall(section_text))
            identifiers.add(anchor)
            for token in identifiers:
                short = token.split(".")[-1]
                for symbol_id in sorted(symbol_lookup_by_name.get(token, set()) | symbol_lookup_by_name.get(short, set())):
                    graph.add_edge(doc_node, "doc_section_to_symbol", symbol_id)

    for symbol_id, symbol in symbols_by_id.items():
        symbol_terms = {symbol.qualname, symbol.qualname.split(".")[-1], symbol.module}
        for command_node in command_nodes:
            command_name = str(graph.nodes["command"][command_node].get("name") or "")
            if any(term and term.lower() in command_name.lower() for term in symbol_terms):
                graph.add_edge(symbol_id, "symbol_to_command", command_node)
        for section_node in report_section_nodes:
            section_name = str(graph.nodes["report_section"][section_node].get("section_id") or "")
            if any(term and term.lower() in section_name.lower() for term in symbol_terms):
                graph.add_edge(symbol_id, "symbol_to_report_section", section_node)

    return {
        "artifact": str(_DEFAULT_ARTIFACT_PATH),
        "graph": graph.to_payload(),
    }


def emit_impact_index(repo_root: Path | None = None) -> Path:
    root = (repo_root or Path.cwd()).resolve()
    artifact_path = root / _DEFAULT_ARTIFACT_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_impact_index(root)
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact_path


if __name__ == "__main__":
    emit_impact_index()
