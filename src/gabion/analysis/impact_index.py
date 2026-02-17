from __future__ import annotations

import ast
import json
import re
import tokenize
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Iterable

from gabion.analysis.dataflow_audit import report_projection_spec_rows
from gabion.analysis.projection_registry import REGISTERED_SPECS
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import ordered_or_sorted

_DEFAULT_ARTIFACT_PATH = Path("artifacts/audit_reports/impact_index.json")
_PYTHON_SOURCE_ROOTS = ("src", "tests")
_MARKDOWN_GLOBS = ("*.md", "docs/**/*.md", "in/**/*.md")
_COMMAND_PATTERN = re.compile(r"gabion\.[A-Za-z][A-Za-z0-9]+")
_ANCHOR_PATTERN = re.compile(r"<a\s+id=\"(?P<anchor>[A-Za-z0-9_\-:.]+)\"\s*></a>")
_INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_\.]*")
_CONFIDENCE_ORDER = {"explicit": 0, "inferred": 1, "weak": 2}
_TARGET_COMMENT_RE = re.compile(r"#\s*impact-target\s*:\s*(?P<targets>.+)$")
_QUALIFIED_IDENTIFIER_RE = re.compile(r"\b([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)\b")


@dataclass(frozen=True)
class ImpactLink:
    source: str
    source_kind: str
    target: str
    confidence: str


@dataclass(frozen=True)
class ImpactIndex:
    links: tuple[ImpactLink, ...]
    graph: dict[str, object] | None


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
                node_type: [
                    self.nodes[node_type][node_id]
                    for node_id in ordered_or_sorted(
                        self.nodes[node_type].keys(),
                        source="impact_index.graph.node_ids",
                    )
                ]
                for node_type in ordered_or_sorted(
                    self.nodes.keys(),
                    source="impact_index.graph.node_types",
                )
            },
            "adjacency": {
                "forward": {
                    node_id: edges
                    for node_id, edges in ordered_or_sorted(
                        self.forward.items(),
                        source="impact_index.graph.forward",
                        key=lambda item: item[0],
                    )
                },
                "reverse": {
                    node_id: edges
                    for node_id, edges in ordered_or_sorted(
                        self.reverse.items(),
                        source="impact_index.graph.reverse",
                        key=lambda item: item[0],
                    )
                },
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


def build_impact_index(
    repo_root: Path | None = None,
    *,
    root: Path | None = None,
    test_paths: Iterable[Path] | None = None,
    doc_paths: Iterable[Path] | None = None,
    include_graph: bool = True,
) -> ImpactIndex:
    check_deadline()
    resolved_root = (root or repo_root or Path.cwd()).resolve()
    tests = (
        list(test_paths)
        if test_paths is not None
        else list((resolved_root / "tests").rglob("test_*.py"))
    )
    docs = (
        list(doc_paths)
        if doc_paths is not None
        else [*resolved_root.glob("*.md"), *(resolved_root / "docs").glob("*.md")]
    )
    symbols = _collect_symbol_universe(resolved_root)
    links: list[ImpactLink] = []
    for path in ordered_or_sorted(
        tests,
        source="impact_index.tests",
        key=lambda item: str(item),
    ):
        check_deadline()
        links.extend(_links_from_test(path=path, root=resolved_root))
    for path in ordered_or_sorted(
        docs,
        source="impact_index.docs",
        key=lambda item: str(item),
    ):
        check_deadline()
        links.extend(_links_from_doc(path=path, root=resolved_root, symbols=symbols))
    deduped = _dedupe_links(links)
    ordered_links = ordered_or_sorted(
        deduped,
        source="impact_index.links",
        key=lambda item: (
            item.source_kind,
            item.source,
            _CONFIDENCE_ORDER.get(item.confidence, 99),
            item.target,
        ),
    )
    graph_payload = _build_graph_payload(resolved_root) if include_graph else None
    return ImpactIndex(links=tuple(ordered_links), graph=graph_payload)


def emit_impact_index(
    repo_root: Path | None = None,
    *,
    root: Path | None = None,
) -> Path:
    resolved_root = (root or repo_root or Path.cwd()).resolve()
    artifact_path = resolved_root / _DEFAULT_ARTIFACT_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    index = build_impact_index(root=resolved_root)
    payload: dict[str, object] = {
        "artifact": str(_DEFAULT_ARTIFACT_PATH),
        "links": [
            {
                "source": link.source,
                "source_kind": link.source_kind,
                "target": link.target,
                "confidence": link.confidence,
            }
            for link in index.links
        ],
        "graph": index.graph
        if index.graph is not None
        else {"nodes": {}, "adjacency": {"forward": {}, "reverse": {}}},
    }
    artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact_path


def _links_from_test(*, path: Path, root: Path) -> list[ImpactLink]:
    check_deadline()
    text = _read_text(path)
    if text is None:
        return []
    tree = _parse_ast(text)
    if tree is None:
        return []
    rel = _relative(path, root)
    comment_map = _impact_comments(text)
    imports = _import_aliases(tree)
    links: list[ImpactLink] = []
    for node in tree.body:
        check_deadline()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            links.extend(
                _links_from_test_function(
                    node=node,
                    rel=rel,
                    imports=imports,
                    comment_map=comment_map,
                )
            )
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for child in node.body:
                check_deadline()
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    links.extend(
                        _links_from_test_function(
                            node=child,
                            rel=rel,
                            imports=imports,
                            comment_map=comment_map,
                        )
                    )
    return links


def _links_from_test_function(
    *,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    rel: str,
    imports: dict[str, str],
    comment_map: dict[int, list[str]],
) -> list[ImpactLink]:
    check_deadline()
    if not node.name.startswith("test"):
        return []
    source = f"{rel}::{node.name}"
    explicit = _decorator_targets(node.decorator_list)
    if not explicit:
        explicit = _nearest_comment_targets(node.lineno, comment_map)
    if explicit:
        return [
            ImpactLink(
                source=source,
                source_kind="test",
                target=target,
                confidence="explicit",
            )
            for target in explicit
        ]
    inferred = _inferred_targets_from_body(node=node, imports=imports)
    if inferred:
        return [
            ImpactLink(
                source=source,
                source_kind="test",
                target=target,
                confidence="inferred",
            )
            for target in inferred
        ]
    weak_targets = ordered_or_sorted(
        {
            target.rsplit(".", 1)[0] if "." in target else target
            for target in imports.values()
        },
        source="impact_index.weak_targets",
    )
    return [
        ImpactLink(
            source=source,
            source_kind="test",
            target=target,
            confidence="weak",
        )
        for target in weak_targets
    ]


def _links_from_doc(*, path: Path, root: Path, symbols: set[str]) -> list[ImpactLink]:
    check_deadline()
    text = _read_text(path)
    if text is None:
        return []
    rel = _relative(path, root)
    frontmatter, body = _parse_frontmatter(text)
    source = rel
    explicit = _coerce_target_list(frontmatter.get("doc_targets"))
    if explicit:
        return [
            ImpactLink(
                source=source,
                source_kind="doc",
                target=target,
                confidence="explicit",
            )
            for target in explicit
        ]
    inferred = _doc_identifier_mentions(body, symbols)
    if inferred:
        return [
            ImpactLink(
                source=source,
                source_kind="doc",
                target=target,
                confidence="inferred",
            )
            for target in inferred
        ]
    weak = _doc_anchor_matches(body, symbols)
    return [
        ImpactLink(
            source=source,
            source_kind="doc",
            target=target,
            confidence="weak",
        )
        for target in weak
    ]


def _build_graph_payload(root: Path) -> dict[str, object]:
    check_deadline()
    graph = ImpactIndexGraph()

    symbols_by_id: dict[str, SymbolInfo] = {}
    symbol_lookup_by_name: dict[str, set[str]] = {}

    for path in _iter_python_files(root):
        check_deadline()
        rel_path = str(path.relative_to(root))
        module = _module_from_path(path.relative_to(root))
        tree = _parse_python_file(path)
        if tree is None:
            continue
        collector = _SymbolCollector(rel_path, module)
        collector.visit(tree)
        for symbol in collector.symbols:
            check_deadline()
            symbols_by_id[symbol.node_id] = symbol
            short_name = symbol.qualname.split(".")[-1]
            symbol_lookup_by_name.setdefault(short_name, set()).add(symbol.node_id)
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
        check_deadline()
        rel_path = str(path.relative_to(root))
        module = _module_from_path(path.relative_to(root))
        tree = _parse_python_file(path)
        if tree is None:
            continue
        imports = _collect_import_aliases_for_graph(tree)

        class _UsageCollector(ast.NodeVisitor):
            def __init__(self) -> None:
                self.stack: list[str] = []

            def _visit_symbol(self, node: ast.AST, name: str) -> None:
                check_deadline()
                self.stack.append(name)
                caller_qual = ".".join(self.stack)
                caller_id = f"symbol:{module}:{caller_qual}"

                for child in _iter_local_calls(node):
                    check_deadline()
                    call_name = _extract_call_name(child)
                    if not call_name:
                        continue
                    callsite_key = (
                        f"{rel_path}:{child.lineno}:{child.col_offset}:"
                        f"{caller_qual}:{call_name}"
                    )
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
                    for callee_symbol_id in ordered_or_sorted(
                        symbol_lookup_by_name.get(call_name, set()),
                        source="impact_index.graph.callee_symbols",
                    ):
                        check_deadline()
                        graph.add_edge(callee_symbol_id, "symbol_to_caller_symbol", caller_id)
                        graph.add_edge(callsite_node, "callsite_to_symbol", callee_symbol_id)

                if (
                    rel_path.startswith("tests/")
                    and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and name.startswith("test_")
                ):
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
                    for name_ref in ordered_or_sorted(
                        names,
                        source="impact_index.graph.test_name_refs",
                    ):
                        check_deadline()
                        if name_ref in imports and imports[name_ref].startswith("gabion"):
                            target_name = imports[name_ref].split(".")[-1]
                            for symbol_id in ordered_or_sorted(
                                symbol_lookup_by_name.get(target_name, set()),
                                source="impact_index.graph.test_import_symbol",
                            ):
                                check_deadline()
                                graph.add_edge(test_node, "test_to_symbol", symbol_id)
                        for symbol_id in ordered_or_sorted(
                            symbol_lookup_by_name.get(name_ref, set()),
                            source="impact_index.graph.test_name_symbol",
                        ):
                            check_deadline()
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
        check_deadline()
        rel_path = str(path.relative_to(root))
        text = path.read_text(encoding="utf-8")
        anchors = [match.group("anchor") for match in _ANCHOR_PATTERN.finditer(text)]
        if not anchors:
            continue
        for anchor in anchors:
            check_deadline()
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
                check_deadline()
                short = token.split(".")[-1]
                for symbol_id in ordered_or_sorted(
                    symbol_lookup_by_name.get(token, set())
                    | symbol_lookup_by_name.get(short, set()),
                    source="impact_index.graph.doc_symbols",
                ):
                    check_deadline()
                    graph.add_edge(doc_node, "doc_section_to_symbol", symbol_id)

    for symbol_id, symbol in symbols_by_id.items():
        check_deadline()
        symbol_terms = {symbol.qualname, symbol.qualname.split(".")[-1], symbol.module}
        for command_node in command_nodes:
            check_deadline()
            command_name = str(graph.nodes["command"][command_node].get("name") or "")
            if any(term and term.lower() in command_name.lower() for term in symbol_terms):
                graph.add_edge(symbol_id, "symbol_to_command", command_node)
        for section_node in report_section_nodes:
            check_deadline()
            section_name = str(graph.nodes["report_section"][section_node].get("section_id") or "")
            if any(term and term.lower() in section_name.lower() for term in symbol_terms):
                graph.add_edge(symbol_id, "symbol_to_report_section", section_node)

    return graph.to_payload()


def _iter_python_files(root: Path) -> Iterable[Path]:
    for source_root in _PYTHON_SOURCE_ROOTS:
        check_deadline()
        target = root / source_root
        if not target.exists():
            continue
        yield from target.rglob("*.py")


def _module_from_path(path: Path) -> str:
    parts = list(path.with_suffix("").parts)
    if parts and parts[0] in _PYTHON_SOURCE_ROOTS:
        parts = parts[1:]
    return ".".join(parts)


def _parse_python_file(path: Path) -> ast.AST | None:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError:
        return None


def _extract_call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _collect_import_aliases_for_graph(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        check_deadline()
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                check_deadline()
                aliases[alias.asname or alias.name] = f"{module}.{alias.name}".strip(".")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                check_deadline()
                aliases[alias.asname or alias.name] = alias.name
    return aliases


def _iter_local_calls(node: ast.AST) -> Iterable[ast.Call]:
    for child in ast.iter_child_nodes(node):
        check_deadline()
        if isinstance(
            child,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            continue
        if isinstance(child, ast.Call):
            yield child
        yield from _iter_local_calls(child)


def _iter_local_names(node: ast.AST) -> Iterable[str]:
    for child in ast.iter_child_nodes(node):
        check_deadline()
        if isinstance(
            child,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            continue
        if isinstance(child, ast.Name):
            yield child.id
        yield from _iter_local_names(child)


def _iter_markdown_files(root: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in _MARKDOWN_GLOBS:
        check_deadline()
        for path in root.glob(pattern):
            check_deadline()
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            yield path


def _section_node_id(path: str, anchor: str) -> str:
    return f"doc_section:{path}#{anchor}"


def _emit_registry_sections(graph: ImpactIndexGraph) -> list[str]:
    section_ids: list[str] = []
    for row in report_projection_spec_rows():
        check_deadline()
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
        check_deadline()
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
        check_deadline()
        path = root / rel_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for match in ordered_or_sorted(
            set(_COMMAND_PATTERN.findall(text)),
            source="impact_index.graph.command_patterns",
        ):
            check_deadline()
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
            check_deadline()
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


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _parse_ast(text: str) -> ast.Module | None:
    try:
        return ast.parse(text)
    except SyntaxError:
        return None


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _impact_comments(text: str) -> dict[int, list[str]]:
    comments: dict[int, list[str]] = {}
    for token in tokenize.generate_tokens(StringIO(text).readline):
        check_deadline()
        if token.type != tokenize.COMMENT:
            continue
        match = _TARGET_COMMENT_RE.match(token.string)
        if not match:
            continue
        targets = _split_targets(match.group("targets"))
        if targets:
            comments[token.start[0]] = targets
    return comments


def _split_targets(raw: str) -> list[str]:
    values = [item.strip() for item in re.split(r"[,\s]+", raw) if item.strip()]
    return ordered_or_sorted(set(values), source="impact_index.split_targets")


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in tree.body:
        check_deadline()
        if isinstance(node, ast.Import):
            for alias in node.names:
                check_deadline()
                name = alias.asname or alias.name.split(".")[-1]
                aliases[name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                check_deadline()
                if alias.name == "*":
                    continue
                name = alias.asname or alias.name
                aliases[name] = f"{module}.{alias.name}" if module else alias.name
    return aliases


def _decorator_targets(decorators: list[ast.expr]) -> list[str]:
    targets: list[str] = []
    for decorator in decorators:
        check_deadline()
        if not isinstance(decorator, ast.Call):
            continue
        name = _call_name(decorator.func)
        if name not in {"impact_target", "impact_targets"}:
            continue
        for arg in decorator.args:
            check_deadline()
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                targets.extend(_split_targets(arg.value))
            elif isinstance(arg, (ast.List, ast.Tuple)):
                for item in arg.elts:
                    check_deadline()
                    if isinstance(item, ast.Constant) and isinstance(item.value, str):
                        targets.extend(_split_targets(item.value))
    return ordered_or_sorted(set(targets), source="impact_index.decorator_targets")


def _call_name(expr: ast.expr) -> str:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return ""


def _nearest_comment_targets(line: int, comment_map: dict[int, list[str]]) -> list[str]:
    idx = line - 1
    while idx > 0:
        check_deadline()
        targets = comment_map.get(idx)
        if targets:
            return targets
        idx -= 1
    return []


def _inferred_targets_from_body(
    *,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    imports: dict[str, str],
) -> list[str]:
    mentioned: set[str] = set()
    for child in ast.walk(node):
        check_deadline()
        if isinstance(child, ast.Name) and child.id in imports:
            mentioned.add(imports[child.id])
        elif isinstance(child, ast.Attribute):
            path = _attribute_path(child)
            if not path:
                continue
            root = path[0]
            if root in imports:
                mentioned.add(".".join([imports[root], *path[1:]]))
    return ordered_or_sorted(mentioned, source="impact_index.inferred_targets")


def _attribute_path(node: ast.Attribute) -> list[str]:
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return list(reversed(parts))
    return []


def _parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    if not text.startswith("---\n"):
        return {}, text
    lines = text.splitlines()
    end = None
    for index in range(1, len(lines)):
        check_deadline()
        if lines[index].strip() == "---":
            end = index
            break
    if end is None:
        return {}, text
    raw = lines[1:end]
    body = "\n".join(lines[end + 1 :])
    payload: dict[str, object] = {}
    current: str | None = None
    items: list[str] = []
    for line in raw:
        check_deadline()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current is not None:
            items.append(stripped[2:].strip().strip("\"'"))
            continue
        if current is not None:
            payload[current] = items
            current = None
            items = []
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "doc_targets" and not value:
            current = key
            items = []
            continue
        payload[key] = value.strip().strip("\"'")
    if current is not None:
        payload[current] = items
    return payload, body


def _coerce_target_list(value: object) -> list[str]:
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        return _split_targets(raw)
    if isinstance(value, list):
        return ordered_or_sorted(
            {str(item).strip() for item in value if str(item).strip()},
            source="impact_index.coerce_target_list",
        )
    return []


def _collect_symbol_universe(root: Path) -> set[str]:
    symbols: set[str] = set()
    src = root / "src"
    if not src.exists():
        return symbols
    for path in src.rglob("*.py"):
        check_deadline()
        rel = path.relative_to(src).as_posix()
        module = rel.removesuffix(".py").replace("/", ".")
        if module.endswith(".__init__"):
            module = module.rsplit(".", 1)[0]
        tree = _parse_ast(_read_text(path) or "")
        if tree is None:
            continue
        for node in tree.body:
            check_deadline()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.add(f"{module}.{node.name}")
    return symbols


def _doc_identifier_mentions(body: str, symbols: set[str]) -> list[str]:
    mentioned = {match.group(1) for match in _QUALIFIED_IDENTIFIER_RE.finditer(body)}
    return ordered_or_sorted(
        symbols.intersection(mentioned),
        source="impact_index.doc_mentions",
    )


def _doc_anchor_matches(body: str, symbols: set[str]) -> list[str]:
    anchors = {
        match.group(1).strip().lower() for match in re.finditer(r"\(#([^)]+)\)", body)
    }
    weak: set[str] = set()
    for symbol in symbols:
        check_deadline()
        tail = symbol.rsplit(".", 1)[-1].lower().replace("_", "-")
        if tail in anchors:
            weak.add(symbol)
    return ordered_or_sorted(weak, source="impact_index.doc_anchors")


def _dedupe_links(links: Iterable[ImpactLink]) -> list[ImpactLink]:
    deduped: dict[tuple[str, str, str], ImpactLink] = {}
    for item in links:
        check_deadline()
        key = (item.source_kind, item.source, item.target)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = item
            continue
        if _CONFIDENCE_ORDER.get(item.confidence, 99) < _CONFIDENCE_ORDER.get(
            existing.confidence,
            99,
        ):
            deduped[key] = item
    return list(deduped.values())


if __name__ == "__main__":
    emit_impact_index()
