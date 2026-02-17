from __future__ import annotations

import ast
import re
import tokenize
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable

from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import ordered_or_sorted

_CONFIDENCE_ORDER = {"explicit": 0, "inferred": 1, "weak": 2}
_TARGET_COMMENT_RE = re.compile(r"#\s*impact-target\s*:\s*(?P<targets>.+)$")
_IDENTIFIER_RE = re.compile(r"\b([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)\b")


@dataclass(frozen=True)
class ImpactLink:
    source: str
    source_kind: str
    target: str
    confidence: str


@dataclass(frozen=True)
class ImpactIndex:
    links: tuple[ImpactLink, ...]


def build_impact_index(
    *,
    root: Path,
    test_paths: Iterable[Path] | None = None,
    doc_paths: Iterable[Path] | None = None,
) -> ImpactIndex:
    check_deadline()
    root = root.resolve()
    tests = list(test_paths) if test_paths is not None else list((root / "tests").rglob("test_*.py"))
    docs = list(doc_paths) if doc_paths is not None else [*root.glob("*.md"), *(root / "docs").glob("*.md")]
    symbols = _collect_symbol_universe(root)
    links: list[ImpactLink] = []
    for path in ordered_or_sorted(tests, source="impact_index.tests", key=lambda item: str(item)):
        links.extend(_links_from_test(path=path, root=root))
    for path in ordered_or_sorted(docs, source="impact_index.docs", key=lambda item: str(item)):
        links.extend(_links_from_doc(path=path, root=root, symbols=symbols))
    deduped = _dedupe_links(links)
    ordered = ordered_or_sorted(
        deduped,
        source="impact_index.links",
        key=lambda item: (
            item.source_kind,
            item.source,
            _CONFIDENCE_ORDER.get(item.confidence, 99),
            item.target,
        ),
    )
    return ImpactIndex(links=tuple(ordered))


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
            ImpactLink(source=source, source_kind="test", target=target, confidence="explicit")
            for target in explicit
        ]
    inferred = _inferred_targets_from_body(node=node, imports=imports)
    if inferred:
        return [
            ImpactLink(source=source, source_kind="test", target=target, confidence="inferred")
            for target in inferred
        ]
    weak_targets = ordered_or_sorted(
        {target.rsplit(".", 1)[0] if "." in target else target for target in imports.values()},
        source="impact_index.weak_targets",
    )
    return [
        ImpactLink(source=source, source_kind="test", target=target, confidence="weak")
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
            ImpactLink(source=source, source_kind="doc", target=target, confidence="explicit")
            for target in explicit
        ]
    inferred = _doc_identifier_mentions(body, symbols)
    if inferred:
        return [
            ImpactLink(source=source, source_kind="doc", target=target, confidence="inferred")
            for target in inferred
        ]
    weak = _doc_anchor_matches(body, symbols)
    return [
        ImpactLink(source=source, source_kind="doc", target=target, confidence="weak")
        for target in weak
    ]


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


def _inferred_targets_from_body(*, node: ast.FunctionDef | ast.AsyncFunctionDef, imports: dict[str, str]) -> list[str]:
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
    mentioned = {match.group(1) for match in _IDENTIFIER_RE.finditer(body)}
    return ordered_or_sorted(symbols.intersection(mentioned), source="impact_index.doc_mentions")


def _doc_anchor_matches(body: str, symbols: set[str]) -> list[str]:
    anchors = {match.group(1).strip().lower() for match in re.finditer(r"\(#([^)]+)\)", body)}
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
        if _CONFIDENCE_ORDER.get(item.confidence, 99) < _CONFIDENCE_ORDER.get(existing.confidence, 99):
            deduped[key] = item
    return list(deduped.values())
