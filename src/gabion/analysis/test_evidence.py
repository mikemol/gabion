from __future__ import annotations

import ast
import json
import re
import tokenize
from collections import Counter
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable

from gabion.analysis import evidence_keys
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import ordered_or_sorted

EVIDENCE_TAG = "gabion:evidence"
_TAG_RE = re.compile(r"#\s*gabion:evidence\s+(?P<ids>.+)")


@dataclass(frozen=True)
class TestEvidence:
    test_id: str
    path: str
    line: int
    evidence: tuple["EvidenceItem", ...]
    status: str


@dataclass(frozen=True)
class TestEvidenceTag:
    test_id: str
    path: str
    line: int
    tags: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceItem:
    key: dict[str, object]
    display: str

    @property
    def identity(self) -> str:
        return evidence_keys.key_identity(self.key)


def build_test_evidence_payload(
    paths: Iterable[Path],
    *,
    root: Path,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    root_display: str | None = None,
) -> dict[str, object]:
    check_deadline()
    root = root.resolve()
    display_root = "."
    exclude_set = {str(item) for item in (exclude or [])}
    include_list = [str(item) for item in (include or [])]
    files = _collect_test_files(paths, root=root, exclude=exclude_set)
    entries: list[TestEvidence] = []
    for path in files:
        entries.extend(_extract_file_evidence(path, root))

    test_ids = [entry.test_id for entry in entries]
    duplicates = sorted(
        {
            test_id
            for test_id, count in Counter(test_ids).items()
            if count > 1
        }
    )
    if duplicates:
        preview = ", ".join(duplicates[:5])
        suffix = "" if len(duplicates) <= 5 else f" (+{len(duplicates) - 5} more)"
        raise ValueError(f"Duplicate test_id entries found: {preview}{suffix}")

    tests_sorted = sorted(entries, key=lambda entry: entry.test_id)
    tests_payload = []
    evidence_index: dict[str, dict[str, object]] = {}
    for entry in tests_sorted:
        evidence_payload = [
            {"key": item.key, "display": item.display} for item in entry.evidence
        ]
        tests_payload.append(
            {
                "test_id": entry.test_id,
                "file": entry.path,
                "line": entry.line,
                "evidence": evidence_payload,
                "status": entry.status,
            }
        )
        for item in entry.evidence:
            identity = item.identity
            record = evidence_index.get(identity)
            if record is None:
                record = {
                    "key": item.key,
                    "display": item.display,
                    "tests": [],
                }
                evidence_index[identity] = record
            record["tests"].append(entry.test_id)

    evidence_payload = []
    for identity in sorted(evidence_index):
        record = evidence_index[identity]
        tests = sorted(record["tests"])
        evidence_payload.append(
            {
                "key": record["key"],
                "display": record["display"],
                "tests": tests,
            }
        )

    return {
        "schema_version": 2,
        "scope": {
            "root": display_root,
            "include": include_list,
            "exclude": sorted(exclude_set),
        },
        "tests": tests_payload,
        "evidence_index": evidence_payload,
    }


def collect_test_tags(
    paths: Iterable[Path],
    *,
    root: Path,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> list[TestEvidenceTag]:
    check_deadline()
    root = root.resolve()
    exclude_set = {str(item) for item in (exclude or [])}
    files = _collect_test_files(paths, root=root, exclude=exclude_set)
    entries: list[TestEvidenceTag] = []
    for path in files:
        entries.extend(_extract_file_tags(path, root))
    return sorted(entries, key=lambda entry: entry.test_id)


def write_test_evidence(
    payload: dict[str, object],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _collect_test_files(
    paths: Iterable[Path],
    *,
    root: Path,
    exclude: set[str],
) -> list[Path]:
    check_deadline()
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            for candidate in ordered_or_sorted(
                path.rglob("test_*.py"),
                source="_collect_test_files.candidates",
                key=lambda item: str(item),
            ):
                if _should_exclude(candidate, root, exclude):
                    continue
                files.append(candidate)
        else:
            if path.suffix == ".py" and not _should_exclude(path, root, exclude):
                files.append(path)
    return files


def _should_exclude(path: Path, root: Path, exclude: set[str]) -> bool:
    if not exclude:
        return False
    rel = str(path.resolve().relative_to(root))
    return any(rel.startswith(pattern) for pattern in exclude)


def _extract_file_evidence(path: Path, root: Path) -> list[TestEvidence]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    comments = _evidence_comments(text)
    rel_path = str(path.resolve().relative_to(root))
    collector = _TestCollector(lines, comments, rel_path)
    collector.visit(tree)
    return collector.entries


def _extract_file_tags(path: Path, root: Path) -> list[TestEvidenceTag]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    comments = _evidence_comments(text)
    rel_path = str(path.resolve().relative_to(root))
    collector = _TagCollector(lines, comments, rel_path)
    collector.visit(tree)
    return collector.entries


def _evidence_comments(text: str) -> dict[int, list[str]]:
    check_deadline()
    comments: dict[int, list[str]] = {}
    for token in tokenize.generate_tokens(StringIO(text).readline):
        if token.type != tokenize.COMMENT:
            continue
        match = _TAG_RE.match(token.string)
        if not match:
            continue
        ids = [item for item in match.group("ids").split() if item]
        if ids:
            comments[token.start[0]] = ids
    return comments


def _find_evidence_tags(
    lines: list[str],
    comment_map: dict[int, list[str]],
    start_line: int,
) -> list[str]:
    check_deadline()
    idx = start_line - 1
    while idx > 0:
        line = lines[idx - 1]
        stripped = line.strip()
        if not stripped:
            idx -= 1
            continue
        if stripped.startswith("@"):
            idx -= 1
            continue
        if stripped.startswith("#"):
            return comment_map.get(idx, [])
        break
    return []


def _normalize_evidence_items(values: Iterable[str]) -> tuple[EvidenceItem, ...]:
    check_deadline()
    items: list[EvidenceItem] = []
    seen: set[str] = set()
    for token in values:
        display = str(token).strip()
        if not display:
            continue
        key = evidence_keys.parse_display(display)
        if key is None:
            key = evidence_keys.make_opaque_key(display)
        key = evidence_keys.normalize_key(key)
        identity = evidence_keys.key_identity(key)
        if identity in seen:
            continue
        seen.add(identity)
        rendered = evidence_keys.render_display(key)
        if evidence_keys.is_opaque(key):
            rendered = display
        items.append(EvidenceItem(key=key, display=rendered))
    items.sort(key=lambda item: item.identity)
    return tuple(items)


def _is_test_function(name: str, class_stack: list[str]) -> bool:
    if not name.startswith("test"):
        return False
    if not class_stack:
        return True
    return class_stack[-1].startswith("Test")


class _TestCollector(ast.NodeVisitor):
    def __init__(
        self,
        lines: list[str],
        comment_map: dict[int, list[str]],
        rel_path: str,
    ) -> None:
        self._lines = lines
        self._comment_map = comment_map
        self._rel_path = rel_path
        self._class_stack: list[str] = []
        self.entries: list[TestEvidence] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        check_deadline()
        self._class_stack.append(node.name)
        for child in node.body:
            self.visit(child)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._handle_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._handle_function(node)

    def _handle_function(self, node: ast.AST) -> None:
        name = getattr(node, "name", "")
        if not _is_test_function(name, self._class_stack):
            return
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            start_line = min(getattr(dec, "lineno", node.lineno) for dec in decorators)
        else:
            start_line = getattr(node, "lineno", 1)
        evidence = _find_evidence_tags(self._lines, self._comment_map, start_line)
        items = _normalize_evidence_items(evidence)
        qualname = "::".join([*self._class_stack, name])
        test_id = f"{self._rel_path}::{qualname}"
        status = "mapped" if items else "unmapped"
        self.entries.append(
            TestEvidence(
                test_id=test_id,
                path=self._rel_path,
                line=getattr(node, "lineno", 0),
                evidence=items,
                status=status,
            )
        )


class _TagCollector(ast.NodeVisitor):
    def __init__(
        self,
        lines: list[str],
        comment_map: dict[int, list[str]],
        rel_path: str,
    ) -> None:
        self._lines = lines
        self._comment_map = comment_map
        self._rel_path = rel_path
        self._class_stack: list[str] = []
        self.entries: list[TestEvidenceTag] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        check_deadline()
        self._class_stack.append(node.name)
        for child in node.body:
            self.visit(child)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._handle_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._handle_function(node)

    def _handle_function(self, node: ast.AST) -> None:
        name = getattr(node, "name", "")
        if not _is_test_function(name, self._class_stack):
            return
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            start_line = min(getattr(dec, "lineno", node.lineno) for dec in decorators)
        else:
            start_line = getattr(node, "lineno", 1)
        raw_tags = _find_evidence_tags(self._lines, self._comment_map, start_line)
        tags = tuple(dict.fromkeys(raw_tags))
        qualname = "::".join([*self._class_stack, name])
        test_id = f"{self._rel_path}::{qualname}"
        self.entries.append(
            TestEvidenceTag(
                test_id=test_id,
                path=self._rel_path,
                line=getattr(node, "lineno", 0),
                tags=tags,
            )
        )
