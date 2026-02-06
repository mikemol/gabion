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

EVIDENCE_TAG = "gabion:evidence"
_TAG_RE = re.compile(r"#\s*gabion:evidence\s+(?P<ids>.+)")


@dataclass(frozen=True)
class TestEvidence:
    test_id: str
    path: str
    line: int
    evidence: tuple[str, ...]
    status: str


def build_test_evidence_payload(
    paths: Iterable[Path],
    *,
    root: Path,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    root_display: str | None = None,
) -> dict[str, object]:
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
    tests_payload = [
        {
            "test_id": entry.test_id,
            "file": entry.path,
            "line": entry.line,
            "evidence": list(entry.evidence),
            "status": entry.status,
        }
        for entry in tests_sorted
    ]

    evidence_index: dict[str, list[str]] = {}
    for entry in tests_sorted:
        for evidence_id in entry.evidence:
            evidence_index.setdefault(evidence_id, []).append(entry.test_id)

    evidence_payload = [
        {"evidence_id": evidence_id, "tests": sorted(test_ids)}
        for evidence_id, test_ids in sorted(evidence_index.items())
    ]

    return {
        "schema_version": 1,
        "scope": {
            "root": display_root,
            "include": include_list,
            "exclude": sorted(exclude_set),
        },
        "tests": tests_payload,
        "evidence_index": evidence_payload,
    }


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
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            for candidate in sorted(path.rglob("test_*.py")):
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


def _evidence_comments(text: str) -> dict[int, list[str]]:
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
        unique = tuple(sorted(set(evidence)))
        qualname = "::".join([*self._class_stack, name])
        test_id = f"{self._rel_path}::{qualname}"
        status = "mapped" if unique else "unmapped"
        self.entries.append(
            TestEvidence(
                test_id=test_id,
                path=self._rel_path,
                line=getattr(node, "lineno", 0),
                evidence=unique,
                status=status,
            )
        )
