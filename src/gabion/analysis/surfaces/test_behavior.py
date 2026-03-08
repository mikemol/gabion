from __future__ import annotations

import ast
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.surfaces import test_evidence
from gabion.json_types import JSONObject
from gabion.order_contract import sort_once

_BEHAVIOR_TAG_RE = re.compile(r"#\s*gabion:behavior\s+(?P<body>.+)")
_EVIDENCE_TAG_RE = re.compile(r"#\s*gabion:evidence\b")
_FACET_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_ALLOWED_PRIMARY_VALUES = ("desired", "allowed_unwanted", "verboten")
_ALLOWED_BEHAVIOR_KEYS = {"primary", "facets"}
_MAX_ERROR_PREVIEW = 200


@dataclass(frozen=True)
class TestBehavior:
    test_id: str
    path: str
    line: int
    primary: str
    facets: tuple[str, ...]


@dataclass(frozen=True)
class _BehaviorParseResult:
    kind: str
    primary: str
    facets: tuple[str, ...]
    error: str


def build_test_behavior_payload(
    paths: Iterable[Path],
    *,
    root: Path,
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
) -> JSONObject:
    check_deadline()
    root = root.resolve()
    display_root = "."
    exclude_set = {str(item) for item in exclude}
    include_list = [str(item) for item in include]
    files = test_evidence._collect_test_files(paths, root=root, exclude=exclude_set)
    entries: list[TestBehavior] = []
    violations: list[str] = []
    for path in files:
        check_deadline()
        file_entries, file_violations = _extract_file_behavior(path, root)
        entries.extend(file_entries)
        violations.extend(file_violations)

    test_ids = [entry.test_id for entry in entries]
    duplicates = sort_once(
        {test_id for test_id, count in Counter(test_ids).items() if count > 1},
        source="test_behavior.build_test_behavior_payload.duplicates",
    )
    if duplicates:
        preview = ", ".join(duplicates[:5])
        suffix = "" if len(duplicates) <= 5 else f" (+{len(duplicates) - 5} more)"
        violations.append(f"Duplicate test_id entries found: {preview}{suffix}")
    if violations:
        raise ValueError(_render_violations(violations))

    sorted_entries = sort_once(
        entries,
        key=lambda entry: entry.test_id,
        source="test_behavior.build_test_behavior_payload.entries",
    )
    tests_payload = [
        {
            "test_id": entry.test_id,
            "file": entry.path,
            "line": entry.line,
            "primary": entry.primary,
            "facets": list(entry.facets),
        }
        for entry in sorted_entries
    ]
    summary = {
        key: sum(1 for entry in sorted_entries if entry.primary == key)
        for key in _ALLOWED_PRIMARY_VALUES
    }
    return {
        "schema_version": 1,
        "scope": {
            "root": display_root,
            "include": include_list,
            "exclude": sort_once(exclude_set, source="test_behavior.build_test_behavior_payload.exclude"),
        },
        "tests": tests_payload,
        "summary": summary,
    }


def collect_test_behavior_contract_violations(
    paths: Iterable[Path],
    *,
    root: Path,
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
) -> list[str]:
    try:
        build_test_behavior_payload(paths, root=root, include=include, exclude=exclude)
    except ValueError as exc:
        lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
        return lines
    return []


def write_test_behavior(
    payload: JSONObject,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _render_violations(violations: list[str]) -> str:
    rendered = sort_once(
        violations,
        source="test_behavior._render_violations",
    )
    if len(rendered) <= _MAX_ERROR_PREVIEW:
        return "\n".join(rendered)
    remaining = len(rendered) - _MAX_ERROR_PREVIEW
    preview = [*rendered[:_MAX_ERROR_PREVIEW], f"... (+{remaining} more)"]
    return "\n".join(preview)


def _extract_file_behavior(path: Path, root: Path) -> tuple[list[TestBehavior], list[str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return [], []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [], []
    lines = text.splitlines()
    rel_path = str(path.resolve().relative_to(root))
    evidence_comments = test_evidence._evidence_comments(text)
    canonical_rel_path = test_evidence._canonical_test_rel_path(rel_path, evidence_comments)
    collector = _BehaviorCollector(lines, canonical_rel_path)
    collector.visit(tree)
    return collector.entries, collector.violations


def _parse_behavior_comment(comment_line: str) -> _BehaviorParseResult:
    match = _BEHAVIOR_TAG_RE.match(comment_line)
    if match is None:
        return _BehaviorParseResult(kind="absent", primary="", facets=(), error="")
    body = match.group("body").strip()
    if not body:
        return _BehaviorParseResult(
            kind="error",
            primary="",
            facets=(),
            error="behavior tag must declare key/value fields",
        )
    fields: dict[str, str] = {}
    for token in body.split():
        check_deadline()
        if "=" not in token:
            return _BehaviorParseResult(
                kind="error",
                primary="",
                facets=(),
                error=f"malformed behavior token {token!r}; expected key=value",
            )
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            return _BehaviorParseResult(
                kind="error",
                primary="",
                facets=(),
                error=f"malformed behavior token {token!r}; expected non-empty key and value",
            )
        if key in fields:
            return _BehaviorParseResult(
                kind="error",
                primary="",
                facets=(),
                error=f"duplicate behavior key {key!r}",
            )
        fields[key] = value
    unknown = set(fields) - _ALLOWED_BEHAVIOR_KEYS
    if unknown:
        return _BehaviorParseResult(
            kind="error",
            primary="",
            facets=(),
            error=(
                "unknown behavior keys: "
                f"{sort_once(unknown, source='test_behavior._parse_behavior_comment.unknown')}"
            ),
        )
    primary = fields.get("primary")
    if primary is None:
        return _BehaviorParseResult(
            kind="error",
            primary="",
            facets=(),
            error="missing required behavior key 'primary'",
        )
    if primary not in _ALLOWED_PRIMARY_VALUES:
        return _BehaviorParseResult(
            kind="error",
            primary="",
            facets=(),
            error=f"invalid primary {primary!r}; expected one of {list(_ALLOWED_PRIMARY_VALUES)}",
        )
    facets_value = fields.get("facets")
    facets: tuple[str, ...] = ()
    if facets_value is not None:
        raw_facets = [item.strip() for item in facets_value.split(",")]
        if not raw_facets or any(item == "" for item in raw_facets):
            return _BehaviorParseResult(
                kind="error",
                primary="",
                facets=(),
                error="facets must be a comma-delimited list without empty values",
            )
        for facet in raw_facets:
            check_deadline()
            if not _FACET_TOKEN_RE.match(facet):
                return _BehaviorParseResult(
                    kind="error",
                    primary="",
                    facets=(),
                    error=f"invalid facet {facet!r}; expected lowercase token",
                )
        facets = tuple(
            sort_once(
                set(raw_facets),
                source="test_behavior._parse_behavior_comment.facets",
            )
        )
    return _BehaviorParseResult(
        kind="label",
        primary=primary,
        facets=facets,
        error="",
    )


def _find_behavior_label(lines: list[str], start_line: int) -> _BehaviorParseResult:
    seen: list[_BehaviorParseResult] = []
    idx = start_line - 1
    while idx > 0:
        check_deadline()
        line = lines[idx - 1]
        stripped = line.strip()
        if not stripped:
            idx -= 1
            continue
        if stripped.startswith("@"):
            idx -= 1
            continue
        if stripped.startswith("#"):
            parsed = _parse_behavior_comment(stripped)
            if parsed.kind == "error":
                return parsed
            if parsed.kind == "label":
                seen.append(parsed)
                idx -= 1
                continue
            if _EVIDENCE_TAG_RE.match(stripped):
                idx -= 1
                continue
            if seen:
                break
            return _BehaviorParseResult(kind="missing", primary="", facets=(), error="")
        if seen:
            break
        return _BehaviorParseResult(kind="missing", primary="", facets=(), error="")
    if len(seen) > 1:
        return _BehaviorParseResult(
            kind="error",
            primary="",
            facets=(),
            error="duplicate gabion:behavior tags for a single test",
        )
    if not seen:
        return _BehaviorParseResult(kind="missing", primary="", facets=(), error="")
    return seen[0]


class _BehaviorCollector(ast.NodeVisitor):
    def __init__(
        self,
        lines: list[str],
        rel_path: str,
    ) -> None:
        self._lines = lines
        self._rel_path = rel_path
        self._class_stack: list[str] = []
        self.entries: list[TestBehavior] = []
        self.violations: list[str] = []

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
        if not test_evidence._is_test_function(name, self._class_stack):
            return
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            start_line = min(getattr(dec, "lineno", node.lineno) for dec in decorators)
        else:
            start_line = getattr(node, "lineno", 1)
        qualname = test_evidence._canonical_test_qualname("::".join([*self._class_stack, name]))
        test_id = f"{self._rel_path}::{qualname}"
        result = _find_behavior_label(self._lines, start_line)
        line = int(getattr(node, "lineno", 0) or 0)
        if result.kind == "error":
            self.violations.append(f"{self._rel_path}:{line}: {test_id}: {result.error}")
            return
        if result.kind != "label":
            self.violations.append(
                f"{self._rel_path}:{line}: {test_id}: missing gabion:behavior tag"
            )
            return
        self.entries.append(
            TestBehavior(
                test_id=test_id,
                path=self._rel_path,
                line=line,
                primary=result.primary,
                facets=result.facets,
            )
        )
