# gabion:decision_protocol_module
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

from gabion.analysis.semantics import evidence_keys
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.order_contract import sort_once

EVIDENCE_TAG = "gabion:evidence"
_TAG_RE = re.compile(r"#\s*gabion:evidence\s+(?P<ids>.+)")
_LEGACY_TEST_ID_RE = re.compile(r"(?P<path>tests/test_[A-Za-z0-9_]+\.py)::")
_QUALNAME_COMPAT_REWRITE = ("legacy_dataflow_monolith", "dataflow_audit")
_LEGACY_PATH_ALIAS = {
    "tests/gabion/analysis/dataflow_raw_runtime_edges_cases.py": "tests/test_dataflow_run_edges.py",
    "tests/gabion/analysis/dataflow_raw_runtime_cases.py": "tests/test_dataflow_run.py",
    "tests/gabion/analysis/dataflow_structure_reuse_cases.py": "tests/test_structure_reuse.py",
    "tests/gabion/analysis/dataflow_structure_reuse_edges_cases.py": "tests/test_structure_reuse_edges.py",
    "tests/gabion/analysis/evidence_suggestions_edges_cases.py": "tests/test_test_evidence_suggestions_edges.py",
    "tests/gabion/cli/cli_payload_cases.py": "tests/test_cli_payloads.py",
    "tests/gabion/ingest/test_adapter_contract.py": "tests/test_ingest_adapter_contract.py",
    "tests/gabion/ingest/test_registry.py": "tests/test_ingest_registry.py",
    "tests/gabion/refactor/engine_cases.py": "tests/test_refactor_engine.py",
    "tests/gabion/refactor/engine_edges_cases.py": "tests/test_refactor_engine_edges.py",
    "tests/gabion/refactor/engine_helpers_cases.py": "tests/test_refactor_engine_helpers.py",
    "tests/gabion/refactor/engine_more_cases.py": "tests/test_refactor_engine_more.py",
    "tests/gabion/refactor/test_engine.py": "tests/test_refactor_engine.py",
    "tests/gabion/refactor/test_idempotency.py": "tests/test_refactor_idempotency.py",
    "tests/gabion/refactor/test_plan.py": "tests/test_refactor_plan.py",
    "tests/gabion/lsp_client/lsp_client_smoke_cases.py": "tests/test_lsp_smoke.py",
    "tests/gabion/server/server_code_action_stub_cases.py": "tests/test_code_action_stub.py",
    "tests/gabion/server_core/command_orchestrator_coverage_cases.py": "tests/test_server_core_orchestrator_coverage.py",
    "tests/gabion/server_core/command_orchestrator_edges_cases.py": "tests/test_server_core_orchestrator_edges.py",
    "tests/gabion/server_core/test_command_reducers.py": "tests/test_server_core_reducers.py",
    "tests/gabion/synthesis/merge_cases.py": "tests/test_synthesis_merge.py",
    "tests/gabion/synthesis/merge_integration_cases.py": "tests/test_synthesis_merge_integration.py",
    "tests/gabion/synthesis/test_contextvar_emission.py": "tests/test_synthesis_contextvar_emission.py",
    "tests/gabion/synthesis/test_control_context.py": "tests/test_synthesis_control_context.py",
    "tests/gabion/synthesis/test_merge.py": "tests/test_synthesis_merge.py",
    "tests/gabion/synthesis/test_naming.py": "tests/test_synthesis_naming.py",
    "tests/gabion/synthesis/test_protocols.py": "tests/test_synthesis_protocols.py",
    "tests/gabion/synthesis/test_schedule.py": "tests/test_synthesis_schedule.py",
    "tests/gabion/synthesis/test_stubs.py": "tests/test_synthesis_stubs.py",
    "tests/gabion/synthesis/test_tiers.py": "tests/test_synthesis_tiers.py",
    "tests/gabion/synthesis/test_types.py": "tests/test_synthesis_types.py",
    "tests/gabion/test_root_plan.py": "tests/test_plan.py",
    "tests/gabion/tooling/test_emit_advisory_and_gates.py": "tests/test_tooling_emit_advisory_and_gates.py",
    "tests/gabion/tooling/test_governance_rules.py": "tests/test_governance_rules_policy.py",
}


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
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
) -> dict[str, object]:
    check_deadline()
    root = root.resolve()
    display_root = "."
    exclude_set = {str(item) for item in exclude}
    include_list = [str(item) for item in include]
    files = _collect_test_files(paths, root=root, exclude=exclude_set)
    entries: list[TestEvidence] = []
    for path in files:
        entries.extend(_extract_file_evidence(path, root))

    test_ids = [entry.test_id for entry in entries]
    duplicates = sort_once(
        {
            test_id
            for test_id, count in Counter(test_ids).items()
            if count > 1
        }, 
    source = 'src/gabion/analysis/test_evidence.py:67')
    if duplicates:
        preview = ", ".join(duplicates[:5])
        suffix = "" if len(duplicates) <= 5 else f" (+{len(duplicates) - 5} more)"
        raise ValueError(f"Duplicate test_id entries found: {preview}{suffix}")

    tests_sorted = sort_once(entries, key=lambda entry: entry.test_id, source = 'src/gabion/analysis/test_evidence.py:79')
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
    for identity in sort_once(evidence_index, source = 'src/gabion/analysis/test_evidence.py:108'):
        record = evidence_index[identity]
        tests = sort_once(record["tests"], source = 'src/gabion/analysis/test_evidence.py:110')
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
            "exclude": sort_once(exclude_set, source = 'src/gabion/analysis/test_evidence.py:124'),
        },
        "tests": tests_payload,
        "evidence_index": evidence_payload,
    }


def collect_test_tags(
    paths: Iterable[Path],
    *,
    root: Path,
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
) -> list[TestEvidenceTag]:
    check_deadline()
    root = root.resolve()
    exclude_set = {str(item) for item in exclude}
    files = _collect_test_files(paths, root=root, exclude=exclude_set)
    entries: list[TestEvidenceTag] = []
    for path in files:
        entries.extend(_extract_file_tags(path, root))
    return sort_once(entries, key=lambda entry: entry.test_id, source = 'src/gabion/analysis/test_evidence.py:145')


def write_test_evidence(
    payload: dict[str, object],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
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
    seen: set[Path] = set()
    for path in paths:
        if path.is_dir():
            candidates = sort_once(
                [*path.rglob("test_*.py"), *path.rglob("*_cases.py")],
                source="_collect_test_files.candidates",
                key=lambda item: str(item),
            )
            for candidate in candidates:
                if candidate in seen:
                    continue
                if _should_exclude(candidate, root, exclude):
                    continue
                seen.add(candidate)
                files.append(candidate)
        else:
            if path.suffix == ".py" and not _should_exclude(path, root, exclude):
                if path not in seen:
                    seen.add(path)
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
    canonical_rel_path = _canonical_test_rel_path(rel_path, comments)
    collector = _TestCollector(lines, comments, canonical_rel_path)
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
    canonical_rel_path = _canonical_test_rel_path(rel_path, comments)
    collector = _TagCollector(lines, comments, canonical_rel_path)
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


def _legacy_test_paths_from_tags(tags: Iterable[str]) -> set[str]:
    check_deadline()
    paths: set[str] = set()
    for tag in tags:
        check_deadline()
        for match in _LEGACY_TEST_ID_RE.finditer(tag):
            check_deadline()
            paths.add(match.group("path"))
    return paths


def _legacy_path_fallback(rel_path: str) -> str:
    aliased = _LEGACY_PATH_ALIAS.get(rel_path)
    if aliased is not None:
        return aliased
    if not rel_path.startswith("tests/gabion/"):
        return rel_path
    basename = Path(rel_path).name
    if basename == "test_integration.py":
        return rel_path
    if basename.startswith("test_"):
        return f"tests/{basename}"
    if basename.endswith("_cases.py"):
        stem = basename[: -len("_cases.py")]
        return f"tests/test_{stem}.py"
    return rel_path


def _canonical_test_rel_path(rel_path: str, comment_map: dict[int, list[str]]) -> str:
    check_deadline()
    comment_tags = [tag for tags in comment_map.values() for tag in tags]
    legacy_paths = sort_once(
        _legacy_test_paths_from_tags(comment_tags),
        source="test_evidence._canonical_test_rel_path.legacy_paths",
    )
    fallback = _legacy_path_fallback(rel_path)
    if not legacy_paths:
        return fallback
    if len(legacy_paths) == 1:
        return legacy_paths[0]
    if fallback in legacy_paths:
        return fallback
    return legacy_paths[0]


def _canonical_test_qualname(qualname: str) -> str:
    old_token, new_token = _QUALNAME_COMPAT_REWRITE
    return qualname.replace(old_token, new_token)


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
    items = sort_once(
        items,
        source="test_evidence._normalize_evidence_items.items",
        # Lexical evidence identity key stabilizes normalized evidence rows.
        key=lambda item: item.identity,
    )
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
        qualname = _canonical_test_qualname("::".join([*self._class_stack, name]))
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
        qualname = _canonical_test_qualname("::".join([*self._class_stack, name]))
        test_id = f"{self._rel_path}::{qualname}"
        self.entries.append(
            TestEvidenceTag(
                test_id=test_id,
                path=self._rel_path,
                line=getattr(node, "lineno", 0),
                tags=tags,
            )
        )
