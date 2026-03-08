#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from gabion.runtime_shape_dispatch import json_list_or_none, json_mapping_or_none

BASELINE_VERSION = 1
WAIVER_VERSION = 1


_APPROVED_BOUNDARY_ALLOWLIST: dict[str, set[str] | None] = {
    "src/gabion/cli.py": None,
    "src/gabion/lsp_client.py": None,
    "src/gabion/server.py": None,
    "src/gabion/schema.py": None,
    "src/gabion/json_types.py": None,
}


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    column: int
    qualname: str
    kind: str
    message: str
    call: str
    structured_hash: str

    @property
    def key(self) -> str:
        return f"{self.path}:{self.qualname}:{self.kind}:{self.structured_hash}"

    @property
    def legacy_key(self) -> str:
        return f"{self.path}:{self.qualname}:{self.line}:{self.kind}"

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: [{self.qualname}] {self.message}"


@dataclass(frozen=True)
class InvalidWaiver:
    index: int
    reason: str


@dataclass(frozen=True)
class WaiverLoadResult:
    allowed_keys: set[str]
    invalid_waivers: list[InvalidWaiver]


class _RuntimeNarrowingBoundaryVisitor(ast.NodeVisitor):
    def __init__(self, *, rel_path: str, source: str) -> None:
        self._rel_path = rel_path
        self._source = source
        self.violations: list[Violation] = []
        self._qualname_stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Call(self, node: ast.Call) -> None:
        kind, call_name = _call_kind(node)
        if kind is not None and call_name is not None and not _is_allowlisted(self._rel_path, self._qualname):
            line = int(getattr(node, "lineno", 1) or 1)
            column = int(getattr(node, "col_offset", 0) or 0) + 1
            call_text = ast.get_source_segment(self._source, node) or call_name
            structural_identity = _structured_hash(
                self._rel_path,
                self._qualname,
                kind,
                call_text,
                str(column),
            )
            self.violations.append(
                Violation(
                    path=self._rel_path,
                    line=line,
                    column=column,
                    qualname=self._qualname,
                    kind=kind,
                    call=call_text,
                    structured_hash=structural_identity,
                    message=(
                        f"{call_name} runtime narrowing is boundary-only; "
                        "normalize at approved ingress modules/functions"
                    ),
                )
            )
        self.generic_visit(node)

    @property
    def _qualname(self) -> str:
        return ".".join(self._qualname_stack) if self._qualname_stack else "<module>"

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._qualname_stack.append(node.name)
        self.generic_visit(node)
        self._qualname_stack.pop()


def collect_violations(*, rel_path: str, source: str, tree: ast.AST) -> list[Violation]:
    visitor = _RuntimeNarrowingBoundaryVisitor(rel_path=rel_path, source=source)
    visitor.visit(tree)
    return visitor.violations


def _is_allowlisted(rel_path: str, qualname: str) -> bool:
    allowed_qualnames = _APPROVED_BOUNDARY_ALLOWLIST.get(rel_path)
    if allowed_qualnames is None:
        return rel_path in _APPROVED_BOUNDARY_ALLOWLIST
    return qualname in allowed_qualnames


def _call_kind(node: ast.Call) -> tuple[str | None, str | None]:
    dotted = _dotted_name(node.func)
    if dotted == "isinstance":
        return "isinstance_call", dotted
    if dotted in {"cast", "typing.cast"}:
        return "cast_call", dotted
    return None, None


def _dotted_name(node: ast.AST) -> str | None:
    match node:
        case ast.Name(id=identifier):
            return identifier
        case ast.Attribute(value=value, attr=attr):
            parent = _dotted_name(value)
            if parent is None:
                return None
            return f"{parent}.{attr}"
        case _:
            return None


def _structured_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _load_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json_mapping_or_none(json.loads(path.read_text(encoding="utf-8")))
    if payload is None:
        return set()
    raw_items = json_list_or_none(payload.get("violations"))
    if raw_items is None:
        return set()
    keys: set[str] = set()
    for item in raw_items:
        item_mapping = json_mapping_or_none(item)
        if item_mapping is None:
            continue
        path_value = str(item_mapping.get("path", "") or "")
        qualname = str(item_mapping.get("qualname", "") or "")
        kind = str(item_mapping.get("kind", "") or "")
        structured_hash = item_mapping.get("structured_hash")
        call = str(item_mapping.get("call", "") or "")
        column = item_mapping.get("column")
        line = item_mapping.get("line")
        if not path_value or not qualname or not kind:
            continue
        match structured_hash:
            case str() as structured_hash_value if structured_hash_value:
                keys.add(f"{path_value}:{qualname}:{kind}:{structured_hash_value}")
                continue
            case _:
                pass
        match column:
            case int() as column_value if call:
                migrated_hash = _structured_hash(
                    path_value,
                    qualname,
                    kind,
                    call,
                    str(column_value),
                )
                keys.add(f"{path_value}:{qualname}:{kind}:{migrated_hash}")
            case _:
                pass
        match line:
            case int() as line_value:
                keys.add(f"{path_value}:{qualname}:{line_value}:{kind}")
            case _:
                pass
    return keys


def load_waivers(path: Path) -> WaiverLoadResult:
    if not path.exists():
        return WaiverLoadResult(allowed_keys=set(), invalid_waivers=[])
    payload = json_mapping_or_none(json.loads(path.read_text(encoding="utf-8")))
    if payload is None:
        return WaiverLoadResult(allowed_keys=set(), invalid_waivers=[InvalidWaiver(index=0, reason="waiver payload must be an object")])

    waivers_raw = json_list_or_none(payload.get("waivers"))
    if waivers_raw is None:
        return WaiverLoadResult(allowed_keys=set(), invalid_waivers=[InvalidWaiver(index=0, reason="waivers must be a list")])

    required = ("path", "qualname", "kind", "rationale", "scope", "expiry", "owner")
    allowed_keys: set[str] = set()
    invalid_waivers: list[InvalidWaiver] = []
    for index, waiver in enumerate(waivers_raw, start=1):
        waiver_mapping = json_mapping_or_none(waiver)
        if waiver_mapping is None:
            invalid_waivers.append(InvalidWaiver(index=index, reason="waiver must be an object"))
            continue
        missing = [field for field in required if field not in waiver_mapping]
        if missing:
            invalid_waivers.append(InvalidWaiver(index=index, reason=f"missing fields: {', '.join(missing)}"))
            continue

        path_value = str(waiver_mapping.get("path", "") or "")
        qualname = str(waiver_mapping.get("qualname", "") or "")
        kind = str(waiver_mapping.get("kind", "") or "")
        structured_hash = waiver_mapping.get("structured_hash")
        line = waiver_mapping.get("line")
        if not path_value or not qualname or kind not in {"isinstance_call", "cast_call"}:
            invalid_waivers.append(InvalidWaiver(index=index, reason="path, qualname, and valid kind are required"))
            continue
        match structured_hash:
            case str() as structured_hash_value if structured_hash_value:
                allowed_keys.add(f"{path_value}:{qualname}:{kind}:{structured_hash_value}")
                continue
            case _:
                pass
        call = str(waiver_mapping.get("call", "") or "")
        column = waiver_mapping.get("column")
        match column:
            case int() as column_value if call:
                migrated_hash = _structured_hash(
                    path_value,
                    qualname,
                    kind,
                    call,
                    str(column_value),
                )
                allowed_keys.add(f"{path_value}:{qualname}:{kind}:{migrated_hash}")
                continue
            case _:
                pass
        match line:
            case int() as line_value:
                allowed_keys.add(f"{path_value}:{qualname}:{line_value}:{kind}")
            case _:
                invalid_waivers.append(InvalidWaiver(index=index, reason="line_or_structured_hash_required"))
                continue

    return WaiverLoadResult(allowed_keys=allowed_keys, invalid_waivers=invalid_waivers)


__all__ = [
    "BASELINE_VERSION",
    "WAIVER_VERSION",
    "InvalidWaiver",
    "Violation",
    "WaiverLoadResult",
    "collect_violations",
    "load_waivers",
]
