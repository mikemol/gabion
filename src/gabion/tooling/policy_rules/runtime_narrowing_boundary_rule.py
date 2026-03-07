#!/usr/bin/env python3
# gabion:boundary_normalization_module gabion:decision_protocol_module
from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return set()
    raw_items = payload.get("violations")
    if not isinstance(raw_items, list):
        return set()
    keys: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        path_value = str(item.get("path", "") or "")
        qualname = str(item.get("qualname", "") or "")
        kind = str(item.get("kind", "") or "")
        structured_hash = item.get("structured_hash")
        call = str(item.get("call", "") or "")
        column = item.get("column")
        line = item.get("line")
        if not path_value or not qualname or not kind:
            continue
        if isinstance(structured_hash, str) and structured_hash:
            keys.add(f"{path_value}:{qualname}:{kind}:{structured_hash}")
            continue
        if call and isinstance(column, int):
            migrated_hash = _structured_hash(
                path_value,
                qualname,
                kind,
                call,
                str(column),
            )
            keys.add(f"{path_value}:{qualname}:{kind}:{migrated_hash}")
        if isinstance(line, int):
            keys.add(f"{path_value}:{qualname}:{line}:{kind}")
    return keys


def load_waivers(path: Path) -> WaiverLoadResult:
    if not path.exists():
        return WaiverLoadResult(allowed_keys=set(), invalid_waivers=[])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return WaiverLoadResult(allowed_keys=set(), invalid_waivers=[InvalidWaiver(index=0, reason="waiver payload must be an object")])

    waivers_raw = payload.get("waivers")
    if not isinstance(waivers_raw, list):
        return WaiverLoadResult(allowed_keys=set(), invalid_waivers=[InvalidWaiver(index=0, reason="waivers must be a list")])

    required = ("path", "qualname", "kind", "rationale", "scope", "expiry", "owner")
    allowed_keys: set[str] = set()
    invalid_waivers: list[InvalidWaiver] = []
    for index, waiver in enumerate(waivers_raw, start=1):
        if not isinstance(waiver, dict):
            invalid_waivers.append(InvalidWaiver(index=index, reason="waiver must be an object"))
            continue
        missing = [field for field in required if field not in waiver]
        if missing:
            invalid_waivers.append(InvalidWaiver(index=index, reason=f"missing fields: {', '.join(missing)}"))
            continue

        path_value = str(waiver.get("path", "") or "")
        qualname = str(waiver.get("qualname", "") or "")
        kind = str(waiver.get("kind", "") or "")
        structured_hash = waiver.get("structured_hash")
        line = waiver.get("line")
        if not path_value or not qualname or kind not in {"isinstance_call", "cast_call"}:
            invalid_waivers.append(InvalidWaiver(index=index, reason="path, qualname, and valid kind are required"))
            continue
        if isinstance(structured_hash, str) and structured_hash:
            allowed_keys.add(f"{path_value}:{qualname}:{kind}:{structured_hash}")
            continue
        call = str(waiver.get("call", "") or "")
        column = waiver.get("column")
        if call and isinstance(column, int):
            migrated_hash = _structured_hash(
                path_value,
                qualname,
                kind,
                call,
                str(column),
            )
            allowed_keys.add(f"{path_value}:{qualname}:{kind}:{migrated_hash}")
            continue
        if not isinstance(line, int):
            invalid_waivers.append(InvalidWaiver(index=index, reason="line_or_structured_hash_required"))
            continue
        allowed_keys.add(f"{path_value}:{qualname}:{line}:{kind}")

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
