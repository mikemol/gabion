#!/usr/bin/env python3
# gabion:decision_protocol_module
from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

BASELINE_VERSION = 1
WAIVER_VERSION = 1


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    column: int
    qualname: str
    kind: str
    message: str
    scope: str
    annotation: str
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


class _TypingSurfaceVisitor(ast.NodeVisitor):
    def __init__(self, *, rel_path: str, source: str) -> None:
        self._rel_path = rel_path
        self._source = source
        self._scope = _forbidden_scope(rel_path)
        self.violations: list[Violation] = []
        self._qualname_stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_annotation(node.annotation, node=node)
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._qualname_stack.append(node.name)
        self._check_annotation(node.returns, node=node)
        for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            self._check_annotation(arg.annotation, node=arg)
        self._check_annotation(node.args.vararg.annotation if node.args.vararg is not None else None, node=node)
        self._check_annotation(node.args.kwarg.annotation if node.args.kwarg is not None else None, node=node)
        self.generic_visit(node)
        self._qualname_stack.pop()

    def _check_annotation(self, annotation: ast.AST | None, *, node: ast.AST) -> None:
        if annotation is None or self._scope is None:
            return
        kind = _annotation_kind(annotation)
        if kind is None:
            return
        line = int(getattr(node, "lineno", 1) or 1)
        column = int(getattr(node, "col_offset", 0) or 0) + 1
        qualname = ".".join(self._qualname_stack) if self._qualname_stack else "<module>"
        annotation_text = ast.get_source_segment(self._source, annotation) or "<unknown>"
        structural_identity = _structured_hash(
            self._rel_path,
            qualname,
            kind,
            self._scope,
            annotation_text,
            str(column),
        )
        self.violations.append(
            Violation(
                path=self._rel_path,
                line=line,
                column=column,
                qualname=qualname,
                kind=kind,
                scope=self._scope,
                annotation=annotation_text,
                message=(
                    f"{annotation_text!r} annotation is forbidden in {self._scope}; "
                    "use DTO/Protocol/dataclass carrier or TypedDict/Pydantic model"
                ),
                structured_hash=structural_identity,
            )
        )


def _forbidden_scope(rel_path: str) -> str | None:
    if rel_path.startswith("src/gabion/analysis/"):
        return "semantic_core"
    if "/stages/" in rel_path or "stage_contract" in rel_path or rel_path.endswith("_stage.py"):
        return "stage_contracts"
    if "/reducers/" in rel_path or rel_path.endswith("_reducer.py"):
        return "reducers"
    return None


def _annotation_kind(annotation: ast.AST) -> str | None:
    if _is_any(annotation):
        return "any_annotation"
    if _is_bare_object(annotation):
        return "bare_object_annotation"
    if _is_dict_str_object(annotation):
        return "dict_str_object_annotation"
    return None


def _is_any(node: ast.AST) -> bool:
    match node:
        case ast.Name(id="Any"):
            return True
        case ast.Attribute(attr="Any"):
            return True
        case _:
            return False


def _is_bare_object(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "object"


def _is_dict_str_object(node: ast.AST) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    base_name = _dotted_name(node.value)
    if base_name not in {"dict", "typing.Dict", "Dict"}:
        return False
    slice_node = node.slice
    if isinstance(slice_node, ast.Tuple) and len(slice_node.elts) == 2:
        key_node, value_node = slice_node.elts
    else:
        return False
    return _is_str_node(key_node) and _is_bare_object(value_node)


def _is_str_node(node: ast.AST) -> bool:
    match node:
        case ast.Name(id="str"):
            return True
        case ast.Constant(value="str"):
            return True
        case _:
            return False


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
        scope = str(item.get("scope", "") or "")
        annotation = str(item.get("annotation", "") or "")
        column = item.get("column")
        line = item.get("line")
        if not path_value or not qualname or not kind:
            continue
        match structured_hash:
            case str() as structured_hash_value if structured_hash_value:
                keys.add(f"{path_value}:{qualname}:{kind}:{structured_hash_value}")
                continue
            case _:
                pass
        match column:
            case int() as column_value if scope and annotation:
                migrated_hash = _structured_hash(
                    path_value,
                    qualname,
                    kind,
                    scope,
                    annotation,
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return WaiverLoadResult(allowed_keys=set(), invalid_waivers=[InvalidWaiver(index=0, reason="waiver_file_not_object")])
    raw_waivers = payload.get("waivers")
    if not isinstance(raw_waivers, list):
        return WaiverLoadResult(allowed_keys=set(), invalid_waivers=[InvalidWaiver(index=0, reason="waivers_not_list")])

    keys: set[str] = set()
    invalid: list[InvalidWaiver] = []
    required = ("path", "qualname", "kind", "rationale", "scope", "expiry", "owner")
    for index, raw in enumerate(raw_waivers, start=1):
        if not isinstance(raw, dict):
            invalid.append(InvalidWaiver(index=index, reason="waiver_not_object"))
            continue
        missing = [field for field in required if raw.get(field) in (None, "")]
        if missing:
            invalid.append(InvalidWaiver(index=index, reason=f"missing_fields:{','.join(missing)}"))
            continue
        structured_hash = raw.get("structured_hash")
        line = raw.get("line")
        match structured_hash:
            case str() as structured_hash_value if structured_hash_value:
                key = f"{raw['path']}:{raw['qualname']}:{raw['kind']}:{structured_hash_value}"
                keys.add(key)
                continue
            case _:
                pass
        scope = str(raw.get("scope", "") or "")
        annotation = str(raw.get("annotation", "") or "")
        column = raw.get("column")
        match column:
            case int() as column_value if scope and annotation:
                migrated_hash = _structured_hash(
                    str(raw["path"]),
                    str(raw["qualname"]),
                    str(raw["kind"]),
                    scope,
                    annotation,
                    str(column_value),
                )
                keys.add(f"{raw['path']}:{raw['qualname']}:{raw['kind']}:{migrated_hash}")
                continue
            case _:
                pass
        match line:
            case int() as line_value:
                key = f"{raw['path']}:{raw['qualname']}:{line_value}:{raw['kind']}"
                keys.add(key)
            case _:
                invalid.append(InvalidWaiver(index=index, reason="line_or_structured_hash_required"))
                continue
    return WaiverLoadResult(allowed_keys=keys, invalid_waivers=invalid)


def collect_violations(*, rel_path: str, source: str, tree: ast.AST) -> list[Violation]:
    visitor = _TypingSurfaceVisitor(rel_path=rel_path, source=source)
    visitor.visit(tree)
    return visitor.violations
