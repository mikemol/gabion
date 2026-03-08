#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from functools import singledispatch
from pathlib import Path

from gabion.invariants import never

TARGETS = (
    "src/gabion/analysis/**/*.py",
    "src/gabion/synthesis/**/*.py",
    "src/gabion/refactor/**/*.py",
)
BASELINE_VERSION = 1
MODULE_MARKER = "gabion:ambiguity_boundary_module"
FUNCTION_MARKER = "gabion:ambiguity_boundary"


@dataclass(frozen=True)
class Violation:
    rule_id: str
    path: str
    line: int
    column: int
    qualname: str
    message: str

    @property
    def key(self) -> str:
        # Baseline identity is line-insensitive so routine line motion in
        # unchanged semantics does not churn policy deltas.
        return f"{self.rule_id}:{self.path}:{self.qualname}"

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: [{self.rule_id}] [{self.qualname}] {self.message}"


@dataclass(frozen=True)
class _Scope:
    qualname: str
    is_boundary: bool


class _Visitor(ast.NodeVisitor):
    def __init__(self, *, rel_path: str, source_lines: list[str]) -> None:
        self.rel_path = rel_path
        self.source_lines = source_lines
        self.violations: list[Violation] = []
        self.module_boundary = _module_boundary(source_lines)
        self.scope_stack: list[_Scope] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._scope_boundary:
            self.generic_visit(node)
            return
        if _is_isinstance_call(node):
            self._report(node, rule_id="ACP-003", message="runtime type narrowing in deterministic core")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not self._scope_boundary:
            self._check_annotation(node.annotation, node)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        if not self._scope_boundary and node.annotation is not None:
            self._check_annotation(node.annotation, node)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if self._scope_boundary:
            self.generic_visit(node)
            return
        if _looks_like_guard(node.test):
            sentinel = _single_sentinel_stmt(node.body)
            if sentinel is not None:
                self._report(node, rule_id="ACP-002", message=f"sentinel control outcome in core ({sentinel})")
        self.generic_visit(node)

    @property
    def _scope_boundary(self) -> bool:
        if self.scope_stack:
            return self.scope_stack[-1].is_boundary
        return self.module_boundary

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        parent = self.scope_stack[-1].qualname if self.scope_stack else "<module>"
        qualname = node.name if parent == "<module>" else f"{parent}.{node.name}"
        is_boundary = self.module_boundary or _has_marker(self.source_lines, int(getattr(node, "lineno", 1)), FUNCTION_MARKER)
        self.scope_stack.append(_Scope(qualname=qualname, is_boundary=is_boundary))
        if not is_boundary and node.returns is not None:
            self._check_annotation(node.returns, node)
        self.generic_visit(node)
        self.scope_stack.pop()

    def _check_annotation(self, annotation: ast.AST, node: ast.AST) -> None:
        if _annotation_is_dynamic(annotation):
            self._report(node, rule_id="ACP-004", message="dynamic type alternation in deterministic core annotation")

    def _report(self, node: ast.AST, *, rule_id: str, message: str) -> None:
        scope = self.scope_stack[-1] if self.scope_stack else _Scope("<module>", self.module_boundary)
        self.violations.append(
            Violation(
                rule_id=rule_id,
                path=self.rel_path,
                line=int(getattr(node, "lineno", 1)),
                column=int(getattr(node, "col_offset", 0)) + 1,
                qualname=scope.qualname,
                message=message,
            )
        )


def _module_boundary(source_lines: list[str]) -> bool:
    for raw in source_lines[:80]:
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#") and MODULE_MARKER in stripped:
            return True
        if stripped.startswith("\"\"\"") or stripped.startswith("'''"):
            continue
    return False


def _has_marker(source_lines: list[str], line: int, marker: str) -> bool:
    idx = max(0, line - 2)
    while idx >= 0:
        stripped = source_lines[idx].strip()
        if not stripped:
            idx -= 1
            continue
        return stripped.startswith("#") and marker in stripped
    return False


@singledispatch
def _name_id_or_none(node: object) -> str | None:
    never("unregistered runtime type", value_type=type(node).__name__)


@_name_id_or_none.register(ast.Name)
def _(node: ast.Name) -> str | None:
    return node.id


@_name_id_or_none.register(ast.AST)
def _(node: ast.AST) -> str | None:
    _ = node
    return None


@singledispatch
def _is_isinstance_call(node: object) -> bool:
    never("unregistered runtime type", value_type=type(node).__name__)


@_is_isinstance_call.register(ast.Call)
def _(node: ast.Call) -> bool:
    return _name_id_or_none(node.func) == "isinstance"


@_is_isinstance_call.register(ast.AST)
def _(node: ast.AST) -> bool:
    _ = node
    return False


@singledispatch
def _is_none_constant(node: object) -> bool:
    never("unregistered runtime type", value_type=type(node).__name__)


@_is_none_constant.register(type(None))
def _(node: None) -> bool:
    _ = node
    return True


@_is_none_constant.register(ast.Constant)
def _(node: ast.Constant) -> bool:
    return node.value is None


@_is_none_constant.register(ast.AST)
def _(node: ast.AST) -> bool:
    _ = node
    return False


@singledispatch
def _compare_contains_none_constant(node: object) -> bool:
    never("unregistered runtime type", value_type=type(node).__name__)


@_compare_contains_none_constant.register(ast.Compare)
def _(node: ast.Compare) -> bool:
    values = [node.left, *node.comparators]
    for value in values:
        if _is_none_constant(value):
            return True
    return False


@_compare_contains_none_constant.register(ast.AST)
def _(node: ast.AST) -> bool:
    _ = node
    return False


@singledispatch
def _subscript_value_is_dynamic_alias(node: object) -> bool:
    never("unregistered runtime type", value_type=type(node).__name__)


@_subscript_value_is_dynamic_alias.register(ast.Name)
def _(node: ast.Name) -> bool:
    return node.id in {"Optional", "Union"}


@_subscript_value_is_dynamic_alias.register(ast.AST)
def _(node: ast.AST) -> bool:
    _ = node
    return False


@singledispatch
def _is_bit_or_operator(node: object) -> bool:
    never("unregistered runtime type", value_type=type(node).__name__)


@_is_bit_or_operator.register(ast.BitOr)
def _(node: ast.BitOr) -> bool:
    _ = node
    return True


@_is_bit_or_operator.register(ast.operator)
def _(node: ast.operator) -> bool:
    _ = node
    return False


@singledispatch
def _annotation_is_dynamic(node: object) -> bool:
    never("unregistered runtime type", value_type=type(node).__name__)


@_annotation_is_dynamic.register(ast.Name)
def _(node: ast.Name) -> bool:
    return node.id in {"Any", "Optional", "Union"}


@_annotation_is_dynamic.register(ast.Subscript)
def _(node: ast.Subscript) -> bool:
    if _subscript_value_is_dynamic_alias(node.value):
        return True
    return _annotation_is_dynamic(node.value) or _annotation_is_dynamic(node.slice)


@_annotation_is_dynamic.register(ast.BinOp)
def _(node: ast.BinOp) -> bool:
    if _is_bit_or_operator(node.op):
        return True
    return _annotation_is_dynamic(node.left) or _annotation_is_dynamic(node.right)


@_annotation_is_dynamic.register(ast.AST)
def _(node: ast.AST) -> bool:
    for child in ast.iter_child_nodes(node):
        if _annotation_is_dynamic(child):
            return True
    return False


def _looks_like_guard(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if _is_isinstance_call(child):
            return True
        if _compare_contains_none_constant(child):
            return True
    return False


@singledispatch
def _sentinel_return_value(value: object) -> str | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_sentinel_return_value.register(type(None))
def _(value: None) -> str | None:
    _ = value
    return "return None"


@_sentinel_return_value.register(ast.Constant)
def _(value: ast.Constant) -> str | None:
    if value.value is None:
        return "return None"
    return None


@_sentinel_return_value.register(ast.List)
def _(value: ast.List) -> str | None:
    if len(value.elts) == 0:
        return "return []"
    return None


@_sentinel_return_value.register(ast.AST)
def _(value: ast.AST) -> str | None:
    _ = value
    return None


@singledispatch
def _sentinel_stmt_value(stmt: object) -> str | None:
    never("unregistered runtime type", value_type=type(stmt).__name__)


@_sentinel_stmt_value.register(ast.Return)
def _(stmt: ast.Return) -> str | None:
    return _sentinel_return_value(stmt.value)


@_sentinel_stmt_value.register(ast.Continue)
def _(stmt: ast.Continue) -> str | None:
    _ = stmt
    return "continue"


@_sentinel_stmt_value.register(ast.Pass)
def _(stmt: ast.Pass) -> str | None:
    _ = stmt
    return "pass"


@_sentinel_stmt_value.register(ast.stmt)
def _(stmt: ast.stmt) -> str | None:
    _ = stmt
    return None


def _single_sentinel_stmt(body: list[ast.stmt]) -> str | None:
    if len(body) != 1:
        return None
    return _sentinel_stmt_value(body[0])


@singledispatch
def _dict_or_none(value: object) -> dict[object, object] | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_dict_or_none.register(dict)
def _(value: dict[object, object]) -> dict[object, object] | None:
    return value


def _none_dict(value: object) -> dict[object, object] | None:
    _ = value
    return None


for _dict_none_type in (list, tuple, set, str, int, float, bool, type(None)):
    _dict_or_none.register(_dict_none_type)(_none_dict)


@singledispatch
def _list_or_none(value: object) -> list[object] | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_list_or_none.register(list)
def _(value: list[object]) -> list[object] | None:
    return value


@_list_or_none.register(tuple)
def _(value: tuple[object, ...]) -> list[object] | None:
    return list(value)


def _none_list(value: object) -> list[object] | None:
    _ = value
    return None


for _list_none_type in (dict, set, str, int, float, bool, type(None)):
    _list_or_none.register(_list_none_type)(_none_list)


@singledispatch
def _str_or_none(value: object) -> str | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_str_or_none.register(str)
def _(value: str) -> str | None:
    return value


def _none_str(value: object) -> str | None:
    _ = value
    return None


for _str_none_type in (dict, list, tuple, set, int, float, bool, type(None)):
    _str_or_none.register(_str_none_type)(_none_str)


def _baseline_violation_entries(payload: object) -> list[dict[object, object]]:
    payload_mapping = _dict_or_none(payload)
    if payload_mapping is None:
        return []
    raw = _list_or_none(payload_mapping.get("violations"))
    if raw is None:
        return []
    entries: list[dict[object, object]] = []
    for item in raw:
        mapping = _dict_or_none(item)
        if mapping is not None:
            entries.append(mapping)
    return entries


def collect_violations(root: Path) -> list[Violation]:
    violations: list[Violation] = []
    for pattern in TARGETS:
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or any(part == "__pycache__" for part in path.parts):
                continue
            rel = path.relative_to(root).as_posix()
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            visitor = _Visitor(rel_path=rel, source_lines=source.splitlines())
            visitor.visit(tree)
            violations.extend(visitor.violations)
    return violations


def _load_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    keys: set[str] = set()
    for item in _baseline_violation_entries(payload):
        rule_id = _str_or_none(item.get("rule_id"))
        path_value = _str_or_none(item.get("path"))
        qualname = _str_or_none(item.get("qualname"))
        if rule_id is not None and path_value is not None and qualname is not None:
            keys.add(f"{rule_id}:{path_value}:{qualname}")
    return keys


def _write_baseline(path: Path, violations: list[Violation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": BASELINE_VERSION,
        "violations": [
            {
                "rule_id": item.rule_id,
                "path": item.path,
                "qualname": item.qualname,
            }
            for item in sorted(violations, key=lambda v: (v.rule_id, v.path, v.qualname, v.line, v.column))
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(root: Path, baseline: Path | None, baseline_write: bool) -> int:
    violations = collect_violations(root)
    if baseline_write:
        if baseline is None:
            raise SystemExit("--baseline is required with --baseline-write")
        _write_baseline(baseline, violations)
        print(f"wrote ambiguity-contract baseline: {baseline}")
        return 0

    if baseline is None:
        baseline_keys: set[str] = set()
    else:
        baseline_keys = _load_baseline(baseline)

    new_violations = [v for v in violations if v.key not in baseline_keys]
    if new_violations:
        print("ambiguity contract policy violations:")
        for item in sorted(new_violations, key=lambda v: (v.rule_id, v.path, v.line, v.column)):
            print(f"  - {item.render()}")
        return 1
    print("ambiguity contract policy check passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shift-ambiguity-left contract policy check")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-write", action="store_true")
    args = parser.parse_args(argv)
    return run(root=args.root.resolve(), baseline=args.baseline, baseline_write=bool(args.baseline_write))


if __name__ == "__main__":
    raise SystemExit(main())
