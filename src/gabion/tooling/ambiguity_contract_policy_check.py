#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path

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
        return f"{self.rule_id}:{self.path}:{self.qualname}:{self.line}"

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
        if isinstance(node.func, ast.Name) and node.func.id == "isinstance":
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


def _annotation_is_dynamic(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in {"Any", "Optional", "Union"}
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id in {"Optional", "Union"}:
            return True
        return _annotation_is_dynamic(node.value) or _annotation_is_dynamic(node.slice)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return True
    for child in ast.iter_child_nodes(node):
        if _annotation_is_dynamic(child):
            return True
    return False


def _looks_like_guard(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "isinstance":
            return True
        if isinstance(child, ast.Compare):
            values = [child.left, *child.comparators]
            for value in values:
                if isinstance(value, ast.Constant) and value.value is None:
                    return True
    return False


def _single_sentinel_stmt(body: list[ast.stmt]) -> str | None:
    if len(body) != 1:
        return None
    stmt = body[0]
    if isinstance(stmt, ast.Return):
        if stmt.value is None:
            return "return None"
        if isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
            return "return None"
        if isinstance(stmt.value, ast.List) and len(stmt.value.elts) == 0:
            return "return []"
    if isinstance(stmt, ast.Continue):
        return "continue"
    if isinstance(stmt, ast.Pass):
        return "pass"
    return None


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
    raw = payload.get("violations", []) if isinstance(payload, dict) else []
    keys: set[str] = set()
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            rule_id = item.get("rule_id")
            path_value = item.get("path")
            qualname = item.get("qualname")
            line = item.get("line")
            if isinstance(rule_id, str) and isinstance(path_value, str) and isinstance(qualname, str) and isinstance(line, int):
                keys.add(f"{rule_id}:{path_value}:{qualname}:{line}")
    return keys


def _write_baseline(path: Path, violations: list[Violation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": BASELINE_VERSION,
        "violations": [
            asdict(item)
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
