#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

TARGET_GLOB = "src/gabion/**/*.py"
BASELINE_VERSION = 1
MODULE_MARKER = "gabion:decision_protocol_module"
FUNCTION_MARKER = "gabion:decision_protocol"
DECORATOR_NAMES = {
    "decision_protocol",
    "invariants.decision_protocol",
    "gabion.invariants.decision_protocol",
}
BRANCH_NODE_TYPES = (
    ast.If,
    ast.IfExp,
    ast.Match,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.TryStar,
)


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    column: int
    qualname: str
    kind: str
    message: str

    @property
    def key(self) -> str:
        return f"{self.path}:{self.qualname}:{self.line}:{self.kind}"

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: [{self.qualname}] {self.message}"


@dataclass(frozen=True)
class _Scope:
    qualname: str
    is_decision_protocol: bool


class _BranchlessVisitor(ast.NodeVisitor):
    def __init__(self, *, rel_path: str, source_lines: list[str]) -> None:
        self.rel_path = rel_path
        self.source_lines = source_lines
        self.violations: list[Violation] = []
        self.module_decision_protocol = _module_is_decision_protocol(source_lines)
        self.scope_stack: list[_Scope] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_node(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_node(node)

    def visit_If(self, node: ast.If) -> None:
        self._record_branch(node, kind="if")
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._record_branch(node, kind="ifexp")
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self._record_branch(node, kind="match")
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._record_branch(node, kind="for")
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._record_branch(node, kind="async_for")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._record_branch(node, kind="while")
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._record_branch(node, kind="try")
        self.generic_visit(node)

    def visit_TryStar(self, node: ast.TryStar) -> None:
        self._record_branch(node, kind="try_star")
        self.generic_visit(node)

    def _visit_function_node(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        parent_qual = self.scope_stack[-1].qualname if self.scope_stack else "<module>"
        qualname = node.name if parent_qual == "<module>" else f"{parent_qual}.{node.name}"
        is_decision_protocol = self._function_is_decision_protocol(node)
        self.scope_stack.append(_Scope(qualname=qualname, is_decision_protocol=is_decision_protocol))
        self.generic_visit(node)
        self.scope_stack.pop()

    def _function_is_decision_protocol(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        if self.module_decision_protocol:
            return True
        if _has_preceding_marker(
            source_lines=self.source_lines,
            line=int(getattr(node, "lineno", 1)),
            marker=FUNCTION_MARKER,
        ):
            return True
        for decorator in node.decorator_list:
            dotted = _dotted_name(decorator)
            if dotted in DECORATOR_NAMES:
                return True
            if isinstance(decorator, ast.Call):
                call_name = _dotted_name(decorator.func)
                if call_name in DECORATOR_NAMES:
                    return True
        return False

    def _record_branch(self, node: ast.AST, *, kind: str) -> None:
        scope = self.scope_stack[-1] if self.scope_stack else _Scope("<module>", self.module_decision_protocol)
        if scope.is_decision_protocol:
            return
        self.violations.append(
            Violation(
                path=self.rel_path,
                line=int(getattr(node, "lineno", 1)),
                column=int(getattr(node, "col_offset", 0)) + 1,
                qualname=scope.qualname,
                kind=kind,
                message="branch construct outside decision protocol",
            )
        )


def _module_is_decision_protocol(source_lines: list[str]) -> bool:
    for raw_line in source_lines[:80]:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") and MODULE_MARKER in stripped:
            return True
        if stripped.startswith("\"\"\"") or stripped.startswith("'''"):
            continue
    return False


def _has_preceding_marker(*, source_lines: list[str], line: int, marker: str) -> bool:
    idx = max(0, line - 2)
    while idx >= 0:
        stripped = source_lines[idx].strip()
        if not stripped:
            idx -= 1
            continue
        return stripped.startswith("#") and marker in stripped
    return False


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        if parent is None:
            return None
        return f"{parent}.{node.attr}"
    return None


def collect_violations(*, root: Path) -> list[Violation]:
    violations: list[Violation] = []
    for path in sorted(root.glob(TARGET_GLOB)):
        if not path.is_file() or any(part == "__pycache__" for part in path.parts):
            continue
        rel_path = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            violations.append(
                Violation(
                    path=rel_path,
                    line=1,
                    column=1,
                    qualname="<module>",
                    kind="read_error",
                    message="unable to read file while checking branchless policy",
                )
            )
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            violations.append(
                Violation(
                    path=rel_path,
                    line=int(exc.lineno or 1),
                    column=int(exc.offset or 1),
                    qualname="<module>",
                    kind="syntax_error",
                    message="syntax error while checking branchless policy",
                )
            )
            continue
        visitor = _BranchlessVisitor(rel_path=rel_path, source_lines=source.splitlines())
        visitor.visit(tree)
        violations.extend(visitor.violations)
    return violations


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
        line = item.get("line")
        if not path_value or not qualname or not kind or not isinstance(line, int):
            continue
        keys.add(f"{path_value}:{qualname}:{line}:{kind}")
    return keys


def _write_baseline(*, path: Path, violations: list[Violation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": BASELINE_VERSION,
        "violations": [
            asdict(violation)
            for violation in sorted(
                violations,
                key=lambda item: (item.path, item.qualname, item.line, item.kind),
            )
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def run(*, root: Path, baseline: Path | None = None, baseline_write: bool = False) -> int:
    violations = collect_violations(root=root)
    if baseline_write:
        if baseline is None:
            raise SystemExit("--baseline is required with --baseline-write")
        _write_baseline(path=baseline, violations=violations)
        print(f"wrote branchless policy baseline to {baseline}")
        return 0

    if baseline is not None:
        allowed = _load_baseline(baseline)
        violations = [violation for violation in violations if violation.key not in allowed]

    if not violations:
        print("branchless policy check passed")
        return 0

    print("branchless policy violations:")
    for violation in violations:
        print(f"  - {violation.render()}")
    return 1


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--baseline-write", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    baseline = Path(args.baseline).resolve() if args.baseline else None
    return run(
        root=Path(args.root).resolve(),
        baseline=baseline,
        baseline_write=bool(args.baseline_write),
    )


if __name__ == "__main__":
    raise SystemExit(main())
