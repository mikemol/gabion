#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TARGET_GLOBS = (
    "tests/**/*.py",
    "src/**/*.py",
)
_PATCH_CALL_NAMES = {
    "patch",
    "patch.object",
    "patch.dict",
    "patch.multiple",
}
_PATCH_MODULES = {
    "mock",
    "unittest",
    "unittest.mock",
}


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    column: int
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: {self.message}"


@dataclass
class _Scope:
    patch_names: set[str]
    patch_modules: set[str]


class _NoMonkeypatchVisitor(ast.NodeVisitor):
    def __init__(self, *, rel_path: str) -> None:
        self.rel_path = rel_path
        self.violations: list[Violation] = []
        self.scope = _Scope(patch_names=set(), patch_modules=set(_PATCH_MODULES))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = str(node.module or "")
        if module in {"unittest.mock", "mock"}:
            for alias in node.names:
                imported_name = str(alias.name or "")
                local_name = str(alias.asname or imported_name)
                if imported_name.startswith("patch"):
                    self.scope.patch_names.add(local_name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            imported_name = str(alias.name or "")
            local_name = str(alias.asname or imported_name.split(".")[-1])
            if imported_name in _PATCH_MODULES:
                self.scope.patch_modules.add(local_name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_monkeypatch_fixture(node)
        self._check_patch_decorators(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_monkeypatch_fixture(node)
        self._check_patch_decorators(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        dotted = _dotted_name(node.func)
        if dotted is not None and self._is_patch_call_name(dotted):
            self._report(node, "patch-style runtime mutation is forbidden; use dependency injection")
        if isinstance(node.func, ast.Attribute):
            owner = _dotted_name(node.func.value)
            if owner == "monkeypatch":
                self._report(
                    node,
                    "pytest monkeypatch usage is forbidden; use dependency injection seams",
                )
        self.generic_visit(node)

    def _check_monkeypatch_fixture(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        args = [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]
        if node.args.vararg is not None:
            args.append(node.args.vararg)
        if node.args.kwarg is not None:
            args.append(node.args.kwarg)
        for arg in args:
            if arg.arg == "monkeypatch":
                self._report(
                    arg,
                    "monkeypatch fixture is forbidden; inject collaborators explicitly",
                )

    def _check_patch_decorators(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        for decorator in node.decorator_list:
            dotted = _dotted_name(decorator)
            if dotted is not None and self._is_patch_call_name(dotted):
                self._report(
                    decorator,
                    "patch decorator is forbidden; inject collaborators explicitly",
                )
                continue
            if isinstance(decorator, ast.Call):
                dotted_call = _dotted_name(decorator.func)
                if dotted_call is not None and self._is_patch_call_name(dotted_call):
                    self._report(
                        decorator,
                        "patch decorator call is forbidden; inject collaborators explicitly",
                    )

    def _is_patch_call_name(self, dotted: str) -> bool:
        if dotted in _PATCH_CALL_NAMES:
            return dotted in self.scope.patch_names or dotted.split(".", 1)[0] in self.scope.patch_names
        head = dotted.split(".", 1)[0]
        if dotted.endswith(".patch"):
            return head in self.scope.patch_modules
        if dotted.endswith(".patch.object") or dotted.endswith(".patch.dict") or dotted.endswith(
            ".patch.multiple"
        ):
            return head in self.scope.patch_modules
        return False

    def _report(self, node: ast.AST, message: str) -> None:
        self.violations.append(
            Violation(
                path=self.rel_path,
                line=int(getattr(node, "lineno", 1)),
                column=int(getattr(node, "col_offset", 0)) + 1,
                message=message,
            )
        )


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
    for pattern in TARGET_GLOBS:
        for path in sorted(root.glob(pattern)):
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
                        message="unable to read file while checking monkeypatch policy",
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
                        message="syntax error while checking monkeypatch policy",
                    )
                )
                continue
            visitor = _NoMonkeypatchVisitor(rel_path=rel_path)
            visitor.visit(tree)
            violations.extend(visitor.violations)
    return violations


def run(*, root: Path) -> int:
    violations = collect_violations(root=root)
    if not violations:
        print("no-monkeypatch policy check passed")
        return 0
    print("no-monkeypatch policy violations:")
    for violation in violations:
        print(f"  - {violation.render()}")
    return 1


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run(root=Path(args.root).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
