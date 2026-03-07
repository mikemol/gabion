#!/usr/bin/env python3
# gabion:boundary_normalization_module gabion:decision_protocol_module
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from gabion.tooling.runtime.policy_result_schema import make_policy_result, write_policy_result

TARGET_GLOBS = ("tests/**/*.py",)
_SPAWN_CALLS = frozenset({"run", "check_output", "Popen"})


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    column: int
    kind: str
    message: str
    call: str
    key: str

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: {self.kind}: {self.message}"


def _load_allowlist(path: Path) -> set[str]:
    if not path.exists():
        return set()
    allowed: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            allowed.add(line.replace("\\", "/"))
    return allowed


@dataclass
class _ImportScope:
    subprocess_names: set[str]
    subprocess_spawn_names: dict[str, str]


class _SubprocessSpawnVisitor(ast.NodeVisitor):
    def __init__(self, *, rel_path: str) -> None:
        self.rel_path = rel_path
        self.scope = _ImportScope(subprocess_names={"subprocess"}, subprocess_spawn_names={})
        self.violations: list[Violation] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            imported_name = str(alias.name or "")
            if imported_name == "subprocess":
                local_name = str(alias.asname or imported_name)
                self.scope.subprocess_names.add(local_name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if str(node.module or "") != "subprocess":
            self.generic_visit(node)
            return
        for alias in node.names:
            imported_name = str(alias.name or "")
            if imported_name in _SPAWN_CALLS:
                local_name = str(alias.asname or imported_name)
                self.scope.subprocess_spawn_names[local_name] = imported_name
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        dotted = _dotted_name(node.func)
        if dotted is None:
            self.generic_visit(node)
            return

        if "." in dotted:
            head, attr = dotted.rsplit(".", 1)
            if head in self.scope.subprocess_names and attr in _SPAWN_CALLS:
                self._report(
                    node=node,
                    call=dotted,
                    kind=f"subprocess_{attr}",
                    message=(
                        "direct process spawn in tests is disallowed; "
                        "prefer dependency-injected seams and keep a narrow allowlisted integration boundary"
                    ),
                )
                self.generic_visit(node)
                return
        spawn_name = self.scope.subprocess_spawn_names.get(dotted)
        if spawn_name is not None:
            self._report(
                node=node,
                call=dotted,
                kind=f"subprocess_{spawn_name}",
                message=(
                    "direct process spawn in tests is disallowed; "
                    "prefer dependency-injected seams and keep a narrow allowlisted integration boundary"
                ),
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._check_assignment_target(target=target, node=node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_assignment_target(target=node.target, node=node)
        self.generic_visit(node)

    def _check_assignment_target(self, *, target: ast.AST, node: ast.AST) -> None:
        dotted = _dotted_name(target)
        if dotted is None or "." not in dotted:
            return
        head, attr = dotted.rsplit(".", 1)
        if head not in self.scope.subprocess_names or attr not in _SPAWN_CALLS:
            return
        self._report(
            node=node,
            call=dotted,
            kind=f"subprocess_{attr}_reassignment",
            message=(
                "runtime reassignment of subprocess spawn surfaces in tests is disallowed; "
                "inject collaborators via explicit deps instead"
            ),
        )

    def _report(self, *, node: ast.AST, call: str, kind: str, message: str) -> None:
        line = int(getattr(node, "lineno", 1))
        column = int(getattr(node, "col_offset", 0)) + 1
        self.violations.append(
            Violation(
                path=self.rel_path,
                line=line,
                column=column,
                kind=kind,
                message=message,
                call=call,
                key=f"{self.rel_path}:{line}:{kind}",
            )
        )


def _dotted_name(node: ast.AST) -> str | None:
    match node:
        case ast.Name(id=identifier):
            return str(identifier)
        case ast.Attribute(value=value, attr=attr):
            parent = _dotted_name(value)
            if parent is None:
                return None
            return f"{parent}.{attr}"
        case _:
            return None


def collect_violations(
    *,
    root: Path,
    allowlist_path: Path,
) -> list[Violation]:
    allowlisted_paths = _load_allowlist(allowlist_path)
    violations: list[Violation] = []
    for pattern in TARGET_GLOBS:
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or any(part == "__pycache__" for part in path.parts):
                continue
            rel_path = path.relative_to(root).as_posix()
            if rel_path in allowlisted_paths:
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except OSError:
                violations.append(
                    Violation(
                        path=rel_path,
                        line=1,
                        column=1,
                        kind="read_error",
                        message="unable to read test file while checking subprocess hygiene",
                        call="<none>",
                        key=f"{rel_path}:1:read_error",
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
                        kind="syntax_error",
                        message="syntax error while checking subprocess hygiene",
                        call="<none>",
                        key=f"{rel_path}:{int(exc.lineno or 1)}:syntax_error",
                    )
                )
                continue
            visitor = _SubprocessSpawnVisitor(rel_path=rel_path)
            visitor.visit(tree)
            violations.extend(visitor.violations)
    return violations


def _serialize_violation(item: Violation) -> dict[str, object]:
    return {
        "path": item.path,
        "line": item.line,
        "column": item.column,
        "kind": item.kind,
        "call": item.call,
        "message": item.message,
        "key": item.key,
        "render": item.render(),
    }


def run(
    *,
    root: Path,
    allowlist_path: Path,
    output: Path | None = None,
) -> int:
    violations = collect_violations(root=root, allowlist_path=allowlist_path)
    status = "pass" if not violations else "fail"
    if output is not None:
        write_policy_result(
            path=output,
            result=make_policy_result(
                rule_id="test_subprocess_hygiene",
                status=status,
                violations=[_serialize_violation(item) for item in violations],
                baseline_mode="allowlist",
                source_tool="src/gabion/tooling/policy_rules/test_subprocess_hygiene_rule.py",
                input_scope={
                    "root": str(root),
                    "allowlist_path": str(allowlist_path),
                },
            ),
        )
    if not violations:
        print("test-subprocess hygiene policy check passed")
        return 0
    print("test-subprocess hygiene policy violations:")
    for violation in violations:
        print(f"  - {violation.render()}")
    return 1


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--allowlist",
        default="docs/policy/test_subprocess_hygiene_allowlist.txt",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    output = args.output.resolve() if args.output is not None else None
    return run(
        root=Path(args.root).resolve(),
        allowlist_path=Path(args.allowlist).resolve(),
        output=output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
