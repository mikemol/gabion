#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from gabion.tooling.runtime.policy_result_schema import make_policy_result, write_policy_result
from gabion.tooling.runtime.policy_scan_batch import (
    PolicyScanBatch,
    build_policy_scan_batch,
    iter_failure_seeds,
)


_DERIVATION_IDENTITY_FILES = (
    Path("src/gabion/analysis/derivation_contract.py"),
    Path("src/gabion/analysis/derivation_graph.py"),
    Path("src/gabion/analysis/derivation_cache.py"),
)
_ASPF_PATH = Path("src/gabion/analysis/aspf.py")
_ASPF_STRUCTURAL_FUNCTIONS = {
    "_float_structural_atom",
    "structural_key_atom",
    "structural_key_json",
}


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    call: str
    reason: str


class _PolicyVisitor(ast.NodeVisitor):
    def __init__(self, *, path: Path, check_all_functions: bool) -> None:
        self.path = path
        self.check_all_functions = check_all_functions
        self.violations: list[Violation] = []
        self._function_stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _dotted_name(node.func)
        if call_name is None:
            self.generic_visit(node)
            return
        if not self._callsite_in_scope():
            self.generic_visit(node)
            return
        if call_name.startswith("hashlib."):
            self._record(node=node, call=call_name, reason="digest/hashlib not allowed")
        if call_name in {
            "stable_encode.stable_compact_text",
            "stable_encode.stable_compact_bytes",
            "stable_compact_text",
            "stable_compact_bytes",
        }:
            self._record(
                node=node,
                call=call_name,
                reason="text canonicalization not allowed in structural key identity paths",
            )
        if call_name in {"json.dumps", "dumps"}:
            self._record(
                node=node,
                call=call_name,
                reason="json text encoding not allowed in structural key identity paths",
            )
        self.generic_visit(node)

    def _callsite_in_scope(self) -> bool:
        if self.check_all_functions:
            return True
        if not self._function_stack:
            return False
        return self._function_stack[-1] in _ASPF_STRUCTURAL_FUNCTIONS

    def _record(self, *, node: ast.Call, call: str, reason: str) -> None:
        self.violations.append(
            Violation(
                path=str(self.path),
                line=int(node.lineno),
                call=call,
                reason=reason,
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


def _target_files(root: Path) -> list[Path]:
    files = [root / rel_path for rel_path in _DERIVATION_IDENTITY_FILES]
    aspf = root / _ASPF_PATH
    if aspf.exists():
        files.append(aspf)
    return [path for path in files if path.exists()]


def _scan_module(*, module_rel_path: str, tree: ast.AST) -> list[Violation]:
    rel_path = Path(module_rel_path)
    visitor = _PolicyVisitor(
        path=rel_path,
        check_all_functions=rel_path != _ASPF_PATH,
    )
    visitor.visit(tree)
    return visitor.violations


def collect_violations(*, batch: PolicyScanBatch) -> list[Violation]:
    violations: list[Violation] = []
    for seed in iter_failure_seeds(batch=batch):
        violations.append(
            Violation(
                path=seed.path,
                line=seed.line,
                call="<read>" if seed.kind == "read_error" else "<parse>",
                reason="unable to read file" if seed.kind == "read_error" else "syntax error",
            )
        )
    for module in batch.modules:
        violations.extend(
            _scan_module(
                module_rel_path=module.rel_path,
                tree=module.tree,
            )
        )
    return violations


def _serialize_violation(violation: Violation) -> dict[str, object]:
    return {
        "path": violation.path,
        "line": violation.line,
        "call": violation.call,
        "reason": violation.reason,
        "render": f"{violation.path}:{violation.line}: {violation.call}: {violation.reason}",
    }


def run(*, root: Path, output: Path | None = None) -> int:
    batch = build_policy_scan_batch(
        root=root,
        target_globs=(),
        files=_target_files(root),
    )
    violations = collect_violations(batch=batch)
    status = "pass" if not violations else "fail"
    if output is not None:
        write_policy_result(
            path=output,
            result=make_policy_result(
                rule_id="structural_hash",
                status=status,
                violations=[_serialize_violation(item) for item in violations],
                baseline_mode="none",
                source_tool="scripts/policy/structural_hash_policy_check.py",
                input_scope={"root": str(root)},
            ),
        )
    if not violations:
        print("structural-hash policy check passed")
        return 0
    print("structural-hash policy violations:")
    for violation in violations:
        print(
            f"  - {violation.path}:{violation.line}: {violation.call}: {violation.reason}"
        )
    return 1


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run(root=Path(args.root).resolve(), output=args.output.resolve() if args.output else None)


if __name__ == "__main__":
    raise SystemExit(main())
