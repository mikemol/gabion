from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import NamedTuple, TypeGuard


DEFAULT_ARTIFACT_PATH = Path("artifacts/out/structural_anti_pattern_contract.json")
_SOURCE_ROOTS = ("src", "scripts", "tests")


@dataclass(frozen=True)
class StructuralAntiPatternFinding:
    code: str
    rel_path: str
    line: int
    column: int
    symbol: str
    message: str


class _ListAccumulatorState(NamedTuple):
    list_names: frozenset[str]
    has_prefiltered_append: bool
    has_tuple_return: bool
    has_iteration_only_return: bool


def _existing_source_roots(root: Path) -> Iterator[Path]:
    return filter(Path.exists, (root / directory for directory in _SOURCE_ROOTS))


def _python_paths(root: Path) -> Iterator[Path]:
    return chain.from_iterable(
        sorted(base.rglob("*.py")) for base in _existing_source_roots(root)
    )


def _rel_path(path: Path, *, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _parse_module(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _qualname_for_node(module: ast.AST, target: ast.AST) -> str:
    stack: list[str] = []
    result: str | None = None

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            nonlocal result
            stack.append(node.name)
            if node is target:
                result = ".".join(stack)
            else:
                self.generic_visit(node)
            stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            nonlocal result
            stack.append(node.name)
            if node is target:
                result = ".".join(stack)
            else:
                self.generic_visit(node)
            stack.pop()

    Visitor().visit(module)
    if result is not None:
        return result
    if isinstance(target, ast.For):
        return "<for-loop>"
    if isinstance(target, ast.match_case):
        return "<match-case>"
    return "<module>"


def _loop_target_name(target: ast.expr) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    return None


def _is_ast_walk_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Attribute):
        return False
    return isinstance(node.func.value, ast.Name) and node.func.value.id == "ast" and node.func.attr == "walk"


def _is_single_continue_if(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.If)
        and not stmt.orelse
        and len(stmt.body) == 1
        and isinstance(stmt.body[0], ast.Continue)
    )


def _is_setup_assignment(stmt: ast.stmt) -> bool:
    return isinstance(stmt, (ast.Assign, ast.AnnAssign))


def _is_wildcard_case(pattern: ast.pattern) -> bool:
    return isinstance(pattern, ast.MatchAs) and pattern.name is None


def _is_len_guard(test: ast.expr) -> bool:
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    left = test.left
    right = test.comparators[0]
    if not (
        isinstance(left, ast.Call)
        and isinstance(left.func, ast.Name)
        and left.func.id == "len"
        and len(left.args) == 1
        and isinstance(right, ast.Constant)
        and isinstance(right.value, int)
    ):
        return False
    threshold = int(right.value)
    return (
        isinstance(test.ops[0], ast.Lt)
        and threshold == 2
    ) or (
        isinstance(test.ops[0], ast.LtE)
        and threshold == 1
    )


def _references_loop_name(test: ast.expr, *, loop_name: str | None) -> bool:
    if not loop_name:
        return False
    return any(isinstance(node, ast.Name) and node.id == loop_name for node in ast.walk(test))


def _is_for_node(node: ast.AST) -> TypeGuard[ast.For]:
    return isinstance(node, ast.For)


def _is_ast_walk_for_loop(node: ast.AST) -> TypeGuard[ast.For]:
    return isinstance(node, ast.For) and _is_ast_walk_call(node.iter)


def _is_match_node(node: ast.AST) -> TypeGuard[ast.Match]:
    return isinstance(node, ast.Match)


def _is_function_node(
    node: ast.AST,
) -> TypeGuard[ast.FunctionDef | ast.AsyncFunctionDef]:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))


def _iter_prefilter_continue_ifs(body: list[ast.stmt]) -> Iterator[ast.If]:
    for stmt in body:
        if _is_setup_assignment(stmt):
            continue
        if not _is_single_continue_if(stmt):
            break
        yield stmt


def _iter_ast_walk_prefilter_findings(
    module: ast.AST,
    *,
    rel_path: str,
) -> Iterator[StructuralAntiPatternFinding]:
    for node in filter(_is_ast_walk_for_loop, ast.walk(module)):
        loop_name = _loop_target_name(node.target)
        for stmt in _iter_prefilter_continue_ifs(node.body):
            if not _references_loop_name(stmt.test, loop_name=loop_name):
                break
            yield StructuralAntiPatternFinding(
                code="ast_walk_prefilter_in_loop",
                rel_path=rel_path,
                line=int(getattr(stmt, "lineno", 0) or 0),
                column=int(getattr(stmt, "col_offset", 0) or 0) + 1,
                symbol=_qualname_for_node(module, node),
                message="ast.walk loop carries a leading continue-filter that should move into filter(...) or a typed iterator",
            )
            if _is_len_guard(stmt.test):
                yield _len_guard_finding(module, rel_path=rel_path, node=node, stmt=stmt)


def _len_guard_finding(
    module: ast.AST,
    *,
    rel_path: str,
    node: ast.For,
    stmt: ast.If,
) -> StructuralAntiPatternFinding:
    return StructuralAntiPatternFinding(
        code="len_guard_continue",
        rel_path=rel_path,
        line=int(getattr(stmt, "lineno", 0) or 0),
        column=int(getattr(stmt, "col_offset", 0) or 0) + 1,
        symbol=_qualname_for_node(module, node),
        message="len-based continue filter should wrap the iterable before the loop body sees it",
    )


def _iter_len_guard_continue_findings(
    module: ast.AST,
    *,
    rel_path: str,
) -> Iterator[StructuralAntiPatternFinding]:
    for node in filter(_is_for_node, ast.walk(module)):
        for stmt in _iter_prefilter_continue_ifs(node.body):
            if _is_len_guard(stmt.test):
                yield _len_guard_finding(module, rel_path=rel_path, node=node, stmt=stmt)


def _iter_wildcard_soft_fallthrough_findings(
    module: ast.AST,
    *,
    rel_path: str,
) -> Iterator[StructuralAntiPatternFinding]:
    for node in filter(_is_match_node, ast.walk(module)):
        for case in node.cases:
            if not _is_wildcard_case(case.pattern) or not case.body:
                continue
            first_stmt = case.body[0]
            if not isinstance(first_stmt, (ast.Continue, ast.Pass)):
                continue
            yield StructuralAntiPatternFinding(
                code="wildcard_soft_fallthrough",
                rel_path=rel_path,
                line=int(getattr(case.pattern, "lineno", 0) or 0),
                column=int(getattr(case.pattern, "col_offset", 0) or 0) + 1,
                symbol=_qualname_for_node(module, case),
                message="wildcard match branch soft-falls through instead of remaining an unreachable invariant sink",
            )


def _list_accumulator_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    names: set[str] = set()
    for stmt in function.body:
        match stmt:
            case ast.Assign(targets=[ast.Name(id=name)], value=ast.List(elts=[])):
                names.add(name)
            case ast.AnnAssign(target=ast.Name(id=name), value=ast.List(elts=[])):
                names.add(name)
    return frozenset(names)


def _is_append_or_extend_to_names(stmt: ast.stmt, *, names: frozenset[str]) -> bool:
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return False
    func = stmt.value.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id in names
        and func.attr in {"append", "extend"}
    )


def _has_prefiltered_append(
    node: ast.AST,
    *,
    names: frozenset[str],
) -> bool:
    if not isinstance(node, ast.For):
        return False
    saw_prefilter = False
    for stmt in node.body:
        if _is_setup_assignment(stmt):
            continue
        if _is_single_continue_if(stmt):
            saw_prefilter = True
            continue
        if saw_prefilter and _is_append_or_extend_to_names(stmt, names=names):
            return True
    return False


def _returns_tuple_of_names(
    stmt: ast.stmt,
    *,
    names: frozenset[str],
) -> bool:
    if not isinstance(stmt, ast.Return):
        return False
    value = stmt.value
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "tuple"
        and len(value.args) == 1
        and isinstance(value.args[0], ast.Name)
        and value.args[0].id in names
    )


def _returns_iteration_only_name(
    stmt: ast.stmt,
    *,
    names: frozenset[str],
) -> bool:
    return isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name) and stmt.value.id in names


def _list_accumulator_state(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> _ListAccumulatorState:
    names = _list_accumulator_names(function)
    return _ListAccumulatorState(
        list_names=names,
        has_prefiltered_append=any(
            _has_prefiltered_append(node, names=names) for node in ast.walk(function)
        ),
        has_tuple_return=any(
            _returns_tuple_of_names(stmt, names=names) for stmt in function.body
        ),
        has_iteration_only_return=any(
            _returns_iteration_only_name(stmt, names=names) for stmt in function.body
        ),
    )


def _iter_materialization_findings(
    module: ast.AST,
    *,
    rel_path: str,
) -> Iterator[StructuralAntiPatternFinding]:
    for node in filter(_is_function_node, ast.walk(module)):
        state = _list_accumulator_state(node)
        if not state.list_names or not state.has_prefiltered_append:
            continue
        if state.has_tuple_return:
            yield StructuralAntiPatternFinding(
                code="eager_tuple_materialization",
                rel_path=rel_path,
                line=int(getattr(node, "lineno", 0) or 0),
                column=int(getattr(node, "col_offset", 0) or 0) + 1,
                symbol=_qualname_for_node(module, node),
                message="helper materializes a tuple from an append-built list instead of yielding or streaming",
            )
        if state.has_iteration_only_return:
            yield StructuralAntiPatternFinding(
                code="append_after_prefilter",
                rel_path=rel_path,
                line=int(getattr(node, "lineno", 0) or 0),
                column=int(getattr(node, "col_offset", 0) or 0) + 1,
                symbol=_qualname_for_node(module, node),
                message="append-based collector uses leading continue filters and should be a streamed projection",
            )


def collect_findings_for_path(path: Path, *, root: Path) -> list[StructuralAntiPatternFinding]:
    module = _parse_module(path)
    rel_path = _rel_path(path, root=root)
    return list(
        chain(
            _iter_wildcard_soft_fallthrough_findings(module, rel_path=rel_path),
            _iter_ast_walk_prefilter_findings(module, rel_path=rel_path),
            _iter_len_guard_continue_findings(module, rel_path=rel_path),
            _iter_materialization_findings(module, rel_path=rel_path),
        )
    )


def collect_findings(root: Path) -> list[StructuralAntiPatternFinding]:
    findings: list[StructuralAntiPatternFinding] = []
    for path in _python_paths(root):
        findings.extend(collect_findings_for_path(path, root=root))
    return sorted(
        findings,
        key=lambda finding: (
            finding.rel_path,
            finding.line,
            finding.column,
            finding.code,
        ),
    )


def build_payload(root: Path) -> dict[str, object]:
    findings = collect_findings(root)
    by_code: dict[str, int] = {}
    for finding in findings:
        by_code[finding.code] = by_code.get(finding.code, 0) + 1
    return {
        "schema_version": 1,
        "artifact_kind": "structural_anti_pattern_contract",
        "root": root.resolve().as_posix(),
        "counts": {
            "total": len(findings),
            "by_code": dict(sorted(by_code.items())),
        },
        "findings": [asdict(finding) for finding in findings],
    }


def run(
    *,
    root: Path,
    out_path: Path = DEFAULT_ARTIFACT_PATH,
    check: bool = False,
) -> int:
    payload = build_payload(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 1 if check and int(((payload.get("counts") or {}).get("total") or 0)) > 0 else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory structural anti-pattern candidates")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    return run(root=args.root.resolve(), out_path=args.out, check=args.check)


__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "StructuralAntiPatternFinding",
    "build_payload",
    "collect_findings",
    "collect_findings_for_path",
    "main",
    "run",
]
