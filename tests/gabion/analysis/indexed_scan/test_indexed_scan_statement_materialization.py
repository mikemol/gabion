from __future__ import annotations

import ast
from dataclasses import dataclass

from gabion.analysis.indexed_scan.scanners.materialization.statement_materialization import (
    materialize_statement_suite_contains)


@dataclass
class _FakeForest:
    suites: list[tuple[str, str, str, tuple[int, int, int, int], object]]

    def add_suite_site(
        self,
        path_name: str,
        qual: str,
        suite_kind: str,
        *,
        span: tuple[int, int, int, int],
        parent: object,
    ) -> object:
        node_id = (path_name, qual, suite_kind, span, parent)
        self.suites.append(node_id)
        return node_id


def _node_span(node: ast.AST) -> tuple[int, int, int, int] | None:
    line = getattr(node, "lineno", None)
    col = getattr(node, "col_offset", None)
    end_line = getattr(node, "end_lineno", None)
    end_col = getattr(node, "end_col_offset", None)
    if None in {line, col, end_line, end_col}:
        return None
    return (int(line) - 1, int(col), int(end_line) - 1, int(end_col))


# gabion:evidence E:function_site::indexed_scan/statement_materialization.py::gabion.analysis.indexed_scan.statement_materialization.materialize_statement_suite_contains
def test_materialize_statement_suite_contains_emits_nested_suites() -> None:
    module = ast.parse(
        "if cond:\n"
        "    while ok:\n"
        "        pass\n"
        "else:\n"
        "    pass\n"
    )
    forest = _FakeForest(suites=[])
    materialize_statement_suite_contains(
        forest=forest,  # type: ignore[arg-type]
        path_name="mod.py",
        qual="pkg.fn",
        statements=module.body,
        parent_suite=("root",),
        node_span_fn=_node_span,
    )
    suite_kinds = [row[2] for row in forest.suites]
    assert "if_body" in suite_kinds
    assert "if_else" in suite_kinds
    assert "while_body" in suite_kinds


# gabion:evidence E:function_site::indexed_scan/statement_materialization.py::gabion.analysis.indexed_scan.statement_materialization.materialize_statement_suite_contains::missing
def test_materialize_statement_suite_contains_skips_when_span_missing() -> None:
    module = ast.parse("if cond:\n    pass\n")
    forest = _FakeForest(suites=[])
    materialize_statement_suite_contains(
        forest=forest,  # type: ignore[arg-type]
        path_name="mod.py",
        qual="pkg.fn",
        statements=module.body,
        parent_suite=("root",),
        node_span_fn=lambda _node: None,
    )
    assert forest.suites == []


# gabion:evidence E:function_site::indexed_scan/statement_materialization.py::gabion.analysis.indexed_scan.statement_materialization.materialize_statement_suite_contains::for_else
def test_materialize_statement_suite_contains_covers_loop_with_try_branches() -> None:
    module = ast.parse(
        "for item in seq:\n"
        "    pass\n"
        "else:\n"
        "    pass\n"
        "try:\n"
        "    pass\n"
        "except Exception:\n"
        "    pass\n"
        "else:\n"
        "    pass\n"
        "finally:\n"
        "    pass\n"
    )
    forest = _FakeForest(suites=[])
    materialize_statement_suite_contains(
        forest=forest,  # type: ignore[arg-type]
        path_name="mod.py",
        qual="pkg.fn",
        statements=module.body,
        parent_suite=("root",),
        node_span_fn=_node_span,
    )
    suite_kinds = [row[2] for row in forest.suites]
    assert "for_body" in suite_kinds
    assert "for_else" in suite_kinds
    assert "try_body" in suite_kinds
    assert "except_body" in suite_kinds
    assert "try_else" in suite_kinds
    assert "try_finally" in suite_kinds


# gabion:evidence E:function_site::indexed_scan/statement_materialization.py::gabion.analysis.indexed_scan.statement_materialization.materialize_statement_suite_contains::async_for
def test_materialize_statement_suite_contains_covers_async_branches() -> None:
    module = ast.parse(
        "async def _f(xs):\n"
        "    async for item in xs:\n"
        "        pass\n"
    )
    async_fn = module.body[0]
    assert isinstance(async_fn, ast.AsyncFunctionDef)
    forest = _FakeForest(suites=[])
    materialize_statement_suite_contains(
        forest=forest,  # type: ignore[arg-type]
        path_name="mod.py",
        qual="pkg._f",
        statements=async_fn.body,
        parent_suite=("root",),
        node_span_fn=_node_span,
    )
    suite_kinds = [row[2] for row in forest.suites]
    assert "async_for_body" in suite_kinds
