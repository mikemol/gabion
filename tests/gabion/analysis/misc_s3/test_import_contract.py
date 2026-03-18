from __future__ import annotations

import ast
from pathlib import Path
from typing import TypeGuard


def _is_call_node(node: ast.AST) -> TypeGuard[ast.Call]:
    return isinstance(node, ast.Call)


def _sys_path_insert_lines(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines: list[int] = []
    for node in filter(_is_call_node, ast.walk(tree)):
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "insert":
            continue
        target = func.value
        if not isinstance(target, ast.Attribute) or target.attr != "path":
            continue
        if isinstance(target.value, ast.Name) and target.value.id == "sys":
            lines.append(int(node.lineno))
    return lines


# gabion:evidence E:call_footprint::tests/test_import_contract.py::test_only_conftest_mutates_sys_path::test_import_contract.py::tests.test_import_contract._sys_path_insert_lines
# gabion:behavior primary=desired
def test_only_conftest_mutates_sys_path() -> None:
    tests_dir = Path(__file__).resolve().parent
    allowed = tests_dir / "conftest.py"
    offenders: dict[str, list[int]] = {}
    for path in tests_dir.rglob("*.py"):
        rel = path.relative_to(tests_dir.parent)
        hits = _sys_path_insert_lines(path)
        if not hits:
            continue
        if path != allowed:
            offenders[str(rel)] = hits
    assert offenders == {}, f"Unexpected sys.path insert sites: {offenders}"
