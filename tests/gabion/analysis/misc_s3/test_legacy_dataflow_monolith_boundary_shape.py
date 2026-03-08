from __future__ import annotations

import ast
from pathlib import Path


# gabion:behavior primary=allowed_unwanted facets=legacy
def test_monolith_has_no_runtime_logic_defs() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    monolith_path = (
        repo_root
        / "src/gabion/analysis/dataflow/engine/dataflow_indexed_file_scan.py"
    )
    tree = ast.parse(monolith_path.read_text())

    forbidden = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    offenders = [
        f"{type(node).__name__}:{getattr(node, 'name', '<anon>')}@{node.lineno}"
        for node in tree.body
        if isinstance(node, forbidden)
    ]

    assert not offenders, (
        "legacy monolith must remain boundary-only (no runtime logic defs); "
        f"offenders={offenders}"
    )
