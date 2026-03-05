from __future__ import annotations

import ast
import importlib
from pathlib import Path


_FACADE_RELATIVE_PATH = "src/gabion/analysis/dataflow/engine/dataflow_facade.py"
_MAX_FACADE_LOC = 155
_MAX_FACADE_TOP_LEVEL_IMPORTS = 38
_MAX_FACADE_IMPORTED_SYMBOLS = 243
_MAX_FACADE_WILDCARD_IMPORTS = 0


def test_legacy_facade_metrics_stay_within_budget() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    facade_path = repo_root / _FACADE_RELATIVE_PATH
    source = facade_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    loc = len(source.splitlines())
    top_level_imports = sum(
        1 for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    imported_symbols = 0
    wildcard_imports = 0
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_symbols += len(node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        module_path = node.module or ""
        for alias in node.names:
            if alias.name != "*":
                imported_symbols += 1
                continue
            wildcard_imports += 1
            canonical = importlib.import_module(module_path)
            imported_symbols += len(getattr(canonical, "__all__", ()))

    assert loc <= _MAX_FACADE_LOC, (
        "legacy facade LOC budget exceeded; "
        f"loc={loc} max={_MAX_FACADE_LOC}"
    )
    assert top_level_imports <= _MAX_FACADE_TOP_LEVEL_IMPORTS, (
        "legacy facade import budget exceeded; "
        f"imports={top_level_imports} max={_MAX_FACADE_TOP_LEVEL_IMPORTS}"
    )
    assert imported_symbols <= _MAX_FACADE_IMPORTED_SYMBOLS, (
        "legacy facade imported-symbol budget exceeded; "
        f"symbols={imported_symbols} max={_MAX_FACADE_IMPORTED_SYMBOLS}"
    )
    assert wildcard_imports <= _MAX_FACADE_WILDCARD_IMPORTS, (
        "legacy facade wildcard-import budget exceeded; "
        f"wildcards={wildcard_imports} max={_MAX_FACADE_WILDCARD_IMPORTS}"
    )
