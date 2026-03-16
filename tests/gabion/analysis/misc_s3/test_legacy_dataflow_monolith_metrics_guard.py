from __future__ import annotations

import ast
import importlib
from pathlib import Path


_MONOLITH_RELATIVE_PATH = (
    "src/gabion/analysis/dataflow/engine/dataflow_indexed_file_scan.py"
)
_MAX_MONOLITH_LOC = 40
_MAX_MONOLITH_TOP_LEVEL_IMPORTS = 4
_MAX_MONOLITH_IMPORTED_SYMBOLS = 4


# gabion:behavior primary=allowed_unwanted facets=legacy
def test_legacy_monolith_metrics_stay_within_budget() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    monolith_path = repo_root / _MONOLITH_RELATIVE_PATH
    source = monolith_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    loc = len(source.splitlines())
    top_level_imports = sum(
        1 for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    imported_symbols = 0
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
            canonical = importlib.import_module(module_path)
            imported_symbols += len(getattr(canonical, "__all__", ()))

    assert loc <= _MAX_MONOLITH_LOC, (
        "legacy monolith LOC budget exceeded; "
        f"loc={loc} max={_MAX_MONOLITH_LOC}"
    )
    assert top_level_imports <= _MAX_MONOLITH_TOP_LEVEL_IMPORTS, (
        "legacy monolith import budget exceeded; "
        f"imports={top_level_imports} max={_MAX_MONOLITH_TOP_LEVEL_IMPORTS}"
    )
    assert imported_symbols <= _MAX_MONOLITH_IMPORTED_SYMBOLS, (
        "legacy monolith imported-symbol budget exceeded; "
        f"symbols={imported_symbols} max={_MAX_MONOLITH_IMPORTED_SYMBOLS}"
    )
