from __future__ import annotations

import ast
from pathlib import Path


_FACADE_RELATIVE_PATH = "src/gabion/analysis/dataflow/engine/dataflow_facade.py"
_MAX_FACADE_LOC = 300
_MAX_FACADE_TOP_LEVEL_IMPORTS = 45


def test_legacy_facade_metrics_stay_within_budget() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    facade_path = repo_root / _FACADE_RELATIVE_PATH
    source = facade_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    loc = len(source.splitlines())
    top_level_imports = sum(
        1 for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    )

    assert loc <= _MAX_FACADE_LOC, (
        "legacy facade LOC budget exceeded; "
        f"loc={loc} max={_MAX_FACADE_LOC}"
    )
    assert top_level_imports <= _MAX_FACADE_TOP_LEVEL_IMPORTS, (
        "legacy facade import budget exceeded; "
        f"imports={top_level_imports} max={_MAX_FACADE_TOP_LEVEL_IMPORTS}"
    )
