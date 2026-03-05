from __future__ import annotations

import ast
from pathlib import Path


_MONOLITH_RELATIVE_PATH = (
    "src/gabion/analysis/dataflow/engine/dataflow_indexed_file_scan.py"
)
_MAX_MONOLITH_LOC = 420
_MAX_MONOLITH_TOP_LEVEL_IMPORTS = 53


def test_legacy_monolith_metrics_stay_within_budget() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    monolith_path = repo_root / _MONOLITH_RELATIVE_PATH
    source = monolith_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    loc = len(source.splitlines())
    top_level_imports = sum(
        1 for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    )

    assert loc <= _MAX_MONOLITH_LOC, (
        "legacy monolith LOC budget exceeded; "
        f"loc={loc} max={_MAX_MONOLITH_LOC}"
    )
    assert top_level_imports <= _MAX_MONOLITH_TOP_LEVEL_IMPORTS, (
        "legacy monolith import budget exceeded; "
        f"imports={top_level_imports} max={_MAX_MONOLITH_TOP_LEVEL_IMPORTS}"
    )
