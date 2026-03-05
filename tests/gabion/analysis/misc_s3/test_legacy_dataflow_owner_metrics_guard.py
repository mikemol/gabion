from __future__ import annotations

import ast
import importlib
from pathlib import Path


_OWNER_RELATIVE_PATHS = (
    "src/gabion/analysis/dataflow/engine/dataflow_analysis_index_owner.py",
    "src/gabion/analysis/dataflow/engine/dataflow_deadline_runtime_owner.py",
    "src/gabion/analysis/dataflow/engine/dataflow_runtime_reporting_owner.py",
    "src/gabion/analysis/dataflow/engine/dataflow_deadline_summary_owner.py",
)
_MAX_OWNER_LOC = 60
_MAX_OWNER_TOP_LEVEL_IMPORTS = 5
_MAX_OWNER_IMPORTED_SYMBOLS = 40


def _file_metrics(path: Path) -> tuple[int, int, int]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    loc = len(source.splitlines())
    imports = sum(
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
    return loc, imports, imported_symbols


def test_legacy_owner_metrics_stay_within_budget() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    violations: list[str] = []

    for relative_path in _OWNER_RELATIVE_PATHS:
        owner_path = repo_root / relative_path
        loc, imports, imported_symbols = _file_metrics(owner_path)
        if loc > _MAX_OWNER_LOC:
            violations.append(
                f"{relative_path}:loc={loc}>max={_MAX_OWNER_LOC}"
            )
        if imports > _MAX_OWNER_TOP_LEVEL_IMPORTS:
            violations.append(
                f"{relative_path}:imports={imports}>max={_MAX_OWNER_TOP_LEVEL_IMPORTS}"
            )
        if imported_symbols > _MAX_OWNER_IMPORTED_SYMBOLS:
            violations.append(
                f"{relative_path}:symbols={imported_symbols}>max={_MAX_OWNER_IMPORTED_SYMBOLS}"
            )

    assert not violations, (
        "legacy owner module metrics budget exceeded; "
        f"violations={violations}"
    )
