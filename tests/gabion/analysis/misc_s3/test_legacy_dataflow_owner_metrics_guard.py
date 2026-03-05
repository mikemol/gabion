from __future__ import annotations

import ast
from pathlib import Path


_OWNER_RELATIVE_PATHS = (
    "src/gabion/analysis/dataflow/engine/dataflow_analysis_index_owner.py",
    "src/gabion/analysis/dataflow/engine/dataflow_deadline_runtime_owner.py",
    "src/gabion/analysis/dataflow/engine/dataflow_runtime_reporting_owner.py",
    "src/gabion/analysis/dataflow/engine/dataflow_deadline_summary_owner.py",
)
_MAX_OWNER_LOC = 40
_MAX_OWNER_TOP_LEVEL_IMPORTS = 10


def _file_metrics(path: Path) -> tuple[int, int]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    loc = len(source.splitlines())
    imports = sum(
        1 for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    return loc, imports


def test_legacy_owner_metrics_stay_within_budget() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    violations: list[str] = []

    for relative_path in _OWNER_RELATIVE_PATHS:
        owner_path = repo_root / relative_path
        loc, imports = _file_metrics(owner_path)
        if loc > _MAX_OWNER_LOC:
            violations.append(
                f"{relative_path}:loc={loc}>max={_MAX_OWNER_LOC}"
            )
        if imports > _MAX_OWNER_TOP_LEVEL_IMPORTS:
            violations.append(
                f"{relative_path}:imports={imports}>max={_MAX_OWNER_TOP_LEVEL_IMPORTS}"
            )

    assert not violations, (
        "legacy owner module metrics budget exceeded; "
        f"violations={violations}"
    )
