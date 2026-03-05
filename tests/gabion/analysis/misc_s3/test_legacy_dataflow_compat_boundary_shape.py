from __future__ import annotations

import ast
from pathlib import Path


_COMPAT_MODULES = (
    "src/gabion/analysis/dataflow/engine/dataflow_analysis_index_owner.py",
    "src/gabion/analysis/dataflow/engine/dataflow_deadline_runtime_owner.py",
    "src/gabion/analysis/dataflow/engine/dataflow_runtime_reporting_owner.py",
    "src/gabion/analysis/dataflow/engine/dataflow_deadline_summary_owner.py",
    "src/gabion/analysis/dataflow/engine/dataflow_facade.py",
)


def _runtime_logic_offenders(module_path: Path) -> list[str]:
    tree = ast.parse(module_path.read_text())
    offenders: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            offenders.append(
                f"{type(node).__name__}:{getattr(node, 'name', '<anon>')}@{node.lineno}"
            )
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            continue
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        offenders.append(f"{type(node).__name__}:<stmt>@{node.lineno}")
    return offenders


def test_legacy_dataflow_compat_modules_have_no_runtime_logic_defs() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    offenders_by_module: dict[str, list[str]] = {}

    for relative_path in _COMPAT_MODULES:
        module_path = repo_root / relative_path
        offenders = _runtime_logic_offenders(module_path)
        if offenders:
            offenders_by_module[relative_path] = offenders

    assert not offenders_by_module, (
        "legacy compatibility modules must remain boundary-only "
        "(imports/metadata/docstring only); "
        f"offenders={offenders_by_module}"
    )
