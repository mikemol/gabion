from __future__ import annotations

import ast
from pathlib import Path


_LEGACY_MODULES = {
    "gabion.analysis.dataflow.engine.dataflow_analysis_index_owner",
    "gabion.analysis.dataflow.engine.dataflow_deadline_runtime_owner",
    "gabion.analysis.dataflow.engine.dataflow_runtime_reporting_owner",
    "gabion.analysis.dataflow.engine.dataflow_deadline_summary_owner",
    "gabion.analysis.dataflow.engine.dataflow_facade",
}
_LEGACY_FILES = (
    "src/gabion/analysis/dataflow/engine/dataflow_analysis_index_owner.py",
    "src/gabion/analysis/dataflow/engine/dataflow_deadline_runtime_owner.py",
    "src/gabion/analysis/dataflow/engine/dataflow_runtime_reporting_owner.py",
    "src/gabion/analysis/dataflow/engine/dataflow_deadline_summary_owner.py",
    "src/gabion/analysis/dataflow/engine/dataflow_facade.py",
)


def _repo_python_files(root: Path):
    for base in (root / "src", root / "tests"):
        for path in base.rglob("*.py"):
            yield path


# gabion:behavior primary=allowed_unwanted facets=compat,legacy
def test_legacy_compat_modules_retired() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    existing = [rel for rel in _LEGACY_FILES if (repo_root / rel).exists()]
    assert not existing, f"legacy compatibility modules must be deleted; existing={existing}"


# gabion:behavior primary=allowed_unwanted facets=compat,legacy
def test_no_legacy_compat_imports_remain() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    violations: list[str] = []

    for path in _repo_python_files(repo_root):
        rel = path.relative_to(repo_root)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in _LEGACY_MODULES:
                    violations.append(f"{rel}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _LEGACY_MODULES:
                        violations.append(f"{rel}:{node.lineno}")

    assert not violations, (
        "legacy compatibility module imports are prohibited in src/tests; "
        f"violations={sorted(violations)}"
    )
