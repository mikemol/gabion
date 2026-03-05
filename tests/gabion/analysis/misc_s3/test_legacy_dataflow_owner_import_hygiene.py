from __future__ import annotations

import ast
from pathlib import Path


_OWNER_MODULES = (
    "dataflow_analysis_index_owner",
    "dataflow_deadline_runtime_owner",
    "dataflow_deadline_summary_owner",
    "dataflow_runtime_reporting_owner",
)


def _repo_python_files(root: Path):
    for base in (root / "src", root / "tests"):
        for path in base.rglob("*.py"):
            yield path


def test_no_internal_owner_module_imports() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    violations: list[str] = []

    for path in _repo_python_files(repo_root):
        rel = path.relative_to(repo_root)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "gabion.analysis.dataflow.engine":
                    for alias in node.names:
                        if alias.name in _OWNER_MODULES:
                            violations.append(f"{rel}:{node.lineno}")
                for owner_module in _OWNER_MODULES:
                    target = f"gabion.analysis.dataflow.engine.{owner_module}"
                    if module == target:
                        violations.append(f"{rel}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    for owner_module in _OWNER_MODULES:
                        target = f"gabion.analysis.dataflow.engine.{owner_module}"
                        if alias.name == target:
                            violations.append(f"{rel}:{node.lineno}")

    assert not violations, (
        "legacy owner-module imports are prohibited in src/tests; "
        f"violations={sorted(violations)}"
    )
