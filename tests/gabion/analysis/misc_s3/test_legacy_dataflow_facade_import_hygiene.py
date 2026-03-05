from __future__ import annotations

import ast
from pathlib import Path


def _repo_python_files(root: Path):
    for base in (root / "src", root / "tests"):
        for path in base.rglob("*.py"):
            yield path


def test_no_internal_dataflow_facade_imports() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    violations: list[str] = []
    target = "gabion.analysis.dataflow.engine.dataflow_facade"

    for path in _repo_python_files(repo_root):
        rel = path.relative_to(repo_root)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "gabion.analysis.dataflow.engine":
                    for alias in node.names:
                        if alias.name == "dataflow_facade":
                            violations.append(f"{rel}:{node.lineno}")
                if module == target:
                    violations.append(f"{rel}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == target:
                        violations.append(f"{rel}:{node.lineno}")

    assert not violations, (
        "legacy dataflow-facade imports are prohibited in src/tests; "
        f"violations={sorted(violations)}"
    )
