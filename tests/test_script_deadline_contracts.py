from __future__ import annotations

import ast
from pathlib import Path

from gabion.analysis.aspf import Forest
from gabion.analysis.dataflow_audit import AuditConfig, analyze_paths
from tests.order_helpers import contract_sorted

_DEADLINE_SYMBOLS = {"check_deadline", "deadline_scope", "deadline_loop_iter"}


def _uses_deadline_symbol(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in _DEADLINE_SYMBOLS:
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr in _DEADLINE_SYMBOLS:
            return True
    return False


def _script_deadline_paths(repo_root: Path) -> list[Path]:
    scripts_dir = repo_root / "scripts"
    paths = contract_sorted(scripts_dir.glob("*.py"), key=lambda path: str(path))
    return [path for path in paths if _uses_deadline_symbol(path)]


def _deadline_roots(script_paths: list[Path], repo_root: Path) -> set[str]:
    roots: set[str] = set()
    for path in script_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module = ".".join(path.relative_to(repo_root).with_suffix("").parts)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                roots.add(f"{module}.{node.name}")
    return roots


def test_scripts_using_deadlines_require_clock_and_forest_scope() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_paths = _script_deadline_paths(repo_root)
    deadline_roots = _deadline_roots(script_paths, repo_root)

    analysis = analyze_paths(
        script_paths,
        forest=Forest(),
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_deadline_obligations=True,
        config=AuditConfig(
            project_root=repo_root,
            exclude_dirs=set(),
            ignore_params=set(),
            external_filter=True,
            strictness="high",
            deadline_roots=deadline_roots,
        ),
    )
    offenders: list[str] = []
    for entry in analysis.deadline_obligations:
        kind = entry.get("kind")
        if kind not in {"missing_carrier", "unchecked_deadline"}:
            continue
        site = entry.get("site")
        if not isinstance(site, dict):
            continue
        site_path = site.get("path")
        site_fn = site.get("function")
        if isinstance(site_path, str) and site_path.startswith("scripts/"):
            offenders.append(f"{site_path}:{site_fn} ({kind})")
    assert not offenders, (
        "Scripts with deadline semantics must propagate/check deadline carriers; "
        f"offenders: {contract_sorted(offenders)}"
    )


def test_deadline_runtime_provides_scope_guards() -> None:
    path = Path(__file__).resolve().parents[1] / "scripts" / "deadline_runtime.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    fn_nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    ticks_fn = fn_nodes["deadline_scope_from_ticks"]
    lsp_fn = fn_nodes["deadline_scope_from_lsp_env"]

    ticks_calls = {
        node.func.id
        for node in ast.walk(ticks_fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    lsp_calls = {
        node.func.id
        for node in ast.walk(lsp_fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert {"forest_scope", "deadline_scope", "deadline_clock_scope"} <= ticks_calls
    assert "deadline_scope_from_ticks" in lsp_calls
