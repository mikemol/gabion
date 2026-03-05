from __future__ import annotations

import ast
import importlib
from pathlib import Path


def _load(module_path: str):
    return importlib.import_module(module_path)


def _assert_alias_parity(*, owner_module_path: str, canonical_module_path: str) -> None:
    owner = _load(owner_module_path)
    canonical = _load(canonical_module_path)

    owner_all = getattr(owner, "__all__", ())
    assert owner_all, f"{owner_module_path} must expose __all__"

    missing = [name for name in owner_all if not hasattr(canonical, name)]
    assert not missing, (
        f"{owner_module_path} exports symbols missing from canonical owner "
        f"{canonical_module_path}: {missing}"
    )

    mismatched = [
        name
        for name in owner_all
        if getattr(owner, name) is not getattr(canonical, name)
    ]
    assert not mismatched, (
        f"{owner_module_path} exports non-alias symbols vs {canonical_module_path}: "
        f"{mismatched}"
    )


def _assert_symbol_alias(
    *,
    module_path: str,
    canonical_module_path: str,
    symbol: str,
) -> None:
    module = _load(module_path)
    canonical = _load(canonical_module_path)
    assert hasattr(module, symbol)
    assert hasattr(canonical, symbol)
    assert getattr(module, symbol) is getattr(canonical, symbol)


def _monolith_post_phase_aliases() -> tuple[str, ...]:
    repo_root = Path(__file__).resolve().parents[4]
    monolith_path = (
        repo_root / "src/gabion/analysis/dataflow/engine/dataflow_indexed_file_scan.py"
    )
    tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == (
            "gabion.analysis.dataflow.engine.dataflow_post_phase_analyses"
        ):
            return tuple(alias.name for alias in node.names)
    raise AssertionError("monolith post-phase import surface not found")


def _monolith_projection_aliases() -> tuple[str, ...]:
    repo_root = Path(__file__).resolve().parents[4]
    monolith_path = (
        repo_root / "src/gabion/analysis/dataflow/engine/dataflow_indexed_file_scan.py"
    )
    tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == (
            "gabion.analysis.dataflow.engine.dataflow_projection_materialization"
        ):
            return tuple(alias.name for alias in node.names)
    raise AssertionError("monolith projection import surface not found")


def _monolith_resume_aliases() -> tuple[str, ...]:
    repo_root = Path(__file__).resolve().parents[4]
    monolith_path = (
        repo_root / "src/gabion/analysis/dataflow/engine/dataflow_indexed_file_scan.py"
    )
    tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == (
            "gabion.analysis.dataflow.engine.dataflow_resume_serialization"
        ):
            return tuple(alias.name for alias in node.names)
    raise AssertionError("monolith resume import surface not found")


def test_legacy_owner_modules_preserve_alias_parity() -> None:
    _assert_alias_parity(
        owner_module_path=(
            "gabion.analysis.dataflow.engine.dataflow_analysis_index_owner"
        ),
        canonical_module_path="gabion.analysis.dataflow.engine.dataflow_analysis_index",
    )
    _assert_alias_parity(
        owner_module_path=(
            "gabion.analysis.dataflow.engine.dataflow_deadline_runtime_owner"
        ),
        canonical_module_path=(
            "gabion.analysis.dataflow.engine.dataflow_deadline_runtime"
        ),
    )
    _assert_alias_parity(
        owner_module_path=(
            "gabion.analysis.dataflow.engine.dataflow_runtime_reporting_owner"
        ),
        canonical_module_path=(
            "gabion.analysis.dataflow.engine.dataflow_runtime_reporting"
        ),
    )
    _assert_alias_parity(
        owner_module_path=(
            "gabion.analysis.dataflow.engine.dataflow_deadline_summary_owner"
        ),
        canonical_module_path=(
            "gabion.analysis.dataflow.engine.dataflow_deadline_summary"
        ),
    )


def test_legacy_monolith_and_facade_selected_symbol_alias_parity() -> None:
    # Monolith compatibility aliases by phase owner.
    _assert_symbol_alias(
        module_path="gabion.analysis.dataflow.engine.dataflow_indexed_file_scan",
        canonical_module_path="gabion.analysis.dataflow.engine.dataflow_analysis_index",
        symbol="_build_analysis_index",
    )
    _assert_symbol_alias(
        module_path="gabion.analysis.dataflow.engine.dataflow_indexed_file_scan",
        canonical_module_path="gabion.analysis.dataflow.engine.dataflow_deadline_helpers",
        symbol="_DeadlineFunctionCollector",
    )
    _assert_symbol_alias(
        module_path="gabion.analysis.dataflow.engine.dataflow_indexed_file_scan",
        canonical_module_path=(
            "gabion.analysis.dataflow.engine.dataflow_post_phase_analyses"
        ),
        symbol="analyze_deadness_flow_repo",
    )
    _assert_symbol_alias(
        module_path="gabion.analysis.dataflow.engine.dataflow_indexed_file_scan",
        canonical_module_path=(
            "gabion.analysis.dataflow.engine.dataflow_projection_materialization"
        ),
        symbol="_materialize_projection_spec_rows",
    )
    _assert_symbol_alias(
        module_path="gabion.analysis.dataflow.engine.dataflow_indexed_file_scan",
        canonical_module_path=(
            "gabion.analysis.dataflow.engine.dataflow_resume_serialization"
        ),
        symbol="_serialize_file_scan_resume_state",
    )

    # Facade compatibility aliases across canonical owners.
    _assert_symbol_alias(
        module_path="gabion.analysis.dataflow.engine.dataflow_facade",
        canonical_module_path="gabion.analysis.dataflow.engine.dataflow_analysis_index",
        symbol="_build_analysis_index",
    )
    _assert_symbol_alias(
        module_path="gabion.analysis.dataflow.engine.dataflow_facade",
        canonical_module_path="gabion.analysis.dataflow.engine.dataflow_deadline_helpers",
        symbol="_resolve_callee",
    )
    _assert_symbol_alias(
        module_path="gabion.analysis.dataflow.engine.dataflow_facade",
        canonical_module_path=(
            "gabion.analysis.dataflow.engine.dataflow_runtime_reporting"
        ),
        symbol="_report_section_spec",
    )


def test_facade_covers_monolith_post_phase_alias_surface() -> None:
    facade = _load("gabion.analysis.dataflow.engine.dataflow_facade")
    canonical = _load("gabion.analysis.dataflow.engine.dataflow_post_phase_analyses")
    for symbol in _monolith_post_phase_aliases():
        assert hasattr(facade, symbol), (
            "facade must carry full monolith post-phase compatibility surface; "
            f"missing={symbol}"
        )
        assert getattr(facade, symbol) is getattr(canonical, symbol), (
            "facade post-phase symbol must remain an alias to canonical owner; "
            f"symbol={symbol}"
        )


def test_facade_covers_monolith_projection_alias_surface() -> None:
    facade = _load("gabion.analysis.dataflow.engine.dataflow_facade")
    canonical = _load(
        "gabion.analysis.dataflow.engine.dataflow_projection_materialization"
    )
    for symbol in _monolith_projection_aliases():
        assert hasattr(facade, symbol), (
            "facade must carry full monolith projection compatibility surface; "
            f"missing={symbol}"
        )
        assert getattr(facade, symbol) is getattr(canonical, symbol), (
            "facade projection symbol must remain an alias to canonical owner; "
            f"symbol={symbol}"
        )


def test_facade_covers_monolith_resume_alias_surface() -> None:
    facade = _load("gabion.analysis.dataflow.engine.dataflow_facade")
    canonical = _load("gabion.analysis.dataflow.engine.dataflow_resume_serialization")
    for symbol in _monolith_resume_aliases():
        assert hasattr(facade, symbol), (
            "facade must carry full monolith resume compatibility surface; "
            f"missing={symbol}"
        )
        assert getattr(facade, symbol) is getattr(canonical, symbol), (
            "facade resume symbol must remain an alias to canonical owner; "
            f"symbol={symbol}"
        )
