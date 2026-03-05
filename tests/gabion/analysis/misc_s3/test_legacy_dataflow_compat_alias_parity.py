from __future__ import annotations

import importlib


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
