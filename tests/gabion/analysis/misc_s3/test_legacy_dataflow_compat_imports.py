from __future__ import annotations

import importlib
from datetime import date


def _load(module_path: str):
    return importlib.import_module(module_path)


def _assert_lifecycle_metadata(module: object, *, expected_scope: str) -> None:
    lifecycle = getattr(module, "_BOUNDARY_ADAPTER_LIFECYCLE")
    assert isinstance(lifecycle, dict)
    for key in (
        "actor",
        "rationale",
        "scope",
        "start",
        "expiry",
        "rollback_condition",
        "evidence_links",
    ):
        assert key in lifecycle
    for key in (
        "actor",
        "rationale",
        "scope",
        "start",
        "expiry",
        "rollback_condition",
    ):
        value = lifecycle[key]
        assert isinstance(value, str)
        assert value.strip()
    scope = lifecycle["scope"]
    assert scope == expected_scope
    date.fromisoformat(lifecycle["start"])
    evidence_links = lifecycle["evidence_links"]
    assert isinstance(evidence_links, list)
    assert evidence_links
    assert all(isinstance(link, str) and link.strip() for link in evidence_links)
    assert "docs/ws5_decomposition_ledger.md" in evidence_links


def test_legacy_dataflow_compat_modules_import() -> None:
    indexed = _load("gabion.analysis.dataflow.engine.dataflow_indexed_file_scan")
    analysis_owner = _load("gabion.analysis.dataflow.engine.dataflow_analysis_index_owner")
    deadline_owner = _load("gabion.analysis.dataflow.engine.dataflow_deadline_runtime_owner")
    reporting_owner = _load("gabion.analysis.dataflow.engine.dataflow_runtime_reporting_owner")
    summary_owner = _load("gabion.analysis.dataflow.engine.dataflow_deadline_summary_owner")
    facade = _load("gabion.analysis.dataflow.engine.dataflow_facade")

    assert hasattr(indexed, "_build_analysis_index")
    assert hasattr(indexed, "_DeadlineFunctionCollector")

    assert hasattr(analysis_owner, "_build_analysis_index")
    assert hasattr(deadline_owner, "_resolve_callee")
    assert hasattr(reporting_owner, "_report_section_spec")
    assert hasattr(summary_owner, "_summarize_deadline_obligations")
    assert hasattr(facade, "render_report")

    _assert_lifecycle_metadata(
        indexed, expected_scope="dataflow_indexed_file_scan.alias_surface"
    )
    _assert_lifecycle_metadata(
        analysis_owner, expected_scope="dataflow_analysis_index_owner.alias_surface"
    )
    _assert_lifecycle_metadata(
        deadline_owner, expected_scope="dataflow_deadline_runtime_owner.alias_surface"
    )
    _assert_lifecycle_metadata(
        reporting_owner, expected_scope="dataflow_runtime_reporting_owner.alias_surface"
    )
    _assert_lifecycle_metadata(
        summary_owner, expected_scope="dataflow_deadline_summary_owner.alias_surface"
    )
    _assert_lifecycle_metadata(
        facade, expected_scope="dataflow_facade.alias_surface"
    )
