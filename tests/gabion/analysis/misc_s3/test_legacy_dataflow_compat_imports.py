from __future__ import annotations

import importlib


def _load(module_path: str):
    return importlib.import_module(module_path)


def _assert_lifecycle_metadata(module: object) -> None:
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
    assert hasattr(facade, "_report_section_spec")

    _assert_lifecycle_metadata(indexed)
    _assert_lifecycle_metadata(analysis_owner)
    _assert_lifecycle_metadata(deadline_owner)
    _assert_lifecycle_metadata(reporting_owner)
    _assert_lifecycle_metadata(summary_owner)
    _assert_lifecycle_metadata(facade)
