from __future__ import annotations

import importlib


def test_monolith_alias_inventory_matches_materialized_surface() -> None:
    monolith = importlib.import_module(
        "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan"
    )
    inventory = monolith.DATAFLOW_INDEXED_FILE_SCAN_ALIAS_SURFACE_INVENTORY

    assert inventory["star_export_names"] == monolith.__all__
    assert len(inventory["exported_names"]) == len(set(inventory["exported_names"]))
    assert "compatibility_support" in {
        group["group_id"] for group in inventory["module_groups"]
    }
    assert "post_phase_analysis" in {
        group["group_id"] for group in inventory["module_groups"]
    }
    assert inventory["owner_adapter_count"] == 5
    assert inventory["owner_adapter_modules"] == (
        "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan_alias_adapter_compatibility",
        "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan_alias_adapter_decision",
        "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan_alias_adapter_runtime",
        "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan_alias_adapter_analysis",
        "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan_alias_adapter_projection",
    )
    assert "_emit_report" in inventory["exported_names"]
    assert "analyze_file" in inventory["exported_names"]
    assert "FunctionNode" in inventory["exported_names"]
    assert monolith._emit_report is importlib.import_module(
        "gabion.analysis.dataflow.io.dataflow_reporting"
    ).emit_report
    assert monolith.analyze_file is importlib.import_module(
        "gabion.analysis.dataflow.engine.dataflow_analysis_index"
    ).analyze_file


def test_monolith_retirement_telemetry_exposes_remaining_hot_spots() -> None:
    monolith = importlib.import_module(
        "gabion.analysis.dataflow.engine.dataflow_indexed_file_scan"
    )
    telemetry = monolith.DATAFLOW_INDEXED_FILE_SCAN_RETIREMENT_TELEMETRY

    assert telemetry["compatibility_scope"] == "dataflow_indexed_file_scan.alias_surface"
    assert telemetry["exported_alias_count"] == len(
        monolith.DATAFLOW_INDEXED_FILE_SCAN_ALIAS_SURFACE_INVENTORY["exported_names"]
    )
    assert telemetry["owner_adapter_count"] == 5
    assert telemetry["owner_adapter_modules"] == monolith.DATAFLOW_INDEXED_FILE_SCAN_ALIAS_SURFACE_INVENTORY[
        "owner_adapter_modules"
    ]
    assert telemetry["owner_module_spread"] >= 8
    assert telemetry["remaining_hot_spots"]
    assert telemetry["remaining_hot_spots"][0] == {
        "module_path": (
            "gabion.analysis.dataflow.engine.dataflow_post_phase_analyses"
        ),
        "alias_count": 35,
    }
    assert telemetry["compatibility_policy_surfaces"] == {
        "private_symbol_allowlist_path": "docs/policy/private_symbol_import_allowlist.txt",
        "debt_ledger_path": "docs/audits/dataflow_runtime_debt_ledger.md",
        "retirement_ledger_path": "docs/audits/dataflow_runtime_retirement_ledger.md",
        "decomposition_ledger_path": "docs/ws5_decomposition_ledger.md",
    }
