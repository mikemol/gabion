from __future__ import annotations

from pathlib import Path

from gabion.analysis import dataflow_audit as da
from gabion.analysis.decision_flow import (
    build_decision_tables,
    detect_repeated_guard_bundles,
    enforce_decision_protocol_contracts,
)


# gabion:evidence E:function_site::decision_flow.py::gabion.analysis.decision_flow.build_decision_tables E:decision_surface/direct::decision_flow.py::gabion.analysis.decision_flow.build_decision_tables::stale_4aa23e3894e5
def test_tier3_decision_tables_emit_deterministic_ids_and_links() -> None:
    tables = build_decision_tables(
        decision_surfaces=[
            "mod.py:mod.f decision surface params: mode, flag (internal callers (transitive): 1)",
        ],
        value_decision_surfaces=[
            "mod.py:mod.g value-encoded decision params: mode, flag (min/max)",
        ],
    )
    assert len(tables) == 2
    assert all(table["tier"] == 3 for table in tables)
    assert all(str(table["decision_id"]).startswith("schema:") for table in tables)
    assert all(table["analysis_evidence_keys"] for table in tables)
    assert all(table["checklist_nodes"] == ["docs/sppf_checklist.md#decision-flow-tier3"] for table in tables)


# gabion:evidence E:function_site::decision_flow.py::gabion.analysis.decision_flow.build_decision_tables E:function_site::decision_flow.py::gabion.analysis.decision_flow.detect_repeated_guard_bundles E:decision_surface/direct::decision_flow.py::gabion.analysis.decision_flow.build_decision_tables::stale_a23ae4d1adf0
def test_tier2_repeated_guard_detection_collects_bundle() -> None:
    tables = build_decision_tables(
        decision_surfaces=[
            "mod.py:mod.f decision surface params: mode, flag",
            "mod.py:mod.h decision surface params: flag, mode",
        ],
        value_decision_surfaces=[],
    )
    bundles = detect_repeated_guard_bundles(tables)
    assert len(bundles) == 1
    bundle = bundles[0]
    assert bundle["tier"] == 2
    assert bundle["params"] == ["flag", "mode"]
    assert bundle["occurrences"] == 2
    assert bundle["checklist_nodes"] == ["docs/sppf_checklist.md#decision-flow-tier2"]


# gabion:evidence E:function_site::decision_flow.py::gabion.analysis.decision_flow.detect_repeated_guard_bundles E:decision_surface/direct::decision_flow.py::gabion.analysis.decision_flow.detect_repeated_guard_bundles::stale_ead0bfccf83d
def test_detect_repeated_guard_bundles_skips_entries_without_params() -> None:
    bundles = detect_repeated_guard_bundles(
        [
            {"decision_id": "a", "params": []},
            {"decision_id": "b"},
        ]
    )
    assert bundles == []


# gabion:evidence E:function_site::decision_flow.py::gabion.analysis.decision_flow.build_decision_tables E:function_site::decision_flow.py::gabion.analysis.decision_flow.detect_repeated_guard_bundles E:function_site::decision_flow.py::gabion.analysis.decision_flow.enforce_decision_protocol_contracts
def test_tier1_schema_enforcement_reports_contract_drift() -> None:
    tables = build_decision_tables(
        decision_surfaces=[
            "mod.py:mod.f decision surface params: mode, flag",
            "mod.py:mod.h decision surface params: mode, flag",
        ],
        value_decision_surfaces=[],
    )
    tables[0]["analysis_evidence_keys"] = []
    bundles = detect_repeated_guard_bundles(tables)
    violations = enforce_decision_protocol_contracts(
        decision_tables=tables,
        decision_bundles=bundles,
    )
    assert any(v["code"] == "DECISION_PROTOCOL_MISSING_EVIDENCE" for v in violations)


# gabion:evidence E:function_site::decision_flow.py::gabion.analysis.decision_flow.enforce_decision_protocol_contracts E:decision_surface/direct::decision_flow.py::gabion.analysis.decision_flow.enforce_decision_protocol_contracts::stale_077cc474e231
def test_tier1_schema_enforcement_reports_empty_members_missing_table_and_checklist() -> None:
    violations = enforce_decision_protocol_contracts(
        decision_tables=[
            {
                "decision_id": "table-without-checklist",
                "analysis_evidence_keys": ["E:decision"],
                "checklist_nodes": [],
            }
        ],
        decision_bundles=[
            {"bundle_id": "bundle-non-list", "member_decision_ids": "not-a-list"},
            {"bundle_id": "bundle-missing-table", "member_decision_ids": ["missing-table-id"]},
            {"bundle_id": "bundle-missing-checklist", "member_decision_ids": ["table-without-checklist"]},
        ],
    )
    codes = {str(item.get("code")) for item in violations}
    assert "DECISION_PROTOCOL_EMPTY_MEMBERS" in codes
    assert "DECISION_PROTOCOL_MISSING_TABLE" in codes
    assert "DECISION_PROTOCOL_MISSING_CHECKLIST_LINK" in codes


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit.render_decision_snapshot
def test_end_to_end_snapshot_contains_tier3_to_tier1_artifacts() -> None:
    forest = da.Forest()
    site_id = forest.add_site("mod.py", "f")
    paramset_id = forest.add_paramset(["mode", "flag"])
    forest.add_alt("DecisionSurface", (site_id, paramset_id))

    snapshot = da.render_decision_snapshot(
        surfaces=da.DecisionSnapshotSurfaces(
            decision_surfaces=[
                "mod.py:mod.f decision surface params: mode, flag",
                "mod.py:mod.h decision surface params: mode, flag",
            ],
            value_decision_surfaces=[
                "mod.py:mod.g value-encoded decision params: mode, flag (min/max)",
            ],
        ),
        project_root=Path("."),
        forest=forest,
        groups_by_path={},
    )
    assert snapshot["decision_tables"]
    assert snapshot["decision_bundles"]
    assert "decision_protocol_violations" in snapshot
    summary = snapshot["summary"]
    assert summary["decision_tables"] == len(snapshot["decision_tables"])
    assert summary["decision_bundles"] == len(snapshot["decision_bundles"])
