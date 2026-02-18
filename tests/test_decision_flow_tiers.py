from __future__ import annotations

from pathlib import Path

from gabion.analysis import dataflow_audit as da
from gabion.analysis.decision_flow import (
    build_decision_tables,
    detect_repeated_guard_bundles,
    enforce_decision_protocol_contracts,
)


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
    assert all(str(table["decision_id"]).startswith("dataflow:") for table in tables)
    assert all(table["analysis_evidence_keys"] for table in tables)
    assert all(table["checklist_nodes"] == ["docs/sppf_checklist.md#decision-flow-tier3"] for table in tables)


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
