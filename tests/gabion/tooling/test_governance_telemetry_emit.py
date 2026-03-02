from __future__ import annotations

import json
from pathlib import Path

from scripts import governance_telemetry_emit


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# gabion:evidence E:function_site::governance_telemetry_emit.py::scripts.governance_telemetry_emit.main E:decision_surface/direct::governance_telemetry_emit.py::scripts.governance_telemetry_emit.main::stale_c2ac1aba5bc9_61694c3d
def test_emit_governance_telemetry_outputs_schema(tmp_path: Path) -> None:
    docflow = tmp_path / "artifacts/out/docflow_compliance_delta.json"
    obsolescence = tmp_path / "artifacts/out/test_obsolescence_delta.json"
    annotation = tmp_path / "artifacts/out/test_annotation_drift_delta.json"
    ambiguity = tmp_path / "artifacts/out/ambiguity_delta.json"
    branchless = tmp_path / "baselines/branchless_policy_baseline.json"
    defensive = tmp_path / "baselines/defensive_fallback_policy_baseline.json"
    telemetry = tmp_path / "artifacts/out/governance_telemetry.json"
    history = tmp_path / "artifacts/out/governance_telemetry_history.json"
    markdown = tmp_path / "artifacts/audit_reports/governance_telemetry.md"

    _write_json(docflow, {"summary": {"current": {"contradicts": 2}}})
    _write_json(
        obsolescence,
        {
            "summary": {
                "counts": {"current": {"unmapped": 3}},
                "opaque_evidence": {"current": 1},
            }
        },
    )
    _write_json(annotation, {"summary": {"current": {"orphaned": 4}}})
    _write_json(ambiguity, {"summary": {"total": {"current": 5}}})
    _write_json(branchless, {"violations": [{"id": 1}, {"id": 2}]})
    _write_json(defensive, {"violations": [{"id": 1}]})

    assert (
        governance_telemetry_emit.main(
            [
                "--run-id",
                "run-1",
                "--docflow-delta",
                str(docflow),
                "--obsolescence-delta",
                str(obsolescence),
                "--annotation-delta",
                str(annotation),
                "--ambiguity-delta",
                str(ambiguity),
                "--branchless-baseline",
                str(branchless),
                "--defensive-baseline",
                str(defensive),
                "--json-out",
                str(telemetry),
                "--history",
                str(history),
                "--md-out",
                str(markdown),
            ]
        )
        == 0
    )
    payload = json.loads(telemetry.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    loops = {entry["loop_id"]: entry for entry in payload["loops"]}
    assert loops["policy.branchless"]["violation_count"] == 2
    assert loops["docflow.contradictions"]["violation_count"] == 2
    assert loops["delta.obsolescence_unmapped"]["violation_count"] == 3
    assert "Convergence SLOs" in markdown.read_text(encoding="utf-8")


# gabion:evidence E:function_site::governance_telemetry_emit.py::scripts.governance_telemetry_emit.main E:decision_surface/direct::governance_telemetry_emit.py::scripts.governance_telemetry_emit.main::stale_6f6e142c28ec
def test_emit_governance_telemetry_sets_trends_from_history(tmp_path: Path) -> None:
    docflow = tmp_path / "docflow_delta.json"
    branchless = tmp_path / "branchless.json"
    defensive = tmp_path / "defensive.json"
    history = tmp_path / "history.json"
    out_json = tmp_path / "telemetry.json"
    out_md = tmp_path / "telemetry.md"

    _write_json(docflow, {"summary": {"current": {"contradicts": 0}}})
    _write_json(branchless, {"violations": []})
    _write_json(defensive, {"violations": []})
    _write_json(
        history,
        {
            "schema_version": 1,
            "runs": [
                {
                    "run_id": "old-1",
                    "loops": [
                        {"loop_id": "docflow.contradictions", "violation_count": 2},
                        {"loop_id": "policy.branchless", "violation_count": 0},
                    ],
                }
            ],
        },
    )

    assert (
        governance_telemetry_emit.main(
            [
                "--run-id",
                "run-2",
                "--docflow-delta",
                str(docflow),
                "--branchless-baseline",
                str(branchless),
                "--defensive-baseline",
                str(defensive),
                "--history",
                str(history),
                "--json-out",
                str(out_json),
                "--md-out",
                str(out_md),
            ]
        )
        == 0
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    loops = {entry["loop_id"]: entry for entry in payload["loops"]}
    assert loops["docflow.contradictions"]["trend_delta"] == -2
    assert loops["docflow.contradictions"]["time_to_correction_runs"] == 1
    assert len(json.loads(history.read_text(encoding="utf-8"))["runs"]) == 2
