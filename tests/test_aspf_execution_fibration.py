from __future__ import annotations

from pathlib import Path

import pytest

from gabion.analysis import aspf_execution_fibration
from gabion.analysis.aspf_core import AspfOneCell, BasisZeroCell
from gabion.analysis.aspf_morphisms import (
    AspfPrimeBasis,
    DomainPrimeBasis,
    DomainToAspfCofibration,
    DomainToAspfCofibrationEntry,
)


def _trace_payload(
    *,
    trace_json: Path,
    state_json: Path | None = None,
    equivalence_against: list[Path] | None = None,
    import_state: list[Path] | None = None,
    surfaces: list[str] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "aspf_trace_json": trace_json,
        "aspf_opportunities_json": trace_json.with_name("opportunities.json"),
        "aspf_state_json": state_json or trace_json.with_name("state.json"),
    }
    if equivalence_against:
        payload["aspf_equivalence_against"] = equivalence_against
    if import_state:
        payload["aspf_import_state"] = import_state
    if surfaces is not None:
        payload["aspf_semantic_surface"] = surfaces
    return payload


# gabion:evidence E:function_site::aspf_execution_fibration.py::gabion.analysis.aspf_execution_fibration.start_execution_trace
def test_start_execution_trace_is_opt_in(tmp_path: Path) -> None:
    assert (
        aspf_execution_fibration.start_execution_trace(root=tmp_path, payload={})
        is None
    )


# gabion:evidence E:call_footprint::tests/test_aspf_execution_fibration.py::test_finalize_execution_trace_emits_non_drift_with_matching_baseline::aspf_execution_fibration.py::gabion.analysis.aspf_execution_fibration.finalize_execution_trace
def test_finalize_execution_trace_emits_non_drift_with_matching_baseline(
    tmp_path: Path,
) -> None:
    baseline_trace = tmp_path / "baseline_trace.json"
    baseline_state = aspf_execution_fibration.start_execution_trace(
        root=tmp_path,
        payload=_trace_payload(
            trace_json=baseline_trace,
            surfaces=["groups_by_path", "violation_summary"],
        ),
    )
    assert baseline_state is not None
    baseline_artifacts = aspf_execution_fibration.finalize_execution_trace(
        state=baseline_state,
        root=tmp_path,
        semantic_surface_payloads={
            "groups_by_path": {"pkg/mod.py": {"fn": [{"a", "b"}]}},
            "violation_summary": {"violations": 0},
        },
    )
    assert baseline_artifacts is not None

    current_trace = tmp_path / "current_trace.json"
    current_state = aspf_execution_fibration.start_execution_trace(
        root=tmp_path,
        payload=_trace_payload(
            trace_json=current_trace,
            equivalence_against=[baseline_trace],
            surfaces=["groups_by_path", "violation_summary"],
        ),
    )
    assert current_state is not None
    current_artifacts = aspf_execution_fibration.finalize_execution_trace(
        state=current_state,
        root=tmp_path,
        semantic_surface_payloads={
            "groups_by_path": {"pkg/mod.py": {"fn": [{"a", "b"}]}},
            "violation_summary": {"violations": 0},
        },
    )
    assert current_artifacts is not None
    assert current_artifacts.equivalence_payload["verdict"] == "non_drift"
    rows = current_artifacts.equivalence_payload["surface_table"]
    assert isinstance(rows, list)
    group_row = next(
        row for row in rows if isinstance(row, dict) and row.get("surface") == "groups_by_path"
    )
    assert group_row["classification"] == "non_drift"
    assert (tmp_path / "artifacts/out/aspf_equivalence.json").exists()


# gabion:evidence E:call_footprint::tests/test_aspf_execution_fibration.py::test_finalize_execution_trace_marks_only_mismatched_surface_as_drift::aspf_execution_fibration.py::gabion.analysis.aspf_execution_fibration.finalize_execution_trace
def test_finalize_execution_trace_marks_only_mismatched_surface_as_drift(
    tmp_path: Path,
) -> None:
    baseline_trace = tmp_path / "baseline_trace.json"
    baseline_state = aspf_execution_fibration.start_execution_trace(
        root=tmp_path,
        payload=_trace_payload(
            trace_json=baseline_trace,
            surfaces=["groups_by_path", "violation_summary"],
        ),
    )
    assert baseline_state is not None
    aspf_execution_fibration.finalize_execution_trace(
        state=baseline_state,
        root=tmp_path,
        semantic_surface_payloads={
            "groups_by_path": {"pkg/mod.py": {"fn": [{"a", "b"}]}},
            "violation_summary": {"violations": 0},
        },
    )

    current_trace = tmp_path / "current_trace.json"
    current_state = aspf_execution_fibration.start_execution_trace(
        root=tmp_path,
        payload=_trace_payload(
            trace_json=current_trace,
            equivalence_against=[baseline_trace],
            surfaces=["groups_by_path", "violation_summary"],
        ),
    )
    assert current_state is not None
    artifacts = aspf_execution_fibration.finalize_execution_trace(
        state=current_state,
        root=tmp_path,
        semantic_surface_payloads={
            "groups_by_path": {"pkg/mod.py": {"fn": [{"z"}]}},
            "violation_summary": {"violations": 0},
        },
    )
    assert artifacts is not None
    table = artifacts.equivalence_payload["surface_table"]
    assert isinstance(table, list)
    by_surface = {
        str(row.get("surface")): str(row.get("classification"))
        for row in table
        if isinstance(row, dict)
    }
    assert by_surface["groups_by_path"] == "drift"
    assert by_surface["violation_summary"] == "non_drift"


# gabion:evidence E:function_site::aspf_execution_fibration.py::gabion.analysis.aspf_execution_fibration.record_cofibration
def test_record_cofibration_rejects_non_faithful_embedding(tmp_path: Path) -> None:
    state = aspf_execution_fibration.start_execution_trace(
        root=tmp_path,
        payload=_trace_payload(trace_json=tmp_path / "trace.json"),
    )
    assert state is not None
    non_faithful = DomainToAspfCofibration(
        entries=(
            DomainToAspfCofibrationEntry(
                domain=DomainPrimeBasis("domain:a", 2),
                aspf=AspfPrimeBasis("aspf:a", 3),
            ),
        )
    )
    with pytest.raises(ValueError):
        aspf_execution_fibration.record_cofibration(
            state=state,
            canonical_identity_kind="canonical_aspf_execution_surface",
            cofibration=non_faithful,
        )


# gabion:evidence E:function_site::aspf_execution_fibration.py::gabion.analysis.aspf_execution_fibration.merge_imported_trace
def test_merge_imported_trace_preserves_surface_representatives(tmp_path: Path) -> None:
    state = aspf_execution_fibration.start_execution_trace(
        root=tmp_path,
        payload=_trace_payload(trace_json=tmp_path / "trace.json"),
    )
    assert state is not None
    aspf_execution_fibration.merge_imported_trace(
        state=state,
        trace_payload={
            "surface_representatives": {"groups_by_path": "rep:stable"},
            "one_cells": [],
            "two_cell_witnesses": [],
        },
    )
    assert state.surface_representatives["groups_by_path"] == "rep:stable"


# gabion:evidence E:function_site::aspf_execution_fibration.py::gabion.analysis.aspf_execution_fibration.build_opportunities_payload
def test_build_opportunities_payload_emits_materialize_and_fungible_candidates(
    tmp_path: Path,
) -> None:
    state = aspf_execution_fibration.start_execution_trace(
        root=tmp_path,
        payload=_trace_payload(
            trace_json=tmp_path / "trace.json",
            surfaces=["groups_by_path"],
        ),
    )
    assert state is not None
    current_cell = aspf_execution_fibration.register_semantic_surface(
        state=state,
        surface="groups_by_path",
        value={"pkg/mod.py": {"fn": [{"a"}]}},
    )
    assert current_cell is not None
    aspf_execution_fibration.record_1cell(
        state,
        kind="resume_load",
        source_label="runtime:aspf_state",
        target_label="analysis:resume_seed",
        representative="aspf_state_loaded",
        basis_path=("resume", "load", "aspf_state"),
        metadata={"import_state_path": "artifacts/out/aspf_state/session/step.snapshot.json"},
    )
    aspf_execution_fibration.record_1cell(
        state,
        kind="resume_write",
        source_label="analysis:resume_seed",
        target_label="runtime:aspf_state",
        representative="aspf_state_written",
        basis_path=("resume", "write", "aspf_state"),
        metadata={"state_path": "artifacts/out/aspf_state/session/step.snapshot.json"},
    )
    baseline_rep = "baseline-groups-rep"
    baseline_cell = AspfOneCell(
        source=BasisZeroCell("surface:groups_by_path:domain"),
        target=BasisZeroCell("surface:groups_by_path:carrier"),
        representative=baseline_rep,
        basis_path=("groups_by_path", "post", "projection"),
    )
    aspf_execution_fibration.record_2cell_witness(
        state,
        left=baseline_cell,
        right=current_cell,
        witness_id="w:groups-fungible",
        reason="equivalent semantic surface projection",
    )
    equivalence = aspf_execution_fibration.build_equivalence_payload(
        state=state,
        baseline_traces=[{"surface_representatives": {"groups_by_path": baseline_rep}}],
    )
    opportunities = aspf_execution_fibration.build_opportunities_payload(
        state=state,
        equivalence_payload=equivalence,
    )
    rows = opportunities["opportunities"]
    assert isinstance(rows, list)
    kinds = {str(row.get("kind")) for row in rows if isinstance(row, dict)}
    assert "materialize_load_fusion" in kinds
    assert "fungible_execution_path_substitution" in kinds


# gabion:evidence E:call_footprint::tests/test_aspf_execution_fibration.py::test_finalize_execution_trace_allows_state_object_roundtrip_import::aspf_execution_fibration.py::gabion.analysis.aspf_execution_fibration.finalize_execution_trace
def test_finalize_execution_trace_allows_state_object_roundtrip_import(
    tmp_path: Path,
) -> None:
    baseline_trace = tmp_path / "baseline_trace.json"
    baseline_state_json = tmp_path / "state" / "session-a" / "0001_baseline.json"
    baseline_state = aspf_execution_fibration.start_execution_trace(
        root=tmp_path,
        payload=_trace_payload(
            trace_json=baseline_trace,
            state_json=baseline_state_json,
            surfaces=["groups_by_path", "violation_summary"],
        ),
    )
    assert baseline_state is not None
    baseline_artifacts = aspf_execution_fibration.finalize_execution_trace(
        state=baseline_state,
        root=tmp_path,
        semantic_surface_payloads={
            "groups_by_path": {"pkg/mod.py": {"fn": [{"a"}]}},
            "violation_summary": {"violations": 0},
        },
        exit_code=0,
        analysis_state="succeeded",
    )
    assert baseline_artifacts is not None
    assert baseline_state_json.exists()

    current_trace = tmp_path / "current_trace.json"
    current_state_json = tmp_path / "state" / "session-a" / "0002_current.json"
    current_state = aspf_execution_fibration.start_execution_trace(
        root=tmp_path,
        payload=_trace_payload(
            trace_json=current_trace,
            state_json=current_state_json,
            import_state=[baseline_state_json],
            surfaces=["groups_by_path", "violation_summary"],
        ),
    )
    assert current_state is not None
    current_artifacts = aspf_execution_fibration.finalize_execution_trace(
        state=current_state,
        root=tmp_path,
        semantic_surface_payloads={
            "groups_by_path": {"pkg/mod.py": {"fn": [{"a"}]}},
            "violation_summary": {"violations": 0},
        },
        exit_code=0,
        analysis_state="succeeded",
    )
    assert current_artifacts is not None
    assert current_artifacts.equivalence_payload["verdict"] == "non_drift"
    assert current_artifacts.state_payload is not None
    assert current_artifacts.state_path == current_state_json
    state_payload = current_artifacts.state_payload
    assert state_payload["session_id"] == "session-a"
    assert state_payload["step_id"] == "0002_current"
    assert state_payload["analysis_state"] == "succeeded"
