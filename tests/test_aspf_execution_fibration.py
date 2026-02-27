from __future__ import annotations

from pathlib import Path
import json

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
        payload["aspf_equivalence_against"] = [str(path) for path in equivalence_against]
    if import_state:
        payload["aspf_import_state"] = [str(path) for path in import_state]
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
            "cofibration_witnesses": [],
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
    rewrite_plans = opportunities["rewrite_plans"]
    assert isinstance(rewrite_plans, list)
    fungible_plan = next(
        plan for plan in rewrite_plans if isinstance(plan, dict) and plan.get("opportunity_id") == "opp:fungible-substitution:groups_by_path"
    )
    assert fungible_plan["actionability"] == "actionable"
    assert fungible_plan["required_witnesses"] == ["w:groups-fungible"]


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


# gabion:evidence E:function_site::aspf_execution_fibration.py::gabion.analysis.aspf_execution_fibration._publish_event
def test_aspf_event_visitor_receives_finalize_hook(tmp_path: Path) -> None:
    class CollectingVisitor:
        def __init__(self) -> None:
            self.one_cells = 0
            self.surfaces = 0
            self.finalized = 0
            self.on_finalize_calls = 0

        def visit_one_cell_recorded(self, event) -> None:
            self.one_cells += 1

        def visit_two_cell_witness_recorded(self, event) -> None:
            return None

        def visit_cofibration_recorded(self, event) -> None:
            return None

        def visit_semantic_surface_updated(self, event) -> None:
            self.surfaces += 1

        def visit_run_finalized(self, event) -> None:
            self.finalized += 1

        def on_finalize(self, event) -> None:
            self.on_finalize_calls += 1

    state = aspf_execution_fibration.start_execution_trace(
        root=tmp_path,
        payload=_trace_payload(
            trace_json=tmp_path / "trace.json",
            surfaces=["groups_by_path"],
        ),
    )
    assert state is not None
    collector = CollectingVisitor()
    state.event_visitors.append(collector)

    artifacts = aspf_execution_fibration.finalize_execution_trace(
        state=state,
        root=tmp_path,
        semantic_surface_payloads={
            "groups_by_path": {"pkg/mod.py": {"fn": [{"a"}]}}
        },
    )

    assert artifacts is not None
    assert collector.surfaces == 1
    assert collector.one_cells == 1
    assert collector.finalized == 1
    assert collector.on_finalize_calls == 1


# gabion:evidence E:function_site::aspf_execution_fibration.py::gabion.analysis.aspf_execution_fibration.AspfExecutionTraceState.__post_init__
def test_execution_trace_state_preserves_preconfigured_event_visitors(tmp_path: Path) -> None:
    class _Visitor:
        def visit_one_cell_recorded(self, event) -> None:
            pass

        def visit_two_cell_witness_recorded(self, event) -> None:
            pass

        def visit_cofibration_recorded(self, event) -> None:
            pass

        def visit_semantic_surface_updated(self, event) -> None:
            pass

        def visit_run_finalized(self, event) -> None:
            pass

        def on_finalize(self, event) -> None:
            pass

    visitor = _Visitor()
    state = aspf_execution_fibration.AspfExecutionTraceState(
        trace_id="aspf-trace:test",
        controls=aspf_execution_fibration.controls_from_payload(
            {"aspf_trace_json": str(tmp_path / "trace.json")}
        ),
        started_at_utc="2026-01-01T00:00:00Z",
        command_profile="check.run",
        event_visitors=[visitor],
    )
    assert state.event_visitors == [visitor]


def test_start_execution_trace_registers_streaming_sink(tmp_path: Path) -> None:
    state = aspf_execution_fibration.start_execution_trace(
        root=tmp_path,
        payload=_trace_payload(
            trace_json=tmp_path / "trace.json",
            surfaces=["groups_by_path"],
        ),
    )
    assert state is not None
    assert state.event_sinks


def test_finalize_execution_trace_derives_payload_from_sink_index(tmp_path: Path) -> None:
    state = aspf_execution_fibration.start_execution_trace(
        root=tmp_path,
        payload=_trace_payload(
            trace_json=tmp_path / "trace.json",
            surfaces=["groups_by_path"],
        ),
    )
    assert state is not None

    aspf_execution_fibration.register_semantic_surface(
        state=state,
        surface="groups_by_path",
        value={"pkg/mod.py": {"fn": [{"a"}]}, "count": 1},
    )
    state.one_cells.clear()
    state.one_cell_metadata.clear()
    state.two_cell_witnesses.clear()
    state.cofibrations.clear()
    state.surface_representatives.clear()
    state.delta_records.clear()

    artifacts = aspf_execution_fibration.finalize_execution_trace(
        state=state,
        root=tmp_path,
        semantic_surface_payloads={"groups_by_path": {"pkg/mod.py": {"fn": [{"a"}]}}},
    )
    assert artifacts is not None
    assert artifacts.trace_payload["one_cells"]
    assert artifacts.trace_payload["delta_record_count"] > 0


def _one_cell_payload(*, representative: str) -> dict[str, object]:
    return {
        "source": "surface:groups_by_path:domain",
        "target": "surface:groups_by_path:carrier",
        "representative": representative,
        "basis_path": ["groups_by_path", "post", "projection"],
    }


# gabion:evidence E:function_site::aspf_execution_fibration.py::gabion.analysis.aspf_execution_fibration._merge_two_cell_payload
def test_merge_imported_trace_parses_two_cell_witness_payloads(tmp_path: Path) -> None:
    state = aspf_execution_fibration.start_execution_trace(
        root=tmp_path,
        payload=_trace_payload(
            trace_json=tmp_path / "trace.json",
            surfaces=["groups_by_path"],
        ),
    )
    assert state is not None

    aspf_execution_fibration.merge_imported_trace(
        state=state,
        trace_payload={
            "surface_representatives": {"groups_by_path": "rep:baseline"},
            "one_cells": [],
            "two_cell_witnesses": [
                {
                    "left": _one_cell_payload(representative="rep:baseline"),
                    "right": _one_cell_payload(representative="rep:current"),
                    "witness_id": "w:imported",
                    "reason": "imported witness",
                }
            ],
            "cofibration_witnesses": [],
        },
    )

    assert len(state.two_cell_witnesses) == 1
    assert state.two_cell_witnesses[0].witness_id == "w:imported"


def _write_stream_trace_fixture(path: Path, *, trace_payload: dict[str, object]) -> None:
    rows: list[dict[str, object]] = []
    for index, one_cell in enumerate(trace_payload.get("one_cells", [])):
        rows.append({"kind": "one_cell", "sequence": 20 + index, "payload": one_cell})
    for index, witness in enumerate(trace_payload.get("two_cell_witnesses", [])):
        rows.append({"kind": "two_cell", "sequence": 40 + index, "payload": witness})
    for index, cofibration in enumerate(trace_payload.get("cofibration_witnesses", [])):
        rows.append({"kind": "cofibration", "sequence": 60 + index, "payload": cofibration})
    for index, (surface, representative) in enumerate(
        sorted(trace_payload.get("surface_representatives", {}).items())
    ):
        rows.append(
            {
                "kind": "surface_update",
                "sequence": 10 + index,
                "surface": surface,
                "representative": representative,
            }
        )
    # Deliberately reverse write order; importer must reorder by sequence deterministically.
    lines = [
        json.dumps(row, sort_keys=False)
        for row in sorted(rows, key=lambda row: int(row["sequence"]), reverse=True)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# gabion:evidence E:function_site::aspf_execution_fibration.py::gabion.analysis.aspf_execution_fibration.load_trace_stream_payload
def test_merge_imported_trace_paths_streaming_matches_legacy_payload_merge(tmp_path: Path) -> None:
    controls = _trace_payload(trace_json=tmp_path / "trace.json", surfaces=["groups_by_path"])

    legacy_state = aspf_execution_fibration.start_execution_trace(
        root=tmp_path / "legacy",
        payload=controls,
    )
    stream_state = aspf_execution_fibration.start_execution_trace(
        root=tmp_path / "stream",
        payload=controls,
    )
    assert legacy_state is not None
    assert stream_state is not None

    trace_payload = {
        "surface_representatives": {"groups_by_path": "rep:baseline"},
        "one_cells": [_one_cell_payload(representative="rep:baseline")],
        "two_cell_witnesses": [
            {
                "left": _one_cell_payload(representative="rep:baseline"),
                "right": _one_cell_payload(representative="rep:current"),
                "witness_id": "w:imported",
                "reason": "imported witness",
            }
        ],
        "cofibration_witnesses": [
            {
                "canonical_identity_kind": "group",
                "cofibration": {
                    "entries": [
                        {
                            "domain": {"key": "domain", "prime": 2},
                            "aspf": {"key": "aspf", "prime": 2},
                        }
                    ]
                },
            }
        ],
    }

    legacy_path = tmp_path / "import_legacy.json"
    stream_path = tmp_path / "import_stream.jsonl"
    legacy_path.write_text(json.dumps(trace_payload) + "\n", encoding="utf-8")
    _write_stream_trace_fixture(stream_path, trace_payload=trace_payload)

    aspf_execution_fibration.merge_imported_trace_paths(state=legacy_state, paths=(legacy_path,))
    aspf_execution_fibration.merge_imported_trace_paths(state=stream_state, paths=(stream_path,))

    assert [cell.as_dict() for cell in legacy_state.one_cells] == [
        cell.as_dict() for cell in stream_state.one_cells
    ]
    assert [w.as_dict() for w in legacy_state.two_cell_witnesses] == [
        w.as_dict() for w in stream_state.two_cell_witnesses
    ]
    assert [c.as_dict() for c in legacy_state.cofibrations] == [
        c.as_dict() for c in stream_state.cofibrations
    ]
    assert legacy_state.surface_representatives == stream_state.surface_representatives

# gabion:evidence E:function_site::aspf_execution_fibration.py::gabion.analysis.aspf_execution_fibration.build_equivalence_payload

def test_build_equivalence_payload_uses_baseline_two_cell_witness_index(tmp_path: Path) -> None:
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
    baseline_rep = "rep:baseline"
    equivalence = aspf_execution_fibration.build_equivalence_payload(
        state=state,
        baseline_traces=[
            {
                "surface_representatives": {"groups_by_path": baseline_rep},
                "two_cell_witnesses": [
                    {
                        "left": _one_cell_payload(representative=baseline_rep),
                        "right": _one_cell_payload(representative=current_cell.representative),
                        "witness_id": "w:baseline-index",
                        "reason": "baseline index witness",
                    }
                ],
            }
        ],
    )

    rows = equivalence["surface_table"]
    assert isinstance(rows, list)
    row = next(item for item in rows if isinstance(item, dict))
    assert row["classification"] == "non_drift"
    assert row["witness_id"] == "w:baseline-index"
