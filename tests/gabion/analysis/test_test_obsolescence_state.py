from __future__ import annotations

import json
from pathlib import Path

import pytest

from gabion.analysis import test_obsolescence, test_obsolescence_state


# gabion:evidence E:function_site::test_obsolescence_state.py::gabion.analysis.test_obsolescence_state.build_state_payload E:function_site::test_obsolescence_state.py::gabion.analysis.test_obsolescence_state.load_state
def test_state_payload_roundtrip(
    tmp_path: Path,
    make_obsolescence_paramset_ref,
) -> None:
    evidence_by_test = {
        "tests/test_alpha.py::test_a": [make_obsolescence_paramset_ref("x")],
        "tests/test_beta.py::test_b": [make_obsolescence_paramset_ref("x")],
    }
    status_by_test = {
        "tests/test_alpha.py::test_a": "mapped",
        "tests/test_beta.py::test_b": "mapped",
    }
    classification = test_obsolescence.classify_candidates(
        evidence_by_test, status_by_test, {}
    )
    payload = test_obsolescence_state.build_state_payload(
        evidence_by_test,
        status_by_test,
        classification.stale_candidates,
        classification.stale_summary,
        active_tests=classification.active_tests,
        active_summary=classification.active_summary,
    )
    assert payload["version"] == test_obsolescence_state.STATE_VERSION
    assert payload["generated_by_spec_id"]
    assert payload["generated_by_spec"]
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    state = test_obsolescence_state.load_state(str(state_path))
    assert len(state.candidates) == 1
    assert state.baseline.summary["equivalent_witness"] == 1
    assert state.baseline.active["summary"]["active_total"] == 1


# gabion:evidence E:function_site::test_obsolescence_state.py::gabion.analysis.test_obsolescence_state.parse_state_payload E:decision_surface/direct::test_obsolescence_state.py::gabion.analysis.test_obsolescence_state.parse_state_payload::stale_d3c07c5ff55b
def test_state_payload_rejects_bad_version() -> None:
    with pytest.raises(ValueError):
        test_obsolescence_state.parse_state_payload({"version": "nope"})


# gabion:evidence E:function_site::test_obsolescence_state.py::gabion.analysis.test_obsolescence_state.parse_state_payload E:decision_surface/direct::test_obsolescence_state.py::gabion.analysis.test_obsolescence_state.parse_state_payload::stale_2866f575ce7c_27ff8884
def test_state_payload_rejects_bad_baseline() -> None:
    with pytest.raises(ValueError):
        test_obsolescence_state.parse_state_payload(
            {"version": test_obsolescence_state.STATE_VERSION, "baseline": []}
        )


# gabion:evidence E:function_site::test_obsolescence_state.py::gabion.analysis.test_obsolescence_state.parse_state_payload E:decision_surface/direct::test_obsolescence_state.py::gabion.analysis.test_obsolescence_state.parse_state_payload::stale_953f851794b3
def test_state_payload_skips_invalid_candidates(
    make_obsolescence_paramset_ref,
) -> None:
    evidence_by_test = {
        "tests/test_alpha.py::test_a": [make_obsolescence_paramset_ref("x")],
    }
    status_by_test = {
        "tests/test_alpha.py::test_a": "mapped",
    }
    classification = test_obsolescence.classify_candidates(
        evidence_by_test, status_by_test, {}
    )
    payload = test_obsolescence_state.build_state_payload(
        evidence_by_test,
        status_by_test,
        classification.stale_candidates,
        classification.stale_summary,
        active_tests=classification.active_tests,
        active_summary=classification.active_summary,
    )
    payload["candidates"] = ["not-a-mapping"]
    state = test_obsolescence_state.parse_state_payload(payload)
    assert state.candidates == []


# gabion:evidence E:call_footprint::tests/test_test_obsolescence_state.py::test_state_payload_ignores_non_list_candidates_payload::test_obsolescence.py::gabion.analysis.test_obsolescence.classify_candidates::test_obsolescence_state.py::gabion.analysis.test_obsolescence_state.build_state_payload::test_obsolescence_state.py::gabion.analysis.test_obsolescence_state.parse_state_payload
def test_state_payload_ignores_non_list_candidates_payload(
    make_obsolescence_paramset_ref,
) -> None:
    evidence_by_test = {
        "tests/test_alpha.py::test_a": [make_obsolescence_paramset_ref("x")],
    }
    status_by_test = {
        "tests/test_alpha.py::test_a": "mapped",
    }
    classification = test_obsolescence.classify_candidates(
        evidence_by_test, status_by_test, {}
    )
    payload = test_obsolescence_state.build_state_payload(
        evidence_by_test,
        status_by_test,
        classification.stale_candidates,
        classification.stale_summary,
        active_tests=classification.active_tests,
        active_summary=classification.active_summary,
    )
    payload["candidates"] = {"bad": True}
    state = test_obsolescence_state.parse_state_payload(payload)
    assert state.candidates == []


# gabion:evidence E:function_site::test_obsolescence_state.py::gabion.analysis.test_obsolescence_state.load_state E:decision_surface/direct::test_obsolescence_state.py::gabion.analysis.test_obsolescence_state.load_state::stale_6ccd66180a88
def test_state_payload_rejects_non_object(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(["nope"]), encoding="utf-8")
    with pytest.raises(ValueError):
        test_obsolescence_state.load_state(str(state_path))
