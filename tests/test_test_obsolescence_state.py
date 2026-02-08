from __future__ import annotations

import json
from pathlib import Path

import pytest

from gabion.analysis import evidence_keys, test_obsolescence, test_obsolescence_state


def _ref_paramset(value: str) -> test_obsolescence.EvidenceRef:
    key = evidence_keys.make_paramset_key([value])
    identity = evidence_keys.key_identity(key)
    return test_obsolescence.EvidenceRef(
        key=key,
        identity=identity,
        display=evidence_keys.render_display(key),
        opaque=False,
    )


# gabion:evidence E:function_site::test_obsolescence_state.py::gabion.analysis.test_obsolescence_state.build_state_payload E:function_site::test_obsolescence_state.py::gabion.analysis.test_obsolescence_state.load_state
def test_state_payload_roundtrip(tmp_path: Path) -> None:
    evidence_by_test = {
        "tests/test_alpha.py::test_a": [_ref_paramset("x")],
        "tests/test_beta.py::test_b": [_ref_paramset("x")],
    }
    status_by_test = {
        "tests/test_alpha.py::test_a": "mapped",
        "tests/test_beta.py::test_b": "mapped",
    }
    candidates, summary = test_obsolescence.classify_candidates(
        evidence_by_test, status_by_test, {}
    )
    payload = test_obsolescence_state.build_state_payload(
        evidence_by_test, status_by_test, candidates, summary
    )
    assert payload["version"] == test_obsolescence_state.STATE_VERSION
    assert payload["generated_by_spec_id"]
    assert payload["generated_by_spec"]
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    state = test_obsolescence_state.load_state(str(state_path))
    assert len(state.candidates) == 2
    assert state.baseline.summary["equivalent_witness"] == 2


# gabion:evidence E:function_site::test_obsolescence_state.py::gabion.analysis.test_obsolescence_state.parse_state_payload
def test_state_payload_rejects_bad_version() -> None:
    with pytest.raises(ValueError):
        test_obsolescence_state.parse_state_payload({"version": "nope"})


# gabion:evidence E:function_site::test_obsolescence_state.py::gabion.analysis.test_obsolescence_state.parse_state_payload
def test_state_payload_rejects_bad_baseline() -> None:
    with pytest.raises(ValueError):
        test_obsolescence_state.parse_state_payload(
            {"version": test_obsolescence_state.STATE_VERSION, "baseline": []}
        )


# gabion:evidence E:function_site::test_obsolescence_state.py::gabion.analysis.test_obsolescence_state.parse_state_payload
def test_state_payload_skips_invalid_candidates() -> None:
    evidence_by_test = {
        "tests/test_alpha.py::test_a": [_ref_paramset("x")],
    }
    status_by_test = {
        "tests/test_alpha.py::test_a": "mapped",
    }
    candidates, summary = test_obsolescence.classify_candidates(
        evidence_by_test, status_by_test, {}
    )
    payload = test_obsolescence_state.build_state_payload(
        evidence_by_test, status_by_test, candidates, summary
    )
    payload["candidates"] = ["not-a-mapping"]
    state = test_obsolescence_state.parse_state_payload(payload)
    assert state.candidates == []


# gabion:evidence E:function_site::test_obsolescence_state.py::gabion.analysis.test_obsolescence_state.load_state
def test_state_payload_rejects_non_object(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(["nope"]), encoding="utf-8")
    with pytest.raises(ValueError):
        test_obsolescence_state.load_state(str(state_path))
