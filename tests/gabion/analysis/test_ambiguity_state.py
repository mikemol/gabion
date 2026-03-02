from __future__ import annotations

import json
from pathlib import Path

import pytest

from gabion.analysis import ambiguity_state


# gabion:evidence E:function_site::ambiguity_state.py::gabion.analysis.ambiguity_state.build_state_payload E:function_site::ambiguity_state.py::gabion.analysis.ambiguity_state.load_state
def test_state_payload_roundtrip(tmp_path: Path) -> None:
    witnesses = [
        {
            "kind": "z-kind",
            "site": {
                "path": "b.py",
                "function": "beta",
                "span": [2, 0, 2, 4],
            },
            "candidate_count": 3,
        },
        {
            "kind": "a-kind",
            "site": {
                "path": "a.py",
                "function": "alpha",
                "span": [1, 0, 1, 4],
            },
            "candidate_count": 2,
        },
    ]
    payload = ambiguity_state.build_state_payload(witnesses)
    assert payload["version"] == ambiguity_state.STATE_VERSION
    assert payload["generated_by_spec_id"]
    state_path = tmp_path / "ambiguity_state.json"
    state_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    state = ambiguity_state.load_state(str(state_path))
    assert state.baseline.total == 2
    assert list(state.witnesses)[0]["kind"] == "a-kind"


# gabion:evidence E:function_site::ambiguity_state.py::gabion.analysis.ambiguity_state.parse_state_payload E:decision_surface/direct::ambiguity_state.py::gabion.analysis.ambiguity_state.parse_state_payload::stale_d1aabd327aea
def test_state_payload_rejects_bad_version() -> None:
    with pytest.raises(ValueError):
        ambiguity_state.parse_state_payload({"version": "nope"})


# gabion:evidence E:function_site::ambiguity_state.py::gabion.analysis.ambiguity_state.parse_state_payload E:decision_surface/direct::ambiguity_state.py::gabion.analysis.ambiguity_state.parse_state_payload::stale_ca1caebddff1_37efef4b
def test_state_payload_filters_invalid_witness_entries() -> None:
    payload = {
        "version": ambiguity_state.STATE_VERSION,
        "ambiguity_witnesses": [
            "not-a-mapping",
            {"kind": "kind-a", "site": "nope", "candidate_count": 1},
            {
                "kind": "kind-b",
                "site": {"path": "a.py", "function": "alpha", "span": [1, 2, 3]},
                "candidate_count": 2,
            },
        ],
        "generated_by_spec_id": "spec",
        "generated_by_spec": {},
    }
    state = ambiguity_state.parse_state_payload(payload)
    assert len(state.witnesses) == 2


# gabion:evidence E:function_site::ambiguity_state.py::gabion.analysis.ambiguity_state.parse_state_payload E:decision_surface/direct::ambiguity_state.py::gabion.analysis.ambiguity_state.parse_state_payload::stale_c3249b8882e9
def test_state_payload_handles_non_iterable_witnesses() -> None:
    payload = {
        "version": ambiguity_state.STATE_VERSION,
        "ambiguity_witnesses": "nope",
        "generated_by_spec_id": "spec",
        "generated_by_spec": {},
    }
    state = ambiguity_state.parse_state_payload(payload)
    assert state.witnesses == []


# gabion:evidence E:function_site::ambiguity_state.py::gabion.analysis.ambiguity_state.load_state E:decision_surface/direct::ambiguity_state.py::gabion.analysis.ambiguity_state.load_state::stale_4a7a0f5b6a2b
def test_state_payload_rejects_non_object(tmp_path: Path) -> None:
    state_path = tmp_path / "ambiguity_state.json"
    state_path.write_text(json.dumps(["not", "object"]), encoding="utf-8")
    with pytest.raises(ValueError):
        ambiguity_state.load_state(str(state_path))
