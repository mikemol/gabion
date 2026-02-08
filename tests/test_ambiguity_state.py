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


# gabion:evidence E:function_site::ambiguity_state.py::gabion.analysis.ambiguity_state.parse_state_payload
def test_state_payload_rejects_bad_version() -> None:
    with pytest.raises(ValueError):
        ambiguity_state.parse_state_payload({"version": "nope"})
