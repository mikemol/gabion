from __future__ import annotations

import json
from pathlib import Path

from scripts import annotation_drift_orphaned_gate


# gabion:evidence E:function_site::annotation_drift_orphaned_gate.py::scripts.annotation_drift_orphaned_gate.check_gate
def test_annotation_drift_gate_disabled(tmp_path: Path) -> None:
    delta_path = tmp_path / "delta.json"
    payload = {
        "summary": {
            "baseline": {"orphaned": 0},
            "current": {"orphaned": 1},
            "delta": {"orphaned": 1},
        }
    }
    delta_path.write_text(json.dumps(payload, sort_keys=True))
    assert (
        annotation_drift_orphaned_gate.check_gate(delta_path, enabled=False) == 0
    )


# gabion:evidence E:function_site::annotation_drift_orphaned_gate.py::scripts.annotation_drift_orphaned_gate.check_gate
def test_annotation_drift_gate_enabled(tmp_path: Path) -> None:
    delta_path = tmp_path / "delta.json"
    payload = {
        "summary": {
            "baseline": {"orphaned": 0},
            "current": {"orphaned": 1},
            "delta": {"orphaned": 1},
        }
    }
    delta_path.write_text(json.dumps(payload, sort_keys=True))
    assert (
        annotation_drift_orphaned_gate.check_gate(delta_path, enabled=True) == 1
    )
    payload["summary"]["current"]["orphaned"] = 0
    payload["summary"]["delta"]["orphaned"] = 0
    delta_path.write_text(json.dumps(payload, sort_keys=True))
    assert (
        annotation_drift_orphaned_gate.check_gate(delta_path, enabled=True) == 0
    )
