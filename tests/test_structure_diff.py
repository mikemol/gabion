from __future__ import annotations

from pathlib import Path
import sys
import json


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


def test_diff_structure_snapshots_detects_changes() -> None:
    da = _load()
    baseline = {
        "files": [
            {"path": "a.py", "functions": [{"name": "f", "bundles": [["a", "b"]]}]},
        ]
    }
    current = {
        "files": [
            {"path": "a.py", "functions": [{"name": "f", "bundles": [["a", "b"], ["c"]]}]},
        ]
    }
    diff = da.diff_structure_snapshots(baseline, current)
    assert diff["added"][0]["bundle"] == ["c"]
    assert diff["removed"] == []
    assert diff["summary"]["added"] == 1


def test_load_structure_snapshot_reads_json(tmp_path: Path) -> None:
    da = _load()
    payload = {"files": []}
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(payload))
    loaded = da.load_structure_snapshot(path)
    assert loaded == payload
