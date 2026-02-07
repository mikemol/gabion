from __future__ import annotations

from pathlib import Path
import json
import sys

import pytest


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


def _write_snapshot(path: Path, snapshot: dict) -> None:
    path.write_text(json.dumps(snapshot))


def test_load_structure_snapshot_invalid_json(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "bad.json"
    target.write_text("{not-json")
    with pytest.raises(ValueError):
        da.load_structure_snapshot(target)


def test_load_structure_snapshot_non_object(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "list.json"
    target.write_text("[1, 2, 3]")
    with pytest.raises(ValueError):
        da.load_structure_snapshot(target)


def test_bundle_counts_skips_invalid_entries() -> None:
    da = _load()
    snapshot = {
        "files": [
            "not-a-dict",
            {"functions": "not-a-list"},
            {
                "functions": [
                    "not-a-dict",
                    {"bundles": "not-a-list"},
                    {"bundles": ["not-a-bundle"]},
                    {"bundles": [["a", "b"], ["c"]]},
                ]
            },
        ]
    }
    counts = da._bundle_counts_from_snapshot(snapshot)
    assert counts == {("a", "b"): 1, ("c",): 1}


def test_diff_structure_snapshots_counts_and_summary() -> None:
    da = _load()
    baseline = {
        "root": "/baseline",
        "files": [{"functions": [{"bundles": [["a", "b"], ["c"]]}]}],
    }
    current = {
        "root": "/current",
        "files": [{"functions": [{"bundles": [["a", "b"], ["a", "b"], ["d"]]}]}],
    }
    diff = da.diff_structure_snapshots(baseline, current)
    assert diff["summary"] == {
        "added": 1,
        "removed": 1,
        "changed": 1,
        "baseline_total": 2,
        "current_total": 3,
    }
    assert diff["baseline_forest_signature_partial"] is True
    assert diff["baseline_forest_signature_basis"] == "missing"
    assert diff["current_forest_signature_partial"] is True
    assert diff["current_forest_signature_basis"] == "missing"
    assert diff["added"][0]["bundle"] == ["d"]
    assert diff["removed"][0]["bundle"] == ["c"]
    assert diff["changed"][0]["bundle"] == ["a", "b"]


def test_diff_structure_snapshot_files(tmp_path: Path) -> None:
    da = _load()
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _write_snapshot(
        baseline,
        {"root": "base", "files": [{"functions": [{"bundles": [["x"]]}]}]},
    )
    _write_snapshot(
        current,
        {"root": "cur", "files": [{"functions": [{"bundles": [["x"], ["y"]]}]}]},
    )
    diff = da.diff_structure_snapshot_files(baseline, current)
    assert diff["summary"]["added"] == 1
