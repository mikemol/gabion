from __future__ import annotations

from pathlib import Path
import json
from types import SimpleNamespace

import pytest
from gabion.analysis.dataflow.io.dataflow_snapshot_contracts import StructureSnapshotDiffRequest
from gabion.analysis.dataflow.io.dataflow_snapshot_io import (
    _bundle_counts_from_snapshot, diff_structure_snapshot_files, diff_structure_snapshots, load_structure_snapshot)

def _load():
    return SimpleNamespace(
        StructureSnapshotDiffRequest=StructureSnapshotDiffRequest,
        _bundle_counts_from_snapshot=_bundle_counts_from_snapshot,
        diff_structure_snapshot_files=diff_structure_snapshot_files,
        diff_structure_snapshots=diff_structure_snapshots,
        load_structure_snapshot=load_structure_snapshot,
    )

def _write_snapshot(path: Path, snapshot: dict) -> None:
    path.write_text(json.dumps(snapshot))

# gabion:evidence E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.load_structure_snapshot E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.load_structure_snapshot::stale_3573052e7745_e728a4bf
# gabion:behavior primary=verboten facets=invalid
def test_load_structure_snapshot_invalid_json(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "bad.json"
    target.write_text("{not-json")
    with pytest.raises(ValueError):
        da.load_structure_snapshot(target)

# gabion:evidence E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.load_structure_snapshot E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.load_structure_snapshot::stale_0579969a1f47
# gabion:behavior primary=desired
def test_load_structure_snapshot_non_object(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "list.json"
    target.write_text("[1, 2, 3]")
    with pytest.raises(ValueError):
        da.load_structure_snapshot(target)

# gabion:evidence E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._bundle_counts_from_snapshot
# gabion:behavior primary=verboten facets=invalid
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

# gabion:evidence E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.diff_structure_snapshots
# gabion:behavior primary=desired
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

# gabion:evidence E:function_site::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.diff_structure_snapshot_files
# gabion:behavior primary=desired
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
    diff = da.diff_structure_snapshot_files(
        da.StructureSnapshotDiffRequest(baseline_path=baseline, current_path=current)
    )
    assert diff["summary"]["added"] == 1
