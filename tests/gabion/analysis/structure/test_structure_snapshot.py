from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gabion.analysis.aspf.aspf import Forest
from gabion.analysis.dataflow.engine.dataflow_ambiguity_helpers import _populate_bundle_forest
from gabion.analysis.dataflow.io.dataflow_snapshot_io import render_structure_snapshot

def _load():
    return SimpleNamespace(
        Forest=Forest,
        _populate_bundle_forest=_populate_bundle_forest,
        render_structure_snapshot=render_structure_snapshot,
    )

# gabion:evidence E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._infer_root::groups_by_path E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._normalize_snapshot_path::root E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._infer_root::stale_0aac8650094d
# gabion:behavior primary=desired
def test_render_structure_snapshot_orders_entries(tmp_path: Path) -> None:
    da = _load()
    path_a = tmp_path / "a.py"
    path_b = tmp_path / "b.py"
    path_a.write_text("")
    path_b.write_text("")
    groups_by_path = {
        path_b: {"g": [set(["b", "a"]), set(["c"])]},
        path_a: {"f": [set(["d", "c"]), set(["a"])]},
    }
    forest = da.Forest()
    da._populate_bundle_forest(
        forest,
        groups_by_path=groups_by_path,
        file_paths=[path_a, path_b],
        project_root=tmp_path,
        include_all_sites=True,
        ignore_params=set(),
        strictness="high",
        transparent_decorators=None,
        parse_failure_witnesses=[],
    )
    snapshot = da.render_structure_snapshot(
        groups_by_path,
        project_root=tmp_path,
        forest=forest,
    )
    assert snapshot["root"] == str(tmp_path)
    assert "generated_by_forest_spec_id" in snapshot
    assert "generated_by_forest_spec" in snapshot
    assert "forest_signature" in snapshot
    files = snapshot["files"]
    assert [entry["path"] for entry in files] == ["a.py", "b.py"]
    fn_entry = files[0]["functions"][0]
    assert fn_entry["name"] == "f"
    assert fn_entry["bundles"][0] == ["a"]
    assert fn_entry["bundles"][1] == ["c", "d"]

# gabion:evidence E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._infer_root::groups_by_path E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._normalize_snapshot_path::root E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._infer_root::stale_7b78ac9c9e2e
# gabion:behavior primary=desired
def test_render_structure_snapshot_handles_outside_root(tmp_path: Path) -> None:
    da = _load()
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("")
    groups_by_path = {outside: {"f": [set(["a"])]}}
    forest = da.Forest()
    da._populate_bundle_forest(
        forest,
        groups_by_path=groups_by_path,
        file_paths=[outside],
        project_root=root,
        include_all_sites=True,
        ignore_params=set(),
        strictness="high",
        transparent_decorators=None,
        parse_failure_witnesses=[],
    )
    snapshot = da.render_structure_snapshot(
        groups_by_path,
        project_root=root,
        forest=forest,
    )
    assert snapshot["files"][0]["path"] == str(outside)
