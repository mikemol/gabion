from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._infer_root::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::root
def test_render_structure_snapshot_orders_entries(tmp_path: Path) -> None:
    da = _load()
    path_a = tmp_path / "a.py"
    path_b = tmp_path / "b.py"
    groups_by_path = {
        path_b: {"g": [set(["b", "a"]), set(["c"])]},
        path_a: {"f": [set(["d", "c"]), set(["a"])]},
    }
    snapshot = da.render_structure_snapshot(groups_by_path, project_root=tmp_path)
    assert snapshot["root"] == str(tmp_path)
    assert "generated_by_forest_spec_id" in snapshot
    assert "generated_by_forest_spec" in snapshot
    assert "forest_signature" in snapshot
    assert snapshot["forest_signature_partial"] is True
    assert snapshot["forest_signature_basis"] == "bundles_only"
    files = snapshot["files"]
    assert [entry["path"] for entry in files] == ["a.py", "b.py"]
    fn_entry = files[0]["functions"][0]
    assert fn_entry["name"] == "f"
    assert fn_entry["bundles"][0] == ["a"]
    assert fn_entry["bundles"][1] == ["c", "d"]


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._infer_root::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::root
def test_render_structure_snapshot_handles_outside_root(tmp_path: Path) -> None:
    da = _load()
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.py"
    groups_by_path = {outside: {"f": [set(["a"])]}}
    snapshot = da.render_structure_snapshot(groups_by_path, project_root=root)
    assert snapshot["files"][0]["path"] == str(outside)
