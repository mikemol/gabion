from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._infer_root::groups_by_path E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::root
def test_structure_snapshot_includes_invariants(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text("")
    groups_by_path = {path: {"f": [set(["a", "b"])]}}
    prop = da.InvariantProposition(
        form="Equal",
        terms=("a.length", "b.length"),
        scope="mod.py:f",
        source="assert",
    )
    forest = da.Forest()
    da._populate_bundle_forest(
        forest,
        groups_by_path=groups_by_path,
        file_paths=[path],
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
        invariant_propositions=[prop],
    )
    fn_entry = snapshot["files"][0]["functions"][0]
    assert fn_entry["invariants"][0]["form"] == "Equal"
    assert fn_entry["invariants"][0]["terms"] == ["a.length", "b.length"]
