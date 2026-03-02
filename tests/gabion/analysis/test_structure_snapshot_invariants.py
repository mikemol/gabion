from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gabion.analysis.aspf import Forest
from gabion.analysis.dataflow_contracts import InvariantProposition
from gabion.analysis.dataflow_ambiguity_helpers import _populate_bundle_forest
from gabion.analysis.dataflow_snapshot_io import render_structure_snapshot

def _load():
    return SimpleNamespace(
        Forest=Forest,
        InvariantProposition=InvariantProposition,
        _populate_bundle_forest=_populate_bundle_forest,
        render_structure_snapshot=render_structure_snapshot,
    )

# gabion:evidence E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.render_structure_snapshot::forest,invariant_propositions E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._infer_root::groups_by_path E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._normalize_snapshot_path::root E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._infer_root::stale_b0d8f0be62be
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
