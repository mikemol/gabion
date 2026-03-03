from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gabion.analysis.aspf.aspf import Forest
from gabion.analysis.dataflow.io.dataflow_snapshot_contracts import DecisionSnapshotSurfaces
from gabion.analysis.dataflow.io.dataflow_snapshot_io import (
    diff_decision_snapshots, render_decision_snapshot)
from gabion.analysis.projection.pattern_schema_projection import (
    pattern_schema_matches as _pattern_schema_matches)


def _load():
    return SimpleNamespace(
        DecisionSnapshotSurfaces=DecisionSnapshotSurfaces,
        Forest=Forest,
        _pattern_schema_matches=_pattern_schema_matches,
        diff_decision_snapshots=diff_decision_snapshots,
        render_decision_snapshot=render_decision_snapshot,
    )


# gabion:evidence E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.render_decision_snapshot::forest,project_root E:decision_surface/direct::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.render_decision_snapshot::stale_33ce2d4f668a
def test_render_and_diff_decision_snapshots() -> None:
    da = _load()
    forest = da.Forest()
    site_id = forest.add_site("a.py", "f")
    paramset_id = forest.add_paramset(["x"])
    forest.add_alt("DecisionSurface", (site_id, paramset_id))
    baseline = da.render_decision_snapshot(
        surfaces=da.DecisionSnapshotSurfaces(
            decision_surfaces=["a.py:f decision surface params: x"],
            value_decision_surfaces=[],
        ),
        project_root=Path("."),
        forest=forest,
        groups_by_path={},
    )
    assert "generated_by_forest_spec_id" in baseline
    assert "generated_by_forest_spec" in baseline
    assert "forest_signature" in baseline
    current = da.render_decision_snapshot(
        surfaces=da.DecisionSnapshotSurfaces(
            decision_surfaces=[
                "a.py:f decision surface params: x",
                "b.py:g decision surface params: y",
            ],
            value_decision_surfaces=["c.py:h value-encoded decision params: z (min/max)"],
        ),
        project_root=Path("."),
        forest=forest,
        groups_by_path={},
    )
    diff = da.diff_decision_snapshots(baseline, current)
    assert diff["decision_surfaces"]["added"] == ["b.py:g decision surface params: y"]
    assert diff["decision_surfaces"]["removed"] == []
    assert diff["value_decision_surfaces"]["added"] == [
        "c.py:h value-encoded decision params: z (min/max)"
    ]
    assert "baseline_forest_signature" in diff
    assert "current_forest_signature" in diff


# gabion:evidence E:call_footprint::tests/test_decision_snapshots.py::test_render_decision_snapshot_includes_pattern_schema_residue::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan._pattern_schema_matches::dataflow_indexed_file_scan.py::gabion.analysis.dataflow_indexed_file_scan.render_decision_snapshot::test_decision_snapshots.py::tests.test_decision_snapshots._load
def test_render_decision_snapshot_includes_pattern_schema_residue() -> None:
    da = _load()
    forest = da.Forest()
    site_id = forest.add_site("a.py", "f")
    paramset_id = forest.add_paramset(["x"])
    forest.add_alt("DecisionSurface", (site_id, paramset_id))
    source = (
        "def one(paths, *, project_root, ignore_params, strictness, external_filter, "
        "transparent_decorators=None, parse_failure_witnesses=None, analysis_index=None):\n"
        "    return _build_analysis_index(paths, project_root=project_root, "
        "ignore_params=ignore_params, strictness=strictness, external_filter=external_filter, "
        "transparent_decorators=transparent_decorators, parse_failure_witnesses=parse_failure_witnesses)\n"
        "\n"
        "def two(paths, *, project_root, ignore_params, strictness, external_filter, "
        "transparent_decorators=None, parse_failure_witnesses=None, analysis_index=None):\n"
        "    return _build_call_graph(paths, project_root=project_root, ignore_params=ignore_params, "
        "strictness=strictness, external_filter=external_filter, transparent_decorators=transparent_decorators, "
        "parse_failure_witnesses=parse_failure_witnesses, analysis_index=analysis_index)\n"
        "\n"
        "def three(paths, *, project_root, ignore_params, strictness, external_filter, "
        "transparent_decorators=None, parse_failure_witnesses=None, analysis_index=None):\n"
        "    return _build_call_graph(paths, project_root=project_root, ignore_params=ignore_params, "
        "strictness=strictness, external_filter=external_filter, transparent_decorators=transparent_decorators, "
        "parse_failure_witnesses=parse_failure_witnesses, analysis_index=analysis_index)\n"
    )
    pattern_schema_instances = da._pattern_schema_matches(
        groups_by_path={Path("mod.py"): {"f": [set(["a", "b"])], "g": [set(["a", "b"])]}},
        source=source,
    )
    snapshot = da.render_decision_snapshot(
        surfaces=da.DecisionSnapshotSurfaces(
            decision_surfaces=["a.py:f decision surface params: x"],
            value_decision_surfaces=[],
        ),
        project_root=Path("."),
        forest=forest,
        groups_by_path={},
        pattern_schema_instances=pattern_schema_instances,
    )
    assert snapshot["pattern_schema_instances"]
    assert snapshot["pattern_schema_residue"]
    summary = snapshot.get("summary") or {}
    assert summary.get("pattern_schema_instances") == len(
        snapshot["pattern_schema_instances"]
    )
    assert summary.get("pattern_schema_residue") == len(
        snapshot["pattern_schema_residue"]
    )
