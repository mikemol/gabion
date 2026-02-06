from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


def test_render_and_diff_decision_snapshots() -> None:
    da = _load()
    baseline = da.render_decision_snapshot(
        decision_surfaces=["a.py:f decision surface params: x"],
        value_decision_surfaces=[],
        project_root=Path("."),
    )
    current = da.render_decision_snapshot(
        decision_surfaces=[
            "a.py:f decision surface params: x",
            "b.py:g decision surface params: y",
        ],
        value_decision_surfaces=["c.py:h value-encoded decision params: z (min/max)"],
        project_root=Path("."),
    )
    diff = da.diff_decision_snapshots(baseline, current)
    assert diff["decision_surfaces"]["added"] == ["b.py:g decision surface params: y"]
    assert diff["decision_surfaces"]["removed"] == []
    assert diff["value_decision_surfaces"]["added"] == [
        "c.py:h value-encoded decision params: z (min/max)"
    ]
