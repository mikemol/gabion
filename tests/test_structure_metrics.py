from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest
def test_compute_structure_metrics_counts(tmp_path: Path) -> None:
    da = _load()
    path_a = tmp_path / "a.py"
    path_b = tmp_path / "b.py"
    path_a.write_text("")
    path_b.write_text("")
    groups_by_path = {
        path_a: {"f": [set(["a", "b"]), set(["c"])]},
        path_b: {"g": [set(["d"])]},
    }
    forest = da.Forest()
    da._populate_bundle_forest(
        forest,
        groups_by_path=groups_by_path,
        file_paths=[path_a, path_b],
        project_root=None,
        include_all_sites=True,
        ignore_params=set(),
        strictness="high",
        transparent_decorators=None,
    )
    metrics = da.compute_structure_metrics(groups_by_path, forest=forest)
    assert metrics["files"] == 2
    assert metrics["functions"] == 2
    assert metrics["bundles"] == 3
    assert metrics["max_bundle_size"] == 2
    assert metrics["bundle_size_histogram"] == {"1": 2, "2": 1}
    assert "forest_signature" in metrics
