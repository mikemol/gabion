from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


def test_compute_structure_metrics_counts() -> None:
    da = _load()
    groups_by_path = {
        Path("a.py"): {"f": [set(["a", "b"]), set(["c"])]},
        Path("b.py"): {"g": [set(["d"])]},
    }
    metrics = da.compute_structure_metrics(groups_by_path)
    assert metrics["files"] == 2
    assert metrics["functions"] == 2
    assert metrics["bundles"] == 3
    assert metrics["max_bundle_size"] == 2
    assert metrics["bundle_size_histogram"] == {1: 2, 2: 1}
