from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit.compute_structure_metrics::forest
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
    assert metrics["bundle_size_histogram"] == {"1": 2, "2": 1}
    assert "forest_signature" in metrics
    assert metrics["forest_signature_partial"] is True
    assert metrics["forest_signature_basis"] == "bundles_only"
