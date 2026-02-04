from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


def test_compute_structure_reuse_detects_repeated_subtrees() -> None:
    da = _load()
    snapshot = {
        "format_version": 1,
        "root": ".",
        "files": [
            {
                "path": "a.py",
                "functions": [
                    {"name": "f", "bundles": [["a", "b"], ["c"]]},
                ],
            },
            {
                "path": "b.py",
                "functions": [
                    {"name": "g", "bundles": [["b", "a"], ["c"]]},
                ],
            },
        ],
    }
    reuse = da.compute_structure_reuse(snapshot, min_count=2)
    kinds = {entry.get("kind") for entry in reuse.get("reused", [])}
    assert "bundle" in kinds
    assert "function" in kinds
    bundle_entries = [
        entry for entry in reuse["reused"] if entry.get("kind") == "bundle"
    ]
    assert any(entry.get("value") == ["a", "b"] for entry in bundle_entries)
