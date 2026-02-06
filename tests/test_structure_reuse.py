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
    suggestions = reuse.get("suggested_lemmas", [])
    assert any(
        entry.get("kind") == "bundle" and entry.get("suggested_name")
        for entry in suggestions
    )
    replacement_map = reuse.get("replacement_map", {})
    assert any(location.startswith("a.py::f") for location in replacement_map)


def test_render_reuse_lemma_stubs_includes_names() -> None:
    da = _load()
    reuse = {
        "format_version": 1,
        "min_count": 2,
        "suggested_lemmas": [
            {
                "hash": "deadbeef",
                "kind": "bundle",
                "count": 2,
                "suggested_name": "_gabion_bundle_lemma_deadbeef",
                "locations": ["a.py::f::bundle:a,b"],
                "value": ["a", "b"],
            }
        ],
    }
    stubs = da.render_reuse_lemma_stubs(reuse)
    assert "_gabion_bundle_lemma_deadbeef" in stubs


def test_structure_reuse_prefers_declared_bundle_names(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    target.write_text(
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class UserContext:\n"
        "    user_id: int\n"
        "    request_id: str\n"
    )
    snapshot = {
        "format_version": 1,
        "root": str(tmp_path),
        "files": [
            {
                "path": "mod.py",
                "functions": [
                    {"name": "f", "bundles": [["user_id", "request_id"]]},
                    {"name": "g", "bundles": [["request_id", "user_id"]]},
                ],
            }
        ],
    }
    reuse = da.compute_structure_reuse(snapshot, min_count=2)
    suggestions = reuse.get("suggested_lemmas", [])
    assert any(
        entry.get("kind") == "bundle"
        and entry.get("suggested_name") == "UserContext"
        and entry.get("name_source") == "declared_bundle"
        for entry in suggestions
    )
    assert reuse.get("warnings") == []
