from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da
    from gabion.analysis.type_fingerprints import build_fingerprint_registry

    return da, build_fingerprint_registry


def test_fingerprint_warnings_missing_match(tmp_path: Path) -> None:
    da, build_registry = _load()
    path = tmp_path / "mod.py"
    groups_by_path = {path: {"f": [set(["user_id", "user_name"])]}}
    annotations_by_path = {
        path: {"f": {"user_id": "int", "user_name": "str"}}
    }
    registry, index = build_registry({"known": ["int"]})
    warnings = da._compute_fingerprint_warnings(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        index=index,
    )
    assert any("fingerprint" in warning for warning in warnings)


def test_fingerprint_warnings_match_known_entry(tmp_path: Path) -> None:
    da, build_registry = _load()
    path = tmp_path / "mod.py"
    groups_by_path = {path: {"f": [set(["user_id", "user_name"])]}}
    annotations_by_path = {
        path: {"f": {"user_id": "int", "user_name": "str"}}
    }
    registry, index = build_registry({"user_context": ["int", "str"]})
    warnings = da._compute_fingerprint_warnings(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        index=index,
    )
    assert warnings == []
