from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.synthesis.merge import merge_bundles

    return merge_bundles


def test_merge_bundles_combines_overlaps() -> None:
    merge_bundles = _load()
    bundles = [{"a", "b"}, {"a", "b", "c"}]
    merged = merge_bundles(bundles, min_overlap=0.66)
    assert merged == [{"a", "b", "c"}]


def test_merge_bundles_keeps_distinct() -> None:
    merge_bundles = _load()
    bundles = [{"a", "b"}, {"c", "d"}]
    merged = merge_bundles(bundles, min_overlap=0.5)
    assert merged == [{"a", "b"}, {"c", "d"}]
