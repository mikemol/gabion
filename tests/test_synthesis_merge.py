from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.synthesis.merge import _jaccard, merge_bundles

    return _jaccard, merge_bundles


# gabion:evidence E:decision_surface/direct::merge.py::gabion.synthesis.merge._jaccard::left,right E:decision_surface/direct::merge.py::gabion.synthesis.merge.merge_bundles::min_overlap E:decision_surface/value_encoded::merge.py::gabion.synthesis.merge._jaccard::left,right
def test_merge_bundles_combines_overlaps() -> None:
    _jaccard, merge_bundles = _load()
    bundles = [{"a", "b"}, {"a", "b", "c"}]
    merged = merge_bundles(bundles, min_overlap=0.66)
    assert merged == [{"a", "b", "c"}]
    assert _jaccard(set(), set()) == 1.0


# gabion:evidence E:decision_surface/direct::merge.py::gabion.synthesis.merge._jaccard::left,right E:decision_surface/direct::merge.py::gabion.synthesis.merge.merge_bundles::min_overlap E:decision_surface/value_encoded::merge.py::gabion.synthesis.merge._jaccard::left,right
def test_merge_bundles_keeps_distinct() -> None:
    _jaccard, merge_bundles = _load()
    bundles = [{"a", "b"}, {"c", "d"}]
    merged = merge_bundles(bundles, min_overlap=0.5)
    assert merged == [{"a", "b"}, {"c", "d"}]
    assert _jaccard(set(), {"x"}) == 0.0
