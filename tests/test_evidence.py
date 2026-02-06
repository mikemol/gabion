from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import evidence

    return evidence


def test_normalize_bundle_key_accepts_lists_tuples_and_sets() -> None:
    evidence = _load()

    assert evidence.normalize_bundle_key("not-a-sequence") == ""
    assert evidence.normalize_bundle_key([]) == ""
    assert evidence.normalize_bundle_key(["b", "a", "a", "", 1]) == "a,b"
    assert evidence.normalize_bundle_key(("a", "b")) == "a,b"
    assert evidence.normalize_bundle_key({"b", "a"}) == "a,b"


def test_normalize_string_list_handles_multiple_payload_shapes() -> None:
    evidence = _load()

    assert evidence.normalize_string_list(None) == []
    assert evidence.normalize_string_list("a, b, a") == ["a", "b"]
    assert evidence.normalize_string_list(["b", "a", "a"]) == ["a", "b"]
    assert evidence.normalize_string_list(("b", "a")) == ["a", "b"]
    assert evidence.normalize_string_list({"b", "a"}) == ["a", "b"]
    assert evidence.normalize_string_list({"a,b", "c"}) == ["a", "b", "c"]
    assert evidence.normalize_string_list(123) == []


def test_site_from_payload_filters_and_normalizes() -> None:
    evidence = _load()

    assert evidence.Site.from_payload("nope") is None
    site = evidence.Site.from_payload(
        {"path": " a.py ", "function": " f ", "bundle": ["b", "a", 1, "a"]}
    )
    assert site is not None
    assert site.path == "a.py"
    assert site.function == "f"
    assert list(site.bundle) == ["a", "b"]
    assert site.bundle_key() == "a,b"
    assert site.key() == ("a.py", "f", "a,b")

