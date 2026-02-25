from __future__ import annotations

from pathlib import Path

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis import evidence

    return evidence

# gabion:evidence E:decision_surface/direct::evidence.py::gabion.analysis.evidence.normalize_bundle_key::bundle
def test_normalize_bundle_key_accepts_lists_tuples_and_sets() -> None:
    evidence = _load()

    assert evidence.normalize_bundle_key("not-a-sequence") == ""
    assert evidence.normalize_bundle_key([]) == ""
    assert evidence.normalize_bundle_key(["b", "a", "a", "", 1]) == "a,b"
    assert evidence.normalize_bundle_key(("a", "b")) == "a,b"
    assert evidence.normalize_bundle_key({"b", "a"}) == "a,b"

# gabion:evidence E:decision_surface/direct::evidence.py::gabion.analysis.evidence.normalize_string_list::value E:decision_surface/direct::evidence.py::gabion.analysis.evidence.normalize_string_list::stale_e2867cee89d9
def test_normalize_string_list_handles_multiple_payload_shapes() -> None:
    evidence = _load()

    assert evidence.normalize_string_list(None) == []
    assert evidence.normalize_string_list("a, b, a") == ["a", "b"]
    assert evidence.normalize_string_list(["b", "a", "a"]) == ["a", "b"]
    assert evidence.normalize_string_list(("b", "a")) == ["a", "b"]
    assert evidence.normalize_string_list({"b", "a"}) == ["a", "b"]
    assert evidence.normalize_string_list({"a,b", "c"}) == ["a", "b", "c"]
    assert evidence.normalize_string_list(123) == []

# gabion:evidence E:decision_surface/direct::evidence.py::gabion.analysis.evidence.Site.from_payload::payload E:decision_surface/direct::evidence.py::gabion.analysis.evidence.normalize_string_list::value
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


# gabion:evidence E:decision_surface/direct::evidence.py::gabion.analysis.evidence.exception_obligation_summary_for_site::obligations
def test_exception_obligation_summary_for_site_skips_non_matching_and_normalizes_status() -> None:
    evidence = _load()
    site = evidence.Site(path="a.py", function="f", bundle=("a",))
    obligations = [
        {"site": "bad"},
        {"site": {"path": "other.py", "function": "f", "bundle": ["a"]}, "status": "HANDLED"},
        {"site": {"path": "a.py", "function": "f", "bundle": ["b"]}, "status": "DEAD"},
        {"site": {"path": "a.py", "function": "f", "bundle": ["a"]}, "status": "INVALID"},
        {"site": {"path": "a.py", "function": "f", "bundle": ["a"]}, "status": "HANDLED"},
    ]

    summary = evidence.exception_obligation_summary_for_site(obligations, site=site)
    assert summary == {"UNKNOWN": 1, "DEAD": 0, "HANDLED": 1, "total": 2}
