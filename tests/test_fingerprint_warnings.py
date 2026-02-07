from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da
    from gabion.analysis.type_fingerprints import build_fingerprint_registry

    return da, build_fingerprint_registry


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index
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


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index
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


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.format_fingerprint::fingerprint E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index
def test_fingerprint_matches_report_known_entry(tmp_path: Path) -> None:
    da, build_registry = _load()
    path = tmp_path / "mod.py"
    groups_by_path = {path: {"f": [set(["user_id", "user_name"])]}}
    annotations_by_path = {
        path: {"f": {"user_id": "int", "user_name": "str"}}
    }
    registry, index = build_registry({"user_context": ["int", "str"]})
    matches = da._compute_fingerprint_matches(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        index=index,
    )
    assert any("user_context" in match for match in matches)


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences
def test_fingerprint_synth_reports_tail(tmp_path: Path) -> None:
    da, build_registry = _load()
    path = tmp_path / "mod.py"
    groups_by_path = {path: {"f": [set(["user_id", "user_name"]), set(["user_id", "user_name"])]}}
    annotations_by_path = {
        path: {"f": {"user_id": "int", "user_name": "str"}}
    }
    registry, _ = build_registry({"user_context": ["int", "str"]})
    synth, payload = da._compute_fingerprint_synth(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        ctor_registry=None,
        min_occurrences=2,
        version="synth@1",
    )
    assert any("synth@" in line or "synth registry" in line for line in synth)
    assert payload is not None
    assert payload.get("entries")


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::root
def test_fingerprint_provenance_emits_entries(tmp_path: Path) -> None:
    da, build_registry = _load()
    path = tmp_path / "mod.py"
    groups_by_path = {path: {"f": [set(["user_id", "user_name"])]}}
    annotations_by_path = {
        path: {"f": {"user_id": "int", "user_name": "str"}}
    }
    registry, index = build_registry({"user_context": ["int", "str"]})
    provenance = da._compute_fingerprint_provenance(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        index=index,
        ctor_registry=None,
    )
    assert provenance
    entry = provenance[0]
    assert entry["base_keys"] == ["int", "str"]
    assert entry["glossary_matches"] == ["user_context"]


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index
def test_fingerprint_warnings_index_empty(tmp_path: Path) -> None:
    da, _ = _load()
    path = tmp_path / "mod.py"
    groups_by_path = {path: {"f": [set(["user_id"])]}}
    annotations_by_path = {path: {"f": {"user_id": "int"}}}
    warnings = da._compute_fingerprint_warnings(
        groups_by_path,
        annotations_by_path,
        registry=da.PrimeRegistry(),
        index={},
    )
    assert warnings == []


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index
def test_fingerprint_warnings_handles_missing_and_none_annotations(tmp_path: Path) -> None:
    da, build_registry = _load()
    path = tmp_path / "mod.py"
    groups_by_path = {path: {"f": [set(["user_id", "user_name"])]}}

    class _Annots(dict):
        def get(self, key: str, default: object | None = None) -> str:
            return "present"

        def __getitem__(self, key: str) -> object:
            if key == "user_name":
                return None
            return "int"

    annotations_by_path = {path: {"f": _Annots()}}
    registry, index = build_registry({"known": ["int"]})
    warnings = da._compute_fingerprint_warnings(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        index=index,
    )
    assert warnings == []


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index
def test_fingerprint_warnings_remainder_with_ctor_keys(tmp_path: Path) -> None:
    da, _ = _load()

    class _RemainderRegistry(da.PrimeRegistry):
        def get_or_assign(self, key: str) -> int:
            prime = super().get_or_assign(key)
            if key == "int":
                self.primes.pop(key, None)
            return prime

    registry = _RemainderRegistry()
    ctor_registry = da.TypeConstructorRegistry(registry)
    path = tmp_path / "mod.py"
    groups_by_path = {path: {"f": [set(["user_id"])]}}
    annotations_by_path = {path: {"f": {"user_id": "list[int]"}}}
    index = {da.Fingerprint(base=da.FingerprintDimension(product=3, mask=0)): {"known"}}
    warnings = da._compute_fingerprint_warnings(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        index=index,
        ctor_registry=ctor_registry,
    )
    assert any("remainder" in warning for warning in warnings)


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_warnings::index
def test_fingerprint_warnings_soundness_detected(tmp_path: Path) -> None:
    da, _ = _load()
    registry = da.PrimeRegistry()
    registry.primes["int"] = 2
    registry.bit_positions["int"] = 0
    registry.primes["ctor:list"] = 2
    registry.bit_positions["ctor:list"] = 1
    ctor_registry = da.TypeConstructorRegistry(registry)
    path = tmp_path / "mod.py"
    groups_by_path = {path: {"f": [set(["user_id"])]}}
    annotations_by_path = {path: {"f": {"user_id": "list[int]"}}}
    index = {da.Fingerprint(base=da.FingerprintDimension(product=3, mask=0)): {"known"}}
    warnings = da._compute_fingerprint_warnings(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        index=index,
        ctor_registry=ctor_registry,
    )
    assert any("carrier soundness" in warning for warning in warnings)


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.format_fingerprint::fingerprint E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index
def test_fingerprint_matches_cover_remainder_and_missing(tmp_path: Path) -> None:
    da, _ = _load()

    class _RemainderRegistry(da.PrimeRegistry):
        def __init__(self):
            super().__init__()
            self._shadow: dict[str, int] = {}

        def get_or_assign(self, key: str) -> int:
            if key in self._shadow:
                return self._shadow[key]
            prime = super().get_or_assign(key)
            if key == "int":
                self.primes.pop(key, None)
            self._shadow[key] = prime
            return prime

    registry = _RemainderRegistry()
    ctor_registry = da.TypeConstructorRegistry(registry)
    path = tmp_path / "mod.py"
    groups_by_path = {path: {"f": [set(["user_id"])]}}
    annotations_by_path = {path: {"f": {"user_id": "list[int]"}}}
    fingerprint = da.bundle_fingerprint_dimensional(
        ["list[int]"], registry, ctor_registry
    )
    index = {fingerprint: {"ctx"}}
    matches = da._compute_fingerprint_matches(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        index=index,
        ctor_registry=ctor_registry,
    )
    assert any("ctx" in match for match in matches)
    assert any("remainder" in match for match in matches)


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.format_fingerprint::fingerprint E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_matches::index
def test_fingerprint_matches_index_empty(tmp_path: Path) -> None:
    da, _ = _load()
    path = tmp_path / "mod.py"
    groups_by_path = {path: {"f": [set(["user_id"])]}}
    annotations_by_path = {path: {"f": {"user_id": "int"}}}
    matches = da._compute_fingerprint_matches(
        groups_by_path,
        annotations_by_path,
        registry=da.PrimeRegistry(),
        index={},
    )
    assert matches == []


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_provenance::index E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::root
def test_fingerprint_provenance_skips_none_annotations(tmp_path: Path) -> None:
    da, _ = _load()
    path = tmp_path / "mod.py"
    groups_by_path = {path: {"f": [set(["user_id"])]}}
    annotations_by_path = {path: {"f": {"user_id": None}}}
    provenance = da._compute_fingerprint_provenance(
        groups_by_path,
        annotations_by_path,
        registry=da.PrimeRegistry(),
        index=None,
        ctor_registry=None,
    )
    assert provenance == []


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._summarize_fingerprint_provenance::entries,max_examples
def test_summarize_fingerprint_provenance_groups() -> None:
    da, _ = _load()
    entries = [
        {"path": "a.py", "function": "f", "bundle": ["a"], "base_keys": ["int"], "ctor_keys": [], "glossary_matches": []},
        {"path": "b.py", "function": "g", "bundle": ["b"], "base_keys": ["int"], "ctor_keys": [], "glossary_matches": []},
        {"path": "c.py", "function": "h", "bundle": ["c"], "base_keys": ["int"], "ctor_keys": [], "glossary_matches": []},
    ]
    lines = da._summarize_fingerprint_provenance(entries, max_groups=1, max_examples=1)
    assert any("occurrences=3" in line for line in lines)
    assert any("... (2 more)" in line for line in lines)


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences
def test_fingerprint_synth_edges(tmp_path: Path) -> None:
    da, _ = _load()
    path = tmp_path / "mod.py"
    groups_by_path = {path: {"f": [set(["user_id"])]}}
    annotations_by_path = {path: {"f": {"user_id": "int"}}}
    registry = da.PrimeRegistry()
    synth, payload = da._compute_fingerprint_synth(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        ctor_registry=None,
        min_occurrences=1,
        version="synth@1",
    )
    assert synth == []
    assert payload is None


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences
def test_fingerprint_synth_existing_registry_remainder(tmp_path: Path) -> None:
    da, _ = _load()
    registry = da.PrimeRegistry()
    fingerprint = da.Fingerprint(
        base=da.FingerprintDimension(product=registry.get_or_assign("int") * 97, mask=0),
        ctor=da.FingerprintDimension(),
    )
    synth_registry = da.SynthRegistry(registry=registry)
    synth_registry.tails[101] = fingerprint
    synth_registry.primes[fingerprint] = 101
    lines, payload = da._compute_fingerprint_synth(
        {},
        {},
        registry=registry,
        ctor_registry=None,
        min_occurrences=2,
        version="synth@1",
        existing=synth_registry,
    )
    assert any("remainder=" in line for line in lines)
    assert payload is not None


# gabion:evidence E:decision_surface/direct::type_fingerprints.py::gabion.analysis.type_fingerprints.bundle_fingerprint_dimensional::ctor_registry E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._compute_fingerprint_synth::existing,min_occurrences
def test_fingerprint_synth_empty_registry_returns_none(tmp_path: Path) -> None:
    da, _ = _load()
    path = tmp_path / "mod.py"
    groups_by_path = {path: {"f": [set(["user_id"])]}}
    annotations_by_path = {path: {"f": {"user_id": "int"}}}
    registry = da.PrimeRegistry()
    synth, payload = da._compute_fingerprint_synth(
        groups_by_path,
        annotations_by_path,
        registry=registry,
        ctor_registry=None,
        min_occurrences=3,
        version="synth@1",
    )
    assert synth == []
    assert payload is None


def test_build_synth_registry_payload_includes_entries() -> None:
    da, _ = _load()
    registry = da.PrimeRegistry()
    fingerprint = da.Fingerprint(
        base=da.FingerprintDimension(product=registry.get_or_assign("int"), mask=0),
        ctor=da.FingerprintDimension(),
    )
    synth_registry = da.SynthRegistry(registry=registry)
    synth_registry.tails[101] = fingerprint
    synth_registry.primes[fingerprint] = 101
    payload = da._build_synth_registry_payload(
        synth_registry, registry, min_occurrences=2
    )
    assert payload["entries"]
