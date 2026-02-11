from __future__ import annotations

from gabion.analysis.forest_spec import (
    build_forest_spec,
    forest_spec_to_dict,
    forest_spec_hash,
    forest_spec_from_dict,
    forest_spec_metadata,
    normalize_forest_spec,
    _normalize_decision_tiers,
)


# gabion:evidence E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.normalize_forest_spec::spec E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec._normalize_decision_tiers::tiers E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec._normalize_value::value E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec._sorted_strings::values
def test_forest_spec_normalization_idempotent() -> None:
    spec = build_forest_spec(
        include_bundle_forest=True,
        include_decision_surfaces=True,
        include_value_decision_surfaces=True,
        include_never_invariants=True,
        ignore_params={"x", "y"},
        decision_ignore_params={"z"},
        transparent_decorators={"decor"},
        strictness="low",
        decision_tiers={"flag": 2, "mode": 3},
        require_tiers=True,
        external_filter=False,
    )
    normalized = normalize_forest_spec(spec)
    roundtrip = forest_spec_from_dict(normalized)
    normalized_again = normalize_forest_spec(roundtrip)
    assert normalized_again == normalized
    assert forest_spec_hash(spec) == forest_spec_hash(roundtrip)


# gabion:evidence E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.normalize_forest_spec::spec E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec._normalize_decision_tiers::tiers E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec._sorted_strings::values
def test_forest_spec_metadata_contains_id_and_spec() -> None:
    spec = build_forest_spec(
        include_bundle_forest=True,
        include_decision_surfaces=False,
        include_value_decision_surfaces=False,
        include_never_invariants=False,
    )
    metadata = forest_spec_metadata(spec)
    assert "generated_by_forest_spec_id" in metadata
    assert "generated_by_forest_spec" in metadata


# gabion:evidence E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec.build_forest_spec::include_bundle_forest,include_decision_surfaces,include_never_invariants,include_value_decision_surfaces E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec._normalize_decision_tiers::tiers E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec._sorted_strings::values
def test_forest_spec_to_dict_roundtrip_handles_invalid_payload() -> None:
    spec = build_forest_spec(
        include_bundle_forest=True,
        include_decision_surfaces=False,
        include_value_decision_surfaces=False,
        include_never_invariants=False,
    )
    payload = forest_spec_to_dict(spec)
    payload["spec_version"] = "bad"
    payload["collectors"] = [
        "not-a-mapping",
        {"name": "", "outputs": ["x"]},
        {"name": "collector", "outputs": "bad", "params": "bad"},
    ]
    roundtrip = forest_spec_from_dict(payload)
    assert roundtrip.spec_version == 1
    assert roundtrip.collectors[0].name == "collector"


def test_forest_spec_includes_deadline_obligations() -> None:
    spec = build_forest_spec(
        include_bundle_forest=False,
        include_decision_surfaces=False,
        include_value_decision_surfaces=False,
        include_never_invariants=False,
        include_ambiguities=False,
        include_deadline_obligations=True,
    )
    collector_names = {collector.name for collector in spec.collectors}
    assert "deadline_obligations" in collector_names
    assert "DeadlineObligation" in spec.declared_outputs


def test_forest_spec_hash_accepts_string_and_mapping() -> None:
    spec = build_forest_spec(
        include_bundle_forest=True,
        include_decision_surfaces=False,
        include_value_decision_surfaces=False,
        include_never_invariants=False,
    )
    payload = forest_spec_to_dict(spec)
    assert forest_spec_hash("explicit-id") == "explicit-id"
    assert forest_spec_hash(payload) == forest_spec_hash(spec)


# gabion:evidence E:decision_surface/direct::forest_spec.py::gabion.analysis.forest_spec._normalize_decision_tiers::tiers
def test_normalize_decision_tiers_ignores_invalid() -> None:
    tiers = {"": 1, "ok": "bad", "fine": 2}
    assert _normalize_decision_tiers(tiers) == {"fine": 2}
