from __future__ import annotations

from gabion.analysis.forest_spec import (
    build_forest_spec,
    forest_spec_hash,
    forest_spec_from_dict,
    forest_spec_metadata,
    normalize_forest_spec,
)


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
