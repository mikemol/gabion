from __future__ import annotations

from gabion.analysis.projection_exec import apply_spec
from gabion.analysis.projection_normalize import normalize_spec, spec_hash
from gabion.analysis.projection_spec import (
    ProjectionOp,
    ProjectionSpec,
    spec_from_dict,
    spec_to_dict,
)


def test_normalize_idempotent_and_hash_stable() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="demo",
        domain="tests",
        pipeline=(
            ProjectionOp("select", {"predicate": "is_even"}),
            ProjectionOp("select", {"predicates": ["lt_three", "is_even"]}),
            ProjectionOp("sort", {"by": ["value"]}),
        ),
        params={"max_entries": 10},
    )
    normalized = normalize_spec(spec)
    roundtrip = spec_from_dict(normalized)
    normalized_again = normalize_spec(roundtrip)
    assert normalized_again == normalized
    assert spec_hash(spec) == spec_hash(roundtrip)


def test_select_fusion_equivalence() -> None:
    rows = [{"value": 1}, {"value": 2}, {"value": 3}, {"value": 4}]

    def is_even(row, params):
        return int(row.get("value", 0)) % 2 == 0

    def lt_three(row, params):
        return int(row.get("value", 0)) < 3

    spec = ProjectionSpec(
        spec_version=1,
        name="selects",
        domain="tests",
        pipeline=(
            ProjectionOp("select", {"predicate": "is_even"}),
            ProjectionOp("select", {"predicate": "lt_three"}),
        ),
    )
    fused = ProjectionSpec(
        spec_version=1,
        name="selects",
        domain="tests",
        pipeline=(
            ProjectionOp("select", {"predicates": ["lt_three", "is_even"]}),
        ),
    )
    registry = {"is_even": is_even, "lt_three": lt_three}
    assert apply_spec(spec, rows, op_registry=registry) == apply_spec(
        fused, rows, op_registry=registry
    )
    assert spec_hash(spec) == spec_hash(fused)


def test_sort_canonicalization_equivalence() -> None:
    rows = [{"a": 2, "b": 2}, {"a": 1, "b": 3}, {"a": 1, "b": 2}]
    spec_list = ProjectionSpec(
        spec_version=1,
        name="sort",
        domain="tests",
        pipeline=(ProjectionOp("sort", {"by": ["a", "b"]}),),
    )
    spec_dict = ProjectionSpec(
        spec_version=1,
        name="sort",
        domain="tests",
        pipeline=(
            ProjectionOp(
                "sort",
                {"by": [{"field": "a", "order": "asc"}, {"field": "b"}]},
            ),
        ),
    )
    assert normalize_spec(spec_list) == normalize_spec(spec_dict)
    assert apply_spec(spec_list, rows) == apply_spec(spec_dict, rows)


def test_noop_select_elided() -> None:
    rows = [{"value": 1}, {"value": 2}]
    spec = ProjectionSpec(
        spec_version=1,
        name="noop",
        domain="tests",
        pipeline=(ProjectionOp("select", {"predicates": []}),),
    )
    normalized = normalize_spec(spec)
    assert normalized["pipeline"] == []
    assert apply_spec(spec, rows) == rows


def test_limit_roundtrip_and_desc_normalization() -> None:
    rows = [{"value": 1}, {"value": 2}, {"value": 3}]
    spec = ProjectionSpec(
        spec_version=1,
        name="limit",
        domain="tests",
        pipeline=(
            ProjectionOp("sort", {"by": [{"field": "value", "order": "DESC"}]}),
            ProjectionOp("limit", {"count": 2}),
        ),
    )
    payload = spec_to_dict(spec)
    roundtrip = spec_from_dict(payload)
    normalized = normalize_spec(roundtrip)
    by = normalized["pipeline"][0]["params"]["by"][0]
    assert by["order"] == "desc"
    assert apply_spec(roundtrip, rows) == [{"value": 3}, {"value": 2}]


def test_count_by_groups_rows() -> None:
    rows = [
        {"class": "a", "value": 1},
        {"class": "a", "value": 2},
        {"class": "b", "value": 3},
    ]
    spec = ProjectionSpec(
        spec_version=1,
        name="count",
        domain="tests",
        pipeline=(
            ProjectionOp("count_by", {"fields": ["class"]}),
            ProjectionOp("sort", {"by": ["class"]}),
        ),
    )
    result = apply_spec(spec, rows)
    assert result == [
        {"class": "a", "count": 2},
        {"class": "b", "count": 1},
    ]
