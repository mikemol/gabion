from __future__ import annotations

import random

from gabion.analysis.projection_exec import apply_spec
from gabion.analysis.projection_normalize import (
    _normalize_predicates,
    normalize_spec,
    spec_canonical_json,
    spec_hash,
)
from gabion.analysis.projection_spec import (
    ProjectionOp,
    ProjectionSpec,
    spec_from_dict,
    spec_to_dict,
)


# gabion:evidence E:decision_surface/direct::projection_normalize.py::gabion.analysis.projection_normalize.normalize_spec::spec E:decision_surface/direct::projection_normalize.py::gabion.analysis.projection_normalize._normalize_value::value
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


def test_spec_hash_accepts_string_and_mapping() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="demo",
        domain="tests",
        pipeline=(),
    )
    payload = spec_to_dict(spec)
    assert spec_hash("explicit-id") == "explicit-id"
    assert spec_hash(payload) == spec_hash(spec)


def test_spec_canonical_json_is_byte_stable_for_shuffled_params() -> None:
    baseline = None
    entries = [("z", 1), ("a", 2), ("m", 3)]
    for seed in range(20):
        rng = random.Random(seed)
        shuffled = list(entries)
        rng.shuffle(shuffled)
        params = dict(shuffled)
        spec = ProjectionSpec(
            spec_version=1,
            name="stable-json",
            domain="tests",
            pipeline=(
                ProjectionOp("select", {"predicates": ["beta", "alpha"]}),
                ProjectionOp("count_by", {"fields": ["right", "left"]}),
            ),
            params=params,
        )
        encoded = spec_canonical_json(spec)
        if baseline is None:
            baseline = encoded
            continue
        assert encoded == baseline


# gabion:evidence E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec.apply_spec::params_override E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec._sort_value::value
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


# gabion:evidence E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec.apply_spec::params_override E:decision_surface/direct::projection_normalize.py::gabion.analysis.projection_normalize.normalize_spec::spec E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec._sort_value::value E:decision_surface/direct::projection_normalize.py::gabion.analysis.projection_normalize._normalize_value::value
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


# gabion:evidence E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec.apply_spec::params_override E:decision_surface/direct::projection_normalize.py::gabion.analysis.projection_normalize.normalize_spec::spec E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec._sort_value::value E:decision_surface/direct::projection_normalize.py::gabion.analysis.projection_normalize._normalize_value::value
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


# gabion:evidence E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec.apply_spec::params_override E:decision_surface/direct::projection_normalize.py::gabion.analysis.projection_normalize.normalize_spec::spec E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec._sort_value::value E:decision_surface/direct::projection_normalize.py::gabion.analysis.projection_normalize._normalize_value::value
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


# gabion:evidence E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec.apply_spec::params_override E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec._sort_value::value
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


def test_spec_from_dict_ignores_non_list_pipeline_payload() -> None:
    spec = spec_from_dict(
        {
            "spec_version": 1,
            "name": "demo",
            "domain": "tests",
            "pipeline": {"op": "select"},
        }
    )
    assert spec.pipeline == ()


def test_normalize_spec_handles_empty_unknown_and_mixed_param_shapes() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="mixed",
        domain="tests",
        pipeline=(
            ProjectionOp("select", {"predicates": ["ok", "", 1]}),
            ProjectionOp("project", {"fields": ["x", "", 1]}),
            ProjectionOp("sort", {"by": ""}),
            ProjectionOp("sort", {"by": {"field": "x"}}),
            ProjectionOp("sort", {"by": ["", 1, {"field": "x", "order": "UP"}]}),
            ProjectionOp("   ", {"ignored": True}),
        ),
    )
    normalized = normalize_spec(spec)
    assert normalized["pipeline"] == [
        {"op": "select", "params": {"predicates": ["ok"]}},
        {"op": "project", "params": {"fields": ["x"]}},
        {"op": "sort", "params": {"by": [{"field": "x", "order": "asc"}]}},
    ]


def test_normalize_spec_drops_project_with_non_string_non_list_fields() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="drop-project",
        domain="tests",
        pipeline=(ProjectionOp("project", {"fields": {"field": "x"}}),),
    )
    normalized = normalize_spec(spec)
    assert normalized["pipeline"] == []


def test_normalize_select_predicates_stable_under_permuted_discovery_order() -> None:
    spec_a = ProjectionSpec(
        spec_version=1,
        name="predicates",
        domain="tests",
        pipeline=(
            ProjectionOp("select", {"predicates": ["beta", "alpha", "beta"]}),
        ),
    )
    spec_b = ProjectionSpec(
        spec_version=1,
        name="predicates",
        domain="tests",
        pipeline=(
            ProjectionOp("select", {"predicates": ["alpha", "beta"]}),
        ),
    )

    assert normalize_spec(spec_a)["pipeline"] == normalize_spec(spec_b)["pipeline"]


def test_count_by_output_stable_under_permuted_discovery_order() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="count-stable",
        domain="tests",
        pipeline=(ProjectionOp("count_by", {"fields": ["class"]}),),
    )
    rows_a = [
        {"class": "b", "value": 1},
        {"class": "a", "value": 2},
        {"class": "b", "value": 3},
        {"class": "a", "value": 4},
    ]
    rows_b = [
        {"class": "a", "value": 2},
        {"class": "b", "value": 1},
        {"class": "a", "value": 4},
        {"class": "b", "value": 3},
    ]

    assert apply_spec(spec, rows_a) == apply_spec(spec, rows_b) == [
        {"class": "a", "count": 2},
        {"class": "b", "count": 2},
    ]


def test_apply_spec_params_override_replaces_normalized_params() -> None:
    rows = [{"value": 1}, {"value": 3}, {"value": 5}]

    def above_threshold(row, params):
        return int(row.get("value", 0)) >= int(params["threshold"])

    spec = ProjectionSpec(
        spec_version=1,
        name="params-override",
        domain="tests",
        pipeline=(ProjectionOp("select", {"predicate": "above_threshold"}),),
        params={"threshold": 4},
    )
    result = apply_spec(
        spec,
        rows,
        op_registry={"above_threshold": above_threshold},
        params_override={"threshold": 3},
    )
    assert result == [{"value": 3}, {"value": 5}]


def test_normalize_select_predicates_drops_whitespace_entries() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="predicates-whitespace",
        domain="tests",
        pipeline=(ProjectionOp("select", {"predicates": ["  ", "alpha"]}),),
    )
    assert normalize_spec(spec)["pipeline"] == [
        {"op": "select", "params": {"predicates": ["alpha"]}}
    ]


def test_normalize_predicates_skips_whitespace_only_values() -> None:
    assert _normalize_predicates(["\t", "beta"]) == ["beta"]
