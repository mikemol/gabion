from __future__ import annotations

from gabion.analysis.projection_exec import apply_spec, _hashable, _sort_value
from gabion.analysis.projection_normalize import (
    _extract_predicates,
    _normalize_fields,
    _normalize_group_fields,
    _normalize_limit,
    _normalize_pipeline,
    _normalize_sort_by,
    _normalize_value,
)
from gabion.analysis.projection_spec import ProjectionOp, ProjectionSpec, spec_from_dict


# gabion:evidence E:function_site::projection_spec.py::gabion.analysis.projection_spec.spec_from_dict
def test_spec_from_dict_handles_invalid_entries() -> None:
    payload = {
        "spec_version": "bad",
        "name": 123,
        "domain": None,
        "params": "not-a-mapping",
        "pipeline": [
            "not-a-mapping",
            {"op": "", "params": {}},
            {"op": "select", "params": "bad"},
        ],
    }
    spec = spec_from_dict(payload)
    assert spec.spec_version == 1
    assert spec.name == "123"
    assert spec.domain == ""
    assert spec.params == {}
    assert spec.pipeline == (ProjectionOp(op="select", params={}),)


# gabion:evidence E:decision_surface/direct::projection_normalize.py::gabion.analysis.projection_normalize._normalize_fields::value E:decision_surface/direct::projection_normalize.py::gabion.analysis.projection_normalize._normalize_limit::value E:decision_surface/direct::projection_normalize.py::gabion.analysis.projection_normalize._normalize_sort_by::value E:decision_surface/direct::projection_normalize.py::gabion.analysis.projection_normalize._normalize_value::value E:decision_surface/direct::projection_normalize.py::gabion.analysis.projection_normalize._normalize_fields::stale_53512762626b
def test_normalize_pipeline_skips_empty_and_unknown_ops() -> None:
    pipeline = (
        ProjectionOp("select", {"predicates": [" "]}),
        ProjectionOp("project", {"fields": []}),
        ProjectionOp("count_by", {"field": ""}),
        ProjectionOp("sort", {"by": []}),
        ProjectionOp("limit", {"count": -1}),
        ProjectionOp("custom", {"b": 2, "a": 1}),
    )
    normalized = _normalize_pipeline(pipeline)
    assert normalized == [
        {"op": "custom", "params": {"a": 1, "b": 2}},
    ]


# gabion:evidence E:decision_surface/direct::projection_normalize.py::gabion.analysis.projection_normalize._normalize_fields::value E:decision_surface/direct::projection_normalize.py::gabion.analysis.projection_normalize._normalize_limit::value E:decision_surface/direct::projection_normalize.py::gabion.analysis.projection_normalize._normalize_sort_by::value E:decision_surface/direct::projection_normalize.py::gabion.analysis.projection_normalize._normalize_value::value E:decision_surface/direct::projection_normalize.py::gabion.analysis.projection_normalize._normalize_fields::stale_2cb8e760719e_6a522c1d
def test_normalize_helpers_cover_branches() -> None:
    preds = _extract_predicates({"predicate": "one", "predicates": ["two", " "]})
    assert preds == ["one", "two", ""]
    assert _extract_predicates({"predicates": "three"}) == ["three"]

    assert _normalize_fields("solo") == ["solo"]
    assert _normalize_fields(["a", "a", "b"]) == ["a", "b"]
    assert _normalize_group_fields(["b", "a"]) == ["a", "b"]

    assert _normalize_sort_by(None) == []
    assert _normalize_sort_by("value") == [{"field": "value", "order": "asc"}]
    assert _normalize_sort_by(
        [
            {"key": "count", "order": 123},
            {"name": "score", "order": "DESC"},
            {"field": "other", "order": "up"},
            {"field": ""},
        ]
    ) == [
        {"field": "count", "order": "asc"},
        {"field": "score", "order": "desc"},
        {"field": "other", "order": "asc"},
    ]

    assert _normalize_limit(None) is None
    assert _normalize_limit("bad") is None
    assert _normalize_limit(-1) is None
    assert _normalize_limit(3) == 3

    assert _normalize_value({"b": 2, "a": 1}) == {"a": 1, "b": 2}
    assert _normalize_value([{"b": 2, "a": 1}]) == [{"a": 1, "b": 2}]


# gabion:evidence E:call_footprint::tests/test_projection_exec_edges.py::test_normalize_pipeline_stable_under_shuffled_upstream_order::projection_normalize.py::gabion.analysis.projection_normalize._normalize_pipeline
def test_normalize_pipeline_stable_under_shuffled_upstream_order() -> None:
    pipeline_a = (
        ProjectionOp("select", {"predicates": ["beta", "alpha", "beta"]}),
        ProjectionOp("count_by", {"fields": ["group_b", "group_a", "group_b"]}),
    )
    pipeline_b = (
        ProjectionOp("select", {"predicates": ["alpha", "beta"]}),
        ProjectionOp("count_by", {"fields": ["group_a", "group_b"]}),
    )

    assert _normalize_pipeline(pipeline_a) == _normalize_pipeline(pipeline_b)


# gabion:evidence E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec.apply_spec::params_override E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec._sort_value::value E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec._sort_value::stale_1581b2052cbd
def test_apply_spec_with_custom_normalizer_handles_invalid_ops() -> None:
    rows = [
        {"group": ["a"], "value": 1},
        {"group": ["a"], "value": 2},
    ]

    def keep(_row, params):
        return int(params.get("threshold", 0)) <= 2

    def normalize(_spec):
        return {
            "params": {"threshold": 2},
            "pipeline": [
                "not-a-mapping",
                {"op": "select", "params": {"predicates": "keep"}},
                {"op": "select", "params": {"predicates": {"bad": True}}},
                {"op": "select", "params": {"predicates": [1, "missing", "keep"]}},
                {"op": "project", "params": {"fields": 1}},
                {"op": "project", "params": {"fields": "group"}},
                {"op": "count_by", "params": {"fields": ["group", 5]}},
                {"op": "sort", "params": {"by": {"field": "count", "order": 1}}},
                {"op": "sort", "params": {"by": "bad"}},
                {"op": "limit", "params": {"count": "bad"}},
                {"op": "custom", "params": "bad"},
            ],
        }

    result = apply_spec(
        ProjectionSpec(spec_version=1, name="demo", domain="tests"),
        rows,
        op_registry={"keep": keep},
        normalize=normalize,
    )
    assert result == [{"group": ["a"], "count": 2}]


# gabion:evidence E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec._sort_value::value E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec._sort_value::stale_f5d27306e19e
def test_sort_value_and_hashable_helpers() -> None:
    assert _sort_value(None) == (1, "")
    assert _sort_value(3) == (0, 3)
    assert _sort_value({"a": 1})[0] == 0
    assert _hashable({"a": 1}) == "{\"a\":1}"
    assert _hashable(2) == 2


# gabion:evidence E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec.apply_spec::params_override E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec._sort_value::value E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec._sort_value::stale_bab431281bf2
def test_apply_spec_count_by_and_sort_edges() -> None:
    rows = [{"group": "a"}, {"group": "b"}]

    def normalize(_spec):
        return {
            "pipeline": [
                {"op": "count_by", "params": {"fields": "group"}},
                {"op": "count_by", "params": {"fields": []}},
                {"op": "sort", "params": {"by": ["bad", {"field": 1}]}},
            ]
        }

    result = apply_spec(
        ProjectionSpec(spec_version=1, name="demo", domain="tests"),
        rows,
        normalize=normalize,
    )
    assert {row["count"] for row in result} == {1}


# gabion:evidence E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec.apply_spec::params_override E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec.apply_spec::stale_e87e1ec18193
def test_apply_spec_traverse_flattens_sequences() -> None:
    rows = [
        {
            "suite": "s1",
            "candidates": [{"qual": "a"}, {"qual": "b"}],
        }
    ]
    spec = ProjectionSpec(
        spec_version=1,
        name="traverse",
        domain="tests",
        pipeline=(
            ProjectionOp(
                "traverse",
                {
                    "field": "candidates",
                    "merge": True,
                    "prefix": "candidate_",
                    "index": "candidate_index",
                },
            ),
        ),
    )
    result = apply_spec(spec, rows)
    assert result == [
        {"suite": "s1", "candidate_index": 0, "candidate_qual": "a"},
        {"suite": "s1", "candidate_index": 1, "candidate_qual": "b"},
    ]


# gabion:evidence E:call_footprint::tests/test_projection_exec_edges.py::test_apply_spec_traverse_as_field_and_keep::projection_exec.py::gabion.analysis.projection_exec.apply_spec
def test_apply_spec_traverse_as_field_and_keep() -> None:
    rows = [
        {
            "suite": "s2",
            "candidates": ["x", "y"],
        }
    ]
    spec = ProjectionSpec(
        spec_version=1,
        name="traverse",
        domain="tests",
        pipeline=(
            ProjectionOp(
                "traverse",
                {
                    "field": "candidates",
                    "merge": False,
                    "as": "candidate",
                    "keep": True,
                },
            ),
        ),
    )
    result = apply_spec(spec, rows)
    assert result == [
        {"suite": "s2", "candidates": ["x", "y"], "candidate": "x"},
        {"suite": "s2", "candidates": ["x", "y"], "candidate": "y"},
    ]


# gabion:evidence E:call_footprint::tests/test_projection_exec_edges.py::test_apply_spec_traverse_handles_invalid_params::projection_exec.py::gabion.analysis.projection_exec.apply_spec
def test_apply_spec_traverse_handles_invalid_params() -> None:
    rows = [
        {"items": [{"a": 1}, {1: "b"}], "other": 3},
        {"items": "not-list"},
    ]
    spec = ProjectionSpec(
        spec_version=1,
        name="traverse",
        domain="tests",
        pipeline=(
            ProjectionOp(
                "traverse",
                {
                    "field": "items",
                    "merge": "yes",
                    "keep": "no",
                    "prefix": 123,
                    "as": 456,
                    "index": 789,
                },
            ),
        ),
    )
    result = apply_spec(spec, rows)
    assert result == [
        {"other": 3, "a": 1},
        {"other": 3, "1": "b"},
    ]


# gabion:evidence E:call_footprint::tests/test_projection_exec_edges.py::test_apply_spec_traverse_skips_when_field_invalid::projection_exec.py::gabion.analysis.projection_exec.apply_spec
def test_apply_spec_traverse_skips_when_field_invalid() -> None:
    rows = [{"items": [1, 2, 3]}]
    spec = ProjectionSpec(
        spec_version=1,
        name="traverse",
        domain="tests",
        pipeline=(ProjectionOp("traverse", {"field": 123}),),
    )
    assert apply_spec(spec, rows) == rows
