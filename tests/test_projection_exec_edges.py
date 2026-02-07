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


def test_sort_value_and_hashable_helpers() -> None:
    assert _sort_value(None) == (1, "")
    assert _sort_value(3) == (0, 3)
    assert _sort_value({"a": 1})[0] == 0
    assert _hashable({"a": 1}) == "{\"a\":1}"
    assert _hashable(2) == 2


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
