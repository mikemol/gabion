from __future__ import annotations

from gabion.analysis.projection.projection_exec import (
    _hashable,
    _normalize_execution_projection_op,
    _sort_value,
    apply_spec,
)
from gabion.analysis.projection.projection_normalize import (
    _extract_predicates,
    _normalize_fields,
    _normalize_group_fields,
    _normalize_limit,
    _normalize_pipeline,
    _normalize_sort_by,
    _normalize_value,
)
from gabion.analysis.projection.projection_spec import ProjectionOp, ProjectionSpec, spec_from_dict


# gabion:evidence E:function_site::projection_spec.py::gabion.analysis.projection_spec.spec_from_dict
# gabion:behavior primary=verboten facets=edge,invalid
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
# gabion:behavior primary=verboten facets=edge,empty
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
# gabion:behavior primary=verboten facets=edge
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
# gabion:behavior primary=verboten facets=edge
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
# gabion:behavior primary=verboten facets=edge,invalid
def test_apply_spec_handles_invalid_and_semantic_only_ops_at_exec_ingress() -> None:
    rows = [
        {"group": ["a"], "value": 1},
        {"group": ["a"], "value": 2},
    ]

    def keep(_row, params):
        return int(params.get("threshold", 0)) <= 2

    spec = ProjectionSpec(
        spec_version=1,
        name="demo",
        domain="tests",
        params={"threshold": 2},
        pipeline=(
            ProjectionOp("select", {"predicates": "keep"}),
            ProjectionOp("select", {"predicates": {"bad": True}}),
            ProjectionOp("select", {"predicates": [1, "missing", "keep"]}),
            ProjectionOp("project", {"fields": 1}),
            ProjectionOp("project", {"fields": "group"}),
            ProjectionOp("count_by", {"fields": ["group", 5]}),
            ProjectionOp("sort", {"by": {"field": "count", "order": 1}}),
            ProjectionOp("sort", {"by": "bad"}),
            ProjectionOp("limit", {"count": "bad"}),
            ProjectionOp("reflect", {"surface": "projection_fiber"}),
            ProjectionOp("custom", {"bad": True}),
        ),
    )

    result = apply_spec(
        spec,
        rows,
        op_registry={"keep": keep},
    )
    assert result == [{"group": ["a"], "count": 2}]


# gabion:evidence E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec._sort_value::value E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec._sort_value::stale_f5d27306e19e
# gabion:behavior primary=verboten facets=edge
def test_sort_value_and_hashable_helpers() -> None:
    assert _sort_value(None) == (1, "")
    assert _sort_value(3) == (0, 3)
    assert _sort_value({"a": 1})[0] == 0
    assert _hashable({"a": 1}) == "{\"a\":1}"
    assert _hashable(2) == 2


# gabion:evidence E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec.apply_spec::params_override E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec._sort_value::value E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec._sort_value::stale_bab431281bf2
# gabion:behavior primary=verboten facets=edge
def test_apply_spec_count_by_and_sort_edges() -> None:
    rows = [{"group": "a"}, {"group": "b"}]
    spec = ProjectionSpec(
        spec_version=1,
        name="demo",
        domain="tests",
        pipeline=(
            ProjectionOp("count_by", {"fields": "group"}),
            ProjectionOp("count_by", {"fields": []}),
            ProjectionOp("sort", {"by": ["bad", {"field": 1}]}),
        ),
    )
    result = apply_spec(spec, rows)
    assert {row["count"] for row in result} == {1}


# gabion:evidence E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec.apply_spec::params_override E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec.apply_spec::stale_e87e1ec18193
# gabion:behavior primary=verboten facets=edge
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
# gabion:behavior primary=verboten facets=edge
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
# gabion:behavior primary=verboten facets=edge,invalid
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
# gabion:behavior primary=verboten facets=edge,invalid
def test_apply_spec_traverse_skips_when_field_invalid() -> None:
    rows = [{"items": [1, 2, 3]}]
    spec = ProjectionSpec(
        spec_version=1,
        name="traverse",
        domain="tests",
        pipeline=(ProjectionOp("traverse", {"field": 123}),),
    )
    assert apply_spec(spec, rows) == rows


def test_normalized_execution_projection_op_erases_semantic_metadata_and_semantic_only_ops() -> None:
    project_op = _normalize_execution_projection_op(
        index=0,
        op=ProjectionOp(
            "project",
            {
                "fields": ["id"],
                "quotient_face": "projection_fiber.frontier",
            },
        ),
    )
    reflect_op = _normalize_execution_projection_op(
        index=1,
        op=ProjectionOp("reflect", {"surface": "projection_fiber"}),
    )
    custom_op = _normalize_execution_projection_op(
        index=2,
        op=ProjectionOp("custom", {"a": 1}),
    )

    assert project_op.op_name == "project"
    assert project_op.params == {"fields": ["id"]}
    assert reflect_op.op_name == ""
    assert reflect_op.params == {}
    assert custom_op.op_name == ""
    assert custom_op.params == {}


def test_apply_spec_erases_semantic_projection_compatibility_at_exec_ingress() -> None:
    spec = ProjectionSpec(
        spec_version=1,
        name="semantic-compat",
        domain="tests",
        pipeline=(
            ProjectionOp(
                "project",
                {
                    "fields": ["id"],
                    "quotient_face": "projection_fiber.frontier",
                },
            ),
            ProjectionOp(
                "reflect",
                {"surface": "projection_fiber"},
            ),
        ),
    )

    rows = [{"id": 1, "status": "ok"}]
    assert apply_spec(spec, rows) == [{"id": 1}]
