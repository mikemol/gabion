from __future__ import annotations

from gabion.analysis.projection.projection_exec import (
    apply_execution_ops,
)
from gabion.analysis.projection.projection_exec_protocol import (
    CountByExecutionOp,
    LimitExecutionOp,
    ProjectExecutionOp,
    SelectExecutionOp,
    SortExecutionOp,
    SortKey,
    TraverseExecutionOp,
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
# gabion:behavior primary=desired
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
# gabion:behavior primary=allowed_unwanted facets=edge,noop
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
# gabion:behavior primary=desired facets=edge
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


# gabion:evidence E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec.apply_execution_ops::runtime_params E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec._sort_value::value
# gabion:behavior primary=desired
def test_apply_execution_ops_applies_typed_pipeline() -> None:
    rows = [
        {"group": ["a"], "value": 1},
        {"group": ["a"], "value": 2},
        {"group": ["b"], "value": 3},
    ]

    def keep(row, params):
        return int(row.get("value", 0)) <= int(params.get("threshold", 0))

    result = apply_execution_ops(
        (
            SelectExecutionOp(
                source_index=0,
                op_name="select",
                predicates=("keep",),
            ),
            ProjectExecutionOp(
                source_index=1,
                op_name="project",
                fields=("group",),
            ),
            CountByExecutionOp(
                source_index=2,
                op_name="count_by",
                fields=("group",),
            ),
            SortExecutionOp(
                source_index=3,
                op_name="sort",
                keys=(SortKey(field="count", order="desc"),),
            ),
        ),
        rows,
        op_registry={"keep": keep},
        runtime_params={"threshold": 2},
    )
    assert result == [{"group": ["a"], "count": 2}]

# gabion:evidence E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec.apply_execution_ops::runtime_params E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec._sort_value::value
# gabion:behavior primary=desired facets=edge
def test_apply_execution_ops_skips_empty_typed_ops() -> None:
    rows = [{"group": "a"}, {"group": "b"}]
    result = apply_execution_ops(
        (
            CountByExecutionOp(
                source_index=0,
                op_name="count_by",
                fields=(),
            ),
            SortExecutionOp(
                source_index=1,
                op_name="sort",
                keys=(),
            ),
        ),
        rows,
    )
    assert result == rows


# gabion:behavior primary=desired
def test_apply_execution_ops_applies_limit_directly() -> None:
    rows = [{"group": "a"}, {"group": "b"}, {"group": "c"}]
    result = apply_execution_ops(
        (
            LimitExecutionOp(
                source_index=0,
                op_name="limit",
                count=2,
            ),
        ),
        rows,
    )
    assert result == [{"group": "a"}, {"group": "b"}]


# gabion:evidence E:decision_surface/direct::projection_exec.py::gabion.analysis.projection_exec.apply_execution_ops::runtime_params
# gabion:behavior primary=desired facets=edge
def test_apply_execution_ops_traverse_flattens_sequences() -> None:
    rows = [
        {
            "suite": "s1",
            "candidates": [{"qual": "a"}, {"qual": "b"}],
        }
    ]
    result = apply_execution_ops(
        (
            TraverseExecutionOp(
                source_index=0,
                op_name="traverse",
                field="candidates",
                merge=True,
                prefix="candidate_",
                index_field="candidate_index",
            ),
        ),
        rows,
    )
    assert result == [
        {"suite": "s1", "candidate_index": 0, "candidate_qual": "a"},
        {"suite": "s1", "candidate_index": 1, "candidate_qual": "b"},
    ]


# gabion:evidence E:call_footprint::tests/test_projection_exec_edges.py::test_apply_execution_ops_traverse_as_field_and_keep::projection_exec.py::gabion.analysis.projection_exec.apply_execution_ops
# gabion:behavior primary=desired facets=edge
def test_apply_execution_ops_traverse_as_field_and_keep() -> None:
    rows = [
        {
            "suite": "s2",
            "candidates": ["x", "y"],
        }
    ]
    result = apply_execution_ops(
        (
            TraverseExecutionOp(
                source_index=0,
                op_name="traverse",
                field="candidates",
                merge=False,
                keep=True,
                as_field="candidate",
            ),
        ),
        rows,
    )
    assert result == [
        {"suite": "s2", "candidates": ["x", "y"], "candidate": "x"},
        {"suite": "s2", "candidates": ["x", "y"], "candidate": "y"},
    ]


# gabion:evidence E:call_footprint::tests/test_projection_exec_edges.py::test_apply_execution_ops_traverse_stringifies_merged_non_string_keys::projection_exec.py::gabion.analysis.projection_exec.apply_execution_ops
# gabion:behavior primary=desired facets=edge
def test_apply_execution_ops_traverse_stringifies_merged_non_string_keys() -> None:
    rows = [
        {"items": [{"a": 1}, {1: "b"}], "other": 3},
        {"items": "not-list"},
    ]
    result = apply_execution_ops(
        (
            TraverseExecutionOp(
                source_index=0,
                op_name="traverse",
                field="items",
                merge=True,
            ),
        ),
        rows,
    )
    assert result == [
        {"other": 3, "a": 1},
        {"other": 3, "1": "b"},
    ]
