from __future__ import annotations

from collections.abc import Callable, Mapping

from gabion.analysis.foundation.json_types import JSONValue


def materialize_suite_order_spec(*, forest, suite_order_relation_runner, row_to_site_runner, projection_spec, projection_apply_runner, materialize_rows_runner) -> None:
    relation, suite_index = suite_order_relation_runner(forest)
    if not relation:
        return
    projected = projection_apply_runner(projection_spec, relation)
    materialize_rows_runner(
        spec=projection_spec,
        projected=projected,
        forest=forest,
        row_to_site=lambda row: row_to_site_runner(row, suite_index),
    )


def materialize_ambiguity_suite_agg_spec(*, forest, ambiguity_relation_runner, row_to_suite_runner, projection_spec, projection_apply_runner, materialize_rows_runner) -> None:
    relation = ambiguity_relation_runner(forest)
    if not relation:
        return
    projected = projection_apply_runner(projection_spec, relation)
    materialize_rows_runner(
        spec=projection_spec,
        projected=projected,
        forest=forest,
        row_to_site=lambda row: row_to_suite_runner(row, forest),
    )


def materialize_ambiguity_virtual_set_spec(*, forest, ambiguity_relation_runner, row_to_suite_runner, projection_spec, projection_apply_runner, materialize_rows_runner, count_gt_1_runner) -> None:
    relation = ambiguity_relation_runner(forest)
    if not relation:
        return
    projected = projection_apply_runner(
        projection_spec,
        relation,
        op_registry={"count_gt_1": count_gt_1_runner},
    )
    materialize_rows_runner(
        spec=projection_spec,
        projected=projected,
        forest=forest,
        row_to_site=lambda row: row_to_suite_runner(row, forest),
    )
