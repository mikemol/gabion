from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AmbiguityProjectionInput:
    forest: object


@dataclass(frozen=True)
class AmbiguityProjectionOutput:
    suite_order_materialized: bool
    suite_aggregate_materialized: bool
    virtual_set_materialized: bool


def materialize_ambiguity_projection(
    *,
    data: AmbiguityProjectionInput,
    suite_order_runner,
    suite_agg_runner,
    virtual_set_runner,
) -> AmbiguityProjectionOutput:
    suite_order_runner(forest=data.forest)
    suite_agg_runner(forest=data.forest)
    virtual_set_runner(forest=data.forest)
    return AmbiguityProjectionOutput(
        suite_order_materialized=True,
        suite_aggregate_materialized=True,
        virtual_set_materialized=True,
    )
