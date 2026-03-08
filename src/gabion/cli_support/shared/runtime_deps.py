from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, TypeAlias, cast

import typer

from gabion.json_types import JSONObject
from gabion.runtime_shape_dispatch import json_mapping_or_none
from gabion.tooling.runtime import ci_watch as tooling_ci_watch

CliRunDataflowRawArgvFn: TypeAlias = Callable[[list[str]], None]
CliRunCheckFn: TypeAlias = Callable[..., JSONObject]
CliRunSppfSyncFn: TypeAlias = Callable[..., int]
CliRunCheckDeltaGatesFn: TypeAlias = Callable[[], int]
CliRunCiWatchFn: TypeAlias = Callable[
    [tooling_ci_watch.StatusWatchOptions],
    tooling_ci_watch.StatusWatchResult,
]


class CliRuntimeDeps(Protocol):
    run_dataflow_raw_argv_fn: CliRunDataflowRawArgvFn
    run_check_fn: CliRunCheckFn
    run_sppf_sync_fn: CliRunSppfSyncFn
    run_check_delta_gates_fn: CliRunCheckDeltaGatesFn
    run_ci_watch_fn: CliRunCiWatchFn


@dataclass(frozen=True)
class CliRuntimeDepsBundle:
    run_dataflow_raw_argv_fn: CliRunDataflowRawArgvFn
    run_check_fn: CliRunCheckFn
    run_sppf_sync_fn: CliRunSppfSyncFn
    run_check_delta_gates_fn: CliRunCheckDeltaGatesFn
    run_ci_watch_fn: CliRunCiWatchFn


# gabion:decision_protocol
def context_callable_dep(
    *,
    ctx: typer.Context,
    key: str,
    default: Callable[..., object],
    never_fn: Callable[..., object],
) -> Callable[..., object]:
    obj_mapping = json_mapping_or_none(ctx.obj)
    if obj_mapping is None:
        return default
    candidate = obj_mapping.get(key)
    if candidate is None:
        return default
    if callable(candidate):
        return candidate
    never_fn(
        "invalid cli dependency override",
        dependency=key,
        value_type=type(candidate).__name__,
    )
    return default  # pragma: no cover - never() raises


def context_cli_runtime_deps(
    *,
    ctx: typer.Context,
    default_run_dataflow_raw_argv_fn: CliRunDataflowRawArgvFn,
    default_run_check_fn: CliRunCheckFn,
    default_run_sppf_sync_fn: CliRunSppfSyncFn,
    default_run_check_delta_gates_fn: CliRunCheckDeltaGatesFn,
    default_run_ci_watch_fn: CliRunCiWatchFn,
    never_fn: Callable[..., object],
) -> CliRuntimeDepsBundle:
    return CliRuntimeDepsBundle(
        run_dataflow_raw_argv_fn=cast(
            CliRunDataflowRawArgvFn,
            context_callable_dep(
                ctx=ctx,
                key="run_dataflow_raw_argv",
                default=default_run_dataflow_raw_argv_fn,
                never_fn=never_fn,
            ),
        ),
        run_check_fn=cast(
            CliRunCheckFn,
            context_callable_dep(
                ctx=ctx,
                key="run_check",
                default=default_run_check_fn,
                never_fn=never_fn,
            ),
        ),
        run_sppf_sync_fn=cast(
            CliRunSppfSyncFn,
            context_callable_dep(
                ctx=ctx,
                key="run_sppf_sync",
                default=default_run_sppf_sync_fn,
                never_fn=never_fn,
            ),
        ),
        run_check_delta_gates_fn=cast(
            CliRunCheckDeltaGatesFn,
            context_callable_dep(
                ctx=ctx,
                key="run_check_delta_gates",
                default=default_run_check_delta_gates_fn,
                never_fn=never_fn,
            ),
        ),
        run_ci_watch_fn=cast(
            CliRunCiWatchFn,
            context_callable_dep(
                ctx=ctx,
                key="run_ci_watch",
                default=default_run_ci_watch_fn,
                never_fn=never_fn,
            ),
        ),
    )
