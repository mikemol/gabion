# gabion:decision_protocol_module
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

from gabion.analysis.foundation.json_types import JSONObject


@dataclass(frozen=True)
class AnalyzeIngestedFileDeps:
    adapt_ingest_carrier_to_analysis_maps_fn: Callable[
        [object],
        tuple[
            dict[str, dict[str, object]],
            dict[str, list[object]],
            dict[str, list[str]],
            dict[str, dict[str, tuple[int, int, int, int]]],
            set[str],
        ],
    ]
    profiling_v1_payload_fn: Callable[..., JSONObject]
    monotonic_ns_fn: Callable[[], int]
    group_by_signature_fn: Callable[[dict[str, object]], list[set[str]]]
    callsite_evidence_for_bundle_fn: Callable[[list[object], set[str]], list[JSONObject]]
    propagate_groups_fn: Callable[
        [
            list[object],
            dict[str, list[set[str]]],
            dict[str, list[str]],
            str,
            set[str],
        ],
        list[set[str]],
    ]
    union_groups_fn: Callable[[list[set[str]]], list[set[str]]]
    check_deadline_fn: Callable[[], None]


def analyze_ingested_file(
    ingest_carrier,
    *,
    recursive: bool,
    config,
    on_profile=None,
    deps: AnalyzeIngestedFileDeps,
) -> tuple[
    dict[str, list[set[str]]],
    dict[str, dict[str, tuple[int, int, int, int]]],
    dict[str, list[list[JSONObject]]],
]:
    (
        fn_use,
        fn_calls,
        fn_param_orders,
        fn_param_spans,
        opaque_callees,
    ) = deps.adapt_ingest_carrier_to_analysis_maps_fn(ingest_carrier)

    profile_stage_ns: dict[str, int] = {
        "file_scan.grouping": 0,
        "file_scan.propagation": 0,
        "file_scan.bundle_sites": 0,
    }

    def _emit_file_profile() -> None:
        if on_profile is not None:
            on_profile(
                deps.profiling_v1_payload_fn(
                    stage_ns=profile_stage_ns,
                    counters=Counter(),
                )
            )

    grouping_started_ns = deps.monotonic_ns_fn()
    groups_by_fn = {
        fn: deps.group_by_signature_fn(use_map) for fn, use_map in fn_use.items()
    }
    profile_stage_ns["file_scan.grouping"] += deps.monotonic_ns_fn() - grouping_started_ns

    if not recursive:
        bundle_started_ns = deps.monotonic_ns_fn()
        bundle_sites_by_fn: dict[str, list[list[JSONObject]]] = {}
        for fn_key, bundles in groups_by_fn.items():
            deps.check_deadline_fn()
            calls = fn_calls.get(fn_key, [])
            bundle_sites_by_fn[fn_key] = [
                deps.callsite_evidence_for_bundle_fn(calls, bundle) for bundle in bundles
            ]
        profile_stage_ns["file_scan.bundle_sites"] += (
            deps.monotonic_ns_fn() - bundle_started_ns
        )
        _emit_file_profile()
        return groups_by_fn, fn_param_spans, bundle_sites_by_fn

    propagation_started_ns = deps.monotonic_ns_fn()
    changed = True
    while changed:
        deps.check_deadline_fn()
        changed = False
        for fn in fn_use:
            deps.check_deadline_fn()
            propagated = deps.propagate_groups_fn(
                fn_calls[fn],
                groups_by_fn,
                fn_param_orders,
                config.strictness,
                opaque_callees,
            )
            if not propagated:
                continue
            combined = deps.union_groups_fn(groups_by_fn.get(fn, []) + propagated)
            if combined != groups_by_fn.get(fn, []):
                groups_by_fn[fn] = combined
                changed = True
    profile_stage_ns["file_scan.propagation"] += (
        deps.monotonic_ns_fn() - propagation_started_ns
    )

    bundle_started_ns = deps.monotonic_ns_fn()
    bundle_sites_by_fn: dict[str, list[list[JSONObject]]] = {}
    for fn_key, bundles in groups_by_fn.items():
        deps.check_deadline_fn()
        calls = fn_calls.get(fn_key, [])
        bundle_sites_by_fn[fn_key] = [
            deps.callsite_evidence_for_bundle_fn(calls, bundle) for bundle in bundles
        ]
    profile_stage_ns["file_scan.bundle_sites"] += deps.monotonic_ns_fn() - bundle_started_ns
    _emit_file_profile()
    return groups_by_fn, fn_param_spans, bundle_sites_by_fn
