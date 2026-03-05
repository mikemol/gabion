# gabion:boundary_normalization_module
from __future__ import annotations

"""Canonical ingest-analysis helpers extracted from the indexed monolith."""

import time
from collections import defaultdict
from typing import cast

from gabion.analysis.dataflow.engine.dataflow_analysis_index import (
    _profiling_v1_payload,
)
from gabion.analysis.dataflow.engine.dataflow_contracts import AuditConfig, CallArgs, ParamUse
from gabion.analysis.dataflow.engine.dataflow_post_phase_analyses import (
    _callsite_evidence_for_bundle,
)
from gabion.analysis.foundation.json_types import JSONObject
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.indexed_scan.scanners.analyze_ingested_file import (
    AnalyzeIngestedFileDeps as _AnalyzeIngestedFileDeps,
    analyze_ingested_file as _analyze_ingested_file_impl,
)
from gabion.analysis.indexed_scan.scanners.flow.group_propagation import (
    PropagateGroupsDeps as _PropagateGroupsDeps,
    propagate_groups as _propagate_groups_impl,
)
from gabion.order_contract import sort_once


def _group_by_signature(use_map: dict[str, ParamUse]) -> list[set[str]]:
    check_deadline()
    sig_map: dict[tuple[tuple[str, str], ...], list[str]] = defaultdict(list)
    for name, info in use_map.items():
        check_deadline()
        if info.non_forward:
            continue
        sig = tuple(
            sort_once(
                info.direct_forward,
                source="gabion.analysis.dataflow_ingested_analysis_support._group_by_signature.site_1",
            )
        )
        if not sig:
            continue
        sig_map[sig].append(name)
    return [set(names) for names in sig_map.values() if len(names) > 1]


def _union_groups(groups: list[set[str]]) -> list[set[str]]:
    check_deadline()
    changed = True
    while changed:
        check_deadline()
        changed = False
        out = []
        while groups:
            check_deadline()
            base = groups.pop()
            merged = True
            while merged:
                check_deadline()
                merged = False
                for i, other in enumerate(groups):
                    check_deadline()
                    if base & other:
                        base |= other
                        groups.pop(i)
                        merged = True
                        changed = True
                        break
            out.append(base)
        groups = out
    return groups


def _propagate_groups(
    call_args: list[CallArgs],
    callee_groups: dict[str, list[set[str]]],
    callee_param_orders: dict[str, list[str]],
    strictness: str,
    opaque_callees=None,
) -> list[set[str]]:
    return cast(
        list[set[str]],
        _propagate_groups_impl(
            cast(list[object], call_args),
            callee_groups,
            callee_param_orders,
            strictness,
            opaque_callees=opaque_callees,
            deps=_PropagateGroupsDeps(check_deadline_fn=check_deadline),
        ),
    )


def _adapt_ingest_carrier_to_analysis_maps(ingest_carrier):
    return (
        dict(ingest_carrier.function_use),
        dict(ingest_carrier.function_calls),
        dict(ingest_carrier.function_param_orders),
        dict(ingest_carrier.function_param_spans),
        set(ingest_carrier.opaque_callees),
    )


def analyze_ingested_file(
    ingest_carrier,
    *,
    recursive: bool,
    config: AuditConfig,
    on_profile=None,
) -> tuple[
    dict[str, list[set[str]]],
    dict[str, dict[str, tuple[int, int, int, int]]],
    dict[str, list[list[JSONObject]]],
]:
    return _analyze_ingested_file_impl(
        ingest_carrier,
        recursive=recursive,
        config=config,
        on_profile=on_profile,
        deps=_AnalyzeIngestedFileDeps(
            adapt_ingest_carrier_to_analysis_maps_fn=_adapt_ingest_carrier_to_analysis_maps,
            profiling_v1_payload_fn=_profiling_v1_payload,
            monotonic_ns_fn=time.monotonic_ns,
            group_by_signature_fn=_group_by_signature,
            callsite_evidence_for_bundle_fn=_callsite_evidence_for_bundle,
            propagate_groups_fn=_propagate_groups,
            union_groups_fn=_union_groups,
            check_deadline_fn=check_deadline,
        ),
    )


__all__ = [
    "_group_by_signature",
    "_propagate_groups",
    "_union_groups",
    "analyze_ingested_file",
]
