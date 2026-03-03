# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from collections import defaultdict

from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.order_contract import sort_once


def _merge_counts_by_knobs(
    counts: dict[tuple[str, ...], int],
    knob_names: set[str],
) -> dict[tuple[str, ...], int]:
    check_deadline()
    if not knob_names:
        return counts
    bundles = [set(bundle) for bundle in counts]
    merged: dict[tuple[str, ...], int] = defaultdict(int)
    for bundle_key, count in counts.items():
        check_deadline()
        bundle = set(bundle_key)
        target = bundle
        for other in bundles:
            check_deadline()
            if bundle and bundle.issubset(other):
                extra = set(other) - bundle
                if extra and extra.issubset(knob_names):
                    if len(other) < len(target) or target == bundle:
                        target = set(other)
        merged[
            tuple(
                sort_once(
                    target,
                    source="dataflow_bundle_merge._merge_counts_by_knobs.target",
                )
            )
        ] += count
    return merged


__all__ = ["_merge_counts_by_knobs"]
