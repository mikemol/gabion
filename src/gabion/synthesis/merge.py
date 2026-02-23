# gabion:decision_protocol_module
from __future__ import annotations

from typing import Iterable, List, Set
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import sort_once


def _jaccard(left: Set[str], right: Set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union)


def merge_bundles(
    bundles: Iterable[Set[str]],
    min_overlap: float = 0.75,
) -> List[Set[str]]:
    check_deadline()
    merged: List[Set[str]] = [set(b) for b in bundles]
    changed = True
    while changed:
        check_deadline()
        changed = False
        merged = sort_once(
            merged,
            source="merge_bundles.iteration.merged",
            # Primary key is bundle size; secondary key is lexicalized member tuple.
            key=lambda b: (
                len(b),
                sort_once(
                    b,
                    source="merge_bundles.iteration.sort_key",
                ),
            ),
        )
        result: List[Set[str]] = []
        while merged:
            check_deadline()
            current = merged.pop(0)
            merged_any = False
            for idx, other in enumerate(list(merged)):
                check_deadline()
                if _jaccard(current, other) >= min_overlap:
                    current |= other
                    merged.pop(idx)
                    merged_any = True
                    changed = True
                    break
            if merged_any:
                merged.append(current)
            else:
                result.append(current)
        merged = result
    merged = sort_once(
        merged,
        source="merge_bundles.final.merged",
        # Primary key is bundle size; secondary key is lexicalized member tuple.
        key=lambda b: (
            len(b),
            sort_once(
                b,
                source="merge_bundles.final.sort_key",
            ),
        ),
    )
    return merged
