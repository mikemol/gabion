from __future__ import annotations

from typing import Iterable, List, Set


def _jaccard(left: Set[str], right: Set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union)


def merge_bundles(
    bundles: Iterable[Set[str]],
    min_overlap: float = 0.75,
) -> List[Set[str]]:
    merged: List[Set[str]] = [set(b) for b in bundles]
    changed = True
    while changed:
        changed = False
        merged.sort(key=lambda b: (len(b), sorted(b)))
        result: List[Set[str]] = []
        while merged:
            current = merged.pop(0)
            merged_any = False
            for idx, other in enumerate(list(merged)):
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
    merged.sort(key=lambda b: (len(b), sorted(b)))
    return merged
