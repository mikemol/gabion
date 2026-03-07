from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable


def normalize_snapshot_path(path: Path, root: object) -> str:
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def iter_monotonic_paths(
    paths: Iterable[Path],
    *,
    source: str,
    analysis_collection_resume_path_key_fn: Callable[[Path], str],
    check_deadline_fn: Callable[[], None],
    never_fn: Callable[..., object],
) -> list[Path]:
    ordered: list[Path] = []
    previous_path_key = ""
    has_previous = False
    for path in paths:
        check_deadline_fn()
        path_key = analysis_collection_resume_path_key_fn(path)
        if has_previous and previous_path_key > path_key:
            never_fn(
                "path order regression",
                source=source,
                previous_path=previous_path_key,
                current_path=path_key,
            )
        previous_path_key = path_key
        has_previous = True
        ordered.append(path)
    return ordered

