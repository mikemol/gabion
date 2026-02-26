# gabion:decision_protocol_module
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping


def normalize_timeout_total_ticks(
    payload: Mapping[str, object],
    *,
    default_ticks: int,
    never_fn: Callable[..., object],
) -> int:
    explicit_tick_limit_value = payload.get("analysis_tick_limit")
    if explicit_tick_limit_value in (None, ""):
        return default_ticks
    try:
        explicit_tick_limit = int(explicit_tick_limit_value)
    except (TypeError, ValueError):
        never_fn("invalid analysis tick limit", tick_limit=explicit_tick_limit_value)
        return default_ticks
    if explicit_tick_limit <= 0:
        never_fn("invalid analysis tick limit", tick_limit=explicit_tick_limit_value)
        return default_ticks
    return min(default_ticks, explicit_tick_limit)


def initial_collection_progress(*, total_files: int) -> dict[str, int]:
    return {
        "completed_files": 0,
        "in_progress_files": 0,
        "remaining_files": 0,
        "total_files": int(total_files),
    }


def initial_paths_count(paths_value: object) -> int:
    return len(paths_value) if isinstance(paths_value, list) else 1


def normalize_paths(raw_paths: object, *, root: Path) -> list[Path]:
    if isinstance(raw_paths, list) and raw_paths:
        return [Path(str(path_value)) for path_value in raw_paths]
    return [root]
