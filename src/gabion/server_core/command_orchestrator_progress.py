from __future__ import annotations

import hashlib
from dataclasses import dataclass

from typing import Iterator, Mapping, Sequence

from gabion.analysis import report_projection_phase_rank
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import never
from gabion.json_types import JSONObject, JSONValue
from gabion.json_utils import canonical_json_text as _canonical_json_text
from gabion.order_contract import sort_once
from gabion.server_core.coercion_contract import (
    bool_optional as _bool_optional,
    float_optional as _float_optional,
    int_optional as _int_optional,
    json_mapping_default_empty as _json_mapping_default_empty,
    json_mapping_optional as _json_mapping_optional,
    non_negative_int_optional as _non_negative_int_optional,
    non_string_sequence_optional as _non_string_sequence_optional,
    str_optional as _str_optional,
)
from gabion.server_core import dataflow_runtime_contract as runtime_contract

_PHASE_PRIMARY_UNITS: Mapping[str, str] = runtime_contract.PHASE_PRIMARY_UNITS

_REPORT_PHASE_RANK_BY_NAME: dict[str, int] = {
    "collection": report_projection_phase_rank("collection"),
    "forest": report_projection_phase_rank("forest"),
    "edge": report_projection_phase_rank("edge"),
    "post": report_projection_phase_rank("post"),
}


@dataclass(frozen=True)
class _CollectionSemanticStateFact:
    path: str
    phase: str
    processed_functions: frozenset[str]
    processed_count: int
    processed_digest: str


@dataclass(frozen=True)
class _CollectionSemanticPathDelta:
    added_processed_count: int
    regressed_processed_count: int
    changed_in_progress_paths: int
    unchanged_in_progress_paths: int


@dataclass(frozen=True)
class _CollectionSemanticHydrationDelta:
    hydrated_paths_delta: int
    hydrated_paths_regressed: int


@dataclass(frozen=True)
class _CollectionSemanticHydrationFact:
    hydrated_paths: frozenset[str]
    hydrated_count: int
    hydrated_digest: str


@dataclass(frozen=True)
class _CollectionSemanticCumulativeTotals:
    cumulative_new_processed_functions: int
    cumulative_completed_files_delta: int
    cumulative_hydrated_paths_delta: int
    cumulative_regressed_functions: int


@dataclass(frozen=True)
class _CollectionSemanticPathTotals:
    added_processed_count: int
    regressed_processed_count: int
    changed_in_progress_paths: int
    unchanged_in_progress_paths: int


def _count_one(_: object) -> int:
    return 1


def _text_is_present(value: str | None) -> bool:
    return value is not None


def _iter_present_text_values(values: Sequence[object]) -> Iterator[str]:
    yield from filter(_text_is_present, map(_str_optional, values))


def _raw_path_state_item(
    item: tuple[object, object],
) -> tuple[str | None, Mapping[str, JSONValue] | None]:
    raw_path, raw_state = item
    return _str_optional(raw_path), _json_mapping_optional(raw_state)


def _path_state_is_present(
    item: tuple[str | None, Mapping[str, JSONValue] | None],
) -> bool:
    path_value, state_value = item
    return path_value is not None and state_value is not None


def _coerce_path_state_item(
    item: tuple[str | None, Mapping[str, JSONValue] | None],
) -> tuple[str, Mapping[str, JSONValue]]:
    path_value, state_value = item
    if path_value is None or state_value is None:
        never("missing path/state item after present-item filtering")
    return path_value, state_value


def _iter_present_path_state_items(
    in_progress: Mapping[str, JSONValue],
) -> Iterator[tuple[str, Mapping[str, JSONValue]]]:
    yield from map(
        _coerce_path_state_item,
        filter(_path_state_is_present, map(_raw_path_state_item, in_progress.items())),
    )


def _next_ordered_path(previous_path: str | None, current_path: str) -> str:
    if previous_path is not None and previous_path > current_path:
        never(
            "in_progress_scan_by_path path order regression",
            previous_path=previous_path,
            current_path=current_path,
        )
    return current_path


def _iter_ordered_present_path_state_items(
    in_progress: Mapping[str, JSONValue],
) -> Iterator[tuple[str, Mapping[str, JSONValue]]]:
    previous_path: str | None = None
    for path_value, state_value in _iter_present_path_state_items(in_progress):
        check_deadline()
        previous_path = _next_ordered_path(previous_path, path_value)
        yield path_value, state_value


def _text_key_item(item: tuple[object, JSONValue]) -> tuple[str | None, JSONValue]:
    raw_key, value = item
    return _str_optional(raw_key), value


def _text_key_item_is_present(item: tuple[str | None, JSONValue]) -> bool:
    key, _value = item
    return key is not None


def _coerce_text_key_item(item: tuple[str | None, JSONValue]) -> tuple[str, JSONValue]:
    key, value = item
    if key is None:
        never("missing key after present-key filtering")
    return key, value


def _iter_text_key_items(mapping: Mapping[str, JSONValue]) -> Iterator[tuple[str, JSONValue]]:
    yield from map(
        _coerce_text_key_item,
        filter(_text_key_item_is_present, map(_text_key_item, mapping.items())),
    )


def _normalized_dimension_row(item: tuple[str, JSONValue]) -> tuple[str, JSONObject] | None:
    dim_name_text, dim_payload = item
    dim_payload_mapping = _json_mapping_optional(dim_payload)
    if dim_payload_mapping is None:
        return None
    raw_done = dim_payload_mapping.get("done")
    raw_total = dim_payload_mapping.get("total")
    dim_done = _non_negative_int_optional(raw_done)
    dim_total = _non_negative_int_optional(raw_total)
    if dim_done is None or dim_total is None:
        return None
    if dim_total:
        dim_done = min(dim_done, dim_total)
    return dim_name_text, {"done": dim_done, "total": dim_total}


def _dimension_row_is_present(item: tuple[str, JSONObject] | None) -> bool:
    return item is not None


def _coerce_dimension_row(item: tuple[str, JSONObject] | None) -> tuple[str, JSONObject]:
    if item is None:
        never("missing dimension after present-dimension filtering")
    return item


def _iter_dimension_rows(
    dimensions_mapping: Mapping[str, JSONValue],
) -> Iterator[tuple[str, JSONObject]]:
    yield from map(
        _coerce_dimension_row,
        filter(_dimension_row_is_present, map(_normalized_dimension_row, _iter_text_key_items(dimensions_mapping))),
    )

def _report_projection_phase_rank_optional(phase_name: object) -> int | None:
    phase_text = _str_optional(phase_name)
    return _REPORT_PHASE_RANK_BY_NAME.get(phase_text or "")

def _analysis_resume_progress(
    *,
    collection_resume: Mapping[str, JSONValue] | None,
    total_files: int,
) -> dict[str, int]:
    normalized_collection_resume = _json_mapping_optional(collection_resume)
    if normalized_collection_resume is None:
        normalized_total_files = max(total_files, 0)
        return {
            "completed_files": 0,
            "in_progress_files": 0,
            "remaining_files": normalized_total_files,
            "total_files": normalized_total_files,
        }
    completed_paths = _non_string_sequence_optional(
        normalized_collection_resume.get("completed_paths")
    )
    completed = sum(
        map(_count_one, _iter_present_text_values(completed_paths or ()))
    )
    in_progress_scan = _json_mapping_optional(
        normalized_collection_resume.get("in_progress_scan_by_path")
    )
    in_progress = sum(
        map(
            _count_one,
            _iter_present_path_state_items(in_progress_scan or {}),
        )
    )
    if total_files >= 0:
        # A timeout can occur before path discovery populates total_files.
        # Preserve observed checkpoint progress instead of clamping it away.
        observed_files = completed + in_progress
        if observed_files > total_files:
            total_files = observed_files
        completed = min(completed, total_files)
        in_progress = min(in_progress, max(total_files - completed, 0))
    remaining = max(total_files - completed, 0)
    return {
        "completed_files": completed,
        "in_progress_files": in_progress,
        "remaining_files": remaining,
        "total_files": total_files,
    }

def _normalize_progress_work(
    *,
    work_done: object | None,
    work_total: object | None,
) -> tuple[int | None, int | None]:
    normalized_done = _non_negative_int_optional(work_done)
    normalized_total = _non_negative_int_optional(work_total)
    if (
        normalized_done is not None
        and normalized_total is not None
        and normalized_done > normalized_total
    ):
        normalized_done = normalized_total
    return normalized_done, normalized_total

def _phase_primary_unit_for_phase(phase: str) -> str:
    primary_unit = _PHASE_PRIMARY_UNITS.get(phase)
    if primary_unit is not None:
        return primary_unit
    never("unknown phase for primary-unit lookup", phase=phase)

def _build_phase_progress_v2(
    *,
    phase: str,
    collection_progress: Mapping[str, JSONValue],
    semantic_progress: Mapping[str, JSONValue] | None = None,
    work_done: object | None,
    work_total: object | None,
    phase_progress_v2: Mapping[str, JSONValue] | None = None,
) -> tuple[JSONObject, int, int]:
    normalized_work_done, normalized_work_total = _normalize_progress_work(
        work_done=work_done,
        work_total=work_total,
    )
    if normalized_work_done is None or normalized_work_total is None:
        if phase == "collection":
            raw_completed = collection_progress.get("completed_files")
            raw_total = collection_progress.get("total_files")
            normalized_completed = _non_negative_int_optional(raw_completed)
            normalized_total = _non_negative_int_optional(raw_total)
            if normalized_completed is not None and normalized_total is not None:
                normalized_work_done = normalized_completed
                normalized_work_total = normalized_total
    if normalized_work_done is None:
        normalized_work_done = 0
    if normalized_work_total is None:
        normalized_work_total = 0
    if normalized_work_total:
        normalized_work_done = min(normalized_work_done, normalized_work_total)

    primary_unit_for_phase = _phase_primary_unit_for_phase(phase)
    normalized: JSONObject = {
        "format_version": 1,
        "schema": "gabion/phase_progress_v2",
        "primary_unit": primary_unit_for_phase,
        "primary_done": normalized_work_done,
        "primary_total": normalized_work_total,
        "dimensions": {
            primary_unit_for_phase: {
                "done": normalized_work_done,
                "total": normalized_work_total,
            }
        },
        "inventory": {},
    }
    phase_progress_v2_mapping = _json_mapping_optional(phase_progress_v2)
    if phase_progress_v2_mapping is not None:
        normalized.update(dict(_iter_text_key_items(phase_progress_v2_mapping)))
    primary_unit = str(normalized.get("primary_unit", "") or "").strip()
    if not primary_unit:
        primary_unit = primary_unit_for_phase
    raw_primary_done = normalized.get("primary_done")
    raw_primary_total = normalized.get("primary_total")
    normalized_primary_done = _non_negative_int_optional(raw_primary_done)
    normalized_primary_total = _non_negative_int_optional(raw_primary_total)
    primary_done = (
        normalized_primary_done
        if normalized_primary_done is not None
        else normalized_work_done
    )
    primary_total = (
        normalized_primary_total
        if normalized_primary_total is not None
        else normalized_work_total
    )
    if primary_total:
        primary_done = min(primary_done, primary_total)
    normalized["primary_unit"] = primary_unit
    normalized["primary_done"] = primary_done
    normalized["primary_total"] = primary_total
    raw_dimensions = normalized.get("dimensions")
    dimensions: JSONObject = {}
    dimensions_mapping = _json_mapping_optional(raw_dimensions)
    if dimensions_mapping is not None:
        dimensions.update(dict(_iter_dimension_rows(dimensions_mapping)))
    if primary_unit not in dimensions:
        dimensions[primary_unit] = {"done": primary_done, "total": primary_total}
    semantic_progress_mapping = _json_mapping_optional(semantic_progress)
    if phase == "collection" and semantic_progress_mapping is not None:
        raw_cumulative_new = semantic_progress_mapping.get(
            "cumulative_new_processed_functions"
        )
        raw_cumulative_completed = semantic_progress_mapping.get(
            "cumulative_completed_files_delta"
        )
        raw_cumulative_hydrated = semantic_progress_mapping.get(
            "cumulative_hydrated_paths_delta"
        )
        raw_cumulative_regressed = semantic_progress_mapping.get(
            "cumulative_regressed_functions"
        )
        semantic_new = _non_negative_int_optional(raw_cumulative_new) or 0
        semantic_completed = _non_negative_int_optional(raw_cumulative_completed) or 0
        semantic_hydrated = _non_negative_int_optional(raw_cumulative_hydrated) or 0
        semantic_regressed = _non_negative_int_optional(raw_cumulative_regressed) or 0
        if semantic_hydrated > 0 or semantic_regressed > 0:
            dimensions["hydrated_paths_delta"] = {
                "done": semantic_hydrated,
                "total": semantic_hydrated + semantic_regressed,
            }
        semantic_done = semantic_new + semantic_completed + semantic_hydrated
        semantic_total = semantic_done + semantic_regressed
        if semantic_done > 0 or semantic_regressed > 0:
            dimensions["semantic_progress_points"] = {
                "done": semantic_done,
                "total": semantic_total,
            }
    normalized["dimensions"] = dimensions
    raw_inventory = normalized.get("inventory")
    inventory: JSONObject = {}
    inventory_mapping = _json_mapping_optional(raw_inventory)
    if inventory_mapping is not None:
        inventory.update(dict(_iter_text_key_items(inventory_mapping)))
    normalized["inventory"] = inventory
    return normalized, primary_done, primary_total

def _completed_path_set(
    collection_resume: Mapping[str, JSONValue] | None,
) -> set[str]:
    normalized_collection_resume = _json_mapping_optional(collection_resume)
    if normalized_collection_resume is None:
        return set()
    raw_completed_paths = normalized_collection_resume.get("completed_paths")
    completed_paths = _non_string_sequence_optional(raw_completed_paths)
    if completed_paths is None:
        return set()
    return set(_iter_present_text_values(completed_paths))

def _in_progress_scan_states(
    collection_resume: Mapping[str, JSONValue] | None,
) -> dict[str, Mapping[str, JSONValue]]:
    normalized_collection_resume = _json_mapping_optional(collection_resume)
    if normalized_collection_resume is None:
        return {}
    raw_in_progress = normalized_collection_resume.get("in_progress_scan_by_path")
    in_progress_mapping = _json_mapping_optional(raw_in_progress)
    if in_progress_mapping is None:
        return {}
    return dict(_iter_ordered_present_path_state_items(in_progress_mapping))

def _state_processed_functions(state: Mapping[str, JSONValue]) -> set[str]:
    raw_processed = state.get("processed_functions")
    processed_entries = _non_string_sequence_optional(raw_processed)
    if processed_entries is None:
        return set()
    return set(_iter_present_text_values(processed_entries))

def _state_processed_count(state: Mapping[str, JSONValue]) -> int:
    processed_functions = _state_processed_functions(state)
    if processed_functions:
        return len(processed_functions)
    raw_count = state.get("processed_functions_count")
    raw_count_value = _int_optional(raw_count)
    if raw_count_value is not None:
        return max(0, raw_count_value)
    return 0

def _state_processed_digest(state: Mapping[str, JSONValue]) -> str:
    processed_functions = _state_processed_functions(state)
    if processed_functions:
        return hashlib.sha1(
            _canonical_json_text(sort_once(processed_functions, source = 'src/gabion/server.py:1371')).encode("utf-8")
        ).hexdigest()
    raw_digest = state.get("processed_functions_digest")
    digest_text = _str_optional(raw_digest)
    if digest_text:
        return digest_text
    return hashlib.sha1(
        _canonical_json_text({"count": _state_processed_count(state)}).encode("utf-8")
    ).hexdigest()


def _collection_semantic_state_fact(
    path_key: str,
    state: Mapping[str, JSONValue],
) -> _CollectionSemanticStateFact:
    phase = state.get("phase")
    phase_text = _str_optional(phase) or "unknown"
    processed_functions = frozenset(_state_processed_functions(state))
    processed_count = (
        len(processed_functions)
        if processed_functions
        else _state_processed_count(state)
    )
    return _CollectionSemanticStateFact(
        path=path_key,
        phase=phase_text,
        processed_functions=processed_functions,
        processed_count=processed_count,
        processed_digest=_state_processed_digest(state),
    )


def _iter_collection_semantic_state_facts(
    states: Mapping[str, Mapping[str, JSONValue]],
) -> Iterator[_CollectionSemanticStateFact]:
    for path_key, state in states.items():
        check_deadline()
        yield _collection_semantic_state_fact(path_key, state)


def _collection_semantic_state_fact_index(
    collection_resume: Mapping[str, JSONValue] | None,
) -> dict[str, _CollectionSemanticStateFact]:
    states = _in_progress_scan_states(collection_resume)
    return {
        fact.path: fact for fact in _iter_collection_semantic_state_facts(states)
    }


def _iter_collection_semantic_path_keys(
    previous_state_facts: Mapping[str, _CollectionSemanticStateFact],
    current_state_facts: Mapping[str, _CollectionSemanticStateFact],
) -> Iterator[str]:
    yielded_paths: set[str] = set()
    for path_key in previous_state_facts:
        check_deadline()
        yielded_paths.add(path_key)
        yield path_key
    for path_key in current_state_facts:
        check_deadline()
        if path_key in yielded_paths:
            continue
        yield path_key


def _collection_semantic_path_delta(
    *,
    previous_fact: _CollectionSemanticStateFact | None,
    current_fact: _CollectionSemanticStateFact | None,
    current_completed_paths: set[str],
) -> _CollectionSemanticPathDelta:
    path_key = (
        current_fact.path if current_fact is not None else previous_fact.path
        if previous_fact is not None
        else never("path delta requires at least one state fact")
    )
    if (
        previous_fact is not None
        and current_fact is None
        and path_key in current_completed_paths
    ):
        return _CollectionSemanticPathDelta(
            added_processed_count=0,
            regressed_processed_count=0,
            changed_in_progress_paths=1,
            unchanged_in_progress_paths=0,
        )
    previous_keys = (
        previous_fact.processed_functions
        if previous_fact is not None
        else frozenset()
    )
    current_keys = (
        current_fact.processed_functions
        if current_fact is not None
        else frozenset()
    )
    if previous_keys or current_keys:
        added_count = len(current_keys - previous_keys)
        regressed_count = len(previous_keys - current_keys)
    else:
        previous_count = previous_fact.processed_count if previous_fact is not None else 0
        current_count = current_fact.processed_count if current_fact is not None else 0
        added_count = max(0, current_count - previous_count)
        regressed_count = max(0, previous_count - current_count)
    return _CollectionSemanticPathDelta(
        added_processed_count=added_count,
        regressed_processed_count=regressed_count,
        changed_in_progress_paths=0 if added_count == 0 and regressed_count == 0 else 1,
        unchanged_in_progress_paths=1 if added_count == 0 and regressed_count == 0 else 0,
    )


def _iter_collection_semantic_path_deltas(
    *,
    previous_state_facts: Mapping[str, _CollectionSemanticStateFact],
    current_state_facts: Mapping[str, _CollectionSemanticStateFact],
    current_completed_paths: set[str],
) -> Iterator[_CollectionSemanticPathDelta]:
    for path_key in _iter_collection_semantic_path_keys(
        previous_state_facts, current_state_facts
    ):
        yield _collection_semantic_path_delta(
            previous_fact=previous_state_facts.get(path_key),
            current_fact=current_state_facts.get(path_key),
            current_completed_paths=current_completed_paths,
        )

def _analysis_index_resume_hydrated_paths(
    collection_resume: Mapping[str, JSONValue] | None,
) -> set[str]:
    normalized_collection_resume = _json_mapping_optional(collection_resume)
    if normalized_collection_resume is None:
        return set()
    raw_resume = normalized_collection_resume.get("analysis_index_resume")
    resume_mapping = _json_mapping_optional(raw_resume)
    if resume_mapping is None:
        return set()
    raw_hydrated = resume_mapping.get("hydrated_paths")
    hydrated_paths = _non_string_sequence_optional(raw_hydrated)
    if hydrated_paths is None:
        return set()
    return set(_iter_present_text_values(hydrated_paths))


def _normalized_hydrated_paths_count(resume_mapping: Mapping[str, JSONValue]) -> int:
    raw_count = resume_mapping.get("hydrated_paths_count")
    raw_count_value = _int_optional(raw_count)
    if raw_count_value is None:
        return 0
    return max(0, raw_count_value)

def _analysis_index_resume_hydrated_count(
    collection_resume: Mapping[str, JSONValue] | None,
) -> int:
    hydrated = _analysis_index_resume_hydrated_paths(collection_resume)
    hydrated_count = len(hydrated)
    if hydrated_count:
        return hydrated_count
    normalized_collection_resume = _json_mapping_optional(collection_resume)
    raw_resume = None
    if normalized_collection_resume is not None:
        raw_resume = normalized_collection_resume.get("analysis_index_resume")
    resume_mapping = _json_mapping_optional(raw_resume)
    if resume_mapping is None:
        return 0
    return _normalized_hydrated_paths_count(resume_mapping)


def _analysis_index_resume_hydration_fact(
    collection_resume: Mapping[str, JSONValue] | None,
) -> _CollectionSemanticHydrationFact:
    hydrated_paths = frozenset(_analysis_index_resume_hydrated_paths(collection_resume))
    hydrated_count = (
        len(hydrated_paths)
        if hydrated_paths
        else _analysis_index_resume_hydrated_count(collection_resume)
    )
    return _CollectionSemanticHydrationFact(
        hydrated_paths=hydrated_paths,
        hydrated_count=hydrated_count,
        hydrated_digest=_analysis_index_resume_hydrated_digest(collection_resume),
    )


def _analysis_index_resume_hydrated_digest(
    collection_resume: Mapping[str, JSONValue] | None,
) -> str:
    hydrated = _analysis_index_resume_hydrated_paths(collection_resume)
    if hydrated:
        return hashlib.sha1(
            _canonical_json_text(sort_once(hydrated, source = 'src/gabion/server.py:1418')).encode("utf-8")
        ).hexdigest()
    normalized_collection_resume = _json_mapping_optional(collection_resume)
    if normalized_collection_resume is None:
        return hashlib.sha1(b"[]").hexdigest()
    raw_resume = normalized_collection_resume.get("analysis_index_resume")
    resume_mapping = _json_mapping_optional(raw_resume)
    if resume_mapping is None:
        return hashlib.sha1(b"[]").hexdigest()
    raw_digest = resume_mapping.get("hydrated_paths_digest")
    digest_text = _str_optional(raw_digest)
    if digest_text:
        return digest_text
    return hashlib.sha1(
        _canonical_json_text({"count": _analysis_index_resume_hydrated_count(collection_resume)}).encode("utf-8")
    ).hexdigest()


def _collection_semantic_hydration_delta(
    *,
    previous_hydration: _CollectionSemanticHydrationFact,
    current_hydration: _CollectionSemanticHydrationFact,
) -> _CollectionSemanticHydrationDelta:
    if previous_hydration.hydrated_paths or current_hydration.hydrated_paths:
        return _CollectionSemanticHydrationDelta(
            hydrated_paths_delta=len(
                current_hydration.hydrated_paths - previous_hydration.hydrated_paths
            ),
            hydrated_paths_regressed=len(
                previous_hydration.hydrated_paths - current_hydration.hydrated_paths
            ),
        )
    return _CollectionSemanticHydrationDelta(
        hydrated_paths_delta=max(
            0, current_hydration.hydrated_count - previous_hydration.hydrated_count
        ),
        hydrated_paths_regressed=max(
            0, previous_hydration.hydrated_count - current_hydration.hydrated_count
        ),
    )

def _analysis_index_resume_signature(
    collection_resume: Mapping[str, JSONValue] | None,
) -> tuple[int, str, int, int, str, str]:
    hydrated_count = _analysis_index_resume_hydrated_count(collection_resume)
    hydrated_digest = _analysis_index_resume_hydrated_digest(collection_resume)
    normalized_collection_resume = _json_mapping_optional(collection_resume)
    if normalized_collection_resume is None:
        return (hydrated_count, hydrated_digest, 0, 0, "", hydrated_digest)
    raw_resume = normalized_collection_resume.get("analysis_index_resume")
    resume_mapping = _json_mapping_optional(raw_resume)
    if resume_mapping is None:
        return (hydrated_count, hydrated_digest, 0, 0, "", hydrated_digest)
    function_count = _int_optional(resume_mapping.get("function_count"))
    class_count = _int_optional(resume_mapping.get("class_count"))
    phase = _str_optional(resume_mapping.get("phase"))
    resume_digest = _str_optional(resume_mapping.get("resume_digest"))
    function_count_value = function_count if function_count is not None else 0
    class_count_value = class_count if class_count is not None else 0
    phase_value = phase if phase is not None else ""
    if not resume_digest:
        resume_digest = hydrated_digest
    return (
        hydrated_count,
        hydrated_digest,
        function_count_value,
        class_count_value,
        phase_value,
        resume_digest,
    )


def _collection_semantic_state_row(
    fact: _CollectionSemanticStateFact,
) -> JSONObject:
    return {
        "path": fact.path,
        "phase": fact.phase,
        "processed_functions_count": fact.processed_count,
        "processed_functions_digest": fact.processed_digest,
    }


def _iter_collection_semantic_state_rows(
    state_facts: Mapping[str, _CollectionSemanticStateFact],
) -> Iterator[JSONObject]:
    for fact in state_facts.values():
        check_deadline()
        yield _collection_semantic_state_row(fact)


def _collection_semantic_processed_total(
    state_facts: Mapping[str, _CollectionSemanticStateFact],
) -> int:
    return sum(fact.processed_count for fact in state_facts.values())


def _reduce_collection_semantic_path_deltas(
    path_deltas: Iterator[_CollectionSemanticPathDelta],
) -> _CollectionSemanticPathTotals:
    added_processed_count = 0
    regressed_processed_count = 0
    changed_in_progress_paths = 0
    unchanged_in_progress_paths = 0
    for path_delta in path_deltas:
        check_deadline()
        added_processed_count += path_delta.added_processed_count
        regressed_processed_count += path_delta.regressed_processed_count
        changed_in_progress_paths += path_delta.changed_in_progress_paths
        unchanged_in_progress_paths += path_delta.unchanged_in_progress_paths
    return _CollectionSemanticPathTotals(
        added_processed_count=added_processed_count,
        regressed_processed_count=regressed_processed_count,
        changed_in_progress_paths=changed_in_progress_paths,
        unchanged_in_progress_paths=unchanged_in_progress_paths,
    )


def _collection_semantic_cumulative_totals(
    *,
    path_totals: _CollectionSemanticPathTotals,
    completed_delta: int,
    completed_regressed: int,
    hydration_delta: _CollectionSemanticHydrationDelta,
    cumulative: Mapping[str, JSONValue] | None,
) -> _CollectionSemanticCumulativeTotals:
    cumulative_totals = _CollectionSemanticCumulativeTotals(
        cumulative_new_processed_functions=path_totals.added_processed_count,
        cumulative_completed_files_delta=completed_delta,
        cumulative_hydrated_paths_delta=hydration_delta.hydrated_paths_delta,
        cumulative_regressed_functions=(
            path_totals.regressed_processed_count
            + completed_regressed
            + hydration_delta.hydrated_paths_regressed
        ),
    )
    cumulative_mapping = _json_mapping_optional(cumulative)
    if cumulative_mapping is None:
        return cumulative_totals
    raw_cumulative_new = _int_optional(
        cumulative_mapping.get("cumulative_new_processed_functions")
    )
    raw_cumulative_completed = _int_optional(
        cumulative_mapping.get("cumulative_completed_files_delta")
    )
    raw_cumulative_hydrated = _int_optional(
        cumulative_mapping.get("cumulative_hydrated_paths_delta")
    )
    raw_cumulative_regressed = _int_optional(
        cumulative_mapping.get("cumulative_regressed_functions")
    )
    return _CollectionSemanticCumulativeTotals(
        cumulative_new_processed_functions=(
            cumulative_totals.cumulative_new_processed_functions
            + (max(0, raw_cumulative_new) if raw_cumulative_new is not None else 0)
        ),
        cumulative_completed_files_delta=(
            cumulative_totals.cumulative_completed_files_delta
            + (
                max(0, raw_cumulative_completed)
                if raw_cumulative_completed is not None
                else 0
            )
        ),
        cumulative_hydrated_paths_delta=(
            cumulative_totals.cumulative_hydrated_paths_delta
            + (
                max(0, raw_cumulative_hydrated)
                if raw_cumulative_hydrated is not None
                else 0
            )
        ),
        cumulative_regressed_functions=(
            cumulative_totals.cumulative_regressed_functions
            + (
                max(0, raw_cumulative_regressed)
                if raw_cumulative_regressed is not None
                else 0
            )
        ),
    )

def _collection_semantic_witness(
    *,
    collection_resume: Mapping[str, JSONValue] | None,
) -> JSONObject:
    state_facts = _collection_semantic_state_fact_index(collection_resume)
    state_rows = list(_iter_collection_semantic_state_rows(state_facts))
    processed_total = _collection_semantic_processed_total(state_facts)
    index_signature = _analysis_index_resume_signature(collection_resume)
    hydration_fact = _analysis_index_resume_hydration_fact(collection_resume)
    digest = hashlib.sha1(
        _canonical_json_text(
            {
                "in_progress": state_rows,
                "index_hydrated_paths_count": hydration_fact.hydrated_count,
                "index_hydrated_paths_digest": hydration_fact.hydrated_digest,
                "index_resume_digest": index_signature[5],
                "index_function_count": index_signature[2],
                "index_class_count": index_signature[3],
            }
        ).encode("utf-8")
    ).hexdigest()
    return {
        "witness_digest": digest,
        "in_progress_paths": len(state_rows),
        "processed_functions_total": processed_total,
        "index_hydrated_paths_count": hydration_fact.hydrated_count,
        "index_hydrated_paths_digest": hydration_fact.hydrated_digest,
        "index_resume_digest": index_signature[5],
        "index_function_count": index_signature[2],
        "index_class_count": index_signature[3],
    }

def _collection_semantic_progress(
    *,
    previous_collection_resume: Mapping[str, JSONValue] | None,
    collection_resume: Mapping[str, JSONValue],
    total_files: int,
    cumulative: Mapping[str, JSONValue] | None = None,
) -> JSONObject:
    previous_state_facts = _collection_semantic_state_fact_index(previous_collection_resume)
    current_state_facts = _collection_semantic_state_fact_index(collection_resume)
    current_completed_paths = _completed_path_set(collection_resume)
    prev_progress = _analysis_resume_progress(
        collection_resume=previous_collection_resume,
        total_files=total_files,
    )
    current_progress = _analysis_resume_progress(
        collection_resume=collection_resume,
        total_files=total_files,
    )
    path_totals = _reduce_collection_semantic_path_deltas(
        _iter_collection_semantic_path_deltas(
            previous_state_facts=previous_state_facts,
            current_state_facts=current_state_facts,
            current_completed_paths=current_completed_paths,
        )
    )
    completed_delta = max(
        0, current_progress["completed_files"] - prev_progress["completed_files"]
    )
    completed_regressed = max(
        0, prev_progress["completed_files"] - current_progress["completed_files"]
    )
    hydration_delta = _collection_semantic_hydration_delta(
        previous_hydration=_analysis_index_resume_hydration_fact(
            previous_collection_resume
        ),
        current_hydration=_analysis_index_resume_hydration_fact(collection_resume),
    )
    cumulative_totals = _collection_semantic_cumulative_totals(
        path_totals=path_totals,
        completed_delta=completed_delta,
        completed_regressed=completed_regressed,
        hydration_delta=hydration_delta,
        cumulative=cumulative,
    )
    current_witness = _collection_semantic_witness(collection_resume=collection_resume)
    previous_resume_mapping = _json_mapping_optional(previous_collection_resume)
    previous_witness = (
        _collection_semantic_witness(collection_resume=previous_resume_mapping)
        if previous_resume_mapping is not None
        else {"witness_digest": None}
    )
    substantive_progress = (
        (
            cumulative_totals.cumulative_new_processed_functions > 0
            or cumulative_totals.cumulative_completed_files_delta > 0
            or cumulative_totals.cumulative_hydrated_paths_delta > 0
        )
        and cumulative_totals.cumulative_regressed_functions == 0
    )
    return {
        "current_witness_digest": current_witness.get("witness_digest"),
        "previous_witness_digest": previous_witness.get("witness_digest"),
        "new_processed_functions_count": path_totals.added_processed_count,
        "regressed_processed_functions_count": path_totals.regressed_processed_count,
        "completed_files_delta": completed_delta,
        "completed_files_regressed": completed_regressed,
        "hydrated_paths_delta": hydration_delta.hydrated_paths_delta,
        "hydrated_paths_regressed": hydration_delta.hydrated_paths_regressed,
        "changed_in_progress_paths": path_totals.changed_in_progress_paths,
        "unchanged_in_progress_paths": path_totals.unchanged_in_progress_paths,
        "cumulative_new_processed_functions": cumulative_totals.cumulative_new_processed_functions,
        "cumulative_completed_files_delta": cumulative_totals.cumulative_completed_files_delta,
        "cumulative_hydrated_paths_delta": cumulative_totals.cumulative_hydrated_paths_delta,
        "cumulative_regressed_functions": cumulative_totals.cumulative_regressed_functions,
        "monotonic_progress": cumulative_totals.cumulative_regressed_functions == 0,
        "substantive_progress": substantive_progress,
    }


analysis_index_resume_hydrated_count = _analysis_index_resume_hydrated_count
analysis_index_resume_signature = _analysis_index_resume_signature
analysis_resume_progress = _analysis_resume_progress
build_phase_progress_v2 = _build_phase_progress_v2
collection_semantic_progress = _collection_semantic_progress
in_progress_scan_states = _in_progress_scan_states
normalize_progress_work = _normalize_progress_work
report_projection_phase_rank_optional = _report_projection_phase_rank_optional
