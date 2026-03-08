from __future__ import annotations

import hashlib
import json

from functools import singledispatch
from typing import Mapping, Sequence

from gabion.analysis import report_projection_phase_rank
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import never
from gabion.json_types import JSONObject, JSONValue
from gabion.order_contract import sort_once
from gabion.server_core import dataflow_runtime_contract as runtime_contract

_PHASE_PRIMARY_UNITS: Mapping[str, str] = runtime_contract.PHASE_PRIMARY_UNITS

_NONE_TYPE = type(None)


def _canonical_json_text(payload: object) -> str:
    return json.dumps(payload, sort_keys=False, separators=(",", ":"), ensure_ascii=True)

@singledispatch
def _json_mapping_or_none(value: object) -> dict[str, JSONValue] | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_json_mapping_or_none.register(dict)
def _(value: dict[str, JSONValue]) -> dict[str, JSONValue] | None:
    return value


def _json_mapping_none(value: object) -> dict[str, JSONValue] | None:
    _ = value
    return None


for _runtime_type in (list, tuple, set, str, int, float, bool, _NONE_TYPE):
    _json_mapping_or_none.register(_runtime_type)(_json_mapping_none)

def _json_mapping_or_empty(value: object) -> dict[str, JSONValue]:
    mapping = _json_mapping_or_none(value)
    if mapping is not None:
        return mapping
    return {}

@singledispatch
def _int_or_none(value: object) -> int | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_int_or_none.register
def _(value: int) -> int | None:
    return value


def _int_none(value: object) -> int | None:
    _ = value
    return None


for _runtime_type in (bool, float, str, list, tuple, set, dict, _NONE_TYPE):
    _int_or_none.register(_runtime_type)(_int_none)

def _non_negative_int_or_none(value: object) -> int | None:
    int_value = _int_or_none(value)
    return max(int_value, 0) if int_value is not None else None

@singledispatch
def _non_string_sequence_or_none(value: object) -> Sequence[object] | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_non_string_sequence_or_none.register
def _(value: list) -> Sequence[object] | None:
    return value


@_non_string_sequence_or_none.register
def _(value: tuple) -> Sequence[object] | None:
    return value


@_non_string_sequence_or_none.register
def _(value: set) -> Sequence[object] | None:
    return value


def _sequence_none(value: object) -> Sequence[object] | None:
    _ = value
    return None


for _runtime_type in (str, dict, int, float, bool, _NONE_TYPE):
    _non_string_sequence_or_none.register(_runtime_type)(_sequence_none)


@singledispatch
def _str_or_none(value: object) -> str | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_str_or_none.register
def _(value: str) -> str | None:
    return value


def _str_none(value: object) -> str | None:
    _ = value
    return None


for _runtime_type in (int, float, bool, list, tuple, set, dict, _NONE_TYPE):
    _str_or_none.register(_runtime_type)(_str_none)


@singledispatch
def _bool_or_none(value: object) -> bool | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_bool_or_none.register
def _(value: bool) -> bool | None:
    return value


def _bool_none(value: object) -> bool | None:
    _ = value
    return None


for _runtime_type in (int, float, str, list, tuple, set, dict, _NONE_TYPE):
    _bool_or_none.register(_runtime_type)(_bool_none)


@singledispatch
def _float_or_none(value: object) -> float | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_float_or_none.register
def _(value: float) -> float | None:
    return value


def _float_none(value: object) -> float | None:
    _ = value
    return None


for _runtime_type in (int, bool, str, list, tuple, set, dict, _NONE_TYPE):
    _float_or_none.register(_runtime_type)(_float_none)

_REPORT_PHASE_RANK_BY_NAME: dict[str, int] = {
    "collection": report_projection_phase_rank("collection"),
    "forest": report_projection_phase_rank("forest"),
    "edge": report_projection_phase_rank("edge"),
    "post": report_projection_phase_rank("post"),
}

def _report_projection_phase_rank_or_none(phase_name: object) -> int | None:
    phase_text = _str_or_none(phase_name)
    return _REPORT_PHASE_RANK_BY_NAME.get(phase_text or "")

def _analysis_resume_progress(
    *,
    collection_resume: Mapping[str, JSONValue] | None,
    total_files: int,
) -> dict[str, int]:
    normalized_collection_resume = _json_mapping_or_none(collection_resume)
    if normalized_collection_resume is None:
        normalized_total_files = max(total_files, 0)
        return {
            "completed_files": 0,
            "in_progress_files": 0,
            "remaining_files": normalized_total_files,
            "total_files": normalized_total_files,
        }
    completed_paths = _non_string_sequence_or_none(
        normalized_collection_resume.get("completed_paths")
    )
    completed = 0
    if completed_paths is not None:
        completed = sum(
            1
            for path in completed_paths
            if _str_or_none(path) is not None
        )
    in_progress_scan = _json_mapping_or_none(
        normalized_collection_resume.get("in_progress_scan_by_path")
    )
    in_progress = 0
    if in_progress_scan is not None:
        for path, state in in_progress_scan.items():
            path_text = _str_or_none(path)
            state_mapping = _json_mapping_or_none(state)
            if path_text is not None and state_mapping is not None:
                in_progress += 1
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
    normalized_done = _non_negative_int_or_none(work_done)
    normalized_total = _non_negative_int_or_none(work_total)
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
            normalized_completed = _non_negative_int_or_none(raw_completed)
            normalized_total = _non_negative_int_or_none(raw_total)
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
    phase_progress_v2_mapping = _json_mapping_or_none(phase_progress_v2)
    if phase_progress_v2_mapping is not None:
        for key, value in phase_progress_v2_mapping.items():
            key_text = _str_or_none(key)
            if key_text is not None:
                normalized[key_text] = value
    primary_unit = str(normalized.get("primary_unit", "") or "").strip()
    if not primary_unit:
        primary_unit = primary_unit_for_phase
    raw_primary_done = normalized.get("primary_done")
    raw_primary_total = normalized.get("primary_total")
    normalized_primary_done = _non_negative_int_or_none(raw_primary_done)
    normalized_primary_total = _non_negative_int_or_none(raw_primary_total)
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
    dimensions_mapping = _json_mapping_or_none(raw_dimensions)
    if dimensions_mapping is not None:
        for dim_name, dim_payload in dimensions_mapping.items():
            dim_name_text = _str_or_none(dim_name)
            dim_payload_mapping = _json_mapping_or_none(dim_payload)
            if dim_name_text is not None and dim_payload_mapping is not None:
                raw_done = dim_payload_mapping.get("done")
                raw_total = dim_payload_mapping.get("total")
                dim_done = _non_negative_int_or_none(raw_done)
                dim_total = _non_negative_int_or_none(raw_total)
                if dim_done is not None and dim_total is not None:
                    if dim_total:
                        dim_done = min(dim_done, dim_total)
                    dimensions[dim_name_text] = {"done": dim_done, "total": dim_total}
    if primary_unit not in dimensions:
        dimensions[primary_unit] = {"done": primary_done, "total": primary_total}
    semantic_progress_mapping = _json_mapping_or_none(semantic_progress)
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
        semantic_new = _non_negative_int_or_none(raw_cumulative_new) or 0
        semantic_completed = _non_negative_int_or_none(raw_cumulative_completed) or 0
        semantic_hydrated = _non_negative_int_or_none(raw_cumulative_hydrated) or 0
        semantic_regressed = _non_negative_int_or_none(raw_cumulative_regressed) or 0
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
    inventory_mapping = _json_mapping_or_none(raw_inventory)
    if inventory_mapping is not None:
        for inv_key, inv_value in inventory_mapping.items():
            inv_key_text = _str_or_none(inv_key)
            if inv_key_text is not None:
                inventory[inv_key_text] = inv_value
    normalized["inventory"] = inventory
    return normalized, primary_done, primary_total

def _completed_path_set(
    collection_resume: Mapping[str, JSONValue] | None,
) -> set[str]:
    normalized_collection_resume = _json_mapping_or_none(collection_resume)
    if normalized_collection_resume is None:
        return set()
    raw_completed_paths = normalized_collection_resume.get("completed_paths")
    completed_paths = _non_string_sequence_or_none(raw_completed_paths)
    if completed_paths is None:
        return set()
    return {
        path_text
        for path_text in (_str_or_none(path) for path in completed_paths)
        if path_text is not None
    }

def _in_progress_scan_states(
    collection_resume: Mapping[str, JSONValue] | None,
) -> dict[str, Mapping[str, JSONValue]]:
    states: dict[str, Mapping[str, JSONValue]] = {}
    normalized_collection_resume = _json_mapping_or_none(collection_resume)
    if normalized_collection_resume is None:
        return states
    raw_in_progress = normalized_collection_resume.get("in_progress_scan_by_path")
    in_progress_mapping = _json_mapping_or_none(raw_in_progress)
    if in_progress_mapping is None:
        return states
    previous_path: str | None = None
    for raw_path, raw_state in in_progress_mapping.items():
        check_deadline()
        path_text = _str_or_none(raw_path)
        if path_text is not None:
            if previous_path is not None and previous_path > path_text:
                never(
                    "in_progress_scan_by_path path order regression",
                    previous_path=previous_path,
                    current_path=path_text,
                )
            previous_path = path_text
            state_mapping = _json_mapping_or_none(raw_state)
            if state_mapping is not None:
                states[path_text] = state_mapping
    return states

def _state_processed_functions(state: Mapping[str, JSONValue]) -> set[str]:
    raw_processed = state.get("processed_functions")
    processed_entries = _non_string_sequence_or_none(raw_processed)
    if processed_entries is None:
        return set()
    return {
        entry_text
        for entry_text in (_str_or_none(entry) for entry in processed_entries)
        if entry_text is not None
    }

def _state_processed_count(state: Mapping[str, JSONValue]) -> int:
    processed_functions = _state_processed_functions(state)
    if processed_functions:
        return len(processed_functions)
    raw_count = state.get("processed_functions_count")
    raw_count_value = _int_or_none(raw_count)
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
    digest_text = _str_or_none(raw_digest)
    if digest_text:
        return digest_text
    return hashlib.sha1(
        _canonical_json_text({"count": _state_processed_count(state)}).encode("utf-8")
    ).hexdigest()

def _analysis_index_resume_hydrated_paths(
    collection_resume: Mapping[str, JSONValue] | None,
) -> set[str]:
    normalized_collection_resume = _json_mapping_or_none(collection_resume)
    if normalized_collection_resume is None:
        return set()
    raw_resume = normalized_collection_resume.get("analysis_index_resume")
    resume_mapping = _json_mapping_or_none(raw_resume)
    if resume_mapping is None:
        return set()
    raw_hydrated = resume_mapping.get("hydrated_paths")
    hydrated_paths = _non_string_sequence_or_none(raw_hydrated)
    if hydrated_paths is None:
        return set()
    return {
        entry_text
        for entry_text in (_str_or_none(entry) for entry in hydrated_paths)
        if entry_text is not None
    }

def _analysis_index_resume_hydrated_count(
    collection_resume: Mapping[str, JSONValue] | None,
) -> int:
    hydrated = _analysis_index_resume_hydrated_paths(collection_resume)
    hydrated_count = len(hydrated)
    if hydrated_count:
        return hydrated_count
    normalized_collection_resume = _json_mapping_or_none(collection_resume)
    raw_resume = (
        normalized_collection_resume.get("analysis_index_resume")
        if normalized_collection_resume is not None
        else None
    )
    resume_mapping = _json_mapping_or_none(raw_resume)
    raw_count = resume_mapping.get("hydrated_paths_count") if resume_mapping is not None else None
    raw_count_value = _int_or_none(raw_count)
    return max(0, raw_count_value) if raw_count_value is not None else 0

def _analysis_index_resume_hydrated_digest(
    collection_resume: Mapping[str, JSONValue] | None,
) -> str:
    hydrated = _analysis_index_resume_hydrated_paths(collection_resume)
    if hydrated:
        return hashlib.sha1(
            _canonical_json_text(sort_once(hydrated, source = 'src/gabion/server.py:1418')).encode("utf-8")
        ).hexdigest()
    normalized_collection_resume = _json_mapping_or_none(collection_resume)
    if normalized_collection_resume is None:
        return hashlib.sha1(b"[]").hexdigest()
    raw_resume = normalized_collection_resume.get("analysis_index_resume")
    resume_mapping = _json_mapping_or_none(raw_resume)
    if resume_mapping is None:
        return hashlib.sha1(b"[]").hexdigest()
    raw_digest = resume_mapping.get("hydrated_paths_digest")
    digest_text = _str_or_none(raw_digest)
    if digest_text:
        return digest_text
    return hashlib.sha1(
        _canonical_json_text({"count": _analysis_index_resume_hydrated_count(collection_resume)}).encode("utf-8")
    ).hexdigest()

def _analysis_index_resume_signature(
    collection_resume: Mapping[str, JSONValue] | None,
) -> tuple[int, str, int, int, str, str]:
    hydrated_count = _analysis_index_resume_hydrated_count(collection_resume)
    hydrated_digest = _analysis_index_resume_hydrated_digest(collection_resume)
    normalized_collection_resume = _json_mapping_or_none(collection_resume)
    if normalized_collection_resume is None:
        return (hydrated_count, hydrated_digest, 0, 0, "", hydrated_digest)
    raw_resume = normalized_collection_resume.get("analysis_index_resume")
    resume_mapping = _json_mapping_or_none(raw_resume)
    if resume_mapping is None:
        return (hydrated_count, hydrated_digest, 0, 0, "", hydrated_digest)
    function_count = _int_or_none(resume_mapping.get("function_count"))
    class_count = _int_or_none(resume_mapping.get("class_count"))
    phase = _str_or_none(resume_mapping.get("phase"))
    resume_digest = _str_or_none(resume_mapping.get("resume_digest"))
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

def _collection_semantic_witness(
    *,
    collection_resume: Mapping[str, JSONValue] | None,
) -> JSONObject:
    states = _in_progress_scan_states(collection_resume)
    state_rows: list[JSONObject] = []
    processed_total = 0
    for path_key, state in states.items():
        check_deadline()
        phase = state.get("phase")
        phase_text = _str_or_none(phase) or "unknown"
        processed_count = _state_processed_count(state)
        processed_total += processed_count
        state_rows.append(
            {
                "path": path_key,
                "phase": phase_text,
                "processed_functions_count": processed_count,
                "processed_functions_digest": _state_processed_digest(state),
            }
        )
    index_signature = _analysis_index_resume_signature(collection_resume)
    digest = hashlib.sha1(
        _canonical_json_text(
            {
                "in_progress": state_rows,
                "index_hydrated_paths_count": _analysis_index_resume_hydrated_count(
                    collection_resume
                ),
                "index_hydrated_paths_digest": _analysis_index_resume_hydrated_digest(
                    collection_resume
                ),
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
        "index_hydrated_paths_count": _analysis_index_resume_hydrated_count(
            collection_resume
        ),
        "index_hydrated_paths_digest": _analysis_index_resume_hydrated_digest(
            collection_resume
        ),
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
    previous_states = _in_progress_scan_states(previous_collection_resume)
    current_states = _in_progress_scan_states(collection_resume)
    current_completed_paths = _completed_path_set(collection_resume)
    prev_progress = _analysis_resume_progress(
        collection_resume=previous_collection_resume,
        total_files=total_files,
    )
    current_progress = _analysis_resume_progress(
        collection_resume=collection_resume,
        total_files=total_files,
    )
    added_processed = 0
    regressed_processed = 0
    unchanged_in_progress_paths = 0
    changed_in_progress_paths = 0
    seen_paths: set[str] = set()

    def _accumulate_progress(path_key: str) -> None:
        nonlocal added_processed
        nonlocal regressed_processed
        nonlocal unchanged_in_progress_paths
        nonlocal changed_in_progress_paths
        previous_state = previous_states.get(path_key)
        current_state = current_states.get(path_key)
        if (
            previous_state is not None
            and current_state is None
            and path_key in current_completed_paths
        ):
            # Moving a path from in-progress to completed is monotonic progress,
            # not a semantic regression in processed functions.
            changed_in_progress_paths += 1
            return
        previous_keys = (
            _state_processed_functions(previous_state) if previous_state is not None else set()
        )
        current_keys = (
            _state_processed_functions(current_state) if current_state is not None else set()
        )
        if previous_keys or current_keys:
            added = current_keys - previous_keys
            regressed = previous_keys - current_keys
            added_count = len(added)
            regressed_count = len(regressed)
        else:
            previous_count = (
                _state_processed_count(previous_state) if previous_state is not None else 0
            )
            current_count = (
                _state_processed_count(current_state) if current_state is not None else 0
            )
            added_count = max(0, current_count - previous_count)
            regressed_count = max(0, previous_count - current_count)
        added_processed += added_count
        regressed_processed += regressed_count
        if added_count == 0 and regressed_count == 0:
            unchanged_in_progress_paths += 1
        else:
            changed_in_progress_paths += 1

    for path_key in previous_states:
        check_deadline()
        seen_paths.add(path_key)
        _accumulate_progress(path_key)
    for path_key in current_states:
        check_deadline()
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        _accumulate_progress(path_key)
    completed_delta = max(
        0, current_progress["completed_files"] - prev_progress["completed_files"]
    )
    completed_regressed = max(
        0, prev_progress["completed_files"] - current_progress["completed_files"]
    )
    previous_hydrated_paths = _analysis_index_resume_hydrated_paths(
        previous_collection_resume
    )
    current_hydrated_paths = _analysis_index_resume_hydrated_paths(collection_resume)
    if previous_hydrated_paths or current_hydrated_paths:
        hydrated_delta = len(current_hydrated_paths - previous_hydrated_paths)
        hydrated_regressed = len(previous_hydrated_paths - current_hydrated_paths)
    else:
        previous_hydrated_count = _analysis_index_resume_hydrated_count(
            previous_collection_resume
        )
        current_hydrated_count = _analysis_index_resume_hydrated_count(collection_resume)
        hydrated_delta = max(0, current_hydrated_count - previous_hydrated_count)
        hydrated_regressed = max(0, previous_hydrated_count - current_hydrated_count)
    cumulative_new = added_processed
    cumulative_completed_delta = completed_delta
    cumulative_hydrated_delta = hydrated_delta
    cumulative_regressed = regressed_processed + completed_regressed + hydrated_regressed
    cumulative_mapping = _json_mapping_or_none(cumulative)
    if cumulative_mapping is not None:
        raw_cumulative_new = _int_or_none(
            cumulative_mapping.get("cumulative_new_processed_functions")
        )
        raw_cumulative_completed = _int_or_none(
            cumulative_mapping.get("cumulative_completed_files_delta")
        )
        raw_cumulative_hydrated = _int_or_none(
            cumulative_mapping.get("cumulative_hydrated_paths_delta")
        )
        raw_cumulative_regressed = _int_or_none(
            cumulative_mapping.get("cumulative_regressed_functions")
        )
        if raw_cumulative_new is not None:
            cumulative_new += max(0, raw_cumulative_new)
        if raw_cumulative_completed is not None:
            cumulative_completed_delta += max(0, raw_cumulative_completed)
        if raw_cumulative_hydrated is not None:
            cumulative_hydrated_delta += max(0, raw_cumulative_hydrated)
        if raw_cumulative_regressed is not None:
            cumulative_regressed += max(0, raw_cumulative_regressed)
    current_witness = _collection_semantic_witness(collection_resume=collection_resume)
    previous_resume_mapping = _json_mapping_or_none(previous_collection_resume)
    previous_witness = (
        _collection_semantic_witness(collection_resume=previous_resume_mapping)
        if previous_resume_mapping is not None
        else {"witness_digest": None}
    )
    substantive_progress = (
        (
            cumulative_new > 0
            or cumulative_completed_delta > 0
            or cumulative_hydrated_delta > 0
        )
        and cumulative_regressed == 0
    )
    return {
        "current_witness_digest": current_witness.get("witness_digest"),
        "previous_witness_digest": previous_witness.get("witness_digest"),
        "new_processed_functions_count": added_processed,
        "regressed_processed_functions_count": regressed_processed,
        "completed_files_delta": completed_delta,
        "completed_files_regressed": completed_regressed,
        "hydrated_paths_delta": hydrated_delta,
        "hydrated_paths_regressed": hydrated_regressed,
        "changed_in_progress_paths": changed_in_progress_paths,
        "unchanged_in_progress_paths": unchanged_in_progress_paths,
        "cumulative_new_processed_functions": cumulative_new,
        "cumulative_completed_files_delta": cumulative_completed_delta,
        "cumulative_hydrated_paths_delta": cumulative_hydrated_delta,
        "cumulative_regressed_functions": cumulative_regressed,
        "monotonic_progress": cumulative_regressed == 0,
        "substantive_progress": substantive_progress,
    }
