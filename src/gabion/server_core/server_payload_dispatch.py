from __future__ import annotations

from functools import singledispatch
from typing import Callable, Mapping, Sequence

from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.commands import boundary_order
from gabion.invariants import never
from gabion.json_types import JSONObject, JSONValue
from gabion.server_core import command_orchestrator_primitives as orchestrator_primitives
from gabion.server_core import command_orchestrator_progress as progress_dispatch

_NONE_TYPE = type(None)

_int_or_none = progress_dispatch._int_or_none
_non_negative_int_or_none = progress_dispatch._non_negative_int_or_none
_json_mapping_or_none = progress_dispatch._json_mapping_or_none
_json_mapping_or_empty = progress_dispatch._json_mapping_or_empty
_non_string_sequence_or_none = progress_dispatch._non_string_sequence_or_none
_str_or_none = progress_dispatch._str_or_none
_bool_or_none = progress_dispatch._bool_or_none
_float_or_none = progress_dispatch._float_or_none
_report_projection_phase_rank_or_none = progress_dispatch._report_projection_phase_rank_or_none
_in_progress_scan_states = progress_dispatch._in_progress_scan_states


@singledispatch
def _list_or_none(value: object) -> list[object] | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_list_or_none.register(list)
def _(value: list[object]) -> list[object] | None:
    return value


def _list_none(value: object) -> list[object] | None:
    _ = value
    return None


for _runtime_type in (tuple, set, dict, str, int, float, bool, _NONE_TYPE):
    _list_or_none.register(_runtime_type)(_list_none)


def _string_entries(value: object) -> list[str]:
    entries = _non_string_sequence_or_none(value)
    if entries is None:
        return []
    normalized: list[str] = []
    for item in entries:
        item_text = _str_or_none(item)
        if item_text is not None:
            normalized.append(item_text)
    return normalized


def _in_progress_scan_state_payload(
    state: Mapping[str, JSONValue] | None,
) -> dict[str, object]:
    state_mapping = _json_mapping_or_none(state)
    if state_mapping is None:
        return {}
    normalized_state: dict[str, object] = {}
    phase = _str_or_none(state_mapping.get("phase"))
    if phase is not None:
        normalized_state["phase"] = phase
    processed_entries = _string_entries(state_mapping.get("processed_functions"))
    if processed_entries:
        normalized_state["processed_functions"] = processed_entries
    raw_processed_count = _non_negative_int_or_none(
        state_mapping.get("processed_functions_count")
    )
    if raw_processed_count is not None:
        normalized_state["processed_functions_count"] = raw_processed_count
    processed_digest = _str_or_none(state_mapping.get("processed_functions_digest"))
    if processed_digest is not None:
        normalized_state["processed_functions_digest"] = processed_digest
    function_count = _non_negative_int_or_none(state_mapping.get("function_count"))
    if function_count is not None:
        normalized_state["function_count"] = function_count
    raw_fn_names = _json_mapping_or_none(state_mapping.get("fn_names"))
    if raw_fn_names is not None:
        normalized_fn_names: dict[str, object] = {}
        for name, value in raw_fn_names.items():
            name_text = _str_or_none(name)
            if name_text is not None:
                normalized_fn_names[name_text] = value
        normalized_state["fn_names"] = normalized_fn_names
    return normalized_state


def _collection_resume_payload(
    collection_resume: Mapping[str, JSONValue] | None,
) -> JSONObject:
    resume_mapping = _json_mapping_or_none(collection_resume)
    if resume_mapping is None:
        return {}

    completed_paths = _string_entries(resume_mapping.get("completed_paths"))
    raw_in_progress = _json_mapping_or_none(resume_mapping.get("in_progress_scan_by_path"))
    in_progress_scan_by_path: dict[str, object] = {}
    previous_path: str | None = None
    if raw_in_progress is not None:
        for raw_path, raw_state in raw_in_progress.items():
            check_deadline()
            path_text = _str_or_none(raw_path)
            state_mapping = _json_mapping_or_none(raw_state)
            if path_text is None or state_mapping is None:
                continue
            if previous_path is not None and previous_path > path_text:
                never(
                    "in_progress_scan_by_path path order regression",
                    previous_path=previous_path,
                    current_path=path_text,
                )
            previous_path = path_text
            in_progress_scan_by_path[path_text] = _in_progress_scan_state_payload(
                state_mapping
            )

    raw_index_resume = _json_mapping_or_none(resume_mapping.get("analysis_index_resume"))
    analysis_index_resume: dict[str, object] | None = None
    if raw_index_resume is not None:
        hydrated_paths = _string_entries(raw_index_resume.get("hydrated_paths"))
        hydrated_paths_count = _non_negative_int_or_none(
            raw_index_resume.get("hydrated_paths_count")
        )
        function_count = _non_negative_int_or_none(raw_index_resume.get("function_count"))
        class_count = _non_negative_int_or_none(raw_index_resume.get("class_count"))
        hydrated_paths_digest = _str_or_none(raw_index_resume.get("hydrated_paths_digest"))
        phase = _str_or_none(raw_index_resume.get("phase"))
        resume_digest = _str_or_none(raw_index_resume.get("resume_digest"))
        analysis_index_resume = {
            "hydrated_paths": hydrated_paths,
            "hydrated_paths_count": hydrated_paths_count if hydrated_paths_count is not None else 0,
            "hydrated_paths_digest": hydrated_paths_digest or "",
            "function_count": function_count if function_count is not None else 0,
            "class_count": class_count if class_count is not None else 0,
            "phase": phase or "",
            "resume_digest": resume_digest or "",
        }

    return {
        "completed_paths": completed_paths,
        "in_progress_scan_by_path": in_progress_scan_by_path,
        "analysis_index_resume": analysis_index_resume,
    }


def _phase_progress_payload(
    phase_progress_v2: Mapping[str, JSONValue] | None,
) -> JSONObject | None:
    phase_payload = _json_mapping_or_none(phase_progress_v2)
    if phase_payload is None:
        return None
    raw_dimensions = _json_mapping_or_none(phase_payload.get("dimensions"))
    dimensions: dict[str, dict[str, int]] = {}
    if raw_dimensions is not None:
        for dim_name, raw_payload in raw_dimensions.items():
            dim_name_text = _str_or_none(dim_name)
            dim_payload = _json_mapping_or_none(raw_payload)
            if dim_name_text is None or dim_payload is None:
                continue
            raw_done = _non_negative_int_or_none(dim_payload.get("done"))
            raw_total = _non_negative_int_or_none(dim_payload.get("total"))
            if raw_done is not None and raw_total is not None:
                dimensions[dim_name_text] = {"done": raw_done, "total": raw_total}
    primary_done = _non_negative_int_or_none(phase_payload.get("primary_done"))
    primary_total = _non_negative_int_or_none(phase_payload.get("primary_total"))
    return {
        "primary_unit": _str_or_none(phase_payload.get("primary_unit")) or "",
        "primary_done": primary_done,
        "primary_total": primary_total,
        "dimensions": dimensions,
    }


def _normalize_dataflow_boundary_controls(
    payload: dict[str, JSONValue],
) -> dict[str, JSONValue]:
    normalized_updates: dict[str, JSONValue] = {}
    for key in ("language", "ingest_profile"):
        raw_value = payload.get(key)
        if raw_value is None:
            continue
        raw_text = _str_or_none(raw_value)
        if raw_text is None:
            never(  # pragma: no cover - invariant sink
                "invalid dataflow boundary control type",
                control=key,
                value_type=type(raw_value).__name__,
            )
        normalized_value = raw_text.strip().lower()
        if not normalized_value:
            never(  # pragma: no cover - invariant sink
                "empty dataflow boundary control",
                control=key,
            )
        normalized_updates[key] = normalized_value
    if not normalized_updates:
        return payload
    return boundary_order.apply_boundary_updates_once(
        payload,
        normalized_updates,
        source="server._normalize_dataflow_boundary_controls",
    )


def _analysis_manifest_digest_from_witness(
    input_witness: JSONObject,
    *,
    manifest_format_version: int,
    digest_fn: Callable[[JSONObject], str],
) -> str | None:
    check_deadline()
    files = _list_or_none(input_witness.get("files"))
    if files is None:
        return None
    manifest_files: list[JSONObject] = []
    for raw_entry in files:
        check_deadline()
        entry = _json_mapping_or_none(raw_entry)
        if entry is None:
            return None
        path_value = _str_or_none(entry.get("path"))
        if path_value is None:
            return None
        manifest_entry: JSONObject = {"path": path_value}
        missing_value = _bool_or_none(entry.get("missing"))
        if missing_value is not None:
            manifest_entry["missing"] = missing_value
        size_value = _int_or_none(entry.get("size"))
        if size_value is not None:
            manifest_entry["size"] = size_value
        content_sha1_value = _str_or_none(entry.get("content_sha1"))
        if content_sha1_value:
            manifest_entry["content_sha1"] = content_sha1_value
        manifest_files.append(manifest_entry)

    config = _json_mapping_or_none(input_witness.get("config"))
    if config is None:
        return None
    config_payload: JSONObject = {}
    for key in (
        "exclude_dirs",
        "ignore_params",
        "strictness",
        "external_filter",
        "transparent_decorators",
    ):
        check_deadline()
        value = config.get(key)
        list_value = _list_or_none(value)
        if list_value is not None:
            config_payload[key] = [str(item) for item in list_value]
            continue
        bool_value = _bool_or_none(value)
        if bool_value is not None:
            config_payload[key] = bool_value
            continue
        int_value = _int_or_none(value)
        if int_value is not None:
            config_payload[key] = int_value
            continue
        str_value = _str_or_none(value)
        if str_value is not None:
            config_payload[key] = str_value
            continue
        return None

    root = _str_or_none(input_witness.get("root"))
    recursive = _bool_or_none(input_witness.get("recursive"))
    include_invariant_propositions = _bool_or_none(
        input_witness.get("include_invariant_propositions")
    )
    include_wl_refinement = _bool_or_none(input_witness.get("include_wl_refinement"))
    if (
        root is None
        or recursive is None
        or include_invariant_propositions is None
        or include_wl_refinement is None
    ):
        return None
    manifest: JSONObject = {
        "format_version": manifest_format_version,
        "root": root,
        "recursive": recursive,
        "include_invariant_propositions": include_invariant_propositions,
        "include_wl_refinement": include_wl_refinement,
        "config": config_payload,
        "files": manifest_files,
    }
    return digest_fn(manifest)


def _normalize_csv_or_iterable_names(value: object, *, strict: bool) -> list[str]:
    return orchestrator_primitives._normalize_csv_or_iterable_names(value, strict=strict)
