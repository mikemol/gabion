from __future__ import annotations

from functools import singledispatch
from typing import Callable, Iterator, Mapping, Sequence

from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.commands import boundary_order
from gabion.invariants import never
from gabion.json_types import JSONObject, JSONValue
from gabion.server_core import command_orchestrator_primitives as orchestrator_primitives
from gabion.server_core import command_orchestrator_progress as progress_dispatch
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

_NONE_TYPE = type(None)

_report_projection_phase_rank_optional = progress_dispatch.report_projection_phase_rank_optional
_in_progress_scan_states = progress_dispatch.in_progress_scan_states


@singledispatch
def _list_optional(value: object) -> list[object] | None:
    never("unregistered runtime type", value_type=type(value).__name__)


@_list_optional.register(list)
def _sd_reg_1(value: list[object]) -> list[object] | None:
    return value


def _list_none(value: object) -> list[object] | None:
    _ = value
    return None


_list_optional.register(tuple)(_list_none)
_list_optional.register(set)(_list_none)
_list_optional.register(dict)(_list_none)
_list_optional.register(str)(_list_none)
_list_optional.register(int)(_list_none)
_list_optional.register(float)(_list_none)
_list_optional.register(bool)(_list_none)
_list_optional.register(_NONE_TYPE)(_list_none)


def _str_is_present(value: str | None) -> bool:
    return value is not None


def _iter_present_text_values(entries: Sequence[object]) -> Iterator[str]:
    yield from filter(_str_is_present, map(_str_optional, entries))


def _string_entries(value: object) -> list[str]:
    entries = _non_string_sequence_optional(value) or ()
    return list(_iter_present_text_values(entries))


@singledispatch
def _analysis_manifest_config_value(value: object) -> JSONValue:
    never("unregistered runtime type", value_type=type(value).__name__)


@_analysis_manifest_config_value.register(bool)
def _sd_reg_2(value: bool) -> JSONValue:
    return value


@_analysis_manifest_config_value.register(int)
def _sd_reg_3(value: int) -> JSONValue:
    return value


@_analysis_manifest_config_value.register(str)
def _sd_reg_4(value: str) -> JSONValue:
    return value


@_analysis_manifest_config_value.register(list)
def _sd_reg_5(value: list[object]) -> JSONValue:
    return list(map(str, value))


def _analysis_manifest_invalid_none(value: object) -> JSONValue:
    _ = value
    never("analysis manifest config field missing")


def _analysis_manifest_invalid_mapping(value: object) -> JSONValue:
    _ = value
    never("invalid analysis manifest config mapping")


_analysis_manifest_config_value.register(_NONE_TYPE)(_analysis_manifest_invalid_none)
_analysis_manifest_config_value.register(dict)(_analysis_manifest_invalid_mapping)


def _required_mapping(value: object, *, field_name: str) -> Mapping[str, JSONValue]:
    mapping_value = _json_mapping_optional(value)
    if mapping_value is None:
        never("missing required analysis witness mapping", field=field_name)
    return mapping_value


def _required_str(value: object, *, field_name: str) -> str:
    value_text = _str_optional(value)
    if value_text is None:
        never("missing required analysis witness text field", field=field_name)
    return value_text


def _required_bool(value: object, *, field_name: str) -> bool:
    flag = _bool_optional(value)
    if flag is None:
        never("missing required analysis witness bool field", field=field_name)
    return flag


def _manifest_entry_from_raw(raw_entry: object) -> JSONObject:
    entry = _required_mapping(raw_entry, field_name="files[]")
    manifest_entry: JSONObject = {"path": _required_str(entry.get("path"), field_name="path")}
    maybe_missing = _bool_optional(entry.get("missing"))
    if maybe_missing is not None:
        manifest_entry["missing"] = maybe_missing
    maybe_size = _int_optional(entry.get("size"))
    if maybe_size is not None:
        manifest_entry["size"] = maybe_size
    maybe_content_sha1 = _str_optional(entry.get("content_sha1"))
    if maybe_content_sha1:
        manifest_entry["content_sha1"] = maybe_content_sha1
    return manifest_entry


def _iter_manifest_file_entries(raw_entries: Sequence[object]) -> Iterator[JSONObject]:
    for raw_entry in raw_entries:
        check_deadline()
        yield _manifest_entry_from_raw(raw_entry)


_MANIFEST_CONFIG_KEYS: tuple[str, ...] = (
    "exclude_dirs",
    "ignore_params",
    "strictness",
    "external_filter",
    "transparent_decorators",
)


def _iter_manifest_config_items(
    config_payload: Mapping[str, JSONValue],
) -> Iterator[tuple[str, JSONValue]]:
    for key in _MANIFEST_CONFIG_KEYS:
        check_deadline()
        yield key, _analysis_manifest_config_value(config_payload.get(key))


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


def _path_state_item(
    item: tuple[object, object],
) -> tuple[str | None, Mapping[str, JSONValue] | None]:
    raw_path, raw_state = item
    return _str_optional(raw_path), _json_mapping_optional(raw_state)


def _path_state_item_is_present(
    item: tuple[str | None, Mapping[str, JSONValue] | None],
) -> bool:
    path_value, state_value = item
    return path_value is not None and state_value is not None


def _coerce_path_state_item(
    item: tuple[str | None, Mapping[str, JSONValue] | None],
) -> tuple[str, Mapping[str, JSONValue]]:
    path_value, state_value = item
    if path_value is None or state_value is None:
        never("missing path/state after present-item filtering")
    return path_value, state_value


def _iter_present_path_state_items(
    mapping: Mapping[str, JSONValue],
) -> Iterator[tuple[str, Mapping[str, JSONValue]]]:
    yield from map(
        _coerce_path_state_item,
        filter(_path_state_item_is_present, map(_path_state_item, mapping.items())),
    )


def _next_ordered_path(previous_path: str | None, current_path: str) -> str:
    if previous_path is not None and previous_path > current_path:
        never(
            "in_progress_scan_by_path path order regression",
            previous_path=previous_path,
            current_path=current_path,
        )
    return current_path


def _in_progress_resume_payload_row(
    item: tuple[str, Mapping[str, JSONValue]],
) -> tuple[str, object]:
    path_value, state_mapping = item
    return path_value, _in_progress_scan_state_payload(state_mapping)


def _iter_ordered_in_progress_resume_payload_rows(
    mapping: Mapping[str, JSONValue],
) -> Iterator[tuple[str, object]]:
    previous_path: str | None = None
    for item in _iter_present_path_state_items(mapping):
        check_deadline()
        path_value, _state_mapping = item
        previous_path = _next_ordered_path(previous_path, path_value)
        yield _in_progress_resume_payload_row(item)


def _dimension_row_optional(item: tuple[str, JSONValue]) -> tuple[str, dict[str, int]] | None:
    dim_name_text, raw_payload = item
    dim_payload = _json_mapping_optional(raw_payload)
    if dim_payload is None:
        return None
    raw_done = _non_negative_int_optional(dim_payload.get("done"))
    raw_total = _non_negative_int_optional(dim_payload.get("total"))
    if raw_done is None or raw_total is None:
        return None
    return dim_name_text, {"done": raw_done, "total": raw_total}


def _dimension_row_is_present(item: tuple[str, dict[str, int]] | None) -> bool:
    return item is not None


def _coerce_dimension_row(item: tuple[str, dict[str, int]] | None) -> tuple[str, dict[str, int]]:
    if item is None:
        never("missing dimension row after present-row filtering")
    return item


def _iter_dimension_rows(mapping: Mapping[str, JSONValue]) -> Iterator[tuple[str, dict[str, int]]]:
    yield from map(
        _coerce_dimension_row,
        filter(_dimension_row_is_present, map(_dimension_row_optional, _iter_text_key_items(mapping))),
    )


def _boundary_control_source_items(
    payload: Mapping[str, JSONValue],
) -> Iterator[tuple[str, JSONValue | None]]:
    for key in ("language", "ingest_profile"):
        yield key, payload.get(key)


def _boundary_control_optional(
    item: tuple[str, JSONValue | None],
) -> tuple[str, str] | None:
    key, raw_value = item
    if raw_value is None:
        return None
    raw_text = _str_optional(raw_value)
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
    return key, normalized_value


def _boundary_control_is_present(item: tuple[str, str] | None) -> bool:
    return item is not None


def _coerce_boundary_control(item: tuple[str, str] | None) -> tuple[str, str]:
    if item is None:
        never("missing boundary control item after present-item filtering")
    return item


def _iter_boundary_control_updates(
    payload: Mapping[str, JSONValue],
) -> Iterator[tuple[str, str]]:
    yield from map(
        _coerce_boundary_control,
        filter(_boundary_control_is_present, map(_boundary_control_optional, _boundary_control_source_items(payload))),
    )


def _in_progress_scan_state_payload(
    state: Mapping[str, JSONValue] | None,
) -> dict[str, object]:
    state_mapping = _json_mapping_default_empty(state)
    normalized_state: dict[str, object] = {}
    phase = _str_optional(state_mapping.get("phase"))
    if phase is not None:
        normalized_state["phase"] = phase
    processed_entries = _string_entries(state_mapping.get("processed_functions"))
    if processed_entries:
        normalized_state["processed_functions"] = processed_entries
    raw_processed_count = _non_negative_int_optional(
        state_mapping.get("processed_functions_count")
    )
    if raw_processed_count is not None:
        normalized_state["processed_functions_count"] = raw_processed_count
    processed_digest = _str_optional(state_mapping.get("processed_functions_digest"))
    if processed_digest is not None:
        normalized_state["processed_functions_digest"] = processed_digest
    function_count = _non_negative_int_optional(state_mapping.get("function_count"))
    if function_count is not None:
        normalized_state["function_count"] = function_count
    raw_fn_names = _json_mapping_default_empty(state_mapping.get("fn_names"))
    normalized_fn_names = dict(_iter_text_key_items(raw_fn_names))
    if normalized_fn_names:
        normalized_state["fn_names"] = normalized_fn_names
    return normalized_state


def _collection_resume_payload(
    collection_resume: Mapping[str, JSONValue] | None,
) -> JSONObject:
    resume_mapping = _json_mapping_default_empty(collection_resume)

    completed_paths = _string_entries(resume_mapping.get("completed_paths"))
    raw_in_progress = _json_mapping_default_empty(
        resume_mapping.get("in_progress_scan_by_path")
    )
    in_progress_scan_by_path = dict(
        _iter_ordered_in_progress_resume_payload_rows(raw_in_progress)
    )

    raw_index_resume = _json_mapping_optional(resume_mapping.get("analysis_index_resume"))
    analysis_index_resume: dict[str, object] | None = None
    if raw_index_resume is not None:
        hydrated_paths = _string_entries(raw_index_resume.get("hydrated_paths"))
        hydrated_paths_count = _non_negative_int_optional(
            raw_index_resume.get("hydrated_paths_count")
        )
        function_count = _non_negative_int_optional(raw_index_resume.get("function_count"))
        class_count = _non_negative_int_optional(raw_index_resume.get("class_count"))
        hydrated_paths_digest = _str_optional(raw_index_resume.get("hydrated_paths_digest"))
        phase = _str_optional(raw_index_resume.get("phase"))
        resume_digest = _str_optional(raw_index_resume.get("resume_digest"))
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
    phase_payload = _json_mapping_optional(phase_progress_v2)
    if phase_payload is None:
        return None
    raw_dimensions = _json_mapping_default_empty(phase_payload.get("dimensions"))
    dimensions = dict(_iter_dimension_rows(raw_dimensions))
    primary_done = _non_negative_int_optional(phase_payload.get("primary_done"))
    primary_total = _non_negative_int_optional(phase_payload.get("primary_total"))
    return {
        "primary_unit": _str_optional(phase_payload.get("primary_unit")) or "",
        "primary_done": primary_done,
        "primary_total": primary_total,
        "dimensions": dimensions,
    }


def _normalize_dataflow_boundary_controls(
    payload: dict[str, JSONValue],
) -> dict[str, JSONValue]:
    normalized_updates = dict(_iter_boundary_control_updates(payload))
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
) -> str:
    check_deadline()
    raw_files = _list_optional(input_witness.get("files"))
    if raw_files is None:
        never("missing required analysis witness sequence", field="files")
    manifest_files = list(_iter_manifest_file_entries(raw_files))

    config = _required_mapping(input_witness.get("config"), field_name="config")
    config_payload = dict(_iter_manifest_config_items(config))

    manifest: JSONObject = {
        "format_version": manifest_format_version,
        "root": _required_str(input_witness.get("root"), field_name="root"),
        "recursive": _required_bool(input_witness.get("recursive"), field_name="recursive"),
        "include_invariant_propositions": _required_bool(
            input_witness.get("include_invariant_propositions"),
            field_name="include_invariant_propositions",
        ),
        "include_wl_refinement": _required_bool(
            input_witness.get("include_wl_refinement"),
            field_name="include_wl_refinement",
        ),
        "config": config_payload,
        "files": manifest_files,
    }
    return digest_fn(manifest)


def _normalize_csv_or_iterable_names(value: object, *, strict: bool) -> list[str]:
    return orchestrator_primitives.normalize_csv_or_iterable_names(value, strict=strict)
