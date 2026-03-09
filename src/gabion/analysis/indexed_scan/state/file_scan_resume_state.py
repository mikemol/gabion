from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, cast


@dataclass(frozen=True)
class FileScanResumeStateLoadDeps:
    empty_state_fn: Callable[[], tuple[object, ...]]
    payload_with_phase_fn: Callable[..., object]
    mapping_sections_fn: Callable[..., object]
    load_resume_map_fn: Callable[..., object]
    deserialize_param_use_map_fn: Callable[..., object]
    mapping_or_none_fn: Callable[..., object]
    deserialize_call_args_list_fn: Callable[..., object]
    sequence_or_none_fn: Callable[..., object]
    str_list_from_sequence_fn: Callable[..., object]
    deserialize_param_spans_for_resume_fn: Callable[..., object]
    str_tuple_from_sequence_fn: Callable[..., object]
    deadline_loop_iter_fn: Callable[..., object]
    iter_valid_key_entries_fn: Callable[..., object]


@dataclass(frozen=True)
class FileScanResumeStateSerializeDeps:
    sort_once_fn: Callable[..., object]
    check_deadline_fn: Callable[[], None]
    serialize_param_use_map_fn: Callable[..., object]
    serialize_call_args_list_fn: Callable[..., object]
    serialize_param_spans_for_resume_fn: Callable[..., object]


def serialize_file_scan_resume_state(
    *,
    fn_use: Mapping[str, Mapping[str, object]],
    fn_calls: Mapping[str, object],
    fn_param_orders: Mapping[str, object],
    fn_param_spans: Mapping[str, Mapping[str, tuple[int, int, int, int]]],
    fn_names: Mapping[str, str],
    fn_lexical_scopes: Mapping[str, object],
    fn_class_names: Mapping[str, object],
    opaque_callees: set[str],
    deps: FileScanResumeStateSerializeDeps,
):
    fn_use_payload: dict[str, object] = {}
    fn_calls_payload: dict[str, object] = {}
    fn_param_orders_payload: dict[str, object] = {}
    fn_param_spans_payload: dict[str, object] = {}
    fn_names_payload: dict[str, object] = {}
    fn_lexical_scopes_payload: dict[str, object] = {}
    fn_class_names_payload: dict[str, object] = {}
    for fn_key in deps.sort_once_fn(
        fn_use,
        source="gabion.analysis.dataflow_indexed_file_scan._serialize_file_scan_resume_state.site_1",
    ):
        deps.check_deadline_fn()
        fn_use_payload[fn_key] = deps.serialize_param_use_map_fn(fn_use[fn_key])
    for fn_key in deps.sort_once_fn(
        fn_calls,
        source="gabion.analysis.dataflow_indexed_file_scan._serialize_file_scan_resume_state.site_2",
    ):
        deps.check_deadline_fn()
        fn_calls_payload[fn_key] = deps.serialize_call_args_list_fn(fn_calls[fn_key])
    for fn_key in deps.sort_once_fn(
        fn_param_orders,
        source="gabion.analysis.dataflow_indexed_file_scan._serialize_file_scan_resume_state.site_3",
    ):
        deps.check_deadline_fn()
        fn_param_orders_payload[fn_key] = list(fn_param_orders[fn_key])
    for fn_key in deps.sort_once_fn(
        fn_param_spans,
        source="gabion.analysis.dataflow_indexed_file_scan._serialize_file_scan_resume_state.site_4",
    ):
        deps.check_deadline_fn()
        fn_param_spans_payload[fn_key] = deps.serialize_param_spans_for_resume_fn(
            {fn_key: dict(fn_param_spans[fn_key])}
        ).get(fn_key, {})
    for fn_key in deps.sort_once_fn(
        fn_names,
        source="gabion.analysis.dataflow_indexed_file_scan._serialize_file_scan_resume_state.site_5",
    ):
        deps.check_deadline_fn()
        fn_names_payload[fn_key] = fn_names[fn_key]
    for fn_key in deps.sort_once_fn(
        fn_lexical_scopes,
        source="gabion.analysis.dataflow_indexed_file_scan._serialize_file_scan_resume_state.site_6",
    ):
        deps.check_deadline_fn()
        fn_lexical_scopes_payload[fn_key] = list(fn_lexical_scopes[fn_key])
    for fn_key in deps.sort_once_fn(
        fn_class_names,
        source="gabion.analysis.dataflow_indexed_file_scan._serialize_file_scan_resume_state.site_7",
    ):
        deps.check_deadline_fn()
        fn_class_names_payload[fn_key] = fn_class_names[fn_key]
    return {
        "phase": "function_scan",
        "fn_use": fn_use_payload,
        "fn_calls": fn_calls_payload,
        "fn_param_orders": fn_param_orders_payload,
        "fn_param_spans": fn_param_spans_payload,
        "fn_names": fn_names_payload,
        "fn_lexical_scopes": fn_lexical_scopes_payload,
        "fn_class_names": fn_class_names_payload,
        "opaque_callees": deps.sort_once_fn(
            opaque_callees,
            source="gabion.analysis.dataflow_indexed_file_scan._serialize_file_scan_resume_state.site_8",
        ),
        "processed_functions": deps.sort_once_fn(
            fn_use.keys(),
            source="gabion.analysis.dataflow_indexed_file_scan._serialize_file_scan_resume_state.site_9",
        ),
    }


def load_file_scan_resume_state(
    *,
    payload,
    valid_fn_keys: set[str],
    deps: FileScanResumeStateLoadDeps,
):
    (
        fn_use,
        fn_calls,
        fn_param_orders,
        fn_param_spans,
        fn_names,
        fn_lexical_scopes,
        fn_class_names,
        opaque_callees,
    ) = deps.empty_state_fn()
    payload = deps.payload_with_phase_fn(payload, phase="function_scan")
    if payload is None:
        return deps.empty_state_fn()
    sections = deps.mapping_sections_fn(
        payload,
        section_keys=(
            "fn_use",
            "fn_calls",
            "fn_param_orders",
            "fn_param_spans",
            "fn_names",
            "fn_lexical_scopes",
            "fn_class_names",
        ),
    )
    if sections is None:
        return deps.empty_state_fn()
    (
        raw_use,
        raw_calls,
        raw_param_orders,
        raw_param_spans,
        raw_names,
        raw_scopes,
        raw_class_names,
    ) = sections

    def _iter_deserialized_param_use(raw_value):
        raw_mapping = deps.mapping_or_none_fn(raw_value)
        if raw_mapping is not None:
            yield deps.deserialize_param_use_map_fn(raw_mapping)

    def _iter_deserialized_call_args(raw_value):
        raw_sequence = deps.sequence_or_none_fn(raw_value)
        if raw_sequence is not None:
            yield deps.deserialize_call_args_list_fn(raw_sequence)

    def _iter_param_order_list(raw_value):
        raw_sequence = deps.sequence_or_none_fn(raw_value)
        if raw_sequence is not None:
            yield deps.str_list_from_sequence_fn(raw_value)

    def _iter_param_spans(raw_value):
        raw_mapping = deps.mapping_or_none_fn(raw_value)
        if raw_mapping is not None:
            yield deps.deserialize_param_spans_for_resume_fn({"_": raw_mapping}).get("_", {})

    def _iter_string(raw_value):
        match raw_value:
            case str() as raw_text:
                yield raw_text

    def _iter_scope_tuple(raw_value):
        raw_sequence = deps.sequence_or_none_fn(raw_value)
        if raw_sequence is not None:
            yield deps.str_tuple_from_sequence_fn(raw_value)

    fn_use = deps.load_resume_map_fn(
        payload=raw_use,
        valid_keys=valid_fn_keys,
        parser=_iter_deserialized_param_use,
    )
    fn_calls = deps.load_resume_map_fn(
        payload=raw_calls,
        valid_keys=valid_fn_keys,
        parser=_iter_deserialized_call_args,
    )
    fn_param_orders = deps.load_resume_map_fn(
        payload=raw_param_orders,
        valid_keys=valid_fn_keys,
        parser=_iter_param_order_list,
    )
    fn_param_spans = deps.load_resume_map_fn(
        payload=raw_param_spans,
        valid_keys=valid_fn_keys,
        parser=_iter_param_spans,
    )

    fn_names = deps.load_resume_map_fn(
        payload=raw_names,
        valid_keys=valid_fn_keys,
        parser=_iter_string,
    )
    fn_lexical_scopes = deps.load_resume_map_fn(
        payload=raw_scopes,
        valid_keys=valid_fn_keys,
        parser=_iter_scope_tuple,
    )
    fn_class_names = {}
    for fn_key, raw_value in deps.deadline_loop_iter_fn(
        deps.iter_valid_key_entries_fn(
            payload=raw_class_names,
            valid_keys=valid_fn_keys,
        )
    ):
        match raw_value:
            case None:
                fn_class_names[fn_key] = None
            case str() as class_name:
                fn_class_names[fn_key] = class_name
    raw_opaque = payload.get("opaque_callees")
    raw_opaque_entries = deps.sequence_or_none_fn(raw_opaque)
    if raw_opaque_entries is not None:
        for entry in deps.deadline_loop_iter_fn(raw_opaque_entries):
            match entry:
                case str() as opaque_callee if opaque_callee in valid_fn_keys:
                    opaque_callees.add(opaque_callee)
    return (
        fn_use,
        fn_calls,
        fn_param_orders,
        fn_param_spans,
        fn_names,
        fn_lexical_scopes,
        fn_class_names,
        opaque_callees,
    )
