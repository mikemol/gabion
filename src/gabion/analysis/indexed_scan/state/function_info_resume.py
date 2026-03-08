from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence, cast

from gabion.analysis.foundation.json_types import JSONObject, JSONValue


@dataclass(frozen=True)
class SerializeFunctionInfoForResumeDeps:
    sort_once_fn: Callable[..., list[str]]
    serialize_call_args_list_fn: Callable[..., list[JSONObject]]


def serialize_function_info_for_resume(
    info,
    *,
    deps: SerializeFunctionInfoForResumeDeps,
) -> JSONObject:
    payload: JSONObject = {
        "name": info.name,
        "qual": info.qual,
        "path": str(info.path),
        "params": list(info.params),
        "annots": {
            param: info.annots[param]
            for param in deps.sort_once_fn(
                info.annots,
                source="gabion.analysis.indexed_scan.function_info_resume.serialize_function_info_for_resume.annots",
            )
        },
        "calls": deps.serialize_call_args_list_fn(info.calls),
        "unused_params": deps.sort_once_fn(
            info.unused_params,
            source="gabion.analysis.indexed_scan.function_info_resume.serialize_function_info_for_resume.unused_params",
        ),
        "unknown_key_carriers": deps.sort_once_fn(
            info.unknown_key_carriers,
            source="gabion.analysis.indexed_scan.function_info_resume.serialize_function_info_for_resume.unknown_key_carriers",
        ),
        "defaults": deps.sort_once_fn(
            info.defaults,
            source="gabion.analysis.indexed_scan.function_info_resume.serialize_function_info_for_resume.defaults",
        ),
        "transparent": bool(info.transparent),
        "class_name": info.class_name,
        "scope": list(info.scope),
        "lexical_scope": list(info.lexical_scope),
        "decision_params": deps.sort_once_fn(
            info.decision_params,
            source="gabion.analysis.indexed_scan.function_info_resume.serialize_function_info_for_resume.decision_params",
        ),
        "decision_surface_reasons": {
            param: deps.sort_once_fn(
                info.decision_surface_reasons.get(param, set()),
                source="gabion.analysis.indexed_scan.function_info_resume.serialize_function_info_for_resume.decision_surface_reasons",
            )
            for param in deps.sort_once_fn(
                info.decision_surface_reasons,
                source="gabion.analysis.indexed_scan.function_info_resume.serialize_function_info_for_resume.decision_surface_reason_keys",
            )
        },
        "value_decision_params": deps.sort_once_fn(
            info.value_decision_params,
            source="gabion.analysis.indexed_scan.function_info_resume.serialize_function_info_for_resume.value_decision_params",
        ),
        "value_decision_reasons": deps.sort_once_fn(
            info.value_decision_reasons,
            source="gabion.analysis.indexed_scan.function_info_resume.serialize_function_info_for_resume.value_decision_reasons",
        ),
        "positional_params": list(info.positional_params),
        "kwonly_params": list(info.kwonly_params),
        "vararg": info.vararg,
        "kwarg": info.kwarg,
        "param_spans": {
            param: [int(value) for value in info.param_spans[param]]
            for param in deps.sort_once_fn(
                info.param_spans,
                source="gabion.analysis.indexed_scan.function_info_resume.serialize_function_info_for_resume.param_spans",
            )
        },
    }
    if info.function_span is not None:
        payload["function_span"] = [int(value) for value in info.function_span]
    return payload


@dataclass(frozen=True)
class DeserializeFunctionInfoForResumeDeps:
    sequence_or_none_fn: Callable[..., object]
    str_list_from_sequence_fn: Callable[..., list[str]]
    str_or_none_fn: Callable[..., object]
    mapping_or_empty_fn: Callable[..., Mapping[str, JSONValue]]
    check_deadline_fn: Callable[[], None]
    deserialize_call_args_list_fn: Callable[..., list[object]]
    str_set_from_sequence_fn: Callable[..., set[str]]
    str_tuple_from_sequence_fn: Callable[..., tuple[str, ...]]
    int_tuple4_or_none_fn: Callable[..., object]
    function_info_ctor: Callable[..., object]


def deserialize_function_info_for_resume(
    payload: Mapping[str, JSONValue],
    *,
    allowed_paths: Mapping[str, Path],
    deps: DeserializeFunctionInfoForResumeDeps,
) -> object:
    name = deps.str_or_none_fn(payload.get("name"))
    qual = deps.str_or_none_fn(payload.get("qual"))
    path_key = deps.str_or_none_fn(payload.get("path"))
    raw_params = payload.get("params")
    params_payload_raw = deps.sequence_or_none_fn(raw_params)
    path = allowed_paths.get(path_key) if path_key is not None else None
    if (
        name is not None
        and qual is not None
        and path is not None
        and params_payload_raw is not None
    ):
        params_payload = cast(Sequence[JSONValue], params_payload_raw)
        params = deps.str_list_from_sequence_fn(params_payload)
        raw_annots = payload.get("annots")
        annots: dict[str, JSONValue] = {}
        for param, annot in deps.mapping_or_empty_fn(raw_annots).items():
            deps.check_deadline_fn()
            param_text = deps.str_or_none_fn(param)
            annot_text = deps.str_or_none_fn(annot)
            if param_text is not None and (annot is None or annot_text is not None):
                annots[param_text] = annot if annot is None else annot_text
        raw_calls = payload.get("calls")
        raw_calls_payload = cast(
            Sequence[JSONValue],
            deps.sequence_or_none_fn(raw_calls) or (),
        )
        calls = deps.deserialize_call_args_list_fn(raw_calls_payload)
        unused_params = deps.str_set_from_sequence_fn(payload.get("unused_params"))
        unknown_key_carriers = deps.str_set_from_sequence_fn(payload.get("unknown_key_carriers"))
        defaults = deps.str_set_from_sequence_fn(payload.get("defaults"))
        class_name = deps.str_or_none_fn(payload.get("class_name"))
        scope = deps.str_tuple_from_sequence_fn(payload.get("scope"))
        lexical_scope = deps.str_tuple_from_sequence_fn(payload.get("lexical_scope"))
        decision_params = deps.str_set_from_sequence_fn(payload.get("decision_params"))
        decision_surface_reasons: dict[str, set[str]] = {}
        for param, raw_reasons in deps.mapping_or_empty_fn(
            payload.get("decision_surface_reasons")
        ).items():
            deps.check_deadline_fn()
            param_text = deps.str_or_none_fn(param)
            if param_text is not None:
                reasons = deps.str_set_from_sequence_fn(raw_reasons)
                if reasons:
                    decision_surface_reasons[param_text] = reasons
        value_decision_params = deps.str_set_from_sequence_fn(payload.get("value_decision_params"))
        value_decision_reasons = deps.str_set_from_sequence_fn(payload.get("value_decision_reasons"))
        positional_params = deps.str_tuple_from_sequence_fn(payload.get("positional_params"))
        kwonly_params = deps.str_tuple_from_sequence_fn(payload.get("kwonly_params"))
        vararg = deps.str_or_none_fn(payload.get("vararg"))
        kwarg = deps.str_or_none_fn(payload.get("kwarg"))
        param_spans: dict[str, tuple[int, int, int, int]] = {}
        for param, raw_span in deps.mapping_or_empty_fn(payload.get("param_spans")).items():
            deps.check_deadline_fn()
            param_text = deps.str_or_none_fn(param)
            if param_text is not None:
                span = deps.int_tuple4_or_none_fn(raw_span)
                if span is not None:
                    param_spans[param_text] = cast(tuple[int, int, int, int], span)
        function_span = deps.int_tuple4_or_none_fn(payload.get("function_span"))
        return deps.function_info_ctor(
            name=name,
            qual=qual,
            path=path,
            params=params,
            annots=annots,
            calls=calls,
            unused_params=unused_params,
            unknown_key_carriers=unknown_key_carriers,
            defaults=defaults,
            transparent=bool(payload.get("transparent", True)),
            class_name=class_name,
            scope=scope,
            lexical_scope=lexical_scope,
            decision_params=decision_params,
            decision_surface_reasons=decision_surface_reasons,
            value_decision_params=value_decision_params,
            value_decision_reasons=value_decision_reasons,
            positional_params=positional_params,
            kwonly_params=kwonly_params,
            vararg=vararg,
            kwarg=kwarg,
            param_spans=param_spans,
            function_span=function_span,
        )
    return None
