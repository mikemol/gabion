from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from gabion.analysis.dataflow.engine.dataflow_contracts import InvariantProposition
from gabion.analysis.foundation.json_types import JSONObject, JSONValue
from gabion.invariants import never
from gabion.order_contract import sort_once
from gabion.runtime_shape_dispatch import str_or_none

from gabion.analysis.foundation.resume_codec import (
    allowed_path_lookup, load_allowed_paths_from_sequence, mapping_or_none, mapping_payload, mapping_sections, payload_with_format, sequence_or_none)
from gabion.analysis.foundation.timeout_context import check_deadline


def analysis_collection_resume_path_key(path: Path) -> str:
    return str(path)


def serialize_groups_for_resume(
    groups: dict[str, list[set[str]]],
    *,
    check_deadline_fn: Callable[[], None] = check_deadline,
    sort_once_fn: Callable[..., object] = sort_once,
) -> dict[str, list[list[str]]]:
    payload: dict[str, list[list[str]]] = {}
    for fn_name in sort_once_fn(
        groups,
        source="indexed_scan.resume_state.serialize_groups_for_resume.fn_name",
    ):
        check_deadline_fn()
        bundles = groups[fn_name]
        normalized = [
            sort_once_fn(
                (str(param) for param in bundle),
                source="indexed_scan.resume_state.serialize_groups_for_resume.bundle",
            )
            for bundle in bundles
        ]
        normalized = sort_once_fn(
            normalized,
            source="indexed_scan.resume_state.serialize_groups_for_resume.normalized",
        )
        payload[fn_name] = [list(bundle) for bundle in normalized]
    return payload


def deserialize_groups_for_resume(
    payload: Mapping[str, JSONValue],
    *,
    check_deadline_fn: Callable[[], None] = check_deadline,
    sequence_or_none_fn: Callable[[JSONValue], object] = sequence_or_none,
) -> dict[str, list[set[str]]]:
    groups: dict[str, list[set[str]]] = {}
    for fn_name, raw_groups in payload.items():
        check_deadline_fn()
        normalized_fn_name = str_or_none(fn_name)
        if normalized_fn_name is not None:
            groups_payload = sequence_or_none_fn(raw_groups)
            if groups_payload is not None:
                normalized: list[set[str]] = []
                for raw_bundle in groups_payload:
                    check_deadline_fn()
                    bundle_payload = sequence_or_none_fn(raw_bundle)
                    if bundle_payload is not None:
                        normalized.append({str(param) for param in bundle_payload if str(param)})
                groups[normalized_fn_name] = normalized
    return groups


def serialize_param_spans_for_resume(
    spans: dict[str, dict[str, tuple[int, int, int, int]]],
    *,
    check_deadline_fn: Callable[[], None] = check_deadline,
    sort_once_fn: Callable[..., object] = sort_once,
) -> dict[str, dict[str, list[int]]]:
    payload: dict[str, dict[str, list[int]]] = {}
    for fn_name in sort_once_fn(
        spans,
        source="indexed_scan.resume_state.serialize_param_spans_for_resume.fn_name",
    ):
        check_deadline_fn()
        param_spans = spans[fn_name]
        payload[fn_name] = {}
        for param_name in sort_once_fn(
            param_spans,
            source="indexed_scan.resume_state.serialize_param_spans_for_resume.param_name",
        ):
            check_deadline_fn()
            span = param_spans[param_name]
            payload[fn_name][param_name] = [int(part) for part in span]
    return payload


def deserialize_param_spans_for_resume(
    payload: Mapping[str, JSONValue],
    *,
    check_deadline_fn: Callable[[], None] = check_deadline,
    sequence_or_none_fn: Callable[[JSONValue], object] = sequence_or_none,
) -> dict[str, dict[str, tuple[int, int, int, int]]]:
    spans: dict[str, dict[str, tuple[int, int, int, int]]] = {}
    for fn_name, raw_map in payload.items():
        check_deadline_fn()
        param_map = mapping_or_none(raw_map)
        normalized_fn_name = str_or_none(fn_name)
        if normalized_fn_name is not None and param_map is not None:
            fn_spans: dict[str, tuple[int, int, int, int]] = {}
            for param_name, raw_span in param_map.items():
                check_deadline_fn()
                span_parts = sequence_or_none_fn(raw_span)
                normalized_param_name = str_or_none(param_name)
                if normalized_param_name is not None and span_parts is not None and len(span_parts) == 4:
                    try:
                        start_line, start_column, end_line, end_column = (
                            int(part) for part in span_parts
                        )
                    except (TypeError, ValueError):
                        continue
                    fn_spans[normalized_param_name] = (
                        start_line,
                        start_column,
                        end_line,
                        end_column,
                    )
            spans[normalized_fn_name] = fn_spans
    return spans


def serialize_bundle_sites_for_resume(
    bundle_sites: dict[str, list[list[JSONObject]]],
    *,
    check_deadline_fn: Callable[[], None] = check_deadline,
    sort_once_fn: Callable[..., object] = sort_once,
    sequence_or_none_fn: Callable[[JSONValue], object] = sequence_or_none,
    mapping_or_none_fn: Callable[[JSONValue], object] = mapping_or_none,
) -> dict[str, list[list[JSONObject]]]:
    payload: dict[str, list[list[JSONObject]]] = {}
    for fn_name in sort_once_fn(
        bundle_sites,
        source="indexed_scan.resume_state.serialize_bundle_sites_for_resume.fn_name",
    ):
        check_deadline_fn()
        fn_sites = bundle_sites[fn_name]
        encoded_fn_sites: list[list[JSONObject]] = []
        for bundle in fn_sites:
            check_deadline_fn()
            encoded_bundle: list[JSONObject] = []
            bundle_entries = sequence_or_none_fn(cast(JSONValue, bundle))
            if bundle_entries is not None:
                for site in bundle_entries:
                    check_deadline_fn()
                    site_mapping = mapping_or_none_fn(site)
                    if site_mapping is not None:
                        encoded_bundle.append(
                            {str(key): site_mapping[key] for key in site_mapping}
                        )
                encoded_fn_sites.append(encoded_bundle)
        payload[fn_name] = encoded_fn_sites
    return payload


def deserialize_bundle_sites_for_resume(
    payload: Mapping[str, JSONValue],
    *,
    check_deadline_fn: Callable[[], None] = check_deadline,
    sequence_or_none_fn: Callable[[JSONValue], object] = sequence_or_none,
    mapping_or_none_fn: Callable[[JSONValue], object] = mapping_or_none,
) -> dict[str, list[list[JSONObject]]]:
    bundle_sites: dict[str, list[list[JSONObject]]] = {}
    for fn_name, raw_sites in payload.items():
        check_deadline_fn()
        site_groups = sequence_or_none_fn(raw_sites)
        normalized_fn_name = str_or_none(fn_name)
        if normalized_fn_name is not None and site_groups is not None:
            fn_sites: list[list[JSONObject]] = []
            for raw_bundle in site_groups:
                check_deadline_fn()
                bundle_entries = sequence_or_none_fn(raw_bundle)
                if bundle_entries is not None:
                    bundle: list[JSONObject] = []
                    for site in bundle_entries:
                        check_deadline_fn()
                        site_mapping = mapping_or_none_fn(site)
                        if site_mapping is not None:
                            bundle.append({str(key): site_mapping[key] for key in site_mapping})
                    fn_sites.append(bundle)
            bundle_sites[normalized_fn_name] = fn_sites
    return bundle_sites


def serialize_invariants_for_resume(
    invariants: Sequence[InvariantProposition],
    *,
    check_deadline_fn: Callable[[], None] = check_deadline,
    sort_once_fn: Callable[..., object] = sort_once,
) -> list[JSONObject]:
    payload: list[JSONObject] = []
    for proposition in sort_once_fn(
        invariants,
        key=lambda proposition: (
            proposition.form,
            proposition.terms,
            proposition.scope or "",
            proposition.source or "",
        ),
        source="indexed_scan.resume_state.serialize_invariants_for_resume",
    ):
        check_deadline_fn()
        payload.append(proposition.as_dict())
    return payload


def deserialize_invariants_for_resume(
    payload: Sequence[JSONValue],
    *,
    normalize_invariant_proposition_fn: Callable[..., InvariantProposition],
    check_deadline_fn: Callable[[], None] = check_deadline,
    mapping_or_none_fn: Callable[[JSONValue], object] = mapping_or_none,
    sequence_or_none_fn: Callable[[JSONValue], object] = sequence_or_none,
) -> list[InvariantProposition]:
    invariants: list[InvariantProposition] = []
    for entry in payload:
        check_deadline_fn()
        entry_mapping = mapping_or_none_fn(entry)
        if entry_mapping is not None:
            form = str_or_none(entry_mapping.get("form"))
            terms = sequence_or_none_fn(entry_mapping.get("terms"))
            if form is not None and terms is not None:
                normalized_terms: list[str] = []
                for term in terms:
                    check_deadline_fn()
                    normalized_term = str_or_none(term)
                    if normalized_term is not None:
                        normalized_terms.append(normalized_term)
                scope = str_or_none(entry_mapping.get("scope"))
                source = str_or_none(entry_mapping.get("source"))
                invariant_id = str_or_none(entry_mapping.get("invariant_id"))
                confidence_raw = entry_mapping.get("confidence")
                confidence = (
                    float(confidence_raw)
                    if type(confidence_raw) in {int, float}
                    else None
                )
                raw_evidence = entry_mapping.get("evidence_keys")
                evidence_keys: tuple[str, ...] = ()
                evidence_sequence = sequence_or_none_fn(raw_evidence)
                if evidence_sequence is not None:
                    evidence_keys = tuple(
                        str(item) for item in evidence_sequence if str(item).strip()
                    )
                normalized = normalize_invariant_proposition_fn(
                    InvariantProposition(
                        form=form,
                        terms=tuple(normalized_terms),
                        scope=scope,
                        source=source,
                        invariant_id=invariant_id,
                        confidence=confidence,
                        evidence_keys=evidence_keys,
                    ),
                    default_scope=scope or "",
                    default_source=source or "resume",
                )
                invariants.append(normalized)
    return invariants


def build_analysis_collection_resume_payload(
    *,
    groups_by_path: Mapping[Path, dict[str, list[set[str]]]],
    param_spans_by_path: Mapping[Path, dict[str, dict[str, tuple[int, int, int, int]]]],
    bundle_sites_by_path: Mapping[Path, dict[str, list[list[JSONObject]]]],
    invariant_propositions: Sequence[InvariantProposition],
    completed_paths: set[Path],
    in_progress_scan_by_path: Mapping[Path, JSONObject],
    analysis_index_resume = None,
    file_stage_timings_v1_by_path = None,
    format_version: int,
    path_key_fn: Callable[[Path], str] = analysis_collection_resume_path_key,
    check_deadline_fn: Callable[[], None] = check_deadline,
    sort_once_fn: Callable[..., object] = sort_once,
    serialize_groups_for_resume_fn: Callable[..., object] = serialize_groups_for_resume,
    serialize_param_spans_for_resume_fn: Callable[..., object] = serialize_param_spans_for_resume,
    serialize_bundle_sites_for_resume_fn: Callable[..., object] = serialize_bundle_sites_for_resume,
    serialize_invariants_for_resume_fn: Callable[..., object] = serialize_invariants_for_resume,
    mapping_or_none_fn: Callable[[JSONValue], object] = mapping_or_none,
    never_fn: Callable[..., None] = never,
) -> JSONObject:
    check_deadline_fn()
    groups_payload: JSONObject = {}
    spans_payload: JSONObject = {}
    sites_payload: JSONObject = {}
    in_progress_scan_payload: JSONObject = {}
    completed_keys = sort_once_fn(
        (path_key_fn(path) for path in completed_paths),
        source="indexed_scan.resume_state.build_analysis_collection_resume_payload.completed",
    )
    for path_key in completed_keys:
        check_deadline_fn()
        path = Path(path_key)
        groups_payload[path_key] = serialize_groups_for_resume_fn(
            groups_by_path.get(path, {}),
        )
        spans_payload[path_key] = serialize_param_spans_for_resume_fn(
            param_spans_by_path.get(path, {}),
        )
        sites_payload[path_key] = serialize_bundle_sites_for_resume_fn(
            bundle_sites_by_path.get(path, {}),
        )
    previous_path_key = None
    for path in in_progress_scan_by_path:
        check_deadline_fn()
        path_key = path_key_fn(path)
        if previous_path_key is not None and previous_path_key > path_key:
            never_fn(
                "in_progress_scan_by_path path order regression",
                previous_path=previous_path_key,
                current_path=path_key,
            )
        previous_path_key = path_key
        in_progress_scan_payload[path_key] = {
            str(key): in_progress_scan_by_path[path][key]
            for key in in_progress_scan_by_path[path]
        }
    payload: JSONObject = {
        "format_version": format_version,
        "completed_paths": completed_keys,
        "groups_by_path": groups_payload,
        "param_spans_by_path": spans_payload,
        "bundle_sites_by_path": sites_payload,
        "in_progress_scan_by_path": in_progress_scan_payload,
        "invariant_propositions": serialize_invariants_for_resume_fn(
            invariant_propositions,
        ),
    }
    if file_stage_timings_v1_by_path:
        payload["file_stage_timings_v1_by_path"] = {
            path_key_fn(path): {
                str(key): value
                for key, value in file_stage_timings_v1_by_path[path].items()
            }
            for path in sort_once_fn(
                file_stage_timings_v1_by_path,
                key=path_key_fn,
                source="indexed_scan.resume_state.build_analysis_collection_resume_payload.timings",
            )
        }
    analysis_index_resume_mapping = mapping_or_none_fn(analysis_index_resume)
    if analysis_index_resume_mapping is not None:
        payload["analysis_index_resume"] = {
            str(key): analysis_index_resume_mapping[key]
            for key in analysis_index_resume_mapping
        }
    return payload


def empty_analysis_collection_resume_payload() -> tuple[dict, dict, dict, list, set, dict, object]:
    return ({}, {}, {}, [], set(), {}, None)


def load_analysis_collection_resume_payload(
    *,
    payload,
    file_paths: Sequence[Path],
    include_invariant_propositions: bool,
    format_version: int,
    path_key_fn: Callable[[Path], str] = analysis_collection_resume_path_key,
    deserialize_groups_for_resume_fn: Callable[..., object] = deserialize_groups_for_resume,
    deserialize_param_spans_for_resume_fn: Callable[..., object] = deserialize_param_spans_for_resume,
    deserialize_bundle_sites_for_resume_fn: Callable[..., object] = deserialize_bundle_sites_for_resume,
    deserialize_invariants_for_resume_fn: Callable[..., object] = deserialize_invariants_for_resume,
    empty_payload_fn: Callable[[], tuple[dict, dict, dict, list, set, dict, object]] = empty_analysis_collection_resume_payload,
    payload_with_format_fn: Callable[..., object] = payload_with_format,
    mapping_sections_fn: Callable[..., object] = mapping_sections,
    mapping_payload_fn: Callable[[JSONValue], object] = mapping_payload,
    allowed_path_lookup_fn: Callable[..., dict[str, Path]] = allowed_path_lookup,
    load_allowed_paths_from_sequence_fn: Callable[..., list[Path]] = load_allowed_paths_from_sequence,
    mapping_or_none_fn: Callable[[JSONValue], object] = mapping_or_none,
    sequence_or_none_fn: Callable[[JSONValue], object] = sequence_or_none,
    check_deadline_fn: Callable[[], None] = check_deadline,
):
    (
        groups_by_path,
        param_spans_by_path,
        bundle_sites_by_path,
        invariant_propositions,
        completed_paths,
        in_progress_scan_by_path,
        analysis_index_resume,
    ) = empty_payload_fn()
    payload = payload_with_format_fn(
        payload,
        format_version=format_version,
    )
    if payload is None:
        return empty_payload_fn()
    sections = mapping_sections_fn(
        payload,
        section_keys=(
            "groups_by_path",
            "param_spans_by_path",
            "bundle_sites_by_path",
        ),
    )
    if sections is None:
        return empty_payload_fn()
    groups_payload, spans_payload, sites_payload = sections
    in_progress_scan_payload = mapping_payload_fn(payload.get("in_progress_scan_by_path"))
    completed_payload = payload.get("completed_paths")
    if in_progress_scan_payload is None:
        in_progress_scan_payload = {}
    allowed_paths = allowed_path_lookup_fn(
        file_paths,
        key_fn=path_key_fn,
    )
    for path in load_allowed_paths_from_sequence_fn(
        completed_payload,
        allowed_paths=allowed_paths,
    ):
        check_deadline_fn()
        path_key = path_key_fn(path)
        raw_groups = mapping_or_none_fn(groups_payload.get(path_key))
        raw_spans = mapping_or_none_fn(spans_payload.get(path_key))
        raw_sites = mapping_or_none_fn(sites_payload.get(path_key))
        if raw_groups is not None and raw_spans is not None and raw_sites is not None:
            groups_by_path[path] = deserialize_groups_for_resume_fn(raw_groups)
            param_spans_by_path[path] = deserialize_param_spans_for_resume_fn(raw_spans)
            bundle_sites_by_path[path] = deserialize_bundle_sites_for_resume_fn(raw_sites)
            completed_paths.add(path)
    if include_invariant_propositions:
        raw_invariants = sequence_or_none_fn(payload.get("invariant_propositions"))
        if raw_invariants is not None:
            invariant_propositions = deserialize_invariants_for_resume_fn(raw_invariants)
    for raw_path, raw_state in in_progress_scan_payload.items():
        check_deadline_fn()
        raw_state_mapping = mapping_or_none_fn(raw_state)
        path = allowed_paths.get(raw_path)
        if raw_state_mapping is not None and path is not None and path not in completed_paths:
            in_progress_scan_by_path[path] = {
                str(key): raw_state_mapping[key] for key in raw_state_mapping
            }
    raw_analysis_index_resume = payload.get("analysis_index_resume")
    raw_analysis_index_mapping = mapping_or_none_fn(raw_analysis_index_resume)
    if raw_analysis_index_mapping is not None:
        analysis_index_resume = {
            str(key): raw_analysis_index_mapping[key]
            for key in raw_analysis_index_mapping
        }
    return (
        groups_by_path,
        param_spans_by_path,
        bundle_sites_by_path,
        invariant_propositions,
        completed_paths,
        in_progress_scan_by_path,
        analysis_index_resume,
    )


__all__ = [
    "analysis_collection_resume_path_key",
    "build_analysis_collection_resume_payload",
    "deserialize_bundle_sites_for_resume",
    "deserialize_groups_for_resume",
    "deserialize_invariants_for_resume",
    "deserialize_param_spans_for_resume",
    "empty_analysis_collection_resume_payload",
    "load_analysis_collection_resume_payload",
    "serialize_bundle_sites_for_resume",
    "serialize_groups_for_resume",
    "serialize_invariants_for_resume",
    "serialize_param_spans_for_resume",
]
