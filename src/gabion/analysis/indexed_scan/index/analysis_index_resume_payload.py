# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from gabion.analysis.foundation.json_types import JSONObject, JSONValue


@dataclass(frozen=True)
class SerializeAnalysisIndexResumePayloadDeps:
    resume_cache_identity_pair_ctor: Callable[..., object]
    cache_identity_from_boundary_required_fn: Callable[..., object]
    analysis_collection_resume_path_key_fn: Callable[[Path], str]
    sort_once_fn: Callable[..., list[object]]
    serialize_function_info_for_resume_fn: Callable[..., JSONObject]
    serialize_symbol_table_for_resume_fn: Callable[..., JSONObject]
    serialize_class_info_for_resume_fn: Callable[..., JSONObject]
    mapping_or_none_fn: Callable[..., object]
    with_analysis_index_resume_variants_fn: Callable[..., JSONObject]


def serialize_analysis_index_resume_payload(
    *,
    hydrated_paths: set[Path],
    by_qual: Mapping[str, object],
    symbol_table,
    class_index: Mapping[str, object],
    index_cache_identity: str,
    projection_cache_identity: str,
    profiling_v1=None,
    previous_payload=None,
    deps: SerializeAnalysisIndexResumePayloadDeps,
) -> JSONObject:
    identities = deps.resume_cache_identity_pair_ctor(
        canonical_index=deps.cache_identity_from_boundary_required_fn(
            index_cache_identity,
            field="index_cache_identity",
        ),
        canonical_projection=deps.cache_identity_from_boundary_required_fn(
            projection_cache_identity,
            field="projection_cache_identity",
        ),
    )
    hydrated_path_keys = deps.sort_once_fn(
        (
            deps.analysis_collection_resume_path_key_fn(path)
            for path in hydrated_paths
        ),
        source="_serialize_analysis_index_resume_payload.hydrated_paths",
    )
    ordered_function_items = list(
        deps.sort_once_fn(
            by_qual.items(),
            source="_serialize_analysis_index_resume_payload.functions_by_qual",
        )
    )
    ordered_class_items = list(
        deps.sort_once_fn(
            class_index.items(),
            source="_serialize_analysis_index_resume_payload.class_index",
        )
    )
    resume_digest = hashlib.sha1(
        json.dumps(
            {
                "hydrated_paths": hydrated_path_keys,
                "function_quals": [qual for qual, _ in ordered_function_items],
                "class_quals": [qual for qual, _ in ordered_class_items],
            },
            sort_keys=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    payload: JSONObject = {
        "format_version": 1,
        "phase": "analysis_index_hydration",
        "resume_digest": resume_digest,
        **identities.encode(),
        "hydrated_paths": hydrated_path_keys,
        "hydrated_paths_count": len(hydrated_path_keys),
        "function_count": len(by_qual),
        "class_count": len(class_index),
        "functions_by_qual": {
            qual: deps.serialize_function_info_for_resume_fn(info)
            for qual, info in ordered_function_items
        },
        "symbol_table": deps.serialize_symbol_table_for_resume_fn(symbol_table),
        "class_index": {
            qual: deps.serialize_class_info_for_resume_fn(class_info)
            for qual, class_info in ordered_class_items
        },
    }
    profiling_payload = deps.mapping_or_none_fn(profiling_v1)
    if profiling_payload is not None:
        payload["profiling_v1"] = {
            str(key): profiling_payload[key] for key in profiling_payload
        }
    return deps.with_analysis_index_resume_variants_fn(
        payload=payload,
        previous_payload=previous_payload,
    )


@dataclass(frozen=True)
class LoadAnalysisIndexResumePayloadDeps:
    symbol_table_ctor: Callable[[], object]
    payload_with_format_fn: Callable[..., object]
    cache_identity_from_boundary_fn: Callable[..., object]
    analysis_index_resume_variants_fn: Callable[..., dict[str, JSONObject]]
    resume_variant_for_identity_fn: Callable[..., object]
    allowed_path_lookup_fn: Callable[..., Mapping[str, Path]]
    analysis_collection_resume_path_key_fn: Callable[[Path], str]
    load_allowed_paths_from_sequence_fn: Callable[..., list[Path]]
    mapping_or_none_fn: Callable[..., object]
    check_deadline_fn: Callable[[], None]
    deserialize_function_info_for_resume_fn: Callable[..., object]
    deserialize_symbol_table_for_resume_fn: Callable[..., object]
    deserialize_class_info_for_resume_fn: Callable[..., object]


def load_analysis_index_resume_payload(
    *,
    payload,
    file_paths: Sequence[Path],
    expected_index_cache_identity: str = "",
    expected_projection_cache_identity: str = "",
    deps: LoadAnalysisIndexResumePayloadDeps,
) -> tuple[set[Path], dict[str, object], object, dict[str, object]]:
    hydrated_paths: set[Path] = set()
    by_qual: dict[str, object] = {}
    symbol_table = deps.symbol_table_ctor()
    class_index: dict[str, object] = {}
    payload = deps.payload_with_format_fn(payload, format_version=1)
    if payload is None:
        return hydrated_paths, by_qual, symbol_table, class_index
    expected_index_identity = deps.cache_identity_from_boundary_fn(expected_index_cache_identity)
    expected_projection_identity = deps.cache_identity_from_boundary_fn(
        expected_projection_cache_identity
    )
    selected_payload = payload
    if expected_index_identity is not None:
        selected_identity = deps.cache_identity_from_boundary_fn(
            selected_payload.get("index_cache_identity")
        )
        if selected_identity != expected_index_identity:
            variants = deps.analysis_index_resume_variants_fn(payload)
            variant = deps.resume_variant_for_identity_fn(variants, expected_index_identity)
            if variant is None:
                return hydrated_paths, by_qual, symbol_table, class_index
            selected_payload = variant
    if expected_projection_identity is not None:
        projection_identity = deps.cache_identity_from_boundary_fn(
            selected_payload.get("projection_cache_identity")
        )
        if projection_identity != expected_projection_identity:
            return hydrated_paths, by_qual, symbol_table, class_index
    allowed_paths = deps.allowed_path_lookup_fn(
        file_paths,
        key_fn=deps.analysis_collection_resume_path_key_fn,
    )
    hydrated_paths = set(
        deps.load_allowed_paths_from_sequence_fn(
            selected_payload.get("hydrated_paths"),
            allowed_paths=allowed_paths,
        )
    )
    raw_functions_mapping = deps.mapping_or_none_fn(selected_payload.get("functions_by_qual"))
    if raw_functions_mapping is not None:
        for qual, raw_info in raw_functions_mapping.items():
            deps.check_deadline_fn()
            raw_info_mapping = deps.mapping_or_none_fn(raw_info)
            if type(qual) is str and raw_info_mapping is not None:
                info = deps.deserialize_function_info_for_resume_fn(
                    raw_info_mapping,
                    allowed_paths=allowed_paths,
                )
                if info is not None:
                    by_qual[qual] = info
    raw_symbol_table_mapping = deps.mapping_or_none_fn(selected_payload.get("symbol_table"))
    if raw_symbol_table_mapping is not None:
        symbol_table = deps.deserialize_symbol_table_for_resume_fn(raw_symbol_table_mapping)
    raw_class_index_mapping = deps.mapping_or_none_fn(selected_payload.get("class_index"))
    if raw_class_index_mapping is not None:
        for qual, raw_class in raw_class_index_mapping.items():
            deps.check_deadline_fn()
            raw_class_mapping = deps.mapping_or_none_fn(raw_class)
            if type(qual) is str and raw_class_mapping is not None:
                class_info = deps.deserialize_class_info_for_resume_fn(raw_class_mapping)
                if class_info is not None:
                    class_index[qual] = class_info
    return hydrated_paths, by_qual, symbol_table, class_index
