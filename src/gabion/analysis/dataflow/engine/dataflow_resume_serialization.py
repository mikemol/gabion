from __future__ import annotations

from gabion.json_types import JSONObject, JSONValue
"""Resume serialization owners extracted from the legacy dataflow monolith."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, cast

from gabion.analysis.dataflow.engine.dataflow_contracts import (
    CallArgs,
    ClassInfo,
    FunctionInfo,
    InvariantProposition,
    OptionalFloat,
    ParamUse,
    SymbolTable,
)
from gabion.analysis.foundation.json_types import JSONObject, JSONValue
from gabion.analysis.foundation.resume_codec import (
    allowed_path_lookup,
    int_str_pairs_from_sequence,
    int_tuple4_or_none,
    iter_valid_key_entries,
    load_allowed_paths_from_sequence,
    load_resume_map,
    mapping_or_empty,
    mapping_or_none,
    mapping_payload,
    mapping_sections,
    payload_with_format,
    payload_with_phase,
    sequence_or_none,
    str_list_from_sequence,
    str_map_from_mapping,
    str_pair_set_from_sequence,
    str_set_from_sequence,
    str_tuple_from_sequence,
)
from gabion.analysis.foundation.timeout_context import check_deadline, deadline_loop_iter
from gabion.analysis.indexed_scan.index.analysis_index_resume_payload import (
    LoadAnalysisIndexResumePayloadDeps,
    SerializeAnalysisIndexResumePayloadDeps,
    load_analysis_index_resume_payload as _load_analysis_index_resume_payload_impl,
    serialize_analysis_index_resume_payload as _serialize_analysis_index_resume_payload_impl,
)
from gabion.analysis.indexed_scan.state.file_scan_resume_state import (
    FileScanResumeStateLoadDeps,
    FileScanResumeStateSerializeDeps,
    load_file_scan_resume_state as _load_file_scan_resume_state_impl,
    serialize_file_scan_resume_state as _serialize_file_scan_resume_state_impl,
)
from gabion.analysis.indexed_scan.state.function_info_resume import (
    DeserializeFunctionInfoForResumeDeps,
    SerializeFunctionInfoForResumeDeps,
    deserialize_function_info_for_resume as _deserialize_function_info_for_resume_impl,
    serialize_function_info_for_resume as _serialize_function_info_for_resume_impl,
)
from gabion.analysis.indexed_scan.state.resume_state import (
    analysis_collection_resume_path_key as _analysis_collection_resume_path_key_impl,
    build_analysis_collection_resume_payload as _build_analysis_collection_resume_payload_impl,
    deserialize_bundle_sites_for_resume as _deserialize_bundle_sites_for_resume_impl,
    deserialize_groups_for_resume as _deserialize_groups_for_resume_impl,
    deserialize_invariants_for_resume as _deserialize_invariants_for_resume_impl,
    deserialize_param_spans_for_resume as _deserialize_param_spans_for_resume_impl,
    empty_analysis_collection_resume_payload as _empty_analysis_collection_resume_payload_impl,
    load_analysis_collection_resume_payload as _load_analysis_collection_resume_payload_impl,
    serialize_bundle_sites_for_resume as _serialize_bundle_sites_for_resume_impl,
    serialize_groups_for_resume as _serialize_groups_for_resume_impl,
    serialize_invariants_for_resume as _serialize_invariants_for_resume_impl,
    serialize_param_spans_for_resume as _serialize_param_spans_for_resume_impl,
)
from gabion.analysis.indexed_scan.state.symbol_table_resume import (
    DeserializeSymbolTableForResumeDeps,
    SerializeSymbolTableForResumeDeps,
    deserialize_symbol_table_for_resume as _deserialize_symbol_table_for_resume_impl,
    serialize_symbol_table_for_resume as _serialize_symbol_table_for_resume_impl,
)
from gabion.invariants import NeverThrown, never
from gabion.order_contract import sort_once
from gabion.runtime_shape_dispatch import str_or_none

_ANALYSIS_COLLECTION_RESUME_FORMAT_VERSION = 2
_ANALYSIS_INDEX_RESUME_VARIANTS_KEY = "resume_variants"
_ANALYSIS_INDEX_RESUME_MAX_VARIANTS = 4

_CACHE_IDENTITY_PREFIX = "aspf:sha1:"
_CACHE_IDENTITY_DIGEST_HEX = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class _CacheIdentity:
    value: str

    @classmethod
    def from_digest(cls, digest: str) -> "_CacheIdentity | None":
        cleaned = str(digest or "").strip().lower()
        if not _CACHE_IDENTITY_DIGEST_HEX.fullmatch(cleaned):
            return None
        return cls(f"{_CACHE_IDENTITY_PREFIX}{cleaned}")

    @classmethod
    def from_boundary(cls, raw_identity) -> "_CacheIdentity | None":
        identity = str(raw_identity or "").strip()
        if not identity:
            return None
        if identity.startswith(_CACHE_IDENTITY_PREFIX):
            digest = identity[len(_CACHE_IDENTITY_PREFIX) :]
            return cls.from_digest(digest)
        return cls.from_digest(identity)

    @classmethod
    def from_boundary_required(cls, raw_identity, *, field: str) -> "_CacheIdentity":
        identity = cls.from_boundary(raw_identity)
        if identity is None:
            never("invalid cache identity", field=field)
            return cls(value="")  # pragma: no cover - never() raises
        return identity


@dataclass(frozen=True)
class _ResumeCacheIdentityPair:
    canonical_index: _CacheIdentity
    canonical_projection: _CacheIdentity

    def encode(self) -> dict[str, str]:
        return {
            "index_cache_identity": self.canonical_index.value,
            "projection_cache_identity": self.canonical_projection.value,
        }

    @classmethod
    def decode_required(cls, payload: Mapping[str, JSONValue]) -> "_ResumeCacheIdentityPair":
        return cls(
            canonical_index=_CacheIdentity.from_boundary_required(
                payload.get("index_cache_identity"),
                field="index_cache_identity",
            ),
            canonical_projection=_CacheIdentity.from_boundary_required(
                payload.get("projection_cache_identity"),
                field="projection_cache_identity",
            ),
        )


def _invariant_digest(payload: Mapping[str, JSONValue], *, prefix: str) -> str:
    encoded = json.dumps(payload, sort_keys=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.blake2s(encoded, digest_size=12).hexdigest()
    return f"{prefix}:{digest}"


def _invariant_confidence(value: OptionalFloat) -> float:
    if value is None:
        return 1.0
    return max(0.0, min(1.0, float(value)))


def _compute_invariant_id(
    *,
    form: str,
    terms: tuple[str, ...],
    scope: str,
    source: str,
) -> str:
    payload = {
        "form": form,
        "terms": list(terms),
        "scope": scope,
        "source": source,
    }
    return _invariant_digest(payload, prefix="inv")


def _compute_invariant_evidence_key(
    *,
    invariant_id: str,
    form: str,
    terms: tuple[str, ...],
    scope: str,
) -> str:
    term_display = ",".join(terms)
    return f"E:invariant::{scope}::{form}::{term_display}::{invariant_id}"


def _normalize_invariant_proposition(
    proposition: InvariantProposition,
    *,
    default_scope: str,
    default_source: str,
) -> InvariantProposition:
    scope = proposition.scope or default_scope
    source = proposition.source or default_source
    invariant_id = proposition.invariant_id or _compute_invariant_id(
        form=proposition.form,
        terms=proposition.terms,
        scope=scope,
        source=source,
    )
    evidence_keys = proposition.evidence_keys or (
        _compute_invariant_evidence_key(
            invariant_id=invariant_id,
            form=proposition.form,
            terms=proposition.terms,
            scope=scope,
        ),
    )
    return InvariantProposition(
        form=proposition.form,
        terms=proposition.terms,
        scope=scope,
        source=source,
        invariant_id=invariant_id,
        confidence=_invariant_confidence(proposition.confidence),
        evidence_keys=tuple(str(key) for key in evidence_keys),
    )


def _analysis_collection_resume_path_key(path: Path) -> str:
    return _analysis_collection_resume_path_key_impl(path)


def _serialize_param_use(value: ParamUse) -> JSONObject:
    return {
        "direct_forward": [
            [callee, slot]
            for callee, slot in sort_once(
                value.direct_forward,
                source="gabion.analysis.dataflow_indexed_file_scan._serialize_param_use.site_1",
            )
        ],
        "non_forward": bool(value.non_forward),
        "current_aliases": sort_once(
            value.current_aliases,
            source="gabion.analysis.dataflow_indexed_file_scan._serialize_param_use.site_2",
        ),
        "forward_sites": [
            {
                "callee": callee,
                "slot": slot,
                "spans": [
                    list(span)
                    for span in sort_once(
                        spans,
                        source="gabion.analysis.dataflow_indexed_file_scan._serialize_param_use.site_3",
                    )
                ],
            }
            for (callee, slot), spans in sort_once(
                value.forward_sites.items(),
                source="gabion.analysis.dataflow_indexed_file_scan._serialize_param_use.site_4",
            )
        ],
        "unknown_key_carrier": bool(value.unknown_key_carrier),
        "unknown_key_sites": [
            list(span)
            for span in sort_once(
                value.unknown_key_sites,
                source="gabion.analysis.dataflow_indexed_file_scan._serialize_param_use.site_5",
            )
        ],
    }


def _deserialize_param_use(payload: Mapping[str, JSONValue]) -> ParamUse:
    direct_forward = str_pair_set_from_sequence(payload.get("direct_forward"))
    current_aliases = str_set_from_sequence(payload.get("current_aliases"))
    forward_sites: dict[tuple[str, str], set[tuple[int, int, int, int]]] = {}
    for raw_entry in sequence_or_none(payload.get("forward_sites")) or ():
        check_deadline()
        entry = mapping_or_none(raw_entry)
        if entry is not None:
            callee = str_or_none(entry.get("callee"))
            slot = str_or_none(entry.get("slot"))
            if callee is not None and slot is not None:
                span_set: set[tuple[int, int, int, int]] = set()
                for raw_span in sequence_or_none(entry.get("spans")) or ():
                    check_deadline()
                    span = int_tuple4_or_none(raw_span)
                    if span is not None:
                        span_set.add(span)
                forward_sites[(callee, slot)] = span_set
    non_forward = bool(payload.get("non_forward"))
    unknown_key_carrier = bool(payload.get("unknown_key_carrier"))
    unknown_key_sites: set[tuple[int, int, int, int]] = set()
    for raw_span in sequence_or_none(payload.get("unknown_key_sites")) or ():
        check_deadline()
        span = int_tuple4_or_none(raw_span)
        if span is not None:
            unknown_key_sites.add(span)
    return ParamUse(
        direct_forward=direct_forward,
        non_forward=non_forward,
        current_aliases=current_aliases,
        forward_sites=forward_sites,
        unknown_key_carrier=unknown_key_carrier,
        unknown_key_sites=unknown_key_sites,
    )


def _serialize_param_use_map(
    use_map: Mapping[str, ParamUse],
) -> JSONObject:
    payload: JSONObject = {}
    for param_name in sort_once(
        use_map,
        source="gabion.analysis.dataflow_indexed_file_scan._serialize_param_use_map.site_1",
    ):
        check_deadline()
        payload[param_name] = _serialize_param_use(use_map[param_name])
    return payload


def _deserialize_param_use_map(
    payload: Mapping[str, JSONValue],
) -> dict[str, ParamUse]:
    use_map: dict[str, ParamUse] = {}
    for param_name, raw_value in payload.items():
        check_deadline()
        raw_mapping = mapping_or_none(raw_value)
        normalized_param_name = str_or_none(param_name)
        if normalized_param_name is not None and raw_mapping is not None:
            use_map[normalized_param_name] = _deserialize_param_use(raw_mapping)
    return use_map


def _serialize_call_args(call: CallArgs) -> JSONObject:
    payload: JSONObject = {
        "callee": call.callee,
        "pos_map": {
            key: call.pos_map[key]
            for key in sort_once(
                call.pos_map,
                source="gabion.analysis.dataflow_indexed_file_scan._serialize_call_args.site_1",
            )
        },
        "kw_map": {
            key: call.kw_map[key]
            for key in sort_once(
                call.kw_map,
                source="gabion.analysis.dataflow_indexed_file_scan._serialize_call_args.site_2",
            )
        },
        "const_pos": {
            key: call.const_pos[key]
            for key in sort_once(
                call.const_pos,
                source="gabion.analysis.dataflow_indexed_file_scan._serialize_call_args.site_3",
            )
        },
        "const_kw": {
            key: call.const_kw[key]
            for key in sort_once(
                call.const_kw,
                source="gabion.analysis.dataflow_indexed_file_scan._serialize_call_args.site_4",
            )
        },
        "non_const_pos": sort_once(
            call.non_const_pos,
            source="gabion.analysis.dataflow_indexed_file_scan._serialize_call_args.site_5",
        ),
        "non_const_kw": sort_once(
            call.non_const_kw,
            source="gabion.analysis.dataflow_indexed_file_scan._serialize_call_args.site_6",
        ),
        "star_pos": [[idx, name] for idx, name in call.star_pos],
        "star_kw": list(call.star_kw),
        "is_test": call.is_test,
        "callable_kind": call.callable_kind,
        "callable_source": call.callable_source,
    }
    if call.span is not None:
        payload["span"] = list(call.span)
    return payload


def _deserialize_call_args(payload: Mapping[str, JSONValue]):
    callee = payload.get("callee")
    if type(callee) is not str:
        return None
    star_pos = int_str_pairs_from_sequence(payload.get("star_pos"))
    span = int_tuple4_or_none(payload.get("span"))
    return CallArgs(
        callee=callee,
        pos_map=str_map_from_mapping(payload.get("pos_map")),
        kw_map=str_map_from_mapping(payload.get("kw_map")),
        const_pos=str_map_from_mapping(payload.get("const_pos")),
        const_kw=str_map_from_mapping(payload.get("const_kw")),
        non_const_pos=str_set_from_sequence(payload.get("non_const_pos")),
        non_const_kw=str_set_from_sequence(payload.get("non_const_kw")),
        star_pos=star_pos,
        star_kw=sort_once(
            str_set_from_sequence(payload.get("star_kw")),
            source="gabion.analysis.dataflow_indexed_file_scan._deserialize_call_args.site_1",
        ),
        is_test=bool(payload.get("is_test")),
        span=span,
        callable_kind=str(payload.get("callable_kind") or "function"),
        callable_source=str(payload.get("callable_source") or "symbol"),
    )


def _serialize_call_args_list(call_args: Sequence[CallArgs]) -> list[JSONObject]:
    return [_serialize_call_args(call) for call in call_args]


def _deserialize_call_args_list(payload: Sequence[JSONValue]) -> list[CallArgs]:
    call_args: list[CallArgs] = []
    for raw_entry in payload:
        check_deadline()
        entry_mapping = mapping_or_none(raw_entry)
        if entry_mapping is not None:
            call = _deserialize_call_args(entry_mapping)
            if call is not None:
                call_args.append(call)
    return call_args


def _serialize_function_info_for_resume(info: FunctionInfo) -> JSONObject:
    return cast(
        JSONObject,
        _serialize_function_info_for_resume_impl(
            info,
            deps=SerializeFunctionInfoForResumeDeps(
                sort_once_fn=sort_once,
                serialize_call_args_list_fn=_serialize_call_args_list,
            ),
        ),
    )


def _deserialize_function_info_for_resume(
    payload: Mapping[str, JSONValue],
    *,
    allowed_paths: Mapping[str, Path],
):
    return _deserialize_function_info_for_resume_impl(
        payload,
        allowed_paths=allowed_paths,
        deps=DeserializeFunctionInfoForResumeDeps(
            sequence_or_none_fn=sequence_or_none,
            str_list_from_sequence_fn=str_list_from_sequence,
            str_or_none_fn=str_or_none,
            mapping_or_empty_fn=mapping_or_empty,
            check_deadline_fn=check_deadline,
            deserialize_call_args_list_fn=_deserialize_call_args_list,
            str_set_from_sequence_fn=str_set_from_sequence,
            str_tuple_from_sequence_fn=str_tuple_from_sequence,
            int_tuple4_or_none_fn=int_tuple4_or_none,
            function_info_ctor=FunctionInfo,
        ),
    )


def _serialize_class_info_for_resume(class_info: ClassInfo) -> JSONObject:
    return {
        "qual": class_info.qual,
        "module": class_info.module,
        "bases": list(class_info.bases),
        "methods": sort_once(
            class_info.methods,
            source="gabion.analysis.dataflow_indexed_file_scan._serialize_class_info_for_resume.site_1",
        ),
    }


def _deserialize_class_info_for_resume(
    payload: Mapping[str, JSONValue],
):
    qual = payload.get("qual")
    module = payload.get("module")
    if type(qual) is not str or type(module) is not str:
        return None
    bases = str_list_from_sequence(payload.get("bases"))
    methods = str_set_from_sequence(payload.get("methods"))
    return ClassInfo(
        qual=qual,
        module=module,
        bases=bases,
        methods=methods,
    )


def _serialize_symbol_table_for_resume(table: SymbolTable) -> JSONObject:
    return cast(
        JSONObject,
        _serialize_symbol_table_for_resume_impl(
            table,
            deps=SerializeSymbolTableForResumeDeps(
                sort_once_fn=sort_once,
            ),
        ),
    )


def _deserialize_symbol_table_for_resume(payload: Mapping[str, JSONValue]) -> SymbolTable:
    return cast(
        SymbolTable,
        _deserialize_symbol_table_for_resume_impl(
            payload,
            deps=DeserializeSymbolTableForResumeDeps(
                symbol_table_ctor=SymbolTable,
                sequence_or_none_fn=sequence_or_none,
                check_deadline_fn=check_deadline,
                str_set_from_sequence_fn=str_set_from_sequence,
                mapping_or_none_fn=mapping_or_none,
                mapping_or_empty_fn=mapping_or_empty,
            ),
        ),
    )


def _analysis_index_resume_variant_payload(payload: Mapping[str, JSONValue]) -> JSONObject:
    variant_payload = {
        str(key): payload[key]
        for key in payload
        if str(key) != _ANALYSIS_INDEX_RESUME_VARIANTS_KEY
    }
    try:
        identities = _ResumeCacheIdentityPair.decode_required(variant_payload)
    except NeverThrown:
        return cast(JSONObject, variant_payload)
    variant_payload.update(identities.encode())
    return cast(JSONObject, variant_payload)


def _analysis_index_resume_variants(
    payload=None,
) -> dict[str, JSONObject]:
    variants: dict[str, JSONObject] = {}
    if payload is None:
        return variants
    raw_variants = payload.get(_ANALYSIS_INDEX_RESUME_VARIANTS_KEY)
    raw_variants_mapping = mapping_or_none(raw_variants)
    if raw_variants_mapping is not None:
        for identity, raw_variant in raw_variants_mapping.items():
            check_deadline()
            raw_variant_mapping = mapping_or_none(raw_variant)
            variant_identity = _CacheIdentity.from_boundary(identity)
            if variant_identity is not None and raw_variant_mapping is not None:
                variant_payload = payload_with_format(raw_variant_mapping, format_version=1)
                if variant_payload is not None:
                    variants[variant_identity.value] = _analysis_index_resume_variant_payload(
                        variant_payload
                    )
    return variants


def _with_analysis_index_resume_variants(
    *,
    payload: JSONObject,
    previous_payload,
) -> JSONObject:
    identities = _ResumeCacheIdentityPair.decode_required(payload)
    variants = _analysis_index_resume_variants(previous_payload)
    payload.update(identities.encode())
    variants[identities.canonical_index.value] = _analysis_index_resume_variant_payload(payload)
    ordered_variant_keys = [
        key
        for key in sort_once(
            variants.keys(),
            source="gabion.analysis.dataflow_indexed_file_scan._with_analysis_index_resume_variants.site_1",
        )
        if key != identities.canonical_index.value
    ]
    ordered_variant_keys.append(identities.canonical_index.value)
    if len(ordered_variant_keys) > _ANALYSIS_INDEX_RESUME_MAX_VARIANTS:
        ordered_variant_keys = ordered_variant_keys[-_ANALYSIS_INDEX_RESUME_MAX_VARIANTS :]
    payload[_ANALYSIS_INDEX_RESUME_VARIANTS_KEY] = {
        key: variants[key] for key in ordered_variant_keys
    }
    return payload


def _resume_variant_for_identity(
    variants: Mapping[str, JSONObject],
    expected_identity: _CacheIdentity,
):
    direct = variants.get(expected_identity.value)
    if direct is not None:
        return direct
    return None


def _serialize_analysis_index_resume_payload(
    *,
    hydrated_paths: set[Path],
    by_qual: Mapping[str, FunctionInfo],
    symbol_table: SymbolTable,
    class_index: Mapping[str, ClassInfo],
    index_cache_identity: str,
    projection_cache_identity: str,
    profiling_v1=None,
    previous_payload=None,
) -> JSONObject:
    return _serialize_analysis_index_resume_payload_impl(
        hydrated_paths=hydrated_paths,
        by_qual=cast(Mapping[str, object], by_qual),
        symbol_table=symbol_table,
        class_index=cast(Mapping[str, object], class_index),
        index_cache_identity=index_cache_identity,
        projection_cache_identity=projection_cache_identity,
        profiling_v1=profiling_v1,
        previous_payload=previous_payload,
        deps=SerializeAnalysisIndexResumePayloadDeps(
            resume_cache_identity_pair_ctor=_ResumeCacheIdentityPair,
            cache_identity_from_boundary_required_fn=_CacheIdentity.from_boundary_required,
            analysis_collection_resume_path_key_fn=_analysis_collection_resume_path_key,
            sort_once_fn=sort_once,
            serialize_function_info_for_resume_fn=_serialize_function_info_for_resume,
            serialize_symbol_table_for_resume_fn=_serialize_symbol_table_for_resume,
            serialize_class_info_for_resume_fn=_serialize_class_info_for_resume,
            mapping_or_none_fn=mapping_or_none,
            with_analysis_index_resume_variants_fn=_with_analysis_index_resume_variants,
        ),
    )


def _load_analysis_index_resume_payload(
    *,
    payload,
    file_paths: Sequence[Path],
    expected_index_cache_identity: str = "",
    expected_projection_cache_identity: str = "",
) -> tuple[set[Path], dict[str, FunctionInfo], SymbolTable, dict[str, ClassInfo]]:
    hydrated_paths, by_qual_raw, symbol_table_raw, class_index_raw = (
        _load_analysis_index_resume_payload_impl(
            payload=payload,
            file_paths=file_paths,
            expected_index_cache_identity=expected_index_cache_identity,
            expected_projection_cache_identity=expected_projection_cache_identity,
            deps=LoadAnalysisIndexResumePayloadDeps(
                symbol_table_ctor=SymbolTable,
                payload_with_format_fn=payload_with_format,
                cache_identity_from_boundary_fn=_CacheIdentity.from_boundary,
                analysis_index_resume_variants_fn=_analysis_index_resume_variants,
                resume_variant_for_identity_fn=_resume_variant_for_identity,
                allowed_path_lookup_fn=allowed_path_lookup,
                analysis_collection_resume_path_key_fn=_analysis_collection_resume_path_key,
                load_allowed_paths_from_sequence_fn=load_allowed_paths_from_sequence,
                mapping_or_none_fn=mapping_or_none,
                check_deadline_fn=check_deadline,
                deserialize_function_info_for_resume_fn=_deserialize_function_info_for_resume,
                deserialize_symbol_table_for_resume_fn=_deserialize_symbol_table_for_resume,
                deserialize_class_info_for_resume_fn=_deserialize_class_info_for_resume,
            ),
        )
    )
    return (
        hydrated_paths,
        cast(dict[str, FunctionInfo], by_qual_raw),
        cast(SymbolTable, symbol_table_raw),
        cast(dict[str, ClassInfo], class_index_raw),
    )


def _serialize_groups_for_resume(
    groups: dict[str, list[set[str]]],
) -> dict[str, list[list[str]]]:
    return _serialize_groups_for_resume_impl(
        groups,
        check_deadline_fn=check_deadline,
        sort_once_fn=sort_once,
    )


def _deserialize_groups_for_resume(
    payload: Mapping[str, JSONValue],
) -> dict[str, list[set[str]]]:
    return _deserialize_groups_for_resume_impl(
        payload,
        check_deadline_fn=check_deadline,
        sequence_or_none_fn=sequence_or_none,
    )


def _serialize_param_spans_for_resume(
    spans: dict[str, dict[str, tuple[int, int, int, int]]],
) -> dict[str, dict[str, list[int]]]:
    return _serialize_param_spans_for_resume_impl(
        spans,
        check_deadline_fn=check_deadline,
        sort_once_fn=sort_once,
    )


def _deserialize_param_spans_for_resume(
    payload: Mapping[str, JSONValue],
) -> dict[str, dict[str, tuple[int, int, int, int]]]:
    return _deserialize_param_spans_for_resume_impl(
        payload,
        check_deadline_fn=check_deadline,
        sequence_or_none_fn=sequence_or_none,
    )


def _serialize_bundle_sites_for_resume(
    bundle_sites: dict[str, list[list[JSONObject]]],
) -> dict[str, list[list[JSONObject]]]:
    return _serialize_bundle_sites_for_resume_impl(
        bundle_sites,
        check_deadline_fn=check_deadline,
        sort_once_fn=sort_once,
        sequence_or_none_fn=sequence_or_none,
        mapping_or_none_fn=mapping_or_none,
    )


def _deserialize_bundle_sites_for_resume(
    payload: Mapping[str, JSONValue],
) -> dict[str, list[list[JSONObject]]]:
    return _deserialize_bundle_sites_for_resume_impl(
        payload,
        check_deadline_fn=check_deadline,
        sequence_or_none_fn=sequence_or_none,
        mapping_or_none_fn=mapping_or_none,
    )


def _serialize_invariants_for_resume(
    invariants: Sequence[InvariantProposition],
) -> list[JSONObject]:
    return _serialize_invariants_for_resume_impl(
        invariants,
        check_deadline_fn=check_deadline,
        sort_once_fn=sort_once,
    )


def _deserialize_invariants_for_resume(
    payload: Sequence[JSONValue],
) -> list[InvariantProposition]:
    return _deserialize_invariants_for_resume_impl(
        payload,
        normalize_invariant_proposition_fn=_normalize_invariant_proposition,
        check_deadline_fn=check_deadline,
        mapping_or_none_fn=mapping_or_none,
        sequence_or_none_fn=sequence_or_none,
    )


def _serialize_file_scan_resume_state(
    *,
    fn_use: Mapping[str, Mapping[str, ParamUse]],
    fn_calls: Mapping[str, Sequence[CallArgs]],
    fn_param_orders: Mapping[str, Sequence[str]],
    fn_param_spans: Mapping[str, Mapping[str, tuple[int, int, int, int]]],
    fn_names: Mapping[str, str],
    fn_lexical_scopes: Mapping[str, Sequence[str]],
    fn_class_names: Mapping[str, object],
    opaque_callees: set[str],
) -> JSONObject:
    return cast(
        JSONObject,
        _serialize_file_scan_resume_state_impl(
            fn_use=fn_use,
            fn_calls=fn_calls,
            fn_param_orders=fn_param_orders,
            fn_param_spans=fn_param_spans,
            fn_names=fn_names,
            fn_lexical_scopes=fn_lexical_scopes,
            fn_class_names=fn_class_names,
            opaque_callees=opaque_callees,
            deps=FileScanResumeStateSerializeDeps(
                sort_once_fn=sort_once,
                check_deadline_fn=check_deadline,
                serialize_param_use_map_fn=_serialize_param_use_map,
                serialize_call_args_list_fn=_serialize_call_args_list,
                serialize_param_spans_for_resume_fn=_serialize_param_spans_for_resume,
            ),
        ),
    )


def _empty_file_scan_resume_state():
    return ({}, {}, {}, {}, {}, {}, {}, set())


def _load_file_scan_resume_state(
    *,
    payload,
    valid_fn_keys: set[str],
):
    return _load_file_scan_resume_state_impl(
        payload=payload,
        valid_fn_keys=valid_fn_keys,
        deps=FileScanResumeStateLoadDeps(
            empty_state_fn=_empty_file_scan_resume_state,
            payload_with_phase_fn=payload_with_phase,
            mapping_sections_fn=mapping_sections,
            load_resume_map_fn=load_resume_map,
            deserialize_param_use_map_fn=_deserialize_param_use_map,
            mapping_or_none_fn=mapping_or_none,
            deserialize_call_args_list_fn=_deserialize_call_args_list,
            sequence_or_none_fn=sequence_or_none,
            str_list_from_sequence_fn=str_list_from_sequence,
            deserialize_param_spans_for_resume_fn=_deserialize_param_spans_for_resume,
            str_tuple_from_sequence_fn=str_tuple_from_sequence,
            deadline_loop_iter_fn=deadline_loop_iter,
            iter_valid_key_entries_fn=iter_valid_key_entries,
        ),
    )


def _build_analysis_collection_resume_payload(
    *,
    groups_by_path: Mapping[Path, dict[str, list[set[str]]]],
    param_spans_by_path: Mapping[Path, dict[str, dict[str, tuple[int, int, int, int]]]],
    bundle_sites_by_path: Mapping[Path, dict[str, list[list[JSONObject]]]],
    invariant_propositions: Sequence[InvariantProposition],
    completed_paths: set[Path],
    in_progress_scan_by_path: Mapping[Path, JSONObject],
    analysis_index_resume=None,
    file_stage_timings_v1_by_path=None,
) -> JSONObject:
    return _build_analysis_collection_resume_payload_impl(
        groups_by_path=groups_by_path,
        param_spans_by_path=param_spans_by_path,
        bundle_sites_by_path=bundle_sites_by_path,
        invariant_propositions=invariant_propositions,
        completed_paths=completed_paths,
        in_progress_scan_by_path=in_progress_scan_by_path,
        analysis_index_resume=analysis_index_resume,
        file_stage_timings_v1_by_path=file_stage_timings_v1_by_path,
        format_version=_ANALYSIS_COLLECTION_RESUME_FORMAT_VERSION,
        path_key_fn=_analysis_collection_resume_path_key,
        check_deadline_fn=check_deadline,
        sort_once_fn=sort_once,
        serialize_groups_for_resume_fn=_serialize_groups_for_resume,
        serialize_param_spans_for_resume_fn=_serialize_param_spans_for_resume,
        serialize_bundle_sites_for_resume_fn=_serialize_bundle_sites_for_resume,
        serialize_invariants_for_resume_fn=_serialize_invariants_for_resume,
        mapping_or_none_fn=mapping_or_none,
        never_fn=never,
    )


def _empty_analysis_collection_resume_payload():
    return _empty_analysis_collection_resume_payload_impl()


def _load_analysis_collection_resume_payload(
    *,
    payload,
    file_paths: Sequence[Path],
    include_invariant_propositions: bool,
):
    return _load_analysis_collection_resume_payload_impl(
        payload=payload,
        file_paths=file_paths,
        include_invariant_propositions=include_invariant_propositions,
        format_version=_ANALYSIS_COLLECTION_RESUME_FORMAT_VERSION,
        path_key_fn=_analysis_collection_resume_path_key,
        deserialize_groups_for_resume_fn=_deserialize_groups_for_resume,
        deserialize_param_spans_for_resume_fn=_deserialize_param_spans_for_resume,
        deserialize_bundle_sites_for_resume_fn=_deserialize_bundle_sites_for_resume,
        deserialize_invariants_for_resume_fn=_deserialize_invariants_for_resume,
        empty_payload_fn=_empty_analysis_collection_resume_payload,
        payload_with_format_fn=payload_with_format,
        mapping_sections_fn=mapping_sections,
        mapping_payload_fn=mapping_payload,
        allowed_path_lookup_fn=allowed_path_lookup,
        load_allowed_paths_from_sequence_fn=load_allowed_paths_from_sequence,
        mapping_or_none_fn=mapping_or_none,
        sequence_or_none_fn=sequence_or_none,
        check_deadline_fn=check_deadline,
    )


__all__ = [
    "_CACHE_IDENTITY_DIGEST_HEX",
    "_CACHE_IDENTITY_PREFIX",
    "_CacheIdentity",
    "_analysis_collection_resume_path_key",
    "_build_analysis_collection_resume_payload",
    "_deserialize_function_info_for_resume",
    "_deserialize_invariants_for_resume",
    "_deserialize_symbol_table_for_resume",
    "_load_analysis_collection_resume_payload",
    "_load_analysis_index_resume_payload",
    "_load_file_scan_resume_state",
    "_invariant_confidence",
    "_invariant_digest",
    "_normalize_invariant_proposition",
    "_serialize_analysis_index_resume_payload",
    "_serialize_file_scan_resume_state",
]
