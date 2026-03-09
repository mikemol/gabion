# gabion:decision_protocol_module
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping

from gabion.analysis.foundation.resume_codec import mapping_optional
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.json_types import JSONValue
from gabion.order_contract import OrderPolicy, sort_once

_EMPTY_FIELD_MAP: Mapping[str, JSONValue] = {}


def canonical_protocol_specs(
    protocols: Iterable[Mapping[str, JSONValue]],
) -> Iterator[Mapping[str, JSONValue]]:
    """Canonicalize synthesis protocol ordering for markdown emission."""

    indexed = list(enumerate(protocols))

    def _key(item: tuple[int, Mapping[str, JSONValue]]) -> str:
        index, spec = item
        name = str(spec.get("name", ""))
        tier = str(spec.get("tier", ""))
        fields = spec.get("fields", [])
        evidence = spec.get("evidence", [])
        return "\x1f".join(
            (
                name,
                tier,
                ",".join(canonical_field_display_parts(fields)),
                ",".join(canonical_string_values(evidence)),
                f"{index:020d}",
            )
        )

    ordered = sort_once(
        indexed,
        source="artifact_ordering.canonical_protocol_specs",
        key=_key,
        policy=OrderPolicy.SORT,
    )
    return map(_indexed_protocol_spec, ordered)


def canonical_field_display_parts(fields: Iterable[JSONValue]) -> Iterator[str]:
    check_deadline()
    normalized_field_maps = map(_field_mapping_or_empty, fields)
    present_field_maps = filter(bool, normalized_field_maps)
    parts = map(_field_display_part, present_field_maps)
    return filter(bool, parts)


def canonical_string_values(values: Iterable[JSONValue]) -> Iterator[str]:
    return iter(
        sort_once(
            map(str, values),
            source="artifact_ordering.canonical_string_values",
            policy=OrderPolicy.SORT,
        )
    )


def canonical_count_summary_items(counts: Mapping[str, int]) -> list[tuple[str, int]]:
    return sort_once(
        counts.items(),
        source="artifact_ordering.canonical_count_summary_items",
        key=lambda item: (-int(item[1]), item[0]),
        policy=OrderPolicy.SORT,
    )


def canonical_doc_scope(scope: Iterable[str]) -> Iterator[str]:
    normalized_scope = map(_normalize_scope_entry, scope)
    cleaned_scope = tuple(filter(bool, normalized_scope))
    if not cleaned_scope:
        return iter(("repo", "artifacts"))
    return iter(
        sort_once(
            cleaned_scope,
            source="artifact_ordering.canonical_doc_scope",
            policy=OrderPolicy.SORT,
        )
    )


def canonical_mapping_keys(mapping: Mapping[str, JSONValue]) -> Iterator[str]:
    return iter(
        sort_once(
            map(str, mapping.keys()),
            source="artifact_ordering.canonical_mapping_keys",
            policy=OrderPolicy.SORT,
        )
    )


def _indexed_protocol_spec(
    item: tuple[int, Mapping[str, JSONValue]],
) -> Mapping[str, JSONValue]:
    _, spec = item
    return spec


def _field_mapping_or_empty(field: JSONValue) -> Mapping[str, JSONValue]:
    check_deadline()
    field_map = mapping_optional(field)
    if field_map is None:
        return _EMPTY_FIELD_MAP
    return field_map

def _field_display_part(field_map: Mapping[str, JSONValue]) -> str:
    name = str(field_map.get("name", "")).strip()
    if not name:
        return ""
    type_hint = str(field_map.get("type_hint") or "Any")
    return f"{name}: {type_hint}"


def _normalize_scope_entry(scope_entry: str) -> str:
    return str(scope_entry).strip()
