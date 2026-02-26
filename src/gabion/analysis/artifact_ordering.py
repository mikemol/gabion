# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from collections.abc import Iterable, Mapping

from gabion.analysis.resume_codec import mapping_or_none
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import OrderPolicy, sort_once


def canonical_protocol_specs(
    protocols: Iterable[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Canonicalize synthesis protocol ordering for markdown emission."""

    indexed = list(enumerate(protocols))

    def _key(item: tuple[int, Mapping[str, object]]) -> tuple[str, str, str, str, int]:
        index, spec = item
        name = str(spec.get("name", ""))
        tier = str(spec.get("tier", ""))
        fields = spec.get("fields", [])
        evidence = spec.get("evidence", [])
        return (
            name,
            tier,
            ",".join(canonical_field_display_parts(fields)),
            ",".join(canonical_string_values(evidence)),
            index,
        )

    ordered = sort_once(
        indexed,
        source="artifact_ordering.canonical_protocol_specs",
        key=_key,
        policy=OrderPolicy.SORT,
    )
    return [spec for _, spec in ordered]


def canonical_field_display_parts(fields: Iterable[object]) -> list[str]:
    check_deadline()
    parts: list[str] = []
    for field in fields:
        check_deadline()
        field_map = mapping_or_none(field)
        if field_map is not None:
            fname = str(field_map.get("name", "")).strip()
            if fname:
                type_hint = str(field_map.get("type_hint") or "Any")
                parts.append(f"{fname}: {type_hint}")
    return parts


def canonical_string_values(values: Iterable[object]) -> list[str]:
    return sort_once(
        (str(value) for value in values),
        source="artifact_ordering.canonical_string_values",
        policy=OrderPolicy.SORT,
    )


def canonical_count_summary_items(counts: Mapping[str, int]) -> list[tuple[str, int]]:
    return sort_once(
        counts.items(),
        source="artifact_ordering.canonical_count_summary_items",
        key=lambda item: (-int(item[1]), item[0]),
        policy=OrderPolicy.SORT,
    )


def canonical_doc_scope(scope: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    for entry in scope:
        match entry:
            case str() as entry_text if entry_text.strip():
                cleaned.append(entry_text.strip())
            case _:
                pass
    if not cleaned:
        return ["repo", "artifacts"]
    return sort_once(
        cleaned,
        source="artifact_ordering.canonical_doc_scope",
        policy=OrderPolicy.SORT,
    )


def canonical_mapping_keys(mapping: Mapping[str, object]) -> list[str]:
    return sort_once(
        (str(key) for key in mapping.keys()),
        source="artifact_ordering.canonical_mapping_keys",
        policy=OrderPolicy.SORT,
    )
