# gabion:decision_protocol_module
from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from typing import cast

from gabion.analysis.foundation.json_types import JSONObject
from gabion.analysis.foundation.marker_protocol import (
    DEFAULT_MARKER_ALIASES,
    DEFAULT_MARKER_KIND_MAPPING_CONFIG,
    MarkerKind,
    MarkerKindMappingConfig,
    MarkerLifecycleState,
    marker_identity,
    normalize_marker_payload,
    resolve_marker_kind_for_profile,
)


def keyword_string_literal(
    call: ast.Call,
    key: str,
    *,
    check_deadline_fn: Callable[[], None],
) -> str:
    check_deadline_fn()
    for kw in call.keywords:
        check_deadline_fn()
        if kw.arg != key:
            continue
        if type(kw.value) is ast.Constant:
            value = cast(ast.Constant, kw.value).value
            if type(value) is str:
                return value
    return ""


def keyword_links_literal(
    call: ast.Call,
    *,
    check_deadline_fn: Callable[[], None],
    sort_once_fn: Callable[..., object],
) -> list[JSONObject]:
    check_deadline_fn()
    for kw in call.keywords:
        check_deadline_fn()
        if kw.arg != "links":
            continue
        if type(kw.value) is not ast.List:
            return []
        links: list[JSONObject] = []
        for item in cast(ast.List, kw.value).elts:
            check_deadline_fn()
            if type(item) is not ast.Dict:
                continue
            dict_node = cast(ast.Dict, item)
            payload: JSONObject = {}
            for raw_key, raw_value in zip(dict_node.keys, dict_node.values, strict=False):
                check_deadline_fn()
                if type(raw_key) is not ast.Constant or type(raw_value) is not ast.Constant:
                    continue
                key_value = cast(ast.Constant, raw_key).value
                value_value = cast(ast.Constant, raw_value).value
                if type(key_value) is str and type(value_value) is str:
                    payload[key_value] = value_value
            kind = str(payload.get("kind", "")).strip()
            value = str(payload.get("value", "")).strip()
            if kind and value:
                links.append({"kind": kind, "value": value})
        return sort_once_fn(
            links,
            key=lambda item: (str(item.get("kind", "")), str(item.get("value", ""))),
            source="indexed_scan.marker_metadata.keyword_links_literal",
        )
    return []


def never_reason(
    call: ast.Call,
    *,
    check_deadline_fn: Callable[[], None],
) -> object:
    check_deadline_fn()
    if call.args and type(call.args[0]) is ast.Constant:
        value = cast(ast.Constant, call.args[0]).value
        if type(value) is str:
            return value
    for kw in call.keywords:
        check_deadline_fn()
        if kw.arg == "reason" and type(kw.value) is ast.Constant:
            value = cast(ast.Constant, kw.value).value
            if type(value) is str:
                return value
    return None


def marker_alias_kind_map(
    marker_aliases: Sequence[str],
    *,
    check_deadline_fn: Callable[[], None],
) -> tuple[set[str], dict[str, MarkerKind]]:
    check_deadline_fn()
    alias_map: dict[str, MarkerKind] = {}
    for marker_kind, aliases in DEFAULT_MARKER_ALIASES.items():
        check_deadline_fn()
        for alias in aliases:
            check_deadline_fn()
            alias_map[alias] = marker_kind
            alias_map[alias.split(".")[-1]] = marker_kind

    active_aliases = set(marker_aliases)
    if not active_aliases:
        active_aliases = set(alias_map)
    else:
        for alias in active_aliases:
            check_deadline_fn()
            alias_map.setdefault(alias, MarkerKind.NEVER)
            if "." in alias:
                alias_map.setdefault(alias.split(".")[-1], MarkerKind.NEVER)
    return active_aliases, alias_map


def marker_kind_for_call(
    call: ast.Call,
    *,
    alias_map: Mapping[str, MarkerKind],
    check_deadline_fn: Callable[[], None],
    decorator_name_fn: Callable[..., str],
) -> MarkerKind:
    check_deadline_fn()
    name = decorator_name_fn(call.func) or ""
    if not name:
        return MarkerKind.NEVER
    if name in alias_map:
        return alias_map[name]
    return alias_map.get(name.split(".")[-1], MarkerKind.NEVER)


def never_marker_metadata(
    call: ast.Call,
    never_id: str,
    reason: str,
    *,
    marker_kind: MarkerKind,
    marker_kind_mapping: MarkerKindMappingConfig = DEFAULT_MARKER_KIND_MAPPING_CONFIG,
    check_deadline_fn: Callable[[], None],
    sort_once_fn: Callable[..., object],
) -> JSONObject:
    check_deadline_fn()
    resolved_marker_kind = resolve_marker_kind_for_profile(
        marker_kind,
        mapping_config=marker_kind_mapping,
    )
    owner = keyword_string_literal(call, "owner", check_deadline_fn=check_deadline_fn)
    expiry = keyword_string_literal(call, "expiry", check_deadline_fn=check_deadline_fn)
    links = keyword_links_literal(
        call,
        check_deadline_fn=check_deadline_fn,
        sort_once_fn=sort_once_fn,
    )
    payload = normalize_marker_payload(
        reason=reason,
        env={},
        marker_kind=resolved_marker_kind,
        owner=owner,
        expiry=expiry,
        lifecycle_state=MarkerLifecycleState.ACTIVE,
        links=tuple(cast(dict[str, str], link) for link in links),
    )
    return {
        "marker_kind": resolved_marker_kind.value,
        "marker_id": marker_identity(payload),
        "marker_site_id": never_id,
        "owner": owner,
        "expiry": expiry,
        "links": links,
    }


__all__ = [
    "keyword_links_literal",
    "keyword_string_literal",
    "marker_alias_kind_map",
    "marker_kind_for_call",
    "never_marker_metadata",
    "never_reason",
]
