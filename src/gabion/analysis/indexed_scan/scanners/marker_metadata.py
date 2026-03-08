from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import singledispatch

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
from gabion.invariants import never


@dataclass(frozen=True)
class _StringLiteralCarrier:
    is_string: bool
    text: str


def _not_string_literal(value: str) -> _StringLiteralCarrier:
    _ = value
    return _StringLiteralCarrier(is_string=False, text="")


_EXPR_NODE_TYPES: tuple[type[ast.expr], ...] = (
    ast.Attribute,
    ast.Await,
    ast.BinOp,
    ast.BoolOp,
    ast.Call,
    ast.Compare,
    ast.Constant,
    ast.Dict,
    ast.DictComp,
    ast.FormattedValue,
    ast.GeneratorExp,
    ast.IfExp,
    ast.Interpolation,
    ast.JoinedStr,
    ast.Lambda,
    ast.List,
    ast.ListComp,
    ast.Name,
    ast.NamedExpr,
    ast.Set,
    ast.SetComp,
    ast.Slice,
    ast.Starred,
    ast.Subscript,
    ast.TemplateStr,
    ast.Tuple,
    ast.UnaryOp,
    ast.Yield,
    ast.YieldFrom,
)


@singledispatch
def _string_literal_from_constant_value(value: str) -> _StringLiteralCarrier:
    never("unregistered runtime type", value_type=type(value).__name__)


@_string_literal_from_constant_value.register
def _(value: str) -> _StringLiteralCarrier:
    return _StringLiteralCarrier(is_string=True, text=value)


for _runtime_type in (int, float, complex, bool, bytes, tuple, frozenset, type(None)):
    _string_literal_from_constant_value.register(_runtime_type)(_not_string_literal)


@singledispatch
def _string_literal_from_node(node: ast.AST) -> _StringLiteralCarrier:
    never("unregistered runtime type", value_type=type(node).__name__)


@_string_literal_from_node.register
def _(node: ast.Constant) -> _StringLiteralCarrier:
    return _string_literal_from_constant_value(node.value)


def _string_literal_from_non_constant_node(node: ast.AST) -> _StringLiteralCarrier:
    _ = node
    return _StringLiteralCarrier(is_string=False, text="")


for _node_type in _EXPR_NODE_TYPES:
    if _node_type is not ast.Constant:
        _string_literal_from_node.register(_node_type)(_string_literal_from_non_constant_node)


@singledispatch
def _list_elements_from_node(node: ast.AST) -> tuple[ast.AST, ...]:
    never("unregistered runtime type", value_type=type(node).__name__)


@_list_elements_from_node.register
def _(node: ast.List) -> tuple[ast.AST, ...]:
    return tuple(node.elts)


def _list_elements_from_non_list_node(node: ast.AST) -> tuple[ast.AST, ...]:
    _ = node
    return ()


for _node_type in _EXPR_NODE_TYPES:
    if _node_type is not ast.List:
        _list_elements_from_node.register(_node_type)(_list_elements_from_non_list_node)


@singledispatch
def _dict_items_from_node(node: ast.AST) -> tuple[tuple[ast.expr, ast.expr], ...]:
    never("unregistered runtime type", value_type=type(node).__name__)


@_dict_items_from_node.register
def _(node: ast.Dict) -> tuple[tuple[ast.expr, ast.expr], ...]:
    pairs: list[tuple[ast.expr, ast.expr]] = []
    for raw_key, raw_value in zip(node.keys, node.values, strict=False):
        if raw_key is not None:
            pairs.append((raw_key, raw_value))
    return tuple(pairs)


def _dict_items_from_non_dict_node(node: ast.AST) -> tuple[tuple[ast.expr, ast.expr], ...]:
    _ = node
    return ()


for _node_type in _EXPR_NODE_TYPES:
    if _node_type is not ast.Dict:
        _dict_items_from_node.register(_node_type)(_dict_items_from_non_dict_node)


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
        value_carrier = _string_literal_from_node(kw.value)
        if value_carrier.is_string:
            return value_carrier.text
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
        list_items = _list_elements_from_node(kw.value)
        if not list_items:
            return []
        links: list[JSONObject] = []
        for item in list_items:
            check_deadline_fn()
            payload: JSONObject = {}
            for raw_key, raw_value in _dict_items_from_node(item):
                check_deadline_fn()
                key_carrier = _string_literal_from_node(raw_key)
                value_carrier = _string_literal_from_node(raw_value)
                if not key_carrier.is_string or not value_carrier.is_string:
                    continue
                payload[key_carrier.text] = value_carrier.text
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
    if call.args:
        arg_reason = _string_literal_from_node(call.args[0])
        if arg_reason.is_string:
            return arg_reason.text
    for kw in call.keywords:
        check_deadline_fn()
        if kw.arg != "reason":
            continue
        kw_reason = _string_literal_from_node(kw.value)
        if kw_reason.is_string:
            return kw_reason.text
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
