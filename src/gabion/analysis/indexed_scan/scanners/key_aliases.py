from __future__ import annotations

import ast
from functools import singledispatch
from re import Pattern
from typing import Callable, Hashable, Mapping

from gabion.invariants import never

CacheIdentityAliasesFn = Callable[[str], tuple[str, ...]]
CheckDeadlineFn = Callable[[], None]


@singledispatch
def _is_tuple_key(value) -> bool:
    never("unregistered runtime type", value_type=type(value).__name__)


@_is_tuple_key.register(tuple)
def _is_tuple_key_tuple(value: tuple[Hashable, ...]) -> bool:
    return True


def _is_not_tuple_key(value) -> bool:
    return False


for _runtime_type in (str, int, float, bool, complex, bytes, frozenset, type(None)):
    _is_tuple_key.register(_runtime_type)(_is_not_tuple_key)


@singledispatch
def _tuple_key(value) -> tuple[Hashable, ...]:
    never("unregistered runtime type", value_type=type(value).__name__)


@_tuple_key.register(tuple)
def _tuple_key_tuple(value: tuple[Hashable, ...]) -> tuple[Hashable, ...]:
    return value


@singledispatch
def _is_string_key(value) -> bool:
    never("unregistered runtime type", value_type=type(value).__name__)


@_is_string_key.register(str)
def _is_string_key_str(value: str) -> bool:
    return True


def _is_not_string_key(value) -> bool:
    return False


for _runtime_type in (
    tuple,
    int,
    float,
    bool,
    complex,
    bytes,
    frozenset,
    type(None),
):
    _is_string_key.register(_runtime_type)(_is_not_string_key)


@singledispatch
def _string_key(value) -> str:
    never("unregistered runtime type", value_type=type(value).__name__)


@_string_key.register(str)
def _string_key_str(value: str) -> str:
    return value


@singledispatch
def _is_exact_int(value) -> bool:
    never("unregistered runtime type", value_type=type(value).__name__)


@_is_exact_int.register(int)
def _is_exact_int_int(value: int) -> bool:
    return True


def _is_not_exact_int(value) -> bool:
    return False


for _runtime_type in (
    bool,
    float,
    complex,
    str,
    bytes,
    tuple,
    list,
    dict,
    set,
    frozenset,
    type(None),
):
    _is_exact_int.register(_runtime_type)(_is_not_exact_int)


@singledispatch
def _stage_cache_key_aliases_dispatch(
    key: Hashable,
    *,
    cache_identity_aliases_fn: CacheIdentityAliasesFn,
    cache_identity_prefix: str,
    cache_identity_digest_hex: Pattern[str],
    node_id_type: type,
) -> tuple[Hashable, ...]:
    never("unregistered runtime type", value_type=type(key).__name__)


@_stage_cache_key_aliases_dispatch.register(tuple)
def _stage_cache_key_aliases_dispatch_tuple(
    key: tuple[Hashable, ...],
    *,
    cache_identity_aliases_fn: CacheIdentityAliasesFn,
    cache_identity_prefix: str,
    cache_identity_digest_hex: Pattern[str],
    node_id_type: type,
) -> tuple[Hashable, ...]:
    if len(key) == 2 and _is_tuple_key(key[1]):
        scoped_identity = key[0]
        parse_key = _tuple_key(key[1])
        parse_aliases = stage_cache_key_aliases(
            parse_key,
            cache_identity_aliases_fn=cache_identity_aliases_fn,
            cache_identity_prefix=cache_identity_prefix,
            cache_identity_digest_hex=cache_identity_digest_hex,
            node_id_type=node_id_type,
        )
        if len(parse_aliases) > 1:
            return tuple((scoped_identity, alias) for alias in parse_aliases)
        return (key,)
    if len(key) == 4 and key[0] == "parse" and _is_string_key(key[2]):
        identity = _string_key(key[2])
        aliases = cache_identity_aliases_fn(identity)
        if len(aliases) == 1 and identity.startswith(cache_identity_prefix):
            digest = identity[len(cache_identity_prefix) :]
            if cache_identity_digest_hex.fullmatch(digest):
                aliases = (aliases[0], digest)
        if len(aliases) > 1:
            return tuple((key[0], key[1], alias, key[3]) for alias in aliases)
    return (key,)


def _stage_cache_key_aliases_passthrough(
    key: Hashable,
    *,
    cache_identity_aliases_fn: CacheIdentityAliasesFn,
    cache_identity_prefix: str,
    cache_identity_digest_hex: Pattern[str],
    node_id_type: type,
) -> tuple[Hashable, ...]:
    return (key,)


for _runtime_type in (str, int, float, bool, complex, bytes, frozenset, type(None)):
    _stage_cache_key_aliases_dispatch.register(_runtime_type)(
        _stage_cache_key_aliases_passthrough
    )


_NODE_STAGE_ALIAS_TYPES: set[type] = set()


def _register_node_stage_alias_dispatch(node_id_type: type) -> None:
    if node_id_type in _NODE_STAGE_ALIAS_TYPES:
        return

    @_stage_cache_key_aliases_dispatch.register(node_id_type)
    def _stage_cache_key_aliases_dispatch_node(
        key: Hashable,
        *,
        cache_identity_aliases_fn: CacheIdentityAliasesFn,
        cache_identity_prefix: str,
        cache_identity_digest_hex: Pattern[str],
        node_id_type: type,
    ) -> tuple[Hashable, ...]:
        if getattr(key, "kind", None) != "ParseStageCacheIdentity":
            return (key,)
        raw_node_key = getattr(key, "key", None)
        if not _is_tuple_key(raw_node_key):
            return (key,)
        node_key = _tuple_key(raw_node_key)
        if len(node_key) != 3:
            return (key,)
        stage_value, identity, detail = node_key
        if _is_string_key(stage_value) and _is_string_key(identity):
            legacy_key = ("parse", _string_key(stage_value), _string_key(identity), detail)
            aliases = stage_cache_key_aliases(
                legacy_key,
                cache_identity_aliases_fn=cache_identity_aliases_fn,
                cache_identity_prefix=cache_identity_prefix,
                cache_identity_digest_hex=cache_identity_digest_hex,
                node_id_type=node_id_type,
            )
            return (key, *aliases)
        return (key,)

    _NODE_STAGE_ALIAS_TYPES.add(node_id_type)


def stage_cache_key_aliases(
    key: Hashable,
    *,
    cache_identity_aliases_fn: CacheIdentityAliasesFn,
    cache_identity_prefix: str,
    cache_identity_digest_hex: Pattern[str],
    node_id_type: type,
) -> tuple[Hashable, ...]:
    _register_node_stage_alias_dispatch(node_id_type)
    return _stage_cache_key_aliases_dispatch(
        key,
        cache_identity_aliases_fn=cache_identity_aliases_fn,
        cache_identity_prefix=cache_identity_prefix,
        cache_identity_digest_hex=cache_identity_digest_hex,
        node_id_type=node_id_type,
    )


def normalize_key_expr(
    node: ast.AST,
    *,
    const_bindings: Mapping[str, ast.AST],
    check_deadline_fn: CheckDeadlineFn,
    literal_eval_error_types: tuple[type[BaseException], ...],
) -> Hashable:
    """Normalize deterministic subscript key forms."""
    check_deadline_fn()
    node_type = type(node)
    normalized_key = None
    if node_type is ast.Constant:
        value = node.value
        value_type = type(value)
        if value_type in {str, int}:
            normalized_key = ("literal", value_type.__name__, value)
    elif node_type is ast.UnaryOp and type(node.op) in {ast.USub, ast.UAdd}:
        evaluated_value = None
        try:
            evaluated_value = ast.literal_eval(node)
        except literal_eval_error_types:
            pass
        if _is_exact_int(evaluated_value):
            normalized_key = ("literal", "int", evaluated_value)
    elif node_type is ast.Name:
        bound = const_bindings.get(node.id)
        if bound is not None:
            normalized_key = normalize_key_expr(
                bound,
                const_bindings=const_bindings,
                check_deadline_fn=check_deadline_fn,
                literal_eval_error_types=literal_eval_error_types,
            )
    elif node_type is ast.Tuple:
        items: list[Hashable] = []
        complete = True
        for elt in node.elts:
            check_deadline_fn()
            normalized_item = normalize_key_expr(
                elt,
                const_bindings=const_bindings,
                check_deadline_fn=check_deadline_fn,
                literal_eval_error_types=literal_eval_error_types,
            )
            if normalized_item is None:
                complete = False
            else:
                items.append(normalized_item)
        if complete:
            normalized_key = ("tuple", tuple(items))
    return normalized_key
