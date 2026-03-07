# gabion:decision_protocol_module
from __future__ import annotations

import ast
from re import Pattern
from typing import Callable, Hashable, Mapping

CacheIdentityAliasesFn = Callable[[str], tuple[str, ...]]
CheckDeadlineFn = Callable[[], None]


def stage_cache_key_aliases(
    key: Hashable,
    *,
    cache_identity_aliases_fn: CacheIdentityAliasesFn,
    cache_identity_prefix: str,
    cache_identity_digest_hex: Pattern[str],
    node_id_type: type,
) -> tuple[Hashable, ...]:
    if type(key) is tuple and len(key) == 2 and type(key[1]) is tuple:
        scoped_identity = key[0]
        parse_key = key[1]
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
    if type(key) is tuple and len(key) == 4 and key[0] == "parse" and type(key[2]) is str:
        identity = key[2]
        aliases = cache_identity_aliases_fn(identity)
        identity_text = str(identity)
        if len(aliases) == 1 and identity_text.startswith(cache_identity_prefix):
            digest = identity_text[len(cache_identity_prefix) :]
            if cache_identity_digest_hex.fullmatch(digest):
                aliases = (aliases[0], digest)
        if len(aliases) > 1:
            return tuple((key[0], key[1], alias, key[3]) for alias in aliases)
    if (
        type(key) is node_id_type
        and getattr(key, "kind", None) == "ParseStageCacheIdentity"
        and len(getattr(key, "key", ())) == 3
    ):
        stage_value, identity, detail = key.key
        if type(stage_value) is str and type(identity) is str:
            legacy_key = ("parse", stage_value, identity, detail)
            aliases = stage_cache_key_aliases(
                legacy_key,
                cache_identity_aliases_fn=cache_identity_aliases_fn,
                cache_identity_prefix=cache_identity_prefix,
                cache_identity_digest_hex=cache_identity_digest_hex,
                node_id_type=node_id_type,
            )
            return (key, *aliases)
    return (key,)


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
        if type(evaluated_value) is int:
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
