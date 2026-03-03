# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from gabion.analysis.json_types import JSONObject, JSONValue
from gabion.analysis.marker_protocol import (
    DEFAULT_MARKER_ALIASES,
    MarkerKind,
    MarkerLifecycleState,
    marker_identity,
    normalize_marker_payload,
)


def _keyword_string_literal(
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
        kw_value = kw.value
        if type(kw_value) is ast.Constant:
            constant_value = cast(ast.Constant, kw_value).value
            if type(constant_value) is str:
                return constant_value
    return ""


def _keyword_links_literal(
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
        kw_value = kw.value
        if type(kw_value) is not ast.List:
            return []
        links: list[JSONObject] = []
        for item in cast(ast.List, kw_value).elts:
            check_deadline_fn()
            if type(item) is not ast.Dict:
                continue
            dict_node = cast(ast.Dict, item)
            payload: JSONObject = {}
            for raw_key, raw_value in zip(dict_node.keys, dict_node.values, strict=False):
                check_deadline_fn()
                if type(raw_key) is not ast.Constant:
                    continue
                if type(raw_value) is not ast.Constant:
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
            source="indexed_scan.never_invariants.keyword_links_literal",
        )
    return []


def _never_marker_metadata(
    call: ast.Call,
    never_id: str,
    reason: str,
    *,
    marker_kind: MarkerKind = MarkerKind.NEVER,
    check_deadline_fn: Callable[[], None],
    sort_once_fn: Callable[..., object],
) -> JSONObject:
    check_deadline_fn()
    owner = _keyword_string_literal(call, "owner", check_deadline_fn=check_deadline_fn)
    expiry = _keyword_string_literal(call, "expiry", check_deadline_fn=check_deadline_fn)
    links = _keyword_links_literal(
        call,
        check_deadline_fn=check_deadline_fn,
        sort_once_fn=sort_once_fn,
    )
    payload = normalize_marker_payload(
        reason=reason,
        env={},
        marker_kind=marker_kind,
        owner=owner,
        expiry=expiry,
        lifecycle_state=MarkerLifecycleState.ACTIVE,
        links=tuple(cast(dict[str, str], link) for link in links),
    )
    return {
        "marker_kind": marker_kind.value,
        "marker_id": marker_identity(payload),
        "marker_site_id": never_id,
        "owner": owner,
        "expiry": expiry,
        "links": links,
    }


def _marker_alias_kind_map(
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
        active_aliases = set(alias_map.keys())
    else:
        for alias in active_aliases:
            check_deadline_fn()
            alias_map.setdefault(alias, MarkerKind.NEVER)
            if "." in alias:
                alias_map.setdefault(alias.split(".")[-1], MarkerKind.NEVER)
    return active_aliases, alias_map


def _marker_kind_for_call(
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
    short = name.split(".")[-1]
    return alias_map.get(short, MarkerKind.NEVER)


def never_reason(
    call: ast.Call,
    *,
    check_deadline_fn: Callable[[], None],
) -> object:
    check_deadline_fn()
    if call.args:
        first = call.args[0]
        if type(first) is ast.Constant:
            first_value = cast(ast.Constant, first).value
            if type(first_value) is str:
                return first_value
    for kw in call.keywords:
        check_deadline_fn()
        if kw.arg == "reason":
            kw_value = kw.value
            if type(kw_value) is ast.Constant:
                constant_value = cast(ast.Constant, kw_value).value
                if type(constant_value) is str:
                    return constant_value
    return None


def keyword_string_literal(
    call: ast.Call,
    key: str,
    *,
    check_deadline_fn: Callable[[], None],
) -> str:
    return _keyword_string_literal(call, key, check_deadline_fn=check_deadline_fn)


def keyword_links_literal(
    call: ast.Call,
    *,
    check_deadline_fn: Callable[[], None],
    sort_once_fn: Callable[..., object],
) -> list[JSONObject]:
    return _keyword_links_literal(
        call,
        check_deadline_fn=check_deadline_fn,
        sort_once_fn=sort_once_fn,
    )


def collect_never_invariants(
    paths: list[Path],
    *,
    project_root,
    ignore_params: set[str],
    forest,
    marker_aliases: Sequence[str],
    deadness_witnesses,
    check_deadline_fn: Callable[[], None],
    parent_annotator_factory: Callable[[], object],
    collect_functions_fn: Callable[[ast.AST], list[ast.AST]],
    param_names_fn: Callable[..., list[str]],
    normalize_snapshot_path_fn: Callable[..., str],
    enclosing_function_node_fn: Callable[..., object],
    enclosing_scopes_fn: Callable[..., list[str]],
    function_key_fn: Callable[..., str],
    exception_param_names_fn: Callable[..., list[str]],
    node_span_fn: Callable[..., object],
    dead_env_map_fn: Callable[..., dict[tuple[str, str], dict[str, tuple[JSONValue, JSONObject]]]],
    branch_reachability_under_env_fn: Callable[..., object],
    is_reachability_false_fn: Callable[..., bool],
    is_reachability_true_fn: Callable[..., bool],
    names_in_expr_fn: Callable[..., set[str]],
    sort_once_fn: Callable[..., object],
    order_policy_sort,
    order_policy_enforce,
    is_marker_call_fn: Callable[..., bool],
    decorator_name_fn: Callable[..., str],
    require_not_none_fn: Callable[..., object],
) -> list[JSONObject]:
    check_deadline_fn()
    invariants: list[JSONObject] = []
    effective_aliases, alias_kind_map = _marker_alias_kind_map(
        marker_aliases,
        check_deadline_fn=check_deadline_fn,
    )
    env_by_site = dead_env_map_fn(deadness_witnesses)
    for path in paths:
        check_deadline_fn()
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        parent_annotator = parent_annotator_factory()
        parent_annotator.visit(tree)
        parents = parent_annotator.parents
        params_by_fn: dict[ast.AST, set[str]] = {}
        for fn in collect_functions_fn(tree):
            check_deadline_fn()
            params_by_fn[fn] = set(param_names_fn(fn, ignore_params))
        path_value = normalize_snapshot_path_fn(path, project_root)
        for node in ast.walk(tree):
            check_deadline_fn()
            if type(node) is not ast.Call:
                continue
            call_node = cast(ast.Call, node)
            if not is_marker_call_fn(call_node, effective_aliases):
                continue
            fn_node = enclosing_function_node_fn(call_node, parents)
            if fn_node is None:
                function = "<module>"
                params = set()
            else:
                scopes = enclosing_scopes_fn(fn_node, parents)
                function = function_key_fn(scopes, fn_node.name)
                params = params_by_fn.get(fn_node, set())
            bundle = exception_param_names_fn(call_node, params)
            span = node_span_fn(call_node)
            lineno = getattr(call_node, "lineno", 0)
            col = getattr(call_node, "col_offset", 0)
            never_id = f"never:{path_value}:{function}:{lineno}:{col}"
            reason = str(never_reason(call_node, check_deadline_fn=check_deadline_fn) or "")
            marker_kind = _marker_kind_for_call(
                call_node,
                alias_map=alias_kind_map,
                check_deadline_fn=check_deadline_fn,
                decorator_name_fn=decorator_name_fn,
            )
            marker_metadata = _never_marker_metadata(
                call_node,
                never_id,
                reason,
                marker_kind=marker_kind,
                check_deadline_fn=check_deadline_fn,
                sort_once_fn=sort_once_fn,
            )
            status = "OBLIGATION"
            witness_ref = None
            environment_ref: JSONValue = None
            undecidable_reason = None
            env_entries = env_by_site.get((path_value, function), {})
            if env_entries:
                env = {name: value for name, (value, _) in env_entries.items()}
                reachability = branch_reachability_under_env_fn(call_node, parents, env)
                if is_reachability_false_fn(reachability):
                    names: set[str] = set()
                    current = parents.get(call_node)
                    while current is not None:
                        check_deadline_fn()
                        if type(current) is ast.If:
                            names.update(names_in_expr_fn(cast(ast.If, current).test))
                        current = parents.get(current)
                    ordered_names = sort_once_fn(
                        names,
                        source="indexed_scan.never_invariants.names.proven_unreachable",
                        policy=order_policy_sort,
                    )
                    for name in sort_once_fn(
                        ordered_names,
                        source="indexed_scan.never_invariants.names.proven_unreachable.enforce",
                        policy=order_policy_enforce,
                    ):
                        check_deadline_fn()
                        if name not in env_entries:
                            continue
                        _, witness = env_entries[name]
                        status = "PROVEN_UNREACHABLE"
                        witness_ref = witness.get("deadness_id")
                        environment_ref = witness.get("environment") or {}
                        break
                    if status == "PROVEN_UNREACHABLE" and not environment_ref:
                        environment_ref = env
                elif is_reachability_true_fn(reachability):
                    status = "VIOLATION"
                    environment_ref = env
                else:
                    names: set[str] = set()
                    current = parents.get(call_node)
                    while current is not None:
                        check_deadline_fn()
                        if type(current) is ast.If:
                            names.update(names_in_expr_fn(cast(ast.If, current).test))
                        current = parents.get(current)
                    undecidable_params = sort_once_fn(
                        (n for n in names if n not in env_entries),
                        source="indexed_scan.never_invariants.undecidable_params",
                        policy=order_policy_sort,
                    )
                    if undecidable_params:
                        undecidable_reason = f"depends on params: {', '.join(undecidable_params)}"
            entry: JSONObject = {
                "never_id": never_id,
                "site": {
                    "path": path_value,
                    "function": function,
                    "bundle": bundle,
                },
                "status": status,
                "reason": reason,
                "marker_kind": marker_metadata.get("marker_kind", MarkerKind.NEVER.value),
                "marker_id": marker_metadata.get("marker_id", never_id),
                "marker_site_id": marker_metadata.get("marker_site_id", never_id),
                "owner": marker_metadata.get("owner", ""),
                "expiry": marker_metadata.get("expiry", ""),
                "links": marker_metadata.get("links", []),
            }
            normalized_span = span or (lineno, col, lineno, col)
            if undecidable_reason:
                entry["undecidable_reason"] = undecidable_reason
            if witness_ref is not None:
                entry["witness_ref"] = witness_ref
            if environment_ref is not None:
                entry["environment_ref"] = environment_ref
            entry["span"] = list(normalized_span)
            invariants.append(entry)
            site_id = forest.add_suite_site(
                path.name,
                function,
                "call",
                span=normalized_span,
            )
            suite_node = require_not_none_fn(
                forest.nodes.get(site_id),
                reason="suite site missing from forest",
                strict=True,
                path=path_value,
                function=function,
            )
            site_payload = cast(dict[str, object], entry["site"])
            site_payload["suite_id"] = str(suite_node.meta.get("suite_id", "") or "")
            site_payload["suite_kind"] = "call"
            paramset_id = forest.add_paramset(bundle)
            evidence: dict[str, object] = {"path": path.name, "qual": function}
            if reason:
                evidence["reason"] = reason
            evidence["marker_id"] = str(marker_metadata.get("marker_id", never_id))
            evidence["marker_site_id"] = str(marker_metadata.get("marker_site_id", never_id))
            marker_links = marker_metadata.get("links")
            if type(marker_links) is list and marker_links:
                evidence["links"] = marker_links
            marker_owner = str(marker_metadata.get("owner", "")).strip()
            if marker_owner:
                evidence["owner"] = marker_owner
            marker_expiry = str(marker_metadata.get("expiry", "")).strip()
            if marker_expiry:
                evidence["expiry"] = marker_expiry
            evidence["span"] = list(normalized_span)
            forest.add_alt("NeverInvariantSink", (site_id, paramset_id), evidence=evidence)
    return sort_once_fn(
        invariants,
        key=lambda entry: (
            str(entry.get("site", {}).get("path", "")),
            str(entry.get("site", {}).get("function", "")),
            ",".join(entry.get("site", {}).get("bundle", []) or []),
            str(entry.get("never_id", "")),
        ),
        source="indexed_scan.never_invariants.collect_never_invariants",
    )


__all__ = [
    "collect_never_invariants",
    "keyword_links_literal",
    "keyword_string_literal",
    "never_reason",
]
