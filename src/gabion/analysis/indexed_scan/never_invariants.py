# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from gabion.analysis.json_types import JSONObject
from gabion.analysis.marker_protocol import MarkerKind

from .ast_context import (
    enclosing_function_context,
)
from .context_walkers import (
    empty_param_annotations,
    iter_nodes_of_types,
    iter_parsed_path_contexts,
)
from .marker_metadata import (
    keyword_links_literal,
    keyword_string_literal,
    marker_alias_kind_map,
    marker_kind_for_call,
    never_marker_metadata,
    never_reason,
)
from .reachability import decide_never_reachability


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
    effective_aliases, alias_kind_map = marker_alias_kind_map(
        marker_aliases,
        check_deadline_fn=check_deadline_fn,
    )
    env_by_site = dead_env_map_fn(deadness_witnesses)
    for context in iter_parsed_path_contexts(
        paths,
        project_root=project_root,
        ignore_params=ignore_params,
        check_deadline_fn=check_deadline_fn,
        parent_annotator_factory=parent_annotator_factory,
        collect_functions_fn=collect_functions_fn,
        param_names_fn=param_names_fn,
        normalize_snapshot_path_fn=normalize_snapshot_path_fn,
        param_annotations_fn=empty_param_annotations,
    ):
        for node in iter_nodes_of_types(
            context.tree,
            (ast.Call,),
            check_deadline_fn=check_deadline_fn,
        ):
            call_node = cast(ast.Call, node)
            if not is_marker_call_fn(call_node, effective_aliases):
                continue

            function, params, _ = enclosing_function_context(
                call_node,
                parents=context.parents,
                params_by_fn=context.params_by_fn,
                param_annotations_by_fn=context.param_annotations_by_fn,
                enclosing_function_node_fn=enclosing_function_node_fn,
                enclosing_scopes_fn=enclosing_scopes_fn,
                function_key_fn=function_key_fn,
            )

            bundle = exception_param_names_fn(call_node, params)
            span = node_span_fn(call_node)
            lineno = getattr(call_node, "lineno", 0)
            col = getattr(call_node, "col_offset", 0)
            never_id = f"never:{context.path_value}:{function}:{lineno}:{col}"
            reason = str(never_reason(call_node, check_deadline_fn=check_deadline_fn) or "")

            marker_kind = marker_kind_for_call(
                call_node,
                alias_map=alias_kind_map,
                check_deadline_fn=check_deadline_fn,
                decorator_name_fn=decorator_name_fn,
            )
            marker_metadata = never_marker_metadata(
                call_node,
                never_id,
                reason,
                marker_kind=marker_kind,
                check_deadline_fn=check_deadline_fn,
                sort_once_fn=sort_once_fn,
            )

            reachability_decision = decide_never_reachability(
                call_node,
                parents=context.parents,
                env_entries=env_by_site.get((context.path_value, function), {}),
                branch_reachability_under_env_fn=branch_reachability_under_env_fn,
                is_reachability_false_fn=is_reachability_false_fn,
                is_reachability_true_fn=is_reachability_true_fn,
                names_in_expr_fn=names_in_expr_fn,
                sort_once_fn=sort_once_fn,
                order_policy_sort=order_policy_sort,
                order_policy_enforce=order_policy_enforce,
                check_deadline_fn=check_deadline_fn,
            )

            entry: JSONObject = {
                "never_id": never_id,
                "site": {
                    "path": context.path_value,
                    "function": function,
                    "bundle": bundle,
                },
                "status": reachability_decision.status,
                "reason": reason,
                "marker_kind": marker_metadata.get("marker_kind", MarkerKind.NEVER.value),
                "marker_id": marker_metadata.get("marker_id", never_id),
                "marker_site_id": marker_metadata.get("marker_site_id", never_id),
                "owner": marker_metadata.get("owner", ""),
                "expiry": marker_metadata.get("expiry", ""),
                "links": marker_metadata.get("links", []),
            }
            normalized_span = span or (lineno, col, lineno, col)
            if reachability_decision.undecidable_reason:
                entry["undecidable_reason"] = reachability_decision.undecidable_reason
            if reachability_decision.witness_ref is not None:
                entry["witness_ref"] = reachability_decision.witness_ref
            if reachability_decision.environment_ref is not None:
                entry["environment_ref"] = reachability_decision.environment_ref
            entry["span"] = list(normalized_span)
            invariants.append(entry)

            site_id = forest.add_suite_site(
                Path(context.path_value).name,
                function,
                "call",
                span=normalized_span,
            )
            suite_node = require_not_none_fn(
                forest.nodes.get(site_id),
                reason="suite site missing from forest",
                strict=True,
                path=context.path_value,
                function=function,
            )
            site_payload = cast(dict[str, object], entry["site"])
            site_payload["suite_id"] = str(suite_node.meta.get("suite_id", "") or "")
            site_payload["suite_kind"] = "call"
            paramset_id = forest.add_paramset(bundle)

            evidence: dict[str, object] = {"path": Path(context.path_value).name, "qual": function}
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
