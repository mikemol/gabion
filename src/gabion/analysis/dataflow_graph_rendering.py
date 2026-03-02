# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import os
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from gabion.analysis.aspf import Alt, Forest, NodeId
from gabion.analysis.json_types import JSONObject, JSONValue
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import sort_once


@dataclass(frozen=True)
class BundleProjection:
    nodes: dict[NodeId, dict[str, str]]
    adj: dict[NodeId, set[NodeId]]
    bundle_map: dict[NodeId, tuple[str, ...]]
    bundle_counts: dict[tuple[str, ...], int]
    declared_global: set[tuple[str, ...]]
    declared_by_path: dict[str, set[tuple[str, ...]]]
    documented_by_path: dict[str, set[tuple[str, ...]]]
    root: Path
    path_lookup: dict[str, Path]


def _alt_input(alt: Alt, kind: str):
    for node_id in alt.inputs:
        check_deadline()
        if node_id.kind == kind:
            return node_id
    return None


def _paramset_key(forest: Forest, paramset_id: NodeId) -> tuple[str, ...]:
    node = forest.nodes.get(paramset_id)
    if node is not None:
        params = node.meta.get("params")
        if type(params) is list:
            return tuple(str(p) for p in cast(list[JSONValue], params))
    return tuple(str(p) for p in paramset_id.key)


def has_bundles(groups_by_path: dict[Path, dict[str, list[set[str]]]]) -> bool:
    check_deadline()
    for groups in groups_by_path.values():
        check_deadline()
        for bundles in groups.values():
            check_deadline()
            if bundles:
                return True
    return False


def bundle_projection_from_forest(
    forest: Forest,
    *,
    file_paths: list[Path],
) -> BundleProjection:
    check_deadline()
    nodes: dict[NodeId, dict[str, str]] = {}
    adj: dict[NodeId, set[NodeId]] = defaultdict(set)
    bundle_map: dict[NodeId, tuple[str, ...]] = {}
    bundle_counts: dict[tuple[str, ...], int] = defaultdict(int)

    for alt in forest.alts:
        check_deadline()
        if alt.kind == "SignatureBundle":
            site_id = _alt_input(alt, "FunctionSite")
            paramset_id = _alt_input(alt, "ParamSet")
            if site_id is not None and paramset_id is not None:
                site_node = forest.nodes.get(site_id)
                if site_node is not None:
                    path = str(site_node.meta.get("path", "?"))
                    qual = str(site_node.meta.get("qual", "?"))
                    nodes[site_id] = {
                        "kind": "fn",
                        "label": f"{path}:{qual}",
                        "path": path,
                        "qual": qual,
                    }
                    bundle_key = _paramset_key(forest, paramset_id)
                    nodes[paramset_id] = {
                        "kind": "bundle",
                        "label": ", ".join(bundle_key),
                    }
                    bundle_map[paramset_id] = bundle_key
                    adj[site_id].add(paramset_id)
                    adj[paramset_id].add(site_id)
                    bundle_counts[bundle_key] += 1

    declared_global: set[tuple[str, ...]] = set()
    declared_by_path: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    documented_by_path: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for alt in forest.alts:
        check_deadline()
        paramset_id = _alt_input(alt, "ParamSet")
        if paramset_id is not None:
            bundle_key = _paramset_key(forest, paramset_id)
            if alt.kind == "ConfigBundle":
                declared_global.add(bundle_key)
                path = str(alt.evidence.get("path") or "")
                if path:
                    declared_by_path[path].add(bundle_key)
            elif alt.kind in ("MarkerBundle", "DataclassCallBundle"):
                path = str(alt.evidence.get("path") or "")
                if path:
                    documented_by_path[path].add(bundle_key)

    if file_paths:
        root = Path(os.path.commonpath([str(path) for path in file_paths]))
    else:
        root = Path(".")

    path_lookup: dict[str, Path] = {}
    for path in file_paths:
        check_deadline()
        path_lookup.setdefault(path.name, path)

    return BundleProjection(
        nodes=nodes,
        adj=adj,
        bundle_map=bundle_map,
        bundle_counts=bundle_counts,
        declared_global=declared_global,
        declared_by_path=declared_by_path,
        documented_by_path=documented_by_path,
        root=root,
        path_lookup=path_lookup,
    )


def bundle_site_index(
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    bundle_sites_by_path: dict[Path, dict[str, list[list[JSONObject]]]],
) -> dict[tuple[str, str, tuple[str, ...]], list[list[JSONObject]]]:
    check_deadline()
    index: dict[tuple[str, str, tuple[str, ...]], list[list[JSONObject]]] = {}
    for path, groups in groups_by_path.items():
        check_deadline()
        fn_sites = bundle_sites_by_path.get(path, {})
        for fn_name, bundles in groups.items():
            check_deadline()
            sites = fn_sites.get(fn_name, [])
            for idx, bundle in enumerate(bundles):
                check_deadline()
                bundle_key = tuple(
                    sort_once(
                        bundle,
                        source="src/gabion/analysis/dataflow_graph_rendering.py:bundle_site_index",
                    )
                )
                entry = index.setdefault((path.name, fn_name, bundle_key), [])
                if idx < len(sites):
                    entry.append(sites[idx])
    return index


def emit_dot(forest: Forest) -> str:
    check_deadline()
    if type(forest) is not Forest:
        raise RuntimeError("forest required for dataflow dot output")
    projection = bundle_projection_from_forest(forest, file_paths=[])
    lines = [
        "digraph dataflow_grammar {",
        "  rankdir=LR;",
        "  node [fontsize=10];",
    ]
    for node_id, meta in projection.nodes.items():
        check_deadline()
        label = meta["label"].replace('"', "'")
        if meta["kind"] == "fn":
            lines.append(f'  {abs(hash(node_id.sort_key()))} [shape=box,label="{label}"];')
        else:
            lines.append(
                f'  {abs(hash(node_id.sort_key()))} [shape=ellipse,label="{label}"];'
            )
    for src, targets in projection.adj.items():
        check_deadline()
        for dst in targets:
            check_deadline()
            if projection.nodes.get(src, {}).get("kind") == "fn":
                lines.append(f"  {abs(hash(src.sort_key()))} -> {abs(hash(dst.sort_key()))};")
    lines.append("}")
    return "\n".join(lines)


def render_dot(forest: Forest) -> str:
    return emit_dot(forest)


def connected_components(
    nodes: dict[NodeId, dict[str, str]],
    adj: dict[NodeId, set[NodeId]],
) -> list[list[NodeId]]:
    check_deadline()
    seen: set[NodeId] = set()
    components: list[list[NodeId]] = []
    for node in nodes:
        check_deadline()
        if node in seen:
            continue
        queue: deque[NodeId] = deque([node])
        seen.add(node)
        component: list[NodeId] = []
        while queue:
            check_deadline()
            current = queue.popleft()
            component.append(current)
            for nxt in adj.get(current, ()):  # pragma: no branch
                check_deadline()
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append(
            sort_once(
                component,
                key=lambda node_id: node_id.sort_key(),
                source="src/gabion/analysis/dataflow_graph_rendering.py:connected_components",
            )
        )
    return components


def render_mermaid_component(
    nodes: dict[NodeId, dict[str, str]],
    bundle_map: dict[NodeId, tuple[str, ...]],
    bundle_counts: dict[tuple[str, ...], int],
    adj: dict[NodeId, set[NodeId]],
    component: list[NodeId],
    declared_global: set[tuple[str, ...]],
    declared_by_path: dict[str, set[tuple[str, ...]]],
    documented_by_path: dict[str, set[tuple[str, ...]]],
) -> tuple[str, str]:
    check_deadline()
    lines = ["```mermaid", "flowchart LR"]
    fn_nodes = [node for node in component if nodes[node]["kind"] == "fn"]
    bundle_nodes = [node for node in component if nodes[node]["kind"] == "bundle"]

    for node in fn_nodes:
        check_deadline()
        label = nodes[node]["label"].replace('"', "'")
        lines.append(f'  {abs(hash(node.sort_key()))}["{label}"]')
    for node in bundle_nodes:
        check_deadline()
        label = nodes[node]["label"].replace('"', "'")
        lines.append(f'  {abs(hash(node.sort_key()))}(({label}))')
    for node in component:
        check_deadline()
        for nxt in adj.get(node, ()):  # pragma: no branch
            check_deadline()
            if nxt in component and nodes[node]["kind"] == "fn":
                lines.append(f"  {abs(hash(node.sort_key()))} --> {abs(hash(nxt.sort_key()))}")

    lines.append("  classDef fn fill:#cfe8ff,stroke:#2b6cb0,stroke-width:1px;")
    lines.append("  classDef bundle fill:#ffe9c6,stroke:#c05621,stroke-width:1px;")
    if fn_nodes:
        lines.append(
            "  class "
            + ",".join(str(abs(hash(node.sort_key()))) for node in fn_nodes)
            + " fn;"
        )
    if bundle_nodes:
        lines.append(
            "  class "
            + ",".join(str(abs(hash(node.sort_key()))) for node in bundle_nodes)
            + " bundle;"
        )
    lines.append("```")

    observed = [bundle_map[node] for node in bundle_nodes if node in bundle_map]
    component_paths: set[str] = set()
    for node in fn_nodes:
        check_deadline()
        component_paths.add(nodes[node]["path"])

    declared_local: set[tuple[str, ...]] = set()
    documented: set[tuple[str, ...]] = set()
    for path in component_paths:
        check_deadline()
        declared_local |= declared_by_path.get(path, set())
        documented |= documented_by_path.get(path, set())

    observed_norm = {
        tuple(
            sort_once(
                bundle,
                source="src/gabion/analysis/dataflow_graph_rendering.py:render_mermaid_component.observed",
            )
        )
        for bundle in observed
    }
    observed_only = (
        sort_once(
            observed_norm - declared_global,
            source="src/gabion/analysis/dataflow_graph_rendering.py:render_mermaid_component.observed_only",
        )
        if declared_global
        else sort_once(
            observed_norm,
            source="src/gabion/analysis/dataflow_graph_rendering.py:render_mermaid_component.observed_all",
        )
    )
    declared_only = sort_once(
        declared_local - observed_norm,
        source="src/gabion/analysis/dataflow_graph_rendering.py:render_mermaid_component.declared_only",
    )
    documented_only = sort_once(
        observed_norm & documented,
        source="src/gabion/analysis/dataflow_graph_rendering.py:render_mermaid_component.documented_only",
    )

    def _tier(bundle: tuple[str, ...]) -> str:
        count = bundle_counts.get(bundle, 1)
        if count > 1:
            return "tier-2"
        return "tier-3"

    summary_lines = [
        f"Functions: {len(fn_nodes)}",
        f"Observed bundles: {len(observed_norm)}",
    ]
    if not declared_local:
        summary_lines.append("Declared Config bundles: none found for this component.")
    if observed_only:
        summary_lines.append("Observed-only bundles (not declared in Configs):")
        for bundle in observed_only:
            check_deadline()
            tier = _tier(bundle)
            documented_flag = "documented" if bundle in documented else "undocumented"
            summary_lines.append(
                f"  - {', '.join(bundle)} ({tier}, {documented_flag})"
            )
    if documented_only:
        summary_lines.append(
            "Documented bundles (dataflow-bundle markers or local dataclass calls):"
        )
        summary_lines.extend(f"  - {', '.join(bundle)}" for bundle in documented_only)
    if declared_only:
        summary_lines.append("Declared Config bundles not observed in this component:")
        summary_lines.extend(f"  - {', '.join(bundle)}" for bundle in declared_only)

    return "\n".join(lines), "\n".join(summary_lines)


def _normalize_snapshot_path(path: Path, root: object) -> str:
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def render_component_callsite_evidence(
    *,
    component: list[NodeId],
    nodes: dict[NodeId, dict[str, str]],
    bundle_map: dict[NodeId, tuple[str, ...]],
    bundle_counts: dict[tuple[str, ...], int],
    adj: dict[NodeId, set[NodeId]],
    documented_by_path: dict[str, set[tuple[str, ...]]],
    declared_global: set[tuple[str, ...]],
    bundle_site_index: dict[tuple[str, str, tuple[str, ...]], list[list[JSONObject]]],
    root: Path,
    path_lookup: dict[str, Path],
    max_sites_per_bundle: int = 5,
) -> list[str]:
    check_deadline()
    fn_nodes = [node for node in component if nodes[node]["kind"] == "fn"]
    bundle_nodes = [node for node in component if nodes[node]["kind"] == "bundle"]

    component_paths: set[str] = set()
    for node in fn_nodes:
        check_deadline()
        component_paths.add(nodes[node]["path"])

    documented: set[tuple[str, ...]] = set()
    for path in component_paths:
        check_deadline()
        documented |= documented_by_path.get(path, set())

    bundle_key_by_node: dict[NodeId, tuple[str, ...]] = {}
    for node in bundle_nodes:
        check_deadline()
        key = tuple(
            sort_once(
                bundle_map[node],
                source="src/gabion/analysis/dataflow_graph_rendering.py:render_component_callsite_evidence.key",
            )
        )
        bundle_key_by_node[node] = key

    ordered_nodes = sort_once(
        bundle_key_by_node,
        key=lambda node_id: (node_id.sort_key(), bundle_key_by_node.get(node_id, ())),
        source="src/gabion/analysis/dataflow_graph_rendering.py:render_component_callsite_evidence.ordered_nodes",
    )

    lines: list[str] = []
    for bundle_id in ordered_nodes:
        check_deadline()
        bundle_key = bundle_key_by_node[bundle_id]
        observed_only = (not declared_global) or (bundle_key not in declared_global)
        if not observed_only or bundle_key in documented:
            continue
        tier = "tier-2" if bundle_counts.get(bundle_key, 1) > 1 else "tier-3"
        adjacent_sites = [
            node_id
            for node_id in sort_once(
                adj.get(bundle_id, set()),
                key=lambda node: node.sort_key(),
                source="src/gabion/analysis/dataflow_graph_rendering.py:render_component_callsite_evidence.adjacent_sites",
            )
            if nodes.get(node_id, {}).get("kind") == "fn"
        ]
        for site_id in adjacent_sites:
            check_deadline()
            path_name = nodes[site_id]["path"]
            fn_name = nodes[site_id]["qual"]
            evidence_sets = bundle_site_index.get((path_name, fn_name, bundle_key), [])
            if not evidence_sets:
                continue
            path = path_lookup.get(path_name, Path(path_name))
            evidence_entries: list[JSONObject] = []
            for entry in evidence_sets:
                check_deadline()
                evidence_entries.extend(entry)
            for site in evidence_entries[:max_sites_per_bundle]:
                check_deadline()
                start_line, start_col, end_line, end_col = site["span"]
                loc = f"{start_line + 1}:{start_col + 1}-{end_line + 1}:{end_col + 1}"
                rel = _normalize_snapshot_path(path, root)
                callee = str(site.get("callee") or "")
                params = ", ".join(site.get("params") or [])
                slots = ", ".join(site.get("slots") or [])
                bundle_label = ", ".join(bundle_key)
                lines.append(
                    f"{rel}:{loc}: {fn_name} -> {callee} forwards {params} "
                    f"({tier}, undocumented bundle: {bundle_label}; slots: {slots})"
                )
    return lines


__all__ = [
    "BundleProjection",
    "bundle_projection_from_forest",
    "bundle_site_index",
    "connected_components",
    "emit_dot",
    "has_bundles",
    "render_component_callsite_evidence",
    "render_dot",
    "render_mermaid_component",
]
