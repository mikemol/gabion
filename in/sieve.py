#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def bucket_int(
    value: int,
    *,
    bounds=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
) -> str:
    for bound in bounds:
        if value <= bound:
            return f"<= {bound}"
    return f"> {bounds[-1]}"


def jaccard(left: Set[int], right: Set[int]) -> float:
    if not left and not right:
        return 1.0
    inter = len(left & right)
    union = len(left | right)
    return inter / union if union else 0.0


def cosine_sparse(left: Counter[int], right: Counter[int]) -> float:
    if not left and not right:
        return 1.0
    dot = 0.0
    for key, left_value in left.items():
        right_value = right.get(key)
        if right_value:
            dot += left_value * right_value
    left_norm = sum(v * v for v in left.values()) ** 0.5
    right_norm = sum(v * v for v in right.values()) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def parse_focus_files(raw: str) -> List[str]:
    if not raw.strip():
        return []
    values = [part.strip() for part in raw.split(",")]
    seen: Set[str] = set()
    ordered: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def default_sim_threshold(metric: str) -> float:
    if metric == "cosine":
        return 0.70
    return 0.55


@dataclass
class Node:
    kind: str
    file: str
    qualname: str
    sym_type: str | None
    loc_introduced: int


def load_scout(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes_raw: Dict[str, dict] = data["nodes"]
    edges_raw: List[dict] = data["edges"]

    nodes: Dict[str, Node] = {}
    for key, value in nodes_raw.items():
        nid = value["node_id"]
        nodes[key] = Node(
            kind=nid["kind"],
            file=nid["file"],
            qualname=nid["qualname"],
            sym_type=value.get("sym_type"),
            loc_introduced=int(value.get("loc_introduced", 0)),
        )

    edges: List[Tuple[str, str, str, int]] = []
    for edge in edges_raw:
        kind = edge["kind"]
        if kind in {"containment", "import"}:
            continue
        src = edge["src"]
        dst = edge["dst"]
        weight = int(edge["weight"])
        if src in nodes and dst in nodes and nodes[src].kind == "sym" and nodes[dst].kind == "sym":
            edges.append((src, dst, kind, weight))

    file_rows: Dict[str, dict] = {}
    for file_row in data.get("files", []):
        file_path = str(file_row.get("file", "") or "")
        if file_path:
            file_rows[file_path] = file_row

    return data["budget"], str(data.get("budget_metric", "code")), nodes, edges, file_rows


def tarjan_scc(nodes: List[str], succ: Dict[str, Set[str]]) -> List[List[str]]:
    index = 0
    stack: List[str] = []
    on_stack: Set[str] = set()
    idx: Dict[str, int] = {}
    low: Dict[str, int] = {}
    out: List[List[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        idx[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in succ.get(v, set()):
            if w not in idx:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], idx[w])

        if low[v] == idx[v]:
            comp: List[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            out.append(comp)

    for v in nodes:
        if v not in idx:
            strongconnect(v)
    return out


def build_scc_clusters(nodes: Dict[str, Node], edges: List[Tuple[str, str, str, int]]):
    file_syms: Dict[str, List[str]] = defaultdict(list)
    for key, node in nodes.items():
        if node.kind == "sym" and "." not in node.qualname:
            file_syms[node.file].append(key)

    edges_by_file: Dict[str, List[Tuple[str, str, str, int]]] = defaultdict(list)
    for src, dst, kind, weight in edges:
        if nodes[src].file == nodes[dst].file:
            edges_by_file[nodes[src].file].append((src, dst, kind, weight))

    clusters: Dict[str, List[str]] = {}
    cluster_meta: Dict[str, dict] = {}

    for file_path, syms in file_syms.items():
        sorted_syms = sorted(syms)
        succ: Dict[str, Set[str]] = defaultdict(set)
        for src, dst, _kind, _weight in edges_by_file.get(file_path, []):
            if src in syms and dst in syms:
                succ[src].add(dst)
        comps = tarjan_scc(sorted_syms, succ)
        for idx, comp in enumerate(comps):
            cid = f"atom:{file_path}:{idx}"
            sorted_members = sorted(comp)
            clusters[cid] = sorted_members
            cluster_meta[cid] = {
                "file": file_path,
                "size": len(sorted_members),
                "loc": sum(nodes[key].loc_introduced for key in sorted_members),
            }

    return clusters, cluster_meta, edges_by_file


def wl_labels(
    *,
    members: List[str],
    edges_internal: List[Tuple[str, str, str, int]],
    nodes: Dict[str, Node],
    rounds: int = 2,
) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for member in members:
        node = nodes[member]
        labels[member] = f"{node.sym_type}|loc{bucket_int(node.loc_introduced)}"

    out_adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    in_adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for src, dst, kind, _weight in edges_internal:
        if src in labels and dst in labels:
            out_adj[src].append((kind, dst))
            in_adj[dst].append((kind, src))

    for _ in range(rounds):
        new_labels: Dict[str, str] = {}
        for member in members:
            out_sig = sorted((kind, labels[dst]) for (kind, dst) in out_adj.get(member, []))
            in_sig = sorted((kind, labels[src]) for (kind, src) in in_adj.get(member, []))
            payload = {"self": labels[member], "out": out_sig, "in": in_sig}
            new_labels[member] = stable_hash(json.dumps(payload, sort_keys=True))
        labels = new_labels

    return labels


def motifs_for_cluster(
    *,
    members: List[str],
    edges_internal: List[Tuple[str, str, str, int]],
    nodes: Dict[str, Node],
    feature_intern: Dict[str, int],
    wl_rounds: int = 2,
    include_paths2: bool = True,
) -> Tuple[Set[int], Counter[int]]:
    labels = wl_labels(members=members, edges_internal=edges_internal, nodes=nodes, rounds=wl_rounds)
    member_set = set(members)

    out_edges: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
    for src, dst, kind, weight in edges_internal:
        if src in member_set and dst in member_set:
            out_edges[src].append((kind, dst, weight))

    feats_set: Set[int] = set()
    feats_count: Counter[int] = Counter()

    def intern(token: str) -> int:
        if token not in feature_intern:
            feature_intern[token] = len(feature_intern) + 1
        return feature_intern[token]

    for src, dst, kind, weight in edges_internal:
        if src not in member_set or dst not in member_set:
            continue
        token = f"E|{labels[src]}|{kind}|{labels[dst]}"
        feature_id = intern(token)
        feats_set.add(feature_id)
        feats_count[feature_id] += max(1, weight)

    if include_paths2:
        for src in members:
            for kind1, mid, weight1 in out_edges.get(src, []):
                for kind2, dst, weight2 in out_edges.get(mid, []):
                    token = f"P2|{labels[src]}|{kind1}|{labels[mid]}|{kind2}|{labels[dst]}"
                    feature_id = intern(token)
                    feats_set.add(feature_id)
                    feats_count[feature_id] += max(1, weight1) + max(1, weight2)

    return feats_set, feats_count


def topk_similar(
    *,
    cluster_ids: List[str],
    features_set: Dict[str, Set[int]],
    features_count: Dict[str, Counter[int]],
    k: int,
    use: str,
) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}

    inv: Dict[int, List[str]] = defaultdict(list)
    for cid in cluster_ids:
        for feature in features_set[cid]:
            inv[feature].append(cid)

    for cid in cluster_ids:
        candidates: Set[str] = set()
        for feature in features_set[cid]:
            for other in inv[feature]:
                if other != cid:
                    candidates.add(other)

        scored: List[Tuple[float, str]] = []
        for other in candidates:
            if use == "jaccard":
                score = jaccard(features_set[cid], features_set[other])
            else:
                score = cosine_sparse(features_count[cid], features_count[other])
            scored.append((score, other))

        scored.sort(reverse=True)
        out[cid] = [{"other": other, "score": float(score)} for score, other in scored[:k]]

    return out


def build_candidate_groups_for_file(
    *,
    file_path: str,
    file_cluster_ids: List[str],
    clusters: Dict[str, List[str]],
    cluster_meta: Dict[str, dict],
    nodes: Dict[str, Node],
    similarity_topk: Dict[str, List[dict]],
    sim_threshold: float,
    max_groups_per_file: int,
) -> List[dict]:
    file_cluster_set = set(file_cluster_ids)
    seeds = sorted(
        file_cluster_ids,
        key=lambda cid: (int(cluster_meta[cid]["loc"]), cid),
        reverse=True,
    )

    candidates: List[dict] = []
    seen_signatures: Set[Tuple[str, ...]] = set()

    for seed in seeds:
        group: Set[str] = {seed}
        frontier: List[str] = [seed]
        similarity_scores: List[float] = []

        while frontier:
            current = frontier.pop(0)
            for link in similarity_topk.get(current, []):
                other = str(link["other"])
                score = float(link["score"])
                if score < sim_threshold:
                    continue
                if other not in file_cluster_set or other in group:
                    continue
                group.add(other)
                frontier.append(other)
                similarity_scores.append(score)

        signature = tuple(sorted(group))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        group_loc = sum(int(cluster_meta[cid]["loc"]) for cid in signature)
        if group_loc <= 0:
            continue
        mean_similarity = (
            (sum(similarity_scores) / len(similarity_scores)) if similarity_scores else 0.0
        )
        score = group_loc * (1.0 + mean_similarity)

        member_node_keys = sorted(
            {node_key for cid in signature for node_key in clusters[cid]}
        )
        member_symbols = sorted(
            {nodes[key].qualname for key in member_node_keys if key in nodes}
        )

        candidates.append(
            {
                "group_id": f"grp:{stable_hash(file_path + '|' + ','.join(signature))}",
                "seed_cluster": seed,
                "clusters": list(signature),
                "cluster_count": len(signature),
                "group_loc": group_loc,
                "estimated_removable_loc": group_loc,
                "mean_similarity": round(mean_similarity, 6),
                "score": round(score, 6),
                "member_node_keys": member_node_keys,
                "member_symbols": member_symbols,
            }
        )

    candidates.sort(
        key=lambda row: (
            float(row["score"]),
            int(row["group_loc"]),
            str(row["group_id"]),
        ),
        reverse=True,
    )
    return candidates[:max_groups_per_file]


def build_ranked_extraction_plan(
    *,
    budget: int,
    budget_metric: str,
    focus_files: List[str],
    file_rows: Dict[str, dict],
    clusters: Dict[str, List[str]],
    cluster_meta: Dict[str, dict],
    nodes: Dict[str, Node],
    similarity_topk: Dict[str, List[dict]],
    sim_threshold: float,
    max_groups_per_file: int,
) -> List[dict]:
    if focus_files:
        target_files = list(focus_files)
    else:
        target_files = [
            file_path
            for file_path, row in file_rows.items()
            if int(row.get("loc_phys", 0)) > budget
        ]

    ranked_plan: List[dict] = []

    for file_path in target_files:
        row = file_rows.get(file_path, {})
        current_phys_loc = int(row.get("loc_phys", 0) or 0)
        meets_budget_now = current_phys_loc <= budget

        file_cluster_ids = [
            cid for cid, meta in cluster_meta.items() if str(meta.get("file", "")) == file_path
        ]

        candidate_groups = build_candidate_groups_for_file(
            file_path=file_path,
            file_cluster_ids=file_cluster_ids,
            clusters=clusters,
            cluster_meta=cluster_meta,
            nodes=nodes,
            similarity_topk=similarity_topk,
            sim_threshold=sim_threshold,
            max_groups_per_file=max_groups_per_file,
        )

        remaining = current_phys_loc
        used_clusters: Set[str] = set()
        selected_groups: List[dict] = []

        for group in candidate_groups:
            if remaining <= budget:
                break
            group_clusters = set(group["clusters"])
            if used_clusters & group_clusters:
                continue
            removable = max(0, int(group["estimated_removable_loc"]))
            if removable <= 0:
                continue

            remaining = max(0, remaining - removable)
            selected_groups.append(
                {
                    "group_id": group["group_id"],
                    "estimated_removable_loc": removable,
                    "score": group["score"],
                    "clusters": list(group["clusters"]),
                    "member_symbols": list(group["member_symbols"]),
                }
            )
            used_clusters |= group_clusters

        unresolved_over_by = max(0, remaining - budget)

        ranked_plan.append(
            {
                "file": file_path,
                "budget": budget,
                "budget_metric": budget_metric,
                "current_phys_loc": current_phys_loc,
                "meets_budget_now": meets_budget_now,
                "candidate_groups": candidate_groups,
                "convergence": {
                    "predicted_post_loc": remaining,
                    "meets_budget": remaining <= budget,
                    "selected_groups": selected_groups,
                    "unresolved_over_by": unresolved_over_by,
                },
            }
        )

    return ranked_plan


def build_correction_unit_backlog(
    *,
    ranked_plan: List[dict],
    scout_path: Path,
    similarity_path: Path,
    plan_path: Path,
    budget: int,
) -> List[dict]:
    rows: List[dict] = []
    counter = 1

    for file_plan in ranked_plan:
        file_path = str(file_plan["file"])
        current_phys = int(file_plan["current_phys_loc"])
        blocking = "yes" if current_phys > budget else "no"

        selected = list(file_plan["convergence"]["selected_groups"])
        if not selected and current_phys > budget:
            candidates = list(file_plan["candidate_groups"])
            if candidates:
                selected = [
                    {
                        "group_id": candidates[0]["group_id"],
                        "estimated_removable_loc": candidates[0]["estimated_removable_loc"],
                        "member_symbols": candidates[0]["member_symbols"],
                    }
                ]
            else:
                selected = [
                    {
                        "group_id": "none",
                        "estimated_removable_loc": 0,
                        "member_symbols": [],
                    }
                ]

        for group in selected:
            target_cu = f"SCOUT-CU-{counter:03d}"
            debt_id = f"SCOUT-{stable_hash(file_path + '|' + str(group['group_id']) + '|' + target_cu)}"
            symbol_sample = list(group.get("member_symbols", []))[:6]
            sample_text = ", ".join(symbol_sample)
            fix_action = (
                f"extract group {group['group_id']} from {file_path} "
                f"(estimated_removable_loc={int(group.get('estimated_removable_loc', 0))})"
            )
            if sample_text:
                fix_action += f"; symbol_sample={sample_text}"

            rows.append(
                {
                    "debt_id": debt_id,
                    "surface": file_path,
                    "signal_source": "paginator+sieve ranked extraction scout",
                    "blocking?": blocking,
                    "target_cu": target_cu,
                    "status": "open",
                    "evidence_links": "; ".join(
                        [
                            str(scout_path),
                            str(similarity_path),
                            str(plan_path),
                            f"group={group['group_id']}",
                        ]
                    ),
                    "owner": "codex",
                    "expiry": "",
                    "fix_forward_action": fix_action,
                }
            )
            counter += 1

    return rows


def write_backlog_markdown(rows: List[dict], path: Path) -> None:
    headers = [
        "debt_id",
        "surface",
        "signal_source",
        "blocking?",
        "target_cu",
        "status",
        "evidence_links",
        "owner",
        "expiry",
        "fix_forward_action",
    ]

    def esc(value: object) -> str:
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(esc(row.get(h, "")) for h in headers) + " |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute internal-wiring similarity between SCC atoms from scout.json"
    )
    parser.add_argument("--scout", type=Path, default=Path("out/scout/scout.json"))
    parser.add_argument("--out", type=Path, default=Path("out/scout/similarity_atoms.json"))
    parser.add_argument("--plan-out", type=Path, default=Path("out/scout/refactor_plan.json"))
    parser.add_argument("--focus-files", default="", help="Comma-separated repo-relative files to plan.")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--wl-rounds", type=int, default=2)
    parser.add_argument("--no-path2", action="store_true")
    parser.add_argument("--metric", choices=["jaccard", "cosine"], default="jaccard")
    parser.add_argument("--sim-threshold", type=float, default=None)
    parser.add_argument("--max-groups-per-file", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    budget, budget_metric, nodes, edges, file_rows = load_scout(args.scout)
    clusters, meta, edges_by_file = build_scc_clusters(nodes, edges)

    feature_intern: Dict[str, int] = {}
    feats_set: Dict[str, Set[int]] = {}
    feats_cnt: Dict[str, Counter[int]] = {}

    for cid, members in clusters.items():
        file_path = str(meta[cid]["file"])
        member_set = set(members)
        internal_edges = [
            edge
            for edge in edges_by_file.get(file_path, [])
            if edge[0] in member_set and edge[1] in member_set
        ]
        sset, scnt = motifs_for_cluster(
            members=members,
            edges_internal=internal_edges,
            nodes=nodes,
            feature_intern=feature_intern,
            wl_rounds=args.wl_rounds,
            include_paths2=(not args.no_path2),
        )
        feats_set[cid] = sset
        feats_cnt[cid] = scnt

    cluster_ids = sorted(clusters.keys())
    similarity_topk = topk_similar(
        cluster_ids=cluster_ids,
        features_set=feats_set,
        features_count=feats_cnt,
        k=args.topk,
        use=args.metric,
    )

    sim_threshold = (
        float(args.sim_threshold)
        if args.sim_threshold is not None
        else default_sim_threshold(args.metric)
    )
    focus_files = parse_focus_files(args.focus_files)

    ranked_plan = build_ranked_extraction_plan(
        budget=budget,
        budget_metric=budget_metric,
        focus_files=focus_files,
        file_rows=file_rows,
        clusters=clusters,
        cluster_meta=meta,
        nodes=nodes,
        similarity_topk=similarity_topk,
        sim_threshold=sim_threshold,
        max_groups_per_file=args.max_groups_per_file,
    )

    cluster_members = {cluster_id: list(members) for cluster_id, members in clusters.items()}

    similarity_payload = {
        "budget": budget,
        "budget_metric": budget_metric,
        "metric": args.metric,
        "wl_rounds": args.wl_rounds,
        "include_paths2": (not args.no_path2),
        "sim_threshold": sim_threshold,
        "focus_files": focus_files,
        "clusters": meta,
        "cluster_members": cluster_members,
        "similarity_topk": similarity_topk,
        "feature_vocab_size": len(feature_intern),
        "ranked_extraction_plan": ranked_plan,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(similarity_payload, indent=2, sort_keys=True), encoding="utf-8")

    backlog_rows = build_correction_unit_backlog(
        ranked_plan=ranked_plan,
        scout_path=args.scout,
        similarity_path=args.out,
        plan_path=args.plan_out,
        budget=budget,
    )

    plan_payload = {
        "budget": budget,
        "budget_metric": budget_metric,
        "focus_files": focus_files,
        "metric": args.metric,
        "sim_threshold": sim_threshold,
        "max_groups_per_file": args.max_groups_per_file,
        "ranked_extraction_plan": ranked_plan,
        "correction_unit_backlog": backlog_rows,
    }
    args.plan_out.parent.mkdir(parents=True, exist_ok=True)
    args.plan_out.write_text(json.dumps(plan_payload, indent=2, sort_keys=True), encoding="utf-8")

    backlog_md_path = args.plan_out.with_name("refactor_backlog.md")
    write_backlog_markdown(backlog_rows, backlog_md_path)

    print(f"Wrote {args.out} (clusters={len(cluster_ids)}, vocab={len(feature_intern)})")
    print(f"Wrote {args.plan_out} (files={len(ranked_plan)}, backlog_rows={len(backlog_rows)})")
    print(f"Wrote {backlog_md_path}")


if __name__ == "__main__":
    main()
