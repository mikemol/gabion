#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gabion.order_contract import ordered_or_sorted
from gabion.tooling.runtime.projection_fiber_semantics_summary import (
    projection_fiber_semantics_summary_from_payload,
)

ACTIVE_FAMILIES: tuple[str, ...] = (
    "branchless",
    "fiber_filter_processor_contract",
    "fiber_loop_structure_contract",
    "defensive_fallback",
    "fiber_scalar_sentinel_contract",
)

DEFAULT_MIN_SEED_FAMILIES = 5
DEFAULT_MIN_SEED_TOTAL = 80
DEFAULT_RING2_SIMILARITY = 0.98
DEFAULT_RING2_MIN_TOTAL = 60
DEFAULT_RING2_LIMIT = 8
DEFAULT_RING2_WEIGHT = 0.35
DEFAULT_RING1_BACKLOG_FILE_THRESHOLD = 20


@dataclass(frozen=True)
class QueueConfig:
    min_seed_families: int = DEFAULT_MIN_SEED_FAMILIES
    min_seed_total: int = DEFAULT_MIN_SEED_TOTAL
    ring2_similarity_threshold: float = DEFAULT_RING2_SIMILARITY
    ring2_min_total: int = DEFAULT_RING2_MIN_TOTAL
    ring2_limit: int = DEFAULT_RING2_LIMIT
    ring2_weight: float = DEFAULT_RING2_WEIGHT
    ring1_backlog_file_threshold: int = DEFAULT_RING1_BACKLOG_FILE_THRESHOLD


def _sorted[T](values: list[T], *, key=None) -> list[T]:
    return ordered_or_sorted(values, source="scripts.policy.hotspot_neighborhood_queue", key=key)


def _count_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _file_family_counts(payload: dict[str, Any], families: tuple[str, ...]) -> dict[str, Counter[str]]:
    violations = payload.get("violations")
    if not isinstance(violations, dict):
        return {}
    counts_by_file: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for family in families:
        family_items = violations.get(family, [])
        if not isinstance(family_items, list):
            continue
        for item in family_items:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            if not isinstance(path, str) or not path.strip():
                continue
            counts_by_file[path][family] += 1
    return dict(counts_by_file)


def _vector_for_path(
    *,
    counts_by_file: dict[str, Counter[str]],
    path: str,
    families: tuple[str, ...],
) -> tuple[int, ...]:
    file_counts = counts_by_file.get(path, Counter())
    return tuple(int(file_counts.get(family, 0)) for family in families)


def _file_total(counts: Counter[str]) -> int:
    return sum(int(value) for value in counts.values())


def _file_family_count(counts: Counter[str]) -> int:
    return sum(1 for value in counts.values() if value > 0)


def _cosine_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    left_norm = math.sqrt(sum(item * item for item in left))
    right_norm = math.sqrt(sum(item * item for item in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = sum(left_item * right_item for left_item, right_item in zip(left, right))
    return dot / (left_norm * right_norm)


def _ring1_scope(path: str) -> str:
    return str(Path(path).parent).replace("\\", "/")


def _seed_paths(
    *,
    counts_by_file: dict[str, Counter[str]],
    config: QueueConfig,
) -> list[str]:
    candidates = [
        path
        for path, counts in counts_by_file.items()
        if _file_family_count(counts) >= config.min_seed_families
        and _file_total(counts) >= config.min_seed_total
    ]
    return _sorted(candidates, key=lambda path: (-_file_total(counts_by_file[path]), path))


def _ring1_paths_for_seed(*, counts_by_file: dict[str, Counter[str]], seed_path: str) -> list[str]:
    scope = _ring1_scope(seed_path)
    ring1 = [path for path in counts_by_file if _ring1_scope(path) == scope]
    return _sorted(ring1, key=lambda path: (-_file_total(counts_by_file[path]), path))


def _ring2_neighbors_for_seed(
    *,
    counts_by_file: dict[str, Counter[str]],
    seed_path: str,
    ring1_paths: set[str],
    families: tuple[str, ...],
    config: QueueConfig,
) -> list[dict[str, Any]]:
    seed_vector = _vector_for_path(counts_by_file=counts_by_file, path=seed_path, families=families)
    neighbors: list[dict[str, Any]] = []
    for path, counts in counts_by_file.items():
        if path in ring1_paths or path == seed_path:
            continue
        total = _file_total(counts)
        if total < config.ring2_min_total:
            continue
        vector = _vector_for_path(counts_by_file=counts_by_file, path=path, families=families)
        similarity = _cosine_similarity(seed_vector, vector)
        if similarity < config.ring2_similarity_threshold:
            continue
        neighbors.append(
            {
                "path": path,
                "similarity": round(similarity, 6),
                "total": total,
                "family_count": _file_family_count(counts),
                "counts_by_family": {family: int(counts.get(family, 0)) for family in families},
            }
        )
    neighbors = _sorted(
        neighbors,
        key=lambda item: (
            -float(item["similarity"]),
            -int(item["total"]),
            str(item["path"]),
        ),
    )
    return neighbors[: config.ring2_limit]


def _ring1_payload(
    *,
    counts_by_file: dict[str, Counter[str]],
    ring1_paths: list[str],
    families: tuple[str, ...],
) -> dict[str, Any]:
    files_payload = [
        {
            "path": path,
            "total": _file_total(counts_by_file[path]),
            "family_count": _file_family_count(counts_by_file[path]),
            "counts_by_family": {
                family: int(counts_by_file[path].get(family, 0))
                for family in families
            },
        }
        for path in ring1_paths
    ]
    ring1_family_counts = Counter[str]()
    for path in ring1_paths:
        ring1_family_counts.update(counts_by_file[path])
    return {
        "files": files_payload,
        "file_count": len(ring1_paths),
        "total": sum(int(item["total"]) for item in files_payload),
        "family_count": sum(
            1 for family in families if int(ring1_family_counts.get(family, 0)) > 0
        ),
        "counts_by_family": {
            family: int(ring1_family_counts.get(family, 0))
            for family in families
        },
    }


def _family_totals(*, counts_by_file: dict[str, Counter[str]], families: tuple[str, ...]) -> dict[str, int]:
    totals = Counter[str]()
    for counts in counts_by_file.values():
        totals.update(counts)
    return {family: int(totals.get(family, 0)) for family in families}


def _equal_family_score(
    *,
    counts_by_family: dict[str, int],
    totals_by_family: dict[str, int],
    families: tuple[str, ...],
) -> float:
    score = 0.0
    for family in families:
        total = int(totals_by_family.get(family, 0))
        if total <= 0:
            continue
        factor = int(counts_by_family.get(family, 0)) / total
        if factor <= 0.0:
            continue
        score += math.log1p(factor)
    return score


def _projection_fiber_semantic_previews(
    *,
    summary: object | None,
) -> list[dict[str, Any]]:
    if summary is None:
        return []
    semantic_previews = getattr(summary, "semantic_previews", None)
    if not isinstance(semantic_previews, tuple):
        return []
    normalized = [
        item.as_payload()
        for item in semantic_previews
        if hasattr(item, "as_payload")
    ]
    return _sorted(
        normalized,
        key=lambda item: (
            str(item.get("spec_name", "")),
            str(item.get("quotient_face", "")),
            str(item.get("path", "")),
            str(item.get("qualname", "")),
            str(item.get("structural_path", "")),
        ),
    )
def _projection_fiber_overlap(
    *,
    semantic_previews: list[dict[str, Any]],
    ring1_paths: list[str],
    ring2_paths: list[str],
) -> dict[str, Any]:
    ring1_set = set(ring1_paths)
    ring2_set = set(ring2_paths)
    matched_previews: list[dict[str, Any]] = []
    ring1_match_count = 0
    ring2_match_count = 0
    for preview in semantic_previews:
        path = preview.get("path")
        if not isinstance(path, str) or not path.strip():
            continue
        location = ""
        if path in ring1_set:
            location = "ring_1"
            ring1_match_count += 1
        elif path in ring2_set:
            location = "ring_2"
            ring2_match_count += 1
        if not location:
            continue
        matched_previews.append(
            {
                "location": location,
                "spec_name": str(preview.get("spec_name", "")),
                "quotient_face": str(preview.get("quotient_face", "")),
                "path": path,
                "qualname": str(preview.get("qualname", "")),
                "structural_path": str(preview.get("structural_path", "")),
            }
        )
    return {
        "match_count": len(matched_previews),
        "ring_1_match_count": ring1_match_count,
        "ring_2_match_count": ring2_match_count,
        "matched_previews": matched_previews,
    }


def analyze(
    *,
    payload: dict[str, Any],
    config: QueueConfig | None = None,
) -> dict[str, Any]:
    local_config = config if config is not None else QueueConfig()
    families = ACTIVE_FAMILIES
    counts_by_file = _file_family_counts(payload, families)
    projection_fiber_summary = projection_fiber_semantics_summary_from_payload(payload)
    semantic_previews = _projection_fiber_semantic_previews(
        summary=projection_fiber_summary,
    )
    family_totals = _family_totals(counts_by_file=counts_by_file, families=families)
    seed_paths = _seed_paths(counts_by_file=counts_by_file, config=local_config)

    neighborhood_candidates: list[dict[str, Any]] = []
    for seed_path in seed_paths:
        ring1_paths = _ring1_paths_for_seed(counts_by_file=counts_by_file, seed_path=seed_path)
        ring1_payload = _ring1_payload(
            counts_by_file=counts_by_file,
            ring1_paths=ring1_paths,
            families=families,
        )
        ring2_neighbors = _ring2_neighbors_for_seed(
            counts_by_file=counts_by_file,
            seed_path=seed_path,
            ring1_paths=set(ring1_paths),
            families=families,
            config=local_config,
        )
        ring2_paths = [
            str(item["path"])
            for item in ring2_neighbors
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        ]
        ring2_total = sum(int(item["total"]) for item in ring2_neighbors)
        ring2_counts = Counter[str]()
        for neighbor in ring2_neighbors:
            counts = neighbor.get("counts_by_family")
            if not isinstance(counts, dict):
                continue
            for family in families:
                ring2_counts[family] += int(counts.get(family, 0) or 0)
        ring1_equal_family_score = _equal_family_score(
            counts_by_family={
                family: int(ring1_payload["counts_by_family"].get(family, 0))
                for family in families
            },
            totals_by_family=family_totals,
            families=families,
        )
        ring2_equal_family_score = _equal_family_score(
            counts_by_family={family: int(ring2_counts.get(family, 0)) for family in families},
            totals_by_family=family_totals,
            families=families,
        )
        ring1_balanced_component = (
            math.log1p(ring1_equal_family_score)
            if ring1_equal_family_score > 0.0
            else 0.0
        )
        ring2_balanced_component_input = (
            local_config.ring2_weight * ring2_equal_family_score
        )
        ring2_balanced_component = (
            math.log1p(ring2_balanced_component_input)
            if ring2_balanced_component_input > 0.0
            else 0.0
        )
        balanced_score = ring1_balanced_component + ring2_balanced_component
        seed_counts = counts_by_file.get(seed_path, Counter())
        neighborhood_candidates.append(
            {
                "ring_1_scope": _ring1_scope(seed_path),
                "seed_path": seed_path,
                "seed_total": _file_total(seed_counts),
                "seed_family_count": _file_family_count(seed_counts),
                "seed_counts_by_family": {
                    family: int(seed_counts.get(family, 0))
                    for family in families
                },
                "ring_1": ring1_payload,
                "ring_2": ring2_neighbors,
                "projection_fiber_overlap": _projection_fiber_overlap(
                    semantic_previews=semantic_previews,
                    ring1_paths=ring1_paths,
                    ring2_paths=ring2_paths,
                ),
                "score": {
                    "balanced": round(balanced_score, 6),
                    "ring_1_total": int(ring1_payload["total"]),
                    "ring_2_advisory_total": ring2_total,
                    "ring_1_equal_family_score": round(ring1_equal_family_score, 6),
                    "ring_2_equal_family_score": round(ring2_equal_family_score, 6),
                    "ring_1_balanced_component": round(ring1_balanced_component, 6),
                    "ring_2_balanced_component": round(ring2_balanced_component, 6),
                    "ring_2_weight": local_config.ring2_weight,
                },
            }
        )

    # Keep one representative neighborhood per ring-1 scope (highest balanced score).
    neighborhoods_by_scope: dict[str, dict[str, Any]] = {}
    for candidate in neighborhood_candidates:
        scope = str(candidate["ring_1_scope"])
        current = neighborhoods_by_scope.get(scope)
        if current is None:
            neighborhoods_by_scope[scope] = candidate
            continue
        candidate_score = float(candidate["score"]["balanced"])
        current_score = float(current["score"]["balanced"])
        if candidate_score > current_score:
            neighborhoods_by_scope[scope] = candidate
            continue
        if candidate_score == current_score and str(candidate["seed_path"]) < str(current["seed_path"]):
            neighborhoods_by_scope[scope] = candidate

    deduped = list(neighborhoods_by_scope.values())
    neighborhoods = _sorted(
        [
            item
            for item in deduped
            if int(item["ring_1"]["file_count"]) <= local_config.ring1_backlog_file_threshold
        ],
        key=lambda item: (
            -float(item["score"]["balanced"]),
            -int(item["ring_1"]["total"]),
            str(item["ring_1_scope"]),
            str(item["seed_path"]),
        ),
    )
    large_zone_backlog = _sorted(
        [
            item
            for item in deduped
            if int(item["ring_1"]["file_count"]) > local_config.ring1_backlog_file_threshold
        ],
        key=lambda item: (
            -int(item["ring_1"]["file_count"]),
            -float(item["score"]["balanced"]),
            str(item["ring_1_scope"]),
            str(item["seed_path"]),
        ),
    )
    for index, neighborhood in enumerate(neighborhoods, start=1):
        neighborhood["rank"] = index
    for index, neighborhood in enumerate(large_zone_backlog, start=1):
        neighborhood["backlog_rank"] = index
        neighborhood["backlog_reason"] = (
            "ring_1_scope exceeds max file threshold; retained as large-zone backlog"
        )

    counts_payload = payload.get("counts")
    source_counts = counts_payload if isinstance(counts_payload, dict) else {}
    projection_fiber_decision = (
        dict(projection_fiber_summary.decision.items())
        if projection_fiber_summary is not None
        else {}
    )
    projection_fiber_semantic_bundle_count = (
        projection_fiber_summary.compiled_projection_semantic_bundle_count
        if projection_fiber_summary is not None
        else 0
    )
    return {
        "format_version": 1,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source": {
            "source_generated_at_utc": payload.get("generated_at_utc"),
            "inventory_hash": payload.get("inventory_hash"),
            "rule_set_hash": payload.get("rule_set_hash"),
            "policy_results_hash": payload.get("policy_results_hash"),
            "changed_scope_hash": payload.get("changed_scope_hash"),
            "projection_fiber_decision": projection_fiber_decision,
            "projection_fiber_semantic_bundle_count": projection_fiber_semantic_bundle_count,
            "projection_fiber_semantic_preview_count": len(semantic_previews),
            "projection_fiber_semantic_previews": semantic_previews,
        },
        "config": {
            "active_families": list(families),
            "min_seed_families": local_config.min_seed_families,
            "min_seed_total": local_config.min_seed_total,
            "ring_2_similarity_threshold": local_config.ring2_similarity_threshold,
            "ring_2_min_total": local_config.ring2_min_total,
            "ring_2_limit": local_config.ring2_limit,
            "ring_2_weight": local_config.ring2_weight,
            "ring_1_backlog_file_threshold": local_config.ring1_backlog_file_threshold,
            "ring_1_definition": "full same-directory neighborhood",
            "scoring": "balanced_5_family_logsum",
        },
        "counts": {
            "source_counts": {
                family: _count_int(source_counts.get(family))
                for family in families
            },
            "seed_count": len(seed_paths),
            "neighborhood_count": len(neighborhoods),
            "large_zone_backlog_count": len(large_zone_backlog),
        },
        "neighborhoods": neighborhoods,
        "large_zone_backlog": large_zone_backlog,
    }


def _markdown_summary(payload: dict[str, Any]) -> str:
    source = payload.get("source")
    source_mapping = source if isinstance(source, dict) else {}
    semantics_decision = source_mapping.get("projection_fiber_decision")
    semantics_decision_mapping = (
        semantics_decision if isinstance(semantics_decision, dict) else {}
    )
    semantics_bundle_count = _count_int(
        source_mapping.get("projection_fiber_semantic_bundle_count")
    )
    semantic_previews = source_mapping.get("projection_fiber_semantic_previews")
    semantic_preview_list = semantic_previews if isinstance(semantic_previews, list) else []
    lines = [
        "# Hotspot Neighborhood Queue",
        "",
        f"- generated_at_utc: {payload.get('generated_at_utc', '')}",
        f"- neighborhoods: {payload.get('counts', {}).get('neighborhood_count', 0)}",
        f"- large_zone_backlog: {payload.get('counts', {}).get('large_zone_backlog_count', 0)}",
    ]
    if semantics_decision_mapping or semantics_bundle_count > 0 or semantic_preview_list:
        lines.extend(
            [
                f"- projection_fiber_decision: {semantics_decision_mapping.get('rule_id', '')}",
                (
                    "- projection_fiber_semantic_bundles: "
                    f"{semantics_bundle_count}"
                ),
            ]
        )
        if semantic_preview_list:
            lines.extend(
                [
                    "",
                    "## Projection Fiber Semantic Previews",
                    "",
                    "| spec | quotient_face | path | qualname | structural_path |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            for item in semantic_preview_list:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    "| {spec} | {face} | {path} | {qualname} | {structural_path} |".format(
                        spec=str(item.get("spec_name", "")),
                        face=str(item.get("quotient_face", "")),
                        path=str(item.get("path", "")),
                        qualname=str(item.get("qualname", "")),
                        structural_path=str(item.get("structural_path", "")),
                    )
                )
    lines.extend(
        [
            "",
            "| rank | ring_1_scope | seed_path | pf_overlap | score | ring_1_total | ring_2_total |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    neighborhoods = payload.get("neighborhoods")
    if isinstance(neighborhoods, list):
        for item in neighborhoods:
            if not isinstance(item, dict):
                continue
            lines.append(
                "| {rank} | {scope} | {seed} | {overlap} | {score:.3f} | {ring1} | {ring2} |".format(
                    rank=int(item.get("rank", 0) or 0),
                    scope=str(item.get("ring_1_scope", "")),
                    seed=str(item.get("seed_path", "")),
                    overlap=int(
                        ((item.get("projection_fiber_overlap") or {}).get("match_count", 0))
                        or 0
                    ),
                    score=float((item.get("score") or {}).get("balanced", 0.0) or 0.0),
                    ring1=int((item.get("score") or {}).get("ring_1_total", 0) or 0),
                    ring2=int((item.get("score") or {}).get("ring_2_advisory_total", 0) or 0),
                )
            )
        matched_neighborhoods = [
            item
            for item in neighborhoods
            if isinstance(item, dict)
            and int(((item.get("projection_fiber_overlap") or {}).get("match_count", 0)) or 0)
            > 0
        ]
        if matched_neighborhoods:
            lines.extend(
                [
                    "",
                    "## Projection Fiber Queue Overlap",
                    "",
                    "| rank | location | path | qualname | structural_path |",
                    "| ---: | --- | --- | --- | --- |",
                ]
            )
            for item in matched_neighborhoods:
                overlap = item.get("projection_fiber_overlap")
                if not isinstance(overlap, dict):
                    continue
                previews = overlap.get("matched_previews")
                if not isinstance(previews, list):
                    continue
                rank = int(item.get("rank", 0) or 0)
                for preview in previews:
                    if not isinstance(preview, dict):
                        continue
                    lines.append(
                        "| {rank} | {location} | {path} | {qualname} | {structural_path} |".format(
                            rank=rank,
                            location=str(preview.get("location", "")),
                            path=str(preview.get("path", "")),
                            qualname=str(preview.get("qualname", "")),
                            structural_path=str(preview.get("structural_path", "")),
                        )
                    )
    backlog = payload.get("large_zone_backlog")
    if isinstance(backlog, list) and backlog:
        lines.extend(
            [
                "",
                "## Large-Zone Backlog",
                "",
                "| backlog_rank | ring_1_scope | seed_path | ring_1_file_count | ring_1_total |",
                "| ---: | --- | --- | ---: | ---: |",
            ]
        )
        for item in backlog:
            if not isinstance(item, dict):
                continue
            lines.append(
                "| {rank} | {scope} | {seed} | {files} | {total} |".format(
                    rank=int(item.get("backlog_rank", 0) or 0),
                    scope=str(item.get("ring_1_scope", "")),
                    seed=str(item.get("seed_path", "")),
                    files=int((item.get("ring_1") or {}).get("file_count", 0) or 0),
                    total=int((item.get("ring_1") or {}).get("total", 0) or 0),
                )
            )
    lines.append("")
    return "\n".join(lines)


def _write_queue_outputs(
    *,
    queue: dict[str, Any],
    out_path: Path,
    markdown_out: Path | None = None,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(_markdown_summary(queue), encoding="utf-8")
    return 0


def run(
    *,
    source_artifact_path: Path,
    out_path: Path,
    markdown_out: Path | None = None,
    config: QueueConfig | None = None,
) -> int:
    payload = json.loads(source_artifact_path.read_text(encoding="utf-8"))
    queue = analyze(
        payload=payload,
        config=config,
    )
    return _write_queue_outputs(
        queue=queue,
        out_path=out_path,
        markdown_out=markdown_out,
    )


def run_from_payload(
    *,
    payload: dict[str, Any],
    out_path: Path,
    markdown_out: Path | None = None,
    config: QueueConfig | None = None,
) -> int:
    queue = analyze(
        payload=payload,
        config=config,
    )
    return _write_queue_outputs(
        queue=queue,
        out_path=out_path,
        markdown_out=markdown_out,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build geometric hotspot neighborhood queue from a source artifact payload."
    )
    parser.add_argument(
        "--source-artifact",
        required=True,
    )
    parser.add_argument("--out", default="artifacts/out/hotspot_neighborhood_queue.json")
    parser.add_argument("--markdown-out", default="artifacts/out/hotspot_neighborhood_queue.md")
    parser.add_argument("--min-seed-families", type=int, default=DEFAULT_MIN_SEED_FAMILIES)
    parser.add_argument("--min-seed-total", type=int, default=DEFAULT_MIN_SEED_TOTAL)
    parser.add_argument("--ring2-similarity-threshold", type=float, default=DEFAULT_RING2_SIMILARITY)
    parser.add_argument("--ring2-min-total", type=int, default=DEFAULT_RING2_MIN_TOTAL)
    parser.add_argument("--ring2-limit", type=int, default=DEFAULT_RING2_LIMIT)
    parser.add_argument("--ring2-weight", type=float, default=DEFAULT_RING2_WEIGHT)
    parser.add_argument(
        "--ring1-backlog-file-threshold",
        type=int,
        default=DEFAULT_RING1_BACKLOG_FILE_THRESHOLD,
    )
    args = parser.parse_args(argv)
    config = QueueConfig(
        min_seed_families=args.min_seed_families,
        min_seed_total=args.min_seed_total,
        ring2_similarity_threshold=args.ring2_similarity_threshold,
        ring2_min_total=args.ring2_min_total,
        ring2_limit=args.ring2_limit,
        ring2_weight=args.ring2_weight,
        ring1_backlog_file_threshold=args.ring1_backlog_file_threshold,
    )
    return run(
        source_artifact_path=Path(args.source_artifact).resolve(),
        out_path=Path(args.out).resolve(),
        markdown_out=Path(args.markdown_out).resolve(),
        config=config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
