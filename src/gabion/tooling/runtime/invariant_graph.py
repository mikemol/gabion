from __future__ import annotations

from collections import defaultdict
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from gabion.analysis.aspf.aspf_lattice_algebra import meet
from gabion.analysis.semantics import impact_index
from gabion.foundation.replayable_stream import ReplayableStream, stream_from_factory
from gabion.frontmatter import parse_strict_yaml_frontmatter
from gabion.order_contract import ordered_or_sorted
from gabion.tooling.policy_substrate.invariant_graph import (
    InvariantGraph,
    blocker_chains,
    build_invariant_graph,
    build_invariant_planning_bundle,
    build_invariant_ledger_alignments,
    build_invariant_ledger_delta_projections,
    build_invariant_ledger_projections,
    build_invariant_workstreams,
    compare_invariant_ledger_projections,
    compare_invariant_workstreams,
    load_invariant_graph,
    load_invariant_ledger_alignments,
    load_invariant_ledger_deltas,
    load_invariant_ledger_projections,
    load_invariant_workstreams,
    trace_nodes,
    write_invariant_graph,
    write_invariant_ledger_alignments,
    write_invariant_ledger_alignments_markdown,
    write_invariant_ledger_deltas,
    write_invariant_ledger_deltas_markdown,
    write_invariant_ledger_projections,
    write_invariant_workstreams,
)
from gabion.tooling.policy_substrate.workstream_registry import WorkstreamRegistry

_DEFAULT_ARTIFACT = Path("artifacts/out/invariant_graph.json")
_DEFAULT_WORKSTREAMS_ARTIFACT = Path("artifacts/out/invariant_workstreams.json")
_DEFAULT_LEDGER_ARTIFACT = Path("artifacts/out/invariant_ledger_projections.json")
_DEFAULT_LEDGER_DELTAS_ARTIFACT = Path("artifacts/out/invariant_ledger_deltas.json")
_DEFAULT_LEDGER_DELTAS_MARKDOWN_ARTIFACT = Path(
    "artifacts/out/invariant_ledger_deltas.md"
)
_DEFAULT_LEDGER_ALIGNMENTS_ARTIFACT = Path(
    "artifacts/out/invariant_ledger_alignments.json"
)
_DEFAULT_LEDGER_ALIGNMENTS_MARKDOWN_ARTIFACT = Path(
    "artifacts/out/invariant_ledger_alignments.md"
)


@dataclass(frozen=True)
class _ProfileObservation:
    profiler: str
    metric_kind: str
    unit: str
    artifact_node_wire: str
    site_identity: str
    rel_path: str
    qualname: str
    line: int
    structural_identity: str
    inclusive_value: float


@dataclass(frozen=True)
class _MatchedProfileObservation:
    node_id: str
    profiler: str
    metric_kind: str
    unit: str
    rel_path: str
    qualname: str
    line: int
    title: str
    inclusive_value: float


@dataclass(frozen=True)
class _PerfInfimumBucket:
    metric_kind: str
    unit: str
    node_id: str
    title: str
    node_kind: str
    rel_path: str
    qualname: str
    line: int
    depth: int
    total_inclusive_value: float
    matched_leaf_node_count: int
    profiler_count: int
    direct_match_node_count: int
    is_global_infimum: bool
    is_virtual_intersection: bool


@dataclass(frozen=True)
class _PerfDslOverlay:
    doc_ids: tuple[str, ...]
    doc_paths: tuple[str, ...]
    target_symbols: tuple[str, ...]
    candidate_node_ids: tuple[str, ...]


def _sorted[T](values: list[T], *, key=None) -> list[T]:
    return ordered_or_sorted(
        values,
        source="tooling.runtime.invariant_graph",
        key=key,
    )


def _load_or_build_graph(
    *,
    root: Path,
    artifact: Path,
    declared_registries: tuple[WorkstreamRegistry, ...] | None = None,
) -> InvariantGraph:
    if artifact.exists():
        return load_invariant_graph(artifact)
    return build_invariant_graph(root, declared_registries=declared_registries)


def _descendant_ids(graph: InvariantGraph, node_id: str) -> tuple[str, ...]:
    node_by_id = graph.node_by_id()
    edges_from = graph.edges_from()
    pending = [node_id]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for edge in edges_from.get(current, ()):
            if edge.edge_kind == "contains" and edge.target_id in node_by_id:
                pending.append(edge.target_id)
    return tuple(_sorted(list(seen)))


def _doc_id_paths(root: Path) -> dict[str, tuple[str, ...]]:
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for path in root.rglob("*.md"):
        rel_parts = path.relative_to(root).parts
        if not rel_parts:
            continue
        if rel_parts[0] in {"artifacts", "out", ".git", ".venv", "__pycache__"}:
            continue
        frontmatter, _body = parse_strict_yaml_frontmatter(
            path.read_text(encoding="utf-8"),
            require_parser=False,
        )
        doc_id = frontmatter.get("doc_id")
        if isinstance(doc_id, str) and doc_id:
            grouped[doc_id].append(str(path.relative_to(root)))
    return {
        doc_id: tuple(_sorted(paths))
        for doc_id, paths in grouped.items()
    }


def _module_name_from_rel_path(rel_path: str) -> str:
    parts = list(Path(rel_path).with_suffix("").parts)
    if parts and parts[0] in {"src", "tests"}:
        parts = parts[1:]
    return ".".join(parts)


def _symbol_name_for_node(node) -> str:
    if not node.rel_path or not node.qualname:
        return ""
    module_name = _module_name_from_rel_path(node.rel_path)
    if not module_name:
        return ""
    return f"{module_name}.{node.qualname}"


def _resolve_perf_dsl_overlay(
    *,
    root: Path,
    graph: InvariantGraph,
    scope_node_ids: tuple[str, ...],
) -> _PerfDslOverlay:
    node_by_id = graph.node_by_id()
    doc_ids = tuple(
        _sorted(
            list(
                {
                    doc_id
                    for node_id in scope_node_ids
                    for doc_id in (
                        ()
                        if node_by_id.get(node_id) is None
                        else node_by_id[node_id].doc_ids
                    )
                }
            )
        )
    )
    if not doc_ids:
        return _PerfDslOverlay(
            doc_ids=(),
            doc_paths=(),
            target_symbols=(),
            candidate_node_ids=(),
        )
    doc_paths_by_id = _doc_id_paths(root)
    doc_paths = tuple(
        _sorted(
            list(
                {
                    path
                    for doc_id in doc_ids
                    for path in doc_paths_by_id.get(doc_id, ())
                }
            )
        )
    )
    if not doc_paths:
        return _PerfDslOverlay(
            doc_ids=doc_ids,
            doc_paths=(),
            target_symbols=(),
            candidate_node_ids=(),
        )
    existing_doc_paths = tuple(
        _sorted(
            list(
                {
                    rel_path
                    for rel_path in doc_paths
                    if (root / rel_path).exists() and (root / rel_path).is_file()
                }
            )
        )
    )
    if not existing_doc_paths:
        return _PerfDslOverlay(
            doc_ids=doc_ids,
            doc_paths=(),
            target_symbols=(),
            candidate_node_ids=(),
        )
    symbol_universe = {
        symbol_name
        for node_id in scope_node_ids
        for symbol_name in (
            (_symbol_name_for_node(node_by_id[node_id]),)
            if node_by_id.get(node_id) is not None
            else ()
        )
        if symbol_name
    }
    if not symbol_universe:
        return _PerfDslOverlay(
            doc_ids=doc_ids,
            doc_paths=existing_doc_paths,
            target_symbols=(),
            candidate_node_ids=(),
        )
    links = []
    for rel_path in existing_doc_paths:
        links.extend(
            impact_index._links_from_doc(
                path=root / rel_path,
                root=root,
                symbols=symbol_universe,
            )
        )
    target_symbols = tuple(
        _sorted(
            list(
                {
                    link.target
                    for link in links
                    if link.source_kind == "doc" and link.source in existing_doc_paths
                }
            )
        )
    )
    if not target_symbols:
        return _PerfDslOverlay(
            doc_ids=doc_ids,
            doc_paths=existing_doc_paths,
            target_symbols=(),
            candidate_node_ids=(),
        )
    candidate_node_ids: set[str] = set()
    for node_id in scope_node_ids:
        node = node_by_id.get(node_id)
        if node is None:
            continue
        symbol_name = _symbol_name_for_node(node)
        if not symbol_name:
            continue
        for target_symbol in target_symbols:
            if symbol_name == target_symbol or symbol_name.startswith(
                f"{target_symbol}."
            ):
                candidate_node_ids.add(node_id)
                break
    return _PerfDslOverlay(
        doc_ids=doc_ids,
        doc_paths=existing_doc_paths,
        target_symbols=target_symbols,
        candidate_node_ids=tuple(_sorted(list(candidate_node_ids))),
    )


def _ancestor_ids(graph: InvariantGraph, node_id: str) -> tuple[str, ...]:
    node_by_id = graph.node_by_id()
    edges_to = graph.edges_to()
    pending = [node_id]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for edge in edges_to.get(current, ()):
            if edge.edge_kind == "contains" and edge.source_id in node_by_id:
                pending.append(edge.source_id)
    return tuple(_sorted(list(seen)))


def _containment_depths(graph: InvariantGraph) -> dict[str, int]:
    node_by_id = graph.node_by_id()
    edges_to = graph.edges_to()
    cache: dict[str, int] = {}

    def _depth(node_id: str) -> int:
        cached = cache.get(node_id)
        if cached is not None:
            return cached
        parent_ids = [
            edge.source_id
            for edge in edges_to.get(node_id, ())
            if edge.edge_kind == "contains" and edge.source_id in node_by_id
        ]
        if not parent_ids:
            cache[node_id] = 0
            return 0
        depth = 1 + max(_depth(parent_id) for parent_id in parent_ids)
        cache[node_id] = depth
        return depth

    for node_id in node_by_id:
        _depth(node_id)
    return cache


def _coverage_and_signal_ids(
    graph: InvariantGraph,
    *,
    node_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    node_by_id = graph.node_by_id()
    edges_from = graph.edges_from()
    edges_to = graph.edges_to()
    test_case_ids = {
        edge.target_id
        for node_id in node_ids
        for edge in edges_from.get(node_id, ())
        if edge.edge_kind == "covered_by"
        and edge.target_id in node_by_id
        and node_by_id[edge.target_id].node_kind == "test_case"
    }
    signal_ids = {
        edge.source_id
        for node_id in node_ids
        for edge in edges_to.get(node_id, ())
        if edge.edge_kind == "blocks"
        and edge.source_id in node_by_id
        and node_by_id[edge.source_id].node_kind == "policy_signal"
    }
    return (tuple(_sorted(list(test_case_ids))), tuple(_sorted(list(signal_ids))))


def _load_impacted_tests(path: Path | None) -> tuple[str, ...]:
    if path is None or not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return ()
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        return ()
    impacted_tests = selection.get("impacted_tests", [])
    if not isinstance(impacted_tests, list):
        return ()
    return tuple(_sorted([str(item) for item in impacted_tests if isinstance(item, str)]))


def _normalized_profile_line(value: object) -> int | None:
    match value:
        case bool():
            return None
        case int() as line if line > 0:
            return line
        case float() as line if line.is_integer() and line > 0:
            return int(line)
        case str() as line:
            stripped = line.strip()
            if not stripped:
                return None
            try:
                parsed = int(stripped)
            except ValueError:
                return None
            return parsed if parsed > 0 else None
        case _:
            return None


def _normalized_profile_value(value: object) -> float | None:
    match value:
        case bool():
            return None
        case int() | float() as number:
            parsed = float(number)
            return parsed if parsed >= 0 else None
        case str() as number:
            stripped = number.strip()
            if not stripped:
                return None
            try:
                parsed = float(stripped)
            except ValueError:
                return None
            return parsed if parsed >= 0 else None
        case _:
            return None


def _profile_observation_from_payload(
    raw_sample: object,
    *,
    profiler: object,
    metric_kind: object,
    unit: object,
) -> _ProfileObservation | None:
    if not isinstance(raw_sample, dict):
        return None
    raw_artifact_node = raw_sample.get("artifact_node")
    artifact_node_wire = ""
    site_identity = ""
    structural_identity = ""
    rel_path = ""
    qualname = ""
    line = 0
    if isinstance(raw_artifact_node, dict):
        artifact_node_wire = str(raw_artifact_node.get("wire", "")).strip()
        site_identity = str(raw_artifact_node.get("site_identity", "")).strip()
        structural_identity = str(
            raw_artifact_node.get("structural_identity", "")
        ).strip()
        rel_path = str(raw_artifact_node.get("rel_path", "")).strip()
        qualname = str(raw_artifact_node.get("qualname", "")).strip()
        line = _normalized_profile_line(raw_artifact_node.get("line")) or 0
    if not site_identity:
        site_identity = str(raw_sample.get("site_identity", "")).strip()
    if not structural_identity:
        structural_identity = str(raw_sample.get("structural_identity", "")).strip()
    if not rel_path:
        rel_path = str(raw_sample.get("rel_path", "")).strip()
    if not qualname:
        qualname = str(raw_sample.get("qualname", "")).strip()
    if line <= 0:
        line = _normalized_profile_line(raw_sample.get("line")) or 0
    inclusive_value = _normalized_profile_value(raw_sample.get("inclusive_value"))
    has_identity = bool(site_identity or structural_identity)
    has_location = bool(rel_path and qualname and line > 0)
    if inclusive_value is None or not (has_identity or has_location):
        return None
    return _ProfileObservation(
        profiler=str(profiler).strip() or "unknown",
        metric_kind=str(metric_kind).strip() or "value",
        unit=str(unit).strip() or "value",
        artifact_node_wire=artifact_node_wire,
        site_identity=site_identity,
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        structural_identity=structural_identity,
        inclusive_value=inclusive_value,
    )


def _load_profile_observations(path: Path | None) -> tuple[_ProfileObservation, ...]:
    if path is None or not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return ()
    observations: list[_ProfileObservation] = []
    raw_profiles = payload.get("profiles", [])
    if isinstance(raw_profiles, list):
        for raw_profile in raw_profiles:
            if not isinstance(raw_profile, dict):
                continue
            samples = raw_profile.get("samples", [])
            if not isinstance(samples, list):
                continue
            for raw_sample in samples:
                observation = _profile_observation_from_payload(
                    raw_sample,
                    profiler=raw_profile.get("profiler", ""),
                    metric_kind=raw_profile.get("metric_kind", ""),
                    unit=raw_profile.get("unit", ""),
                )
                if observation is not None:
                    observations.append(observation)
    raw_observations = payload.get("observations", [])
    if isinstance(raw_observations, list):
        for raw_observation in raw_observations:
            observation = _profile_observation_from_payload(
                raw_observation,
                profiler=payload.get("profiler", raw_observation.get("profiler", "")),
                metric_kind=payload.get(
                    "metric_kind", raw_observation.get("metric_kind", "")
                ),
                unit=payload.get("unit", raw_observation.get("unit", "")),
            )
            if observation is not None:
                observations.append(observation)
    return tuple(
        _sorted(
            observations,
            key=lambda item: (
                item.profiler,
                item.metric_kind,
                item.rel_path,
                item.line,
                item.qualname,
                item.inclusive_value,
            ),
        )
    )


def _match_profile_observations(
    *,
    graph: InvariantGraph,
    descendant_ids: tuple[str, ...],
    observations: tuple[_ProfileObservation, ...],
) -> tuple[_MatchedProfileObservation, ...]:
    node_by_id = graph.node_by_id()
    structural_node_ids: dict[str, str] = {}
    site_node_ids: dict[str, str] = {}
    path_line_qualname_node_ids: dict[tuple[str, int, str], str] = {}
    for node_id in descendant_ids:
        node = node_by_id.get(node_id)
        if node is None:
            continue
        if node.structural_identity:
            structural_node_ids.setdefault(node.structural_identity, node_id)
        if node.site_identity:
            site_node_ids.setdefault(node.site_identity, node_id)
        if node.rel_path and node.line > 0 and node.qualname:
            path_line_qualname_node_ids.setdefault(
                (node.rel_path, node.line, node.qualname),
                node_id,
            )
    matched: list[_MatchedProfileObservation] = []
    for observation in observations:
        matched_node_id = ""
        if observation.structural_identity:
            matched_node_id = structural_node_ids.get(observation.structural_identity, "")
        if not matched_node_id and observation.site_identity:
            matched_node_id = site_node_ids.get(observation.site_identity, "")
        if not matched_node_id:
            matched_node_id = path_line_qualname_node_ids.get(
                (observation.rel_path, observation.line, observation.qualname),
                "",
            )
        if not matched_node_id:
            continue
        node = node_by_id[matched_node_id]
        matched.append(
            _MatchedProfileObservation(
                node_id=matched_node_id,
                profiler=observation.profiler,
                metric_kind=observation.metric_kind,
                unit=observation.unit,
                rel_path=node.rel_path,
                qualname=node.qualname,
                line=node.line,
                title=node.title,
                inclusive_value=observation.inclusive_value,
            )
        )
    return tuple(
        _sorted(
            matched,
            key=lambda item: (
                item.metric_kind,
                item.unit,
                -item.inclusive_value,
                item.profiler,
                item.rel_path,
                item.line,
                item.qualname,
            ),
        )
    )


def _perf_infimum_buckets(
    *,
    graph: InvariantGraph,
    descendant_ids: tuple[str, ...],
    matched: tuple[_MatchedProfileObservation, ...],
) -> tuple[_PerfInfimumBucket, ...]:
    if not matched:
        return ()
    node_by_id = graph.node_by_id()
    descendant_id_set = set(descendant_ids)
    depth_by_node = _containment_depths(graph)
    ancestor_cache: dict[str, tuple[str, ...]] = {}
    buckets: list[_PerfInfimumBucket] = []
    by_metric: defaultdict[tuple[str, str], list[_MatchedProfileObservation]] = defaultdict(list)
    for observation in matched:
        by_metric[(observation.metric_kind, observation.unit)].append(observation)
    for (metric_kind, unit), metric_observations in by_metric.items():
        matched_node_ids = tuple(
            _sorted(list({item.node_id for item in metric_observations}))
        )
        if len(matched_node_ids) < 2:
            continue
        global_common_ids: tuple[str, ...] = tuple(
            item
            for item in ancestor_cache.setdefault(
                matched_node_ids[0],
                _ancestor_ids(graph, matched_node_ids[0]),
            )
            if item in descendant_id_set
        )
        for matched_node_id in matched_node_ids[1:]:
            global_common_ids = tuple(
                item
                for item in meet(
                    left_ids=global_common_ids,
                    right_ids=tuple(
                        item
                        for item in ancestor_cache.setdefault(
                            matched_node_id,
                            _ancestor_ids(graph, matched_node_id),
                        )
                        if item in descendant_id_set
                    ),
                ).result_ids
                if item in descendant_id_set
            )
        if global_common_ids:
            deepest_global_depth = max(depth_by_node.get(item, 0) for item in global_common_ids)
            global_infimum_ids = {
                item
                for item in global_common_ids
                if depth_by_node.get(item, 0) == deepest_global_depth
            }
        else:
            global_infimum_ids = set()
        ancestor_aggregate: defaultdict[
            str,
            dict[str, object],
        ] = defaultdict(
            lambda: {
                "matched_leaf_node_ids": set(),
                "profilers": set(),
                "direct_match_node_ids": set(),
                "total_inclusive_value": 0.0,
            }
        )
        for observation in metric_observations:
            ancestor_ids = tuple(
                item
                for item in ancestor_cache.setdefault(
                    observation.node_id,
                    _ancestor_ids(graph, observation.node_id),
                )
                if item in descendant_id_set
            )
            for ancestor_id in ancestor_ids:
                aggregate = ancestor_aggregate[ancestor_id]
                aggregate["matched_leaf_node_ids"].add(observation.node_id)
                aggregate["profilers"].add(observation.profiler)
                aggregate["total_inclusive_value"] = float(
                    aggregate["total_inclusive_value"]
                ) + observation.inclusive_value
                if observation.node_id == ancestor_id:
                    aggregate["direct_match_node_ids"].add(observation.node_id)
        for ancestor_id, aggregate in ancestor_aggregate.items():
            matched_leaf_node_ids = aggregate["matched_leaf_node_ids"]
            if len(matched_leaf_node_ids) < 2:
                continue
            node = node_by_id.get(ancestor_id)
            if node is None:
                continue
            direct_match_node_ids = aggregate["direct_match_node_ids"]
            buckets.append(
                _PerfInfimumBucket(
                    metric_kind=metric_kind,
                    unit=unit,
                    node_id=ancestor_id,
                    title=node.title,
                    node_kind=node.node_kind,
                    rel_path=node.rel_path,
                    qualname=node.qualname,
                    line=node.line,
                    depth=depth_by_node.get(ancestor_id, 0),
                    total_inclusive_value=float(aggregate["total_inclusive_value"]),
                    matched_leaf_node_count=len(matched_leaf_node_ids),
                    profiler_count=len(aggregate["profilers"]),
                    direct_match_node_count=len(direct_match_node_ids),
                    is_global_infimum=ancestor_id in global_infimum_ids,
                    is_virtual_intersection=ancestor_id not in matched_leaf_node_ids,
                )
            )
    return tuple(
        _sorted(
            buckets,
            key=lambda item: (
                item.metric_kind,
                item.unit,
                -item.total_inclusive_value,
                -item.matched_leaf_node_count,
                -item.depth,
                item.node_id,
            ),
        )
    )


def _format_score_components(components: object) -> str:
    if not isinstance(components, (list, tuple)) or not components:
        return "none"
    parts: list[str] = []
    for item in components:
        if isinstance(item, dict):
            kind = str(item.get("kind", "component"))
            score = int(item.get("score", 0))
            rationale = str(item.get("rationale", ""))
        else:
            kind = str(getattr(item, "kind", "component"))
            score = int(getattr(item, "score", 0))
            rationale = str(getattr(item, "rationale", ""))
        parts.append(f"{kind}:{score}:{rationale}")
    return " | ".join(parts) if parts else "none"


def _format_owner_resolution_options(options: object) -> str:
    if not isinstance(options, (list, tuple)) or not options:
        return "none"
    parts: list[str] = []
    for item in options:
        if isinstance(item, dict):
            resolution_kind = str(item.get("resolution_kind", "unknown"))
            object_id = str(item.get("object_id", "<none>"))
            score = int(item.get("score", 0))
            components = item.get("score_components", ())
            selection_rank = int(item.get("selection_rank", 0))
            opportunity_cost_score = int(item.get("opportunity_cost_score", 0))
            opportunity_cost_reason = str(item.get("opportunity_cost_reason", "frontier"))
            opportunity_cost_components = item.get("opportunity_cost_components", ())
        else:
            resolution_kind = str(getattr(item, "resolution_kind", "unknown"))
            object_id = str(getattr(item, "object_id", "<none>"))
            score = int(getattr(item, "score", 0))
            components = getattr(item, "score_components", ())
            selection_rank = int(getattr(item, "selection_rank", 0))
            opportunity_cost_score = int(getattr(item, "opportunity_cost_score", 0))
            opportunity_cost_reason = str(
                getattr(item, "opportunity_cost_reason", "frontier")
            )
            opportunity_cost_components = getattr(item, "opportunity_cost_components", ())
        parts.append(
            (
                f"{resolution_kind}:{object_id}:{score}:"
                f"{_format_score_components(components)}:"
                f"rank={selection_rank}:"
                f"opp={opportunity_cost_score}:{opportunity_cost_reason}:"
                f"{_format_score_components(opportunity_cost_components)}"
            )
        )
    return " || ".join(parts) if parts else "none"


def _format_repo_followup_cohort(cohort: object) -> str:
    if not isinstance(cohort, (list, tuple)) or not cohort:
        return "none"
    parts: list[str] = []
    for item in cohort:
        if isinstance(item, dict):
            followup_family = str(item.get("followup_family", "followup"))
            action_kind = str(item.get("action_kind", "action"))
            object_id = item.get("object_id")
            owner_root_object_id = item.get("owner_root_object_id")
            diagnostic_code = item.get("diagnostic_code")
            target_doc_id = item.get("target_doc_id")
            title = str(item.get("title", ""))
            utility_score = int(item.get("utility_score", 0))
            selection_rank = int(item.get("selection_rank", 0))
            selection_reason = str(item.get("selection_reason", "none"))
        else:
            followup_family = str(getattr(item, "followup_family", "followup"))
            action_kind = str(getattr(item, "action_kind", "action"))
            object_id = getattr(item, "object_id", None)
            owner_root_object_id = getattr(item, "owner_root_object_id", None)
            diagnostic_code = getattr(item, "diagnostic_code", None)
            target_doc_id = getattr(item, "target_doc_id", None)
            title = str(getattr(item, "title", ""))
            utility_score = int(getattr(item, "utility_score", 0))
            selection_rank = int(getattr(item, "selection_rank", 0))
            selection_reason = str(getattr(item, "selection_reason", "none"))
        if diagnostic_code is not None:
            label = f"{followup_family}:{action_kind}:{diagnostic_code}:{title}"
        elif object_id is not None:
            label = f"{followup_family}:{action_kind}:{object_id}:{title}"
        elif target_doc_id is not None:
            label = f"{followup_family}:{action_kind}:{target_doc_id}:{title}"
        else:
            label = f"{followup_family}:{action_kind}:{title}"
        parts.append(
            (
                f"{label}@{utility_score}:root={owner_root_object_id or '<none>'}:"
                f"rank={selection_rank}:{selection_reason}"
            )
        )
    return " || ".join(parts) if parts else "none"


def _workstream_by_object_id(
    *,
    graph: InvariantGraph,
    root: Path | None,
    object_id: str,
):
    projection = build_invariant_workstreams(graph, root=root)
    for item in projection.iter_workstreams():
        if item.object_id.wire() == object_id:
            return item
    return None


def _print_summary(*, graph: InvariantGraph, root: Path) -> None:
    payload = graph.as_payload()
    counts = payload.get("counts", {})
    workstreams = build_invariant_workstreams(graph, root=root)
    diagnostic_summary = workstreams.diagnostic_summary()
    recommended_repo_followup = workstreams.recommended_repo_followup()
    recommended_repo_code_followup = workstreams.recommended_repo_code_followup()
    recommended_repo_human_followup = workstreams.recommended_repo_human_followup()
    recommended_repo_followup_lane = workstreams.recommended_repo_followup_lane()
    recommended_repo_code_followup_lane = workstreams.recommended_repo_code_followup_lane()
    recommended_repo_human_followup_lane = (
        workstreams.recommended_repo_human_followup_lane()
    )
    recommended_repo_followup_frontier_tradeoff = (
        workstreams.recommended_repo_followup_frontier_tradeoff()
    )
    recommended_repo_followup_frontier_explanation = (
        workstreams.recommended_repo_followup_frontier_explanation()
    )
    recommended_repo_followup_decision_protocol = (
        workstreams.recommended_repo_followup_decision_protocol()
    )
    recommended_repo_followup_frontier_triad = (
        workstreams.recommended_repo_followup_frontier_triad()
    )
    recommended_repo_followup_same_class_tradeoff = (
        workstreams.recommended_repo_followup_same_class_tradeoff()
    )
    recommended_repo_followup_cross_class_tradeoff = (
        workstreams.recommended_repo_followup_cross_class_tradeoff()
    )
    repo_followup_lanes = workstreams.repo_followup_lanes()
    repo_diagnostic_lanes = workstreams.repo_diagnostic_lanes()
    print(f"root: {graph.root}")
    print(f"workstreams: {len(graph.workstream_root_ids)}")
    print(f"nodes: {counts.get('node_count', 0)}")
    print(f"edges: {counts.get('edge_count', 0)}")
    print(f"diagnostics: {counts.get('diagnostic_count', 0)}")
    print(f"dominant_followup_class: {workstreams.dominant_repo_followup_class()}")
    print(f"next_human_followup_family: {workstreams.next_repo_human_followup_family()}")
    print(
        "diagnostic_summary: unmatched_policy_signals={signals} :: unresolved_dependencies={dependencies} :: workspace_preservation={workspace} :: workspace_orphans={workspace_orphans}".format(
            signals=diagnostic_summary.unmatched_policy_signal_count,
            dependencies=diagnostic_summary.unresolved_blocking_dependency_count,
            workspace=diagnostic_summary.workspace_preservation_count,
            workspace_orphans=diagnostic_summary.orphaned_workspace_change_count,
        )
    )
    if recommended_repo_followup is None:
        print("recommended_repo_followup: <none>")
    elif recommended_repo_followup.diagnostic_code is not None:
        print(
            "recommended_repo_followup: {family} :: diagnostic={diagnostic} :: owner={owner} :: seed={seed} :: seed_object={seed_object} :: owner_kind={owner_kind} :: owner_score={owner_score} :: owner_options={owner_options} :: runner_up_owner={runner_up_owner} :: runner_up_kind={runner_up_kind} :: runner_up_score={runner_up_score} :: owner_choice_margin={owner_choice_margin} :: owner_choice_margin_components={owner_choice_margin_components} :: owner_option_tradeoff={owner_option_tradeoff} :: owner_option_tradeoff_components={owner_option_tradeoff_components} :: count={count} :: action={action} :: utility={utility} :: utility_components={utility_components} :: certainty={certainty} :: scope={scope} :: runner_up_followup={runner_up_followup} :: frontier_choice_margin={frontier_choice_margin} :: frontier_choice_margin_components={frontier_choice_margin_components} :: rank={rank} :: opportunity={opportunity} :: opportunity_components={opportunity_components}".format(
                family=recommended_repo_followup.followup_family,
                diagnostic=recommended_repo_followup.diagnostic_code,
                owner=recommended_repo_followup.owner_object_id or "<none>",
                seed=recommended_repo_followup.owner_seed_path or "<none>",
                seed_object=recommended_repo_followup.owner_seed_object_id or "<none>",
                owner_kind=recommended_repo_followup.owner_resolution_kind or "none",
                owner_score=(
                    "none"
                    if recommended_repo_followup.owner_resolution_score is None
                    else recommended_repo_followup.owner_resolution_score
                ),
                owner_options=_format_owner_resolution_options(
                    recommended_repo_followup.owner_resolution_options
                ),
                runner_up_owner=(
                    recommended_repo_followup.runner_up_owner_object_id or "<none>"
                ),
                runner_up_kind=(
                    recommended_repo_followup.runner_up_owner_resolution_kind or "none"
                ),
                runner_up_score=(
                    "none"
                    if recommended_repo_followup.runner_up_owner_resolution_score is None
                    else recommended_repo_followup.runner_up_owner_resolution_score
                ),
                owner_choice_margin=(
                    "none"
                    if recommended_repo_followup.owner_choice_margin_score is None
                    else (
                        f"{recommended_repo_followup.owner_choice_margin_score}:"
                        f"{recommended_repo_followup.owner_choice_margin_reason}"
                    )
                ),
                owner_choice_margin_components=_format_score_components(
                    recommended_repo_followup.owner_choice_margin_components
                ),
                owner_option_tradeoff=(
                    "none"
                    if recommended_repo_followup.owner_option_tradeoff_score is None
                    else (
                        f"{recommended_repo_followup.owner_option_tradeoff_score}:"
                        f"{recommended_repo_followup.owner_option_tradeoff_reason}"
                    )
                ),
                owner_option_tradeoff_components=_format_score_components(
                    recommended_repo_followup.owner_option_tradeoff_components
                ),
                count=recommended_repo_followup.count,
                action=recommended_repo_followup.recommended_action or "none",
                utility=(
                    f"{recommended_repo_followup.utility_score}:{recommended_repo_followup.utility_reason}"
                ),
                utility_components=_format_score_components(
                    recommended_repo_followup.utility_components
                ),
                certainty=(
                    f"{recommended_repo_followup.selection_certainty_kind}:"
                    f"{recommended_repo_followup.cofrontier_followup_count}"
                ),
                scope=(
                    f"{recommended_repo_followup.selection_scope_kind}:"
                    f"{recommended_repo_followup.selection_scope_id or '<none>'}"
                ),
                runner_up_followup=(
                    "<none>"
                    if recommended_repo_followup.runner_up_followup_family is None
                    else (
                        f"{recommended_repo_followup.runner_up_followup_family}:"
                        f"{recommended_repo_followup.runner_up_followup_class or 'none'}:"
                        f"{recommended_repo_followup.runner_up_followup_object_id or '<none>'}:"
                        f"{recommended_repo_followup.runner_up_followup_utility_score}"
                    )
                ),
                frontier_choice_margin=(
                    "none"
                    if recommended_repo_followup.frontier_choice_margin_score is None
                    else (
                        f"{recommended_repo_followup.frontier_choice_margin_score}:"
                        f"{recommended_repo_followup.frontier_choice_margin_reason}"
                    )
                ),
                frontier_choice_margin_components=_format_score_components(
                    recommended_repo_followup.frontier_choice_margin_components
                ),
                rank=recommended_repo_followup.selection_rank,
                opportunity=(
                    f"{recommended_repo_followup.opportunity_cost_score}:"
                    f"{recommended_repo_followup.opportunity_cost_reason}"
                ),
                opportunity_components=_format_score_components(
                    recommended_repo_followup.opportunity_cost_components
                ),
            )
        )
        print(
            "recommended_repo_followup_cohort: {count} :: {cohort}".format(
                count=len(recommended_repo_followup.cofrontier_followup_cohort),
                cohort=_format_repo_followup_cohort(
                    recommended_repo_followup.cofrontier_followup_cohort
                ),
            )
        )
    elif recommended_repo_followup.action_kind == "doc_alignment":
        print(
            "recommended_repo_followup: {family} :: owner={owner} :: target_doc={target_doc} :: alignment={alignment} :: action={action} :: utility={utility} :: utility_components={utility_components} :: certainty={certainty} :: scope={scope} :: runner_up_followup={runner_up_followup} :: frontier_choice_margin={frontier_choice_margin} :: frontier_choice_margin_components={frontier_choice_margin_components} :: rank={rank} :: opportunity={opportunity} :: opportunity_components={opportunity_components}".format(
                family=recommended_repo_followup.followup_family,
                owner=recommended_repo_followup.owner_object_id or "<none>",
                target_doc=recommended_repo_followup.target_doc_id or "<none>",
                alignment=recommended_repo_followup.alignment_status or "none",
                action=recommended_repo_followup.recommended_action or "none",
                utility=(
                    f"{recommended_repo_followup.utility_score}:{recommended_repo_followup.utility_reason}"
                ),
                utility_components=_format_score_components(
                    recommended_repo_followup.utility_components
                ),
                certainty=(
                    f"{recommended_repo_followup.selection_certainty_kind}:"
                    f"{recommended_repo_followup.cofrontier_followup_count}"
                ),
                scope=(
                    f"{recommended_repo_followup.selection_scope_kind}:"
                    f"{recommended_repo_followup.selection_scope_id or '<none>'}"
                ),
                runner_up_followup=(
                    "<none>"
                    if recommended_repo_followup.runner_up_followup_family is None
                    else (
                        f"{recommended_repo_followup.runner_up_followup_family}:"
                        f"{recommended_repo_followup.runner_up_followup_class or 'none'}:"
                        f"{recommended_repo_followup.runner_up_followup_object_id or '<none>'}:"
                        f"{recommended_repo_followup.runner_up_followup_utility_score}"
                    )
                ),
                frontier_choice_margin=(
                    "none"
                    if recommended_repo_followup.frontier_choice_margin_score is None
                    else (
                        f"{recommended_repo_followup.frontier_choice_margin_score}:"
                        f"{recommended_repo_followup.frontier_choice_margin_reason}"
                    )
                ),
                frontier_choice_margin_components=_format_score_components(
                    recommended_repo_followup.frontier_choice_margin_components
                ),
                rank=recommended_repo_followup.selection_rank,
                opportunity=(
                    f"{recommended_repo_followup.opportunity_cost_score}:"
                    f"{recommended_repo_followup.opportunity_cost_reason}"
                ),
                opportunity_components=_format_score_components(
                    recommended_repo_followup.opportunity_cost_components
                ),
            )
        )
        print(
            "recommended_repo_followup_cohort: {count} :: {cohort}".format(
                count=len(recommended_repo_followup.cofrontier_followup_cohort),
                cohort=_format_repo_followup_cohort(
                    recommended_repo_followup.cofrontier_followup_cohort
                ),
            )
        )
    else:
        print(
            "recommended_repo_followup: {family} :: owner={owner} :: {action_kind} :: {object_id} :: count={count} :: blocker={blocker} :: utility={utility} :: utility_components={utility_components} :: certainty={certainty} :: scope={scope} :: runner_up_followup={runner_up_followup} :: frontier_choice_margin={frontier_choice_margin} :: frontier_choice_margin_components={frontier_choice_margin_components} :: rank={rank} :: opportunity={opportunity} :: opportunity_components={opportunity_components}".format(
                family=recommended_repo_followup.followup_family,
                owner=recommended_repo_followup.owner_object_id or "<none>",
                action_kind=recommended_repo_followup.action_kind,
                object_id=recommended_repo_followup.object_id or "<none>",
                count=recommended_repo_followup.count,
                blocker=recommended_repo_followup.readiness_class or "none",
                utility=(
                    f"{recommended_repo_followup.utility_score}:{recommended_repo_followup.utility_reason}"
                ),
                utility_components=_format_score_components(
                    recommended_repo_followup.utility_components
                ),
                certainty=(
                    f"{recommended_repo_followup.selection_certainty_kind}:"
                    f"{recommended_repo_followup.cofrontier_followup_count}"
                ),
                scope=(
                    f"{recommended_repo_followup.selection_scope_kind}:"
                    f"{recommended_repo_followup.selection_scope_id or '<none>'}"
                ),
                runner_up_followup=(
                    "<none>"
                    if recommended_repo_followup.runner_up_followup_family is None
                    else (
                        f"{recommended_repo_followup.runner_up_followup_family}:"
                        f"{recommended_repo_followup.runner_up_followup_class or 'none'}:"
                        f"{recommended_repo_followup.runner_up_followup_object_id or '<none>'}:"
                        f"{recommended_repo_followup.runner_up_followup_utility_score}"
                    )
                ),
                frontier_choice_margin=(
                    "none"
                    if recommended_repo_followup.frontier_choice_margin_score is None
                    else (
                        f"{recommended_repo_followup.frontier_choice_margin_score}:"
                        f"{recommended_repo_followup.frontier_choice_margin_reason}"
                    )
                ),
                frontier_choice_margin_components=_format_score_components(
                    recommended_repo_followup.frontier_choice_margin_components
                ),
                rank=recommended_repo_followup.selection_rank,
                opportunity=(
                    f"{recommended_repo_followup.opportunity_cost_score}:"
                    f"{recommended_repo_followup.opportunity_cost_reason}"
                ),
                opportunity_components=_format_score_components(
                    recommended_repo_followup.opportunity_cost_components
                ),
            )
        )
        print(
            "recommended_repo_followup_cohort: {count} :: {cohort}".format(
                count=len(recommended_repo_followup.cofrontier_followup_cohort),
                cohort=_format_repo_followup_cohort(
                    recommended_repo_followup.cofrontier_followup_cohort
                ),
            )
        )
    if recommended_repo_code_followup is None:
        print("recommended_repo_code_followup: <none>")
    else:
        print(
            "recommended_repo_code_followup: {family} :: owner={owner} :: {action_kind} :: {object_id} :: count={count} :: blocker={blocker} :: utility={utility} :: utility_components={utility_components} :: certainty={certainty} :: scope={scope} :: runner_up_followup={runner_up_followup} :: frontier_choice_margin={frontier_choice_margin} :: frontier_choice_margin_components={frontier_choice_margin_components} :: rank={rank} :: opportunity={opportunity} :: opportunity_components={opportunity_components}".format(
                family=recommended_repo_code_followup.followup_family,
                owner=recommended_repo_code_followup.owner_object_id or "<none>",
                action_kind=recommended_repo_code_followup.action_kind,
                object_id=recommended_repo_code_followup.object_id or "<none>",
                count=recommended_repo_code_followup.count,
                blocker=recommended_repo_code_followup.readiness_class or "none",
                utility=(
                    f"{recommended_repo_code_followup.utility_score}:{recommended_repo_code_followup.utility_reason}"
                ),
                utility_components=_format_score_components(
                    recommended_repo_code_followup.utility_components
                ),
                certainty=(
                    f"{recommended_repo_code_followup.selection_certainty_kind}:"
                    f"{recommended_repo_code_followup.cofrontier_followup_count}"
                ),
                scope=(
                    f"{recommended_repo_code_followup.selection_scope_kind}:"
                    f"{recommended_repo_code_followup.selection_scope_id or '<none>'}"
                ),
                runner_up_followup=(
                    "<none>"
                    if recommended_repo_code_followup.runner_up_followup_family is None
                    else (
                        f"{recommended_repo_code_followup.runner_up_followup_family}:"
                        f"{recommended_repo_code_followup.runner_up_followup_class or 'none'}:"
                        f"{recommended_repo_code_followup.runner_up_followup_object_id or '<none>'}:"
                        f"{recommended_repo_code_followup.runner_up_followup_utility_score}"
                    )
                ),
                frontier_choice_margin=(
                    "none"
                    if recommended_repo_code_followup.frontier_choice_margin_score is None
                    else (
                        f"{recommended_repo_code_followup.frontier_choice_margin_score}:"
                        f"{recommended_repo_code_followup.frontier_choice_margin_reason}"
                    )
                ),
                frontier_choice_margin_components=_format_score_components(
                    recommended_repo_code_followup.frontier_choice_margin_components
                ),
                rank=recommended_repo_code_followup.selection_rank,
                opportunity=(
                    f"{recommended_repo_code_followup.opportunity_cost_score}:"
                    f"{recommended_repo_code_followup.opportunity_cost_reason}"
                ),
                opportunity_components=_format_score_components(
                    recommended_repo_code_followup.opportunity_cost_components
                ),
            )
        )
    if recommended_repo_human_followup is None:
        print("recommended_repo_human_followup: <none>")
    elif recommended_repo_human_followup.diagnostic_code is not None:
        print(
            "recommended_repo_human_followup: {family} :: diagnostic={diagnostic} :: owner={owner} :: seed={seed} :: seed_object={seed_object} :: owner_kind={owner_kind} :: owner_score={owner_score} :: owner_options={owner_options} :: runner_up_owner={runner_up_owner} :: runner_up_kind={runner_up_kind} :: runner_up_score={runner_up_score} :: owner_choice_margin={owner_choice_margin} :: owner_choice_margin_components={owner_choice_margin_components} :: owner_option_tradeoff={owner_option_tradeoff} :: owner_option_tradeoff_components={owner_option_tradeoff_components} :: count={count} :: action={action} :: utility={utility} :: utility_components={utility_components} :: certainty={certainty} :: scope={scope} :: runner_up_followup={runner_up_followup} :: frontier_choice_margin={frontier_choice_margin} :: frontier_choice_margin_components={frontier_choice_margin_components} :: rank={rank} :: opportunity={opportunity} :: opportunity_components={opportunity_components}".format(
                family=recommended_repo_human_followup.followup_family,
                diagnostic=recommended_repo_human_followup.diagnostic_code,
                owner=recommended_repo_human_followup.owner_object_id or "<none>",
                seed=recommended_repo_human_followup.owner_seed_path or "<none>",
                seed_object=recommended_repo_human_followup.owner_seed_object_id or "<none>",
                owner_kind=recommended_repo_human_followup.owner_resolution_kind or "none",
                owner_score=(
                    "none"
                    if recommended_repo_human_followup.owner_resolution_score is None
                    else recommended_repo_human_followup.owner_resolution_score
                ),
                owner_options=_format_owner_resolution_options(
                    recommended_repo_human_followup.owner_resolution_options
                ),
                runner_up_owner=(
                    recommended_repo_human_followup.runner_up_owner_object_id
                    or "<none>"
                ),
                runner_up_kind=(
                    recommended_repo_human_followup.runner_up_owner_resolution_kind
                    or "none"
                ),
                runner_up_score=(
                    "none"
                    if recommended_repo_human_followup.runner_up_owner_resolution_score is None
                    else recommended_repo_human_followup.runner_up_owner_resolution_score
                ),
                owner_choice_margin=(
                    "none"
                    if recommended_repo_human_followup.owner_choice_margin_score is None
                    else (
                        f"{recommended_repo_human_followup.owner_choice_margin_score}:"
                        f"{recommended_repo_human_followup.owner_choice_margin_reason}"
                    )
                ),
                owner_choice_margin_components=_format_score_components(
                    recommended_repo_human_followup.owner_choice_margin_components
                ),
                owner_option_tradeoff=(
                    "none"
                    if recommended_repo_human_followup.owner_option_tradeoff_score
                    is None
                    else (
                        f"{recommended_repo_human_followup.owner_option_tradeoff_score}:"
                        f"{recommended_repo_human_followup.owner_option_tradeoff_reason}"
                    )
                ),
                owner_option_tradeoff_components=_format_score_components(
                    recommended_repo_human_followup.owner_option_tradeoff_components
                ),
                count=recommended_repo_human_followup.count,
                action=recommended_repo_human_followup.recommended_action or "none",
                utility=(
                    f"{recommended_repo_human_followup.utility_score}:{recommended_repo_human_followup.utility_reason}"
                ),
                utility_components=_format_score_components(
                    recommended_repo_human_followup.utility_components
                ),
                certainty=(
                    f"{recommended_repo_human_followup.selection_certainty_kind}:"
                    f"{recommended_repo_human_followup.cofrontier_followup_count}"
                ),
                scope=(
                    f"{recommended_repo_human_followup.selection_scope_kind}:"
                    f"{recommended_repo_human_followup.selection_scope_id or '<none>'}"
                ),
                runner_up_followup=(
                    "<none>"
                    if recommended_repo_human_followup.runner_up_followup_family is None
                    else (
                        f"{recommended_repo_human_followup.runner_up_followup_family}:"
                        f"{recommended_repo_human_followup.runner_up_followup_class or 'none'}:"
                        f"{recommended_repo_human_followup.runner_up_followup_object_id or '<none>'}:"
                        f"{recommended_repo_human_followup.runner_up_followup_utility_score}"
                    )
                ),
                frontier_choice_margin=(
                    "none"
                    if recommended_repo_human_followup.frontier_choice_margin_score
                    is None
                    else (
                        f"{recommended_repo_human_followup.frontier_choice_margin_score}:"
                        f"{recommended_repo_human_followup.frontier_choice_margin_reason}"
                    )
                ),
                frontier_choice_margin_components=_format_score_components(
                    recommended_repo_human_followup.frontier_choice_margin_components
                ),
                rank=recommended_repo_human_followup.selection_rank,
                opportunity=(
                    f"{recommended_repo_human_followup.opportunity_cost_score}:"
                    f"{recommended_repo_human_followup.opportunity_cost_reason}"
                ),
                opportunity_components=_format_score_components(
                    recommended_repo_human_followup.opportunity_cost_components
                ),
            )
        )
    else:
        print(
            "recommended_repo_human_followup: {family} :: target_doc={target_doc} :: alignment={alignment} :: action={action} :: utility={utility} :: utility_components={utility_components} :: certainty={certainty} :: scope={scope} :: runner_up_followup={runner_up_followup} :: frontier_choice_margin={frontier_choice_margin} :: frontier_choice_margin_components={frontier_choice_margin_components} :: rank={rank} :: opportunity={opportunity} :: opportunity_components={opportunity_components}".format(
                family=recommended_repo_human_followup.followup_family,
                target_doc=recommended_repo_human_followup.target_doc_id or "<none>",
                alignment=recommended_repo_human_followup.alignment_status or "none",
                action=recommended_repo_human_followup.recommended_action or "none",
                utility=(
                    f"{recommended_repo_human_followup.utility_score}:{recommended_repo_human_followup.utility_reason}"
                ),
                utility_components=_format_score_components(
                    recommended_repo_human_followup.utility_components
                ),
                certainty=(
                    f"{recommended_repo_human_followup.selection_certainty_kind}:"
                    f"{recommended_repo_human_followup.cofrontier_followup_count}"
                ),
                scope=(
                    f"{recommended_repo_human_followup.selection_scope_kind}:"
                    f"{recommended_repo_human_followup.selection_scope_id or '<none>'}"
                ),
                runner_up_followup=(
                    "<none>"
                    if recommended_repo_human_followup.runner_up_followup_family is None
                    else (
                        f"{recommended_repo_human_followup.runner_up_followup_family}:"
                        f"{recommended_repo_human_followup.runner_up_followup_class or 'none'}:"
                        f"{recommended_repo_human_followup.runner_up_followup_object_id or '<none>'}:"
                        f"{recommended_repo_human_followup.runner_up_followup_utility_score}"
                    )
                ),
                frontier_choice_margin=(
                    "none"
                    if recommended_repo_human_followup.frontier_choice_margin_score
                    is None
                    else (
                        f"{recommended_repo_human_followup.frontier_choice_margin_score}:"
                        f"{recommended_repo_human_followup.frontier_choice_margin_reason}"
                    )
                ),
                frontier_choice_margin_components=_format_score_components(
                    recommended_repo_human_followup.frontier_choice_margin_components
                ),
                rank=recommended_repo_human_followup.selection_rank,
                opportunity=(
                    f"{recommended_repo_human_followup.opportunity_cost_score}:"
                    f"{recommended_repo_human_followup.opportunity_cost_reason}"
                ),
                opportunity_components=_format_score_components(
                    recommended_repo_human_followup.opportunity_cost_components
                ),
            )
        )
    if recommended_repo_followup_lane is None:
        print("recommended_repo_followup_lane: <none>")
    else:
        print(
            "recommended_repo_followup_lane: {family} :: class={klass} :: roots={roots} :: rank={rank} :: utility={utility} :: utility_components={utility_components} :: opportunity={opportunity} :: opportunity_components={opportunity_components}".format(
                family=recommended_repo_followup_lane.followup_family,
                klass=recommended_repo_followup_lane.followup_class,
                roots=",".join(recommended_repo_followup_lane.root_object_ids)
                or "<none>",
                rank=recommended_repo_followup_lane.selection_rank,
                utility=(
                    f"{recommended_repo_followup_lane.lane_utility_score}:{recommended_repo_followup_lane.lane_utility_reason}"
                ),
                utility_components=_format_score_components(
                    recommended_repo_followup_lane.lane_utility_components
                ),
                opportunity=(
                    f"{recommended_repo_followup_lane.opportunity_cost_score}:{recommended_repo_followup_lane.opportunity_cost_reason}"
                ),
                opportunity_components=_format_score_components(
                    recommended_repo_followup_lane.opportunity_cost_components
                ),
            )
        )
    if recommended_repo_code_followup_lane is None:
        print("recommended_repo_code_followup_lane: <none>")
    else:
        print(
            "recommended_repo_code_followup_lane: {family} :: class={klass} :: roots={roots} :: rank={rank} :: utility={utility} :: utility_components={utility_components} :: opportunity={opportunity} :: opportunity_components={opportunity_components}".format(
                family=recommended_repo_code_followup_lane.followup_family,
                klass=recommended_repo_code_followup_lane.followup_class,
                roots=",".join(recommended_repo_code_followup_lane.root_object_ids)
                or "<none>",
                rank=recommended_repo_code_followup_lane.selection_rank,
                utility=(
                    f"{recommended_repo_code_followup_lane.lane_utility_score}:{recommended_repo_code_followup_lane.lane_utility_reason}"
                ),
                utility_components=_format_score_components(
                    recommended_repo_code_followup_lane.lane_utility_components
                ),
                opportunity=(
                    f"{recommended_repo_code_followup_lane.opportunity_cost_score}:{recommended_repo_code_followup_lane.opportunity_cost_reason}"
                ),
                opportunity_components=_format_score_components(
                    recommended_repo_code_followup_lane.opportunity_cost_components
                ),
            )
        )
    if recommended_repo_human_followup_lane is None:
        print("recommended_repo_human_followup_lane: <none>")
    else:
        print(
            "recommended_repo_human_followup_lane: {family} :: class={klass} :: roots={roots} :: rank={rank} :: utility={utility} :: utility_components={utility_components} :: opportunity={opportunity} :: opportunity_components={opportunity_components}".format(
                family=recommended_repo_human_followup_lane.followup_family,
                klass=recommended_repo_human_followup_lane.followup_class,
                roots=",".join(recommended_repo_human_followup_lane.root_object_ids)
                or "<none>",
                rank=recommended_repo_human_followup_lane.selection_rank,
                utility=(
                    f"{recommended_repo_human_followup_lane.lane_utility_score}:{recommended_repo_human_followup_lane.lane_utility_reason}"
                ),
                utility_components=_format_score_components(
                    recommended_repo_human_followup_lane.lane_utility_components
                ),
                opportunity=(
                    f"{recommended_repo_human_followup_lane.opportunity_cost_score}:{recommended_repo_human_followup_lane.opportunity_cost_reason}"
                ),
                opportunity_components=_format_score_components(
                    recommended_repo_human_followup_lane.opportunity_cost_components
                ),
            )
        )
    if recommended_repo_followup_frontier_tradeoff is None:
        print("recommended_repo_followup_frontier_tradeoff: <none>")
    else:
        print(
            "recommended_repo_followup_frontier_tradeoff: {frontier_family}:{frontier_class}:{frontier_utility} :: runner_up={runner_up_family}:{runner_up_class}:{runner_up_utility} :: margin={margin} :: margin_components={margin_components}".format(
                frontier_family=(
                    recommended_repo_followup_frontier_tradeoff.frontier_followup_family
                ),
                frontier_class=(
                    recommended_repo_followup_frontier_tradeoff.frontier_followup_class
                ),
                frontier_utility=(
                    f"{recommended_repo_followup_frontier_tradeoff.frontier_lane_utility_score}:"
                    f"{recommended_repo_followup_frontier_tradeoff.frontier_lane_utility_reason}"
                ),
                runner_up_family=(
                    recommended_repo_followup_frontier_tradeoff.runner_up_followup_family
                ),
                runner_up_class=(
                    recommended_repo_followup_frontier_tradeoff.runner_up_followup_class
                ),
                runner_up_utility=(
                    f"{recommended_repo_followup_frontier_tradeoff.runner_up_lane_utility_score}:"
                    f"{recommended_repo_followup_frontier_tradeoff.runner_up_lane_utility_reason}"
                ),
                margin=(
                    f"{recommended_repo_followup_frontier_tradeoff.margin_score}:"
                    f"{recommended_repo_followup_frontier_tradeoff.margin_reason}"
                ),
                margin_components=_format_score_components(
                    recommended_repo_followup_frontier_tradeoff.margin_components
                ),
            )
        )
    if recommended_repo_followup_frontier_explanation is None:
        print("recommended_repo_followup_frontier_explanation: <none>")
    else:
        print(
            "recommended_repo_followup_frontier_explanation: frontier={frontier_family}:{frontier_class}:{frontier_action}:{frontier_target}:{frontier_policy_ids}:{frontier_utility} :: same_class={same_class_runner_up}:{same_class_utility}:{same_class_margin}:{same_class_margin_components} :: cross_class={cross_class_runner_up}:{cross_class_utility}:{cross_class_margin}:{cross_class_margin_components} :: rationale={rationale_kind}:{rationale_reason}:{rationale_components}".format(
                frontier_family=(
                    recommended_repo_followup_frontier_explanation.frontier_followup_family
                ),
                frontier_class=(
                    recommended_repo_followup_frontier_explanation.frontier_followup_class
                ),
                frontier_action=(
                    recommended_repo_followup_frontier_explanation.frontier_action_kind
                ),
                frontier_target=(
                    recommended_repo_followup_frontier_explanation.frontier_object_id
                    or recommended_repo_followup_frontier_explanation.frontier_diagnostic_code
                    or recommended_repo_followup_frontier_explanation.frontier_target_doc_id
                    or "<none>"
                ),
                frontier_policy_ids=",".join(
                    recommended_repo_followup_frontier_explanation.frontier_policy_ids
                )
                or "<none>",
                frontier_utility=(
                    f"{recommended_repo_followup_frontier_explanation.frontier_utility_score}:"
                    f"{recommended_repo_followup_frontier_explanation.frontier_utility_reason}"
                ),
                same_class_runner_up=(
                    recommended_repo_followup_frontier_explanation.same_class_runner_up_object_id
                    or recommended_repo_followup_frontier_explanation.same_class_runner_up_diagnostic_code
                    or recommended_repo_followup_frontier_explanation.same_class_runner_up_target_doc_id
                    or "<none>"
                ),
                same_class_utility=(
                    "<none>"
                    if recommended_repo_followup_frontier_explanation.same_class_runner_up_utility_score is None
                    or recommended_repo_followup_frontier_explanation.same_class_runner_up_utility_reason is None
                    else f"{recommended_repo_followup_frontier_explanation.same_class_runner_up_utility_score}:{recommended_repo_followup_frontier_explanation.same_class_runner_up_utility_reason}"
                ),
                same_class_margin=(
                    "<none>"
                    if recommended_repo_followup_frontier_explanation.same_class_margin_score is None
                    or recommended_repo_followup_frontier_explanation.same_class_margin_reason is None
                    else f"{recommended_repo_followup_frontier_explanation.same_class_margin_score}:{recommended_repo_followup_frontier_explanation.same_class_margin_reason}"
                ),
                same_class_margin_components=_format_score_components(
                    recommended_repo_followup_frontier_explanation.same_class_margin_components
                ),
                cross_class_runner_up=(
                    recommended_repo_followup_frontier_explanation.cross_class_runner_up_object_id
                    or recommended_repo_followup_frontier_explanation.cross_class_runner_up_diagnostic_code
                    or recommended_repo_followup_frontier_explanation.cross_class_runner_up_target_doc_id
                    or "<none>"
                ),
                cross_class_utility=(
                    "<none>"
                    if recommended_repo_followup_frontier_explanation.cross_class_runner_up_utility_score is None
                    or recommended_repo_followup_frontier_explanation.cross_class_runner_up_utility_reason is None
                    else f"{recommended_repo_followup_frontier_explanation.cross_class_runner_up_utility_score}:{recommended_repo_followup_frontier_explanation.cross_class_runner_up_utility_reason}"
                ),
                cross_class_margin=(
                    "<none>"
                    if recommended_repo_followup_frontier_explanation.cross_class_margin_score is None
                    or recommended_repo_followup_frontier_explanation.cross_class_margin_reason is None
                    else f"{recommended_repo_followup_frontier_explanation.cross_class_margin_score}:{recommended_repo_followup_frontier_explanation.cross_class_margin_reason}"
                ),
                cross_class_margin_components=_format_score_components(
                    recommended_repo_followup_frontier_explanation.cross_class_margin_components
                ),
                rationale_kind=(
                    recommended_repo_followup_frontier_explanation.recommendation_rationale_kind
                ),
                rationale_reason=(
                    recommended_repo_followup_frontier_explanation.recommendation_rationale_reason
                ),
                rationale_components=_format_score_components(
                    recommended_repo_followup_frontier_explanation.recommendation_rationale_components
                ),
            )
        )
    if recommended_repo_followup_decision_protocol is None:
        print("recommended_repo_followup_decision_protocol: <none>")
    else:
        print(
            "recommended_repo_followup_decision_protocol: {frontier_family}:{frontier_class}:{frontier_action}:{frontier_target}:{frontier_policy_ids}:{frontier_utility} :: mode={mode} :: pressure=same_class:{same_class_pressure}|cross_class:{cross_class_pressure} :: reason={reason} :: components={components}".format(
                frontier_family=(
                    recommended_repo_followup_decision_protocol.frontier_followup_family
                ),
                frontier_class=(
                    recommended_repo_followup_decision_protocol.frontier_followup_class
                ),
                frontier_action=(
                    recommended_repo_followup_decision_protocol.frontier_action_kind
                ),
                frontier_target=(
                    recommended_repo_followup_decision_protocol.frontier_object_id
                    or recommended_repo_followup_decision_protocol.frontier_diagnostic_code
                    or recommended_repo_followup_decision_protocol.frontier_target_doc_id
                    or "<none>"
                ),
                frontier_policy_ids=",".join(
                    recommended_repo_followup_decision_protocol.frontier_policy_ids
                )
                or "<none>",
                frontier_utility=(
                    f"{recommended_repo_followup_decision_protocol.frontier_utility_score}:"
                    f"{recommended_repo_followup_decision_protocol.frontier_utility_reason}"
                ),
                mode=recommended_repo_followup_decision_protocol.decision_mode,
                same_class_pressure=(
                    recommended_repo_followup_decision_protocol.same_class_pressure
                ),
                cross_class_pressure=(
                    recommended_repo_followup_decision_protocol.cross_class_pressure
                ),
                reason=recommended_repo_followup_decision_protocol.decision_reason,
                components=_format_score_components(
                    recommended_repo_followup_decision_protocol.decision_components
                ),
            )
        )
    if recommended_repo_followup_frontier_triad is None:
        print("recommended_repo_followup_frontier_triad: <none>")
    else:
        same_class_tradeoff = recommended_repo_followup_frontier_triad.same_class_tradeoff
        cross_class_tradeoff = recommended_repo_followup_frontier_triad.cross_class_tradeoff
        print(
            "recommended_repo_followup_frontier_triad: frontier={frontier_family}:{frontier_class}:{frontier_action}:{frontier_target}:{frontier_policy_ids}:{frontier_utility} :: same_class_runner_up={same_class_runner_up} :: same_class_margin={same_class_margin} :: cross_class_runner_up={cross_class_runner_up} :: cross_class_margin={cross_class_margin}".format(
                frontier_family=recommended_repo_followup_frontier_triad.frontier_followup_family,
                frontier_class=recommended_repo_followup_frontier_triad.frontier_followup_class,
                frontier_action=recommended_repo_followup_frontier_triad.frontier_action_kind,
                frontier_target=(
                    recommended_repo_followup_frontier_triad.frontier_object_id
                    or recommended_repo_followup_frontier_triad.frontier_diagnostic_code
                    or recommended_repo_followup_frontier_triad.frontier_target_doc_id
                    or "<none>"
                ),
                frontier_policy_ids=",".join(
                    recommended_repo_followup_frontier_triad.frontier_policy_ids
                )
                or "<none>",
                frontier_utility=(
                    f"{recommended_repo_followup_frontier_triad.frontier_utility_score}:"
                    f"{recommended_repo_followup_frontier_triad.frontier_utility_reason}"
                ),
                same_class_runner_up=(
                    "<none>"
                    if same_class_tradeoff is None
                    else (
                        f"{same_class_tradeoff.runner_up_followup_family}:"
                        f"{same_class_tradeoff.runner_up_followup_class}:"
                        f"{same_class_tradeoff.runner_up_action_kind}:"
                        f"{same_class_tradeoff.runner_up_object_id or same_class_tradeoff.runner_up_diagnostic_code or same_class_tradeoff.runner_up_target_doc_id or '<none>'}:"
                        f"{','.join(same_class_tradeoff.runner_up_policy_ids) or '<none>'}:"
                        f"{same_class_tradeoff.runner_up_utility_score}:"
                        f"{same_class_tradeoff.runner_up_utility_reason}"
                    )
                ),
                same_class_margin=(
                    "<none>"
                    if same_class_tradeoff is None
                    else f"{same_class_tradeoff.margin_score}:{same_class_tradeoff.margin_reason}"
                ),
                cross_class_runner_up=(
                    "<none>"
                    if cross_class_tradeoff is None
                    else (
                        f"{cross_class_tradeoff.runner_up_followup_family}:"
                        f"{cross_class_tradeoff.runner_up_followup_class}:"
                        f"{cross_class_tradeoff.runner_up_action_kind}:"
                        f"{cross_class_tradeoff.runner_up_object_id or cross_class_tradeoff.runner_up_diagnostic_code or cross_class_tradeoff.runner_up_target_doc_id or '<none>'}:"
                        f"{cross_class_tradeoff.runner_up_utility_score}:"
                        f"{cross_class_tradeoff.runner_up_utility_reason}"
                    )
                ),
                cross_class_margin=(
                    "<none>"
                    if cross_class_tradeoff is None
                    else f"{cross_class_tradeoff.margin_score}:{cross_class_tradeoff.margin_reason}"
                ),
            )
        )
    if recommended_repo_followup_same_class_tradeoff is None:
        print("recommended_repo_followup_same_class_tradeoff: <none>")
    else:
        print(
            "recommended_repo_followup_same_class_tradeoff: {frontier_family}:{frontier_class}:{frontier_action}:{frontier_target}:{frontier_policy_ids}:{frontier_utility} :: runner_up={runner_up_family}:{runner_up_class}:{runner_up_action}:{runner_up_target}:{runner_up_policy_ids}:{runner_up_utility} :: margin={margin} :: margin_components={margin_components}".format(
                frontier_family=(
                    recommended_repo_followup_same_class_tradeoff.frontier_followup_family
                ),
                frontier_class=(
                    recommended_repo_followup_same_class_tradeoff.frontier_followup_class
                ),
                frontier_action=(
                    recommended_repo_followup_same_class_tradeoff.frontier_action_kind
                ),
                frontier_target=(
                    recommended_repo_followup_same_class_tradeoff.frontier_object_id
                    or recommended_repo_followup_same_class_tradeoff.frontier_diagnostic_code
                    or recommended_repo_followup_same_class_tradeoff.frontier_target_doc_id
                    or "<none>"
                ),
                frontier_policy_ids=",".join(
                    recommended_repo_followup_same_class_tradeoff.frontier_policy_ids
                )
                or "<none>",
                frontier_utility=(
                    f"{recommended_repo_followup_same_class_tradeoff.frontier_utility_score}:"
                    f"{recommended_repo_followup_same_class_tradeoff.frontier_utility_reason}"
                ),
                runner_up_family=(
                    recommended_repo_followup_same_class_tradeoff.runner_up_followup_family
                ),
                runner_up_class=(
                    recommended_repo_followup_same_class_tradeoff.runner_up_followup_class
                ),
                runner_up_action=(
                    recommended_repo_followup_same_class_tradeoff.runner_up_action_kind
                ),
                runner_up_target=(
                    recommended_repo_followup_same_class_tradeoff.runner_up_object_id
                    or recommended_repo_followup_same_class_tradeoff.runner_up_diagnostic_code
                    or recommended_repo_followup_same_class_tradeoff.runner_up_target_doc_id
                    or "<none>"
                ),
                runner_up_policy_ids=",".join(
                    recommended_repo_followup_same_class_tradeoff.runner_up_policy_ids
                )
                or "<none>",
                runner_up_utility=(
                    f"{recommended_repo_followup_same_class_tradeoff.runner_up_utility_score}:"
                    f"{recommended_repo_followup_same_class_tradeoff.runner_up_utility_reason}"
                ),
                margin=(
                    f"{recommended_repo_followup_same_class_tradeoff.margin_score}:"
                    f"{recommended_repo_followup_same_class_tradeoff.margin_reason}"
                ),
                margin_components=_format_score_components(
                    recommended_repo_followup_same_class_tradeoff.margin_components
                ),
            )
        )
    if recommended_repo_followup_cross_class_tradeoff is None:
        print("recommended_repo_followup_cross_class_tradeoff: <none>")
    else:
        print(
            "recommended_repo_followup_cross_class_tradeoff: {frontier_family}:{frontier_class}:{frontier_action}:{frontier_target}:{frontier_utility} :: runner_up={runner_up_family}:{runner_up_class}:{runner_up_action}:{runner_up_target}:{runner_up_utility} :: margin={margin} :: margin_components={margin_components}".format(
                frontier_family=(
                    recommended_repo_followup_cross_class_tradeoff.frontier_followup_family
                ),
                frontier_class=(
                    recommended_repo_followup_cross_class_tradeoff.frontier_followup_class
                ),
                frontier_action=(
                    recommended_repo_followup_cross_class_tradeoff.frontier_action_kind
                ),
                frontier_target=(
                    recommended_repo_followup_cross_class_tradeoff.frontier_object_id
                    or recommended_repo_followup_cross_class_tradeoff.frontier_diagnostic_code
                    or recommended_repo_followup_cross_class_tradeoff.frontier_target_doc_id
                    or "<none>"
                ),
                frontier_utility=(
                    f"{recommended_repo_followup_cross_class_tradeoff.frontier_utility_score}:"
                    f"{recommended_repo_followup_cross_class_tradeoff.frontier_utility_reason}"
                ),
                runner_up_family=(
                    recommended_repo_followup_cross_class_tradeoff.runner_up_followup_family
                ),
                runner_up_class=(
                    recommended_repo_followup_cross_class_tradeoff.runner_up_followup_class
                ),
                runner_up_action=(
                    recommended_repo_followup_cross_class_tradeoff.runner_up_action_kind
                ),
                runner_up_target=(
                    recommended_repo_followup_cross_class_tradeoff.runner_up_object_id
                    or recommended_repo_followup_cross_class_tradeoff.runner_up_diagnostic_code
                    or recommended_repo_followup_cross_class_tradeoff.runner_up_target_doc_id
                    or "<none>"
                ),
                runner_up_utility=(
                    f"{recommended_repo_followup_cross_class_tradeoff.runner_up_utility_score}:"
                    f"{recommended_repo_followup_cross_class_tradeoff.runner_up_utility_reason}"
                ),
                margin=(
                    f"{recommended_repo_followup_cross_class_tradeoff.margin_score}:"
                    f"{recommended_repo_followup_cross_class_tradeoff.margin_reason}"
                ),
                margin_components=_format_score_components(
                    recommended_repo_followup_cross_class_tradeoff.margin_components
                ),
            )
        )
    print("repo_followup_lanes:")
    for lane in repo_followup_lanes:
        best = lane.best_followup
        target = best.object_id or best.target_doc_id or best.diagnostic_code or "<none>"
        print(
            "- {family} :: class={klass} :: roots={roots} :: actions={actions} :: best={action_kind}::{target} :: owner_strength={owner_strength} :: utility={utility} :: lane_utility={lane_utility} :: lane_components={lane_components} :: rank={rank} :: opportunity={opportunity} :: opportunity_components={opportunity_components}".format(
                family=lane.followup_family,
                klass=lane.followup_class,
                roots=",".join(lane.root_object_ids) or "<none>",
                actions=lane.action_count,
                action_kind=best.action_kind,
                target=target,
                owner_strength=(
                    "none"
                    if lane.strongest_owner_resolution_kind is None
                    else (
                        f"{lane.strongest_owner_resolution_kind}:{lane.strongest_owner_resolution_score}"
                    )
                ),
                utility=f"{lane.strongest_utility_score}:{lane.strongest_utility_reason}",
                lane_utility=f"{lane.lane_utility_score}:{lane.lane_utility_reason}",
                lane_components=_format_score_components(
                    lane.lane_utility_components
                ),
                rank=lane.selection_rank,
                opportunity=f"{lane.opportunity_cost_score}:{lane.opportunity_cost_reason}",
                opportunity_components=_format_score_components(
                    lane.opportunity_cost_components
                ),
            )
        )
    print("repo_diagnostic_lanes:")
    for lane in repo_diagnostic_lanes:
        policy_ids = ", ".join(lane.policy_ids) if lane.policy_ids else "<none>"
        best_option = lane.candidate_owner_options[0] if lane.candidate_owner_options else None
        runner_up_option = lane.runner_up_candidate_owner_option
        print(
            "- {title} :: code={code} :: severity={severity} :: count={count} :: source={source} :: policy_ids={policy_ids} :: owner_status={owner_status} :: owner={owner} :: seed={seed} :: seed_object={seed_object} :: best_option={best_option} :: best_option_components={best_option_components} :: runner_up_option={runner_up_option} :: runner_up_components={runner_up_components} :: choice_margin={choice_margin} :: action={action}".format(
                title=lane.title,
                code=lane.diagnostic_code,
                severity=lane.severity,
                count=lane.count,
                source=(
                    f"{lane.rel_path}::{lane.qualname}"
                    if lane.rel_path
                    else "<none>"
                ),
                policy_ids=policy_ids,
                owner_status=lane.candidate_owner_status,
                owner=lane.candidate_owner_object_id or "<none>",
                seed=lane.candidate_owner_seed_path or "<none>",
                seed_object=lane.candidate_owner_seed_object_id or "<none>",
                best_option=(
                    "<none>"
                    if best_option is None
                    else f"{best_option.resolution_kind}:{best_option.object_id}:{best_option.score}"
                ),
                best_option_components=(
                    "none"
                    if best_option is None
                    else _format_score_components(best_option.score_components)
                ),
                runner_up_option=(
                    "<none>"
                    if runner_up_option is None
                    else (
                        f"{runner_up_option.resolution_kind}:"
                        f"{runner_up_option.object_id}:"
                        f"{runner_up_option.score}"
                    )
                ),
                runner_up_components=(
                    "none"
                    if runner_up_option is None
                    else _format_score_components(runner_up_option.score_components)
                ),
                choice_margin=(
                    "none"
                    if lane.candidate_owner_choice_margin_score is None
                    else (
                        f"{lane.candidate_owner_choice_margin_score}:"
                        f"{lane.candidate_owner_choice_margin_reason}"
                    )
                ),
                action=lane.recommended_action,
            )
        )
    node_kind_counts = counts.get("node_kind_counts", {})
    if isinstance(node_kind_counts, dict):
        print("node kinds:")
        for key, value in _sorted(
            [(str(key), int(value)) for key, value in node_kind_counts.items()],
            key=lambda item: item[0],
        ):
            print(f"- {key}: {value}")
    edge_kind_counts = counts.get("edge_kind_counts", {})
    if isinstance(edge_kind_counts, dict):
        print("edge kinds:")
        for key, value in _sorted(
            [(str(key), int(value)) for key, value in edge_kind_counts.items()],
            key=lambda item: item[0],
        ):
            print(f"- {key}: {value}")


def _ownership_chain(graph: InvariantGraph, node_id: str) -> tuple[str, ...]:
    node_by_id = graph.node_by_id()
    edges_to = graph.edges_to()
    chain: list[str] = []
    current = node_id
    seen: set[str] = set()
    while current not in seen:
        seen.add(current)
        chain.append(current)
        parents = [
            edge.source_id
            for edge in edges_to.get(current, ())
            if edge.edge_kind == "contains" and edge.source_id in node_by_id
        ]
        if not parents:
            break
        current = _sorted(parents)[0]
    return tuple(reversed(chain))


def _print_trace(*, graph: InvariantGraph, raw_id: str) -> int:
    nodes = trace_nodes(graph, raw_id)
    if not nodes:
        print(f"no invariant-graph nodes matched: {raw_id}")
        return 1
    node_by_id = graph.node_by_id()
    edges_from = graph.edges_from()
    edges_to = graph.edges_to()
    for index, node in enumerate(nodes, start=1):
        if index > 1:
            print("")
        print(f"node: {node.node_id}")
        print(f"kind: {node.node_kind}")
        print(f"title: {node.title}")
        if node.marker_id:
            print(f"marker_id: {node.marker_id}")
        if node.object_ids:
            print(f"object_ids: {', '.join(node.object_ids)}")
        if node.doc_ids:
            print(f"doc_ids: {', '.join(node.doc_ids)}")
        if node.policy_ids:
            print(f"policy_ids: {', '.join(node.policy_ids)}")
        if node.invariant_ids:
            print(f"invariant_ids: {', '.join(node.invariant_ids)}")
        print(f"site_identity: {node.site_identity}")
        print(f"structural_identity: {node.structural_identity}")
        if node.rel_path:
            print(
                f"location: {node.rel_path}:{node.line}:{node.column} "
                f"({node.qualname or '<none>'})"
            )
        if node.reasoning_control:
            print(f"reasoning.control: {node.reasoning_control}")
        if node.blocking_dependencies:
            print(
                "blocking_dependencies: "
                + ", ".join(node.blocking_dependencies)
            )
        chain = _ownership_chain(graph, node.node_id)
        if chain:
            print("ownership_chain:")
            for value in chain:
                owner = node_by_id.get(value)
                print(f"- {value} :: {owner.title if owner is not None else '<missing>'}")
        outbound = [
            edge for edge in edges_from.get(node.node_id, ()) if edge.target_id in node_by_id
        ]
        inbound = [
            edge for edge in edges_to.get(node.node_id, ()) if edge.source_id in node_by_id
        ]
        print("outbound_edges:")
        for edge in outbound:
            target = node_by_id[edge.target_id]
            print(f"- {edge.edge_kind}: {target.node_id} :: {target.title}")
        if not outbound:
            print("- <none>")
        print("inbound_edges:")
        for edge in inbound:
            source = node_by_id[edge.source_id]
            print(f"- {edge.edge_kind}: {source.node_id} :: {source.title}")
        if not inbound:
            print("- <none>")
        descendant_ids = _descendant_ids(graph, node.node_id)
        coverage_ids, signal_ids = _coverage_and_signal_ids(graph, node_ids=descendant_ids)
        print(f"coverage_summary: tests={len(coverage_ids)}")
        if coverage_ids:
            print("covering_tests:")
            for test_case_id in coverage_ids[:20]:
                print(f"- {node_by_id[test_case_id].title}")
        print(f"policy_signal_summary: signals={len(signal_ids)}")
        if signal_ids:
            print("policy_signals:")
            for signal_id in signal_ids[:20]:
                signal_node = node_by_id[signal_id]
                print(f"- {signal_node.title} :: {signal_node.reasoning_summary}")
    return 0


def _print_blockers(*, graph: InvariantGraph, object_id: str) -> int:
    grouped = blocker_chains(graph, object_id=object_id)
    if not grouped:
        print(f"no blocker chains for object_id: {object_id}")
        return 1
    node_by_id = graph.node_by_id()
    for seam_class, chains in _sorted(list(grouped.items()), key=lambda item: item[0]):
        print(f"{seam_class}:")
        for chain in chains:
            rendered = " -> ".join(
                f"{node_id} ({node_by_id[node_id].title})"
                for node_id in chain
                if node_id in node_by_id
            )
            print(f"- {rendered}")
    return 0


def _ledger_projection_item(
    *,
    ledger_artifact: Path,
    object_id: str,
) -> dict[str, object] | None:
    if not ledger_artifact.exists():
        return None
    payload = load_invariant_ledger_projections(ledger_artifact)
    ledgers = payload.get("ledgers", [])
    if not isinstance(ledgers, list):
        return None
    for item in ledgers:
        if isinstance(item, dict) and str(item.get("object_id", "")) == object_id:
            return item
    return None


def _print_workstream(*, graph: InvariantGraph, root: Path, object_id: str) -> int:
    workstream = _workstream_by_object_id(graph=graph, root=root, object_id=object_id)
    if workstream is None:
        print(f"no workstream projection for object_id: {object_id}")
        return 1
    health_summary = workstream.health_summary()
    print(f"object_id: {workstream.object_id.wire()}")
    print(f"title: {workstream.title}")
    print(f"status: {workstream.status}")
    print(f"touchsites: {workstream.touchsite_count}")
    print(f"collapsible_touchsites: {workstream.collapsible_touchsite_count}")
    print(f"surviving_touchsites: {workstream.surviving_touchsite_count}")
    print(f"policy_signals: {workstream.policy_signal_count}")
    print(f"coverage_count: {workstream.coverage_count}")
    print(f"diagnostics: {workstream.diagnostic_count}")
    print(
        "health_summary: covered={covered} :: uncovered={uncovered} :: governed={governed} :: diagnosed={diagnosed}".format(
            covered=health_summary.covered_touchsite_count,
            uncovered=health_summary.uncovered_touchsite_count,
            governed=health_summary.governed_touchsite_count,
            diagnosed=health_summary.diagnosed_touchsite_count,
        )
    )
    print(
        "touchsite_blockers: ready={ready} :: coverage_gap={coverage_gap} :: policy={policy} :: diagnostic={diagnostic}".format(
            ready=health_summary.ready_touchsite_count,
            coverage_gap=health_summary.coverage_gap_touchsite_count,
            policy=health_summary.policy_blocked_touchsite_count,
            diagnostic=health_summary.diagnostic_blocked_touchsite_count,
        )
    )
    print(
        "health_cuts: touchpoints(ready={tp_ready}, coverage_gap={tp_gap}, policy={tp_policy}, diagnostic={tp_diag}) :: "
        "subqueues(ready={sq_ready}, coverage_gap={sq_gap}, policy={sq_policy}, diagnostic={sq_diag})".format(
            tp_ready=health_summary.ready_touchpoint_cut_count,
            tp_gap=health_summary.coverage_gap_touchpoint_cut_count,
            tp_policy=health_summary.policy_blocked_touchpoint_cut_count,
            tp_diag=health_summary.diagnostic_blocked_touchpoint_cut_count,
            sq_ready=health_summary.ready_subqueue_cut_count,
            sq_gap=health_summary.coverage_gap_subqueue_cut_count,
            sq_policy=health_summary.policy_blocked_subqueue_cut_count,
            sq_diag=health_summary.diagnostic_blocked_subqueue_cut_count,
        )
    )
    print(f"dominant_blocker_class: {workstream.dominant_blocker_class()}")
    print(f"recommended_remediation_family: {workstream.recommended_remediation_family()}")
    recommended_cut = workstream.recommended_cut()
    recommended_ready_cut = workstream.recommended_ready_cut()
    recommended_coverage_gap_cut = workstream.recommended_coverage_gap_cut()
    recommended_policy_blocked_cut = workstream.recommended_policy_blocked_cut()
    recommended_diagnostic_blocked_cut = workstream.recommended_diagnostic_blocked_cut()
    recommended_cut_frontier_explanation = (
        workstream.recommended_cut_frontier_explanation()
    )
    recommended_cut_decision_protocol = workstream.recommended_cut_decision_protocol()
    recommended_cut_frontier_stability = workstream.recommended_cut_frontier_stability()
    if recommended_cut is None:
        print("recommended_cut: <none>")
    else:
        print(
            "recommended_cut: {cut_kind} :: {object_id} :: touchsites={touchsites} :: surviving={surviving}".format(
                cut_kind=recommended_cut.cut_kind,
                object_id=recommended_cut.object_id.wire(),
                touchsites=recommended_cut.touchsite_count,
                surviving=recommended_cut.surviving_touchsite_count,
            )
        )
    if recommended_ready_cut is None:
        print("recommended_ready_cut: <none>")
    else:
        print(
            "recommended_ready_cut: {cut_kind} :: {object_id} :: touchsites={touchsites} :: uncovered={uncovered}".format(
                cut_kind=recommended_ready_cut.cut_kind,
                object_id=recommended_ready_cut.object_id.wire(),
                touchsites=recommended_ready_cut.touchsite_count,
                uncovered=recommended_ready_cut.uncovered_touchsite_count,
            )
        )
    if recommended_coverage_gap_cut is None:
        print("recommended_coverage_gap_cut: <none>")
    else:
        print(
            "recommended_coverage_gap_cut: {cut_kind} :: {object_id} :: touchsites={touchsites} :: uncovered={uncovered}".format(
                cut_kind=recommended_coverage_gap_cut.cut_kind,
                object_id=recommended_coverage_gap_cut.object_id.wire(),
                touchsites=recommended_coverage_gap_cut.touchsite_count,
                uncovered=recommended_coverage_gap_cut.uncovered_touchsite_count,
            )
        )
    if recommended_policy_blocked_cut is None:
        print("recommended_policy_blocked_cut: <none>")
    else:
        print(
            "recommended_policy_blocked_cut: {cut_kind} :: {object_id} :: touchsites={touchsites} :: signals={signals}".format(
                cut_kind=recommended_policy_blocked_cut.cut_kind,
                object_id=recommended_policy_blocked_cut.object_id.wire(),
                touchsites=recommended_policy_blocked_cut.touchsite_count,
                signals=recommended_policy_blocked_cut.policy_signal_count,
            )
        )
    if recommended_diagnostic_blocked_cut is None:
        print("recommended_diagnostic_blocked_cut: <none>")
    else:
        print(
            "recommended_diagnostic_blocked_cut: {cut_kind} :: {object_id} :: touchsites={touchsites} :: diagnostics={diagnostics}".format(
                cut_kind=recommended_diagnostic_blocked_cut.cut_kind,
                object_id=recommended_diagnostic_blocked_cut.object_id.wire(),
                touchsites=recommended_diagnostic_blocked_cut.touchsite_count,
                diagnostics=recommended_diagnostic_blocked_cut.diagnostic_count,
            )
        )
    if recommended_cut_frontier_explanation is None:
        print("recommended_cut_frontier_explanation: <none>")
    else:
        same_kind_tradeoff = recommended_cut_frontier_explanation.same_kind_tradeoff
        cross_kind_tradeoff = recommended_cut_frontier_explanation.cross_kind_tradeoff
        print(
            "recommended_cut_frontier_explanation: frontier={frontier_kind}::{frontier_object_id}::{frontier_readiness} :: same_kind={same_kind_object_id}:{same_kind_margin_kind}:{same_kind_margin_score} :: cross_kind={cross_kind_object_id}:{cross_kind_margin_kind}:{cross_kind_margin_score} :: rationale={rationale_kind}".format(
                frontier_kind=recommended_cut_frontier_explanation.frontier_cut_kind,
                frontier_object_id=recommended_cut_frontier_explanation.frontier_object_id,
                frontier_readiness=recommended_cut_frontier_explanation.frontier_readiness_class,
                same_kind_object_id=(
                    "<none>"
                    if same_kind_tradeoff is None
                    else same_kind_tradeoff.runner_up_object_id
                ),
                same_kind_margin_kind=(
                    "none"
                    if same_kind_tradeoff is None
                    else same_kind_tradeoff.margin_kind
                ),
                same_kind_margin_score=(
                    "none"
                    if same_kind_tradeoff is None
                    else same_kind_tradeoff.margin_score
                ),
                cross_kind_object_id=(
                    "<none>"
                    if cross_kind_tradeoff is None
                    else cross_kind_tradeoff.runner_up_object_id
                ),
                cross_kind_margin_kind=(
                    "none"
                    if cross_kind_tradeoff is None
                    else cross_kind_tradeoff.margin_kind
                ),
                cross_kind_margin_score=(
                    "none"
                    if cross_kind_tradeoff is None
                    else cross_kind_tradeoff.margin_score
                ),
                rationale_kind=recommended_cut_frontier_explanation.recommendation_rationale_kind,
            )
        )
    if recommended_cut_decision_protocol is None:
        print("recommended_cut_decision_protocol: <none>")
    else:
        print(
            "recommended_cut_decision_protocol: {cut_kind} :: {object_id} :: mode={mode} :: pressure=same_kind:{same_kind}|cross_kind:{cross_kind}".format(
                cut_kind=recommended_cut_decision_protocol.frontier_cut_kind,
                object_id=recommended_cut_decision_protocol.frontier_object_id,
                mode=recommended_cut_decision_protocol.decision_mode,
                same_kind=recommended_cut_decision_protocol.same_kind_pressure,
                cross_kind=recommended_cut_decision_protocol.cross_kind_pressure,
            )
        )
    if recommended_cut_frontier_stability is None:
        print("recommended_cut_frontier_stability: <none>")
    else:
        print(
            "recommended_cut_frontier_stability: {cut_kind} :: {object_id} :: kind={kind} :: pressure=same_kind:{same_kind}|cross_kind:{cross_kind}".format(
                cut_kind=recommended_cut_frontier_stability.frontier_cut_kind,
                object_id=recommended_cut_frontier_stability.frontier_object_id,
                kind=recommended_cut_frontier_stability.stability_kind,
                same_kind=recommended_cut_frontier_stability.same_kind_pressure,
                cross_kind=recommended_cut_frontier_stability.cross_kind_pressure,
            )
        )
    recommended_followup = workstream.recommended_followup()
    if recommended_followup is None:
        print("recommended_followup: <none>")
    elif recommended_followup.action_kind == "doc_alignment":
        print(
            "recommended_followup: {family} :: target_doc={target_doc} :: alignment={alignment} :: action={action}".format(
                family=recommended_followup.followup_family,
                target_doc=recommended_followup.target_doc_id or "<none>",
                alignment=recommended_followup.alignment_status or "none",
                action=recommended_followup.recommended_action or "none",
            )
        )
    else:
        print(
            "recommended_followup: {family} :: {action_kind} :: {object_id} :: touchsites={touchsites} :: blocker={blocker}".format(
                family=recommended_followup.followup_family,
                action_kind=recommended_followup.action_kind,
                object_id=recommended_followup.object_id or "<none>",
                touchsites=recommended_followup.touchsite_count,
                blocker=recommended_followup.readiness_class or "none",
            )
        )
    print("ranked_followups:")
    ranked_followups = workstream.ranked_followups()
    if not ranked_followups:
        print("- <none>")
    else:
        for item in ranked_followups:
            if item.action_kind == "doc_alignment":
                print(
                    "- {family} :: target_doc={target_doc} :: alignment={alignment} :: action={action}".format(
                        family=item.followup_family,
                        target_doc=item.target_doc_id or "<none>",
                        alignment=item.alignment_status or "none",
                        action=item.recommended_action or "none",
                    )
                )
            else:
                print(
                    "- {family} :: {action_kind} :: {object_id} :: readiness={readiness} :: touchsites={touchsites} :: surviving={surviving}".format(
                        family=item.followup_family,
                        action_kind=item.action_kind,
                        object_id=item.object_id or "<none>",
                        readiness=item.readiness_class or "none",
                        touchsites=item.touchsite_count,
                        surviving=item.surviving_touchsite_count,
                    )
                )
    print("remediation_lanes:")
    remediation_lanes = workstream.remediation_lanes()
    if not remediation_lanes:
        print("- <none>")
    else:
        for lane in remediation_lanes:
            best_cut = lane.best_cut
            best_cut_rendered = (
                "<none>"
                if best_cut is None
                else "{cut_kind}::{object_id}".format(
                    cut_kind=best_cut.cut_kind,
                    object_id=best_cut.object_id.wire(),
                )
            )
            print(
                "- {family} :: blocker={blocker} :: touchsites={touchsites} :: touchpoints={touchpoints} :: subqueues={subqueues} :: best={best}".format(
                    family=lane.remediation_family,
                    blocker=lane.blocker_class,
                    touchsites=lane.touchsite_count,
                    touchpoints=lane.touchpoint_cut_count,
                    subqueues=lane.subqueue_cut_count,
                    best=best_cut_rendered,
                )
            )
    print("ranked_touchpoint_cuts:")
    touchpoint_cuts = workstream.ranked_touchpoint_cuts()
    if not touchpoint_cuts:
        print("- <none>")
    else:
        for item in touchpoint_cuts:
            print(
                "- {object_id} :: readiness={readiness} :: touchsites={touchsites} :: collapsible={collapsible} :: surviving={surviving} :: uncovered={uncovered}".format(
                    object_id=item.object_id.wire(),
                    readiness=item.readiness_class,
                    touchsites=item.touchsite_count,
                    collapsible=item.collapsible_touchsite_count,
                    surviving=item.surviving_touchsite_count,
                    uncovered=item.uncovered_touchsite_count,
                )
            )
    print("ranked_subqueue_cuts:")
    subqueue_cuts = workstream.ranked_subqueue_cuts()
    if not subqueue_cuts:
        print("- <none>")
    else:
        for item in subqueue_cuts:
            print(
                "- {object_id} :: readiness={readiness} :: touchsites={touchsites} :: collapsible={collapsible} :: surviving={surviving} :: uncovered={uncovered}".format(
                    object_id=item.object_id.wire(),
                    readiness=item.readiness_class,
                    touchsites=item.touchsite_count,
                    collapsible=item.collapsible_touchsite_count,
                    surviving=item.surviving_touchsite_count,
                    uncovered=item.uncovered_touchsite_count,
                )
            )
    print("subqueues:")
    subqueues = tuple(workstream.iter_subqueues())
    if not subqueues:
        print("- <none>")
    else:
        for item in subqueues:
            print(
                "- {object_id} :: {status} :: touchsites={touchsites} :: signals={signals} :: coverage={coverage}".format(
                    object_id=item.object_id.wire(),
                    status=item.status,
                    touchsites=item.touchsite_count,
                    signals=item.policy_signal_count,
                    coverage=item.coverage_count,
                )
            )
    alignment_summary = workstream.doc_alignment_summary
    if alignment_summary is not None:
        print(
            "ledger_alignment_summary: target_docs={target_docs} :: reflected={reflected} :: append_existing={append_existing} :: append_new={append_new} :: missing={missing} :: ambiguous={ambiguous} :: unassigned={unassigned}".format(
                target_docs=alignment_summary.target_doc_count,
                reflected=alignment_summary.reflected_target_doc_count,
                append_existing=alignment_summary.append_pending_existing_target_doc_count,
                append_new=alignment_summary.append_pending_new_target_doc_count,
                missing=alignment_summary.missing_target_doc_count,
                ambiguous=alignment_summary.ambiguous_target_doc_count,
                unassigned=alignment_summary.unassigned_target_doc_count,
            )
        )
        print(
            "dominant_doc_alignment_status: {status}".format(
                status=workstream.dominant_doc_alignment_status()
            )
        )
        print(
            "recommended_doc_alignment_action: {action}".format(
                action=workstream.recommended_doc_alignment_action()
            )
        )
        print(
            "next_human_followup_family: {family}".format(
                family=workstream.next_human_followup_family()
            )
        )
        print(
            "recommended_doc_followup_target_doc_id: {doc_id}".format(
                doc_id=(
                    workstream.recommended_doc_followup_target_doc_id()
                    or "<none>"
                )
            )
        )
        misaligned_target_doc_ids = workstream.misaligned_target_doc_ids()
        print(
            "misaligned_target_doc_ids: "
            + (
                ", ".join(str(value) for value in misaligned_target_doc_ids)
                if misaligned_target_doc_ids
                else "<none>"
            )
        )
        print("documentation_followup_lanes:")
        lane = workstream.documentation_followup_lane()
        if lane is None:
            print("- <none>")
        else:
            print(
                "- {family} :: alignment={alignment} :: target_docs={target_docs} :: misaligned={misaligned} :: best={best} :: action={action}".format(
                    family=lane.followup_family,
                    alignment=lane.alignment_status,
                    target_docs=lane.target_doc_count,
                    misaligned=lane.misaligned_target_doc_count,
                    best=lane.best_target_doc_id or "<none>",
                    action=lane.recommended_action,
                )
            )
    return 0


def _print_blast_radius(
    *,
    graph: InvariantGraph,
    raw_id: str,
    impact_artifact: Path | None,
) -> int:
    node_by_id = graph.node_by_id()
    nodes = trace_nodes(graph, raw_id)
    if not nodes:
        print(f"no invariant-graph nodes matched: {raw_id}")
        return 1
    descendant_ids = tuple(
        _sorted(
            list(
                {
                    descendant_id
                    for node in nodes
                    for descendant_id in _descendant_ids(graph, node.node_id)
                }
            )
        )
    )
    coverage_ids, _signal_ids = _coverage_and_signal_ids(graph, node_ids=descendant_ids)
    impacted_tests = set(_load_impacted_tests(impact_artifact))
    print(f"node_matches: {len(nodes)}")
    print(f"covering_tests: {len(coverage_ids)}")
    if impact_artifact is not None:
        print(f"impact_artifact: {impact_artifact}")
    for test_case_id in coverage_ids:
        test_id = node_by_id[test_case_id].title
        suffix = " [impacted]" if test_id in impacted_tests else ""
        print(f"- {test_id}{suffix}")
    return 0


def _print_perf_heat_map(
    *,
    root: Path,
    graph: InvariantGraph,
    raw_id: str,
    perf_artifact: Path | None,
) -> int:
    if perf_artifact is None:
        print("perf_artifact: <none>")
        return 1
    node_by_id = graph.node_by_id()
    nodes = trace_nodes(graph, raw_id)
    if not nodes:
        print(f"no invariant-graph nodes matched: {raw_id}")
        return 1
    descendant_ids = tuple(
        _sorted(
            list(
                {
                    descendant_id
                    for node in nodes
                    for descendant_id in _descendant_ids(graph, node.node_id)
                }
            )
        )
    )
    overlay = _resolve_perf_dsl_overlay(
        root=root,
        graph=graph,
        scope_node_ids=descendant_ids,
    )
    candidate_node_ids = (
        overlay.candidate_node_ids if overlay.candidate_node_ids else descendant_ids
    )
    observations = _load_profile_observations(perf_artifact)
    matched = _match_profile_observations(
        graph=graph,
        descendant_ids=candidate_node_ids,
        observations=observations,
    )
    infimum_buckets = _perf_infimum_buckets(
        graph=graph,
        descendant_ids=descendant_ids,
        matched=matched,
    )
    print(f"node_matches: {len(nodes)}")
    print(f"perf_artifact: {perf_artifact}")
    print("perf_query_overlay:")
    print(
        "  doc_ids: "
        + (", ".join(overlay.doc_ids) if overlay.doc_ids else "<none>")
    )
    print(
        "  doc_paths: "
        + (", ".join(overlay.doc_paths) if overlay.doc_paths else "<none>")
    )
    print(
        "  target_symbols: "
        + (", ".join(overlay.target_symbols) if overlay.target_symbols else "<none>")
    )
    print(f"  candidate_nodes: {len(candidate_node_ids)}")
    print(
        "  source: "
        + ("dsl_overlay" if overlay.candidate_node_ids else "invariant_descendants")
    )
    print(f"profile_observations: {len(observations)}")
    print(f"matched_profile_observations: {len(matched)}")
    if not matched:
        print("perf_metric_buckets:")
        print("- <none>")
        return 0
    metric_keys = _sorted(
        list({(item.metric_kind, item.unit) for item in matched}),
        key=lambda item: (item[0], item[1]),
    )
    print("perf_metric_buckets:")
    for metric_kind, unit in metric_keys:
        bucket = [
            item
            for item in matched
            if item.metric_kind == metric_kind and item.unit == unit
        ]
        total = sum(item.inclusive_value for item in bucket)
        print(f"- {metric_kind}:{unit} total={total:g}")
        for item in bucket:
            node = node_by_id[item.node_id]
            print(
                "  - {profiler} :: {value:g} :: {path}:{line}::{qualname} :: {title} :: node_kind={node_kind}".format(
                    profiler=item.profiler,
                    value=item.inclusive_value,
                    path=item.rel_path,
                    line=item.line,
                    qualname=item.qualname,
                    title=item.title,
                    node_kind=node.node_kind,
                )
            )
    print("perf_infimum_buckets:")
    if not infimum_buckets:
        print("- <none>")
        return 0
    infimum_metric_keys = _sorted(
        list({(item.metric_kind, item.unit) for item in infimum_buckets}),
        key=lambda item: (item[0], item[1]),
    )
    for metric_kind, unit in infimum_metric_keys:
        print(f"- {metric_kind}:{unit}")
        metric_buckets = [
            item
            for item in infimum_buckets
            if item.metric_kind == metric_kind and item.unit == unit
        ]
        for item in metric_buckets:
            location = (
                f"{item.rel_path}:{item.line}::{item.qualname}"
                if item.rel_path and item.qualname and item.line > 0
                else item.node_id
            )
            print(
                "  - {total:g} :: leaves={leaves} :: depth={depth} :: global_infimum={global_infimum} :: virtual={virtual} :: {location} :: {title} :: node_kind={node_kind}".format(
                    total=item.total_inclusive_value,
                    leaves=item.matched_leaf_node_count,
                    depth=item.depth,
                    global_infimum="yes" if item.is_global_infimum else "no",
                    virtual="yes" if item.is_virtual_intersection else "no",
                    location=location,
                    title=item.title,
                    node_kind=item.node_kind,
                )
            )
    return 0


def _print_compare(
    *,
    root: Path,
    before_workstreams_artifact: Path,
    after_workstreams_artifact: Path,
    ledger_deltas_artifact: Path,
    ledger_deltas_markdown_artifact: Path,
    ledger_alignments_artifact: Path,
    ledger_alignments_markdown_artifact: Path,
    object_id: str | None,
) -> int:
    before_payload = load_invariant_workstreams(before_workstreams_artifact)
    after_payload = load_invariant_workstreams(after_workstreams_artifact)
    drifts = compare_invariant_workstreams(before_payload, after_payload)
    if object_id is not None:
        drifts = tuple(item for item in drifts if item.object_id == object_id)
        if not drifts:
            print(f"no workstream drift for object_id: {object_id}")
            return 1
    print(f"before: {before_workstreams_artifact}")
    print(f"after: {after_workstreams_artifact}")
    classification_counts: dict[str, int] = {}
    for item in drifts:
        classification_counts[item.classification] = (
            classification_counts.get(item.classification, 0) + 1
        )
    print("classification_counts:")
    for key, value in _sorted(
        list(classification_counts.items()),
        key=lambda item: item[0],
    ):
        print(f"- {key}: {value}")
    print("workstream_drifts:")
    for item in drifts:
        print(
            "- {object_id} :: {classification} :: touchsites={before}->{after} ({delta:+d}) :: surviving={before_surviving}->{after_surviving} ({surviving_delta:+d}) :: dominant={before_blocker}->{after_blocker} :: recommended={before_cut}->{after_cut}".format(
                object_id=item.object_id,
                classification=item.classification,
                before=item.before_touchsite_count,
                after=item.after_touchsite_count,
                delta=item.touchsite_delta,
                before_surviving=item.before_surviving_touchsite_count,
                after_surviving=item.after_surviving_touchsite_count,
                surviving_delta=item.surviving_touchsite_delta,
                before_blocker=item.before_dominant_blocker_class,
                after_blocker=item.after_dominant_blocker_class,
                before_cut=item.before_recommended_cut_object_id or "<none>",
                after_cut=item.after_recommended_cut_object_id or "<none>",
            )
        )
        print(
            "  blocker_deltas: ready={ready:+d} :: coverage_gap={coverage:+d} :: policy={policy:+d} :: diagnostic={diagnostic:+d}".format(
                ready=item.blocker_deltas.get("ready_touchsite_count", 0),
                coverage=item.blocker_deltas.get("coverage_gap_touchsite_count", 0),
                policy=item.blocker_deltas.get("policy_blocked_touchsite_count", 0),
                diagnostic=item.blocker_deltas.get("diagnostic_blocked_touchsite_count", 0),
            )
        )
        print(
            "  touchsite_identity_delta: added={added} :: removed={removed}".format(
                added=len(item.added_touchsite_ids),
                removed=len(item.removed_touchsite_ids),
            )
        )
    ledger_deltas = compare_invariant_ledger_projections(before_payload, after_payload)
    if object_id is not None:
        ledger_deltas = tuple(
            item for item in ledger_deltas if item.object_id == object_id
        )
    print("ledger_deltas:")
    for item in ledger_deltas:
        print(
            "- {object_id} :: {classification} :: action={action} :: docs={docs}".format(
                object_id=item.object_id,
                classification=item.classification,
                action=item.recommended_ledger_action,
                docs=(
                    "<none>"
                    if not item.target_doc_ids
                    else ",".join(item.target_doc_ids)
                ),
            )
        )
        print(f"  summary: {item.summary}")
    ledger_delta_projections = build_invariant_ledger_delta_projections(
        root=str(after_payload.get("root", after_workstreams_artifact.parent)),
        before_workstreams_artifact=str(before_workstreams_artifact),
        after_workstreams_artifact=str(after_workstreams_artifact),
        before_payload=before_payload,
        after_payload=after_payload,
    )
    if object_id is not None:
        filtered_deltas = tuple(
            item
            for item in ledger_delta_projections.iter_deltas()
            if item.object_id == object_id
        )
        ledger_delta_projections = type(ledger_delta_projections)(
            root=ledger_delta_projections.root,
            generated_at_utc=ledger_delta_projections.generated_at_utc,
            before_workstreams_artifact=ledger_delta_projections.before_workstreams_artifact,
            after_workstreams_artifact=ledger_delta_projections.after_workstreams_artifact,
            deltas=stream_from_factory(
                lambda filtered_deltas=filtered_deltas: iter(filtered_deltas)
            ),
        )
    write_invariant_ledger_deltas(ledger_deltas_artifact, ledger_delta_projections)
    write_invariant_ledger_deltas_markdown(
        ledger_deltas_markdown_artifact,
        ledger_delta_projections,
    )
    ledger_alignments = build_invariant_ledger_alignments(
        root=root,
        ledger_deltas=ledger_delta_projections,
    )
    write_invariant_ledger_alignments(
        ledger_alignments_artifact,
        ledger_alignments,
    )
    write_invariant_ledger_alignments_markdown(
        ledger_alignments_markdown_artifact,
        ledger_alignments,
    )
    print(f"ledger_delta_artifact: {ledger_deltas_artifact}")
    print(f"ledger_delta_markdown_artifact: {ledger_deltas_markdown_artifact}")
    alignment_payload = ledger_alignments.as_payload()
    status_counts = alignment_payload.get("counts", {}).get("status_counts", {})
    print("ledger_alignment_counts:")
    if isinstance(status_counts, dict):
        for key, value in _sorted(
            [(str(key), int(value)) for key, value in status_counts.items()],
            key=lambda item: item[0],
        ):
            print(f"- {key}: {value}")
    print(f"ledger_alignment_artifact: {ledger_alignments_artifact}")
    print(
        f"ledger_alignment_markdown_artifact: {ledger_alignments_markdown_artifact}"
    )
    return 0


def _print_ledger(*, ledger_artifact: Path, object_id: str | None) -> int:
    payload = load_invariant_ledger_projections(ledger_artifact)
    ledgers = payload.get("ledgers", [])
    if not isinstance(ledgers, list):
        print("invalid invariant ledger projections payload")
        return 1
    filtered = [
        item
        for item in ledgers
        if isinstance(item, dict)
        and (
            object_id is None
            or str(item.get("object_id", "")) == object_id
        )
    ]
    if not filtered:
        if object_id is None:
            print("no ledger projections available")
        else:
            print(f"no ledger projection for object_id: {object_id}")
        return 1
    for item in filtered:
        print(f"object_id: {item.get('object_id', '')}")
        print(f"title: {item.get('title', '')}")
        print(f"status: {item.get('status', '')}")
        doc_ids = item.get("target_doc_ids", [])
        if isinstance(doc_ids, list):
            print(
                "target_doc_ids: "
                + (", ".join(str(value) for value in doc_ids) if doc_ids else "<none>")
            )
        print(
            f"recommended_ledger_action: {item.get('recommended_ledger_action', '')}"
        )
        print(f"summary: {item.get('summary', '')}")
        current_snapshot = item.get("current_snapshot", {})
        if isinstance(current_snapshot, dict):
            recommended_cut = current_snapshot.get("recommended_cut_object_id")
            print(
                "current_snapshot: touchsites={touchsites} :: surviving={surviving} :: "
                "coverage={coverage} :: diagnostics={diagnostics} :: recommended_cut={recommended_cut}".format(
                    touchsites=current_snapshot.get("touchsite_count", 0),
                    surviving=current_snapshot.get("surviving_touchsite_count", 0),
                    coverage=current_snapshot.get("coverage_count", 0),
                    diagnostics=current_snapshot.get("diagnostic_count", 0),
                    recommended_cut=recommended_cut or "<none>",
                )
            )
        alignment_summary = item.get("alignment_summary")
        if isinstance(alignment_summary, dict):
            print(
                "alignment_summary: target_docs={target_docs} :: reflected={reflected} :: append_existing={append_existing} :: append_new={append_new} :: missing={missing} :: ambiguous={ambiguous} :: unassigned={unassigned}".format(
                    target_docs=alignment_summary.get("target_doc_count", 0),
                    reflected=alignment_summary.get("reflected_target_doc_count", 0),
                    append_existing=alignment_summary.get(
                        "append_pending_existing_target_doc_count", 0
                    ),
                    append_new=alignment_summary.get(
                        "append_pending_new_target_doc_count", 0
                    ),
                    missing=alignment_summary.get("missing_target_doc_count", 0),
                    ambiguous=alignment_summary.get("ambiguous_target_doc_count", 0),
                    unassigned=alignment_summary.get("unassigned_target_doc_count", 0),
                )
            )
            print(
                "recommended_doc_alignment_action: {action}".format(
                    action=alignment_summary.get(
                        "recommended_doc_alignment_action", "none"
                    )
                )
            )
        target_doc_alignments = item.get("target_doc_alignments", [])
        if isinstance(target_doc_alignments, list):
            print("target_doc_alignments:")
            if not target_doc_alignments:
                print("- <none>")
            else:
                for alignment in target_doc_alignments:
                    if not isinstance(alignment, dict):
                        continue
                    print(
                        "- {doc_id} :: {status} :: path={path}".format(
                            doc_id=alignment.get("target_doc_id", ""),
                            status=alignment.get("alignment_status", ""),
                            path=alignment.get("target_doc_path", ""),
                        )
                    )
    return 0


def _print_ledger_deltas(
    *,
    ledger_deltas_artifact: Path,
    object_id: str | None,
    doc_id: str | None,
) -> int:
    payload = load_invariant_ledger_deltas(ledger_deltas_artifact)
    deltas = payload.get("deltas", [])
    if not isinstance(deltas, list):
        print("invalid invariant ledger deltas payload")
        return 1
    filtered = [
        item
        for item in deltas
        if isinstance(item, dict)
        and (object_id is None or str(item.get("object_id", "")) == object_id)
        and (
            doc_id is None
            or (
                isinstance(item.get("target_doc_ids"), list)
                and doc_id in [str(value) for value in item.get("target_doc_ids", [])]
            )
        )
    ]
    if not filtered:
        print("no ledger deltas available")
        return 1
    for item in filtered:
        print(f"object_id: {item.get('object_id', '')}")
        print(f"title: {item.get('title', '')}")
        print(f"classification: {item.get('classification', '')}")
        print(f"recommended_ledger_action: {item.get('recommended_ledger_action', '')}")
        doc_ids = item.get("target_doc_ids", [])
        if isinstance(doc_ids, list):
            print(
                "target_doc_ids: "
                + (", ".join(str(value) for value in doc_ids) if doc_ids else "<none>")
            )
        print(f"summary: {item.get('summary', '')}")
    return 0


def _print_ledger_alignments(
    *,
    ledger_alignments_artifact: Path,
    object_id: str | None,
    doc_id: str | None,
    status: str | None,
) -> int:
    payload = load_invariant_ledger_alignments(ledger_alignments_artifact)
    alignments = payload.get("alignments", [])
    if not isinstance(alignments, list):
        print("invalid invariant ledger alignments payload")
        return 1
    filtered = [
        item
        for item in alignments
        if isinstance(item, dict)
        and (object_id is None or str(item.get("object_id", "")) == object_id)
        and (doc_id is None or str(item.get("target_doc_id", "")) == doc_id)
        and (status is None or str(item.get("alignment_status", "")) == status)
    ]
    if not filtered:
        print("no ledger alignments available")
        return 1
    for item in filtered:
        print(f"object_id: {item.get('object_id', '')}")
        print(f"title: {item.get('title', '')}")
        print(f"target_doc_id: {item.get('target_doc_id', '')}")
        print(f"target_doc_path: {item.get('target_doc_path', '')}")
        print(f"alignment_status: {item.get('alignment_status', '')}")
        print(f"summary: {item.get('summary', '')}")
    return 0


def main(
    argv: list[str] | None = None,
    *,
    declared_registries: tuple[WorkstreamRegistry, ...] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--artifact", default=str(_DEFAULT_ARTIFACT))
    parser.add_argument(
        "--workstreams-artifact",
        default=str(_DEFAULT_WORKSTREAMS_ARTIFACT),
    )
    parser.add_argument(
        "--ledger-artifact",
        default=str(_DEFAULT_LEDGER_ARTIFACT),
    )
    parser.add_argument(
        "--ledger-deltas-artifact",
        default=str(_DEFAULT_LEDGER_DELTAS_ARTIFACT),
    )
    parser.add_argument(
        "--ledger-deltas-markdown-artifact",
        default=str(_DEFAULT_LEDGER_DELTAS_MARKDOWN_ARTIFACT),
    )
    parser.add_argument(
        "--ledger-alignments-artifact",
        default=str(_DEFAULT_LEDGER_ALIGNMENTS_ARTIFACT),
    )
    parser.add_argument(
        "--ledger-alignments-markdown-artifact",
        default=str(_DEFAULT_LEDGER_ALIGNMENTS_MARKDOWN_ARTIFACT),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build")
    subparsers.add_parser("summary")

    trace_parser = subparsers.add_parser("trace")
    trace_parser.add_argument("--id", required=True)

    blockers_parser = subparsers.add_parser("blockers")
    blockers_parser.add_argument("--object-id", required=True)

    workstream_parser = subparsers.add_parser("workstream")
    workstream_parser.add_argument("--object-id", required=True)

    ledger_parser = subparsers.add_parser("ledger")
    ledger_parser.add_argument("--object-id", default=None)

    ledger_deltas_parser = subparsers.add_parser("ledger-deltas")
    ledger_deltas_parser.add_argument("--object-id", default=None)
    ledger_deltas_parser.add_argument("--doc-id", default=None)
    ledger_deltas_parser.add_argument(
        "--ledger-deltas-artifact",
        default=str(_DEFAULT_LEDGER_DELTAS_ARTIFACT),
    )

    ledger_alignments_parser = subparsers.add_parser("ledger-alignments")
    ledger_alignments_parser.add_argument("--object-id", default=None)
    ledger_alignments_parser.add_argument("--doc-id", default=None)
    ledger_alignments_parser.add_argument("--status", default=None)
    ledger_alignments_parser.add_argument(
        "--ledger-alignments-artifact",
        default=str(_DEFAULT_LEDGER_ALIGNMENTS_ARTIFACT),
    )

    blast_radius_parser = subparsers.add_parser("blast-radius")
    blast_radius_parser.add_argument("--id", required=True)
    blast_radius_parser.add_argument("--impact-artifact", default=None)

    perf_heat_parser = subparsers.add_parser("perf-heat")
    perf_heat_parser.add_argument("--id", required=True)
    perf_heat_parser.add_argument("--perf-artifact", required=True)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--root", default=".")
    compare_parser.add_argument("--before-workstreams-artifact", required=True)
    compare_parser.add_argument("--after-workstreams-artifact", required=True)
    compare_parser.add_argument("--object-id", default=None)
    compare_parser.add_argument(
        "--ledger-deltas-artifact",
        default=str(_DEFAULT_LEDGER_DELTAS_ARTIFACT),
    )
    compare_parser.add_argument(
        "--ledger-deltas-markdown-artifact",
        default=str(_DEFAULT_LEDGER_DELTAS_MARKDOWN_ARTIFACT),
    )
    compare_parser.add_argument(
        "--ledger-alignments-artifact",
        default=str(_DEFAULT_LEDGER_ALIGNMENTS_ARTIFACT),
    )
    compare_parser.add_argument(
        "--ledger-alignments-markdown-artifact",
        default=str(_DEFAULT_LEDGER_ALIGNMENTS_MARKDOWN_ARTIFACT),
    )

    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    artifact = Path(args.artifact).resolve()
    workstreams_artifact = Path(args.workstreams_artifact).resolve()
    ledger_artifact = Path(args.ledger_artifact).resolve()
    ledger_deltas_artifact = Path(args.ledger_deltas_artifact).resolve()
    ledger_deltas_markdown_artifact = Path(
        args.ledger_deltas_markdown_artifact
    ).resolve()
    ledger_alignments_artifact = Path(args.ledger_alignments_artifact).resolve()
    ledger_alignments_markdown_artifact = Path(
        args.ledger_alignments_markdown_artifact
    ).resolve()

    if args.command == "build":
        bundle = build_invariant_planning_bundle(
            root,
            declared_registries=declared_registries,
        )
        graph = bundle.graph
        workstreams = bundle.workstreams
        write_invariant_graph(artifact, graph)
        write_invariant_workstreams(workstreams_artifact, workstreams)
        write_invariant_ledger_projections(
            ledger_artifact,
            build_invariant_ledger_projections(workstreams, root=root),
        )
        print(str(artifact))
        return 0
    if args.command == "summary":
        _print_summary(
            graph=_load_or_build_graph(
                root=root,
                artifact=artifact,
                declared_registries=declared_registries,
            ),
            root=root,
        )
        return 0
    if args.command == "trace":
        return _print_trace(
            graph=_load_or_build_graph(
                root=root,
                artifact=artifact,
                declared_registries=declared_registries,
            ),
            raw_id=str(args.id),
        )
    if args.command == "blockers":
        return _print_blockers(
            graph=_load_or_build_graph(
                root=root,
                artifact=artifact,
                declared_registries=declared_registries,
            ),
            object_id=str(args.object_id),
        )
    if args.command == "workstream":
        return _print_workstream(
            graph=_load_or_build_graph(
                root=root,
                artifact=artifact,
                declared_registries=declared_registries,
            ),
            root=root,
            object_id=str(args.object_id),
        )
    if args.command == "ledger":
        return _print_ledger(
            ledger_artifact=ledger_artifact,
            object_id=None if args.object_id is None else str(args.object_id),
        )
    if args.command == "ledger-deltas":
        return _print_ledger_deltas(
            ledger_deltas_artifact=ledger_deltas_artifact,
            object_id=None if args.object_id is None else str(args.object_id),
            doc_id=None if args.doc_id is None else str(args.doc_id),
        )
    if args.command == "ledger-alignments":
        return _print_ledger_alignments(
            ledger_alignments_artifact=ledger_alignments_artifact,
            object_id=None if args.object_id is None else str(args.object_id),
            doc_id=None if args.doc_id is None else str(args.doc_id),
            status=None if args.status is None else str(args.status),
        )
    if args.command == "blast-radius":
        impact_artifact = (
            Path(args.impact_artifact).resolve()
            if args.impact_artifact is not None
            else None
        )
        return _print_blast_radius(
            graph=_load_or_build_graph(
                root=root,
                artifact=artifact,
                declared_registries=declared_registries,
            ),
            raw_id=str(args.id),
            impact_artifact=impact_artifact,
        )
    if args.command == "perf-heat":
        return _print_perf_heat_map(
            root=root,
            graph=_load_or_build_graph(
                root=root,
                artifact=artifact,
                declared_registries=declared_registries,
            ),
            raw_id=str(args.id),
            perf_artifact=Path(str(args.perf_artifact)).resolve(),
        )
    if args.command == "compare":
        return _print_compare(
            root=root,
            before_workstreams_artifact=Path(
                str(args.before_workstreams_artifact)
            ).resolve(),
            after_workstreams_artifact=Path(
                str(args.after_workstreams_artifact)
            ).resolve(),
            ledger_deltas_artifact=ledger_deltas_artifact,
            ledger_deltas_markdown_artifact=ledger_deltas_markdown_artifact,
            ledger_alignments_artifact=ledger_alignments_artifact,
            ledger_alignments_markdown_artifact=ledger_alignments_markdown_artifact,
            object_id=None if args.object_id is None else str(args.object_id),
        )
    return 1


__all__ = ["main"]
