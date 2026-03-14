#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gabion.order_contract import ordered_or_sorted
from gabion.tooling.policy_substrate.hotspot_queue_identity import (
    HOTSPOT_QUEUE_ZONE,
    HotspotQueueIdentitySpace,
)
from gabion.tooling.policy_substrate.identity_zone import (
    HierarchicalIdentityGrammar,
    IdentityCarrier,
)
from gabion.tooling.policy_substrate.policy_scanner_identity import (
    POLICY_SCANNER_ZONE,
    policy_scanner_carrier_from_payload,
)
from gabion.tooling.runtime import policy_result_schema
from gabion.tooling.policy_substrate.policy_artifact_stream import (
    ArtifactSourceRef,
    mapping_document,
    render_markdown,
    write_json,
    write_markdown,
)
from gabion.tooling.runtime.projection_fiber_semantics_summary import (
    projection_fiber_decision_from_payload,
    projection_fiber_semantic_bundle_count_from_payload,
    projection_fiber_semantic_previews_from_payload,
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


@dataclass(frozen=True, order=True)
class HotspotScopeRef:
    identity: str
    path: str
    identity_zone: str = field(default=HOTSPOT_QUEUE_ZONE.value, compare=False)
    identity_morphism: dict[str, object] = field(default_factory=dict, compare=False)
    kernel_congruence: dict[str, object] = field(default_factory=dict, compare=False)
    quotient_projection: dict[str, object] = field(default_factory=dict, compare=False)
    reflection_functor: dict[str, object] = field(default_factory=dict, compare=False)
    adjoint_pair: dict[str, object] = field(default_factory=dict, compare=False)
    fiber_witness: dict[str, object] = field(default_factory=dict, compare=False)


@dataclass(frozen=True, order=True)
class HotspotFileRef:
    identity: str
    path: str
    scope: HotspotScopeRef
    identity_zone: str = field(default=HOTSPOT_QUEUE_ZONE.value, compare=False)
    identity_morphism: dict[str, object] = field(default_factory=dict, compare=False)
    kernel_congruence: dict[str, object] = field(default_factory=dict, compare=False)
    quotient_projection: dict[str, object] = field(default_factory=dict, compare=False)
    reflection_functor: dict[str, object] = field(default_factory=dict, compare=False)
    adjoint_pair: dict[str, object] = field(default_factory=dict, compare=False)
    fiber_witness: dict[str, object] = field(default_factory=dict, compare=False)


def _sorted[T](values: list[T], *, key=None) -> list[T]:
    return ordered_or_sorted(values, source="scripts.policy.hotspot_neighborhood_queue", key=key)


def _count_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


@dataclass
class _HotspotIdentityContext:
    identities: HotspotQueueIdentitySpace
    grammar: HierarchicalIdentityGrammar
    _scope_cache: dict[str, HotspotScopeRef] = field(default_factory=dict)
    _file_cache: dict[str, HotspotFileRef] = field(default_factory=dict)


def _file_family_counts(
    payload: dict[str, Any],
    families: tuple[str, ...],
) -> dict[HotspotFileRef, Counter[str]]:
    violations = payload.get("violations")
    if not isinstance(violations, dict):
        return {}
    context = _HotspotIdentityContext(
        identities=HotspotQueueIdentitySpace(),
        grammar=HierarchicalIdentityGrammar(),
    )
    counts_by_path: defaultdict[str, Counter[str]] = defaultdict(Counter)
    source_carriers_by_path: defaultdict[str, list[IdentityCarrier[object, object, object]]] = defaultdict(list)
    for family in families:
        family_items = violations.get(family, [])
        if not isinstance(family_items, list):
            continue
        for item in family_items:
            if not isinstance(item, dict):
                continue
            path = _text(item.get("path"))
            if not path:
                continue
            counts_by_path[path][family] += 1
            source_carrier = _scanner_carrier_from_violation(item)
            if source_carrier is not None:
                source_carriers_by_path[path].append(source_carrier)
    counts_by_file: dict[HotspotFileRef, Counter[str]] = {}
    for path, counts in counts_by_path.items():
        file_ref = _file_ref(
            path=path,
            source_carriers=tuple(source_carriers_by_path.get(path, ())),
            context=context,
            scope_source_carriers=tuple(
                carrier
                for other_path, carriers in source_carriers_by_path.items()
                if str(Path(other_path).parent).replace("\\", "/")
                == str(Path(path).parent).replace("\\", "/")
                for carrier in carriers
            ),
        )
        counts_by_file[file_ref] = counts
    return dict(counts_by_file)


def _vector_for_path(
    *,
    counts_by_file: dict[HotspotFileRef, Counter[str]],
    path: HotspotFileRef,
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


def _text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _identity_payload(ref: HotspotScopeRef | HotspotFileRef) -> dict[str, object]:
    return {
        "identity_zone": ref.identity_zone,
        "identity_morphism": ref.identity_morphism,
        "kernel_congruence": ref.kernel_congruence,
        "quotient_projection": ref.quotient_projection,
        "reflection_functor": ref.reflection_functor,
        "adjoint_pair": ref.adjoint_pair,
        "fiber_witness": ref.fiber_witness,
    }


def _scope_ref(
    path: str,
    *,
    source_carriers: tuple[IdentityCarrier[object, object, object], ...],
    context: _HotspotIdentityContext,
) -> HotspotScopeRef:
    scope_path = str(Path(path).parent).replace("\\", "/")
    cached = context._scope_cache.get(scope_path)
    if cached is not None:
        return cached
    identity = context.identities.item_id(
        item_kind="scope",
        path=scope_path,
        label=scope_path,
    )
    target_carrier = identity.as_carrier()
    context.grammar.add_carrier(target_carrier)
    for source_carrier in source_carriers:
        context.grammar.add_carrier(source_carrier)
    member_source_wires = tuple(
        sorted({carrier.wire() for carrier in source_carriers})
    )
    chosen_wire = member_source_wires[0] if member_source_wires else ""
    kernel = context.grammar.add_kernel_congruence(
        source_zone=POLICY_SCANNER_ZONE.value,
        target_zone=HOTSPOT_QUEUE_ZONE.value,
        source_carrier_wire=chosen_wire or identity.wire(),
        retained_decomposition_kinds=("rel_path_segment",),
        erased_decomposition_kinds=(
            "rule_id",
            "qualname",
            "kind",
            "site_identity",
            "structural_identity",
            "line",
            "column",
        ),
        rationale="scope quotient retains parent path face",
    )
    quotient = context.grammar.add_quotient_projection(
        source_zone=POLICY_SCANNER_ZONE.value,
        target_zone=HOTSPOT_QUEUE_ZONE.value,
        source_carrier_wire=chosen_wire or identity.wire(),
        target_carrier_wire=identity.wire(),
        kernel_congruence_id=kernel.kernel_congruence_id,
        rationale="scanner->hotspot_scope",
    )
    fiber = context.grammar.add_fiber_witness(
        source_zone=POLICY_SCANNER_ZONE.value,
        target_zone=HOTSPOT_QUEUE_ZONE.value,
        target_carrier_wire=identity.wire(),
        member_source_wires=member_source_wires,
        chosen_representative_wire=chosen_wire,
    )
    reflection = context.grammar.add_reflection_functor(
        source_zone=HOTSPOT_QUEUE_ZONE.value,
        target_zone=POLICY_SCANNER_ZONE.value,
        source_carrier_wire=identity.wire(),
        target_carrier_wire=chosen_wire,
        section_kind="least_wire_representative",
        rationale="scope representative chosen from fiber",
    )
    adjoint = context.grammar.add_adjoint_pair(
        left_morphism_id=quotient.quotient_projection_id,
        right_morphism_id=reflection.reflection_functor_id,
        along_zone_boundary=f"{POLICY_SCANNER_ZONE.value}->{HOTSPOT_QUEUE_ZONE.value}",
        law_checks=("fiber_member", "deterministic_section"),
    )
    morphism = context.grammar.add_zone_morphism(
        source_zone=POLICY_SCANNER_ZONE.value,
        target_zone=HOTSPOT_QUEUE_ZONE.value,
        source_carrier_wire=chosen_wire or identity.wire(),
        target_carrier_wire=identity.wire(),
        morphism_kind="quotients_to",
        invertible=False,
        retained_decomposition_kinds=("rel_path_segment",),
        erased_decomposition_kinds=(
            "rule_id",
            "qualname",
            "kind",
            "site_identity",
            "structural_identity",
            "line",
            "column",
        ),
        kernel_congruence_id=kernel.kernel_congruence_id,
        quotient_projection_id=quotient.quotient_projection_id,
        reflection_functor_id=reflection.reflection_functor_id,
        adjoint_pair_id=adjoint.adjoint_pair_id,
        fiber_witness_ids=(fiber.fiber_witness_id,),
    )
    created = HotspotScopeRef(
        identity=identity.wire(),
        path=scope_path,
        identity_morphism=morphism.as_payload(),
        kernel_congruence=kernel.as_payload(),
        quotient_projection=quotient.as_payload(),
        reflection_functor=reflection.as_payload(),
        adjoint_pair=adjoint.as_payload(),
        fiber_witness=fiber.as_payload(),
    )
    context._scope_cache[scope_path] = created
    return created


def _scanner_carrier_from_violation(
    item: dict[str, object],
) -> IdentityCarrier[object, object, object] | None:
    identity_payload = item.get("identity")
    if isinstance(identity_payload, dict):
        return policy_scanner_carrier_from_payload(identity_payload)
    return None


def _file_ref(
    *,
    path: str,
    source_carriers: tuple[IdentityCarrier[object, object, object], ...],
    scope_source_carriers: tuple[IdentityCarrier[object, object, object], ...],
    context: _HotspotIdentityContext,
) -> HotspotFileRef:
    cached = context._file_cache.get(path)
    if cached is not None:
        return cached
    scope_ref = _scope_ref(
        path,
        source_carriers=scope_source_carriers,
        context=context,
    )
    identity = context.identities.item_id(
        item_kind="file",
        path=path,
        label=path,
    )
    target_carrier = identity.as_carrier()
    context.grammar.add_carrier(target_carrier)
    for source_carrier in source_carriers:
        context.grammar.add_carrier(source_carrier)
    member_source_wires = tuple(
        sorted({carrier.wire() for carrier in source_carriers})
    )
    chosen_wire = member_source_wires[0] if member_source_wires else ""
    kernel = context.grammar.add_kernel_congruence(
        source_zone=POLICY_SCANNER_ZONE.value,
        target_zone=HOTSPOT_QUEUE_ZONE.value,
        source_carrier_wire=chosen_wire or identity.wire(),
        retained_decomposition_kinds=("rel_path",),
        erased_decomposition_kinds=(
            "rule_id",
            "qualname",
            "kind",
            "site_identity",
            "structural_identity",
            "line",
            "column",
        ),
        rationale="file quotient retains rel_path face",
    )
    quotient = context.grammar.add_quotient_projection(
        source_zone=POLICY_SCANNER_ZONE.value,
        target_zone=HOTSPOT_QUEUE_ZONE.value,
        source_carrier_wire=chosen_wire or identity.wire(),
        target_carrier_wire=identity.wire(),
        kernel_congruence_id=kernel.kernel_congruence_id,
        rationale="scanner->hotspot_file",
    )
    fiber = context.grammar.add_fiber_witness(
        source_zone=POLICY_SCANNER_ZONE.value,
        target_zone=HOTSPOT_QUEUE_ZONE.value,
        target_carrier_wire=identity.wire(),
        member_source_wires=member_source_wires,
        chosen_representative_wire=chosen_wire,
    )
    reflection = context.grammar.add_reflection_functor(
        source_zone=HOTSPOT_QUEUE_ZONE.value,
        target_zone=POLICY_SCANNER_ZONE.value,
        source_carrier_wire=identity.wire(),
        target_carrier_wire=chosen_wire,
        section_kind="least_wire_representative",
        rationale="file representative chosen from fiber",
    )
    adjoint = context.grammar.add_adjoint_pair(
        left_morphism_id=quotient.quotient_projection_id,
        right_morphism_id=reflection.reflection_functor_id,
        along_zone_boundary=f"{POLICY_SCANNER_ZONE.value}->{HOTSPOT_QUEUE_ZONE.value}",
        law_checks=("fiber_member", "deterministic_section"),
    )
    morphism = context.grammar.add_zone_morphism(
        source_zone=POLICY_SCANNER_ZONE.value,
        target_zone=HOTSPOT_QUEUE_ZONE.value,
        source_carrier_wire=chosen_wire or identity.wire(),
        target_carrier_wire=identity.wire(),
        morphism_kind="quotients_to",
        invertible=False,
        retained_decomposition_kinds=("rel_path",),
        erased_decomposition_kinds=(
            "rule_id",
            "qualname",
            "kind",
            "site_identity",
            "structural_identity",
            "line",
            "column",
        ),
        kernel_congruence_id=kernel.kernel_congruence_id,
        quotient_projection_id=quotient.quotient_projection_id,
        reflection_functor_id=reflection.reflection_functor_id,
        adjoint_pair_id=adjoint.adjoint_pair_id,
        fiber_witness_ids=(fiber.fiber_witness_id,),
    )
    created = HotspotFileRef(
        identity=identity.wire(),
        path=path,
        scope=scope_ref,
        identity_morphism=morphism.as_payload(),
        kernel_congruence=kernel.as_payload(),
        quotient_projection=quotient.as_payload(),
        reflection_functor=reflection.as_payload(),
        adjoint_pair=adjoint.as_payload(),
        fiber_witness=fiber.as_payload(),
    )
    context._file_cache[path] = created
    return created


def _seed_paths(
    *,
    counts_by_file: dict[HotspotFileRef, Counter[str]],
    config: QueueConfig,
) -> list[HotspotFileRef]:
    candidates = [
        path
        for path, counts in counts_by_file.items()
        if _file_family_count(counts) >= config.min_seed_families
        and _file_total(counts) >= config.min_seed_total
    ]
    return _sorted(
        candidates,
        key=lambda path: (-_file_total(counts_by_file[path]), path.path),
    )


def _ring1_paths_for_seed(
    *,
    counts_by_file: dict[HotspotFileRef, Counter[str]],
    seed_path: HotspotFileRef,
) -> list[HotspotFileRef]:
    scope = seed_path.scope
    ring1 = [path for path in counts_by_file if path.scope == scope]
    return _sorted(
        ring1,
        key=lambda path: (-_file_total(counts_by_file[path]), path.path),
    )


def _ring2_neighbors_for_seed(
    *,
    counts_by_file: dict[HotspotFileRef, Counter[str]],
    seed_path: HotspotFileRef,
    ring1_paths: set[HotspotFileRef],
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
                "path": path.path,
                "path_identity": path.identity,
                **_identity_payload(path),
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
    counts_by_file: dict[HotspotFileRef, Counter[str]],
    ring1_paths: list[HotspotFileRef],
    families: tuple[str, ...],
) -> dict[str, Any]:
    files_payload = [
        {
            "path": path.path,
            "path_identity": path.identity,
            **_identity_payload(path),
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


def _family_totals(
    *,
    counts_by_file: dict[HotspotFileRef, Counter[str]],
    families: tuple[str, ...],
) -> dict[str, int]:
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
    semantic_previews = list(projection_fiber_semantic_previews_from_payload(payload))
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
                "ring_1_scope": seed_path.scope.path,
                "ring_1_scope_identity": seed_path.scope.identity,
                **{
                    f"ring_1_scope_{key}": value
                    for key, value in _identity_payload(seed_path.scope).items()
                },
                "seed_path": seed_path.path,
                "seed_path_identity": seed_path.identity,
                **{
                    f"seed_{key}": value
                    for key, value in _identity_payload(seed_path).items()
                },
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
        scope = str(candidate["ring_1_scope_identity"])
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
            str(item["ring_1_scope_identity"]),
            str(item["seed_path_identity"]),
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
            str(item["ring_1_scope_identity"]),
            str(item["seed_path_identity"]),
        ),
    )
    for index, neighborhood in enumerate(neighborhoods, start=1):
        neighborhood["rank"] = index
    for index, neighborhood in enumerate(large_zone_backlog, start=1):
        neighborhood["backlog_rank"] = index
        neighborhood["backlog_reason"] = (
            "ring_1_scope exceeds max file threshold; retained as large-zone backlog"
        )

    return {
        "format_version": 1,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source": {
            "projection_fiber_decision": projection_fiber_decision_from_payload(payload),
            "projection_fiber_semantic_bundle_count": (
                projection_fiber_semantic_bundle_count_from_payload(payload)
            ),
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
                family: int(family_totals.get(family, 0))
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
    return render_markdown(_queue_document(payload))


def _write_queue_outputs(
    *,
    queue: dict[str, Any],
    out_path: Path,
    markdown_out: Path | None = None,
) -> int:
    write_json(out_path, _queue_document(queue))
    if markdown_out is not None:
        write_markdown(markdown_out, _queue_document(queue))
    return 0


def _queue_document(payload: dict[str, Any]):
    return mapping_document(
        identity=ArtifactSourceRef(
            rel_path="<synthetic>",
            qualname="hotspot_neighborhood_queue",
        ),
        title="Hotspot Neighborhood Queue",
        payload=payload,
    )


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


def run_from_inputs(
    *,
    violations_by_rule: dict[str, list[dict[str, Any]]],
    policy_check_result_path: Path | None = None,
    out_path: Path,
    markdown_out: Path | None = None,
    config: QueueConfig | None = None,
) -> int:
    payload: dict[str, Any] = {
        "format_version": 1,
        "violations": violations_by_rule,
    }
    if policy_check_result_path is not None:
        projection_fiber_semantics = _load_projection_fiber_semantics_from_policy_check_result(
            artifact_path=policy_check_result_path,
        )
        if projection_fiber_semantics is not None:
            payload["projection_fiber_semantics"] = projection_fiber_semantics
    return run_from_payload(
        payload=payload,
        out_path=out_path,
        markdown_out=markdown_out,
        config=config,
    )


def _load_projection_fiber_semantics_from_policy_check_result(
    *,
    artifact_path: Path,
) -> dict[str, Any] | None:
    loaded = policy_result_schema.load_policy_result(artifact_path)
    if loaded is None or str(loaded.get("rule_id", "") or "").strip() != "policy_check":
        raise RuntimeError(
            "required child-owned policy result artifact missing before hotspot queue invocation: "
            f"rule_id=policy_check artifact={artifact_path}"
        )
    raw_semantics = loaded.get("projection_fiber_semantics")
    match raw_semantics:
        case dict() as semantics_mapping if semantics_mapping:
            return dict(semantics_mapping)
        case _:
            return None


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
