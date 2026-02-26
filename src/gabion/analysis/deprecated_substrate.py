# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from collections.abc import Mapping, Sequence


class DeprecatedLifecycleState(StrEnum):
    ACTIVE = "active"
    BLOCKED = "blocked"
    RESOLVED = "resolved"


@dataclass(frozen=True)
class DeprecatedBlocker:
    blocker_id: str
    kind: str
    summary: str
    lifecycle: DeprecatedLifecycleState = DeprecatedLifecycleState.BLOCKED
    depends_on: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "DeprecatedBlocker":
        blocker_id = str(payload.get("blocker_id", "") or "").strip()
        kind = str(payload.get("kind", "") or "").strip()
        summary = str(payload.get("summary", "") or "").strip()
        lifecycle = DeprecatedLifecycleState(str(payload.get("lifecycle", "blocked") or "blocked"))
        depends_on_raw = payload.get("depends_on", ())
        depends_on: tuple[str, ...]
        match depends_on_raw:
            case str() | bytes():
                depends_on = ()
            case Sequence() as depends_on_sequence:
                depends_on = tuple(
                    str(item).strip() for item in depends_on_sequence if str(item).strip()
                )
            case _:
                depends_on = ()
        if not blocker_id or not kind or not summary:
            raise ValueError("deprecated blocker requires blocker_id, kind, and summary")
        return cls(
            blocker_id=blocker_id,
            kind=kind,
            summary=summary,
            lifecycle=lifecycle,
            depends_on=depends_on,
        )


@dataclass(frozen=True)
class DeprecatedFiber:
    fiber_id: str
    canonical_aspf_path: tuple[str, ...]
    lifecycle: DeprecatedLifecycleState
    blocker_payload: tuple[DeprecatedBlocker, ...] = ()
    resolution_metadata: object = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "DeprecatedFiber":
        path_raw = payload.get("canonical_aspf_path", ())
        match path_raw:
            case str() | bytes():
                raise ValueError("canonical_aspf_path must be a sequence")
            case Sequence() as path_sequence:
                path = tuple(str(item).strip() for item in path_sequence if str(item).strip())
            case _:
                raise ValueError("canonical_aspf_path must be a sequence")
        lifecycle = DeprecatedLifecycleState(str(payload.get("lifecycle", "active") or "active"))
        blockers_raw = payload.get("blocker_payload", ())
        blockers: list[DeprecatedBlocker] = []
        match blockers_raw:
            case str() | bytes():
                blockers = blockers
            case Sequence() as blockers_sequence:
                for item in blockers_sequence:
                    match item:
                        case Mapping() as blocker_payload:
                            blockers.append(DeprecatedBlocker.from_payload(blocker_payload))
                        case _:
                            pass
            case _:
                blockers = blockers
        resolution_metadata_payload = payload.get("resolution_metadata")
        match resolution_metadata_payload:
            case Mapping() as resolution_metadata:
                metadata = resolution_metadata
            case _:
                metadata = None
        return deprecated(
            canonical_aspf_path=path,
            blockers=tuple(blockers),
            lifecycle=lifecycle,
            fiber_id=str(payload.get("fiber_id", "") or "").strip() or None,
            resolution_metadata=metadata,
        )


def deprecated(
    *,
    canonical_aspf_path: Sequence[str],
    blockers: Sequence[DeprecatedBlocker],
    lifecycle: DeprecatedLifecycleState = DeprecatedLifecycleState.ACTIVE,
    fiber_id: object = None,
    resolution_metadata: object = None,
) -> DeprecatedFiber:
    path = tuple(str(item).strip() for item in canonical_aspf_path if str(item).strip())
    if not path:
        raise ValueError("deprecated() requires canonical ASPF path identity")
    blocker_payload = tuple(blockers)
    if lifecycle is not DeprecatedLifecycleState.RESOLVED and not blocker_payload:
        raise ValueError("deprecated() requires blocker payload unless lifecycle is resolved")
    if lifecycle is DeprecatedLifecycleState.RESOLVED and resolution_metadata is None:
        raise ValueError("resolved deprecated fiber requires resolution metadata")
    if fiber_id is None:
        fiber_id = "aspf:" + "/".join(path)
    resolved_metadata: object = None
    match resolution_metadata:
        case Mapping() as metadata_mapping:
            resolved_metadata = metadata_mapping
        case _:
            resolved_metadata = None
    return DeprecatedFiber(
        fiber_id=str(fiber_id),
        canonical_aspf_path=path,
        lifecycle=lifecycle,
        blocker_payload=blocker_payload,
        resolution_metadata=resolved_metadata,
    )


@dataclass(frozen=True)
class PerfSample:
    stack: tuple[str, ...]
    weight: int = 1


@dataclass(frozen=True)
class DeprecatedExtractionArtifacts:
    perf_fiber_groups: tuple[dict[str, object], ...]
    fiber_group_rankings: tuple[dict[str, object], ...]
    blocker_dag: dict[str, tuple[dict[str, object], ...]]
    informational_signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeprecatedGatingResult:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


def ingest_perf_samples(samples: Sequence[Mapping[str, object]]) -> tuple[PerfSample, ...]:
    parsed: list[PerfSample] = []
    for sample in samples:
        stack_raw = sample.get("stack", ())
        match stack_raw:
            case str() | bytes():
                pass
            case Sequence() as stack_sequence:
                stack = tuple(str(item).strip() for item in stack_sequence if str(item).strip())
                if stack:
                    weight_raw = sample.get("weight", 1)
                    weight = int(weight_raw) if type(weight_raw) is int and weight_raw > 0 else 1
                    parsed.append(PerfSample(stack=stack, weight=weight))
            case _:
                pass
    return tuple(parsed)


def project_stack_to_aspf_fiber_groups(samples: Sequence[PerfSample]) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, ...], int] = {}
    for sample in samples:
        grouped[sample.stack] = grouped.get(sample.stack, 0) + sample.weight
    ordered = sorted(grouped.items(), key=lambda item: (-item[1], item[0]))
    return tuple(
        {
            "fiber_group": "::".join(stack),
            "canonical_aspf_path": list(stack),
            "weight": weight,
        }
        for stack, weight in ordered
    )


def rank_fiber_groups(groups: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    sortable: list[tuple[str, int]] = []
    for group in groups:
        fiber_group = str(group.get("fiber_group", "") or "").strip()
        if not fiber_group:
            continue
        weight = int(group.get("weight", 0) or 0)
        sortable.append((fiber_group, weight))
    ordered = sorted(sortable, key=lambda item: (-item[1], item[0]))
    return tuple(
        {
            "rank": idx + 1,
            "fiber_group": fiber_group,
            "weight": weight,
        }
        for idx, (fiber_group, weight) in enumerate(ordered)
    )


def blocker_dag_for_fibers(fibers: Sequence[DeprecatedFiber]) -> dict[str, tuple[dict[str, object], ...]]:
    nodes: dict[str, dict[str, object]] = {}
    edges: set[tuple[str, str]] = set()
    for fiber in fibers:
        for blocker in fiber.blocker_payload:
            nodes.setdefault(
                blocker.blocker_id,
                {
                    "blocker_id": blocker.blocker_id,
                    "kind": blocker.kind,
                    "summary": blocker.summary,
                    "lifecycle": blocker.lifecycle.value,
                },
            )
            for dep in blocker.depends_on:
                edges.add((blocker.blocker_id, dep))
    ordered_nodes = tuple(nodes[node_id] for node_id in sorted(nodes))
    ordered_edges = tuple(
        {"from": src, "to": dst}
        for src, dst in sorted(edges)
    )
    return {"nodes": ordered_nodes, "edges": ordered_edges}


def build_deprecated_extraction_artifacts(
    *,
    perf_samples: Sequence[PerfSample],
    deprecated_fibers: Sequence[DeprecatedFiber],
    branch_coverage_previous: object = None,
    branch_coverage_current: object = None,
) -> DeprecatedExtractionArtifacts:
    perf_groups = project_stack_to_aspf_fiber_groups(perf_samples)
    rankings = rank_fiber_groups(perf_groups)
    blocker_dag = blocker_dag_for_fibers(deprecated_fibers)
    previous_coverage: Mapping[str, float] = {}
    current_coverage: Mapping[str, float] = {}
    match branch_coverage_previous:
        case Mapping() as previous_payload:
            previous_coverage = previous_payload
        case _:
            pass
    match branch_coverage_current:
        case Mapping() as current_payload:
            current_coverage = current_payload
        case _:
            pass
    info = classify_branch_coverage_loss(
        previous=previous_coverage,
        current=current_coverage,
    )
    return DeprecatedExtractionArtifacts(
        perf_fiber_groups=perf_groups,
        fiber_group_rankings=rankings,
        blocker_dag=blocker_dag,
        informational_signals=tuple(info),
    )


def detect_report_section_extinction(*, previous_sections: Sequence[str], current_sections: Sequence[str]) -> tuple[str, ...]:
    previous = {section.strip() for section in previous_sections if section.strip()}
    current = {section.strip() for section in current_sections if section.strip()}
    return tuple(sorted(previous - current))


def check_semantic_fiber_continuity(*, previous_fibers: Sequence[DeprecatedFiber], current_fibers: Sequence[DeprecatedFiber]) -> tuple[str, ...]:
    previous_ids = {fiber.fiber_id for fiber in previous_fibers}
    current_ids = {fiber.fiber_id for fiber in current_fibers}
    return tuple(sorted(previous_ids - current_ids))


def classify_branch_coverage_loss(*, previous: Mapping[str, float], current: Mapping[str, float]) -> list[str]:
    info: list[str] = []
    for key in sorted(previous):
        before = float(previous.get(key, 0.0))
        after = float(current.get(key, before))
        if after < before:
            info.append(
                f"informational: branch coverage loss for {key} ({before:.3f} -> {after:.3f})"
            )
    return info


def enforce_non_erasability_policy(
    *,
    previous_fibers: Sequence[DeprecatedFiber],
    current_fibers: Sequence[DeprecatedFiber],
) -> DeprecatedGatingResult:
    missing = check_semantic_fiber_continuity(
        previous_fibers=previous_fibers,
        current_fibers=current_fibers,
    )
    errors: list[str] = []
    for missing_id in missing:
        prior = next(f for f in previous_fibers if f.fiber_id == missing_id)
        if not (
            prior.lifecycle is DeprecatedLifecycleState.RESOLVED
            and prior.resolution_metadata is not None
        ):
            errors.append(
                f"deprecated fiber erased without explicit resolution metadata: {missing_id}"
            )
    return DeprecatedGatingResult(errors=tuple(sorted(errors)))
