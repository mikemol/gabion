# gabion:decision_protocol_module
from __future__ import annotations

import hashlib
from functools import reduce
from dataclasses import dataclass
from typing import Iterable

from gabion.analysis.core.prime_identity_adapter import PrimeIdentityAdapter
from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.analysis.foundation.event_algebra import (
    CanonicalRunContext,
    GlobalEventSequencer,
    derive_identity_projection_from_tokens,
)
from gabion.analysis.foundation.identity_space import GlobalIdentitySpace
from gabion.tooling.policy_rules.fiber_diagnostics import (
    FiberApplicabilityBounds,
    FiberCounterfactualBoundary,
    FiberTraceEvent,
)
from gabion.tooling.policy_substrate.aspf_union_view import (
    ASPFUnionView,
    CSTParseFailureEvent,
)
from gabion.tooling.policy_substrate.overlap_eval import evaluate_condition_overlaps
from gabion.tooling.policy_substrate.projection_lens import (
    LensEvent,
    LensSite,
    ProjectionLensSpec,
    run_projection_lenses,
)
from gabion.tooling.policy_substrate.site_identity import canonical_site_identity
from gabion.tooling.policy_substrate.taint_intervals import build_taint_intervals
from gabion.tooling.runtime.deadline_runtime import DeadlineBudget, deadline_scope_from_ticks
from gabion.tooling.runtime.policy_scan_batch import ScanFailureSeed

_DEFAULT_POLICY_TIMEOUT_BUDGET = DeadlineBudget(ticks=120_000, tick_ns=1_000_000)


@dataclass(frozen=True)
class SubstrateDecoration:
    flow_identity: str
    fiber_trace: tuple[FiberTraceEvent, ...]
    applicability_bounds: FiberApplicabilityBounds
    counterfactual_boundary: FiberCounterfactualBoundary
    fiber_id: str
    taint_interval_id: str
    condition_overlap_id: str


def new_run_context(*, rule_name: str) -> CanonicalRunContext:
    return CanonicalRunContext(
        run_id=f"policy:{rule_name}",
        sequencer=GlobalEventSequencer(),
        identity_space=GlobalIdentitySpace(
            allocator=PrimeIdentityAdapter(registry=PrimeRegistry())
        ),
    )


def decorate_site(
    *,
    run_context: CanonicalRunContext,
    rule_name: str,
    rel_path: str,
    qualname: str,
    line: int,
    column: int,
    node_kind: str,
    input_slot: str,
    taint_class: str,
    intro_kind: str,
    condition_kind: str,
    erase_kind: str,
    rationale: str,
    surface: str = "pyast",
) -> SubstrateDecoration:
    with deadline_scope_from_ticks(_DEFAULT_POLICY_TIMEOUT_BUDGET):
        site_id = canonical_site_identity(
            rel_path=rel_path,
            qualname=qualname,
            line=line,
            column=column,
            node_kind=node_kind,
            surface=surface,
        )
        fiber_id = _stable_hash(
            "fiber",
            rule_name,
            rel_path,
            qualname,
            str(line),
            str(column),
            input_slot,
            taint_class,
        )
        flow_identity = _derive_flow_identity(
            run_context=run_context,
            rule_name=rule_name,
            rel_path=rel_path,
            qualname=qualname,
            line=line,
            column=column,
            input_slot=input_slot,
            taint_class=taint_class,
            site_id=site_id,
        )
        site = LensSite(
            site_id=site_id,
            path=rel_path,
            qualname=qualname,
            line=line,
            column=column,
            node_kind=node_kind,
            surface=surface,
            fiber_id=fiber_id,
            taint_class=taint_class,
            input_slot=input_slot,
        )
        events = tuple(
            run_projection_lenses(
                site=site,
                specs=(
                    _intro_spec(event_kind=intro_kind),
                    _condition_spec(event_kind=condition_kind),
                    _erase_spec(event_kind=erase_kind),
                ),
            )
        )
        intervals = tuple(build_taint_intervals(events=events))
        overlaps = tuple(
            evaluate_condition_overlaps(
                intervals=intervals,
                condition_events=iter_condition_events(events),
            )
        )
        interval = _first_interval(intervals=intervals, fiber_id=fiber_id)
        overlap = _first_overlap(
            overlaps=overlaps,
            interval_id=interval.interval_id,
        )

        fiber_trace = tuple(_iter_fiber_trace(events))

        erase_ordinal = int(interval.end_ordinal)
        bounds = FiberApplicabilityBounds(
            current_boundary_before_ordinal=erase_ordinal,
            violation_applies_when_boundary_before_ordinal_gt=max(1, erase_ordinal - 1),
            violation_clears_when_boundary_before_ordinal_lte=max(1, erase_ordinal - 1),
            boundary_domain_max_before_ordinal=erase_ordinal,
            core_entry_before_ordinal=None,
        )
        intro_event = interval.intro_event
        counterfactual = FiberCounterfactualBoundary(
            suggested_boundary_before_ordinal=int(intro_event.ordinal),
            boundary_event_kind=str(intro_event.event_kind),
            boundary_line=int(intro_event.line),
            boundary_column=int(intro_event.column),
            eliminates_violation_without_other_changes=True,
            preserves_prior_normalization=True,
            rationale=rationale,
        )

        return SubstrateDecoration(
            flow_identity=flow_identity,
            fiber_trace=fiber_trace,
            applicability_bounds=bounds,
            counterfactual_boundary=counterfactual,
            fiber_id=fiber_id,
            taint_interval_id=interval.interval_id,
            condition_overlap_id=overlap,
        )


def decorate_failure(
    *,
    run_context: CanonicalRunContext,
    rule_name: str,
    seed: ScanFailureSeed,
    rationale: str,
) -> SubstrateDecoration:
    return decorate_site(
        run_context=run_context,
        rule_name=rule_name,
        rel_path=seed.path,
        qualname="<module>",
        line=int(seed.line),
        column=int(seed.column),
        node_kind="module_failure",
        input_slot="module_failure",
        taint_class="module_failure",
        intro_kind=f"syntax:block_enter:{seed.kind}",
        condition_kind=f"syntax:{seed.kind}",
        erase_kind=f"syntax:boundary:{seed.kind}",
        rationale=rationale,
    )


def cst_failure_seeds(*, union_view: ASPFUnionView):
    return map(_cst_failure_seed, union_view.cst_failures)


def _cst_failure_seed(event: CSTParseFailureEvent) -> ScanFailureSeed:
    return ScanFailureSeed(
        path=event.rel_path,
        line=int(event.line),
        column=int(event.column),
        kind=event.kind,
        detail=event.message,
    )


def _intro_spec(*, event_kind: str) -> ProjectionLensSpec:
    return ProjectionLensSpec(
        name="taint_intro_spec",
        project=lambda site: (
            LensEvent(
                ordinal=1,
                site_id=site.site_id,
                path=site.path,
                qualname=site.qualname,
                line=site.line,
                column=site.column,
                node_kind=site.node_kind,
                surface=site.surface,
                fiber_id=site.fiber_id,
                event_kind=event_kind,
                event_phase="taint_intro",
                input_slot=site.input_slot,
                taint_class=site.taint_class,
                action="taint_intro",
            ),
        ),
    )


def _condition_spec(*, event_kind: str) -> ProjectionLensSpec:
    return ProjectionLensSpec(
        name="condition_spec",
        project=lambda site: (
            LensEvent(
                ordinal=2,
                site_id=site.site_id,
                path=site.path,
                qualname=site.qualname,
                line=site.line,
                column=site.column,
                node_kind=site.node_kind,
                surface=site.surface,
                fiber_id=site.fiber_id,
                event_kind=event_kind,
                event_phase="condition",
                input_slot=site.input_slot,
                taint_class=site.taint_class,
                action="condition",
            ),
        ),
    )


def _erase_spec(*, event_kind: str) -> ProjectionLensSpec:
    return ProjectionLensSpec(
        name="taint_erase_spec",
        project=lambda site: (
            LensEvent(
                ordinal=3,
                site_id=site.site_id,
                path=site.path,
                qualname=site.qualname,
                line=site.line,
                column=site.column,
                node_kind=site.node_kind,
                surface=site.surface,
                fiber_id=site.fiber_id,
                event_kind=event_kind,
                event_phase="taint_erase",
                input_slot=site.input_slot,
                taint_class=site.taint_class,
                action="taint_erase",
            ),
        ),
    )


def _first_interval(*, intervals: Iterable[object], fiber_id: str):
    try:
        return next(
            filter(
                lambda interval: getattr(interval, "fiber_id", "") == fiber_id,
                intervals,
            )
        )
    except StopIteration as exc:
        raise ValueError(f"missing taint interval for fiber_id={fiber_id}") from exc


def _first_overlap(*, overlaps: Iterable[object], interval_id: str) -> str:
    try:
        return next(
            map(
                lambda overlap: str(getattr(overlap, "condition_overlap_id", "")),
                filter(
                    lambda overlap: getattr(overlap, "taint_interval_id", "")
                    == interval_id,
                    overlaps,
                ),
            )
        )
    except StopIteration:
        return _stable_hash(interval_id, "no_overlap")


def _derive_flow_identity(
    *,
    run_context: CanonicalRunContext,
    rule_name: str,
    rel_path: str,
    qualname: str,
    line: int,
    column: int,
    input_slot: str,
    taint_class: str,
    site_id: str,
) -> str:
    projection = derive_identity_projection_from_tokens(
        run_context=run_context,
        tokens=(
            f"fiber.{rule_name}",
            f"path:{rel_path}",
            f"qualname:{qualname}",
            f"line:{line}",
            f"column:{column}",
            f"slot:{input_slot}",
            f"taint:{taint_class}",
            f"site:{site_id}",
        ),
    )
    atoms = ".".join(map(str, projection.basis_path.atoms))
    return f"{projection.basis_path.namespace}:{atoms}"


def _stable_hash(*parts: str) -> str:
    return reduce(_digest_update, parts, hashlib.sha256()).hexdigest()


def _digest_update(digest: object, part: str):
    digest.update(part.encode("utf-8"))
    digest.update(b"\x00")
    return digest


def iter_condition_events(events: tuple[LensEvent, ...]):
    return filter(lambda event: event.action == "condition", events)


def _iter_fiber_trace(events: tuple[LensEvent, ...]):
    for event in events:
        yield FiberTraceEvent(
            ordinal=int(event.ordinal),
            line=int(event.line),
            column=int(event.column),
            event_kind=str(event.event_kind),
            normalization_class=str(event.taint_class),
            input_slot=str(event.input_slot),
            phase_hint=str(event.event_phase),
            pre_core=True,
        )


__all__ = [
    "SubstrateDecoration",
    "cst_failure_seeds",
    "decorate_failure",
    "decorate_site",
    "new_run_context",
]
