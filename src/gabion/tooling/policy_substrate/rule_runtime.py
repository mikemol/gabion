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
from gabion.tooling.policy_substrate.policy_event_kind import (
    PolicyEventKind,
    coerce_policy_event_kind,
    policy_event_kind_segments,
    policy_event_kind_scalar,
)
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
    fiber_trace: list[FiberTraceEvent]
    applicability_bounds: FiberApplicabilityBounds
    counterfactual_boundary: FiberCounterfactualBoundary
    fiber_id: str
    taint_interval_id: str
    condition_overlap_id: str


class MissingTaintIntervalError(ValueError):
    def __init__(self, *, fiber_id: str) -> None:
        self.fiber_id = fiber_id
        super().__init__("missing_taint_interval")


def new_run_context(*, rule_name: str) -> CanonicalRunContext:
    return CanonicalRunContext(
        run_id=rule_name,
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
    intro_kind: PolicyEventKind | str,
    condition_kind: PolicyEventKind | str,
    erase_kind: PolicyEventKind | str,
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
            line,
            column,
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
        intro_kind_value = coerce_policy_event_kind(kind=intro_kind)
        condition_kind_value = coerce_policy_event_kind(kind=condition_kind)
        erase_kind_value = coerce_policy_event_kind(kind=erase_kind)
        events = list(
            run_projection_lenses(
                site=site,
                specs=(
                    _intro_spec(event_kind=intro_kind_value),
                    _condition_spec(event_kind=condition_kind_value),
                    _erase_spec(event_kind=erase_kind_value),
                ),
            )
        )
        intervals = list(build_taint_intervals(events=events))
        overlaps = list(
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

        fiber_trace = list(_iter_fiber_trace(events))

        erase_ordinal = interval.end_ordinal
        bounds = FiberApplicabilityBounds(
            current_boundary_before_ordinal=erase_ordinal,
            violation_applies_when_boundary_before_ordinal_gt=max(1, erase_ordinal - 1),
            violation_clears_when_boundary_before_ordinal_lte=max(1, erase_ordinal - 1),
            boundary_domain_max_before_ordinal=erase_ordinal,
            core_entry_before_ordinal=None,
        )
        intro_event = interval.intro_event
        counterfactual = FiberCounterfactualBoundary(
            suggested_boundary_before_ordinal=intro_event.ordinal,
            boundary_event_kind=policy_event_kind_scalar(kind=intro_event.event_kind),
            boundary_line=intro_event.line,
            boundary_column=intro_event.column,
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
        line=seed.line,
        column=seed.column,
        node_kind="module_failure",
        input_slot="module_failure",
        taint_class="module_failure",
        intro_kind="syntax:block_enter",
        condition_kind="syntax:module_failure",
        erase_kind="syntax:boundary",
        rationale=rationale,
    )


def cst_failure_seeds(*, union_view: ASPFUnionView):
    return map(_cst_failure_seed, union_view.cst_failures)


def _cst_failure_seed(event: CSTParseFailureEvent) -> ScanFailureSeed:
    return ScanFailureSeed(
        path=event.rel_path,
        line=event.line,
        column=event.column,
        kind=event.kind,
        detail=event.message,
    )


def _intro_spec(*, event_kind: PolicyEventKind) -> ProjectionLensSpec:
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


def _condition_spec(*, event_kind: PolicyEventKind) -> ProjectionLensSpec:
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


def _erase_spec(*, event_kind: PolicyEventKind) -> ProjectionLensSpec:
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
        raise MissingTaintIntervalError(fiber_id=fiber_id) from exc


def _first_overlap(*, overlaps: Iterable[object], interval_id: str) -> str:
    try:
        return next(
            map(
                _condition_overlap_id,
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
    _ = run_context
    _ = line
    _ = column
    return _stable_hash(
        "flow",
        rule_name,
        rel_path,
        qualname,
        input_slot,
        taint_class,
        site_id,
    )


def _stable_hash(*parts: object) -> str:
    return reduce(_digest_update, parts, hashlib.sha256()).hexdigest()


def _digest_update(digest: object, part: object):
    digest.update(_hash_part_bytes(part))
    digest.update(b"\x00")
    return digest


def iter_condition_events(events: Iterable[LensEvent]):
    return filter(lambda event: event.action == "condition", events)


def _iter_fiber_trace(events: Iterable[LensEvent]):
    for event in events:
        yield FiberTraceEvent(
            ordinal=event.ordinal,
            line=event.line,
            column=event.column,
            event_kind=policy_event_kind_scalar(kind=event.event_kind),
            normalization_class=event.taint_class,
            input_slot=event.input_slot,
            phase_hint=event.event_phase,
            pre_core=True,
        )


def _condition_overlap_id(overlap: object) -> str:
    value = getattr(overlap, "condition_overlap_id", "")
    match value:
        case str() as text:
            return text
        case _:
            return ""


def _hash_part_bytes(value: object) -> bytes:
    match value:
        case PolicyEventKind() as event_kind:
            return b"\x1f".join(
                map(lambda segment: segment.encode("utf-8"), policy_event_kind_segments(kind=event_kind))
            )
        case bool() as flag:
            return b"1" if flag else b"0"
        case int() as integer:
            return _int_bytes(integer)
        case str() as text:
            return text.encode("utf-8")
        case bytes() as raw:
            return raw
        case _:
            return b"<unsupported>"


def _int_bytes(value: int) -> bytes:
    magnitude = abs(value)
    width = max(1, (magnitude.bit_length() + 7) // 8)
    sign = b"-" if value < 0 else b"+"
    return sign + magnitude.to_bytes(width, byteorder="big", signed=False)


__all__ = [
    "SubstrateDecoration",
    "cst_failure_seeds",
    "decorate_failure",
    "decorate_site",
    "new_run_context",
]
