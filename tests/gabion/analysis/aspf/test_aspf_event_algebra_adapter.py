from __future__ import annotations

import pytest

from gabion.analysis.aspf.aspf_event_algebra_adapter import (
    adapt_aspf_replay_event,
    adapt_aspf_replay_event_or_raise,
)
from gabion.analysis.aspf.aspf_visitors import (
    AspfCofibrationEvent,
    AspfOneCellEvent,
    AspfRunBoundaryEvent,
    AspfSurfaceUpdateEvent,
    AspfTwoCellEvent,
)
from gabion.analysis.core.prime_identity_adapter import PrimeIdentityAdapter
from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.analysis.foundation.event_algebra import (
    CanonicalAdaptationKind,
    CanonicalEventAdaptationError,
    CanonicalRunContext,
    GlobalEventSequencer,
)
from gabion.analysis.foundation.identity_space import GlobalIdentitySpace


def _run_context(*, run_id: str = "run:aspf") -> CanonicalRunContext:
    return CanonicalRunContext(
        run_id=run_id,
        sequencer=GlobalEventSequencer(),
        identity_space=GlobalIdentitySpace(
            allocator=PrimeIdentityAdapter(registry=PrimeRegistry())
        ),
    )


@pytest.mark.parametrize(
    ("event", "expected_kind"),
    [
        (AspfOneCellEvent(index=0, payload={"kind": "resume_load"}), "one_cell"),
        (
            AspfTwoCellEvent(
                index=1,
                payload={
                    "witness_id": "w:1",
                    "left_representative": "rep:a",
                    "right_representative": "rep:b",
                },
            ),
            "two_cell",
        ),
        (
            AspfCofibrationEvent(
                index=2,
                payload={"canonical_identity_kind": "suite_site"},
            ),
            "cofibration",
        ),
        (
            AspfSurfaceUpdateEvent(
                surface="groups_by_path",
                representative="rep:a",
            ),
            "surface_update",
        ),
        (
            AspfRunBoundaryEvent(
                boundary="equivalence_surface_row",
                payload={"surface": "groups_by_path", "classification": "non_drift"},
            ),
            "run_boundary",
        ),
    ],
)
def test_aspf_event_adapter_maps_all_replay_event_kinds(
    event: object,
    expected_kind: str,
) -> None:
    envelope = adapt_aspf_replay_event_or_raise(
        event=event,  # type: ignore[arg-type]
        run_context=_run_context(),
    )
    assert envelope.source == "aspf.trace_replay"
    assert envelope.kind == expected_kind
    assert envelope.identity_projection.basis_path.atoms


def test_aspf_event_adapter_identity_projection_is_deterministic() -> None:
    event = AspfTwoCellEvent(
        index=3,
        payload={
            "witness_id": "w:det",
            "left_representative": "rep:left",
            "right_representative": "rep:right",
        },
    )
    envelope_a = adapt_aspf_replay_event_or_raise(event=event, run_context=_run_context())
    envelope_b = adapt_aspf_replay_event_or_raise(event=event, run_context=_run_context())
    assert envelope_a.identity_projection == envelope_b.identity_projection


def test_aspf_event_adapter_hard_fails_on_identity_derivation_gap() -> None:
    bad_event = AspfSurfaceUpdateEvent(surface="", representative="rep:a")
    decision = adapt_aspf_replay_event(event=bad_event, run_context=_run_context())
    assert decision.kind == CanonicalAdaptationKind.REJECTED
    with pytest.raises(CanonicalEventAdaptationError):
        adapt_aspf_replay_event_or_raise(event=bad_event, run_context=_run_context())
