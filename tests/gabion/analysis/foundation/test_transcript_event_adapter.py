from __future__ import annotations

import pytest

from gabion.analysis.core.prime_identity_adapter import PrimeIdentityAdapter
from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.analysis.foundation.event_algebra import (
    CanonicalAdaptationKind,
    CanonicalEventAdaptationError,
    CanonicalRunContext,
    GlobalEventSequencer,
)
from gabion.analysis.foundation.identity_space import GlobalIdentitySpace
from tests.gabion.analysis.foundation.transcript_event_fixtures import (
    ComponentSealed,
    EdgeFormed,
    NameInterned,
    NodeDiscovered,
    StreamTerminated,
    adapt_transcript_fixture_event,
    adapt_transcript_fixture_event_or_raise,
)


def _run_context(*, run_id: str = "run:transcript") -> CanonicalRunContext:
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
        (
            NodeDiscovered(node_id="N1", module_path="pkg.core", label="node"),
            "node_discovered",
        ),
        (
            EdgeFormed(src_node_id="N1", dst_node_id="N2", relation="imports"),
            "edge_formed",
        ),
        (
            ComponentSealed(component_id="C1", members=("N1", "N2")),
            "component_sealed",
        ),
        (
            StreamTerminated(reason="done", total_events=10),
            "stream_terminated",
        ),
        (
            NameInterned(namespace="symbol", token="N1", atom_id=2),
            "name_interned",
        ),
    ],
)
def test_transcript_fixture_adapter_maps_event_types(
    event: object,
    expected_kind: str,
) -> None:
    envelope = adapt_transcript_fixture_event_or_raise(
        event=event,  # type: ignore[arg-type]
        run_context=_run_context(),
    )
    assert envelope.source == "transcript.scout"
    assert envelope.phase == "scout"
    assert envelope.kind == expected_kind
    assert envelope.identity_projection.basis_path.atoms


def test_transcript_fixture_adapter_enforces_identity_and_sequence() -> None:
    run_context = _run_context(run_id="run:transcript:seq")
    first = adapt_transcript_fixture_event_or_raise(
        event=NodeDiscovered(node_id="N1", module_path="pkg.a"),
        run_context=run_context,
    )
    second = adapt_transcript_fixture_event_or_raise(
        event=EdgeFormed(src_node_id="N1", dst_node_id="N2", relation="calls"),
        run_context=run_context,
    )
    assert first.sequence == 1
    assert first.event_id == "run:transcript:seq:1"
    assert second.sequence == 2
    assert second.event_id == "run:transcript:seq:2"
    assert first.identity_projection != second.identity_projection


def test_transcript_fixture_adapter_rejects_missing_identity_components() -> None:
    bad_event = NameInterned(namespace="symbol", token="", atom_id=0)
    decision = adapt_transcript_fixture_event(event=bad_event, run_context=_run_context())
    assert decision.kind == CanonicalAdaptationKind.REJECTED
    with pytest.raises(CanonicalEventAdaptationError):
        adapt_transcript_fixture_event_or_raise(
            event=bad_event,
            run_context=_run_context(),
        )
