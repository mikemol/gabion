from __future__ import annotations

import pytest

from gabion.analysis.aspf.aspf_event_algebra_adapter import adapt_aspf_replay_event_or_raise
from gabion.analysis.aspf.aspf_visitors import AspfOneCellEvent
from gabion.analysis.core.prime_identity_adapter import PrimeIdentityAdapter
from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.analysis.foundation.event_algebra import (
    CanonicalEventAdaptationError,
    CanonicalRunContext,
    GlobalEventSequencer,
    build_canonical_event_envelope,
    derive_identity_projection_from_tokens,
    envelope_from_decision_or_raise,
    canonical_adaptation_rejected,
)
from gabion.analysis.foundation.identity_space import GlobalIdentitySpace
from tests.gabion.analysis.foundation.transcript_event_fixtures import (
    NodeDiscovered,
    adapt_transcript_fixture_event_or_raise,
)


def _run_context(*, run_id: str = "run:test") -> CanonicalRunContext:
    identity_space = GlobalIdentitySpace(
        allocator=PrimeIdentityAdapter(registry=PrimeRegistry())
    )
    return CanonicalRunContext(
        run_id=run_id,
        sequencer=GlobalEventSequencer(),
        identity_space=identity_space,
    )


# gabion:behavior primary=desired
def test_canonical_event_envelope_constructs_with_required_fields() -> None:
    run_context = _run_context()
    projection = derive_identity_projection_from_tokens(
        run_context=run_context,
        tokens=("test_source", "phase", "kind", "anchor:1"),
    )
    envelope = build_canonical_event_envelope(
        run_context=run_context,
        source="test_source",
        phase="phase",
        kind="kind",
        identity_projection=projection,
        payload={"ok": True},
        causal_refs=("r:1",),
    )
    assert envelope.schema_version == 1
    assert envelope.sequence == 1
    assert envelope.event_id == "run:test:1"
    assert envelope.identity_projection.basis_path.atoms
    assert envelope.payload == {"ok": True}
    assert envelope.causal_refs == ("r:1",)


# gabion:behavior primary=desired
def test_global_sequencer_is_monotonic_across_mixed_source_adaptation() -> None:
    run_context = _run_context(run_id="run:mixed")
    transcript_event = adapt_transcript_fixture_event_or_raise(
        event=NodeDiscovered(node_id="n1", module_path="pkg.mod", label="N"),
        run_context=run_context,
    )
    aspf_event = adapt_aspf_replay_event_or_raise(
        event=AspfOneCellEvent(index=0, payload={"kind": "resume_load"}),
        run_context=run_context,
    )
    another_transcript_event = adapt_transcript_fixture_event_or_raise(
        event=NodeDiscovered(node_id="n2", module_path="pkg.mod", label="M"),
        run_context=run_context,
    )
    assert transcript_event.sequence == 1
    assert aspf_event.sequence == 2
    assert another_transcript_event.sequence == 3


# gabion:behavior primary=verboten facets=missing
def test_missing_identity_projection_rejects_adaptation() -> None:
    run_context = _run_context()
    with pytest.raises(CanonicalEventAdaptationError):
        derive_identity_projection_from_tokens(run_context=run_context, tokens=())
    with pytest.raises(CanonicalEventAdaptationError):
        envelope_from_decision_or_raise(canonical_adaptation_rejected("missing identity"))
