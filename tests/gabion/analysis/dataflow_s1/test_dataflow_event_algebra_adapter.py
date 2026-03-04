from __future__ import annotations

import pytest

from gabion.analysis.core.prime_identity_adapter import PrimeIdentityAdapter
from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.analysis.dataflow.engine.dataflow_event_algebra_adapter import (
    adapt_dataflow_collection_progress_event,
    adapt_dataflow_collection_progress_event_or_raise,
    adapt_dataflow_phase_progress_event,
    adapt_dataflow_phase_progress_event_or_raise,
)
from gabion.analysis.foundation.event_algebra import (
    CanonicalAdaptationKind,
    CanonicalEventAdaptationError,
    CanonicalRunContext,
    GlobalEventSequencer,
)
from gabion.analysis.foundation.identity_space import GlobalIdentitySpace


def _run_context(*, run_id: str = "run:dataflow") -> CanonicalRunContext:
    return CanonicalRunContext(
        run_id=run_id,
        sequencer=GlobalEventSequencer(),
        identity_space=GlobalIdentitySpace(
            allocator=PrimeIdentityAdapter(registry=PrimeRegistry())
        ),
    )


@pytest.mark.parametrize("phase", ["collection", "forest", "edge", "post"])
def test_dataflow_phase_progress_adapter_maps_phases(phase: str) -> None:
    envelope = adapt_dataflow_phase_progress_event_or_raise(
        phase_progress={
            "phase": phase,
            "event_kind": "progress",
            "progress_marker": f"{phase}:tick",
            "work_done": 1,
            "work_total": 2,
        },
        run_context=_run_context(),
    )
    assert envelope.source == "dataflow.phase_progress"
    assert envelope.phase == phase
    assert envelope.kind == "progress"
    assert envelope.identity_projection.basis_path.atoms


def test_dataflow_phase_progress_adapter_maps_transition_payloads() -> None:
    envelope = adapt_dataflow_phase_progress_event_or_raise(
        phase_progress={
            "phase": "post",
            "progress_marker": "ignored",
            "progress_transition_v2": {
                "phase": "post",
                "event_kind": "terminal",
                "root": {
                    "identity": "post_root",
                    "unit": "post_tasks",
                    "done": 6,
                    "total": 6,
                },
                "active_path": ["post_root"],
            },
            "phase_progress_v2": {
                "primary_unit": "post_tasks",
                "primary_done": 6,
                "primary_total": 6,
            },
        },
        run_context=_run_context(),
    )
    assert envelope.kind == "terminal"
    assert envelope.phase == "post"


def test_dataflow_phase_progress_adapter_accepts_integer_anchor_encoder_hook() -> None:
    run_context = _run_context()
    default_envelope = adapt_dataflow_phase_progress_event_or_raise(
        phase_progress={
            "phase": "post",
            "event_kind": "progress",
            "event_seq": 7,
            "progress_marker": "fingerprint:warnings",
        },
        run_context=run_context,
    )
    encoded_envelope = adapt_dataflow_phase_progress_event_or_raise(
        phase_progress={
            "phase": "post",
            "event_kind": "progress",
            "event_seq": 7,
            "progress_marker": "fingerprint:warnings",
        },
        run_context=run_context,
        integer_anchor_encoder=lambda key, value: (f"{key}@{value}", "bit:0"),
    )
    assert default_envelope.identity_projection.basis_path.namespace == (
        encoded_envelope.identity_projection.basis_path.namespace
    )
    assert default_envelope.identity_projection.basis_path.atoms != (
        encoded_envelope.identity_projection.basis_path.atoms
    )


def test_dataflow_phase_progress_adapter_rejects_empty_integer_anchor_token() -> None:
    decision = adapt_dataflow_phase_progress_event(
        phase_progress={
            "phase": "post",
            "event_kind": "progress",
            "event_seq": 3,
        },
        run_context=_run_context(),
        integer_anchor_encoder=lambda _key, _value: (),
    )
    assert decision.kind == CanonicalAdaptationKind.REJECTED


def test_dataflow_collection_adapter_maps_collection_and_index_payloads() -> None:
    collection_envelope = adapt_dataflow_collection_progress_event_or_raise(
        collection_progress={
            "completed_paths": ["a.py"],
            "in_progress_scan_by_path": {
                "b.py": {"phase": "scan_pending"},
            },
        },
        run_context=_run_context(),
    )
    assert collection_envelope.kind == "collection_progress"
    assert collection_envelope.source == "dataflow.collection_progress"

    index_envelope = adapt_dataflow_collection_progress_event_or_raise(
        collection_progress={
            "analysis_index_resume": {
                "index_cache_identity": "idx:1",
                "projection_cache_identity": "proj:1",
                "profiling_v1": {"stage_ns": {"analysis_index.parse_module": 1}},
            }
        },
        run_context=_run_context(),
    )
    assert index_envelope.kind == "analysis_index_progress"


def test_dataflow_adapter_hard_fails_on_insufficient_identity_data() -> None:
    phase_decision = adapt_dataflow_phase_progress_event(
        phase_progress={"phase": "post", "event_kind": "progress"},
        run_context=_run_context(),
    )
    assert phase_decision.kind == CanonicalAdaptationKind.REJECTED
    with pytest.raises(CanonicalEventAdaptationError):
        adapt_dataflow_phase_progress_event_or_raise(
            phase_progress={"phase": "post", "event_kind": "progress"},
            run_context=_run_context(),
        )

    collection_decision = adapt_dataflow_collection_progress_event(
        collection_progress={},
        run_context=_run_context(),
    )
    assert collection_decision.kind == CanonicalAdaptationKind.REJECTED
    with pytest.raises(CanonicalEventAdaptationError):
        adapt_dataflow_collection_progress_event_or_raise(
            collection_progress={},
            run_context=_run_context(),
        )
