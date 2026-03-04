from __future__ import annotations

from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.analysis.foundation.identity_namespace_governance import (
    INTEGER_ANCHOR_NAMESPACE as GOVERNED_INTEGER_ANCHOR_NAMESPACE,
)
from gabion.analysis.foundation.identity_shadow_runtime import (
    BitPrimeIntegerCarrier,
    INTEGER_ANCHOR_NAMESPACE,
    IdentityShadowEmissionKind,
    build_identity_shadow_runtime,
)


def _phase_progress_payload(
    *,
    event_seq: int,
    marker: str = "fingerprint:warnings",
) -> dict[str, object]:
    return {
        "phase": "post",
        "event_kind": "progress",
        "event_seq": event_seq,
        "progress_marker": marker,
        "work_done": event_seq,
        "work_total": 3,
    }


def test_identity_shadow_runtime_accepts_progress_payload_and_emits_seed() -> None:
    runtime = build_identity_shadow_runtime(
        run_id="run:shadow",
        registry=PrimeRegistry(),
    )
    emission = runtime.adapt_progress_payload(
        phase="post",
        progress_payload=_phase_progress_payload(event_seq=1),
    )
    assert emission.kind == IdentityShadowEmissionKind.VALID
    assert isinstance(emission.canonical_event_v1, dict)
    assert emission.canonical_event_error_v1 == ""
    assert emission.identity_allocation_delta_v1
    seed = runtime.identity_seed_payload()
    assert seed.get("version") == "prime-registry-seed@1"
    namespaces = seed.get("namespaces")
    assert isinstance(namespaces, dict)
    assert "path" in namespaces


def test_identity_shadow_runtime_allocation_delta_cursor_is_monotonic() -> None:
    runtime = build_identity_shadow_runtime(
        run_id="run:delta",
        registry=PrimeRegistry(),
    )
    first = runtime.adapt_progress_payload(
        phase="post",
        progress_payload=_phase_progress_payload(event_seq=1),
    )
    second = runtime.adapt_progress_payload(
        phase="post",
        progress_payload=_phase_progress_payload(event_seq=1),
    )
    third = runtime.adapt_progress_payload(
        phase="post",
        progress_payload=_phase_progress_payload(event_seq=2),
    )
    assert first.identity_allocation_delta_v1
    assert second.identity_allocation_delta_v1 == []
    assert len(third.identity_allocation_delta_v1) >= 1
    combined = (
        first.identity_allocation_delta_v1
        + second.identity_allocation_delta_v1
        + third.identity_allocation_delta_v1
    )
    seqs = [int(row["seq"]) for row in combined]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))


def test_identity_shadow_runtime_rejects_progress_without_anchor() -> None:
    runtime = build_identity_shadow_runtime(
        run_id="run:reject",
        registry=PrimeRegistry(),
    )
    emission = runtime.adapt_progress_payload(
        phase="post",
        progress_payload={"phase": "post", "event_kind": "progress"},
    )
    assert emission.kind == IdentityShadowEmissionKind.REJECTED
    assert emission.canonical_event_v1 == {}
    assert isinstance(emission.canonical_event_error_v1, str)
    assert emission.canonical_event_error_v1
    assert emission.identity_allocation_delta_v1 == []


def test_identity_shadow_runtime_is_deterministic_with_live_shared_registry() -> None:
    shared_registry = PrimeRegistry()
    runtime_first = build_identity_shadow_runtime(
        run_id="run:deterministic",
        registry=shared_registry,
    )
    payloads = [
        _phase_progress_payload(event_seq=1),
        _phase_progress_payload(event_seq=2),
    ]
    first_events = [
        runtime_first.adapt_progress_payload(phase="post", progress_payload=payload)
        for payload in payloads
    ]
    runtime_second = build_identity_shadow_runtime(
        run_id="run:deterministic",
        registry=shared_registry,
    )
    second_events = [
        runtime_second.adapt_progress_payload(phase="post", progress_payload=payload)
        for payload in payloads
    ]
    assert [event.canonical_event_v1 for event in first_events] == [
        event.canonical_event_v1 for event in second_events
    ]


def test_bit_prime_integer_carrier_roundtrips_and_sorts_bits() -> None:
    carrier = BitPrimeIntegerCarrier()
    encoded = carrier.encode_anchor_tokens(
        namespace="dataflow.progress.integer_anchor",
        key="event_seq",
        value=13,
    )
    assert encoded == ("sign:+", "bit:0", "bit:2", "bit:3")
    decoded = carrier.decode_anchor_tokens(
        namespace="dataflow.progress.integer_anchor",
        key="event_seq",
        tokens=encoded,
    )
    assert decoded.is_present is True
    assert decoded.value == 13

    negative = carrier.encode_anchor_tokens(
        namespace="dataflow.progress.integer_anchor",
        key="event_seq",
        value=-5,
    )
    assert negative == ("sign:-", "bit:0", "bit:2")
    negative_decoded = carrier.decode_anchor_tokens(
        namespace="dataflow.progress.integer_anchor",
        key="event_seq",
        tokens=negative,
    )
    assert negative_decoded.is_present is True
    assert negative_decoded.value == -5


def test_identity_shadow_runtime_integer_anchor_tokens_scale_by_bit_width() -> None:
    runtime = build_identity_shadow_runtime(
        run_id="run:cardinality",
        registry=PrimeRegistry(),
    )
    for event_seq in range(1, 1025):
        runtime.adapt_progress_payload(
            phase="post",
            progress_payload=_phase_progress_payload(event_seq=event_seq),
        )
    records = runtime.run_context.identity_space.allocation_records()
    event_seq_tokens = {
        record.token
        for record in records
        if record.namespace == "path" and record.token.startswith("event_seq:")
    }
    # Bit-lowered tokens should scale with integer bit width, not event count.
    assert len(event_seq_tokens) <= 16


def test_identity_shadow_runtime_uses_governed_integer_anchor_namespace() -> None:
    assert INTEGER_ANCHOR_NAMESPACE == GOVERNED_INTEGER_ANCHOR_NAMESPACE
