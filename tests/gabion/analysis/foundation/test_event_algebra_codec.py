from __future__ import annotations

import pytest

from gabion.analysis.core.prime_identity_adapter import PrimeIdentityAdapter
from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.analysis.foundation.event_algebra import (
    CanonicalEventAdaptationError,
    CanonicalRunContext,
    GlobalEventSequencer,
    build_canonical_event_envelope,
    decode_canonical_event_json,
    derive_identity_projection_from_tokens,
    encode_canonical_event_json,
)
from gabion.analysis.foundation.event_algebra_codec import (
    CanonicalEventProtoDecodeError,
    decode_canonical_event_proto,
    encode_canonical_event_proto,
)
from gabion.analysis.foundation.identity_space import GlobalIdentitySpace


def _sample_envelope():
    run_context = CanonicalRunContext(
        run_id="run:codec",
        sequencer=GlobalEventSequencer(),
        identity_space=GlobalIdentitySpace(
            allocator=PrimeIdentityAdapter(registry=PrimeRegistry())
        ),
    )
    projection = derive_identity_projection_from_tokens(
        run_context=run_context,
        tokens=("codec", "phase", "kind", "event:1"),
    )
    return build_canonical_event_envelope(
        run_context=run_context,
        source="codec.test",
        phase="phase",
        kind="kind",
        identity_projection=projection,
        payload={"a": 1, "b": {"c": 2}},
        causal_refs=("cause:1", "cause:2"),
    )


# gabion:behavior primary=desired
def test_json_codec_round_trip_is_deterministic() -> None:
    envelope = _sample_envelope()
    encoded_first = encode_canonical_event_json(envelope)
    encoded_second = encode_canonical_event_json(envelope)
    decoded = decode_canonical_event_json(encoded_first)
    assert encoded_first == encoded_second
    assert decoded == envelope


# gabion:behavior primary=desired
def test_proto_codec_round_trip_is_deterministic() -> None:
    envelope = _sample_envelope()
    encoded_first = encode_canonical_event_proto(envelope)
    encoded_second = encode_canonical_event_proto(envelope)
    decoded = decode_canonical_event_proto(encoded_first)
    assert encoded_first == encoded_second
    assert decoded == envelope


# gabion:behavior primary=desired
def test_json_and_proto_codecs_have_semantic_parity() -> None:
    envelope = _sample_envelope()
    decoded_json = decode_canonical_event_json(encode_canonical_event_json(envelope))
    decoded_proto = decode_canonical_event_proto(encode_canonical_event_proto(envelope))
    assert decoded_json == decoded_proto == envelope


# gabion:behavior primary=verboten facets=missing
def test_decode_rejects_missing_required_fields() -> None:
    with pytest.raises(CanonicalEventAdaptationError):
        decode_canonical_event_json('{"schema_version":1}')
    with pytest.raises(CanonicalEventProtoDecodeError):
        decode_canonical_event_proto(b"")
