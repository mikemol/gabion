# gabion:decision_protocol_module
from __future__ import annotations

from itertools import chain
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable, Iterator, Mapping, Protocol

from gabion.analysis.core.prime_identity_adapter import PrimeIdentityAdapter
from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.analysis.dataflow.engine.dataflow_event_algebra_adapter import (
    adapt_dataflow_collection_progress_event,
    adapt_dataflow_phase_progress_event,
)
from gabion.analysis.foundation.event_algebra import (
    CanonicalAdaptationDecision,
    CanonicalAdaptationKind,
    CanonicalEventAdaptationError,
    CanonicalRunContext,
    GlobalEventSequencer,
    _iter_canonical_event_json_items,
    envelope_from_decision_or_raise,
)
from gabion.analysis.foundation.identity_namespace_governance import (
    INTEGER_ANCHOR_NAMESPACE,
)
from gabion.analysis.foundation.identity_space import (
    GlobalIdentitySpace,
    IdentityAllocationRecord,
)
from gabion.analysis.foundation.json_types import JSONObject
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.invariants import never

DEFAULT_IDENTITY_SHADOW_RUN_ID = "gabion.dataflowAudit/progress-v2.shadow"


@dataclass(frozen=True)
class IntegerAnchorDecode:
    is_present: bool
    value: int = 0


class IntegerCarrierProtocol(Protocol):
    def encode_anchor_tokens(
        self,
        *,
        namespace: str,
        key: str,
        value: int,
    ) -> Iterator[str]: ...

    def decode_anchor_tokens(
        self,
        *,
        namespace: str,
        key: str,
        tokens: Iterable[str],
    ) -> IntegerAnchorDecode: ...


@dataclass(frozen=True)
class BitPrimeIntegerCarrier(IntegerCarrierProtocol):
    sign_positive: str = "sign:+"
    sign_negative: str = "sign:-"
    zero_token: str = "zero"
    bit_prefix: str = "bit:"

    def encode_anchor_tokens(
        self,
        *,
        namespace: str,
        key: str,
        value: int,
    ) -> Iterator[str]:
        check_deadline()
        _ = (namespace, key)
        integer_value = int(value)
        sign_token = self.sign_positive if integer_value >= 0 else self.sign_negative
        magnitude = abs(integer_value)
        if magnitude == 0:
            return iter((sign_token, self.zero_token))
        bit_tokens = map(
            lambda bit_index: f"{self.bit_prefix}{bit_index}",
            _iter_set_bit_indices(magnitude),
        )
        return chain((sign_token,), bit_tokens)

    def decode_anchor_tokens(
        self,
        *,
        namespace: str,
        key: str,
        tokens: Iterable[str],
    ) -> IntegerAnchorDecode:
        check_deadline()
        _ = (namespace, key)
        normalized_tokens = tuple(map(str.strip, map(str, tokens)))
        if not normalized_tokens:
            return IntegerAnchorDecode(is_present=False)
        if any(map(lambda token: not token, normalized_tokens)):
            return IntegerAnchorDecode(is_present=False)
        sign_tokens = tuple(filter(lambda token: _is_sign_token(self, token), normalized_tokens))
        if len(sign_tokens) != 1:
            return IntegerAnchorDecode(is_present=False)
        sign = 1 if sign_tokens[0] == self.sign_positive else -1
        zero_tokens = tuple(filter(lambda token: token == self.zero_token, normalized_tokens))
        if len(zero_tokens) > 1:
            return IntegerAnchorDecode(is_present=False)
        bit_tokens = tuple(filter(lambda token: token.startswith(self.bit_prefix), normalized_tokens))
        non_domain_tokens = tuple(
            filter(
                lambda token: _is_non_domain_token(self, token),
                normalized_tokens,
            )
        )
        if non_domain_tokens:
            return IntegerAnchorDecode(is_present=False)
        if zero_tokens and bit_tokens:
            return IntegerAnchorDecode(is_present=False)
        try:
            bit_indices = tuple(
                map(
                    lambda token: _bit_index_from_token(token, bit_prefix=self.bit_prefix),
                    bit_tokens,
                )
            )
        except ValueError:
            return IntegerAnchorDecode(is_present=False)
        if len(set(bit_indices)) != len(bit_indices):
            return IntegerAnchorDecode(is_present=False)
        magnitude = sum(map(lambda bit_index: 1 << bit_index, bit_indices))
        if zero_tokens:
            return IntegerAnchorDecode(is_present=True, value=0)
        if magnitude == 0:
            return IntegerAnchorDecode(is_present=False)
        return IntegerAnchorDecode(is_present=True, value=sign * magnitude)


class IdentityShadowEmissionKind(StrEnum):
    VALID = "valid"
    REJECTED = "rejected"


@dataclass(frozen=True)
class IdentityShadowEmission:
    kind: IdentityShadowEmissionKind
    identity_allocation_delta_v1: tuple[JSONObject, ...]
    canonical_event_v1: JSONObject = field(default_factory=dict)
    canonical_event_error_v1: str = ""

    def sidecar_payload(self) -> dict[str, object]:
        check_deadline()
        payload: dict[str, object] = {
            "identity_allocation_delta_v1": list(
                map(dict, self.identity_allocation_delta_v1)
            )
        }
        match self.kind:
            case IdentityShadowEmissionKind.VALID:
                payload["canonical_event_v1"] = dict(self.canonical_event_v1)
            case IdentityShadowEmissionKind.REJECTED:
                if self.canonical_event_error_v1:
                    payload["canonical_event_error_v1"] = self.canonical_event_error_v1
            case _:
                never(
                    "invalid identity shadow emission kind",
                    emission_kind=self.kind,
                )
        return payload


@dataclass
class IdentityShadowRuntime:
    run_context: CanonicalRunContext
    integer_carrier: IntegerCarrierProtocol = field(
        default_factory=BitPrimeIntegerCarrier
    )
    _allocation_cursor: int = 0

    def adapt_progress_payload(
        self,
        *,
        phase: str,
        progress_payload: Mapping[str, object],
        causal_refs: tuple[str, ...] = (),
    ) -> IdentityShadowEmission:
        check_deadline()
        phase_text = str(phase).strip()
        if phase_text == "collection":
            decision = adapt_dataflow_collection_progress_event(
                collection_progress=progress_payload,
                run_context=self.run_context,
                causal_refs=causal_refs,
                integer_anchor_encoder=self._encode_integer_anchor_tokens,
            )
        else:
            decision = adapt_dataflow_phase_progress_event(
                phase_progress=progress_payload,
                run_context=self.run_context,
                causal_refs=causal_refs,
                integer_anchor_encoder=self._encode_integer_anchor_tokens,
            )
        return self._emit_from_decision(decision=decision)

    def identity_seed_payload(self) -> JSONObject:
        check_deadline()
        seed = self.run_context.identity_space.seed_payload()
        normalized_seed: JSONObject = dict(
            map(
                lambda item: (str(item[0]), item[1]),
                seed.items(),
            )
        )
        return normalized_seed

    def _emit_from_decision(
        self,
        *,
        decision: CanonicalAdaptationDecision,
    ) -> IdentityShadowEmission:
        check_deadline()
        allocation_delta = tuple(self._allocation_delta_since_last_emit())
        match decision.kind:
            case CanonicalAdaptationKind.VALID:
                try:
                    envelope = envelope_from_decision_or_raise(decision)
                except CanonicalEventAdaptationError:
                    return IdentityShadowEmission(
                        kind=IdentityShadowEmissionKind.REJECTED,
                        canonical_event_v1={},
                        identity_allocation_delta_v1=allocation_delta,
                        canonical_event_error_v1="canonical adaptation returned VALID without envelope.",
                    )
                return IdentityShadowEmission(
                    kind=IdentityShadowEmissionKind.VALID,
                    canonical_event_v1=dict(_iter_canonical_event_json_items(envelope)),
                    identity_allocation_delta_v1=allocation_delta,
                )
            case CanonicalAdaptationKind.REJECTED:
                reason = str(decision.reason).strip()
                return IdentityShadowEmission(
                    kind=IdentityShadowEmissionKind.REJECTED,
                    canonical_event_v1={},
                    identity_allocation_delta_v1=allocation_delta,
                    canonical_event_error_v1=reason or "canonical adaptation rejected.",
                )
            case _:
                never(
                    "invalid identity shadow adaptation decision kind",
                    decision_kind=decision.kind,
                )
        return IdentityShadowEmission(
            kind=IdentityShadowEmissionKind.REJECTED,
            canonical_event_v1={},
            identity_allocation_delta_v1=allocation_delta,
            canonical_event_error_v1="invalid canonical adaptation decision kind.",
        )  # pragma: no cover - never() raises

    def _allocation_delta_since_last_emit(self) -> Iterator[JSONObject]:
        check_deadline()
        records = tuple(self.run_context.identity_space.allocation_records())
        cursor = min(max(int(self._allocation_cursor), 0), len(records))
        self._allocation_cursor = len(records)
        return map(
            lambda record: {
                "seq": int(record.seq),
                "namespace": str(record.namespace),
                "token": str(record.token),
                "atom_id": int(record.atom_id),
            },
            records[cursor:],
        )

    def _encode_integer_anchor_tokens(self, key: str, value: int) -> tuple[str, ...]:
        check_deadline()
        encoded = self.integer_carrier.encode_anchor_tokens(
            namespace=INTEGER_ANCHOR_NAMESPACE,
            key=str(key),
            value=int(value),
        )
        normalized_tokens = tuple(
            filter(
                bool,
                map(str.strip, map(str, encoded)),
            )
        )
        if not normalized_tokens:
            raise ValueError("integer carrier produced no anchor tokens.")
        return normalized_tokens


def build_identity_shadow_runtime(
    *,
    run_id: str,
    registry: PrimeRegistry,
    integer_carrier: IntegerCarrierProtocol = BitPrimeIntegerCarrier(),
) -> IdentityShadowRuntime:
    check_deadline()
    run_id_text = str(run_id).strip()
    if not run_id_text:
        raise ValueError("identity shadow runtime requires a non-empty run_id.")
    return IdentityShadowRuntime(
        run_context=CanonicalRunContext(
            run_id=run_id_text,
            sequencer=GlobalEventSequencer(),
            identity_space=GlobalIdentitySpace(
                allocator=PrimeIdentityAdapter(registry=registry),
            ),
        ),
        integer_carrier=integer_carrier,
    )


def _iter_set_bit_indices(magnitude: int) -> Iterator[int]:
    check_deadline()
    bit_index = 0
    pending = int(magnitude)
    while pending > 0:
        check_deadline()
        if pending & 1:
            yield bit_index
        bit_index += 1
        pending >>= 1


def _is_sign_token(carrier: BitPrimeIntegerCarrier, token: str) -> bool:
    return token == carrier.sign_positive or token == carrier.sign_negative


def _is_non_domain_token(carrier: BitPrimeIntegerCarrier, token: str) -> bool:
    return (
        token != carrier.zero_token
        and not token.startswith(carrier.bit_prefix)
        and not _is_sign_token(carrier, token)
    )


def _bit_index_from_token(token: str, *, bit_prefix: str) -> int:
    if not token.startswith(bit_prefix):
        raise ValueError("token is not a bit token")
    suffix = token[len(bit_prefix) :]
    if not suffix.isdigit():
        raise ValueError("bit token suffix must be numeric")
    return int(suffix)


__all__ = [
    "BitPrimeIntegerCarrier",
    "DEFAULT_IDENTITY_SHADOW_RUN_ID",
    "IntegerAnchorDecode",
    "INTEGER_ANCHOR_NAMESPACE",
    "IdentityShadowEmission",
    "IdentityShadowEmissionKind",
    "IdentityShadowRuntime",
    "IntegerCarrierProtocol",
    "build_identity_shadow_runtime",
]
