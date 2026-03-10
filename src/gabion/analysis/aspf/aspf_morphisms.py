from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from gabion.analysis.foundation.frozen_object_map import (
    ObjectEntry,
    make_object_map,
)
from gabion.analysis.foundation.wire_types import WireValue


@dataclass(frozen=True)
class DomainPrimeBasis:
    domain_key: str
    prime: int


@dataclass(frozen=True)
class AspfPrimeBasis:
    aspf_key: str
    prime: int


@dataclass(frozen=True)
class DomainToAspfCofibrationEntry:
    domain: DomainPrimeBasis
    aspf: AspfPrimeBasis


@dataclass(frozen=True)
class DomainToAspfCofibration:
    entries: tuple[DomainToAspfCofibrationEntry, ...]

    def validate_injective(self) -> None:
        targets = list(map(lambda entry: entry.aspf.aspf_key, self.entries))
        _require(
            len(set(targets)) == len(targets),
            "Cofibration must be injective over ASPF basis targets",
        )

    def validate_faithful(self) -> None:
        _require(
            all(map(_primes_are_valid, self.entries)),
            "Cofibration primes must be valid (>1)",
        )
        _require(
            all(map(_primes_are_faithful, self.entries)),
            "Cofibration faithfulness requires prime-preserving embedding",
        )

    def validate(self) -> None:
        _require(
            len(self.entries) > 0,
            "Cofibration requires at least one basis embedding",
        )
        self.validate_injective()
        self.validate_faithful()

    def as_dict(self) -> Mapping[str, WireValue]:
        return make_object_map(
            [
                ObjectEntry(
                    "entries",
                    list(map(_cofibration_entry_payload, self.entries)),
                )
            ]
        )


@dataclass(frozen=True)
class CofibrationWitnessCarrier:
    canonical_identity_kind: str
    cofibration: DomainToAspfCofibration

    def as_dict(self) -> Mapping[str, WireValue]:
        return make_object_map(
            [
                ObjectEntry(
                    "canonical_identity_kind",
                    self.canonical_identity_kind,
                ),
                ObjectEntry("cofibration", self.cofibration.as_dict()),
            ]
        )


def _cofibration_entry_payload(entry: DomainToAspfCofibrationEntry) -> Mapping[str, WireValue]:
    return make_object_map(
        [
            ObjectEntry(
                "domain",
                make_object_map(
                    [
                        ObjectEntry("key", entry.domain.domain_key),
                        ObjectEntry("prime", entry.domain.prime),
                    ]
                ),
            ),
            ObjectEntry(
                "aspf",
                make_object_map(
                    [
                        ObjectEntry("key", entry.aspf.aspf_key),
                        ObjectEntry("prime", entry.aspf.prime),
                    ]
                ),
            ),
        ]
    )


def _primes_are_valid(entry: DomainToAspfCofibrationEntry) -> bool:
    return entry.domain.prime > 1 and entry.aspf.prime > 1


def _primes_are_faithful(entry: DomainToAspfCofibrationEntry) -> bool:
    return entry.domain.prime == entry.aspf.prime


def _noop_validator(_: str) -> None:
    return None


def _raise_validation_error(message: str) -> None:
    raise ValueError(message)


_VALIDATION_HANDLERS: list[Callable[[str], None]] = [_noop_validator, _raise_validation_error]


def _require(condition: bool, message: str) -> None:
    _VALIDATION_HANDLERS[not condition](message)
