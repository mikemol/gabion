# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass


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
        seen_targets: set[str] = set()
        for entry in self.entries:
            target = entry.aspf.aspf_key
            if target in seen_targets:
                raise ValueError("Cofibration must be injective over ASPF basis targets")
            seen_targets.add(target)

    def validate_faithful(self) -> None:
        for entry in self.entries:
            if entry.domain.prime <= 1 or entry.aspf.prime <= 1:
                raise ValueError("Cofibration primes must be valid (>1)")
            if entry.domain.prime != entry.aspf.prime:
                raise ValueError("Cofibration faithfulness requires prime-preserving embedding")

    def validate(self) -> None:
        if not self.entries:
            raise ValueError("Cofibration requires at least one basis embedding")
        self.validate_injective()
        self.validate_faithful()

    def as_dict(self) -> dict[str, object]:
        return {
            "entries": [
                {
                    "domain": {
                        "key": entry.domain.domain_key,
                        "prime": entry.domain.prime,
                    },
                    "aspf": {
                        "key": entry.aspf.aspf_key,
                        "prime": entry.aspf.prime,
                    },
                }
                for entry in self.entries
            ]
        }
