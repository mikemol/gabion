# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
import math

from gabion.analysis.aspf_core import (
    AspfCanonicalIdentityContract,
    AspfOneCell,
    AspfTwoCellWitness,
    BasisZeroCell,
    SuiteSiteEndpoint,
    validate_2cell_compatibility,
)
from gabion.analysis.aspf_decision_surface import (
    RepresentativeSelectionMode,
    RepresentativeSelectionOptions,
    select_representative,
)
from gabion.analysis.aspf_morphisms import (
    AspfPrimeBasis,
    DomainPrimeBasis,
    DomainToAspfCofibration,
    DomainToAspfCofibrationEntry,
)
from gabion.analysis.evidence_keys import fingerprint_identity_layers
from gabion.analysis.json_types import JSONObject, JSONValue
from gabion.analysis.timeout_context import check_deadline
from gabion.analysis.timeout_context import consume_deadline_ticks
from gabion.order_contract import OrderPolicy, sort_once

TYPE_BASE_NAMESPACE = "type_base"
TYPE_CTOR_NAMESPACE = "type_ctor"
EVIDENCE_KIND_NAMESPACE = "evidence_kind"
SITE_KIND_NAMESPACE = "site_kind"
SYNTH_NAMESPACE = "synth"

_NAMESPACE_TO_PREFIX: dict[str, str] = {
    TYPE_BASE_NAMESPACE: "",
    TYPE_CTOR_NAMESPACE: "ctor:",
    EVIDENCE_KIND_NAMESPACE: "evidence:",
    SITE_KIND_NAMESPACE: "site:",
    SYNTH_NAMESPACE: "synth:",
}


def _namespace_key(key: str) -> tuple[str, str]:
    check_deadline()
    for namespace, prefix in _NAMESPACE_TO_PREFIX.items():
        check_deadline()
        if prefix and key.startswith(prefix):
            return namespace, key[len(prefix) :]
    return TYPE_BASE_NAMESPACE, key


def _raw_key(namespace: str, key: str) -> str:
    prefix = _NAMESPACE_TO_PREFIX.get(namespace, "")
    return f"{prefix}{key}"

def _split_top_level(value: str, sep: str) -> list[str]:
    check_deadline()
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in value:
        check_deadline()
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth = max(depth - 1, 0)
        if ch == sep and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _strip_known_prefix(name: str) -> str:
    check_deadline()
    for prefix in ("typing.", "builtins."):
        check_deadline()
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _normalize_base(name: str) -> str:
    base = _strip_known_prefix(name.strip())
    lowered = base.lower()
    if lowered in {"list", "dict", "set", "tuple"}:
        return lowered
    if lowered == "optional":
        return "Optional"
    if lowered == "union":
        return "Union"
    if lowered in {"none", "nonetype", "type(None)"}:
        return "None"
    return base


def canonical_type_key(hint: str) -> str:
    check_deadline()
    raw = hint.strip()
    if not raw:
        return ""
    union_parts = _split_top_level(raw, "|")
    if len(union_parts) > 1:
        normalized = sort_once(
            (part for part in (canonical_type_key(p) for p in union_parts) if part),
            source="canonical_type_key.union_parts",
            policy=OrderPolicy.SORT,
        )
        normalized = sort_once(
            normalized,
            source="canonical_type_key.union_parts.enforce",
            policy=OrderPolicy.ENFORCE,
        )
        return f"Union[{', '.join(normalized)}]"
    if raw.startswith("Optional[") and raw.endswith("]"):
        inner = raw[len("Optional[") : -1]
        parts = _split_top_level(inner, ",")
        normalized = [canonical_type_key(p) for p in parts]
        normalized.append("None")
        normalized = sort_once(
            {item for item in normalized if item},
            source="canonical_type_key.optional_parts",
            policy=OrderPolicy.SORT,
        )
        normalized = sort_once(
            normalized,
            source="canonical_type_key.optional_parts.enforce",
            policy=OrderPolicy.ENFORCE,
        )
        return f"Union[{', '.join(normalized)}]"
    if raw.startswith("Union[") and raw.endswith("]"):
        inner = raw[len("Union[") : -1]
        parts = _split_top_level(inner, ",")
        normalized = sort_once(
            (part for part in (canonical_type_key(p) for p in parts) if part),
            source="canonical_type_key.union_bracket_parts",
            policy=OrderPolicy.SORT,
        )
        normalized = sort_once(
            normalized,
            source="canonical_type_key.union_bracket_parts.enforce",
            policy=OrderPolicy.ENFORCE,
        )
        return f"Union[{', '.join(normalized)}]"
    if "[" in raw and raw.endswith("]"):
        base, inner = raw.split("[", 1)
        inner = inner[:-1]
        normalized_base = _normalize_base(base)
        parts = _split_top_level(inner, ",")
        normalized = [canonical_type_key(p) for p in parts if p.strip()]
        return f"{normalized_base}[{', '.join(normalized)}]"
    return _normalize_base(raw)


def _is_prime(value: int) -> bool:
    check_deadline()
    if value < 2:
        return False
    if value in (2, 3):
        return True
    if value % 2 == 0 or value % 3 == 0:
        return False
    step = 5
    while step * step <= value:
        check_deadline()
        if value % step == 0 or value % (step + 2) == 0:
            return False
        step += 6
    return True


def _next_prime(start: int) -> int:
    check_deadline()
    candidate = max(start, 2)
    while not _is_prime(candidate):
        check_deadline()
        candidate += 1
    return candidate


@dataclass
class PrimeRegistry:
    primes: dict[str, int] = field(default_factory=dict)
    next_candidate: int = 2
    bit_positions: dict[str, int] = field(default_factory=dict)
    next_bit: int = 0

    def get_or_assign(self, key: str) -> int:
        consume_deadline_ticks()
        check_deadline()
        if not key:
            raise ValueError("Type key must be non-empty.")
        existing = self.primes.get(key)
        if existing is not None:
            return existing
        prime = _next_prime(self.next_candidate)
        self.primes[key] = prime
        self.next_candidate = prime + 1
        if key not in self.bit_positions:
            self.bit_positions[key] = self.next_bit
            self.next_bit += 1
        return prime

    def prime_for(self, key: str) -> int | None:
        return self.primes.get(key)

    def key_for_prime(self, prime: int) -> str | None:
        check_deadline()
        for key, value in self.primes.items():
            check_deadline()
            if value == prime:
                return key
        return None

    def bit_for(self, key: str) -> int | None:
        return self.bit_positions.get(key)

    def seed_payload(self) -> JSONObject:
        check_deadline()
        namespace_payload: dict[str, JSONObject] = {}

        def _entry(namespace: str) -> JSONObject:
            payload = namespace_payload.get(namespace)
            if payload is None:
                payload = {"primes": {}, "bit_positions": {}}
                namespace_payload[namespace] = payload
            return payload

        for key, prime in sort_once(
            self.primes.items(),
            source="PrimeRegistry.seed_payload.primes",
            policy=OrderPolicy.SORT,
        ):
            check_deadline()
            namespace, local_key = _namespace_key(key)
            _entry(namespace)["primes"][local_key] = int(prime)
        for key, bit in sort_once(
            self.bit_positions.items(),
            source="PrimeRegistry.seed_payload.bit_positions",
            policy=OrderPolicy.SORT,
        ):
            check_deadline()
            namespace, local_key = _namespace_key(key)
            _entry(namespace)["bit_positions"][local_key] = int(bit)
        return {
            "version": "prime-registry-seed@1",
            "namespaces": namespace_payload,
        }

    def load_seed_payload(self, payload: object) -> None:
        check_deadline()
        if not isinstance(payload, dict):
            return
        namespaces = payload.get("namespaces")
        if not isinstance(namespaces, dict):
            _apply_registry_payload(payload, self)
            return
        flat_primes: dict[str, int] = {}
        flat_bits: dict[str, int] = {}
        for namespace, section in namespaces.items():
            check_deadline()
            if not isinstance(namespace, str) or not isinstance(section, dict):
                continue
            primes = section.get("primes")
            bits = section.get("bit_positions")
            if isinstance(primes, dict):
                for key, value in primes.items():
                    check_deadline()
                    if isinstance(key, str) and isinstance(value, int):
                        flat_primes[_raw_key(namespace, key)] = value
            if isinstance(bits, dict):
                for key, value in bits.items():
                    check_deadline()
                    if isinstance(key, str) and isinstance(value, int):
                        flat_bits[_raw_key(namespace, key)] = value
        _apply_registry_payload({"primes": flat_primes, "bit_positions": flat_bits}, self)


@dataclass(frozen=True)
class FingerprintDimension:
    product: int = 1
    mask: int = 0
    # Sparse exponent-vector sidecar keyed by normalized registry key.
    # Kept out of equality/hash for compatibility with persisted payloads that
    # may not carry sidecars yet.
    exponents: tuple[tuple[str, int], ...] = field(
        default_factory=tuple,
        compare=False,
    )

    def is_empty(self) -> bool:
        return self.product in (0, 1) and self.mask == 0

    def keys_with_remainder(
        self,
        registry: PrimeRegistry,
    ) -> tuple[list[str], int]:
        """Decode this dimension, preferring the sparse exponent sidecar."""
        check_deadline()
        if not self.exponents:
            return fingerprint_to_type_keys_with_remainder(self.product, registry)
        keys: list[str] = []
        for key, exponent in self.exponents:
            check_deadline()
            if exponent <= 0:
                continue
            keys.extend([key] * exponent)
        sidecar_product = 1
        for key in keys:
            check_deadline()
            prime = registry.prime_for(key)
            if prime is None:
                # Incomplete registry basis: preserve soundness by falling back.
                return fingerprint_to_type_keys_with_remainder(self.product, registry)
            sidecar_product *= prime
        if sidecar_product != self.product:
            # Product remains source-of-truth; sidecar is an optimization.
            return fingerprint_to_type_keys_with_remainder(self.product, registry)
        return keys, 1


@dataclass(frozen=True)
class Fingerprint:
    base: FingerprintDimension
    ctor: FingerprintDimension = field(default_factory=FingerprintDimension)
    provenance: FingerprintDimension = field(default_factory=FingerprintDimension)
    synth: FingerprintDimension = field(default_factory=FingerprintDimension)

    def __str__(self) -> str:
        return format_fingerprint(self)


def fingerprint_to_aspf_path(fingerprint: Fingerprint) -> AspfOneCell:
    start = BasisZeroCell(label="fingerprint:start")
    end = BasisZeroCell(label="fingerprint:end")
    return AspfOneCell(
        source=start,
        target=end,
        representative=(
            f"base={fingerprint.base.product}|ctor={fingerprint.ctor.product}|"
            f"prov={fingerprint.provenance.product}|synth={fingerprint.synth.product}"
        ),
        basis_path=(
            f"base:{fingerprint.base.product}",
            f"ctor:{fingerprint.ctor.product}",
            f"prov:{fingerprint.provenance.product}",
            f"synth:{fingerprint.synth.product}",
        ),
    )


def fingerprint_identity_payload(
    fingerprint: Fingerprint,
    *,
    representative_mode: RepresentativeSelectionMode = RepresentativeSelectionMode.LEXICOGRAPHIC_MIN,
) -> JSONObject:
    aspf_path = fingerprint_to_aspf_path(fingerprint)
    candidates = (
        aspf_path.representative,
        "|".join(aspf_path.basis_path),
    )
    rep_witness = select_representative(
        RepresentativeSelectionOptions(mode=representative_mode, candidates=candidates)
    )
    cofibration = DomainToAspfCofibration(
        entries=tuple(
            DomainToAspfCofibrationEntry(
                domain=DomainPrimeBasis(domain_key=key, prime=prime),
                aspf=AspfPrimeBasis(aspf_key=f"aspf:{key}", prime=prime),
            )
            for key, prime in sort_once(
                {
                    "base": fingerprint.base.product,
                    "ctor": fingerprint.ctor.product,
                    "prov": fingerprint.provenance.product,
                    "synth": fingerprint.synth.product,
                }.items(),
                source="fingerprint_identity_payload.cofibration_entries",
                policy=OrderPolicy.SORT,
            )
            if prime > 1
        )
    )
    if cofibration.entries:
        cofibration.validate()
    canonical_identity = AspfCanonicalIdentityContract(
        identity_kind="canonical_aspf_structural_identity",
        source=aspf_path.source,
        target=aspf_path.target,
        representative=rep_witness.selected,
        basis_path=aspf_path.basis_path,
        suite_site_source=SuiteSiteEndpoint(
            kind="SuiteSite",
            key=(aspf_path.source.label, "fingerprint", "source"),
        ),
        suite_site_target=SuiteSiteEndpoint(
            kind="SuiteSite",
            key=(aspf_path.target.label, "fingerprint", "target"),
        ),
    )
    scalar_alias = max(
        1,
        fingerprint.base.product
        * max(1, fingerprint.ctor.product)
        * max(1, fingerprint.provenance.product)
        * max(1, fingerprint.synth.product),
    )
    layers = fingerprint_identity_layers(
        canonical_aspf_path=canonical_identity.as_dict(),
        scalar_prime_product=scalar_alias,
    )
    higher_path_witness = AspfTwoCellWitness(
        left=AspfOneCell(
            source=aspf_path.source,
            target=aspf_path.target,
            representative=rep_witness.selected,
            basis_path=aspf_path.basis_path,
        ),
        right=AspfOneCell(
            source=aspf_path.source,
            target=aspf_path.target,
            representative="|".join(aspf_path.basis_path),
            basis_path=aspf_path.basis_path,
        ),
        witness_id=f"higher:{rep_witness.witness_id}",
        reason="representative_selection_coherence",
    )
    validate_2cell_compatibility(higher_path_witness)
    payload: JSONObject = {
        "canonical_identity_contract": canonical_identity.as_dict(),
        "identity_layers": layers.as_dict(),
        "representative_selection": rep_witness.as_dict(),
        "witness_carriers": {
            "selection_witness": rep_witness.as_dict(),
            "cofibration_witness": cofibration.as_dict() if cofibration.entries else {"entries": []},
            "higher_path_witness": higher_path_witness.as_dict(),
        },
        "derived_aliases": {
            "scalar_prime_product": {
                "value": scalar_alias,
                "canonical": False,
                "alias_of": "canonical_identity_contract",
            },
            "digest_alias": {
                "value": layers.digest_projection["value"],
                "canonical": False,
                "alias_of": "canonical_identity_contract",
            },
        },
    }
    if cofibration.entries:
        payload["cofibration_witness"] = cofibration.as_dict()
    return payload


def _fingerprint_sort_key(fingerprint: Fingerprint) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        fingerprint.base.product,
        fingerprint.base.mask,
        fingerprint.ctor.product,
        fingerprint.ctor.mask,
        fingerprint.provenance.product,
        fingerprint.provenance.mask,
        fingerprint.synth.product,
        fingerprint.synth.mask,
    )


@dataclass
class TypeConstructorRegistry:
    registry: PrimeRegistry
    constructors: dict[str, int] = field(default_factory=dict)

    def get_or_assign(self, constructor: str) -> int:
        check_deadline()
        key = _normalize_base(constructor)
        prime = self.constructors.get(key)
        if prime is not None:
            return prime
        prime = self.registry.get_or_assign(_raw_key(TYPE_CTOR_NAMESPACE, key))
        self.constructors[key] = prime
        return prime


def canonical_type_key_with_constructor(
    hint: str,
    ctor_registry: TypeConstructorRegistry,
) -> str:
    check_deadline()
    raw = hint.strip()
    if not raw:
        return ""
    union_parts = _split_top_level(raw, "|")
    if len(union_parts) > 1:
        normalized = sort_once(
            (
                part
                for part in (
                    canonical_type_key_with_constructor(p, ctor_registry)
                    for p in union_parts
                )
                if part
            ),
            source="canonical_type_key_with_constructor.union_parts",
            policy=OrderPolicy.SORT,
        )
        return f"Union[{', '.join(normalized)}]"
    if raw.startswith("Optional[") and raw.endswith("]"):
        inner = raw[len("Optional[") : -1]
        parts = _split_top_level(inner, ",")
        normalized = [
            canonical_type_key_with_constructor(p, ctor_registry) for p in parts
        ]
        normalized.append("None")
        normalized = sort_once(
            {item for item in normalized if item},
            source="canonical_type_key_with_constructor.optional_parts",
            policy=OrderPolicy.SORT,
        )
        return f"Union[{', '.join(normalized)}]"
    if raw.startswith("Union[") and raw.endswith("]"):
        inner = raw[len("Union[") : -1]
        parts = _split_top_level(inner, ",")
        normalized = sort_once(
            (
                part
                for part in (
                    canonical_type_key_with_constructor(p, ctor_registry)
                    for p in parts
                )
                if part
            ),
            source="canonical_type_key_with_constructor.union_bracket_parts",
            policy=OrderPolicy.SORT,
        )
        return f"Union[{', '.join(normalized)}]"
    if "[" in raw and raw.endswith("]"):
        base, inner = raw.split("[", 1)
        inner = inner[:-1]
        normalized_base = _normalize_base(base)
        ctor_registry.get_or_assign(normalized_base)
        parts = _split_top_level(inner, ",")
        normalized = [
            canonical_type_key_with_constructor(p, ctor_registry)
            for p in parts
            if p.strip()
        ]
        return f"{normalized_base}[{', '.join(normalized)}]"
    return _normalize_base(raw)


def _collect_base_atoms(hint: str, out: list[str]) -> None:
    check_deadline()
    raw = hint.strip()
    if not raw:
        return
    union_parts = _split_top_level(raw, "|")
    if len(union_parts) > 1:
        for part in union_parts:
            check_deadline()
            _collect_base_atoms(part, out)
        return
    if raw.startswith("Optional[") and raw.endswith("]"):
        inner = raw[len("Optional[") : -1]
        for part in _split_top_level(inner, ","):
            check_deadline()
            _collect_base_atoms(part, out)
        out.append("None")
        return
    if raw.startswith("Union[") and raw.endswith("]"):
        inner = raw[len("Union[") : -1]
        for part in _split_top_level(inner, ","):
            check_deadline()
            _collect_base_atoms(part, out)
        return
    if "[" in raw and raw.endswith("]"):
        _, inner = raw.split("[", 1)
        inner = inner[:-1]
        for part in _split_top_level(inner, ","):
            check_deadline()
            _collect_base_atoms(part, out)
        return
    out.append(_normalize_base(raw))


def _collect_constructors_multiset(hint: str, out: list[str]) -> None:
    check_deadline()
    raw = hint.strip()
    if not raw:
        return
    union_parts = _split_top_level(raw, "|")
    if len(union_parts) > 1:
        for part in union_parts:
            check_deadline()
            _collect_constructors_multiset(part, out)
        return
    if raw.startswith("Optional[") and raw.endswith("]"):
        inner = raw[len("Optional[") : -1]
        for part in _split_top_level(inner, ","):
            check_deadline()
            _collect_constructors_multiset(part, out)
        return
    if raw.startswith("Union[") and raw.endswith("]"):
        inner = raw[len("Union[") : -1]
        for part in _split_top_level(inner, ","):
            check_deadline()
            _collect_constructors_multiset(part, out)
        return
    if "[" in raw and raw.endswith("]"):
        base, inner = raw.split("[", 1)
        normalized_base = _normalize_base(base)
        out.append(normalized_base)
        inner = inner[:-1]
        for part in _split_top_level(inner, ","):
            check_deadline()
            _collect_constructors_multiset(part, out)

def _normalize_type_list(value: object) -> list[str]:
    check_deadline()
    items: list[str] = []
    if value is None:
        return items
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            check_deadline()
            if isinstance(item, str):
                items.extend([part.strip() for part in item.split(",") if part.strip()])
    return [item for item in items if item]


def _collect_constructors(hint: str, out: set[str]) -> None:
    check_deadline()
    raw = hint.strip()
    if not raw:
        return
    union_parts = _split_top_level(raw, "|")
    if len(union_parts) > 1:
        for part in union_parts:
            check_deadline()
            _collect_constructors(part, out)
        return
    if raw.startswith("Optional[") and raw.endswith("]"):
        inner = raw[len("Optional[") : -1]
        for part in _split_top_level(inner, ","):
            check_deadline()
            _collect_constructors(part, out)
        return
    if raw.startswith("Union[") and raw.endswith("]"):
        inner = raw[len("Union[") : -1]
        for part in _split_top_level(inner, ","):
            check_deadline()
            _collect_constructors(part, out)
        return
    if "[" in raw and raw.endswith("]"):
        base, inner = raw.split("[", 1)
        normalized_base = _normalize_base(base)
        out.add(normalized_base)
        inner = inner[:-1]
        for part in _split_top_level(inner, ","):
            check_deadline()
            _collect_constructors(part, out)


def _dimension_from_keys(keys: Iterable[str], registry: PrimeRegistry) -> FingerprintDimension:
    check_deadline()
    product = 1
    mask = 0
    exponents: dict[str, int] = {}
    for key in keys:
        check_deadline()
        if not key:
            continue
        prime = registry.get_or_assign(key)
        product *= prime
        exponents[key] = exponents.get(key, 0) + 1
        bit = registry.bit_for(key)
        if bit is not None:
            mask |= 1 << bit
    return FingerprintDimension(
        product=product,
        mask=mask,
        exponents=tuple(
            sort_once(
                exponents.items(),
                source="_dimension_from_keys.exponents",
                policy=OrderPolicy.SORT,
            )
        ),
    )


def _ctor_dimension_from_names(
    names: Iterable[str],
    registry: PrimeRegistry,
) -> FingerprintDimension:
    check_deadline()
    product = 1
    mask = 0
    exponents: dict[str, int] = {}
    for name in names:
        check_deadline()
        if not name:
            continue
        key = _raw_key(TYPE_CTOR_NAMESPACE, _normalize_base(name))
        prime = registry.get_or_assign(key)
        product *= prime
        exponents[key] = exponents.get(key, 0) + 1
        bit = registry.bit_for(key)
        if bit is not None:
            mask |= 1 << bit
    return FingerprintDimension(
        product=product,
        mask=mask,
        exponents=tuple(
            sort_once(
                exponents.items(),
                source="_ctor_dimension_from_names.exponents",
                policy=OrderPolicy.SORT,
            )
        ),
    )


def format_fingerprint(fingerprint: Fingerprint) -> str:
    parts = [f"base={fingerprint.base.product}"]
    if not fingerprint.ctor.is_empty():
        parts.append(f"ctor={fingerprint.ctor.product}")
    if not fingerprint.provenance.is_empty():
        parts.append(f"prov={fingerprint.provenance.product}")
    if not fingerprint.synth.is_empty():
        parts.append(f"synth={fingerprint.synth.product}")
    return "{" + ", ".join(parts) + "}"


def _synth_key(version: str, fingerprint: Fingerprint) -> str:
    return (
        f"synth:{version}:"
        f"{fingerprint.base.product}:{fingerprint.ctor.product}:"
        f"{fingerprint.provenance.product}:{fingerprint.provenance.mask}"
    )


@dataclass
class SynthRegistry:
    registry: PrimeRegistry
    version: str = "synth@1"
    primes: dict[Fingerprint, int] = field(default_factory=dict)
    tails: dict[int, Fingerprint] = field(default_factory=dict)

    def get_or_assign(self, fingerprint: Fingerprint) -> int:
        check_deadline()
        existing = self.primes.get(fingerprint)
        if existing is not None:
            return existing
        key = _synth_key(self.version, fingerprint)
        prime = self.registry.get_or_assign(key)
        self.primes[fingerprint] = prime
        self.tails[prime] = fingerprint
        return prime

    def synth_dimension_for(self, fingerprint: Fingerprint) -> FingerprintDimension | None:
        prime = self.primes.get(fingerprint)
        if prime is None:
            return None
        key = _synth_key(self.version, fingerprint)
        bit = self.registry.bit_for(key)
        mask = 0 if bit is None else 1 << bit
        return FingerprintDimension(product=prime, mask=mask, exponents=((key, 1),))


def build_synth_registry(
    fingerprints: Iterable[Fingerprint],
    registry: PrimeRegistry,
    *,
    min_occurrences: int = 2,
    version: str = "synth@1",
) -> SynthRegistry:
    check_deadline()
    counts: dict[Fingerprint, int] = {}
    for fingerprint in fingerprints:
        check_deadline()
        counts[fingerprint] = counts.get(fingerprint, 0) + 1
    synth_registry = SynthRegistry(registry=registry, version=version)
    candidates = [fp for fp, count in counts.items() if count >= min_occurrences]
    for fingerprint in sort_once(
        candidates,
        source="build_synth_registry.candidates",
        key=_fingerprint_sort_key,
        policy=OrderPolicy.SORT,
    ):
        check_deadline()
        synth_registry.get_or_assign(fingerprint)
    return synth_registry


def apply_synth_dimension(
    fingerprint: Fingerprint,
    synth_registry: SynthRegistry,
) -> Fingerprint:
    synth_dim = synth_registry.synth_dimension_for(fingerprint)
    if synth_dim is None:
        return fingerprint
    return Fingerprint(
        base=fingerprint.base,
        ctor=fingerprint.ctor,
        provenance=fingerprint.provenance,
        synth=synth_dim,
    )


def synth_registry_payload(
    synth_registry: SynthRegistry,
    registry: PrimeRegistry,
    *,
    min_occurrences: int,
) -> JSONObject:
    check_deadline()
    entries: list[JSONObject] = []
    for prime, tail in sort_once(
        synth_registry.tails.items(),
        source="synth_registry_payload.tails",
        policy=OrderPolicy.SORT,
    ):
        check_deadline()
        base_keys, base_remaining = fingerprint_to_type_keys_with_remainder(
            tail.base.product, registry
        )
        ctor_keys, ctor_remaining = fingerprint_to_type_keys_with_remainder(
            tail.ctor.product, registry
        )
        ctor_keys = [
            key[len("ctor:") :] if key.startswith("ctor:") else key
            for key in ctor_keys
        ]
        entries.append(
            {
                "prime": prime,
                "tail": {
                    "base": {
                        "product": tail.base.product,
                        "mask": tail.base.mask,
                    },
                    "ctor": {
                        "product": tail.ctor.product,
                        "mask": tail.ctor.mask,
                    },
                    "provenance": {
                        "product": tail.provenance.product,
                        "mask": tail.provenance.mask,
                    },
                    "synth": {
                        "product": tail.synth.product,
                        "mask": tail.synth.mask,
                    },
                },
                "base_keys": sort_once(
                    base_keys,
                    source="synth_registry_payload.base_keys",
                    policy=OrderPolicy.SORT,
                ),
                "ctor_keys": sort_once(
                    ctor_keys,
                    source="synth_registry_payload.ctor_keys",
                    policy=OrderPolicy.SORT,
                ),
                "remainder": {
                    "base": base_remaining,
                    "ctor": ctor_remaining,
                },
            }
        )
    seed_payload = registry.seed_payload()
    return {
        "version": synth_registry.version,
        "min_occurrences": min_occurrences,
        "entries": entries,
        # Registry basis for deterministic reload across runs/snapshots.
        # This turns the synth registry artifact into a reproducible basis.
        "registry": {
            "primes": {
                key: int(value)
                for key, value in sort_once(
                    registry.primes.items(),
                    source="synth_registry_payload.registry.primes",
                    policy=OrderPolicy.SORT,
                )
            },
            "bit_positions": {
                key: int(value)
                for key, value in sort_once(
                    registry.bit_positions.items(),
                    source="synth_registry_payload.registry.bit_positions",
                    policy=OrderPolicy.SORT,
                )
            },
            "seed": seed_payload,
        },
    }


def load_synth_registry_payload(
    payload: JSONObject,
) -> tuple[list[JSONObject], str, int]:
    version = str(payload.get("version", "synth@1"))
    min_occurrences = int(payload.get("min_occurrences", 2))
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        entries = []
    return entries, version, min_occurrences


def build_synth_registry_from_payload(
    payload: JSONObject,
    registry: PrimeRegistry,
) -> SynthRegistry:
    check_deadline()
    registry_payload = payload.get("registry")
    if isinstance(registry_payload, dict):
        registry.load_seed_payload(registry_payload.get("seed"))
    _apply_registry_payload(registry_payload, registry)
    entries, version, _ = load_synth_registry_payload(payload)
    synth_registry = SynthRegistry(registry=registry, version=version)
    for entry in entries:
        check_deadline()
        if not isinstance(entry, dict):
            continue
        prime = entry.get("prime")
        tail = entry.get("tail", {})
        base = tail.get("base", {}) if isinstance(tail, dict) else {}
        ctor = tail.get("ctor", {}) if isinstance(tail, dict) else {}
        provenance = tail.get("provenance", {}) if isinstance(tail, dict) else {}
        synth = tail.get("synth", {}) if isinstance(tail, dict) else {}
        fingerprint = Fingerprint(
            base=FingerprintDimension(
                product=int(base.get("product", 1)),
                mask=int(base.get("mask", 0)),
            ),
            ctor=FingerprintDimension(
                product=int(ctor.get("product", 1)),
                mask=int(ctor.get("mask", 0)),
            ),
            provenance=FingerprintDimension(
                product=int(provenance.get("product", 1)),
                mask=int(provenance.get("mask", 0)),
            ),
            synth=FingerprintDimension(
                product=int(synth.get("product", 1)),
                mask=int(synth.get("mask", 0)),
            ),
        )
        assigned_prime = synth_registry.get_or_assign(fingerprint)
        if isinstance(prime, int) and assigned_prime != prime:
            synth_registry.tails[prime] = fingerprint
            synth_registry.primes[fingerprint] = prime
    return synth_registry


def _apply_registry_payload(
    payload: object,
    registry: PrimeRegistry,
) -> None:
    """Extend a PrimeRegistry with a serialized basis.

    This is intentionally conservative: we only add missing keys and validate
    that any already-assigned primes/bits are consistent. Conflicts indicate a
    stale or non-deterministic basis and must be handled explicitly.
    """
    check_deadline()
    if not isinstance(payload, dict):
        return
    primes = payload.get("primes")
    bits = payload.get("bit_positions")
    primes_map: dict[str, int] = {}
    bits_map: dict[str, int] = {}
    if isinstance(primes, dict):
        for key, value in primes.items():
            check_deadline()
            if not isinstance(key, str):
                continue
            if isinstance(value, int):
                primes_map[key] = value
    if isinstance(bits, dict):
        for key, value in bits.items():
            check_deadline()
            if not isinstance(key, str):
                continue
            if isinstance(value, int):
                bits_map[key] = value

    for key, prime in primes_map.items():
        check_deadline()
        existing = registry.primes.get(key)
        if existing is not None and existing != prime:
            raise ValueError(
                f"Registry basis mismatch for {key}: have {existing} expected {prime}"
            )
        registry.primes.setdefault(key, prime)

    for key, bit in bits_map.items():
        check_deadline()
        existing = registry.bit_positions.get(key)
        if existing is not None and existing != bit:
            raise ValueError(
                f"Registry basis mismatch for bit {key}: have {existing} expected {bit}"
            )
        registry.bit_positions.setdefault(key, bit)

    if registry.primes and len(set(registry.primes.values())) != len(registry.primes):
        raise ValueError("Registry basis contains duplicate primes.")
    if registry.bit_positions and len(set(registry.bit_positions.values())) != len(
        registry.bit_positions
    ):
        raise ValueError("Registry basis contains duplicate bit positions.")

    if registry.primes:
        registry.next_candidate = max(
            registry.next_candidate, max(registry.primes.values()) + 1
        )
    if registry.bit_positions:
        registry.next_bit = max(registry.next_bit, max(registry.bit_positions.values()) + 1)

    # Older payloads may omit bit positions; assign deterministically for any
    # primes without a bit position so mask carriers remain usable.
    for key in sort_once(
        registry.primes,
        source="_apply_registry_payload.registry.primes",
        policy=OrderPolicy.SORT,
    ):
        check_deadline()
        if key not in registry.bit_positions:
            registry.bit_positions[key] = registry.next_bit
            registry.next_bit += 1

    if registry.bit_positions:
        registry.next_bit = max(
            registry.next_bit, max(registry.bit_positions.values()) + 1
        )


def fingerprint_carrier_soundness(a: FingerprintDimension, b: FingerprintDimension) -> bool:
    if (a.mask & b.mask) == 0:
        return math.gcd(a.product, b.product) == 1
    return True


def bundle_fingerprint_dimensional(
    types: Iterable[str],
    registry: PrimeRegistry,
    ctor_registry: TypeConstructorRegistry | None = None,
) -> Fingerprint:
    check_deadline()
    base_keys: list[str] = []
    ctor_names: list[str] = []
    for hint in types:
        check_deadline()
        _collect_base_atoms(hint, base_keys)
        if ctor_registry is not None:
            _collect_constructors_multiset(hint, ctor_names)
    if ctor_registry is not None:
        for ctor in ctor_names:
            check_deadline()
            ctor_registry.get_or_assign(ctor)
    base_dim = _dimension_from_keys(base_keys, registry)
    ctor_dim = _ctor_dimension_from_names(ctor_names, registry) if ctor_registry else FingerprintDimension()
    return Fingerprint(base=base_dim, ctor=ctor_dim)


def bundle_fingerprint(types: Iterable[str], registry: PrimeRegistry) -> int:
    check_deadline()
    product = 1
    for hint in types:
        check_deadline()
        key = canonical_type_key(hint)
        if not key:
            continue
        product *= registry.get_or_assign(key)
    return product


def bundle_fingerprint_setlike(types: Iterable[str], registry: PrimeRegistry) -> int:
    check_deadline()
    keys: set[str] = set()
    for hint in types:
        check_deadline()
        key = canonical_type_key(hint)
        if not key:
            continue
        keys.add(key)
    product = 1
    for key in sort_once(
        keys,
        source="bundle_fingerprint_setlike.keys",
        policy=OrderPolicy.SORT,
    ):
        check_deadline()
        product *= registry.get_or_assign(key)
    return product


def bundle_fingerprint_with_constructors(
    types: Iterable[str],
    registry: PrimeRegistry,
    ctor_registry: TypeConstructorRegistry,
) -> int:
    check_deadline()
    product = 1
    for hint in types:
        check_deadline()
        key = canonical_type_key_with_constructor(hint, ctor_registry)
        if not key:
            continue
        product *= registry.get_or_assign(key)
    return product


def fingerprint_bitmask(types: Iterable[str], registry: PrimeRegistry) -> int:
    check_deadline()
    mask = 0
    for hint in types:
        check_deadline()
        key = canonical_type_key(hint)
        if not key:
            continue
        registry.get_or_assign(key)
        bit = registry.bit_for(key)
        if bit is None:
            continue
        mask |= 1 << bit
    return mask


def fingerprint_hybrid(types: Iterable[str], registry: PrimeRegistry) -> tuple[int, int]:
    return bundle_fingerprint(types, registry), fingerprint_bitmask(types, registry)


def build_fingerprint_registry(
    spec: dict[str, JSONValue],
    *,
    registry_seed: object = None,
) -> tuple[PrimeRegistry, dict[Fingerprint, set[str]]]:
    check_deadline()
    registry = PrimeRegistry()
    registry.load_seed_payload(registry_seed)
    ctor_registry = TypeConstructorRegistry(registry)
    base_keys: set[str] = set()
    constructor_keys: set[str] = set()
    spec_entries: dict[str, list[str]] = {}
    for name, entry in spec.items():
        check_deadline()
        types = _normalize_type_list(entry)
        spec_entries[str(name)] = types
        if not types:
            continue
        for hint in types:
            check_deadline()
            atoms: list[str] = []
            _collect_base_atoms(hint, atoms)
            base_keys.update(atom for atom in atoms if atom)
            _collect_constructors(hint, constructor_keys)
    for constructor in sort_once(
        constructor_keys,
        source="build_fingerprint_registry.constructor_keys",
        policy=OrderPolicy.SORT,
    ):
        check_deadline()
        ctor_registry.get_or_assign(constructor)
    for key in sort_once(
        base_keys,
        source="build_fingerprint_registry.base_keys",
        policy=OrderPolicy.SORT,
    ):
        check_deadline()
        registry.get_or_assign(key)
    index: dict[Fingerprint, set[str]] = {}
    for name in sort_once(
        spec_entries,
        source="build_fingerprint_registry.spec_entries",
        policy=OrderPolicy.SORT,
    ):
        check_deadline()
        types = spec_entries[name]
        if not types:
            continue
        fingerprint = bundle_fingerprint_dimensional(
            types,
            registry,
            ctor_registry,
        )
        index.setdefault(fingerprint, set()).add(name)
    return registry, index


def fingerprint_to_type_keys_with_remainder(
    fingerprint: int,
    registry: PrimeRegistry,
) -> tuple[list[str], int]:
    check_deadline()
    remaining = fingerprint
    keys: list[str] = []
    if remaining <= 1:
        return keys, remaining
    for key, prime in sort_once(
        registry.primes.items(),
        source="fingerprint_to_type_keys_with_remainder.registry.primes",
        key=lambda item: item[1],
        policy=OrderPolicy.SORT,
    ):
        check_deadline()
        while remaining % prime == 0:
            check_deadline()
            keys.append(key)
            remaining //= prime
        if remaining == 1:
            break
    return keys, remaining


def fingerprint_to_type_keys(
    fingerprint: int,
    registry: PrimeRegistry,
    *,
    strict: bool = False,
) -> list[str]:
    keys, remaining = fingerprint_to_type_keys_with_remainder(
        fingerprint, registry
    )
    if strict and remaining not in (0, 1):
        raise ValueError(
            f"Fingerprint {fingerprint} contains primes not in registry."
        )
    return keys


def fingerprint_gcd(a: int, b: int) -> int:
    return math.gcd(a, b)


def fingerprint_lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // math.gcd(a, b)


def fingerprint_contains(container: int, part: int) -> bool:
    if part == 0:
        return False
    return container % part == 0


def fingerprint_symmetric_diff(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return a or b
    shared = math.gcd(a, b)
    return (a // shared) * (b // shared)
