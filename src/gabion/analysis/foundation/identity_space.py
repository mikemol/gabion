from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
from typing import Protocol

from gabion.analysis.core.identity_namespace import (
    EVIDENCE_KIND_NAMESPACE,
    SITE_KIND_NAMESPACE,
    SYNTH_NAMESPACE,
    TYPE_BASE_NAMESPACE,
    TYPE_CTOR_NAMESPACE,
    raw_key,
)
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.runtime import stable_encode


class IdentityNamespace(StrEnum):
    TYPE_BASE = TYPE_BASE_NAMESPACE
    TYPE_CTOR = TYPE_CTOR_NAMESPACE
    EVIDENCE_KIND = EVIDENCE_KIND_NAMESPACE
    SITE_KIND = SITE_KIND_NAMESPACE
    SYNTH = SYNTH_NAMESPACE
    SYMBOL = "symbol"
    PATH = "path"
    FEATURE = "feature"
    CACHE = "cache"


@dataclass(frozen=True)
class IdentityAllocationRecord:
    seq: int
    namespace: str
    token: str
    atom_id: int


@dataclass(frozen=True)
class IdentityPath:
    namespace: str
    atoms: tuple[int, ...]


@dataclass(frozen=True)
class IdentityProjection:
    basis_path: IdentityPath
    prime_product: int
    digest_alias: str
    witness: dict[str, object]


@dataclass(frozen=True)
class IdentityTokenLookup:
    is_present: bool
    token: str = ""


class IdentityAllocator(Protocol):
    def get_or_assign(self, *, namespace: str, token: str) -> int: ...

    def token_for_id(
        self,
        *,
        namespace: str,
        atom_id: int,
    ) -> IdentityTokenLookup: ...

    def seed_payload(self) -> dict[str, object]: ...

    def load_seed_payload(self, payload: object) -> None: ...


@dataclass
class GlobalIdentitySpace:
    allocator: IdentityAllocator
    _atom_by_token: dict[str, dict[str, int]] = field(default_factory=dict)
    _token_by_atom: dict[str, dict[int, str]] = field(default_factory=dict)
    _allocation_records: list[IdentityAllocationRecord] = field(default_factory=list)
    _next_seq: int = 1

    def intern_atom(self, *, namespace: str, token: str) -> int:
        check_deadline()
        namespace_text = self._normalize_namespace(namespace)
        token_text = str(token)
        if not token_text:
            raise ValueError("Identity token must be non-empty.")
        known = self._atom_by_token.setdefault(namespace_text, {})
        existing = known.get(token_text)
        if existing is not None:
            return existing
        atom_id = int(
            self.allocator.get_or_assign(namespace=namespace_text, token=token_text)
        )
        return self.register_atom(
            namespace=namespace_text,
            token=token_text,
            atom_id=atom_id,
            record_allocation=True,
        )

    def register_atom(
        self,
        *,
        namespace: str,
        token: str,
        atom_id: int,
        record_allocation: bool = True,
    ) -> int:
        check_deadline()
        namespace_text = self._normalize_namespace(namespace)
        token_text = str(token)
        atom_value = int(atom_id)
        if not token_text:
            raise ValueError("Identity token must be non-empty.")
        if atom_value <= 0:
            raise ValueError("Identity atom_id must be a positive integer.")
        known_tokens = self._atom_by_token.setdefault(namespace_text, {})
        known_atoms = self._token_by_atom.setdefault(namespace_text, {})
        existing_atom = known_tokens.get(token_text)
        if existing_atom is not None:
            if int(existing_atom) != atom_value:
                raise ValueError(
                    "Identity token is already interned to a different atom_id."
                )
            return int(existing_atom)
        existing_token = known_atoms.get(atom_value)
        if existing_token is not None and existing_token != token_text:
            raise ValueError(
                "Identity atom_id is already interned to a different token."
            )
        known_tokens[token_text] = atom_value
        known_atoms[atom_value] = token_text
        if record_allocation:
            self._allocation_records.append(
                IdentityAllocationRecord(
                    seq=self._next_seq,
                    namespace=namespace_text,
                    token=token_text,
                    atom_id=atom_value,
                )
            )
            self._next_seq += 1
        return atom_value

    def intern_path(self, *, namespace: str, tokens: object) -> IdentityPath:
        check_deadline()
        namespace_text = self._normalize_namespace(namespace)
        atoms = tuple(
            self.intern_atom(namespace=namespace_text, token=str(token))
            for token in tokens if str(token)
        )
        return IdentityPath(namespace=namespace_text, atoms=atoms)

    def project(self, *, path: IdentityPath) -> IdentityProjection:
        check_deadline()
        prime_product = 1
        for atom in path.atoms:
            check_deadline()
            prime_product *= int(atom)
        digest_payload = stable_encode.stable_compact_text(
            {
                "identity_layer": "ordered_basis_path",
                "namespace": path.namespace,
                "atoms": list(path.atoms),
            }
        ).encode("utf-8")
        digest_alias = f"ids:sha1:{hashlib.sha1(digest_payload).hexdigest()}"
        witness = {
            "identity_layer": "ordered_basis_path",
            "canonical": {
                "namespace": path.namespace,
                "basis_path": list(path.atoms),
                "order_sensitive": True,
            },
            "derived_aliases": {
                "prime_product": {
                    "value": prime_product,
                    "canonical": False,
                    "projection": "commutative_prime_product",
                },
                "digest_alias": {
                    "value": digest_alias,
                    "canonical": False,
                    "projection": "ordered_basis_digest",
                },
            },
            "commutation_witness": {
                "carrier_relation": "ordered_basis_vs_commutative_scalar",
                "order_erased_by_prime_product": True,
                "order_preserved_by_basis_path": True,
            },
        }
        return IdentityProjection(
            basis_path=path,
            prime_product=prime_product,
            digest_alias=digest_alias,
            witness=witness,
        )

    def token_for_atom(
        self,
        *,
        namespace: str,
        atom_id: int,
    ) -> IdentityTokenLookup:
        check_deadline()
        namespace_text = self._normalize_namespace(namespace)
        existing = self._token_by_atom.get(namespace_text, {}).get(int(atom_id))
        if existing is not None:
            return IdentityTokenLookup(is_present=True, token=existing)
        decoded = self.allocator.token_for_id(
            namespace=namespace_text,
            atom_id=int(atom_id),
        )
        if not decoded.is_present:
            return IdentityTokenLookup(is_present=False)
        self.register_atom(
            namespace=namespace_text,
            token=decoded.token,
            atom_id=int(atom_id),
            record_allocation=False,
        )
        return decoded

    def seed_payload(self) -> dict[str, object]:
        check_deadline()
        return self.allocator.seed_payload()

    def load_seed_payload(self, payload: object) -> None:
        check_deadline()
        self.allocator.load_seed_payload(payload)

    def allocation_records(self) -> tuple[IdentityAllocationRecord, ...]:
        check_deadline()
        return tuple(self._allocation_records)

    def allocation_records_payload(self) -> list[dict[str, object]]:
        check_deadline()
        return [
            {
                "seq": record.seq,
                "namespace": record.namespace,
                "token": record.token,
                "atom_id": record.atom_id,
            }
            for record in self._allocation_records
        ]

    @staticmethod
    def _normalize_namespace(namespace: str) -> str:
        check_deadline()
        namespace_text = str(namespace).strip()
        if not namespace_text:
            raise ValueError("Identity namespace must be non-empty.")
        # Ensure all interners agree on namespace -> raw-key semantics.
        raw_key(namespace_text, "")
        return namespace_text
