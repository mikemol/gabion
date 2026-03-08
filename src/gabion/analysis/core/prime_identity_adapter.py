from __future__ import annotations

from dataclasses import dataclass

from gabion.analysis.core.identity_namespace import namespace_key, raw_key
from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.analysis.foundation.identity_space import IdentityTokenLookup
from gabion.analysis.foundation.timeout_context import check_deadline


@dataclass
class PrimeIdentityAdapter:
    """Identity allocator adapter backed by PrimeRegistry."""

    registry: PrimeRegistry

    def get_or_assign(self, *, namespace: str, token: str) -> int:
        check_deadline()
        key = raw_key(str(namespace), str(token))
        return self.registry.get_or_assign(key)

    def token_for_id(self, *, namespace: str, atom_id: int) -> IdentityTokenLookup:
        check_deadline()
        raw = self.registry.key_for_prime(int(atom_id))
        if type(raw) is not str:
            return IdentityTokenLookup(is_present=False)
        decoded_namespace, decoded_token = namespace_key(raw)
        if decoded_namespace != str(namespace):
            return IdentityTokenLookup(is_present=False)
        return IdentityTokenLookup(is_present=True, token=decoded_token)

    def seed_payload(self) -> dict[str, object]:
        check_deadline()
        return self.registry.seed_payload()

    def load_seed_payload(self, payload: object) -> None:
        check_deadline()
        self.registry.load_seed_payload(payload)
