from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
import math


def _split_top_level(value: str, sep: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in value:
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
    for prefix in ("typing.", "builtins."):
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
    raw = hint.strip()
    if not raw:
        return ""
    union_parts = _split_top_level(raw, "|")
    if len(union_parts) > 1:
        normalized = sorted(
            part for part in (canonical_type_key(p) for p in union_parts) if part
        )
        return f"Union[{', '.join(normalized)}]"
    if raw.startswith("Optional[") and raw.endswith("]"):
        inner = raw[len("Optional[") : -1]
        parts = _split_top_level(inner, ",")
        normalized = [canonical_type_key(p) for p in parts]
        normalized.append("None")
        normalized = sorted({item for item in normalized if item})
        return f"Union[{', '.join(normalized)}]"
    if raw.startswith("Union[") and raw.endswith("]"):
        inner = raw[len("Union[") : -1]
        parts = _split_top_level(inner, ",")
        normalized = sorted(
            part for part in (canonical_type_key(p) for p in parts) if part
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
    if value < 2:
        return False
    if value in (2, 3):
        return True
    if value % 2 == 0 or value % 3 == 0:
        return False
    step = 5
    while step * step <= value:
        if value % step == 0 or value % (step + 2) == 0:
            return False
        step += 6
    return True


def _next_prime(start: int) -> int:
    candidate = max(start, 2)
    while not _is_prime(candidate):
        candidate += 1
    return candidate


@dataclass
class PrimeRegistry:
    primes: dict[str, int] = field(default_factory=dict)
    next_candidate: int = 2

    def get_or_assign(self, key: str) -> int:
        if not key:
            raise ValueError("Type key must be non-empty.")
        existing = self.primes.get(key)
        if existing is not None:
            return existing
        prime = _next_prime(self.next_candidate)
        self.primes[key] = prime
        self.next_candidate = prime + 1
        return prime

    def prime_for(self, key: str) -> int | None:
        return self.primes.get(key)

    def key_for_prime(self, prime: int) -> str | None:
        for key, value in self.primes.items():
            if value == prime:
                return key
        return None


def _normalize_type_list(value: object) -> list[str]:
    items: list[str] = []
    if value is None:
        return items
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, str):
                items.extend([part.strip() for part in item.split(",") if part.strip()])
    return [item for item in items if item]


def bundle_fingerprint(types: Iterable[str], registry: PrimeRegistry) -> int:
    product = 1
    for hint in types:
        key = canonical_type_key(hint)
        if not key:
            continue
        product *= registry.get_or_assign(key)
    return product


def build_fingerprint_registry(
    spec: dict[str, object],
) -> tuple[PrimeRegistry, dict[int, set[str]]]:
    registry = PrimeRegistry()
    index: dict[int, set[str]] = {}
    for name, entry in spec.items():
        types = _normalize_type_list(entry)
        if not types:
            continue
        fingerprint = bundle_fingerprint(types, registry)
        index.setdefault(fingerprint, set()).add(str(name))
    return registry, index


def fingerprint_to_type_keys(
    fingerprint: int,
    registry: PrimeRegistry,
) -> list[str]:
    remaining = fingerprint
    keys: list[str] = []
    if remaining <= 1:
        return keys
    for key, prime in sorted(registry.primes.items(), key=lambda item: item[1]):
        while remaining % prime == 0:
            keys.append(key)
            remaining //= prime
        if remaining == 1:
            break
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
