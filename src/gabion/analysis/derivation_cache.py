# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, TypeVar

from gabion.analysis import aspf
from gabion.analysis.derivation_contract import (
    DerivationCacheStats,
    DerivationNodeId,
    DerivationOp,
    DerivationValue,
)
from gabion.analysis.derivation_graph import DerivationGraph, dependency_token
from gabion.order_contract import sort_once


ValueT = TypeVar("ValueT")

_DERIVATION_CACHE_SIZE_ENV = "GABION_DERIVATION_CACHE_MAX_ENTRIES"
_DEFAULT_DERIVATION_CACHE_SIZE = 4096


@dataclass
class DerivationCacheRuntime:
    graph: DerivationGraph = field(default_factory=DerivationGraph)
    max_entries: int = _DEFAULT_DERIVATION_CACHE_SIZE
    _values: OrderedDict[DerivationNodeId, DerivationValue[object]] = field(
        default_factory=OrderedDict
    )
    _known_nodes: dict[DerivationNodeId, None] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    regenerations: int = 0

    def derive(
        self,
        *,
        op: DerivationOp,
        structural_inputs: Mapping[str, object] | Iterable[object],
        compute_fn: Callable[[], ValueT],
        dependencies: Mapping[str, object] | None = None,
        params: Mapping[str, object] | None = None,
        source: str,
        on_cache_event: Callable[[str, DerivationNodeId], None] | None = None,
    ) -> ValueT:
        input_nodes = self._intern_inputs(
            structural_inputs,
            source=f"{source}.inputs",
        )
        dependency_value = dependency_token(
            dependencies or {},
            source=f"{source}.dependencies",
        )
        node_id = self.graph.intern_derived(
            op=op,
            input_nodes=input_nodes,
            params=params or {},
            dependencies=dependency_value,
            source=source,
        )
        was_known = node_id in self._known_nodes
        self._known_nodes[node_id] = None
        cached = self._values.get(node_id)
        if cached is not None:
            self.hits += 1
            self._values.move_to_end(node_id)
            if on_cache_event is not None:
                on_cache_event("cache:hit", node_id)
            return cached.value  # type: ignore[return-value]
        self.misses += 1
        if was_known:
            self.regenerations += 1
            if on_cache_event is not None:
                on_cache_event("cache:regen:start", node_id)
        else:
            if on_cache_event is not None:
                on_cache_event("cache:miss", node_id)
        value = compute_fn()
        entry: DerivationValue[object] = DerivationValue(
            node_id=node_id,
            value=value,
            lineage=input_nodes,
            generated_at_ns=time.monotonic_ns(),
        )
        self._values[node_id] = entry
        self._values.move_to_end(node_id)
        self._evict_if_needed()
        if was_known and on_cache_event is not None:
            on_cache_event("cache:regen:done", node_id)
        return value

    def materialize(self, node_id: DerivationNodeId) -> object | None:
        cached = self._values.get(node_id)
        if cached is None:
            return None
        self._values.move_to_end(node_id)
        return cached.value

    def invalidate(
        self,
        node_or_predicate: DerivationNodeId | Callable[[DerivationNodeId], bool],
    ) -> tuple[DerivationNodeId, ...]:
        if callable(node_or_predicate):
            matches = [
                node_id
                for node_id in self._values
                if node_or_predicate(node_id)
            ]
            invalidated = tuple(
                sort_once(
                    matches,
                    source="src/gabion/analysis/derivation_cache.py:invalidate.matches",
                    key=lambda value: value.sort_key(),
                )
            )
        else:
            invalidated = self.graph.invalidate(node_or_predicate)
        for node_id in invalidated:
            if node_id in self._values:
                del self._values[node_id]
                self.invalidations += 1
        return invalidated

    def stats(self) -> DerivationCacheStats:
        return DerivationCacheStats(
            hits=self.hits,
            misses=self.misses,
            evictions=self.evictions,
            invalidations=self.invalidations,
            regenerations=self.regenerations,
        )

    def to_payload(self) -> dict[str, object]:
        cached_nodes = sort_once(
            self._values,
            source="src/gabion/analysis/derivation_cache.py:to_payload.cached_nodes",
            key=lambda value: value.sort_key(),
        )
        return {
            "format_version": 1,
            "stats": {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "invalidations": self.invalidations,
                "regenerations": self.regenerations,
                "size": len(self._values),
                "max_entries": self.max_entries,
            },
            "cached_nodes": [
                _node_id_payload(node_id, source="derivation_cache.to_payload.cached_node")
                for node_id in cached_nodes
            ],
            "graph": self.graph.to_payload(),
        }

    def _intern_inputs(
        self,
        values: Mapping[str, object] | Iterable[object],
        *,
        source: str,
    ) -> tuple[DerivationNodeId, ...]:
        if isinstance(values, Mapping):
            labels = sort_once(
                values,
                source=f"{source}.labels",
                key=lambda value: value,
            )
            return tuple(
                self.graph.intern_input(
                    input_label=str(label),
                    value=values[label],
                    source=f"{source}.{label}",
                )
                for label in labels
            )
        return tuple(
            self.graph.intern_input(
                input_label=f"arg_{index}",
                value=value,
                source=f"{source}.arg_{index}",
            )
            for index, value in enumerate(values)
        )

    def _evict_if_needed(self) -> None:
        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)
            self.evictions += 1


_GLOBAL_RUNTIME: DerivationCacheRuntime | None = None


def _runtime_max_entries_from_env() -> int:
    raw = os.environ.get(_DERIVATION_CACHE_SIZE_ENV)
    if raw is None:
        return _DEFAULT_DERIVATION_CACHE_SIZE
    try:
        parsed = int(raw)
    except ValueError:
        return _DEFAULT_DERIVATION_CACHE_SIZE
    return max(1, parsed)


def get_global_derivation_cache() -> DerivationCacheRuntime:
    global _GLOBAL_RUNTIME
    if _GLOBAL_RUNTIME is None:
        _GLOBAL_RUNTIME = DerivationCacheRuntime(max_entries=_runtime_max_entries_from_env())
    return _GLOBAL_RUNTIME


def reset_global_derivation_cache(*, max_entries: int | None = None) -> DerivationCacheRuntime:
    global _GLOBAL_RUNTIME
    runtime = DerivationCacheRuntime(
        max_entries=(
            _runtime_max_entries_from_env()
            if max_entries is None
            else max(1, int(max_entries))
        )
    )
    _GLOBAL_RUNTIME = runtime
    return runtime


def _node_id_payload(
    node_id: DerivationNodeId,
    *,
    source: str,
) -> dict[str, object]:
    return {
        "kind": node_id.kind,
        "key": aspf.structural_key_json(
            aspf.structural_key_atom(node_id.key, source=source)
        ),
    }
