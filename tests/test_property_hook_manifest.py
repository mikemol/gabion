from __future__ import annotations

from gabion.analysis.dataflow_audit import (
    InvariantProposition,
    generate_property_hook_manifest,
)


def _sample_invariant(*, invariant_id: str, terms: tuple[str, ...], confidence: float = 1.0) -> InvariantProposition:
    return InvariantProposition(
        form="Equal",
        terms=terms,
        scope="pkg/mod.py:target",
        source="assert",
        invariant_id=invariant_id,
        confidence=confidence,
        evidence_keys=(f"E:invariant::{invariant_id}",),
    )


def test_property_hook_ids_are_stable_across_runs() -> None:
    invariants = [
        _sample_invariant(invariant_id="inv:001", terms=("a", "b")),
        _sample_invariant(invariant_id="inv:002", terms=("a.length", "b.length")),
    ]
    first = generate_property_hook_manifest(invariants)
    second = generate_property_hook_manifest(invariants)
    assert first == second


def test_property_hook_manifest_skips_low_confidence_invariants() -> None:
    invariants = [
        _sample_invariant(invariant_id="inv:low", terms=("a", "b"), confidence=0.2),
    ]
    payload = generate_property_hook_manifest(invariants, min_confidence=0.7)
    assert payload["hooks"] == []
    assert payload["callable_index"] == []


def test_property_hook_manifest_maps_multiple_invariants_to_one_callable() -> None:
    invariants = [
        _sample_invariant(invariant_id="inv:001", terms=("a", "b")),
        _sample_invariant(invariant_id="inv:002", terms=("a.length", "b.length")),
    ]
    payload = generate_property_hook_manifest(invariants)
    hooks = payload["hooks"]
    assert len(hooks) == 2
    assert all(hook["source_invariant_evidence_keys"] for hook in hooks)
    callable_index = payload["callable_index"]
    assert callable_index == [
        {
            "scope": "pkg/mod.py:target",
            "hook_ids": sorted([hook["hook_id"] for hook in hooks]),
        }
    ]
