from __future__ import annotations

from gabion.analysis.dataflow_audit import (
    InvariantProposition,
    _build_property_hook_callable_index,
    _deserialize_invariants_for_resume,
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


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit.generate_property_hook_manifest
def test_property_hook_ids_are_stable_across_runs() -> None:
    invariants = [
        _sample_invariant(invariant_id="inv:001", terms=("a", "b")),
        _sample_invariant(invariant_id="inv:002", terms=("a.length", "b.length")),
    ]
    first = generate_property_hook_manifest(invariants)
    second = generate_property_hook_manifest(invariants)
    assert first == second


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit.generate_property_hook_manifest
def test_property_hook_manifest_skips_low_confidence_invariants() -> None:
    invariants = [
        _sample_invariant(invariant_id="inv:low", terms=("a", "b"), confidence=0.2),
    ]
    payload = generate_property_hook_manifest(invariants, min_confidence=0.7)
    assert payload["hooks"] == []
    assert payload["callable_index"] == []


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit.generate_property_hook_manifest
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


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit.InvariantProposition.as_dict
def test_invariant_proposition_as_dict_emits_optional_metadata() -> None:
    proposition = InvariantProposition(
        form="Equal",
        terms=("a", "b"),
        invariant_id="inv:123",
        confidence=0.6,
        evidence_keys=("E:foo", "E:bar"),
    )
    payload = proposition.as_dict()
    assert payload["invariant_id"] == "inv:123"
    assert payload["confidence"] == 0.6
    assert payload["evidence_keys"] == ["E:foo", "E:bar"]


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit.generate_property_hook_manifest
def test_property_hook_manifest_hypothesis_templates_and_scope_filtering() -> None:
    invariants = [
        _sample_invariant(invariant_id="inv:good", terms=("a", "b")),
        InvariantProposition(form="Equal", terms=("a", "b"), scope="pkg/mod.py", source="assert"),
        InvariantProposition(form="Equal", terms=("a", "b"), scope="pkg/mod.py:", source="assert"),
    ]
    payload = generate_property_hook_manifest(invariants, emit_hypothesis_templates=True)
    hooks = payload["hooks"]
    assert hooks
    assert all("hypothesis_template" in hook for hook in hooks)
    assert all("def test_" in str(hook["hypothesis_template"]) for hook in hooks)
    valid_hook_ids = [
        str(hook["hook_id"])
        for hook in hooks
        if isinstance(hook.get("callable"), dict)
        and str(hook["callable"].get("path", "")) == "pkg/mod.py"
        and str(hook["callable"].get("qual", "")) == "target"
    ]
    assert payload["callable_index"] == [{"scope": "pkg/mod.py:target", "hook_ids": valid_hook_ids}]


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._build_property_hook_callable_index
def test_property_hook_callable_index_skips_invalid_hook_shapes() -> None:
    callable_index = _build_property_hook_callable_index(
        [
            1,
            {"hook_id": "bad", "callable": "not-a-mapping"},
            {"hook_id": "empty", "callable": {"path": "pkg/mod.py", "qual": ""}},
            {"hook_id": "good", "callable": {"path": "pkg/mod.py", "qual": "target"}},
        ]
    )
    assert callable_index == [{"scope": "pkg/mod.py:target", "hook_ids": ["good"]}]


# gabion:evidence E:function_site::dataflow_audit.py::gabion.analysis.dataflow_audit._deserialize_invariants_for_resume
def test_deserialize_invariants_for_resume_normalizes_evidence_keys() -> None:
    invariants = _deserialize_invariants_for_resume(
        [
            {
                "form": "Equal",
                "terms": ["a", "b"],
                "scope": "pkg/mod.py:target",
                "source": "resume",
                "invariant_id": "inv:resume",
                "confidence": 0.5,
                "evidence_keys": ["E:first", "  ", 2, ""],
            }
        ]
    )
    assert len(invariants) == 1
    assert invariants[0].evidence_keys == ("E:first", "2")
