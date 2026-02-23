from gabion.analysis.artifact_ordering import (
    canonical_count_summary_items,
    canonical_doc_scope,
    canonical_field_display_parts,
    canonical_mapping_keys,
    canonical_protocol_specs,
    canonical_string_values,
)


# gabion:evidence E:call_footprint::tests/test_artifact_ordering.py::test_canonical_field_display_parts_skips_non_mapping_entries::artifact_ordering.py::gabion.analysis.artifact_ordering.canonical_field_display_parts
def test_canonical_field_display_parts_skips_non_mapping_entries() -> None:
    parts = canonical_field_display_parts(
        [
            {"name": "z", "type_hint": "int"},
            7,
            {"name": " ", "type_hint": "str"},
            {"name": "a"},
        ]
    )
    assert parts == ["z: int", "a: Any"]


# gabion:evidence E:call_footprint::tests/test_artifact_ordering.py::test_canonical_doc_scope_uses_default_for_empty_cleaned_scope::artifact_ordering.py::gabion.analysis.artifact_ordering.canonical_doc_scope
def test_canonical_doc_scope_uses_default_for_empty_cleaned_scope() -> None:
    assert canonical_doc_scope(["", " ", "\t"]) == ["repo", "artifacts"]
    assert canonical_doc_scope([" docs ", "repo"]) == ["docs", "repo"]


# gabion:evidence E:call_footprint::tests/test_artifact_ordering.py::test_canonical_protocol_specs_and_count_items_are_ordered::artifact_ordering.py::gabion.analysis.artifact_ordering.canonical_count_summary_items::artifact_ordering.py::gabion.analysis.artifact_ordering.canonical_protocol_specs
def test_canonical_protocol_specs_and_count_items_are_ordered() -> None:
    protocols = [
        {"name": "B", "tier": "2", "fields": [{"name": "x", "type_hint": "int"}], "evidence": ["b"]},
        {"name": "A", "tier": "1", "fields": [{"name": "x", "type_hint": "int"}], "evidence": ["a"]},
    ]
    ordered = canonical_protocol_specs(protocols)
    assert [str(spec.get("name")) for spec in ordered] == ["A", "B"]

    counts = canonical_count_summary_items({"z": 1, "a": 2, "m": 2})
    assert counts == [("a", 2), ("m", 2), ("z", 1)]


# gabion:evidence E:call_footprint::tests/test_artifact_ordering.py::test_canonical_string_values_and_mapping_keys_sort::artifact_ordering.py::gabion.analysis.artifact_ordering.canonical_mapping_keys::artifact_ordering.py::gabion.analysis.artifact_ordering.canonical_string_values
def test_canonical_string_values_and_mapping_keys_sort() -> None:
    assert canonical_string_values([3, "1", 2]) == ["1", "2", "3"]
    assert canonical_mapping_keys({"b": 1, "a": 2}) == ["a", "b"]
