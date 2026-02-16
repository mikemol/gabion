from __future__ import annotations

from pathlib import Path

from gabion.analysis.resume_codec import (
    allowed_path_lookup,
    int_str_pairs_from_sequence,
    int_tuple4_or_none,
    iter_valid_key_entries,
    load_allowed_paths_from_sequence,
    load_resume_map,
    mapping_payload,
    mapping_sections,
    mapping_or_empty,
    mapping_or_none,
    payload_with_format,
    payload_with_phase,
    sequence_or_none,
    str_list_from_sequence,
    str_map_from_mapping,
    str_pair_set_from_sequence,
    str_set_from_sequence,
    str_tuple_from_sequence,
)


def test_mapping_helpers() -> None:
    payload = {"a": 1}
    assert mapping_or_none(payload) == payload
    assert mapping_or_none([1, 2, 3]) is None
    assert mapping_payload(payload) == payload
    assert mapping_payload([1, 2, 3]) is None
    assert mapping_or_empty(payload) == payload
    assert mapping_or_empty([1, 2, 3]) == {}


def test_payload_filters() -> None:
    payload = {"phase": "scan", "format_version": 2}
    assert payload_with_phase(payload, phase="scan") == payload
    assert payload_with_phase(payload, phase="other") is None
    assert payload_with_phase(None, phase="scan") is None
    assert payload_with_format(payload, format_version=2) == payload
    assert payload_with_format(payload, format_version=1) is None
    assert payload_with_format(None, format_version=2) is None


def test_mapping_sections() -> None:
    payload = {"a": {"x": 1}, "b": {"y": 2}}
    assert mapping_sections(payload, section_keys=("a", "b")) == (payload["a"], payload["b"])
    assert mapping_sections(payload, section_keys=("a", "missing")) is None
    assert mapping_sections({"a": []}, section_keys=("a",)) is None


def test_sequence_or_none_variants() -> None:
    assert sequence_or_none(123) is None
    assert sequence_or_none("abc") is None
    assert sequence_or_none(b"abc") is None
    assert sequence_or_none("abc", allow_str=True) == "abc"
    assert sequence_or_none([1, 2, 3]) == [1, 2, 3]


def test_string_collection_helpers() -> None:
    values: object = ["a", 1, "b", None]
    assert str_list_from_sequence(values) == ["a", "b"]
    assert str_tuple_from_sequence(values) == ("a", "b")
    assert str_set_from_sequence(values) == {"a", "b"}
    assert str_list_from_sequence(42) == []


def test_str_map_from_mapping_filters_invalid_entries() -> None:
    payload: object = {"a": "x", "b": 1, 3: "y"}  # type: ignore[dict-item]
    assert str_map_from_mapping(payload) == {"a": "x"}
    assert str_map_from_mapping([("a", "x")]) == {}


def test_int_tuple4_or_none() -> None:
    assert int_tuple4_or_none("abcd") is None
    assert int_tuple4_or_none([1, 2, 3]) is None
    assert int_tuple4_or_none([1, 2, "x", 4]) is None
    assert int_tuple4_or_none([1, 2, 3, 4]) == (1, 2, 3, 4)


def test_int_str_pairs_from_sequence_filters_invalid_entries() -> None:
    payload: object = [
        1,
        [1],
        [1, 2],
        ["x", "name"],
        [2, "good"],
    ]
    assert int_str_pairs_from_sequence(payload) == [(2, "good")]
    assert int_str_pairs_from_sequence(None) == []


def test_str_pair_set_from_sequence_filters_invalid_entries() -> None:
    payload: object = [
        1,
        ["a"],
        ["a", 1],
        ["a", "b"],
        ["a", "b"],
    ]
    assert str_pair_set_from_sequence(payload) == {("a", "b")}
    assert str_pair_set_from_sequence(None) == set()


def test_iter_valid_key_entries_filters_keys() -> None:
    payload = {"good": 1, "skip": 2, 3: 4}  # type: ignore[dict-item]
    entries = list(
        iter_valid_key_entries(
            payload=payload,  # type: ignore[arg-type]
            valid_keys={"good"},
        )
    )
    assert entries == [("good", 1)]


def test_load_resume_map_filters_invalid_and_none_parses() -> None:
    payload = {"keep": "1", "drop": "2", 3: "3"}  # type: ignore[dict-item]
    result = load_resume_map(
        payload=payload,  # type: ignore[arg-type]
        valid_keys={"keep", "drop"},
        parser=lambda value: int(value) if value == "1" else None,
    )
    assert result == {"keep": 1}


def test_allowed_path_helpers(tmp_path: Path) -> None:
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    lookup = allowed_path_lookup(
        [first, second],
        key_fn=str,
    )
    assert lookup == {str(first): first, str(second): second}
    loaded = load_allowed_paths_from_sequence(
        [str(first), str(second), str(first), "missing", 1],
        allowed_paths=lookup,
    )
    assert loaded == [first, second]
    assert load_allowed_paths_from_sequence(None, allowed_paths=lookup) == []
