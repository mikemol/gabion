from __future__ import annotations

from pathlib import Path

from gabion.analysis.foundation.resume_codec import (
    allowed_path_lookup, int_str_pairs_from_sequence, iter_int_tuple4_from_sequence, iter_valid_key_entries, load_allowed_paths_from_sequence, load_resume_map, mapping_payload, mapping_sections, mapping_default_empty, mapping_optional, payload_with_format, payload_with_phase, sequence_optional, str_list_from_sequence, str_map_from_mapping, str_pair_set_from_sequence, str_set_from_sequence, str_tuple_from_sequence)


# gabion:evidence E:call_footprint::tests/test_resume_codec.py::test_mapping_helpers::resume_codec.py::gabion.analysis.resume_codec.mapping_default_empty::resume_codec.py::gabion.analysis.resume_codec.mapping_optional::resume_codec.py::gabion.analysis.resume_codec.mapping_payload
# gabion:behavior primary=desired
def test_mapping_helpers() -> None:
    payload = {"a": 1}
    assert mapping_optional(payload) == payload
    assert mapping_optional([1, 2, 3]) is None
    assert mapping_payload(payload) == payload
    assert mapping_payload([1, 2, 3]) is None
    assert mapping_default_empty(payload) == payload
    assert mapping_default_empty([1, 2, 3]) == {}


# gabion:evidence E:call_footprint::tests/test_resume_codec.py::test_payload_filters::resume_codec.py::gabion.analysis.resume_codec.payload_with_format::resume_codec.py::gabion.analysis.resume_codec.payload_with_phase
# gabion:behavior primary=desired
def test_payload_filters() -> None:
    payload = {"phase": "scan", "format_version": 2}
    assert payload_with_phase(payload, phase="scan") == payload
    assert payload_with_phase(payload, phase="other") is None
    assert payload_with_phase(None, phase="scan") is None
    assert payload_with_format(payload, format_version=2) == payload
    assert payload_with_format(payload, format_version=1) is None
    assert payload_with_format(None, format_version=2) is None


# gabion:evidence E:call_footprint::tests/test_resume_codec.py::test_mapping_sections::resume_codec.py::gabion.analysis.resume_codec.mapping_sections
# gabion:behavior primary=desired
def test_mapping_sections() -> None:
    payload = {"a": {"x": 1}, "b": {"y": 2}}
    assert mapping_sections(payload, section_keys=("a", "b")) == (payload["a"], payload["b"])
    assert mapping_sections(payload, section_keys=("a", "missing")) is None
    assert mapping_sections({"a": []}, section_keys=("a",)) is None


# gabion:evidence E:call_footprint::tests/test_resume_codec.py::test_sequence_or_none_variants::resume_codec.py::gabion.analysis.resume_codec.sequence_optional
# gabion:behavior primary=verboten facets=none
def test_sequence_or_none_variants() -> None:
    assert sequence_optional(123) is None
    assert sequence_optional("abc") is None
    assert sequence_optional(b"abc") is None
    assert sequence_optional("abc", allow_str=True) == "abc"
    assert sequence_optional([1, 2, 3]) == [1, 2, 3]


# gabion:evidence E:call_footprint::tests/test_resume_codec.py::test_string_collection_helpers::resume_codec.py::gabion.analysis.resume_codec.str_list_from_sequence::resume_codec.py::gabion.analysis.resume_codec.str_set_from_sequence::resume_codec.py::gabion.analysis.resume_codec.str_tuple_from_sequence
# gabion:behavior primary=desired
def test_string_collection_helpers() -> None:
    values: object = ["a", 1, "b", None]
    assert list(str_list_from_sequence(values)) == ["a", "b"]
    assert tuple(str_tuple_from_sequence(values)) == ("a", "b")
    assert set(str_set_from_sequence(values)) == {"a", "b"}
    assert list(str_list_from_sequence(42)) == []


# gabion:evidence E:call_footprint::tests/test_resume_codec.py::test_str_map_from_mapping_filters_invalid_entries::resume_codec.py::gabion.analysis.resume_codec.str_map_from_mapping
# gabion:behavior primary=verboten facets=invalid
def test_str_map_from_mapping_filters_invalid_entries() -> None:
    payload: object = {"a": "x", "b": 1, 3: "y"}  # type: ignore[dict-item]
    assert dict(str_map_from_mapping(payload)) == {"a": "x"}
    assert dict(str_map_from_mapping([("a", "x")])) == {}


# gabion:evidence E:call_footprint::tests/test_resume_codec.py::test_iter_int_tuple4_from_sequence_filters_invalid_entries::resume_codec.py::gabion.analysis.resume_codec.iter_int_tuple4_from_sequence
# gabion:behavior primary=verboten facets=none
def test_iter_int_tuple4_from_sequence_filters_invalid_entries() -> None:
    assert list(iter_int_tuple4_from_sequence("abcd")) == []
    assert list(iter_int_tuple4_from_sequence([[1, 2, 3]])) == []
    assert list(iter_int_tuple4_from_sequence([[1, 2, "x", 4]])) == []
    assert list(iter_int_tuple4_from_sequence([[1, 2, 3, 4]])) == [(1, 2, 3, 4)]


# gabion:evidence E:call_footprint::tests/test_resume_codec.py::test_int_str_pairs_from_sequence_filters_invalid_entries::resume_codec.py::gabion.analysis.resume_codec.int_str_pairs_from_sequence
# gabion:behavior primary=verboten facets=invalid
def test_int_str_pairs_from_sequence_filters_invalid_entries() -> None:
    payload: object = [
        1,
        [1],
        [1, 2],
        ["x", "name"],
        [2, "good"],
    ]
    assert list(int_str_pairs_from_sequence(payload)) == [(2, "good")]
    assert list(int_str_pairs_from_sequence(None)) == []


# gabion:evidence E:call_footprint::tests/test_resume_codec.py::test_str_pair_set_from_sequence_filters_invalid_entries::resume_codec.py::gabion.analysis.resume_codec.str_pair_set_from_sequence
# gabion:behavior primary=verboten facets=invalid
def test_str_pair_set_from_sequence_filters_invalid_entries() -> None:
    payload: object = [
        1,
        ["a"],
        ["a", 1],
        ["a", "b"],
        ["a", "b"],
    ]
    assert set(str_pair_set_from_sequence(payload)) == {("a", "b")}
    assert set(str_pair_set_from_sequence(None)) == set()


# gabion:evidence E:call_footprint::tests/test_resume_codec.py::test_iter_valid_key_entries_filters_keys::resume_codec.py::gabion.analysis.resume_codec.iter_valid_key_entries
# gabion:behavior primary=desired
def test_iter_valid_key_entries_filters_keys() -> None:
    payload = {"good": 1, "skip": 2, 3: 4}  # type: ignore[dict-item]
    entries = list(
        iter_valid_key_entries(
            payload=payload,  # type: ignore[arg-type]
            valid_keys={"good"},
        )
    )
    assert entries == [("good", 1)]


# gabion:evidence E:call_footprint::tests/test_resume_codec.py::test_load_resume_map_filters_invalid_and_none_parses::resume_codec.py::gabion.analysis.resume_codec.load_resume_map
# gabion:behavior primary=verboten facets=invalid,none
def test_load_resume_map_filters_invalid_and_none_parses() -> None:
    payload = {"keep": "1", "drop": "2", 3: "3"}  # type: ignore[dict-item]

    def _parse_int_one(value: object):
        if value == "1":
            yield int(value)

    result = dict(
        load_resume_map(
        payload=payload,  # type: ignore[arg-type]
        valid_keys={"keep", "drop"},
        parser=_parse_int_one,
        )
    )
    assert result == {"keep": 1}


# gabion:evidence E:call_footprint::tests/test_resume_codec.py::test_allowed_path_helpers::resume_codec.py::gabion.analysis.resume_codec.allowed_path_lookup::resume_codec.py::gabion.analysis.resume_codec.load_allowed_paths_from_sequence
# gabion:behavior primary=desired
def test_allowed_path_helpers(tmp_path: Path) -> None:
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    lookup = dict(
        allowed_path_lookup(
            [first, second],
            key_fn=str,
        )
    )
    assert lookup == {str(first): first, str(second): second}
    loaded = list(
        load_allowed_paths_from_sequence(
            [str(first), str(second), str(first), "missing", 1],
            allowed_paths=lookup,
        )
    )
    assert loaded == [first, second]
    assert list(load_allowed_paths_from_sequence(None, allowed_paths=lookup)) == []
