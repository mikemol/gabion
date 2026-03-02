from __future__ import annotations

import json
from pathlib import Path

from gabion.analysis import evidence_keys, semantic_coverage_map, test_evidence


def _write_test_module(root: Path) -> Path:
    path = root / "tests" / "test_semantic_case.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
# gabion:evidence INV.alpha

def test_alpha() -> None:
    assert True

# gabion:evidence INV.beta

def test_beta() -> None:
    assert True
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


# gabion:evidence E:function_site::semantic_coverage_map.py::gabion.analysis.semantic_coverage_map.build_semantic_coverage_payload E:decision_surface/direct::semantic_coverage_map.py::gabion.analysis.semantic_coverage_map.build_semantic_coverage_payload::stale_39604d8ff929_03566325
def test_semantic_coverage_payload_is_deterministic(tmp_path: Path) -> None:
    test_path = _write_test_module(tmp_path)
    evidence_payload = test_evidence.build_test_evidence_payload(
        [test_path],
        root=tmp_path,
    )
    evidence_path = tmp_path / "out" / "test_evidence.json"
    test_evidence.write_test_evidence(evidence_payload, evidence_path)
    mapping_path = tmp_path / "out" / "semantic_coverage_mapping.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {"obligation": "lemma.beta", "obligation_kind": "lemma", "evidence": "INV.beta"},
                    {"obligation": "invariant.alpha", "obligation_kind": "invariant", "evidence": "INV.alpha"},
                ],
            }
        ),
        encoding="utf-8",
    )

    first = semantic_coverage_map.build_semantic_coverage_payload(
        [test_path],
        root=tmp_path,
        mapping_path=mapping_path,
        evidence_path=evidence_path,
    )
    second = semantic_coverage_map.build_semantic_coverage_payload(
        [test_path],
        root=tmp_path,
        mapping_path=mapping_path,
        evidence_path=evidence_path,
    )

    assert first == second
    assert [entry["obligation"] for entry in first["mapped_obligations"]] == [
        "invariant.alpha",
        "lemma.beta",
    ]


# gabion:evidence E:function_site::semantic_coverage_map.py::gabion.analysis.semantic_coverage_map.build_semantic_coverage_payload E:decision_surface/direct::semantic_coverage_map.py::gabion.analysis.semantic_coverage_map.build_semantic_coverage_payload::stale_fbc33e8075bc
def test_semantic_coverage_reports_unmapped_dead_and_duplicate_entries(tmp_path: Path) -> None:
    test_path = _write_test_module(tmp_path)
    evidence_payload = test_evidence.build_test_evidence_payload(
        [test_path],
        root=tmp_path,
    )
    evidence_path = tmp_path / "out" / "test_evidence.json"
    test_evidence.write_test_evidence(evidence_payload, evidence_path)
    mapping_path = tmp_path / "out" / "semantic_coverage_mapping.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {"obligation": "invariant.alpha", "obligation_kind": "invariant", "evidence": "INV.alpha"},
                    {"obligation": "invariant.alpha", "obligation_kind": "invariant", "evidence": "INV.alpha"},
                    {"obligation": "invariant.dead", "obligation_kind": "invariant", "evidence": "INV.missing"},
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = semantic_coverage_map.build_semantic_coverage_payload(
        [test_path],
        root=tmp_path,
        mapping_path=mapping_path,
        evidence_path=evidence_path,
    )

    assert [entry["obligation"] for entry in payload["unmapped_obligations"]] == [
        "invariant.dead"
    ]
    assert len(payload["dead_mapping_entries"]) == 1
    assert payload["duplicate_mapping_entries"][0]["count"] == 2


# gabion:evidence E:function_site::semantic_coverage_map.py::gabion.analysis.semantic_coverage_map.render_markdown E:function_site::semantic_coverage_map.py::gabion.analysis.semantic_coverage_map.write_semantic_coverage
def test_semantic_coverage_markdown_and_write_emit_sections(tmp_path: Path) -> None:
    payload = {
        "summary": {"mapping_entries": 2},
        "mapped_obligations": [{"obligation": "a"}],
        "unmapped_obligations": [{"obligation": "b"}],
        "dead_mapping_entries": [{"obligation": "c"}],
        "duplicate_mapping_entries": [{"obligation": "d", "count": 2}],
    }
    markdown = semantic_coverage_map.render_markdown(payload)
    assert "Mapped obligations" in markdown
    assert "Unmapped obligations" in markdown
    assert "Dead mapping entries" in markdown
    assert "Duplicate mapping entries" in markdown

    output_path = tmp_path / "artifacts" / "out" / "semantic_coverage_map.json"
    semantic_coverage_map.write_semantic_coverage(payload, output_path=output_path)
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8"))["summary"]["mapping_entries"] == 2


# gabion:evidence E:function_site::semantic_coverage_map.py::gabion.analysis.semantic_coverage_map.load_mapping_entries
def test_semantic_coverage_load_mapping_entries_edge_shapes(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"
    assert semantic_coverage_map.load_mapping_entries(missing_path) == []

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    assert semantic_coverage_map.load_mapping_entries(invalid_json) == []

    non_mapping = tmp_path / "non_mapping.json"
    non_mapping.write_text("[]", encoding="utf-8")
    assert semantic_coverage_map.load_mapping_entries(non_mapping) == []

    non_list_entries = tmp_path / "non_list_entries.json"
    non_list_entries.write_text(json.dumps({"entries": {}}), encoding="utf-8")
    assert semantic_coverage_map.load_mapping_entries(non_list_entries) == []

    mixed_entries = tmp_path / "mixed_entries.json"
    mixed_entries.write_text(
        json.dumps(
            {
                "entries": [
                    1,
                    {"obligation": "", "evidence": "INV.blank"},
                    {"obligation": "inv.alpha", "evidence": "", "obligation_kind": "invariant"},
                    {"obligation": "inv.beta", "evidence": "INV.beta", "obligation_kind": "lemma"},
                ]
            }
        ),
        encoding="utf-8",
    )
    entries = semantic_coverage_map.load_mapping_entries(mixed_entries)
    assert len(entries) == 1
    assert entries[0].obligation == "inv.beta"
    assert entries[0].obligation_kind == "lemma"


# gabion:evidence E:function_site::semantic_coverage_map.py::gabion.analysis.semantic_coverage_map._annotation_index E:function_site::semantic_coverage_map.py::gabion.analysis.semantic_coverage_map._artifact_evidence_index
def test_semantic_coverage_indexes_handle_invalid_and_fallback_records(tmp_path: Path) -> None:
    tags = [
        test_evidence.TestEvidenceTag(
            test_id="tests/test_semantic_case.py::test_alpha",
            path="tests/test_semantic_case.py",
            line=1,
            tags=("   ", "INV.alpha"),
        )
    ]
    annotation_index = semantic_coverage_map._annotation_index(tags)
    assert annotation_index
    assert list(annotation_index.values()) == [{"tests/test_semantic_case.py::test_alpha"}]

    missing_path = tmp_path / "missing_evidence.json"
    assert semantic_coverage_map._artifact_evidence_index(missing_path) == set()

    non_mapping = tmp_path / "non_mapping_evidence.json"
    non_mapping.write_text("[]", encoding="utf-8")
    assert semantic_coverage_map._artifact_evidence_index(non_mapping) == set()

    non_list = tmp_path / "non_list_evidence.json"
    non_list.write_text(json.dumps({"evidence_index": {}}), encoding="utf-8")
    assert semantic_coverage_map._artifact_evidence_index(non_list) == set()

    records_path = tmp_path / "records_evidence.json"
    records_path.write_text(
        json.dumps(
            {
                "evidence_index": [
                    1,
                    {"display": ""},
                    {"display": "INV.alpha"},
                    {"key": {"k": "opaque", "value": "opaque::value"}},
                ]
            }
        ),
        encoding="utf-8",
    )
    identities = semantic_coverage_map._artifact_evidence_index(records_path)
    assert len(identities) == 2


# gabion:evidence E:call_footprint::tests/test_semantic_coverage_map.py::test_semantic_coverage_falls_back_to_opaque_identity_when_display_not_parseable::semantic_coverage_map.py::gabion.analysis.semantic_coverage_map.SemanticCoverageEntry
def test_semantic_coverage_falls_back_to_opaque_identity_when_display_not_parseable() -> None:
    entry = semantic_coverage_map.SemanticCoverageEntry(
        obligation="invariant.x",
        obligation_kind="invariant",
        evidence_display="opaque evidence display",
    )
    assert entry.evidence_identity


# gabion:evidence E:call_footprint::tests/test_semantic_coverage_map.py::test_semantic_coverage_indexes_fallback_for_unparseable_displays::semantic_coverage_map.py::gabion.analysis.semantic_coverage_map._annotation_index::semantic_coverage_map.py::gabion.analysis.semantic_coverage_map._artifact_evidence_index
def test_semantic_coverage_indexes_fallback_for_unparseable_displays(tmp_path: Path) -> None:
    tags = [
        test_evidence.TestEvidenceTag(
            test_id="tests/test_semantic_case.py::test_fallback",
            path="tests/test_semantic_case.py",
            line=1,
            tags=("opaque display",),
        )
    ]
    annotation_index = semantic_coverage_map._annotation_index(tags)
    assert annotation_index
    assert list(annotation_index.values()) == [{"tests/test_semantic_case.py::test_fallback"}]

    records_path = tmp_path / "records_evidence_fallback.json"
    records_path.write_text(
        json.dumps({"evidence_index": [{"display": "opaque display"}]}),
        encoding="utf-8",
    )
    identities = semantic_coverage_map._artifact_evidence_index(records_path)
    assert len(identities) == 1


# gabion:evidence E:call_footprint::tests/test_semantic_coverage_map.py::test_semantic_coverage_uses_parsed_identity_for_parseable_displays::semantic_coverage_map.py::gabion.analysis.semantic_coverage_map.SemanticCoverageEntry::semantic_coverage_map.py::gabion.analysis.semantic_coverage_map._annotation_index::semantic_coverage_map.py::gabion.analysis.semantic_coverage_map._artifact_evidence_index
def test_semantic_coverage_uses_parsed_identity_for_parseable_displays(tmp_path: Path) -> None:
    parsed_display = evidence_keys.render_display(
        evidence_keys.make_function_site_key(path="src/app.py", qual="mod.fn")
    )
    parsed_key = evidence_keys.parse_display(parsed_display)
    assert isinstance(parsed_key, dict)
    expected_identity = evidence_keys.key_identity(evidence_keys.normalize_key(parsed_key))

    entry = semantic_coverage_map.SemanticCoverageEntry(
        obligation="invariant.parsed",
        obligation_kind="invariant",
        evidence_display=parsed_display,
    )
    assert entry.evidence_identity == expected_identity

    tags = [
        test_evidence.TestEvidenceTag(
            test_id="tests/test_semantic_case.py::test_parsed",
            path="tests/test_semantic_case.py",
            line=1,
            tags=(parsed_display,),
        )
    ]
    annotation_index = semantic_coverage_map._annotation_index(tags)
    assert annotation_index[expected_identity] == {"tests/test_semantic_case.py::test_parsed"}

    records_path = tmp_path / "records_evidence_parsed.json"
    records_path.write_text(
        json.dumps({"evidence_index": [{"display": parsed_display}]}),
        encoding="utf-8",
    )
    identities = semantic_coverage_map._artifact_evidence_index(records_path)
    assert expected_identity in identities
