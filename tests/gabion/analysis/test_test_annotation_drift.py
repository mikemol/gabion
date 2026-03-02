from __future__ import annotations

import json
from pathlib import Path

from gabion.analysis import evidence_keys, test_annotation_drift


# gabion:evidence E:function_site::test_annotation_drift.py::gabion.analysis.test_annotation_drift.build_annotation_drift_payload E:decision_surface/direct::test_annotation_drift.py::gabion.analysis.test_annotation_drift.build_annotation_drift_payload::stale_6aee05fe5cd3
def test_annotation_drift_orphaned(tmp_path: Path) -> None:
    test_file = tmp_path / "test_sample.py"
    test_file.write_text(
        "# gabion:evidence E:function_site::mod.py::pkg.fn E:function_site::missing.py::pkg.missing\n"
        "def test_good():\n"
        "    assert True\n\n"
        "# gabion:evidence E:bundle/alias_invariance\n"
        "def test_legacy():\n"
        "    assert True\n\n"
        "# gabion:evidence NOT_A_TAG\n"
        "def test_bad():\n"
        "    assert True\n"
    )

    key = evidence_keys.make_function_site_key(path="mod.py", qual="pkg.fn")
    display = evidence_keys.render_display(key)
    legacy_key = evidence_keys.make_opaque_key("E:bundle/alias_invariance")
    legacy_display = evidence_keys.render_display(legacy_key)
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": ["tests"], "exclude": []},
        "tests": [
            {
                "test_id": "test_sample.py::test_good",
                "file": "test_sample.py",
                "line": 1,
                "status": "mapped",
                "evidence": [{"key": key, "display": display}],
            },
            {
                "test_id": "test_sample.py::test_legacy",
                "file": "test_sample.py",
                "line": 5,
                "status": "mapped",
                "evidence": [{"key": legacy_key, "display": legacy_display}],
            }
        ],
        "evidence_index": [
            {
                "key": key,
                "display": display,
                "tests": ["test_sample.py::test_good"],
            },
            {
                "key": legacy_key,
                "display": legacy_display,
                "tests": ["test_sample.py::test_legacy"],
            },
        ],
    }
    evidence_path = tmp_path / "out" / "test_evidence.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n")

    payload = test_annotation_drift.build_annotation_drift_payload(
        [tmp_path],
        root=tmp_path,
        evidence_path=evidence_path,
    )
    summary = payload.get("summary", {})
    assert summary.get("ok") == 1
    assert summary.get("orphaned") == 2
    assert summary.get("legacy_tag") == 1
    assert payload.get("generated_by_spec_id")
    report_md = test_annotation_drift.render_markdown(payload)
    assert "generated_by_spec_id" in report_md
    assert "orphaned" in report_md


# gabion:evidence E:function_site::test_annotation_drift.py::gabion.analysis.test_annotation_drift.build_annotation_drift_payload E:decision_surface/direct::test_annotation_drift.py::gabion.analysis.test_annotation_drift.build_annotation_drift_payload::stale_aadd8423ac2a_f0495429
def test_annotation_drift_legacy_ambiguous_and_missing(tmp_path: Path) -> None:
    test_file = tmp_path / "test_legacy.py"
    test_file.write_text(
        "# gabion:evidence E:custom\n"
        "def test_ambiguous():\n"
        "    assert True\n\n"
        "# gabion:evidence E:missing\n"
        "def test_missing():\n"
        "    assert True\n"
    )

    key_one = {"k": "custom", "value": 1}
    key_two = {"k": "custom", "value": 2}
    display = "E:custom"
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": ["tests"], "exclude": []},
        "tests": [
            {
                "test_id": "test_legacy.py::test_ambiguous",
                "file": "test_legacy.py",
                "line": 1,
                "status": "mapped",
                "evidence": [{"key": key_one, "display": display}],
            },
            {
                "test_id": "test_legacy.py::test_ambiguous_alt",
                "file": "test_legacy.py",
                "line": 1,
                "status": "mapped",
                "evidence": [{"key": key_two, "display": display}],
            },
        ],
        "evidence_index": [
            {"key": key_one, "display": display, "tests": ["test_legacy.py::test_ambiguous"]},
            {
                "key": key_two,
                "display": display,
                "tests": ["test_legacy.py::test_ambiguous_alt"],
            },
        ],
    }
    evidence_path = tmp_path / "out" / "test_evidence.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n")

    payload = test_annotation_drift.build_annotation_drift_payload(
        [tmp_path],
        root=tmp_path,
        evidence_path=evidence_path,
    )
    summary = payload.get("summary", {})
    assert summary.get("legacy_ambiguous") == 1
    assert summary.get("orphaned") == 1


# gabion:evidence E:function_site::test_annotation_drift.py::gabion.analysis.test_annotation_drift.render_markdown
def test_annotation_drift_render_sections_and_write(tmp_path: Path) -> None:
    payload = {
        "summary": {"ok": 1},
        "entries": [
            "bad",
            {"status": "orphaned", "test_id": "t1", "tag": "E:bad", "reason": "x"},
            {
                "status": "legacy_tag",
                "test_id": "t2",
                "tag": "E:legacy",
                "reason": "legacy_display",
            },
            {
                "status": "legacy_ambiguous",
                "test_id": "t3",
                "tag": "E:legacy",
                "reason": "legacy_display_ambiguous",
            },
        ],
    }
    rendered = test_annotation_drift.render_markdown(payload)
    assert "Orphaned tags:" in rendered
    assert "Legacy tags:" in rendered
    assert "Ambiguous legacy tags:" in rendered

    output_path = tmp_path / "out" / "test_annotation_drift.json"
    test_annotation_drift.write_annotation_drift(payload, output_path=output_path)
    assert output_path.exists()


# gabion:evidence E:function_site::test_annotation_drift.py::gabion.analysis.test_annotation_drift._summarize
def test_annotation_drift_summarize_unknown_status() -> None:
    summary = test_annotation_drift._summarize([{"status": "custom"}])
    assert summary["custom"] == 1


# gabion:evidence E:call_footprint::tests/test_test_annotation_drift.py::test_annotation_drift_render_handles_non_list_entries_payload::test_annotation_drift.py::gabion.analysis.test_annotation_drift.render_markdown
def test_annotation_drift_render_handles_non_list_entries_payload() -> None:
    rendered = test_annotation_drift.render_markdown(
        {"summary": {"ok": 1}, "entries": "not-a-list"}
    )
    assert "Summary" in rendered
