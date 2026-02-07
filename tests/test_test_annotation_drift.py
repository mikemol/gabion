from __future__ import annotations

import json
from pathlib import Path

from gabion.analysis import evidence_keys, test_annotation_drift


# gabion:evidence E:function_site::test_annotation_drift.py::gabion.analysis.test_annotation_drift.build_annotation_drift_payload
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
