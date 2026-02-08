from __future__ import annotations

import json
from pathlib import Path

from gabion.analysis import call_clusters


# gabion:evidence E:function_site::call_clusters.py::gabion.analysis.call_clusters.build_call_clusters_payload
# gabion:evidence E:function_site::call_clusters.py::gabion.analysis.call_clusters.render_markdown
def test_call_clusters_payload_and_render(tmp_path: Path) -> None:
    src_dir = tmp_path / "src" / "pkg"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text("")
    (src_dir / "core.py").write_text(
        "def helper(x):\n"
        "    return x\n"
    )

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_core.py").write_text(
        "from pkg.core import helper\n"
        "\n"
        "def test_helper():\n"
        "    assert helper(1) == 1\n"
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    evidence_payload = {
        "schema_version": 2,
        "scope": {"root": ".", "include": ["tests"], "exclude": []},
        "tests": [
            {
                "test_id": "tests/test_core.py::test_helper",
                "file": "tests/test_core.py",
                "line": 1,
                "evidence": [],
                "status": "unmapped",
            }
        ],
        "evidence_index": [],
    }
    (out_dir / "test_evidence.json").write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n"
    )

    payload = call_clusters.build_call_clusters_payload(
        [tests_dir, tmp_path / "src"],
        root=tmp_path,
        evidence_path=out_dir / "test_evidence.json",
    )
    assert payload["summary"]["clusters"] == 1
    assert payload["summary"]["tests"] == 1
    assert payload["clusters"]
    cluster = payload["clusters"][0]
    assert cluster["key"]["k"] == "call_cluster"
    assert cluster["count"] == 1
    markdown = call_clusters.render_markdown(payload)
    assert "generated_by_spec_id" in markdown
