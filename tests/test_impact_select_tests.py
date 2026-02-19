from __future__ import annotations

import json
from pathlib import Path

from scripts import impact_select_tests


# gabion:evidence E:call_footprint::tests/test_impact_select_tests.py::test_parse_changed_lines_extracts_hunk_lines::impact_select_tests.py::scripts.impact_select_tests._parse_changed_lines
def test_parse_changed_lines_extracts_hunk_lines() -> None:
    diff = """diff --git a/src/gabion/example.py b/src/gabion/example.py
index 1111111..2222222 100644
--- a/src/gabion/example.py
+++ b/src/gabion/example.py
@@ -4,0 +5,2 @@
+line a
+line b
"""
    changed = impact_select_tests._parse_changed_lines(diff)
    assert changed == [
        impact_select_tests.ChangedLine(path="src/gabion/example.py", line=5),
        impact_select_tests.ChangedLine(path="src/gabion/example.py", line=6),
    ]


# gabion:evidence E:call_footprint::tests/test_impact_select_tests.py::test_select_tests_matches_evidence_site_and_changed_test::impact_select_tests.py::scripts.impact_select_tests._select_tests
def test_select_tests_matches_evidence_site_and_changed_test() -> None:
    payload = {
        "tests": [
            {
                "test_id": "tests/test_alpha.py::test_one",
                "file": "tests/test_alpha.py",
                "evidence": [
                    {
                        "key": {
                            "k": "function_site",
                            "site": {"path": "src/gabion/example.py", "span": [5, 0, 9, 0]},
                        }
                    }
                ],
            },
            {
                "test_id": "tests/test_beta.py::test_two",
                "file": "tests/test_beta.py",
                "evidence": [],
            },
        ]
    }
    changed = [
        impact_select_tests.ChangedLine(path="src/gabion/example.py", line=6),
        impact_select_tests.ChangedLine(path="tests/test_beta.py", line=3),
    ]

    impacted, changed_paths, must_run_impacted, confidence = impact_select_tests._select_tests(
        payload,
        changed_lines=changed,
        must_run_tests={"tests/test_alpha.py::test_one"},
    )

    assert impacted == [
        "tests/test_alpha.py::test_one",
        "tests/test_beta.py::test_two",
    ]
    assert changed_paths == ["src/gabion/example.py", "tests/test_beta.py"]
    assert must_run_impacted == ["tests/test_alpha.py::test_one"]
    assert confidence > 0.0


# gabion:evidence E:call_footprint::tests/test_impact_select_tests.py::test_main_falls_back_when_index_missing::impact_select_tests.py::scripts.impact_select_tests.main
def test_main_falls_back_when_index_missing(tmp_path: Path) -> None:
    root = tmp_path

    out_path = root / "artifacts/audit_reports/impact_selection.json"
    exit_code = impact_select_tests.main(
        [
            "--root",
            str(root),
            "--index",
            "out/test_evidence.json",
            "--out",
            str(out_path.relative_to(root)),
            "--no-refresh",
        ],
        git_diff_changed_lines_fn=lambda *_args: [
            impact_select_tests.ChangedLine(path="src/gabion/example.py", line=3)
        ],
    )

    assert exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "full"
    assert "index_missing" in payload["fallback_reasons"]
