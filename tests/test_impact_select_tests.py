from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from gabion.tooling import impact_select_tests
import pytest


# gabion:evidence E:call_footprint::tests/test_impact_select_tests.py::test_parse_changed_lines_extracts_hunk_lines::impact_select_tests.py::gabion.tooling.impact_select_tests._parse_changed_lines
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


# gabion:evidence E:call_footprint::tests/test_impact_select_tests.py::test_select_tests_matches_evidence_site_and_changed_test::impact_select_tests.py::gabion.tooling.impact_select_tests._select_tests
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


# gabion:evidence E:call_footprint::tests/test_impact_select_tests.py::test_main_falls_back_when_index_missing::impact_select_tests.py::gabion.tooling.impact_select_tests.main
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


# gabion:evidence E:call_footprint::tests/test_impact_select_tests.py::test_main_falls_back_when_diff_unavailable::impact_select_tests.py::gabion.tooling.impact_select_tests.main
def test_main_falls_back_when_diff_unavailable(tmp_path: Path) -> None:
    root = tmp_path
    index_path = root / "out/test_evidence.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text('{"tests":[]}\n', encoding="utf-8")

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
        git_diff_changed_lines_fn=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("fatal: no merge base"),
        ),
    )

    assert exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "full"
    assert "diff_unavailable" in payload["fallback_reasons"]
    assert payload["diff"]["error"] == "fatal: no merge base"
    assert payload["diff"]["changed_line_count"] == 0
    assert payload["diff"]["changed_paths"] == []


# gabion:evidence E:call_footprint::tests/test_impact_select_tests.py::test_impact_select_helpers_cover_error_and_stale_paths::impact_select_tests.py::gabion.tooling.impact_select_tests._git_diff_changed_lines::impact_select_tests.py::gabion.tooling.impact_select_tests._select_tests
def test_impact_select_helpers_cover_error_and_stale_paths(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        impact_select_tests._git_diff_changed_lines(
            tmp_path,
            base="__not_a_real_ref__",
            head=None,
        )

    broken_json = tmp_path / "broken.json"
    broken_json.write_text("{bad", encoding="utf-8")
    assert impact_select_tests._load_json(broken_json) is None

    refreshed = impact_select_tests._refresh_index(
        tmp_path,
        tmp_path / "out/test_evidence.json",
        "tests",
    )
    assert refreshed is False

    assert (
        impact_select_tests._site_matches_changed_lines(
            {"path": "src/a.py", "span": ["x", 0, "y", 0]},
            {"src/a.py": {1}},
        )
        is True
    )

    impacted, changed_paths, must_run_impacted, confidence = impact_select_tests._select_tests(
        {},
        changed_lines=[impact_select_tests.ChangedLine(path="src/a.py", line=1)],
        must_run_tests=set(),
    )
    assert impacted == []
    assert changed_paths == ["src/a.py"]
    assert must_run_impacted == []
    assert confidence == 0.0


# gabion:evidence E:call_footprint::tests/test_impact_select_tests.py::test_main_handles_stale_index_without_refresh::impact_select_tests.py::gabion.tooling.impact_select_tests.main
def test_main_handles_stale_index_without_refresh(tmp_path: Path) -> None:
    root = tmp_path
    index_path = root / "out/test_evidence.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text('{"tests":[]}\n', encoding="utf-8")

    # Force stale-index branch.
    stale_time = index_path.stat().st_mtime - 1000
    os.utime(index_path, (stale_time, stale_time))

    out_path = root / "artifacts/audit_reports/impact_selection.json"
    exit_code = impact_select_tests.main(
        [
            "--root",
            str(root),
            "--index",
            "out/test_evidence.json",
            "--out",
            str(out_path.relative_to(root)),
            "--stale-seconds",
            "0",
            "--no-refresh",
            "--must-run-test",
            "tests/test_alpha.py::test_one",
        ],
        git_diff_changed_lines_fn=lambda *_args: [
            impact_select_tests.ChangedLine(path="docs/guide.md", line=1)
        ],
    )

    assert exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "full"
    assert "index_stale" in payload["fallback_reasons"]
    assert payload["selection"]["impacted_docs"] == ["docs/guide.md"]


# gabion:evidence E:call_footprint::tests/test_impact_select_tests.py::test_impact_select_additional_helper_branches::impact_select_tests.py::gabion.tooling.impact_select_tests._read_must_run_tests::impact_select_tests.py::gabion.tooling.impact_select_tests._site_matches_changed_lines
def test_impact_select_additional_helper_branches(tmp_path: Path) -> None:
    diff = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ /dev/null
@@ -1,1 +0,0 @@
-x
diff --git a/src/b.py b/src/b.py
--- a/src/b.py
+++ b/src/b.py
@@ -2,1 +5,0 @@
-x
"""
    assert impact_select_tests._parse_changed_lines(diff) == []

    must_run_file = tmp_path / "must_run.txt"
    must_run_file.write_text("# comment\n\n tests/test_a.py::test_x \n", encoding="utf-8")
    assert impact_select_tests._read_must_run_tests(must_run_file, [" tests/test_b.py::test_y "]) == {
        "tests/test_a.py::test_x",
        "tests/test_b.py::test_y",
    }

    payload = {
        "tests": [
            "not-mapping",
            {
                "test_id": "",
                "file": "tests/test_skip.py",
                "evidence": [],
            },
            {
                "test_id": "tests/test_non_list.py::test_non_list",
                "file": "tests/test_non_list.py",
                "evidence": {"bad": "shape"},
            },
            {
                "test_id": "tests/test_ok.py::test_ok",
                "file": "tests/test_ok.py",
                "evidence": [
                    "bad-item",
                    {"key": "bad-key"},
                    {"key": {"site": {"path": "src/app.py", "span": [1, 0, 1, 0]}}},
                ],
            },
        ]
    }
    impacted, changed_paths, must_run_impacted, confidence = impact_select_tests._select_tests(
        payload,
        changed_lines=[impact_select_tests.ChangedLine(path="src/app.py", line=1)],
        must_run_tests={"tests/test_ok.py::test_ok"},
    )
    assert impacted == ["tests/test_ok.py::test_ok"]
    assert changed_paths == ["src/app.py"]
    assert must_run_impacted == ["tests/test_ok.py::test_ok"]
    assert confidence > 0.0
    assert (
        impact_select_tests._site_matches_changed_lines(
            {"path": ""},
            {"src/app.py": {1}},
        )
        is False
    )
    assert (
        impact_select_tests._site_matches_changed_lines(
            {"path": "src/app.py"},
            {},
        )
        is False
    )
    assert (
        impact_select_tests._site_matches_changed_lines(
            {"path": "src/app.py"},
            {"src/app.py": {1}},
        )
        is True
    )


# gabion:evidence E:call_footprint::tests/test_impact_select_tests.py::test_git_diff_changed_lines_branches_and_main_default_git_diff::impact_select_tests.py::gabion.tooling.impact_select_tests._git_diff_changed_lines::impact_select_tests.py::gabion.tooling.impact_select_tests.main
def test_git_diff_changed_lines_branches_and_main_default_git_diff(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    app_file = root / "src_app.py"
    app_file.write_text("a\n", encoding="utf-8")
    subprocess.run(["git", "add", "src_app.py"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True, text=True)
    app_file.write_text("a\nb\n", encoding="utf-8")
    subprocess.run(["git", "add", "src_app.py"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "update"], cwd=root, check=True, capture_output=True, text=True)

    head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    base_sha = subprocess.check_output(["git", "rev-parse", "HEAD~1"], cwd=root, text=True).strip()

    with_base_and_head = impact_select_tests._git_diff_changed_lines(
        root,
        base=base_sha,
        head=head_sha,
    )
    assert with_base_and_head

    app_file.write_text("a\nb\nc\n", encoding="utf-8")
    without_base = impact_select_tests._git_diff_changed_lines(root, base=None, head=None)
    assert without_base

    index_path = root / "out/test_evidence.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text('{"tests":[]}\n', encoding="utf-8")
    out_path = root / "artifacts/audit_reports/impact_selection.json"
    assert (
        impact_select_tests.main(
            [
                "--root",
                str(root),
                "--index",
                "out/test_evidence.json",
                "--out",
                "artifacts/audit_reports/impact_selection.json",
                "--no-refresh",
            ]
        )
        == 0
    )
    output_payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert output_payload["diff"]["changed_line_count"] >= 1


# gabion:evidence E:call_footprint::tests/test_impact_select_tests.py::test_main_attempts_refresh_when_index_missing::impact_select_tests.py::gabion.tooling.impact_select_tests.main
def test_main_attempts_refresh_when_index_missing(tmp_path: Path) -> None:
    root = tmp_path
    out_path = root / "artifacts/audit_reports/impact_selection.json"
    assert (
        impact_select_tests.main(
            [
                "--root",
                str(root),
                "--index",
                "out/test_evidence.json",
                "--out",
                str(out_path.relative_to(root)),
            ],
            git_diff_changed_lines_fn=lambda *_args: [],
        )
        == 0
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["index"]["refreshed"] is False


# gabion:evidence E:call_footprint::tests/test_impact_select_tests.py::test_impact_select_targets_mode_with_negative_stale_and_unprefixed_paths::impact_select_tests.py::gabion.tooling.impact_select_tests._parse_changed_lines::impact_select_tests.py::gabion.tooling.impact_select_tests._select_tests::impact_select_tests.py::gabion.tooling.impact_select_tests.main
def test_impact_select_targets_mode_with_negative_stale_and_unprefixed_paths(
    tmp_path: Path,
) -> None:
    diff = """diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ src/app.py
@@ -1,0 +1,1 @@
+print("x")
"""
    changed = impact_select_tests._parse_changed_lines(diff)
    assert changed == [impact_select_tests.ChangedLine(path="src/app.py", line=1)]

    payload = {
        "tests": [
            {
                "test_id": "tests/test_app.py::test_app",
                "file": "tests/test_app.py",
                "evidence": [
                    {"key": {"site": {"path": "src/app.py", "span": [10, 0, 10, 0]}}},
                    {"key": {"site": {"path": "src/app.py", "span": [1, 0, 1, 0]}}},
                ],
            }
        ]
    }
    impacted, changed_paths, must_run_impacted, confidence = impact_select_tests._select_tests(
        payload,
        changed_lines=changed,
        must_run_tests={"tests/test_app.py::test_app"},
    )
    assert impacted == ["tests/test_app.py::test_app"]
    assert changed_paths == ["src/app.py"]
    assert must_run_impacted == ["tests/test_app.py::test_app"]
    assert confidence > 0.0

    index_path = tmp_path / "out" / "test_evidence.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    out_path = tmp_path / "artifacts" / "audit_reports" / "impact_selection.json"

    exit_code = impact_select_tests.main(
        [
            "--root",
            str(tmp_path),
            "--index",
            "out/test_evidence.json",
            "--out",
            str(out_path.relative_to(tmp_path)),
            "--stale-seconds",
            "-1",
            "--confidence-threshold",
            "0.5",
            "--no-refresh",
        ],
        git_diff_changed_lines_fn=lambda *_args: changed,
    )
    assert exit_code == 0
    output = json.loads(out_path.read_text(encoding="utf-8"))
    assert output["mode"] == "targeted"
    assert output["fallback_reasons"] == []


# gabion:evidence E:call_footprint::tests/test_impact_select_tests.py::test_impact_select_inner_evidence_loop_break_then_continues_outer_entries::impact_select_tests.py::gabion.tooling.impact_select_tests._select_tests
def test_impact_select_inner_evidence_loop_break_then_continues_outer_entries() -> None:
    payload = {
        "tests": [
            {
                "test_id": "tests/test_app.py::test_match",
                "file": "tests/test_app.py",
                "evidence": [
                    {"key": {"site": {"path": "src/app.py", "span": [3, 0, 3, 0]}}},
                ],
            },
            {
                "test_id": "tests/test_app.py::test_other",
                "file": "tests/test_app.py",
                "evidence": [],
            },
        ]
    }
    impacted, _changed_paths, _must_run_impacted, confidence = impact_select_tests._select_tests(
        payload,
        changed_lines=[impact_select_tests.ChangedLine(path="src/app.py", line=3)],
        must_run_tests=set(),
    )
    assert impacted == ["tests/test_app.py::test_match"]
    assert confidence > 0.0
