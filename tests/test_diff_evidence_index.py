from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from gabion.tooling import diff_evidence_index


def test_parse_changed_lines_and_dev_null_branch() -> None:
    diff = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -1,0 +2,2 @@
+x
+y
diff --git a/src/b.py b/src/b.py
--- a/src/b.py
+++ /dev/null
@@ -1,1 +0,0 @@
-x
"""
    parsed = diff_evidence_index.parse_changed_lines(diff)
    assert parsed == [
        diff_evidence_index.ChangedLine(path="src/a.py", line=2),
        diff_evidence_index.ChangedLine(path="src/a.py", line=3),
    ]


def test_diff_evidence_json_load_and_refresh_failure(tmp_path: Path) -> None:
    broken_json = tmp_path / "broken.json"
    broken_json.write_text("{bad", encoding="utf-8")
    assert diff_evidence_index.load_json(broken_json) is None
    refreshed = diff_evidence_index.refresh_test_evidence_index(
        tmp_path,
        index_path=tmp_path / "out/test_evidence.json",
        tests_root="tests",
    )
    assert refreshed is False


def test_diff_evidence_key_fallback_tree_hash(tmp_path: Path) -> None:
    key = diff_evidence_index.diff_evidence_key(
        root=tmp_path,
        base="abc",
        head="def",
        index_path=tmp_path / "out/test_evidence.json",
    )
    assert key["base_sha"] == "abc"
    assert key["head_sha"] == "def"
    assert key["tree_hash"]


def test_write_diff_evidence_artifacts(tmp_path: Path) -> None:
    changed = [diff_evidence_index.ChangedLine(path="src/app.py", line=7)]
    changed_path = tmp_path / "artifacts/out/changed_lines.json"
    meta_path = tmp_path / "artifacts/out/evidence_index_meta.json"
    diff_evidence_index.write_diff_evidence_artifacts(
        changed_lines_path=changed_path,
        meta_path=meta_path,
        changed_lines=changed,
        key={"base_sha": "a", "head_sha": "b", "tree_hash": "c", "index_path": "x"},
        stale=False,
        refreshed=True,
        index_path=tmp_path / "out/test_evidence.json",
    )
    changed_payload = json.loads(changed_path.read_text(encoding="utf-8"))
    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert changed_payload["changed_lines"] == [{"path": "src/app.py", "line": 7}]
    assert meta_payload["changed_paths"] == ["src/app.py"]


def test_build_diff_evidence_index_on_git_repo(tmp_path: Path) -> None:
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
    target = root / "src_app.py"
    target.write_text("a\n", encoding="utf-8")
    subprocess.run(["git", "add", "src_app.py"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True, text=True)
    target.write_text("a\nb\n", encoding="utf-8")

    index_path = root / "out/test_evidence.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text('{"tests":[]}\n', encoding="utf-8")
    result = diff_evidence_index.build_diff_evidence_index(
        root=root,
        base=None,
        head=None,
        index_path=index_path,
        tests_root="tests",
        stale_seconds=-1,
        no_refresh=True,
    )
    assert result.changed_lines
    assert result.changed_paths == ["src_app.py"]
    assert result.index_payload is not None


def test_build_diff_evidence_index_missing_index_triggers_refresh_attempt(
    tmp_path: Path,
) -> None:
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
    target = root / "src_app.py"
    target.write_text("a\n", encoding="utf-8")
    subprocess.run(["git", "add", "src_app.py"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True, text=True)
    target.write_text("a\nb\n", encoding="utf-8")

    result = diff_evidence_index.build_diff_evidence_index(
        root=root,
        base=None,
        head=None,
        index_path=root / "out/test_evidence.json",
        tests_root="tests",
        stale_seconds=0,
        no_refresh=False,
    )
    assert result.stale is True
    assert result.refreshed is False
    assert result.index_payload is None


def test_git_diff_changed_lines_raises_on_bad_ref(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        diff_evidence_index.git_diff_changed_lines(
            tmp_path,
            base="__bad_ref__",
            head=None,
        )
