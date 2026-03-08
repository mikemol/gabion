from __future__ import annotations

import json
from pathlib import Path

import pytest

from gabion.tooling.impact import diff_evidence_index


# gabion:evidence E:call_footprint::tests/test_diff_evidence_index.py::test_parse_changed_lines_and_dev_null_branch::diff_evidence_index.py::gabion.tooling.diff_evidence_index.build_diff_evidence_index
# gabion:behavior primary=desired
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


# gabion:evidence E:call_footprint::tests/test_diff_evidence_index.py::test_diff_evidence_json_load_and_refresh_failure::diff_evidence_index.py::gabion.tooling.diff_evidence_index.build_diff_evidence_index
# gabion:behavior primary=desired
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


# gabion:evidence E:call_footprint::tests/test_diff_evidence_index.py::test_diff_evidence_key_fallback_tree_hash::diff_evidence_index.py::gabion.tooling.diff_evidence_index.build_diff_evidence_index
# gabion:behavior primary=allowed_unwanted facets=fallback
def test_diff_evidence_key_fallback_tree_hash(tmp_path: Path) -> None:
    key = diff_evidence_index.diff_evidence_key(
        root=tmp_path,
        base="abc",
        head="def",
        index_path=tmp_path / "out/test_evidence.json",
        tree_hash_fn=lambda _root: "fallback-tree-hash",
    )
    assert key["base_sha"] == "abc"
    assert key["head_sha"] == "def"
    assert key["tree_hash"] == "fallback-tree-hash"


# gabion:evidence E:call_footprint::tests/test_diff_evidence_index.py::test_write_diff_evidence_artifacts::diff_evidence_index.py::gabion.tooling.diff_evidence_index.build_diff_evidence_index
# gabion:behavior primary=desired
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


# gabion:evidence E:call_footprint::tests/test_diff_evidence_index.py::test_build_diff_evidence_index_on_git_repo::diff_evidence_index.py::gabion.tooling.diff_evidence_index.build_diff_evidence_index
# gabion:behavior primary=desired
def test_build_diff_evidence_index_with_injected_git_diff(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "out/test_evidence.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text('{"tests":[]}\n', encoding="utf-8")
    observed: dict[str, object] = {}

    def _git_diff(root_path: Path, base: str | None, head: str | None) -> list[diff_evidence_index.ChangedLine]:
        observed["root"] = root_path
        observed["base"] = base
        observed["head"] = head
        return [diff_evidence_index.ChangedLine(path="src_app.py", line=2)]

    def _key(root_path: Path, base: str | None, head: str | None, target: Path) -> dict[str, str]:
        assert root_path == root
        assert target == index_path
        return {
            "base_sha": str(base or ""),
            "head_sha": str(head or ""),
            "tree_hash": "tree-hash",
            "index_path": str(target),
        }

    result = diff_evidence_index.build_diff_evidence_index(
        root=root,
        base=None,
        head=None,
        index_path=index_path,
        tests_root="tests",
        stale_seconds=-1,
        no_refresh=True,
        git_diff_changed_lines_fn=_git_diff,
        diff_evidence_key_fn=_key,
    )
    assert observed["root"] == root
    assert observed["base"] is None
    assert observed["head"] is None
    assert result.changed_lines
    assert result.changed_paths == ["src_app.py"]
    assert result.index_payload is not None


# gabion:evidence E:call_footprint::tests/test_diff_evidence_index.py::test_build_diff_evidence_index_missing_index_triggers_refresh_attempt::diff_evidence_index.py::gabion.tooling.diff_evidence_index.build_diff_evidence_index
# gabion:behavior primary=verboten facets=missing
def test_build_diff_evidence_index_missing_index_triggers_refresh_attempt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    observed: dict[str, object] = {}

    def _git_diff(root_path: Path, base: str | None, head: str | None) -> list[diff_evidence_index.ChangedLine]:
        observed["root"] = root_path
        observed["base"] = base
        observed["head"] = head
        return [diff_evidence_index.ChangedLine(path="src_app.py", line=2)]

    def _refresh(root_path: Path, target: Path, tests_root: str) -> bool:
        observed["refresh"] = (root_path, target, tests_root)
        return False

    result = diff_evidence_index.build_diff_evidence_index(
        root=root,
        base=None,
        head=None,
        index_path=root / "out/test_evidence.json",
        tests_root="tests",
        stale_seconds=0,
        no_refresh=False,
        git_diff_changed_lines_fn=_git_diff,
        refresh_test_evidence_index_fn=_refresh,
        diff_evidence_key_fn=lambda _root, _base, _head, index_path: {
            "base_sha": "",
            "head_sha": "",
            "tree_hash": "fallback",
            "index_path": str(index_path),
        },
    )
    assert observed["root"] == root
    assert observed["refresh"] == (root, root / "out/test_evidence.json", "tests")
    assert result.stale is True
    assert result.refreshed is False
    assert result.index_payload is None


# gabion:evidence E:call_footprint::tests/test_diff_evidence_index.py::test_git_diff_changed_lines_raises_on_bad_ref::diff_evidence_index.py::gabion.tooling.diff_evidence_index.build_diff_evidence_index
# gabion:behavior primary=verboten facets=raises
def test_git_diff_changed_lines_raises_on_bad_ref(tmp_path: Path) -> None:
    class _ProcResult:
        def __init__(self, *, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def _failing_run(_cmd: list[str], _root: Path) -> _ProcResult:
        return _ProcResult(returncode=1, stderr="fatal: bad revision")

    with pytest.raises(RuntimeError):
        diff_evidence_index.git_diff_changed_lines(
            tmp_path,
            base="__bad_ref__",
            head=None,
            run_command_fn=_failing_run,
        )
