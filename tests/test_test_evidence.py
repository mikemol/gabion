from __future__ import annotations

from pathlib import Path

import pytest

from gabion.analysis import test_evidence


# gabion:evidence E:function_site::test_evidence.py::gabion.analysis.test_evidence.build_test_evidence_payload
def test_extracts_evidence_tags_and_unmapped(tmp_path: Path) -> None:
    root = tmp_path
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    source = tests_dir / "test_sample.py"
    source.write_text(
        "\n".join(
            [
                "# gabion:evidence E:bundle/foo E:never/bar",
                "def test_top():",
                "    assert True",
                "",
                "# gabion:evidence E:async/one",
                "async def test_async():",
                "    assert True",
                "",
                "def helper():",
                "    return 1",
                "",
                "class TestWidget:",
                "    # gabion:evidence E:decision/x",
                "    @staticmethod",
                "    def test_method():",
                "        assert True",
                "",
                "def test_unmapped():",
                "    assert True",
            ]
        )
        + "\n"
    )

    payload = test_evidence.build_test_evidence_payload(
        [tests_dir], root=root, include=["tests"], exclude=[]
    )
    tests = payload["tests"]
    assert tests[0]["test_id"].endswith("tests/test_sample.py::TestWidget::test_method")
    assert tests[1]["test_id"].endswith("tests/test_sample.py::test_async")
    assert tests[2]["test_id"].endswith("tests/test_sample.py::test_top")
    assert tests[3]["test_id"].endswith("tests/test_sample.py::test_unmapped")
    assert [item["display"] for item in tests[0]["evidence"]] == ["E:decision/x"]
    assert [item["display"] for item in tests[1]["evidence"]] == ["E:async/one"]
    assert [item["display"] for item in tests[2]["evidence"]] == [
        "E:bundle/foo",
        "E:never/bar",
    ]
    assert tests[3]["status"] == "unmapped"

    index = payload["evidence_index"]
    evidence_ids = [entry["display"] for entry in index]
    assert evidence_ids == ["E:async/one", "E:bundle/foo", "E:decision/x", "E:never/bar"]


# gabion:evidence E:function_site::test_evidence.py::gabion.analysis.test_evidence._find_evidence_tags E:function_site::test_evidence.py::gabion.analysis.test_evidence.build_test_evidence_payload
def test_requires_adjacent_tag_and_skips_bad_files(tmp_path: Path) -> None:
    root = tmp_path
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    source = tests_dir / "test_gap.py"
    source.write_text(
        "\n".join(
            [
                "# gabion:evidence E:bundle/gap",
                "# plain comment",
                "def test_gap():",
                "    assert True",
            ]
        )
        + "\n"
    )
    bad = tests_dir / "test_bad.py"
    bad.write_text("def bad(:\n    pass\n")

    payload = test_evidence.build_test_evidence_payload(
        [tests_dir], root=root, include=["tests"], exclude=[]
    )
    entry = next(
        test for test in payload["tests"] if test["test_id"].endswith("test_gap")
    )
    assert entry["evidence"] == []
    assert entry["status"] == "unmapped"

    lines = ["@decorator", "def test_gap():", "    pass"]
    assert test_evidence._find_evidence_tags(lines, {}, start_line=2) == []


# gabion:evidence E:function_site::test_evidence.py::gabion.analysis.test_evidence.build_test_evidence_payload
def test_excludes_paths(tmp_path: Path) -> None:
    root = tmp_path
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    source = tests_dir / "test_keep.py"
    source.write_text("def test_keep():\n    assert True\n")
    skip_dir = tests_dir / "skip"
    skip_dir.mkdir()
    (skip_dir / "test_skip.py").write_text("def test_skip():\n    assert True\n")

    payload = test_evidence.build_test_evidence_payload(
        [tests_dir], root=root, include=["tests"], exclude=["tests/skip"]
    )
    test_ids = [entry["test_id"] for entry in payload["tests"]]
    assert any("test_keep" in test_id for test_id in test_ids)
    assert not any("test_skip" in test_id for test_id in test_ids)


# gabion:evidence E:function_site::test_evidence.py::gabion.analysis.test_evidence._extract_file_evidence E:function_site::test_evidence.py::gabion.analysis.test_evidence.build_test_evidence_payload E:function_site::test_evidence.py::gabion.analysis.test_evidence.write_test_evidence
def test_handles_missing_and_direct_file_paths(tmp_path: Path) -> None:
    root = tmp_path
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    source = tests_dir / "test_direct.py"
    source.write_text("def test_direct():\n    assert True\n")

    payload = test_evidence.build_test_evidence_payload(
        [source], root=root, include=["tests/test_direct.py"], exclude=[]
    )
    assert payload["tests"][0]["test_id"].endswith("tests/test_direct.py::test_direct")
    assert payload["scope"]["root"] == "."

    out_path = tmp_path / "out" / "test_evidence.json"
    test_evidence.write_test_evidence(payload, out_path)
    assert out_path.exists()

    missing = tmp_path / "tests" / "missing.py"
    entries = test_evidence._extract_file_evidence(missing, root)
    assert entries == []


# gabion:evidence E:function_site::test_evidence.py::gabion.analysis.test_evidence.collect_test_tags
def test_collect_test_tags_handles_async_and_class(tmp_path: Path) -> None:
    root = tmp_path
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    source = tests_dir / "test_tags.py"
    source.write_text(
        "\n".join(
            [
                "# gabion:evidence E:function_site::x.py::pkg.fn",
                "async def test_async():",
                "    assert True",
                "",
                "class TestWidget:",
                "    # gabion:evidence E:function_site::x.py::pkg.fn",
                "    @decorator",
                "    def test_method(self):",
                "        assert True",
                "",
                "def helper():",
                "    return 1",
            ]
        )
        + "\n"
    )
    bad = tests_dir / "test_bad.py"
    bad.write_text("def bad(:\n    pass\n")

    tags = test_evidence.collect_test_tags(
        [tests_dir], root=root, include=["tests"], exclude=[]
    )
    ids = {entry.test_id: entry.tags for entry in tags}
    assert any("test_tags.py::test_async" in test_id for test_id in ids)
    assert any("test_tags.py::TestWidget::test_method" in test_id for test_id in ids)
    assert not any("test_tags.py::helper" in test_id for test_id in ids)

    missing = tests_dir / "missing.py"
    assert test_evidence._extract_file_tags(missing, root) == []


# gabion:evidence E:function_site::test_evidence.py::gabion.analysis.test_evidence.build_test_evidence_payload
def test_rejects_duplicate_test_ids(tmp_path: Path) -> None:
    root = tmp_path
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    source = tests_dir / "test_dupe.py"
    source.write_text(
        "\n".join(
            [
                "def test_dupe():",
                "    assert True",
                "",
                "def test_dupe():",
                "    assert True",
            ]
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="Duplicate test_id"):
        test_evidence.build_test_evidence_payload(
            [tests_dir], root=root, include=["tests"], exclude=[]
        )


# gabion:evidence E:call_footprint::tests/test_test_evidence.py::test_build_test_evidence_merges_duplicate_evidence_identity::test_evidence.py::gabion.analysis.test_evidence.build_test_evidence_payload
def test_build_test_evidence_merges_duplicate_evidence_identity(tmp_path: Path) -> None:
    root = tmp_path
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    source = tests_dir / "test_dupe_evidence.py"
    source.write_text(
        "\n".join(
            [
                "# gabion:evidence E:function_site::m.py::pkg.fn",
                "def test_one():",
                "    assert True",
                "",
                "# gabion:evidence E:function_site::m.py::pkg.fn",
                "def test_two():",
                "    assert True",
            ]
        )
        + "\n"
    )
    payload = test_evidence.build_test_evidence_payload(
        [tests_dir], root=root, include=["tests"], exclude=[]
    )
    assert len(payload["evidence_index"]) == 1
    assert payload["evidence_index"][0]["tests"] == [
        "tests/test_dupe_evidence.py::test_one",
        "tests/test_dupe_evidence.py::test_two",
    ]


# gabion:evidence E:call_footprint::tests/test_test_evidence.py::test_collect_test_files_ignores_non_python_paths::test_evidence.py::gabion.analysis.test_evidence._collect_test_files
def test_collect_test_files_ignores_non_python_paths(tmp_path: Path) -> None:
    root = tmp_path
    txt = tmp_path / "notes.txt"
    txt.write_text("plain text\n", encoding="utf-8")
    files = test_evidence._collect_test_files([txt], root=root, exclude=set())
    assert files == []


# gabion:evidence E:call_footprint::tests/test_test_evidence.py::test_evidence_comments_ignores_empty_ids::test_evidence.py::gabion.analysis.test_evidence._evidence_comments
def test_evidence_comments_ignores_empty_ids() -> None:
    comments = test_evidence._evidence_comments("# gabion:evidence   \n")
    assert comments == {}
