from __future__ import annotations

from pathlib import Path

from gabion.analysis import test_evidence


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
    assert tests[0]["evidence"] == ["E:decision/x"]
    assert tests[1]["evidence"] == ["E:async/one"]
    assert tests[2]["evidence"] == ["E:bundle/foo", "E:never/bar"]
    assert tests[3]["status"] == "unmapped"

    index = payload["evidence_index"]
    evidence_ids = [entry["evidence_id"] for entry in index]
    assert evidence_ids == ["E:async/one", "E:bundle/foo", "E:decision/x", "E:never/bar"]


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

    out_path = tmp_path / "out" / "test_evidence.json"
    test_evidence.write_test_evidence(payload, out_path)
    assert out_path.exists()

    missing = tmp_path / "tests" / "missing.py"
    entries = test_evidence._extract_file_evidence(missing, root)
    assert entries == []
