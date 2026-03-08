from __future__ import annotations

from pathlib import Path

import pytest

from gabion.analysis.surfaces import test_behavior


# gabion:evidence E:function_site::test_behavior.py::gabion.analysis.test_behavior.build_test_behavior_payload
# gabion:behavior primary=desired
def test_build_test_behavior_payload_extracts_primary_and_facets(tmp_path: Path) -> None:
    root = tmp_path
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    source = tests_dir / "test_behavior_sample.py"
    source.write_text(
        "\n".join(
            [
                "# gabion:evidence E:function_site::x.py::pkg.fn",
                "# gabion:behavior primary=desired facets=cli,ordering",
                "def test_top():",
                "    assert True",
                "",
                "class TestWidget:",
                "    # gabion:behavior primary=verboten facets=raises,invalid",
                "    def test_method(self):",
                "        assert True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = test_behavior.build_test_behavior_payload(
        [tests_dir],
        root=root,
        include=["tests"],
        exclude=[],
    )
    tests = payload["tests"]
    assert tests[0]["test_id"].endswith("tests/test_behavior_sample.py::TestWidget::test_method")
    assert tests[0]["primary"] == "verboten"
    assert tests[0]["facets"] == ["invalid", "raises"]
    assert tests[1]["test_id"].endswith("tests/test_behavior_sample.py::test_top")
    assert tests[1]["primary"] == "desired"
    assert tests[1]["facets"] == ["cli", "ordering"]
    assert payload["summary"] == {"desired": 1, "allowed_unwanted": 0, "verboten": 1}


# gabion:evidence E:function_site::test_behavior.py::gabion.analysis.test_behavior.build_test_behavior_payload
# gabion:behavior primary=verboten facets=invalid,missing
def test_build_test_behavior_payload_rejects_missing_or_invalid_tags(tmp_path: Path) -> None:
    root = tmp_path
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    source = tests_dir / "test_invalid_behavior.py"
    source.write_text(
        "\n".join(
            [
                "def test_missing():",
                "    assert True",
                "",
                "# gabion:behavior primary=unknown",
                "def test_bad_primary():",
                "    assert True",
                "",
                "# gabion:behavior primary=desired facets=good,bad/facet",
                "def test_bad_facet():",
                "    assert True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc:
        test_behavior.build_test_behavior_payload(
            [tests_dir],
            root=root,
            include=["tests"],
            exclude=[],
        )
    message = str(exc.value)
    assert "missing gabion:behavior tag" in message
    assert "invalid primary" in message
    assert "invalid facet" in message


# gabion:evidence E:function_site::test_behavior.py::gabion.analysis.test_behavior.collect_test_behavior_contract_violations
# gabion:behavior primary=verboten facets=error
def test_collect_test_behavior_contract_violations_reports_errors(tmp_path: Path) -> None:
    root = tmp_path
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    source = tests_dir / "test_duplicates.py"
    source.write_text(
        "\n".join(
            [
                "# gabion:behavior primary=desired",
                "# gabion:behavior primary=desired",
                "def test_dupe():",
                "    assert True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    violations = test_behavior.collect_test_behavior_contract_violations(
        [tests_dir],
        root=root,
        include=["tests"],
        exclude=[],
    )
    assert any("duplicate gabion:behavior tags" in violation for violation in violations)


# gabion:evidence E:function_site::test_behavior.py::gabion.analysis.test_behavior.write_test_behavior
# gabion:behavior primary=desired
def test_write_test_behavior_payload(tmp_path: Path) -> None:
    output = tmp_path / "out" / "test_behavior.json"
    payload = {
        "schema_version": 1,
        "scope": {"root": ".", "include": ["tests"], "exclude": []},
        "tests": [],
        "summary": {"desired": 0, "allowed_unwanted": 0, "verboten": 0},
    }
    test_behavior.write_test_behavior(payload, output)
    assert output.exists()
