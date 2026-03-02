from __future__ import annotations

from pathlib import Path

from scripts import no_monkeypatch_policy_check as policy


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# gabion:evidence E:call_footprint::tests/test_no_monkeypatch_policy_check.py::test_no_monkeypatch_policy_detects_fixture_and_patch_calls::no_monkeypatch_policy_check.py::scripts.no_monkeypatch_policy_check.collect_violations

def test_no_monkeypatch_policy_detects_fixture_and_patch_calls(tmp_path: Path) -> None:
    _write(
        tmp_path / "tests" / "test_bad.py",
        "from unittest.mock import patch\n\n"
        "@patch('m.f')\n"
        "def test_bad(monkeypatch):\n"
        "    monkeypatch.setattr('x','y')\n"
        "    pass\n",
    )

    violations = policy.collect_violations(root=tmp_path)
    assert violations
    rendered = "\n".join(v.render() for v in violations)
    assert "monkeypatch fixture is forbidden" in rendered
    assert "patch decorator call is forbidden" in rendered


# gabion:evidence E:call_footprint::tests/test_no_monkeypatch_policy_check.py::test_no_monkeypatch_policy_allows_di_style_tests::no_monkeypatch_policy_check.py::scripts.no_monkeypatch_policy_check.run

def test_no_monkeypatch_policy_allows_di_style_tests(tmp_path: Path) -> None:
    _write(
        tmp_path / "tests" / "test_ok.py",
        "def test_ok():\n"
        "    class Runner:\n"
        "        def __call__(self):\n"
        "            return 1\n"
        "    assert Runner()() == 1\n",
    )

    assert policy.run(root=tmp_path) == 0
