from __future__ import annotations

from pathlib import Path

from gabion.tooling.policy_rules import no_monkeypatch_rule as policy
from gabion.tooling.runtime.policy_scan_batch import build_policy_scan_batch


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# gabion:evidence E:call_footprint::tests/test_no_monkeypatch_policy_check.py::test_no_monkeypatch_policy_detects_fixture_and_patch_calls::no_monkeypatch_policy_check.py::scripts.no_monkeypatch_policy_check.collect_violations

# gabion:behavior primary=desired
def test_no_monkeypatch_policy_detects_fixture_and_patch_calls(tmp_path: Path) -> None:
    _write(
        tmp_path / "tests" / "test_bad.py",
        "from unittest.mock import patch\n\n"
        "@patch('m.f')\n"
        "def test_bad(monkeypatch):\n"
        "    monkeypatch.setattr('x','y')\n"
        "    pass\n",
    )

    batch = build_policy_scan_batch(root=tmp_path, target_globs=policy.TARGET_GLOBS)
    violations = policy.collect_violations(batch=batch)
    assert violations
    rendered = "\n".join(v.render() for v in violations)
    assert "monkeypatch fixture is forbidden" in rendered
    assert "patch decorator call is forbidden" in rendered


# gabion:evidence E:call_footprint::tests/test_no_monkeypatch_policy_check.py::test_no_monkeypatch_policy_allows_di_style_tests::no_monkeypatch_policy_check.py::scripts.no_monkeypatch_policy_check.run

# gabion:behavior primary=desired
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


# gabion:evidence E:call_footprint::tests/test_no_monkeypatch_policy_check.py::test_no_monkeypatch_policy_check_writes_policy_result_output::no_monkeypatch_policy_check.py::scripts.no_monkeypatch_policy_check.run
# gabion:behavior primary=desired
def test_no_monkeypatch_policy_check_writes_policy_result_output(tmp_path: Path) -> None:
    _write(tmp_path / "tests" / "test_good.py", "def test_ok():\n    assert True\n")
    out = tmp_path / "out/no_monkeypatch.json"
    result = policy.run(root=tmp_path, output=out)
    assert result == 0
    payload = __import__("json").loads(out.read_text(encoding="utf-8"))
    assert payload["rule_id"] == "no_monkeypatch"
    assert payload["status"] == "pass"
