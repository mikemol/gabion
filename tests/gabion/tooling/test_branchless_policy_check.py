from __future__ import annotations

from pathlib import Path

from scripts import branchless_policy_check as policy


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# gabion:evidence E:call_footprint::tests/test_branchless_policy_check.py::test_branchless_policy_flags_non_protocol_branches::branchless_policy_check.py::scripts.branchless_policy_check.collect_violations

def test_branchless_policy_flags_non_protocol_branches(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "gabion" / "sample.py",
        "def fn(x):\n"
        "    if x:\n"
        "        return 1\n"
        "    return 0\n",
    )
    violations = policy.collect_violations(root=tmp_path)
    assert violations
    assert any(v.kind == "if" for v in violations)


# gabion:evidence E:call_footprint::tests/test_branchless_policy_check.py::test_branchless_policy_allows_marked_decision_protocol::branchless_policy_check.py::scripts.branchless_policy_check.run

def test_branchless_policy_allows_marked_decision_protocol(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "gabion" / "sample.py",
        "from gabion.invariants import decision_protocol\n\n"
        "@decision_protocol\n"
        "def decide(x):\n"
        "    if x:\n"
        "        return 1\n"
        "    return 0\n",
    )
    assert policy.run(root=tmp_path) == 0


# gabion:evidence E:call_footprint::tests/test_branchless_policy_check.py::test_branchless_policy_baseline_write_and_ratchet::branchless_policy_check.py::scripts.branchless_policy_check.run

def test_branchless_policy_baseline_write_and_ratchet(tmp_path: Path) -> None:
    src_file = tmp_path / "src" / "gabion" / "sample.py"
    baseline = tmp_path / "baseline.json"

    _write(
        src_file,
        "def fn(x):\n"
        "    if x:\n"
        "        return 1\n"
        "    return 0\n",
    )

    assert policy.run(root=tmp_path, baseline=baseline, baseline_write=True) == 0
    assert baseline.exists()
    assert policy.run(root=tmp_path, baseline=baseline) == 0

    _write(
        tmp_path / "src" / "gabion" / "newer.py",
        "def extra(y):\n"
        "    if y:\n"
        "        return y\n"
        "    return y\n",
    )
    assert policy.run(root=tmp_path, baseline=baseline) == 1
