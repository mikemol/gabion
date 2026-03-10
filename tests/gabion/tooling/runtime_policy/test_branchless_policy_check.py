from __future__ import annotations

from pathlib import Path

from scripts.policy import branchless_policy_check as policy
from gabion.tooling.runtime.policy_scan_batch import build_policy_scan_batch


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# gabion:evidence E:call_footprint::tests/test_branchless_policy_check.py::test_branchless_policy_flags_non_protocol_branches::branchless_policy_check.py::scripts.branchless_policy_check.collect_violations

# gabion:behavior primary=desired
def test_branchless_policy_flags_non_protocol_branches(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "gabion" / "sample.py",
        "def fn(x):\n"
        "    if x:\n"
        "        return 1\n"
        "    return 0\n",
    )
    batch = build_policy_scan_batch(root=tmp_path, target_globs=(policy.TARGET_GLOB,))
    violations = policy.collect_violations(batch=batch)
    assert violations
    if_violation = next(v for v in violations if v.kind == "if")
    assert list(if_violation.recombination_frontier.required_symbols) == ["x"]
    assert list(if_violation.recombination_frontier.unresolved_symbols) == []
    assert if_violation.recombination_frontier.anchor_line == 1
    assert if_violation.recombination_frontier.execution_event_count >= 1
    assert if_violation.recombination_frontier.execution_frontier_ordinal >= 0


# gabion:evidence E:call_footprint::tests/test_branchless_policy_check.py::test_branchless_policy_allows_marked_decision_protocol::branchless_policy_check.py::scripts.branchless_policy_check.run

# gabion:behavior primary=desired
def test_branchless_policy_marks_decorated_decision_protocol_as_violation(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "gabion" / "sample.py",
        "from gabion.invariants import decision_protocol\n\n"
        "@decision_protocol\n"
        "def decide(x):\n"
        "    if x:\n"
        "        return 1\n"
        "    return 0\n",
    )
    assert policy.run(root=tmp_path) == 1


# gabion:evidence E:call_footprint::tests/test_branchless_policy_check.py::test_branchless_policy_baseline_write_and_ratchet::branchless_policy_check.py::scripts.branchless_policy_check.run

# gabion:behavior primary=desired
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
