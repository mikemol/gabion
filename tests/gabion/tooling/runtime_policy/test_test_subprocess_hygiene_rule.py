from __future__ import annotations

from pathlib import Path

from gabion.tooling.policy_rules import test_subprocess_hygiene_rule as rule
from gabion.tooling.runtime.policy_scan_batch import build_policy_scan_batch


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# gabion:behavior primary=desired
def test_collect_violations_detects_subprocess_spawn_calls(tmp_path: Path) -> None:
    _write(
        tmp_path / "tests" / "x.py",
        "\n".join(
            [
                "import subprocess",
                "subprocess.run(['echo', 'x'])",
            ]
        )
        + "\n",
    )
    _write(
        tmp_path / "tests" / "y.py",
        "\n".join(
            [
                "from subprocess import check_output as co",
                "co(['git', 'status'])",
            ]
        )
        + "\n",
    )
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("", encoding="utf-8")

    batch = build_policy_scan_batch(root=tmp_path, target_globs=rule.TARGET_GLOBS)
    violations = rule.collect_violations(batch=batch, allowlist_path=allowlist)
    paths = {item.path for item in violations}
    kinds = {item.kind for item in violations}
    assert paths == {"tests/x.py", "tests/y.py"}
    assert kinds == {"subprocess_run", "subprocess_check_output"}


# gabion:behavior primary=desired
def test_collect_violations_respects_allowlist_and_non_spawn_calls(tmp_path: Path) -> None:
    _write(
        tmp_path / "tests" / "allowed.py",
        "\n".join(
            [
                "import subprocess",
                "subprocess.run(['echo', 'x'])",
                "subprocess.CalledProcessError(1, ['x'])",
            ]
        )
        + "\n",
    )
    _write(
        tmp_path / "tests" / "no_spawn.py",
        "\n".join(
            [
                "import subprocess",
                "subprocess.CalledProcessError(1, ['x'])",
            ]
        )
        + "\n",
    )
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("tests/allowed.py\n", encoding="utf-8")

    batch = build_policy_scan_batch(root=tmp_path, target_globs=rule.TARGET_GLOBS)
    violations = rule.collect_violations(batch=batch, allowlist_path=allowlist)
    assert violations == []


# gabion:behavior primary=desired
def test_collect_violations_detects_subprocess_spawn_reassignment(tmp_path: Path) -> None:
    _write(
        tmp_path / "tests" / "x.py",
        "\n".join(
            [
                "import subprocess",
                "def _fake(*_args, **_kwargs):",
                "    return 0",
                "subprocess.run = _fake",
            ]
        )
        + "\n",
    )
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("", encoding="utf-8")

    batch = build_policy_scan_batch(root=tmp_path, target_globs=rule.TARGET_GLOBS)
    violations = rule.collect_violations(batch=batch, allowlist_path=allowlist)
    assert len(violations) == 1
    violation = violations[0]
    assert violation.path == "tests/x.py"
    assert violation.kind == "subprocess_run_reassignment"
