from __future__ import annotations

from pathlib import Path

from gabion.tooling.policy_rules import test_sleep_hygiene_rule as rule


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# gabion:behavior primary=desired
def test_collect_violations_detects_time_sleep_calls(tmp_path: Path) -> None:
    _write(
        tmp_path / "tests" / "x.py",
        "\n".join(
            [
                "import time",
                "time.sleep(0.1)",
            ]
        )
        + "\n",
    )
    _write(
        tmp_path / "tests" / "y.py",
        "\n".join(
            [
                "from time import sleep as snooze",
                "snooze(0.2)",
            ]
        )
        + "\n",
    )
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("", encoding="utf-8")

    violations = rule.collect_violations(root=tmp_path, allowlist_path=allowlist)
    paths = {item.path for item in violations}
    kinds = {item.kind for item in violations}
    assert paths == {"tests/x.py", "tests/y.py"}
    assert kinds == {"time_sleep", "time_sleep_import"}


# gabion:behavior primary=desired
def test_collect_violations_respects_allowlist_and_non_sleep_calls(tmp_path: Path) -> None:
    _write(
        tmp_path / "tests" / "allowed.py",
        "\n".join(
            [
                "import time",
                "time.sleep(0.1)",
            ]
        )
        + "\n",
    )
    _write(
        tmp_path / "tests" / "no_sleep.py",
        "\n".join(
            [
                "import time",
                "value = time.monotonic()",
            ]
        )
        + "\n",
    )
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("tests/allowed.py\n", encoding="utf-8")

    violations = rule.collect_violations(root=tmp_path, allowlist_path=allowlist)
    assert violations == []
