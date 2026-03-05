from __future__ import annotations

from pathlib import Path

from scripts.policy import private_symbol_import_guard as guard


# gabion:evidence E:function_site::test_private_symbol_import_guard.py::tests.test_private_symbol_import_guard.test_private_symbol_import_guard_repo_check

def test_private_symbol_import_guard_repo_check(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[4]
    out = tmp_path / "private_symbol_import_report.json"

    rc = guard.main(
        [
            "--root",
            str(repo_root),
            "--allowlist",
            str(repo_root / "docs/policy/private_symbol_import_allowlist.txt"),
            "--baseline",
            str(repo_root / "docs/baselines/private_symbol_import_baseline.json"),
            "--out",
            str(out),
            "--check",
        ]
    )

    assert rc == 0
    assert out.exists()
