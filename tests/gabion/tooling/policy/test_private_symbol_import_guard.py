from __future__ import annotations

import json
from pathlib import Path

from scripts.policy import private_symbol_import_guard as guard


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# gabion:evidence E:function_site::test_private_symbol_import_guard.py::tests.test_private_symbol_import_guard.test_collect_private_import_violations_detects_private_imports
# gabion:behavior primary=desired
def test_collect_private_import_violations_detects_private_imports(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "pkg" / "a.py",
        "\n".join(
            [
                "from pkg.helpers import _private",  # should detect
                "from pkg.helpers import __all__",  # dunder ignored
                "import pkg._internal",  # should detect
                "import pkg.public",  # ignored
            ]
        )
        + "\n",
    )
    _write(tmp_path / "tests" / "x.py", "from pkg.x import _x\n")

    violations = guard._collect_private_import_violations(root=tmp_path)
    keys = {(v.importer, v.module_path, v.symbol) for v in violations}

    assert ("src/pkg/a.py", "pkg.helpers", "_private") in keys
    assert ("src/pkg/a.py", "pkg", "_internal") in keys
    assert ("tests/x.py", "pkg.x", "_x") in keys
    assert all(symbol != "__all__" for _, _, symbol in keys)


# gabion:evidence E:function_site::test_private_symbol_import_guard.py::tests.test_private_symbol_import_guard.test_run_check_fails_on_new_non_exempt_private_imports
# gabion:behavior primary=verboten facets=fail
def test_run_check_fails_on_new_non_exempt_private_imports(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "pkg" / "a.py", "from pkg.helpers import _private\n")

    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("", encoding="utf-8")

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"entries": []}), encoding="utf-8")

    out = tmp_path / "report.json"
    rc = guard.run(
        root=tmp_path,
        allowlist_path=allowlist,
        baseline_path=baseline,
        out_path=out,
        check=True,
        write_baseline=False,
    )

    assert rc == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["totals"]["new_violations"] == 1


# gabion:evidence E:function_site::test_private_symbol_import_guard.py::tests.test_private_symbol_import_guard.test_run_check_passes_when_allowlisted_or_baselined
# gabion:behavior primary=desired
def test_run_check_passes_when_allowlisted_or_baselined(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "pkg" / "a.py", "from pkg.helpers import _private\n")

    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("src/pkg/a.py|pkg.helpers|_private\n", encoding="utf-8")

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"entries": []}), encoding="utf-8")

    out = tmp_path / "report.json"
    rc = guard.run(
        root=tmp_path,
        allowlist_path=allowlist,
        baseline_path=baseline,
        out_path=out,
        check=True,
        write_baseline=False,
    )

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["totals"]["non_exempt"] == 0
