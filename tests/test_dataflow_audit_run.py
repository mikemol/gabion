from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


def _write_bundle_module(path: Path) -> None:
    path.write_text(
        "def callee(x):\n"
        "    return x\n"
        "\n"
        "def caller(a, b):\n"
        "    callee(a)\n"
        "    callee(b)\n"
    )


def _write_type_module(path: Path) -> None:
    path.write_text(
        "def typed_int(x: int):\n"
        "    return x\n"
        "\n"
        "def typed_str(x: str):\n"
        "    return x\n"
        "\n"
        "def type_caller(a):\n"
        "    typed_int(a)\n"
        "\n"
        "def type_conflict(b):\n"
        "    typed_int(b)\n"
        "    typed_str(b)\n"
    )


def test_run_baseline_write_requires_path(tmp_path: Path, capsys) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    _write_bundle_module(target)
    code = da.run([str(target), "--baseline-write"])
    assert code == 2
    captured = capsys.readouterr()
    assert "Baseline path required" in captured.err


def test_run_dot_stdout_short_circuit(tmp_path: Path, capsys) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    _write_bundle_module(target)
    code = da.run([str(target), "--dot", "-"])
    assert code == 0
    captured = capsys.readouterr()
    assert "digraph dataflow_grammar" in captured.out


def test_run_report_baseline_write_and_apply(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    _write_bundle_module(target)
    baseline = tmp_path / "baseline.txt"
    report = tmp_path / "report.md"
    code = da.run(
        [
            str(target),
            "--report",
            str(report),
            "--baseline",
            str(baseline),
            "--baseline-write",
            "--fail-on-violations",
        ]
    )
    assert code == 0
    assert baseline.exists()
    assert report.exists()

    code = da.run(
        [
            str(target),
            "--report",
            str(report),
            "--baseline",
            str(baseline),
            "--fail-on-violations",
        ]
    )
    assert code == 0


def test_run_fail_on_violations_no_report(tmp_path: Path) -> None:
    da = _load()
    target = tmp_path / "mod.py"
    _write_bundle_module(target)
    code = da.run([str(target), "--fail-on-violations"])
    assert code == 1


def test_run_type_audit_early_return(tmp_path: Path, capsys) -> None:
    da = _load()
    target = tmp_path / "types.py"
    _write_type_module(target)
    code = da.run([str(target), "--type-audit", "--type-audit-max", "1"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Type tightening candidates" in captured.out
