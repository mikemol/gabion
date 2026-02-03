from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit

    return dataflow_audit


def _write_conflict_code(path: Path) -> None:
    path.write_text(
        "def callee_int(x: int):\n"
        "    return x\n\n"
        "def callee_str(x: str):\n"
        "    return x\n\n"
        "def caller(a):\n"
        "    return callee_int(a)\n\n"
        "def caller_conflict(b):\n"
        "    callee_int(b)\n"
        "    callee_str(b)\n"
    )


def _write_bundle_code(path: Path) -> None:
    path.write_text(
        "def callee(x):\n"
        "    return x\n"
        "\n"
        "def caller(a, b):\n"
        "    callee(a)\n"
        "    callee(b)\n"
    )


def test_run_baseline_write_requires_path(tmp_path: Path) -> None:
    dataflow_audit = _load()
    code = dataflow_audit.run(
        [
            str(tmp_path),
            "--root",
            str(tmp_path),
            "--baseline-write",
        ]
    )
    assert code == 2


def test_run_dot_only_returns_early(tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    sample.write_text("def f(a, b):\n    return a + b\n")
    dot_path = tmp_path / "graph.dot"
    code = dataflow_audit.run(
        [
            str(tmp_path),
            "--root",
            str(tmp_path),
            "--dot",
            str(dot_path),
        ]
    )
    assert code == 0
    assert dot_path.exists()


def test_run_fail_on_type_ambiguities_with_synthesis_plan(tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "mod.py"
    _write_conflict_code(sample)
    plan_path = tmp_path / "plan.json"
    code = dataflow_audit.run(
        [
            str(tmp_path),
            "--root",
            str(tmp_path),
            "--synthesis-plan",
            str(plan_path),
            "--fail-on-type-ambiguities",
        ]
    )
    assert plan_path.exists()
    assert code == 1


def test_run_report_with_baseline_write(tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_bundle_code(sample)
    report_path = tmp_path / "report.md"
    baseline_path = tmp_path / "baseline.txt"
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--report",
            str(report_path),
            "--baseline",
            str(baseline_path),
            "--baseline-write",
            "--fail-on-violations",
        ]
    )
    assert code == 0
    assert report_path.exists()
    assert baseline_path.exists()


def test_run_report_with_baseline_apply(tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_bundle_code(sample)
    report_path = tmp_path / "report.md"
    baseline_path = tmp_path / "baseline.txt"
    baseline_path.write_text("preexisting\n")
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--report",
            str(report_path),
            "--baseline",
            str(baseline_path),
            "--fail-on-violations",
        ]
    )
    assert report_path.exists()
    assert code == 1


def test_run_type_audit_prints_findings(tmp_path: Path, capsys) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "types.py"
    _write_conflict_code(sample)
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--type-audit",
            "--type-audit-max",
            "5",
            "--transparent-decorators",
            "wrap, deco",
        ]
    )
    captured = capsys.readouterr().out
    assert "Type tightening candidates" in captured
    assert "Type ambiguities" in captured
    assert code == 0


def test_run_synthesis_protocols_only(tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_bundle_code(sample)
    stubs_path = tmp_path / "stubs.py"
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--synthesis-protocols",
            str(stubs_path),
            "--synthesis-merge-overlap",
            "1.5",
        ]
    )
    assert stubs_path.exists()
    assert code == 0


def test_run_synthesis_outputs_to_stdout(tmp_path: Path, capsys) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_bundle_code(sample)
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--synthesis-plan",
            "-",
            "--synthesis-protocols",
            "-",
        ]
    )
    captured = capsys.readouterr().out
    assert "protocol" in captured or "class" in captured
    assert code == 0


def test_run_synthesis_uses_config_overlap(tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_bundle_code(sample)
    config_path = tmp_path / "gabion.toml"
    config_path.write_text("[synthesis]\nmerge_overlap_threshold = 0.5\n")
    plan_path = tmp_path / "plan.json"
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--config",
            str(config_path),
            "--synthesis-plan",
            str(plan_path),
        ]
    )
    assert plan_path.exists()
    assert code == 0


def test_run_refactor_plan_json_stdout(tmp_path: Path, capsys) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_bundle_code(sample)
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--refactor-plan-json",
            "-",
        ]
    )
    captured = capsys.readouterr().out
    assert captured.strip()
    assert code == 0


def test_run_refactor_plan_without_json(tmp_path: Path, capsys) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_bundle_code(sample)
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--refactor-plan",
            "--exclude",
            "in, .venv",
            "--ignore-params",
            "x, y",
        ]
    )
    captured = capsys.readouterr().out
    assert "bundle" in captured
    assert code == 0


def test_run_dot_stdout_returns_early(tmp_path: Path, capsys) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_bundle_code(sample)
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--dot",
            "-",
        ]
    )
    captured = capsys.readouterr().out
    assert "graph" in captured or "digraph" in captured
    assert code == 0


def test_run_fail_on_violations_with_baseline(tmp_path: Path, capsys) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_bundle_code(sample)
    baseline_path = tmp_path / "baseline.txt"
    baseline_path.write_text("preexisting\n")
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--baseline",
            str(baseline_path),
            "--fail-on-violations",
        ]
    )
    captured = capsys.readouterr().out
    assert "bundle" in captured
    assert code == 1


def test_run_fail_on_violations_baseline_write(tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_bundle_code(sample)
    baseline_path = tmp_path / "baseline.txt"
    code = dataflow_audit.run(
        [
            str(sample),
            "--root",
            str(tmp_path),
            "--baseline",
            str(baseline_path),
            "--baseline-write",
            "--fail-on-violations",
        ]
    )
    assert baseline_path.exists()
    assert code == 0
