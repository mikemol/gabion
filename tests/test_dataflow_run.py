from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit

    return dataflow_audit


def _write_sample_code(path: Path) -> None:
    path.write_text(
        "def helper(x, y):\n"
        "    return x + y\n\n"
        "def alpha(a, b, *args, **kwargs):\n"
        "    c = a\n"
        "    d, e = (b, b)\n"
        "    return helper(c, d)\n"
    )


def test_run_generates_outputs(tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_sample_code(sample)
    report = tmp_path / "report.md"
    dot = tmp_path / "graph.dot"
    plan = tmp_path / "plan.json"
    protocols = tmp_path / "protocols.py"
    refactor = tmp_path / "refactor.json"
    snapshot = tmp_path / "structure.json"
    argv = [
        str(tmp_path),
        "--root",
        str(tmp_path),
        "--report",
        str(report),
        "--dot",
        str(dot),
        "--synthesis-plan",
        str(plan),
        "--synthesis-protocols",
        str(protocols),
        "--synthesis-protocols-kind",
        "protocol",
        "--synthesis-min-bundle-size",
        "1",
        "--synthesis-allow-singletons",
        "--refactor-plan",
        "--refactor-plan-json",
        str(refactor),
        "--emit-structure-tree",
        str(snapshot),
        "--type-audit-report",
        "--type-audit-max",
        "5",
        "--transparent-decorators",
        "decorator_a,decorator_b",
    ]
    code = dataflow_audit.run(argv)
    assert code == 0
    assert report.exists()
    assert dot.exists()
    assert plan.exists()
    assert protocols.exists()
    assert refactor.exists()
    assert snapshot.exists()


def test_run_baseline_write_and_apply(tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_sample_code(sample)
    baseline = tmp_path / "baseline.txt"
    report = tmp_path / "report.md"
    code = dataflow_audit.run(
        [
            str(tmp_path),
            "--root",
            str(tmp_path),
            "--baseline",
            str(baseline),
            "--baseline-write",
            "--report",
            str(report),
            "--fail-on-violations",
        ]
    )
    assert code == 0
    assert baseline.exists()

    code = dataflow_audit.run(
        [
            str(tmp_path),
            "--root",
            str(tmp_path),
            "--baseline",
            str(baseline),
            "--report",
            str(report),
            "--fail-on-violations",
        ]
    )
    assert code == 0


def test_run_invalid_strictness_from_config(tmp_path: Path) -> None:
    dataflow_audit = _load()
    sample = tmp_path / "sample.py"
    _write_sample_code(sample)
    config = tmp_path / "gabion.toml"
    config.write_text("[dataflow]\nstrictness = \"weird\"\n")
    code = dataflow_audit.run(
        [
            str(tmp_path),
            "--root",
            str(tmp_path),
            "--config",
            str(config),
            "--report",
            str(tmp_path / "report.md"),
        ]
    )
    assert code == 0
