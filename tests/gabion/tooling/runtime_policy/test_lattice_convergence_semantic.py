from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gabion.tooling.policy_substrate import lattice_convergence_semantic


@dataclass(frozen=True)
class _StubWitness:
    complete: bool
    violation: object | None


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_semantic_lattice_convergence_request_order_is_deterministic(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    _write(
        tmp_path / "a.py",
        "def z(x):\n    if x:\n        return 1\n    return 0\n\nif True:\n    pass\n",
    )
    _write(
        tmp_path / "b.py",
        "def f(items):\n    for item in items:\n        if item:\n            return item\n    return None\n",
    )
    monkeypatch.setattr(
        lattice_convergence_semantic,
        "_CANONICAL_CORPUS",
        ("b.py", "a.py"),
    )
    monkeypatch.setattr(
        lattice_convergence_semantic,
        "_collect_linkage_diagnostics",
        lambda: (),
    )
    monkeypatch.setattr(
        lattice_convergence_semantic.aspf_lattice_algebra,
        "build_fiber_bundle_for_qualname",
        lambda **_: object(),
    )
    monkeypatch.setattr(
        lattice_convergence_semantic.aspf_lattice_algebra,
        "compute_lattice_witness",
        lambda **_: _StubWitness(complete=True, violation=None),
    )

    first = lattice_convergence_semantic.collect_semantic_lattice_convergence(
        repo_root=tmp_path,
    )
    second = lattice_convergence_semantic.collect_semantic_lattice_convergence(
        repo_root=tmp_path,
    )
    assert first.evaluated_requests == second.evaluated_requests
    assert first.policy_data() == second.policy_data()
    assert first.error_count == 0
    assert first.policy_data()["witness_rows"] == []
    assert tuple(request.path for request in first.evaluated_requests) == tuple(
        sorted(request.path for request in first.evaluated_requests)
    )


def test_semantic_lattice_convergence_parse_and_read_failures_increment_error_count(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    _write(
        tmp_path / "bad.py",
        "def broken(:\n    pass\n",
    )
    monkeypatch.setattr(
        lattice_convergence_semantic,
        "_CANONICAL_CORPUS",
        ("missing.py", "bad.py"),
    )
    monkeypatch.setattr(
        lattice_convergence_semantic,
        "_collect_linkage_diagnostics",
        lambda: (),
    )

    report = lattice_convergence_semantic.collect_semantic_lattice_convergence(
        repo_root=tmp_path,
    )
    codes = tuple(item.code for item in report.diagnostics)
    assert codes == (
        "lattice_corpus_parse_failure",
        "lattice_corpus_read_failure",
    )
    assert report.error_count == 2
    witness_rows = report.policy_data()["witness_rows"]
    assert isinstance(witness_rows, list)
    assert len(witness_rows) == 2
    assert all(isinstance(row, dict) for row in witness_rows)
    assert all(row.get("witness_kind") == "unmapped_witness" for row in witness_rows)
    assert all(row.get("mapping_complete") is False for row in witness_rows)
    assert all(row.get("boundary_crossed") is True for row in witness_rows)
    assert all("obligation_state" not in row for row in witness_rows)


def test_semantic_lattice_convergence_counts_incomplete_or_violation_once_per_request(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    _write(
        tmp_path / "one.py",
        "if True:\n    pass\n",
    )
    monkeypatch.setattr(
        lattice_convergence_semantic,
        "_CANONICAL_CORPUS",
        ("one.py",),
    )
    monkeypatch.setattr(
        lattice_convergence_semantic,
        "_collect_linkage_diagnostics",
        lambda: (),
    )
    monkeypatch.setattr(
        lattice_convergence_semantic.aspf_lattice_algebra,
        "build_fiber_bundle_for_qualname",
        lambda **_: object(),
    )
    monkeypatch.setattr(
        lattice_convergence_semantic.aspf_lattice_algebra,
        "compute_lattice_witness",
        lambda **_: _StubWitness(complete=False, violation=object()),
    )

    report = lattice_convergence_semantic.collect_semantic_lattice_convergence(
        repo_root=tmp_path,
    )
    assert report.evaluated_request_count == 1
    assert report.error_count == 1
    assert tuple(item.code for item in report.diagnostics) == (
        "lattice_witness_incomplete_or_violation",
    )
    witness_row = report.policy_data()["witness_rows"][0]
    assert witness_row["witness_incomplete"] is True
    assert witness_row["witness_violation"] is True


def test_semantic_convergence_payload_stays_pre_transform_shape() -> None:
    source = Path("src/gabion/tooling/policy_substrate/lattice_convergence_semantic.py").read_text(
        encoding="utf-8"
    )
    assert "obligation_state" not in source
