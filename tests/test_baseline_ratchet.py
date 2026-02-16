from __future__ import annotations

from pathlib import Path

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.analysis import apply_baseline, load_baseline, resolve_baseline_path, write_baseline

    return apply_baseline, load_baseline, resolve_baseline_path, write_baseline

# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._apply_baseline::baseline_allowlist
def test_baseline_write_and_apply(tmp_path: Path) -> None:
    apply_baseline, load_baseline, resolve_baseline_path, write_baseline = _load()
    baseline_path = resolve_baseline_path("baseline.txt", tmp_path)
    assert baseline_path == tmp_path / "baseline.txt"
    violations = ["violation-a", "violation-b"]
    write_baseline(baseline_path, violations)
    loaded = load_baseline(baseline_path)
    assert loaded == {"violation-a", "violation-b"}
    new, suppressed = apply_baseline(
        ["violation-a", "violation-c", "violation-b"], loaded
    )
    assert new == ["violation-c"]
    assert set(suppressed) == {"violation-a", "violation-b"}
