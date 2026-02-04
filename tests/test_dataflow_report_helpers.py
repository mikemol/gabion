from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


def _write(path: Path, content: str) -> None:
    path.write_text(content)


def test_emit_report_empty_groups() -> None:
    da = _load()
    report, violations = da._emit_report({}, 3)
    assert "No bundle components detected." in report
    assert violations == []


def test_emit_report_component_summary(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    extra_path = tmp_path / "other.py"
    _write(
        path,
        "from dataclasses import dataclass\n"
        "\n"
        "# dataflow-bundle: a, b\n"
        "@dataclass\n"
        "class AppConfig:\n"
        "    c: int\n"
        "    d: int\n"
        "\n"
        "def f(a, b, x, y):\n"
        "    return a\n",
    )
    _write(extra_path, "def g():\n    return 1\n")
    groups_by_path = {
        path: {"f": [set(["a", "b"]), set(["x", "y"])]},
        extra_path: {},
    }
    report, violations = da._emit_report(
        groups_by_path,
        3,
        type_suggestions=["f.a can tighten to int"],
        type_ambiguities=["f.b downstream types conflict: ['int', 'str']"],
        constant_smells=["mod.py:f.a only observed constant 1 across 1 non-test call(s)"],
        unused_arg_smells=["mod.py:f passes param x to unused mod.py:f.x"],
        decision_surfaces=["mod.py:f decision surface params: a"],
        value_decision_surfaces=["mod.py:f value-encoded decision params: a (min/max)"],
        decision_warnings=["mod.py:f decision param 'a' missing decision tier metadata"],
        context_suggestions=["Consider contextvar for mod.py:f decision surface params: a"],
    )
    assert "Observed-only bundles" in report
    assert "Documented bundles" in report
    assert "Declared Config bundles not observed" in report
    assert "Type-flow audit" in report
    assert "Constant-propagation smells" in report
    assert "Unused-argument smells" in report
    assert "Decision surface candidates" in report
    assert "Value-encoded decision surface candidates" in report
    assert "Decision tier warnings" in report
    assert "Contextvar/ambient rewrite suggestions" in report
    assert any("tier-3" in line for line in violations)


def test_emit_report_fingerprint_provenance_summary(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    _write(path, "def f(a, b):\n    return a\n")
    groups_by_path = {path: {}}
    entries = [
        {
            "path": str(path),
            "function": "f",
            "bundle": ["a", "b"],
            "base_keys": ["int", "str"],
            "ctor_keys": [],
            "glossary_matches": ["user_context"],
        },
        {
            "path": str(path),
            "function": "g",
            "bundle": ["x", "y"],
            "base_keys": ["int", "str"],
            "ctor_keys": [],
            "glossary_matches": ["user_context"],
        },
        {
            "path": str(path),
            "function": "h",
            "bundle": ["c"],
            "base_keys": ["float"],
            "ctor_keys": ["list"],
            "glossary_matches": [],
        },
    ]
    report, _ = da._emit_report(
        groups_by_path, 3, fingerprint_provenance=entries
    )
    assert "Fingerprint provenance summary" in report
    assert "glossary=user_context" in report
    assert "base=['float'] ctor=['list']" in report


def test_emit_report_max_components_cutoff(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    _write(path, "def f(a, b):\n    return a\n")
    groups_by_path = {path: {"f": [set(["a", "b"])]}}
    report, _ = da._emit_report(groups_by_path, 0)
    assert "Showing top 0 components" in report


def test_render_mermaid_component_declared_none_documented(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    fn_key = f"fn::{path}::f"
    bundle_a = "bundle::a"
    bundle_b = "bundle::b"
    nodes = {
        fn_key: {"kind": "fn", "label": "f"},
        bundle_a: {"kind": "bundle", "label": "a,b"},
        bundle_b: {"kind": "bundle", "label": "a,b"},
    }
    bundle_map = {bundle_a: {"a", "b"}, bundle_b: {"a", "b"}}
    adj = {fn_key: {bundle_a, bundle_b}}
    component = [fn_key, bundle_a, bundle_b]
    config_bundles_by_path = {tmp_path / "other.py": {"Cfg": {"x", "y"}}}
    documented_bundles_by_path = {path: {("a", "b")}}
    _, summary = da._render_mermaid_component(
        nodes,
        bundle_map,
        adj,
        component,
        config_bundles_by_path,
        documented_bundles_by_path,
    )
    assert "Declared Config bundles: none found for this component." in summary
    assert "Documented bundles" in summary
    assert "(tier-2, documented)" in summary


def test_merge_counts_by_knobs_merges_subset() -> None:
    da = _load()
    counts = {("a", "b"): 1, ("a", "b", "k"): 2}
    merged = da._merge_counts_by_knobs(counts, {"k"})
    assert merged == {("a", "b", "k"): 3}
