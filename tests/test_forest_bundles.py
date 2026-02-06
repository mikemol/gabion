from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import AuditConfig, analyze_paths

    return AuditConfig, analyze_paths


def test_bundle_forest_reuses_paramset(tmp_path: Path) -> None:
    AuditConfig, analyze_paths = _load()
    (tmp_path / "mod.py").write_text(
        "# dataflow-bundle: a, b\n"
        "def h(x):\n"
        "    return x\n"
        "\n"
        "def f(a, b):\n"
        "    h(a)\n"
        "    h(b)\n"
    )
    config = AuditConfig(project_root=tmp_path, external_filter=False)
    analysis = analyze_paths(
        [tmp_path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_decision_surfaces=True,
        include_value_decision_surfaces=True,
        config=config,
    )
    forest = analysis.forest
    assert forest is not None
    paramset_id = next(
        node.node_id
        for node in forest.nodes.values()
        if node.kind == "ParamSet" and node.node_id.key == ("a", "b")
    )
    alt_kinds = {alt.kind for alt in forest.alts if paramset_id in alt.inputs}
    assert "SignatureBundle" in alt_kinds
    assert "MarkerBundle" in alt_kinds
