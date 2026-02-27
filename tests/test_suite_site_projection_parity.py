from __future__ import annotations

from pathlib import Path


def _load():
    from gabion.analysis import dataflow_audit as da

    return da


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._analyze_decision_surface_indexed
def test_decision_surface_function_projection_parity_from_suite_sites(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def choose(flag, mode):\n"
        "    if flag:\n"
        "        return mode\n"
        "    return mode\n"
    )

    forest = da.Forest()
    analysis = da.analyze_paths(
        forest=forest,
        paths=[path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_decision_surfaces=True,
        config=da.AuditConfig(project_root=tmp_path),
    )

    decision_alts = [alt for alt in forest.alts if alt.kind == "DecisionSurface"]
    assert decision_alts
    projected: list[str] = []
    for alt in decision_alts:
        suite_node = forest.nodes[alt.inputs[0]]
        assert suite_node.kind == "SuiteSite"
        assert suite_node.meta.get("suite_kind") == "function_body"
        params = list(forest.nodes[alt.inputs[1]].meta.get("params", []))
        descriptor = str(alt.evidence.get("classification_descriptor", "") or "")
        projected.append(
            f"{suite_node.meta['path']}:{suite_node.meta['qual']} decision surface params: {', '.join(params)} ({descriptor})"
        )

    assert sorted(projected) == sorted(analysis.decision_surfaces)


# gabion:evidence E:never/sink::dataflow_audit.py::gabion.analysis.dataflow_audit._collect_never_invariants
def test_never_invariant_function_projection_parity_from_suite_sites(tmp_path: Path) -> None:
    da = _load()
    path = tmp_path / "never_case.py"
    path.write_text(
        "from gabion.invariants import never\n\n"
        "def fail_fast(flag):\n"
        "    if flag:\n"
        "        never('stop')\n"
    )

    forest = da.Forest()
    analysis = da.analyze_paths(
        forest=forest,
        paths=[path],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        include_never_invariants=True,
        config=da.AuditConfig(project_root=tmp_path),
    )

    sink_alts = [alt for alt in forest.alts if alt.kind == "NeverInvariantSink"]
    assert sink_alts
    projected = {
        (
            str(forest.nodes[alt.inputs[0]].meta.get("path", "")),
            str(forest.nodes[alt.inputs[0]].meta.get("qual", "")),
            tuple(str(item) for item in forest.nodes[alt.inputs[1]].meta.get("params", [])),
        )
        for alt in sink_alts
    }
    for alt in sink_alts:
        suite_node = forest.nodes[alt.inputs[0]]
        assert suite_node.kind == "SuiteSite"
        assert suite_node.meta.get("suite_kind") == "call"

    projected_entries = {
        (
            str(entry.get("site", {}).get("path", "")),
            str(entry.get("site", {}).get("function", "")),
            tuple(str(item) for item in entry.get("site", {}).get("bundle", [])),
        )
        for entry in analysis.never_invariants
    }
    assert projected == projected_entries
