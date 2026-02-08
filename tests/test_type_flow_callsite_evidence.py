from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis.dataflow_audit import (
        CallArgs,
        FunctionInfo,
        _format_type_flow_site,
        analyze_type_flow_repo_with_evidence,
        render_report,
    )

    return (
        CallArgs,
        FunctionInfo,
        _format_type_flow_site,
        analyze_type_flow_repo_with_evidence,
        render_report,
    )


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._format_type_flow_site::call E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._normalize_snapshot_path::root
def test_format_type_flow_site_handles_missing_span(tmp_path: Path) -> None:
    CallArgs, FunctionInfo, _format_type_flow_site, _, _ = _load()
    caller = FunctionInfo(
        name="caller",
        qual="pkg.mod.caller",
        path=tmp_path / "mod.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        scope=(),
    )
    callee = FunctionInfo(
        name="callee",
        qual="pkg.mod.callee",
        path=tmp_path / "mod.py",
        params=[],
        annots={},
        calls=[],
        unused_params=set(),
        scope=(),
    )
    call = CallArgs(
        callee="callee",
        pos_map={},
        kw_map={},
        const_pos={},
        const_kw={},
        non_const_pos=set(),
        non_const_kw=set(),
        star_pos=[],
        star_kw=[],
        is_test=False,
        span=None,
    )
    rendered = _format_type_flow_site(
        caller=caller,
        call=call,
        callee=callee,
        caller_param="x",
        callee_param="y",
        annot="int",
        project_root=tmp_path,
    )
    assert rendered.startswith("mod.py:caller:")


# gabion:evidence E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._emit_report::bundle_sites_by_path,coherence_witnesses,constant_smells,context_suggestions,deadness_witnesses,decision_surfaces,decision_warnings,exception_obligations,fingerprint_matches,fingerprint_provenance,fingerprint_synth,fingerprint_warnings,forest,groups_by_path,handledness_witnesses,invariant_propositions,max_components,never_invariants,rewrite_plans,type_ambiguities,type_callsite_evidence,type_suggestions,unused_arg_smells,value_decision_rewrites,value_decision_surfaces E:decision_surface/direct::dataflow_audit.py::gabion.analysis.dataflow_audit._infer_type_flow::strictness
def test_type_flow_evidence_in_report(tmp_path: Path) -> None:
    _, _, _, analyze_type_flow_repo_with_evidence, render_report = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "def callee(a: int, *args: str, **kwargs: float):\n"
        "    return a\n"
        "\n"
        "def caller(x, z, xs, ys):\n"
        "    callee(1, x)\n"
        "    callee(1, extra=z)\n"
        "    callee(1, *xs, **ys)\n"
    )
    suggestions, ambiguities, evidence = analyze_type_flow_repo_with_evidence(
        [path],
        project_root=tmp_path,
        ignore_params=set(),
        strictness="low",
        external_filter=False,
    )
    assert evidence
    report, _ = render_report(
        {path: {}},
        3,
        type_suggestions=suggestions,
        type_ambiguities=ambiguities,
        type_callsite_evidence=evidence,
    )
    assert "Type-flow callsite evidence:" in report
