from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gabion.analysis import aspf
from gabion.analysis import dataflow_run_outputs


def _run_output_context(root: Path) -> dataflow_run_outputs.DataflowRunOutputContext:
    args = SimpleNamespace(
        report=None,
        dot=None,
        type_audit=False,
        type_audit_max=10,
        type_audit_report=False,
        fail_on_type_ambiguities=False,
        fail_on_violations=False,
        max_components=5,
        synthesis_plan=None,
        synthesis_report=False,
        synthesis_protocols=None,
        synthesis_protocols_kind="dataclass",
        synthesis_max_tier=3,
        synthesis_min_bundle_size=2,
        synthesis_allow_singletons=False,
        synthesis_merge_overlap=None,
        synthesis_property_hook_min_confidence=0.0,
        synthesis_property_hook_hypothesis=False,
        refactor_plan=False,
        refactor_plan_json=None,
        emit_structure_tree=None,
        emit_structure_metrics=None,
    )
    analysis = SimpleNamespace(
        groups_by_path={Path("src/example.py"): {"f": [("a", "b")]}},
        type_ambiguities=[],
        type_suggestions=[],
        forest=aspf.Forest(),
        decision_warnings=[],
        fingerprint_warnings=[],
        parse_failure_witnesses=[],
        decision_surfaces=[],
        value_decision_surfaces=[],
        invariant_propositions=[],
        forest_spec=None,
    )
    return dataflow_run_outputs.DataflowRunOutputContext(
        args=args,
        analysis=analysis,
        paths=[Path("src/example.py")],
        config=SimpleNamespace(project_root=root),
        synth_defaults={},
        baseline_path=None,
        baseline_write=False,
        decision_snapshot_path=None,
        fingerprint_deadness_json=None,
        fingerprint_coherence_json=None,
        fingerprint_rewrite_plans_json=None,
        fingerprint_exception_obligations_json=None,
        fingerprint_handledness_json=None,
    )


# gabion:evidence E:call_footprint::tests/test_dataflow_run_outputs.py::test_apply_run_output_ops_skips_unknown_op_and_reaches_terminal_console_step::dataflow_run_outputs.py::gabion.analysis.dataflow_run_outputs.apply_run_output_ops
def test_apply_run_output_ops_skips_unknown_op_and_reaches_terminal_console_step(
    tmp_path: Path,
) -> None:
    dataflow_run_outputs._bind_audit_symbols()
    context = _run_output_context(tmp_path)
    outcome = dataflow_run_outputs.apply_run_output_ops(
        context=context,
        ops=(
            dataflow_run_outputs.RunOutputOp(op_id="unknown"),
            dataflow_run_outputs.RunOutputOp(op_id="console_and_violations"),
        ),
        emit_report_fn=lambda *_args, **_kwargs: ("", []),
        compute_violations_fn=lambda *_args, **_kwargs: [],
    )
    assert outcome.exit_code == 0
    assert outcome.terminal_phase == "console_and_violations"
