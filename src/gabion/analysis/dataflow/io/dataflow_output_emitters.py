# gabion:decision_protocol_module
from __future__ import annotations

from gabion.analysis.foundation.timeout_context import check_deadline, deadline_loop_iter
from gabion.cli_support.shared.output_emitters import (
    write_json_to_target,
    write_text_to_target,
)


def has_followup_actions(
    args,
    *,
    include_type_audit: bool = True,
    include_structure_tree: bool = False,
    include_structure_metrics: bool = False,
    include_decision_snapshot: bool = False,
) -> bool:
    return bool(
        (include_type_audit and args.type_audit)
        or args.synthesis_plan
        or args.synthesis_report
        or args.synthesis_protocols
        or args.refactor_plan
        or args.refactor_plan_json
        or include_structure_tree
        or include_structure_metrics
        or include_decision_snapshot
    )


def emit_sidecar_outputs(
    *,
    args,
    analysis,
    fingerprint_deadness_json,
    fingerprint_coherence_json,
    fingerprint_rewrite_plans_json,
    fingerprint_exception_obligations_json,
    fingerprint_handledness_json,
) -> None:
    for path, payload, require_content in deadline_loop_iter(
        (
            (args.fingerprint_synth_json, analysis.fingerprint_synth_registry, True),
            (args.fingerprint_provenance_json, analysis.fingerprint_provenance, True),
            (fingerprint_deadness_json, analysis.deadness_witnesses, False),
            (fingerprint_coherence_json, analysis.coherence_witnesses, False),
            (fingerprint_rewrite_plans_json, analysis.rewrite_plans, False),
            (
                fingerprint_exception_obligations_json,
                analysis.exception_obligations,
                False,
            ),
            (fingerprint_handledness_json, analysis.handledness_witnesses, False),
        )
    ):
        check_deadline()
        if not path:
            continue
        if require_content and not payload:
            continue
        write_json_to_target(path, payload)
    if args.lint:
        for line in deadline_loop_iter(analysis.lint_lines):
            check_deadline()
            print(line)


def write_json_or_stdout(target: str, payload: object) -> None:
    write_json_to_target(target, payload)


def write_text_or_stdout(target: str, payload: str) -> None:
    write_text_to_target(target, payload, ensure_trailing_newline=True)
