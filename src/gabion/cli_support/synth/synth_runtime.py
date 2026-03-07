# gabion:decision_protocol_module
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import typer

from gabion.cli_support.shared import payload_builder
from gabion.commands import check_contract
from gabion.commands.check_contract import DataflowFilterBundle
from gabion.json_types import JSONObject
from gabion.lsp_client import run_command

Runner = Callable[..., JSONObject]


def run_synth(
    *,
    paths: list[Path] | None,
    root: Path,
    out_dir: Path,
    no_timestamp: bool,
    config: Optional[Path],
    exclude: Optional[list[str]],
    filter_bundle: DataflowFilterBundle | None,
    allow_external: Optional[bool],
    strictness: Optional[str],
    no_recursive: bool,
    max_components: int,
    type_audit_report: bool,
    type_audit_max: int,
    synthesis_max_tier: int,
    synthesis_min_bundle_size: int,
    synthesis_allow_singletons: bool,
    synthesis_protocols_kind: str,
    refactor_plan: bool,
    fail_on_violations: bool,
    aspf_trace_json: Path | None = None,
    aspf_import_trace: list[Path] | None = None,
    aspf_equivalence_against: list[Path] | None = None,
    aspf_opportunities_json: Path | None = None,
    aspf_state_json: Path | None = None,
    aspf_import_state: list[Path] | None = None,
    aspf_delta_jsonl: Path | None = None,
    aspf_semantic_surface: list[str] | None = None,
    runner: Runner = run_command,
    dispatch_command_fn: Callable[..., JSONObject],
    check_deadline_fn: Callable[[], None],
    dataflow_command: str,
) -> tuple[JSONObject, dict[str, Path], Path | None]:
    check_deadline_fn()
    resolved_filter_bundle = filter_bundle or DataflowFilterBundle(None, None)
    resolved_paths = paths or [Path(".")]
    if strictness is not None and strictness not in {"high", "low"}:
        raise typer.BadParameter("strictness must be 'high' or 'low'")
    if synthesis_protocols_kind not in {"dataclass", "protocol", "contextvar"}:
        raise typer.BadParameter(
            "synthesis-protocols-kind must be 'dataclass', 'protocol', or 'contextvar'"
        )

    output_root = out_dir
    timestamp = None
    if not no_timestamp:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_root = out_dir / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "LATEST.txt").write_text(timestamp)
    output_root.mkdir(parents=True, exist_ok=True)

    report_path = output_root / "dataflow_report.md"
    dot_path = output_root / "dataflow_graph.dot"
    plan_path = output_root / "synthesis_plan.json"
    protocol_path = output_root / "protocol_stubs.py"
    refactor_plan_path = output_root / "refactor_plan.json"
    fingerprint_synth_path = output_root / "fingerprint_synth.json"
    fingerprint_provenance_path = output_root / "fingerprint_provenance.json"
    fingerprint_coherence_path = output_root / "fingerprint_coherence.json"
    fingerprint_rewrite_plans_path = output_root / "fingerprint_rewrite_plans.json"
    fingerprint_exception_obligations_path = (
        output_root / "fingerprint_exception_obligations.json"
    )
    fingerprint_handledness_path = output_root / "fingerprint_handledness.json"

    payload = payload_builder.build_synth_payload(
        options=check_contract.DataflowPayloadCommonOptions(
            paths=resolved_paths,
            root=root,
            config=config,
            report=report_path,
            fail_on_violations=fail_on_violations,
            fail_on_type_ambiguities=False,
            baseline=None,
            baseline_write=None,
            decision_snapshot=None,
            exclude=exclude,
            filter_bundle=resolved_filter_bundle,
            allow_external=allow_external,
            strictness=strictness,
            lint=False,
            aspf_trace_json=aspf_trace_json,
            aspf_import_trace=aspf_import_trace,
            aspf_equivalence_against=aspf_equivalence_against,
            aspf_opportunities_json=aspf_opportunities_json,
            aspf_state_json=aspf_state_json,
            aspf_import_state=aspf_import_state,
            aspf_delta_jsonl=aspf_delta_jsonl,
            aspf_semantic_surface=aspf_semantic_surface,
        ),
        synth_options=payload_builder.SynthPayloadOptions(
            no_recursive=no_recursive,
            max_components=max_components,
            type_audit_report=type_audit_report,
            type_audit_max=type_audit_max,
            synthesis_plan=plan_path,
            synthesis_report=True,
            synthesis_protocols=protocol_path,
            synthesis_protocols_kind=synthesis_protocols_kind,
            synthesis_max_tier=synthesis_max_tier,
            synthesis_min_bundle_size=synthesis_min_bundle_size,
            synthesis_allow_singletons=synthesis_allow_singletons,
            refactor_plan=refactor_plan,
            refactor_plan_json=refactor_plan_path if refactor_plan else None,
            fingerprint_synth_json=fingerprint_synth_path,
            fingerprint_provenance_json=fingerprint_provenance_path,
            fingerprint_coherence_json=fingerprint_coherence_path,
            fingerprint_rewrite_plans_json=fingerprint_rewrite_plans_path,
            fingerprint_exception_obligations_json=fingerprint_exception_obligations_path,
            fingerprint_handledness_json=fingerprint_handledness_path,
        ),
        build_dataflow_payload_common_fn=check_contract.build_dataflow_payload_common,
    )
    payload["dot"] = str(dot_path)
    result = dispatch_command_fn(
        command=dataflow_command,
        payload=payload,
        root=root,
        runner=runner,
    )
    paths_out = {
        "report": report_path,
        "dot": dot_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "refactor": refactor_plan_path,
        "fingerprint_synth": fingerprint_synth_path,
        "fingerprint_provenance": fingerprint_provenance_path,
        "fingerprint_coherence": fingerprint_coherence_path,
        "fingerprint_rewrite_plans": fingerprint_rewrite_plans_path,
        "fingerprint_exception_obligations": fingerprint_exception_obligations_path,
        "fingerprint_handledness": fingerprint_handledness_path,
        "output_root": output_root,
    }
    return result, paths_out, timestamp
