# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import typer

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
    if not paths:
        paths = [Path(".")]
    exclude_dirs: list[str] | None = None
    if exclude is not None:
        exclude_dirs = []
        for entry in exclude:
            check_deadline_fn()
            exclude_dirs.extend([part.strip() for part in entry.split(",") if part.strip()])
    ignore_list, transparent_list = resolved_filter_bundle.to_payload_lists()
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

    payload: JSONObject = {
        "paths": [str(p) for p in paths],
        "root": str(root),
        "config": str(config) if config is not None else None,
        "report": str(report_path),
        "dot": str(dot_path),
        "fail_on_violations": fail_on_violations,
        "no_recursive": no_recursive,
        "max_components": max_components,
        "type_audit_report": type_audit_report,
        "type_audit_max": type_audit_max,
        "exclude": exclude_dirs,
        "ignore_params": ignore_list,
        "transparent_decorators": transparent_list,
        "allow_external": allow_external,
        "strictness": strictness,
        "synthesis_plan": str(plan_path),
        "synthesis_report": True,
        "synthesis_protocols": str(protocol_path),
        "synthesis_protocols_kind": synthesis_protocols_kind,
        "synthesis_max_tier": synthesis_max_tier,
        "synthesis_min_bundle_size": synthesis_min_bundle_size,
        "synthesis_allow_singletons": synthesis_allow_singletons,
        "refactor_plan": refactor_plan,
        "refactor_plan_json": str(refactor_plan_path) if refactor_plan else None,
        "fingerprint_synth_json": str(fingerprint_synth_path),
        "fingerprint_provenance_json": str(fingerprint_provenance_path),
        "fingerprint_coherence_json": str(fingerprint_coherence_path),
        "fingerprint_rewrite_plans_json": str(fingerprint_rewrite_plans_path),
        "fingerprint_exception_obligations_json": str(
            fingerprint_exception_obligations_path
        ),
        "fingerprint_handledness_json": str(fingerprint_handledness_path),
        "aspf_trace_json": str(aspf_trace_json) if aspf_trace_json is not None else None,
        "aspf_import_trace": [str(path) for path in (aspf_import_trace or [])],
        "aspf_equivalence_against": [
            str(path) for path in (aspf_equivalence_against or [])
        ],
        "aspf_opportunities_json": (
            str(aspf_opportunities_json) if aspf_opportunities_json is not None else None
        ),
        "aspf_state_json": str(aspf_state_json) if aspf_state_json is not None else None,
        "aspf_import_state": [str(path) for path in (aspf_import_state or [])],
        "aspf_delta_jsonl": str(aspf_delta_jsonl) if aspf_delta_jsonl is not None else None,
        "aspf_semantic_surface": [
            str(surface) for surface in (aspf_semantic_surface or [])
        ],
    }
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
