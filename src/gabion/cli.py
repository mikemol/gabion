from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from contextlib import ExitStack, contextmanager
from typing import Callable, Generator, List, Mapping, Optional, TypeAlias
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys

from click.core import ParameterSource
import typer
from gabion.analysis.aspf import Forest
from gabion.analysis.timeout_context import (
    Deadline,
    check_deadline,
    deadline_loop_iter,
    deadline_clock_scope,
    deadline_scope,
    forest_scope,
    render_deadline_profile_markdown,
)
from gabion.deadline_clock import GasMeter

DATAFLOW_COMMAND = "gabion.dataflowAudit"
SYNTHESIS_COMMAND = "gabion.synthesisPlan"
REFACTOR_COMMAND = "gabion.refactorProtocol"
STRUCTURE_DIFF_COMMAND = "gabion.structureDiff"
STRUCTURE_REUSE_COMMAND = "gabion.structureReuse"
DECISION_DIFF_COMMAND = "gabion.decisionDiff"
from gabion.lsp_client import (
    CommandRequest,
    run_command,
    run_command_direct,
    _env_timeout_ticks,
    _has_env_timeout,
)
from gabion.json_types import JSONObject
app = typer.Typer(add_completion=False)
Runner: TypeAlias = Callable[..., JSONObject]
DEFAULT_RUNNER: Runner = run_command

_LINT_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d+):(?P<col>\d+):\s*(?P<rest>.*)$")

_DEFAULT_TIMEOUT_TICKS = 100
_DEFAULT_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_CHECK_REPORT_REL_PATH = Path("artifacts/audit_reports/dataflow_report.md")
_DEFAULT_TIMEOUT_PROGRESS_REPORT_REL_PATH = Path(
    "artifacts/audit_reports/timeout_progress.md"
)
_DATAFLOW_AUDIT_ALIAS_MESSAGE = (
    "`dataflow-audit` is deprecated; use `check --profile raw`."
)
_DATAFLOW_AUDIT_MIGRATION_EPILOG = (
    f"DEPRECATION: {_DATAFLOW_AUDIT_ALIAS_MESSAGE}\n"
    "Migration map:\n"
    "  gabion dataflow-audit <paths> -> gabion check --profile raw <paths>\n"
    "  --emit-decision-snapshot -> --decision-snapshot"
)


def _cli_timeout_ticks() -> tuple[int, int]:
    if _has_env_timeout():
        return _env_timeout_ticks()
    return _DEFAULT_TIMEOUT_TICKS, _DEFAULT_TIMEOUT_TICK_NS


def _cli_deadline() -> Deadline:
    ticks, tick_ns = _cli_timeout_ticks()
    return Deadline.from_timeout_ticks(ticks, tick_ns)


def _resolve_check_report_path(report: Path | None, *, root: Path) -> Path:
    if report is not None:
        return report
    return root / _DEFAULT_CHECK_REPORT_REL_PATH


@contextmanager
def _cli_deadline_scope():
    ticks, _tick_ns = _cli_timeout_ticks()
    with ExitStack() as stack:
        stack.enter_context(forest_scope(Forest()))
        stack.enter_context(deadline_scope(_cli_deadline()))
        stack.enter_context(deadline_clock_scope(GasMeter(limit=int(ticks))))
        yield


@dataclass(frozen=True)
class DataflowAuditRequest:
    ctx: typer.Context
    args: List[str] | None = None
    runner: Runner | None = None


@dataclass(frozen=True)
class CheckArtifactFlags:
    emit_test_obsolescence: bool
    emit_test_evidence_suggestions: bool
    emit_call_clusters: bool
    emit_call_cluster_consolidation: bool
    emit_test_annotation_drift: bool


@dataclass(frozen=True)
class SnapshotDiffRequest:
    baseline: Path
    current: Path

    def to_payload(self) -> JSONObject:
        return {"baseline": str(self.baseline), "current": str(self.current)}


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _split_csv_entries(entries: List[str]) -> list[str]:
    check_deadline()
    merged: list[str] = []
    for entry in entries:
        check_deadline()
        merged.extend([part.strip() for part in entry.split(",") if part.strip()])
    return merged


def _split_csv(value: str) -> list[str]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    return items


def _parse_lint_line(line: str) -> dict[str, object] | None:
    match = _LINT_RE.match(line.strip())
    if not match:
        return None
    line_no = int(match.group("line"))
    col_no = int(match.group("col"))
    rest = match.group("rest").strip()
    if not rest:
        return None
    code, _, message = rest.partition(" ")
    return {
        "path": match.group("path"),
        "line": line_no,
        "col": col_no,
        "code": code,
        "message": message,
        "severity": "warning",
    }


def _collect_lint_entries(lines: list[str]) -> list[dict[str, object]]:
    check_deadline()
    entries: list[dict[str, object]] = []
    for line in lines:
        check_deadline()
        parsed = _parse_lint_line(line)
        if parsed is not None:
            entries.append(parsed)
    return entries


def _write_lint_jsonl(target: str, entries: list[dict[str, object]]) -> None:
    payload = "\n".join(json.dumps(entry, sort_keys=True) for entry in entries)
    if target == "-":
        typer.echo(payload)
    else:
        Path(target).write_text(payload + ("\n" if payload else ""))


def _write_lint_sarif(target: str, entries: list[dict[str, object]]) -> None:
    check_deadline()
    rules: dict[str, dict[str, object]] = {}
    results: list[dict[str, object]] = []
    for entry in entries:
        check_deadline()
        code = str(entry.get("code") or "GABION")
        message = str(entry.get("message") or "").strip()
        path = str(entry.get("path") or "")
        line = int(entry.get("line") or 1)
        col = int(entry.get("col") or 1)
        if code not in rules:
            rules[code] = {
                "id": code,
                "name": code,
                "shortDescription": {"text": code},
            }
        results.append(
            {
                "ruleId": code,
                "level": "warning",
                "message": {"text": message or code},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": path},
                            "region": {
                                "startLine": line,
                                "startColumn": col,
                            },
                        }
                    }
                ],
            }
        )
    sarif = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {"driver": {"name": "gabion", "rules": list(rules.values())}},
                "results": results,
            }
        ],
    }
    payload = json.dumps(sarif, indent=2, sort_keys=True)
    if target == "-":
        typer.echo(payload)
    else:
        Path(target).write_text(payload + "\n")


def _emit_lint_outputs(
    lint_lines: list[str],
    *,
    lint: bool,
    lint_jsonl: Optional[Path],
    lint_sarif: Optional[Path],
) -> None:
    check_deadline()
    if lint:
        for line in lint_lines:
            check_deadline()
            typer.echo(line)
    if lint_jsonl or lint_sarif:
        entries = _collect_lint_entries(lint_lines)
        if lint_jsonl is not None:
            _write_lint_jsonl(str(lint_jsonl), entries)
        if lint_sarif is not None:
            _write_lint_sarif(str(lint_sarif), entries)


def _emit_timeout_profile_artifacts(
    result: Mapping[str, object],
    *,
    root: Path,
) -> None:
    timeout_context = result.get("timeout_context")
    if not isinstance(timeout_context, Mapping):
        return
    profile = timeout_context.get("deadline_profile")
    if not isinstance(profile, Mapping):
        return
    artifact_dir = root / "artifacts" / "out"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    profile_json_path = artifact_dir / "deadline_profile.json"
    profile_md_path = artifact_dir / "deadline_profile.md"
    profile_json_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    profile_md_path.write_text(
        render_deadline_profile_markdown(profile) + "\n",
        encoding="utf-8",
    )
    typer.echo(f"Wrote deadline profile JSON: {profile_json_path}")
    typer.echo(f"Wrote deadline profile markdown: {profile_md_path}")


def _render_timeout_progress_markdown(
    *,
    analysis_state: str | None,
    progress: Mapping[str, object],
    deadline_profile: Mapping[str, object] | None = None,
) -> str:
    lines = ["# Timeout Progress", ""]
    if analysis_state:
        lines.append(f"- `analysis_state`: `{analysis_state}`")
    classification = progress.get("classification")
    if isinstance(classification, str):
        lines.append(f"- `classification`: `{classification}`")
    retry_recommended = progress.get("retry_recommended")
    if isinstance(retry_recommended, bool):
        lines.append(f"- `retry_recommended`: `{retry_recommended}`")
    resume_supported = progress.get("resume_supported")
    if isinstance(resume_supported, bool):
        lines.append(f"- `resume_supported`: `{resume_supported}`")
    ticks_consumed = progress.get("ticks_consumed")
    if isinstance(ticks_consumed, int):
        lines.append(f"- `ticks_consumed`: `{ticks_consumed}`")
    tick_limit = progress.get("tick_limit")
    if isinstance(tick_limit, int):
        lines.append(f"- `tick_limit`: `{tick_limit}`")
    ticks_remaining = progress.get("ticks_remaining")
    if isinstance(ticks_remaining, int):
        lines.append(f"- `ticks_remaining`: `{ticks_remaining}`")
    progress_ticks_per_ns = progress.get("ticks_per_ns")
    if isinstance(progress_ticks_per_ns, (int, float)):
        lines.append(f"- `ticks_per_ns`: `{float(progress_ticks_per_ns):.9f}`")
    if isinstance(deadline_profile, Mapping):
        if not isinstance(progress_ticks_per_ns, (int, float)):
            ticks_per_ns = deadline_profile.get("ticks_per_ns")
            if isinstance(ticks_per_ns, (int, float)):
                lines.append(f"- `ticks_per_ns`: `{float(ticks_per_ns):.9f}`")
    resume = progress.get("resume")
    if isinstance(resume, Mapping):
        token = resume.get("resume_token")
        if isinstance(token, Mapping):
            lines.append("")
            lines.append("## Resume Token")
            lines.append("")
            for key in deadline_loop_iter(
                (
                    "phase",
                    "checkpoint_path",
                    "completed_files",
                    "remaining_files",
                    "total_files",
                    "witness_digest",
                )
            ):
                value = token.get(key)
                if value is None:
                    continue
                lines.append(f"- `{key}`: `{value}`")
    obligations = progress.get("incremental_obligations")
    if isinstance(obligations, list) and obligations:
        lines.append("")
        lines.append("## Incremental Obligations")
        lines.append("")
        for entry in deadline_loop_iter(obligations):
            if not isinstance(entry, Mapping):
                continue
            status = str(entry.get("status", "UNKNOWN") or "UNKNOWN")
            contract = str(entry.get("contract", "") or "")
            kind = str(entry.get("kind", "") or "")
            detail = str(entry.get("detail", "") or "")
            section_id = str(entry.get("section_id", "") or "")
            section_suffix = f" section={section_id}" if section_id else ""
            lines.append(
                f"- `{status}` `{contract}` `{kind}`{section_suffix}: {detail}"
            )
    return "\n".join(lines)


def _emit_timeout_progress_artifacts(
    result: Mapping[str, object],
    *,
    root: Path,
) -> None:
    timeout_context = result.get("timeout_context")
    if not isinstance(timeout_context, Mapping):
        return
    progress = timeout_context.get("progress")
    if not isinstance(progress, Mapping):
        return
    profile = timeout_context.get("deadline_profile")
    profile_mapping = profile if isinstance(profile, Mapping) else None
    progress_md_path = root / _DEFAULT_TIMEOUT_PROGRESS_REPORT_REL_PATH
    progress_json_path = progress_md_path.with_suffix(".json")
    progress_md_path.parent.mkdir(parents=True, exist_ok=True)
    payload: JSONObject = {
        "analysis_state": str(result.get("analysis_state", "")),
        "progress": {str(key): progress[key] for key in progress},
    }
    progress_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with _cli_deadline_scope():
        rendered = _render_timeout_progress_markdown(
            analysis_state=str(result.get("analysis_state", "")),
            progress=progress,
            deadline_profile=profile_mapping,
        )
    progress_md_path.write_text(
        rendered + "\n",
        encoding="utf-8",
    )
    typer.echo(f"Wrote timeout progress JSON: {progress_json_path}")
    typer.echo(f"Wrote timeout progress markdown: {progress_md_path}")


def _build_dataflow_payload_common(
    *,
    paths: list[Path],
    root: Path,
    config: Path | None,
    report: Path | None,
    fail_on_violations: bool,
    fail_on_type_ambiguities: bool,
    baseline: Path | None,
    baseline_write: bool | None,
    decision_snapshot: Path | None,
    exclude: list[str] | None,
    ignore_params_csv: str | None,
    transparent_decorators_csv: str | None,
    allow_external: bool | None,
    strictness: str | None,
    lint: bool,
    resume_checkpoint: Path | None,
    emit_timeout_progress_report: bool,
    resume_on_timeout: int,
    deadline_profile: bool = True,
) -> JSONObject:
    # dataflow-bundle: ignore_params_csv, transparent_decorators_csv
    # dataflow-bundle: deadline_profile, emit_timeout_progress_report
    if strictness is not None and strictness not in {"high", "low"}:
        raise typer.BadParameter("strictness must be 'high' or 'low'")
    exclude_dirs = _split_csv_entries(exclude) if exclude is not None else None
    ignore_list = _split_csv(ignore_params_csv) if ignore_params_csv is not None else None
    transparent_list = (
        _split_csv(transparent_decorators_csv)
        if transparent_decorators_csv is not None
        else None
    )
    payload: JSONObject = {
        "paths": [str(p) for p in paths],
        "root": str(root),
        "config": str(config) if config is not None else None,
        "report": str(report) if report is not None else None,
        "fail_on_violations": fail_on_violations,
        "fail_on_type_ambiguities": fail_on_type_ambiguities,
        "baseline": str(baseline) if baseline is not None else None,
        "baseline_write": baseline_write,
        "decision_snapshot": str(decision_snapshot) if decision_snapshot is not None else None,
        "exclude": exclude_dirs,
        "ignore_params": ignore_list,
        "transparent_decorators": transparent_list,
        "allow_external": allow_external,
        "strictness": strictness,
        "lint": lint,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "emit_timeout_progress_report": bool(emit_timeout_progress_report),
        "resume_on_timeout": int(resume_on_timeout),
        "deadline_profile": bool(deadline_profile),
    }
    return payload


def build_check_payload(
    *,
    paths: Optional[List[Path]],
    report: Optional[Path],
    fail_on_violations: bool,
    root: Path,
    config: Optional[Path],
    baseline: Optional[Path],
    baseline_write: bool,
    decision_snapshot: Optional[Path],
    artifact_flags: CheckArtifactFlags,
    emit_test_obsolescence_state: bool,
    test_obsolescence_state: Optional[Path],
    emit_test_obsolescence_delta: bool,
    test_annotation_drift_state: Optional[Path],
    emit_test_annotation_drift_delta: bool,
    write_test_annotation_drift_baseline: bool,
    write_test_obsolescence_baseline: bool,
    emit_ambiguity_delta: bool,
    emit_ambiguity_state: bool,
    ambiguity_state: Optional[Path],
    write_ambiguity_baseline: bool,
    exclude: Optional[List[str]],
    ignore_params_csv: Optional[str],
    transparent_decorators_csv: Optional[str],
    allow_external: Optional[bool],
    strictness: Optional[str],
    fail_on_type_ambiguities: bool,
    lint: bool,
    resume_checkpoint: Optional[Path] = None,
    emit_timeout_progress_report: bool = False,
    resume_on_timeout: int = 0,
    analysis_tick_limit: int | None = None,
) -> JSONObject:
    if not paths:
        paths = [Path(".")]
    if emit_test_obsolescence_delta and write_test_obsolescence_baseline:
        raise typer.BadParameter(
            "Use --emit-test-obsolescence-delta or --write-test-obsolescence-baseline, not both."
        )
    if emit_test_obsolescence_state and test_obsolescence_state is not None:
        raise typer.BadParameter(
            "Use --emit-test-obsolescence-state or --test-obsolescence-state, not both."
        )
    if emit_test_annotation_drift_delta and write_test_annotation_drift_baseline:
        raise typer.BadParameter(
            "Use --emit-test-annotation-drift-delta or --write-test-annotation-drift-baseline, not both."
        )
    if emit_ambiguity_delta and write_ambiguity_baseline:
        raise typer.BadParameter(
            "Use --emit-ambiguity-delta or --write-ambiguity-baseline, not both."
        )
    if emit_ambiguity_state and ambiguity_state is not None:
        raise typer.BadParameter(
            "Use --emit-ambiguity-state or --ambiguity-state, not both."
        )
    baseline_write_value = bool(baseline is not None and baseline_write)
    payload = _build_dataflow_payload_common(
        paths=paths,
        root=root,
        config=config,
        report=report,
        fail_on_violations=fail_on_violations,
        fail_on_type_ambiguities=fail_on_type_ambiguities,
        baseline=baseline,
        baseline_write=baseline_write_value,
        decision_snapshot=decision_snapshot,
        exclude=exclude,
        ignore_params_csv=ignore_params_csv,
        transparent_decorators_csv=transparent_decorators_csv,
        allow_external=allow_external,
        strictness=strictness,
        lint=lint,
        resume_checkpoint=resume_checkpoint,
        emit_timeout_progress_report=emit_timeout_progress_report,
        resume_on_timeout=resume_on_timeout,
    )
    payload.update(
        {
            "emit_test_obsolescence": artifact_flags.emit_test_obsolescence,
            "emit_test_obsolescence_state": emit_test_obsolescence_state,
            "test_obsolescence_state": str(test_obsolescence_state)
            if test_obsolescence_state is not None
            else None,
            "emit_test_obsolescence_delta": emit_test_obsolescence_delta,
            "emit_test_evidence_suggestions": artifact_flags.emit_test_evidence_suggestions,
            "emit_call_clusters": artifact_flags.emit_call_clusters,
            "emit_call_cluster_consolidation": artifact_flags.emit_call_cluster_consolidation,
            "emit_test_annotation_drift": artifact_flags.emit_test_annotation_drift,
            "test_annotation_drift_state": str(test_annotation_drift_state)
            if test_annotation_drift_state is not None
            else None,
            "emit_test_annotation_drift_delta": emit_test_annotation_drift_delta,
            "write_test_annotation_drift_baseline": write_test_annotation_drift_baseline,
            "write_test_obsolescence_baseline": write_test_obsolescence_baseline,
            "emit_ambiguity_delta": emit_ambiguity_delta,
            "emit_ambiguity_state": emit_ambiguity_state,
            "ambiguity_state": str(ambiguity_state) if ambiguity_state is not None else None,
            "write_ambiguity_baseline": write_ambiguity_baseline,
            "type_audit": True if fail_on_type_ambiguities else None,
            "analysis_tick_limit": int(analysis_tick_limit)
            if analysis_tick_limit is not None
            else None,
        }
    )
    return payload


def parse_dataflow_args(argv: list[str]) -> argparse.Namespace:
    parser = dataflow_cli_parser()
    return parser.parse_args(argv)


def parse_dataflow_args_or_exit(
    argv: list[str],
    *,
    parser_fn: Callable[[], argparse.ArgumentParser] | None = None,
) -> argparse.Namespace:
    parser = (parser_fn or dataflow_cli_parser)()
    if any(arg in {"-h", "--help"} for arg in argv):
        parser.print_help()
        raise typer.Exit(code=0)
    try:
        return parser.parse_args(argv)
    except SystemExit as exc:
        raise typer.Exit(code=int(exc.code))


def build_dataflow_payload(opts: argparse.Namespace) -> JSONObject:
    payload = _build_dataflow_payload_common(
        paths=opts.paths,
        root=Path(opts.root),
        config=Path(opts.config) if opts.config is not None else None,
        report=Path(opts.report) if opts.report else None,
        fail_on_violations=opts.fail_on_violations,
        fail_on_type_ambiguities=opts.fail_on_type_ambiguities,
        baseline=Path(opts.baseline) if opts.baseline else None,
        baseline_write=opts.baseline_write if opts.baseline else None,
        decision_snapshot=Path(opts.emit_decision_snapshot)
        if opts.emit_decision_snapshot
        else None,
        exclude=opts.exclude,
        ignore_params_csv=opts.ignore_params,
        transparent_decorators_csv=opts.transparent_decorators,
        allow_external=opts.allow_external,
        strictness=opts.strictness,
        lint=bool(opts.lint or opts.lint_jsonl or opts.lint_sarif),
        resume_checkpoint=Path(opts.resume_checkpoint)
        if opts.resume_checkpoint
        else None,
        emit_timeout_progress_report=bool(opts.emit_timeout_progress_report),
        resume_on_timeout=max(int(opts.resume_on_timeout), 0),
    )
    payload.update(
        {
        "dot": opts.dot,
        "no_recursive": opts.no_recursive,
        "max_components": opts.max_components,
        "type_audit": opts.type_audit,
        "type_audit_report": opts.type_audit_report,
        "type_audit_max": opts.type_audit_max,
        "synthesis_plan": str(opts.synthesis_plan) if opts.synthesis_plan else None,
        "synthesis_report": opts.synthesis_report,
        "synthesis_max_tier": opts.synthesis_max_tier,
        "synthesis_min_bundle_size": opts.synthesis_min_bundle_size,
        "synthesis_allow_singletons": opts.synthesis_allow_singletons,
        "synthesis_protocols": str(opts.synthesis_protocols)
        if opts.synthesis_protocols
        else None,
        "synthesis_protocols_kind": opts.synthesis_protocols_kind,
        "refactor_plan": opts.refactor_plan,
        "refactor_plan_json": str(opts.refactor_plan_json)
        if opts.refactor_plan_json
        else None,
        "fingerprint_synth_json": str(opts.fingerprint_synth_json)
        if opts.fingerprint_synth_json
        else None,
        "fingerprint_provenance_json": str(opts.fingerprint_provenance_json)
        if opts.fingerprint_provenance_json
        else None,
        "fingerprint_deadness_json": str(opts.fingerprint_deadness_json)
        if opts.fingerprint_deadness_json
        else None,
        "fingerprint_coherence_json": str(opts.fingerprint_coherence_json)
        if opts.fingerprint_coherence_json
        else None,
        "fingerprint_rewrite_plans_json": str(opts.fingerprint_rewrite_plans_json)
        if opts.fingerprint_rewrite_plans_json
        else None,
        "fingerprint_exception_obligations_json": str(
            opts.fingerprint_exception_obligations_json
        )
        if opts.fingerprint_exception_obligations_json
        else None,
        "fingerprint_handledness_json": str(opts.fingerprint_handledness_json)
        if opts.fingerprint_handledness_json
        else None,
        "synthesis_merge_overlap": opts.synthesis_merge_overlap,
        "structure_tree": str(opts.emit_structure_tree)
        if opts.emit_structure_tree
        else None,
        "structure_metrics": str(opts.emit_structure_metrics)
        if opts.emit_structure_metrics
        else None,
        }
    )
    return payload


def build_refactor_payload(
    *,
    input_payload: Optional[JSONObject] = None,
    protocol_name: Optional[str],
    bundle: Optional[List[str]],
    field: Optional[List[str]],
    target_path: Optional[Path],
    target_functions: Optional[List[str]],
    compatibility_shim: bool,
    rationale: Optional[str],
) -> JSONObject:
    check_deadline()
    if input_payload is not None:
        return input_payload
    if protocol_name is None or target_path is None:
        raise typer.BadParameter(
            "Provide --protocol-name and --target-path or use --input."
        )
    field_specs: list[dict[str, str | None]] = []
    for spec in field or []:
        check_deadline()
        name, _, hint = spec.partition(":")
        name = name.strip()
        if not name:
            continue
        type_hint = hint.strip() or None
        field_specs.append({"name": name, "type_hint": type_hint})
    if not bundle and field_specs:
        bundle = [spec["name"] for spec in field_specs]
    return {
        "protocol_name": protocol_name,
        "bundle": bundle or [],
        "fields": field_specs,
        "target_path": str(target_path),
        "target_functions": target_functions or [],
        "compatibility_shim": compatibility_shim,
        "rationale": rationale,
    }


def dispatch_command(
    *,
    command: str,
    payload: JSONObject,
    root: Path = Path("."),
    runner: Runner = run_command,
    process_factory: Callable[..., subprocess.Popen] | None = None,
) -> JSONObject:
    ticks, tick_ns = _cli_timeout_ticks()
    if (
        "analysis_timeout_ticks" not in payload
        and "analysis_timeout_ms" not in payload
        and "analysis_timeout_seconds" not in payload
    ):
        payload = dict(payload)
        payload["analysis_timeout_ticks"] = int(ticks)
        payload["analysis_timeout_tick_ns"] = int(tick_ns)
    request = CommandRequest(command, [payload])
    resolved = runner
    if runner is run_command:
        flag = os.getenv("GABION_DIRECT_RUN", "").strip().lower()
        if flag in {"1", "true", "yes", "on"}:
            resolved = run_command_direct
    if resolved is run_command:
        factory = process_factory or subprocess.Popen
        return resolved(
            request,
            root=root,
            timeout_ticks=ticks,
            timeout_tick_ns=tick_ns,
            process_factory=factory,
        )
    return resolved(request, root=root)


def run_check(
    *,
    paths: Optional[List[Path]],
    report: Optional[Path],
    fail_on_violations: bool,
    root: Path,
    config: Optional[Path],
    baseline: Optional[Path],
    baseline_write: bool,
    decision_snapshot: Optional[Path],
    artifact_flags: CheckArtifactFlags,
    emit_test_obsolescence_state: bool,
    test_obsolescence_state: Optional[Path],
    emit_test_obsolescence_delta: bool,
    test_annotation_drift_state: Optional[Path],
    emit_test_annotation_drift_delta: bool,
    write_test_annotation_drift_baseline: bool,
    write_test_obsolescence_baseline: bool,
    emit_ambiguity_delta: bool,
    emit_ambiguity_state: bool,
    ambiguity_state: Optional[Path],
    write_ambiguity_baseline: bool,
    exclude: Optional[List[str]],
    ignore_params_csv: Optional[str],
    transparent_decorators_csv: Optional[str],
    allow_external: Optional[bool],
    strictness: Optional[str],
    fail_on_type_ambiguities: bool,
    lint: bool,
    resume_checkpoint: Optional[Path] = None,
    emit_timeout_progress_report: bool = False,
    resume_on_timeout: int = 0,
    analysis_tick_limit: int | None = None,
    runner: Runner = run_command,
) -> JSONObject:
    # dataflow-bundle: ignore_params_csv, transparent_decorators_csv
    resolved_report = _resolve_check_report_path(report, root=root)
    resolved_report.parent.mkdir(parents=True, exist_ok=True)
    payload = build_check_payload(
        paths=paths,
        report=resolved_report,
        fail_on_violations=fail_on_violations,
        root=root,
        config=config,
        baseline=baseline,
        baseline_write=baseline_write if baseline is not None else False,
        decision_snapshot=decision_snapshot,
        artifact_flags=artifact_flags,
        emit_test_obsolescence_state=emit_test_obsolescence_state,
        test_obsolescence_state=test_obsolescence_state,
        emit_test_obsolescence_delta=emit_test_obsolescence_delta,
        test_annotation_drift_state=test_annotation_drift_state,
        emit_test_annotation_drift_delta=emit_test_annotation_drift_delta,
        write_test_annotation_drift_baseline=write_test_annotation_drift_baseline,
        write_test_obsolescence_baseline=write_test_obsolescence_baseline,
        emit_ambiguity_delta=emit_ambiguity_delta,
        emit_ambiguity_state=emit_ambiguity_state,
        ambiguity_state=ambiguity_state,
        write_ambiguity_baseline=write_ambiguity_baseline,
        exclude=exclude,
        ignore_params_csv=ignore_params_csv,
        transparent_decorators_csv=transparent_decorators_csv,
        allow_external=allow_external,
        strictness=strictness,
        fail_on_type_ambiguities=fail_on_type_ambiguities,
        lint=lint,
        resume_checkpoint=resume_checkpoint,
        emit_timeout_progress_report=emit_timeout_progress_report,
        resume_on_timeout=resume_on_timeout,
        analysis_tick_limit=analysis_tick_limit,
    )
    return dispatch_command(command=DATAFLOW_COMMAND, payload=payload, root=root, runner=runner)


def _run_with_timeout_retries(
    *,
    run_once: Callable[[], JSONObject],
    root: Path,
    emit_timeout_progress_report: bool,
    resume_on_timeout: int,
    emit_timeout_profile_artifacts_fn: Callable[..., None] = _emit_timeout_profile_artifacts,
    emit_timeout_progress_artifacts_fn: Callable[..., None] = _emit_timeout_progress_artifacts,
    echo_fn: Callable[[str], None] = typer.echo,
) -> JSONObject:
    attempt = 0
    result: JSONObject = {}
    while True:
        with _cli_deadline_scope():
            check_deadline()
            result = run_once()
        if result.get("timeout") is not True:
            return result
        emit_timeout_profile_artifacts_fn(result, root=root)
        if emit_timeout_progress_report:
            emit_timeout_progress_artifacts_fn(result, root=root)
        if (
            attempt < max(int(resume_on_timeout), 0)
            and str(result.get("analysis_state", "")) == "timed_out_progress_resume"
        ):
            attempt += 1
            echo_fn(f"Retrying after timeout with progress ({attempt}/{resume_on_timeout})...")
            continue
        raise typer.Exit(code=int(result.get("exit_code", 2)))


def _emit_dataflow_result_outputs(result: JSONObject, opts: argparse.Namespace) -> None:
    with _cli_deadline_scope():
        lint_lines = result.get("lint_lines", []) or []
        _emit_lint_outputs(
            lint_lines,
            lint=opts.lint,
            lint_jsonl=opts.lint_jsonl,
            lint_sarif=opts.lint_sarif,
        )
        if opts.type_audit:
            suggestions = result.get("type_suggestions", [])
            ambiguities = result.get("type_ambiguities", [])
            if suggestions:
                typer.echo("Type tightening candidates:")
                for line in suggestions[: opts.type_audit_max]:
                    check_deadline()
                    typer.echo(f"- {line}")
            if ambiguities:
                typer.echo("Type ambiguities (conflicting downstream expectations):")
                for line in ambiguities[: opts.type_audit_max]:
                    check_deadline()
                    typer.echo(f"- {line}")
        if opts.dot == "-" and "dot" in result:
            typer.echo(result["dot"])
        if opts.synthesis_plan == "-" and "synthesis_plan" in result:
            typer.echo(json.dumps(result["synthesis_plan"], indent=2, sort_keys=True))
        if opts.synthesis_protocols == "-" and "synthesis_protocols" in result:
            typer.echo(result["synthesis_protocols"])
        if opts.refactor_plan_json == "-" and "refactor_plan" in result:
            typer.echo(json.dumps(result["refactor_plan"], indent=2, sort_keys=True))
        if (
            opts.fingerprint_synth_json == "-"
            and "fingerprint_synth_registry" in result
        ):
            typer.echo(
                json.dumps(
                    result["fingerprint_synth_registry"], indent=2, sort_keys=True
                )
            )
        if (
            opts.fingerprint_provenance_json == "-"
            and "fingerprint_provenance" in result
        ):
            typer.echo(
                json.dumps(
                    result["fingerprint_provenance"], indent=2, sort_keys=True
                )
            )
        if opts.fingerprint_deadness_json == "-" and "fingerprint_deadness" in result:
            typer.echo(
                json.dumps(
                    result["fingerprint_deadness"], indent=2, sort_keys=True
                )
            )
        if opts.fingerprint_coherence_json == "-" and "fingerprint_coherence" in result:
            typer.echo(
                json.dumps(
                    result["fingerprint_coherence"], indent=2, sort_keys=True
                )
            )
        if (
            opts.fingerprint_rewrite_plans_json == "-"
            and "fingerprint_rewrite_plans" in result
        ):
            typer.echo(
                json.dumps(
                    result["fingerprint_rewrite_plans"], indent=2, sort_keys=True
                )
            )
        if (
            opts.fingerprint_exception_obligations_json == "-"
            and "fingerprint_exception_obligations" in result
        ):
            typer.echo(
                json.dumps(
                    result["fingerprint_exception_obligations"],
                    indent=2,
                    sort_keys=True,
                )
            )
        if (
            opts.fingerprint_handledness_json == "-"
            and "fingerprint_handledness" in result
        ):
            typer.echo(
                json.dumps(
                    result["fingerprint_handledness"], indent=2, sort_keys=True
                )
            )
        if opts.emit_structure_tree == "-" and "structure_tree" in result:
            typer.echo(json.dumps(result["structure_tree"], indent=2, sort_keys=True))
        if opts.emit_structure_metrics == "-" and "structure_metrics" in result:
            typer.echo(json.dumps(result["structure_metrics"], indent=2, sort_keys=True))
        if opts.emit_decision_snapshot == "-" and "decision_snapshot" in result:
            typer.echo(json.dumps(result["decision_snapshot"], indent=2, sort_keys=True))


def _param_is_command_line(ctx: typer.Context, param: str) -> bool:
    return ctx.get_parameter_source(param) is ParameterSource.COMMANDLINE


def _raw_profile_unsupported_flags(ctx: typer.Context) -> list[str]:
    strict_only_flags = {
        "emit_test_obsolescence": "--emit-test-obsolescence",
        "emit_test_obsolescence_state": "--emit-test-obsolescence-state",
        "test_obsolescence_state": "--test-obsolescence-state",
        "emit_test_obsolescence_delta": "--emit-test-obsolescence-delta",
        "emit_test_evidence_suggestions": "--emit-test-evidence-suggestions",
        "emit_call_clusters": "--emit-call-clusters",
        "emit_call_cluster_consolidation": "--emit-call-cluster-consolidation",
        "emit_test_annotation_drift": "--emit-test-annotation-drift",
        "test_annotation_drift_state": "--test-annotation-drift-state",
        "emit_test_annotation_drift_delta": "--emit-test-annotation-drift-delta",
        "write_test_annotation_drift_baseline": "--write-test-annotation-drift-baseline",
        "write_test_obsolescence_baseline": "--write-test-obsolescence-baseline",
        "emit_ambiguity_delta": "--emit-ambiguity-delta",
        "emit_ambiguity_state": "--emit-ambiguity-state",
        "ambiguity_state": "--ambiguity-state",
        "write_ambiguity_baseline": "--write-ambiguity-baseline",
        "analysis_tick_limit": "--analysis-tick-limit",
    }
    return [
        flag
        for param, flag in strict_only_flags.items()
        if _param_is_command_line(ctx, param)
    ]


def _check_raw_profile_args(
    *,
    ctx: typer.Context,
    paths: Optional[List[Path]],
    report: Optional[Path],
    fail_on_violations: bool,
    root: Path,
    config: Optional[Path],
    decision_snapshot: Optional[Path],
    baseline: Optional[Path],
    baseline_write: bool,
    exclude: Optional[List[str]],
    ignore_params_csv: Optional[str],
    transparent_decorators_csv: Optional[str],
    allow_external: Optional[bool],
    strictness: Optional[str],
    resume_checkpoint: Optional[Path],
    emit_timeout_progress_report: bool,
    resume_on_timeout: int,
    fail_on_type_ambiguities: bool,
    lint: bool,
    lint_jsonl: Optional[Path],
    lint_sarif: Optional[Path],
) -> list[str]:
    argv = [str(path) for path in (paths or [])]
    if _param_is_command_line(ctx, "root"):
        argv.extend(["--root", str(root)])
    if _param_is_command_line(ctx, "config") and config is not None:
        argv.extend(["--config", str(config)])
    if _param_is_command_line(ctx, "report") and report is not None:
        argv.extend(["--report", str(report)])
    if _param_is_command_line(ctx, "decision_snapshot") and decision_snapshot is not None:
        argv.extend(["--emit-decision-snapshot", str(decision_snapshot)])
    if _param_is_command_line(ctx, "baseline") and baseline is not None:
        argv.extend(["--baseline", str(baseline)])
    if _param_is_command_line(ctx, "baseline_write") and baseline_write:
        argv.append("--baseline-write")
    if _param_is_command_line(ctx, "exclude"):
        for entry in deadline_loop_iter(exclude or []):
            argv.extend(["--exclude", entry])
    if _param_is_command_line(ctx, "ignore_params_csv") and ignore_params_csv is not None:
        argv.extend(["--ignore-params", ignore_params_csv])
    if (
        _param_is_command_line(ctx, "transparent_decorators_csv")
        and transparent_decorators_csv is not None
    ):
        argv.extend(["--transparent-decorators", transparent_decorators_csv])
    if _param_is_command_line(ctx, "allow_external") and allow_external is not None:
        argv.append("--allow-external" if allow_external else "--no-allow-external")
    if _param_is_command_line(ctx, "strictness") and strictness is not None:
        argv.extend(["--strictness", strictness])
    if _param_is_command_line(ctx, "resume_checkpoint") and resume_checkpoint is not None:
        argv.extend(["--resume-checkpoint", str(resume_checkpoint)])
    if _param_is_command_line(ctx, "emit_timeout_progress_report") and emit_timeout_progress_report:
        argv.append("--emit-timeout-progress-report")
    if _param_is_command_line(ctx, "resume_on_timeout"):
        argv.extend(["--resume-on-timeout", str(int(resume_on_timeout))])
    if _param_is_command_line(ctx, "fail_on_violations") and fail_on_violations:
        argv.append("--fail-on-violations")
    if _param_is_command_line(ctx, "fail_on_type_ambiguities") and fail_on_type_ambiguities:
        argv.append("--fail-on-type-ambiguities")
    if _param_is_command_line(ctx, "lint") and lint:
        argv.append("--lint")
    if _param_is_command_line(ctx, "lint_jsonl") and lint_jsonl is not None:
        argv.extend(["--lint-jsonl", str(lint_jsonl)])
    if _param_is_command_line(ctx, "lint_sarif") and lint_sarif is not None:
        argv.extend(["--lint-sarif", str(lint_sarif)])
    return argv


def _run_check_raw_profile(
    *,
    ctx: typer.Context,
    paths: Optional[List[Path]],
    report: Optional[Path],
    fail_on_violations: bool,
    root: Path,
    config: Optional[Path],
    decision_snapshot: Optional[Path],
    baseline: Optional[Path],
    baseline_write: bool,
    exclude: Optional[List[str]],
    ignore_params_csv: Optional[str],
    transparent_decorators_csv: Optional[str],
    allow_external: Optional[bool],
    strictness: Optional[str],
    resume_checkpoint: Optional[Path],
    emit_timeout_progress_report: bool,
    resume_on_timeout: int,
    fail_on_type_ambiguities: bool,
    lint: bool,
    lint_jsonl: Optional[Path],
    lint_sarif: Optional[Path],
    run_dataflow_raw_argv_fn: Callable[[list[str]], None] | None = None,
) -> None:
    unsupported = _raw_profile_unsupported_flags(ctx)
    if unsupported:
        rendered = ", ".join(unsupported)
        raise typer.BadParameter(
            f"--profile raw does not support check-only options: {rendered}"
        )
    raw_args = _check_raw_profile_args(
        ctx=ctx,
        paths=paths,
        report=report,
        fail_on_violations=fail_on_violations,
        root=root,
        config=config,
        decision_snapshot=decision_snapshot,
        baseline=baseline,
        baseline_write=baseline_write,
        exclude=exclude,
        ignore_params_csv=ignore_params_csv,
        transparent_decorators_csv=transparent_decorators_csv,
        allow_external=allow_external,
        strictness=strictness,
        resume_checkpoint=resume_checkpoint,
        emit_timeout_progress_report=emit_timeout_progress_report,
        resume_on_timeout=resume_on_timeout,
        fail_on_type_ambiguities=fail_on_type_ambiguities,
        lint=lint,
        lint_jsonl=lint_jsonl,
        lint_sarif=lint_sarif,
    )
    resolved_run = run_dataflow_raw_argv_fn or _run_dataflow_raw_argv
    resolved_run(raw_args + list(ctx.args))


def _warn_dataflow_audit_alias() -> None:
    typer.echo(_DATAFLOW_AUDIT_ALIAS_MESSAGE, err=True)


def _dataflow_alias_migration_epilog() -> str:
    return _DATAFLOW_AUDIT_MIGRATION_EPILOG


def _context_run_dataflow_raw_argv(ctx: typer.Context) -> Callable[[list[str]], None]:
    obj = ctx.obj
    if isinstance(obj, Mapping):
        candidate = obj.get("run_dataflow_raw_argv")
        if callable(candidate):
            return candidate
    return _run_dataflow_raw_argv


def _context_warn_dataflow_audit_alias(ctx: typer.Context) -> Callable[[], None]:
    obj = ctx.obj
    if isinstance(obj, Mapping):
        candidate = obj.get("warn_dataflow_audit_alias")
        if callable(candidate):
            return candidate
    return _warn_dataflow_audit_alias


def _run_dataflow_raw_argv(
    argv: list[str],
    *,
    runner: Runner | None = None,
) -> None:
    opts = parse_dataflow_args_or_exit(argv)
    payload = build_dataflow_payload(opts)
    resolved_runner = runner or run_command
    result = _run_with_timeout_retries(
        run_once=lambda: dispatch_command(
            command=DATAFLOW_COMMAND,
            payload=payload,
            root=Path(opts.root),
            runner=resolved_runner,
        ),
        root=Path(opts.root),
        emit_timeout_progress_report=opts.emit_timeout_progress_report,
        resume_on_timeout=max(int(opts.resume_on_timeout), 0),
    )
    _emit_dataflow_result_outputs(result, opts)
    raise typer.Exit(code=int(result.get("exit_code", 0)))


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def check(
    ctx: typer.Context,
    paths: List[Path] = typer.Argument(None),
    profile: str = typer.Option("strict", "--profile"),
    report: Optional[Path] = typer.Option(None, "--report"),
    fail_on_violations: bool = typer.Option(True, "--fail-on-violations/--no-fail-on-violations"),
    root: Path = typer.Option(Path("."), "--root"),
    config: Optional[Path] = typer.Option(None, "--config"),
    decision_snapshot: Optional[Path] = typer.Option(
        None, "--decision-snapshot", help="Write decision surface snapshot JSON."
    ),
    emit_test_obsolescence: bool = typer.Option(
        False,
        "--emit-test-obsolescence/--no-emit-test-obsolescence",
        help=(
            "Write test obsolescence report (JSON in artifacts/out, markdown in out/)."
        ),
    ),
    emit_test_obsolescence_state: bool = typer.Option(
        False,
        "--emit-test-obsolescence-state/--no-emit-test-obsolescence-state",
        help="Write test obsolescence state to artifacts/out.",
    ),
    test_obsolescence_state: Optional[Path] = typer.Option(
        None,
        "--test-obsolescence-state",
        help="Use precomputed test obsolescence state for delta/report.",
    ),
    emit_test_obsolescence_delta: bool = typer.Option(
        False,
        "--emit-test-obsolescence-delta/--no-emit-test-obsolescence-delta",
        help=(
            "Write test obsolescence delta report (JSON in artifacts/out, markdown in out/)."
        ),
    ),
    emit_test_evidence_suggestions: bool = typer.Option(
        False,
        "--emit-test-evidence-suggestions/--no-emit-test-evidence-suggestions",
        help=(
            "Write test evidence suggestions (JSON in artifacts/out, markdown in out/)."
        ),
    ),
    emit_call_clusters: bool = typer.Option(
        False,
        "--emit-call-clusters/--no-emit-call-clusters",
        help="Write call cluster report (JSON in artifacts/out, markdown in out/).",
    ),
    emit_call_cluster_consolidation: bool = typer.Option(
        False,
        "--emit-call-cluster-consolidation/--no-emit-call-cluster-consolidation",
        help=(
            "Write call cluster consolidation plan (JSON in artifacts/out, markdown in out/)."
        ),
    ),
    emit_test_annotation_drift: bool = typer.Option(
        False,
        "--emit-test-annotation-drift/--no-emit-test-annotation-drift",
        help=(
            "Write test annotation drift report (JSON in artifacts/out, markdown in out/)."
        ),
    ),
    test_annotation_drift_state: Optional[Path] = typer.Option(
        None,
        "--test-annotation-drift-state",
        help="Use precomputed annotation drift state for delta.",
    ),
    emit_test_annotation_drift_delta: bool = typer.Option(
        False,
        "--emit-test-annotation-drift-delta/--no-emit-test-annotation-drift-delta",
        help=(
            "Write test annotation drift delta report (JSON in artifacts/out, markdown in out/)."
        ),
    ),
    write_test_annotation_drift_baseline: bool = typer.Option(
        False,
        "--write-test-annotation-drift-baseline/--no-write-test-annotation-drift-baseline",
        help="Write the current test annotation drift baseline to baselines/.",
    ),
    write_test_obsolescence_baseline: bool = typer.Option(
        False,
        "--write-test-obsolescence-baseline/--no-write-test-obsolescence-baseline",
        help="Write the current test obsolescence baseline to baselines/.",
    ),
    emit_ambiguity_delta: bool = typer.Option(
        False,
        "--emit-ambiguity-delta/--no-emit-ambiguity-delta",
        help="Write ambiguity delta report (JSON in artifacts/out, markdown in out/).",
    ),
    emit_ambiguity_state: bool = typer.Option(
        False,
        "--emit-ambiguity-state/--no-emit-ambiguity-state",
        help="Write ambiguity state to artifacts/out.",
    ),
    ambiguity_state: Optional[Path] = typer.Option(
        None,
        "--ambiguity-state",
        help="Use precomputed ambiguity state for delta.",
    ),
    write_ambiguity_baseline: bool = typer.Option(
        False,
        "--write-ambiguity-baseline/--no-write-ambiguity-baseline",
        help="Write the current ambiguity baseline to baselines/.",
    ),
    baseline: Optional[Path] = typer.Option(
        None, "--baseline", help="Baseline file of allowed violations."
    ),
    baseline_write: bool = typer.Option(
        False, "--baseline-write", help="Write current violations to baseline."
    ),
    exclude: Optional[List[str]] = typer.Option(None, "--exclude"),
    ignore_params_csv: Optional[str] = typer.Option(None, "--ignore-params"),
    transparent_decorators_csv: Optional[str] = typer.Option(
        None, "--transparent-decorators"
    ),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    strictness: Optional[str] = typer.Option(None, "--strictness"),
    resume_checkpoint: Optional[Path] = typer.Option(
        None,
        "--resume-checkpoint",
        help="Checkpoint path for resumable timeout runs.",
    ),
    emit_timeout_progress_report: bool = typer.Option(
        False,
        "--emit-timeout-progress-report/--no-emit-timeout-progress-report",
        help="Write timeout progress report artifacts on timeout.",
    ),
    resume_on_timeout: int = typer.Option(
        0,
        "--resume-on-timeout",
        min=0,
        help="Retry count when timeout reports timed_out_progress_resume.",
    ),
    analysis_tick_limit: Optional[int] = typer.Option(
        None,
        "--analysis-tick-limit",
        min=1,
        help="Deterministic logical timeout budget (ticks).",
    ),
    fail_on_type_ambiguities: bool = typer.Option(
        True, "--fail-on-type-ambiguities/--no-fail-on-type-ambiguities"
    ),
    lint: bool = typer.Option(False, "--lint/--no-lint"),
    lint_jsonl: Optional[Path] = typer.Option(
        None, "--lint-jsonl", help="Write lint JSONL to file or '-' for stdout."
    ),
    lint_sarif: Optional[Path] = typer.Option(
        None, "--lint-sarif", help="Write lint SARIF to file or '-' for stdout."
    ),
) -> None:
    # dataflow-bundle: ignore_params_csv, transparent_decorators_csv
    """Run the dataflow grammar audit with strict defaults."""
    profile_name = profile.strip().lower()
    if profile_name not in {"strict", "raw"}:
        raise typer.BadParameter("profile must be 'strict' or 'raw'")
    if profile_name == "raw":
        run_dataflow_raw_argv_fn = _context_run_dataflow_raw_argv(ctx)
        _run_check_raw_profile(
            ctx=ctx,
            paths=paths,
            report=report,
            fail_on_violations=fail_on_violations,
            root=root,
            config=config,
            decision_snapshot=decision_snapshot,
            baseline=baseline,
            baseline_write=baseline_write,
            exclude=exclude,
            ignore_params_csv=ignore_params_csv,
            transparent_decorators_csv=transparent_decorators_csv,
            allow_external=allow_external,
            strictness=strictness,
            resume_checkpoint=resume_checkpoint,
            emit_timeout_progress_report=emit_timeout_progress_report,
            resume_on_timeout=resume_on_timeout,
            fail_on_type_ambiguities=fail_on_type_ambiguities,
            lint=lint,
            lint_jsonl=lint_jsonl,
            lint_sarif=lint_sarif,
            run_dataflow_raw_argv_fn=run_dataflow_raw_argv_fn,
        )
        return
    extra_tokens = list(ctx.args)
    extra_tokens.extend(
        str(path) for path in (paths or []) if str(path).startswith("-")
    )
    if extra_tokens:
        joined = " ".join(extra_tokens)
        raise typer.BadParameter(
            f"Unknown arguments for strict profile: {joined}. Use --profile raw for raw options."
        )
    lint_enabled = lint or bool(lint_jsonl or lint_sarif)
    result = _run_with_timeout_retries(
        run_once=lambda: run_check(
            paths=paths,
            report=report,
            fail_on_violations=fail_on_violations,
            root=root,
            config=config,
            baseline=baseline,
            baseline_write=baseline_write,
            decision_snapshot=decision_snapshot,
            artifact_flags=CheckArtifactFlags(
                emit_test_obsolescence=emit_test_obsolescence,
                emit_test_evidence_suggestions=emit_test_evidence_suggestions,
                emit_call_clusters=emit_call_clusters,
                emit_call_cluster_consolidation=emit_call_cluster_consolidation,
                emit_test_annotation_drift=emit_test_annotation_drift,
            ),
            emit_test_obsolescence_state=emit_test_obsolescence_state,
            test_obsolescence_state=test_obsolescence_state,
            emit_test_obsolescence_delta=emit_test_obsolescence_delta,
            test_annotation_drift_state=test_annotation_drift_state,
            emit_test_annotation_drift_delta=emit_test_annotation_drift_delta,
            write_test_annotation_drift_baseline=write_test_annotation_drift_baseline,
            write_test_obsolescence_baseline=write_test_obsolescence_baseline,
            emit_ambiguity_delta=emit_ambiguity_delta,
            emit_ambiguity_state=emit_ambiguity_state,
            ambiguity_state=ambiguity_state,
            write_ambiguity_baseline=write_ambiguity_baseline,
            exclude=exclude,
            ignore_params_csv=ignore_params_csv,
            transparent_decorators_csv=transparent_decorators_csv,
            allow_external=allow_external,
            strictness=strictness,
            fail_on_type_ambiguities=fail_on_type_ambiguities,
            lint=lint_enabled,
            resume_checkpoint=resume_checkpoint,
            emit_timeout_progress_report=emit_timeout_progress_report,
            resume_on_timeout=resume_on_timeout,
            analysis_tick_limit=analysis_tick_limit,
        ),
        root=Path(root),
        emit_timeout_progress_report=emit_timeout_progress_report,
        resume_on_timeout=resume_on_timeout,
    )
    with _cli_deadline_scope():
        lint_lines = result.get("lint_lines", []) or []
        _emit_lint_outputs(
            lint_lines,
            lint=lint,
            lint_jsonl=lint_jsonl,
            lint_sarif=lint_sarif,
        )
    raise typer.Exit(code=int(result.get("exit_code", 0)))


def _dataflow_audit(
    request: "DataflowAuditRequest",
) -> None:
    """Run the dataflow grammar audit with explicit options."""
    with _cli_deadline_scope():
        check_deadline()
    argv = list(request.args or []) + list(request.ctx.args)
    _run_dataflow_raw_argv(argv, runner=request.runner)


@app.command(
    "dataflow-audit",
    add_help_option=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def dataflow_audit(
    ctx: typer.Context,
    args: List[str] = typer.Argument(None),
) -> None:
    argv = list(args or []) + list(ctx.args)
    if any(arg in {"-h", "--help"} for arg in argv):
        parse_dataflow_args_or_exit(["--help"])
    _context_warn_dataflow_audit_alias(ctx)()
    _context_run_dataflow_raw_argv(ctx)(argv)


def dataflow_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run dataflow grammar audit in raw profile mode.",
        epilog=_dataflow_alias_migration_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default=None)
    parser.add_argument("--baseline", default=None, help="Baseline file for violations.")
    parser.add_argument(
        "--baseline-write",
        action="store_true",
        help="Write current violations to baseline file.",
    )
    parser.add_argument("--exclude", action="append", default=None)
    parser.add_argument("--ignore-params", default=None)
    parser.add_argument(
        "--transparent-decorators",
        default=None,
        help="Comma-separated decorator names treated as transparent.",
    )
    parser.add_argument(
        "--allow-external",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--strictness", choices=["high", "low"], default=None)
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Checkpoint path for resumable timeout runs.",
    )
    parser.add_argument(
        "--emit-timeout-progress-report",
        action="store_true",
        help="Write timeout progress report artifacts on timeout.",
    )
    parser.add_argument(
        "--resume-on-timeout",
        type=int,
        default=0,
        help="Retry count when timeout returns timed_out_progress_resume.",
    )
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--dot", default=None, help="Write DOT graph to file or '-' for stdout.")
    parser.add_argument(
        "--emit-structure-tree",
        default=None,
        help="Write canonical structure snapshot JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--emit-structure-metrics",
        default=None,
        help="Write structure metrics JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-synth-json",
        default=None,
        help="Write fingerprint synth registry JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-provenance-json",
        default=None,
        help="Write fingerprint provenance JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-deadness-json",
        default=None,
        help="Write fingerprint deadness JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-coherence-json",
        default=None,
        help="Write fingerprint coherence JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-rewrite-plans-json",
        default=None,
        help="Write fingerprint rewrite plans JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-exception-obligations-json",
        default=None,
        help="Write fingerprint exception obligations JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-handledness-json",
        default=None,
        help="Write fingerprint handledness JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--emit-decision-snapshot",
        default=None,
        help="Write decision surface snapshot JSON to file or '-' for stdout.",
    )
    parser.add_argument("--report", default=None, help="Write Markdown report (mermaid) to file.")
    parser.add_argument(
        "--lint",
        action="store_true",
        help="Emit lint-style lines (path:line:col: CODE message).",
    )
    parser.add_argument(
        "--lint-jsonl",
        default=None,
        help="Write lint JSONL to file or '-' for stdout.",
    )
    parser.add_argument(
        "--lint-sarif",
        default=None,
        help="Write lint SARIF to file or '-' for stdout.",
    )
    parser.add_argument("--max-components", type=int, default=10, help="Max components in report.")
    parser.add_argument(
        "--type-audit",
        action="store_true",
        help="Emit type-tightening suggestions based on downstream annotations.",
    )
    parser.add_argument(
        "--type-audit-max",
        type=int,
        default=50,
        help="Max type-tightening entries to print.",
    )
    parser.add_argument(
        "--type-audit-report",
        action="store_true",
        help="Include type-flow audit summary in the markdown report.",
    )
    parser.add_argument(
        "--fail-on-type-ambiguities",
        action="store_true",
        help="Exit non-zero if type ambiguities are detected.",
    )
    parser.add_argument(
        "--fail-on-violations",
        action="store_true",
        help="Exit non-zero if undocumented/undeclared bundle violations are detected.",
    )
    parser.add_argument(
        "--synthesis-plan",
        default=None,
        help="Write synthesis plan JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--synthesis-report",
        action="store_true",
        help="Include synthesis plan summary in the markdown report.",
    )
    parser.add_argument(
        "--synthesis-protocols",
        default=None,
        help="Write protocol/dataclass stubs to file or '-' for stdout.",
    )
    parser.add_argument(
        "--synthesis-protocols-kind",
        choices=["dataclass", "protocol"],
        default="dataclass",
        help="Emit dataclass or typing.Protocol stubs (default: dataclass).",
    )
    parser.add_argument(
        "--synthesis-max-tier",
        type=int,
        default=2,
        help="Max tier to include in synthesis plan.",
    )
    parser.add_argument(
        "--synthesis-min-bundle-size",
        type=int,
        default=2,
        help="Min bundle size to include in synthesis plan.",
    )
    parser.add_argument(
        "--synthesis-allow-singletons",
        action="store_true",
        help="Allow single-field bundles in synthesis plan.",
    )
    parser.add_argument(
        "--synthesis-merge-overlap",
        type=float,
        default=None,
        help="Jaccard overlap threshold for merging bundles (0.0-1.0).",
    )
    parser.add_argument(
        "--refactor-plan",
        action="store_true",
        help="Include refactoring plan summary in the markdown report.",
    )
    parser.add_argument(
        "--refactor-plan-json",
        default=None,
        help="Write refactoring plan JSON to file or '-' for stdout.",
    )
    return parser


def _run_docflow_audit(
    *,
    root: Path,
    fail_on_violations: bool,
    audit_tools_path: Path | None = None,
    spec_from_file_location_fn: Callable[..., object | None] = importlib.util.spec_from_file_location,
    module_from_spec_fn: Callable[..., object] = importlib.util.module_from_spec,
) -> int:
    repo_root = _find_repo_root()
    module_path = audit_tools_path or (repo_root / "scripts" / "audit_tools.py")
    if not module_path.exists():
        typer.secho(
            "audit_tools.py not found; repository layout required",
            err=True,
            fg=typer.colors.RED,
        )
        return 2

    @contextmanager
    def _load_audit_tools() -> Generator[object, None, None]:
        module_name = "gabion_repo_audit_tools"
        spec = spec_from_file_location_fn(module_name, module_path)
        loader = getattr(spec, "loader", None)
        if spec is None or loader is None:
            raise RuntimeError("failed to load audit_tools module")
        module = module_from_spec_fn(spec)
        scripts_root = str(module_path.parent)
        inserted_path = False
        if scripts_root not in sys.path:
            sys.path.insert(0, scripts_root)
            inserted_path = True
        previous_module = sys.modules.get(module_name)
        had_previous = module_name in sys.modules
        sys.modules[module_name] = module
        try:
            loader.exec_module(module)
            yield module
        finally:
            if had_previous:
                sys.modules[module_name] = previous_module  # type: ignore[assignment]
            else:
                sys.modules.pop(module_name, None)
            if inserted_path:
                try:
                    sys.path.remove(scripts_root)
                except ValueError:
                    pass

    try:
        with _load_audit_tools() as module:
            args = ["--root", str(root)]
            if fail_on_violations:
                args.append("--fail-on-violations")
            status = int(module.run_docflow_cli(args))
            if status == 0:
                try:
                    status = int(module.run_sppf_graph_cli([]))
                except Exception as exc:
                    typer.secho(
                        f"docflow: sppf-graph failed: {exc}",
                        err=True,
                        fg=typer.colors.RED,
                    )
                    return 1
            return status
    except Exception as exc:
        typer.secho(
            f"failed to load audit_tools module: {exc}",
            err=True,
            fg=typer.colors.RED,
        )
        return 2


@app.command("docflow-audit")
def docflow_audit(
    root: Path = typer.Option(Path("."), "--root"),
    fail_on_violations: bool = typer.Option(
        False, "--fail-on-violations/--no-fail-on-violations"
    ),
) -> None:
    """Run the docflow audit (governance docs only)."""
    exit_code = _run_docflow_audit(root=root, fail_on_violations=fail_on_violations)
    raise typer.Exit(code=exit_code)


def _run_synth(
    *,
    paths: List[Path] | None,
    root: Path,
    out_dir: Path,
    no_timestamp: bool,
    config: Optional[Path],
    exclude: Optional[List[str]],
    ignore_params_csv: Optional[str],
    transparent_decorators_csv: Optional[str],
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
    runner: Runner = run_command,
) -> tuple[JSONObject, dict[str, Path], Path | None]:
    check_deadline()
    if not paths:
        paths = [Path(".")]
    exclude_dirs: list[str] | None = None
    if exclude is not None:
        exclude_dirs = []
        for entry in exclude:
            check_deadline()
            exclude_dirs.extend([part.strip() for part in entry.split(",") if part.strip()])
    ignore_list: list[str] | None = None
    if ignore_params_csv is not None:
        ignore_list = [p.strip() for p in ignore_params_csv.split(",") if p.strip()]
    transparent_list: list[str] | None = None
    if transparent_decorators_csv is not None:
        transparent_list = [
            p.strip() for p in transparent_decorators_csv.split(",") if p.strip()
        ]
    if strictness is not None and strictness not in {"high", "low"}:
        raise typer.BadParameter("strictness must be 'high' or 'low'")
    if synthesis_protocols_kind not in {"dataclass", "protocol"}:
        raise typer.BadParameter(
            "synthesis-protocols-kind must be 'dataclass' or 'protocol'"
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
    }
    result = dispatch_command(
        command=DATAFLOW_COMMAND,
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


@app.command("synth")
def synth(
    paths: List[Path] = typer.Argument(None),
    root: Path = typer.Option(Path("."), "--root"),
    out_dir: Path = typer.Option(Path("artifacts/synthesis"), "--out-dir"),
    no_timestamp: bool = typer.Option(False, "--no-timestamp"),
    config: Optional[Path] = typer.Option(None, "--config"),
    exclude: Optional[List[str]] = typer.Option(None, "--exclude"),
    ignore_params_csv: Optional[str] = typer.Option(None, "--ignore-params"),
    transparent_decorators_csv: Optional[str] = typer.Option(
        None, "--transparent-decorators"
    ),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    strictness: Optional[str] = typer.Option(None, "--strictness"),
    no_recursive: bool = typer.Option(False, "--no-recursive"),
    max_components: int = typer.Option(10, "--max-components"),
    type_audit_report: bool = typer.Option(
        True, "--type-audit-report/--no-type-audit-report"
    ),
    type_audit_max: int = typer.Option(50, "--type-audit-max"),
    synthesis_max_tier: int = typer.Option(2, "--synthesis-max-tier"),
    synthesis_min_bundle_size: int = typer.Option(2, "--synthesis-min-bundle-size"),
    synthesis_allow_singletons: bool = typer.Option(
        False, "--synthesis-allow-singletons"
    ),
    synthesis_protocols_kind: str = typer.Option(
        "dataclass", "--synthesis-protocols-kind"
    ),
    refactor_plan: bool = typer.Option(True, "--refactor-plan/--no-refactor-plan"),
    fail_on_violations: bool = typer.Option(
        False, "--fail-on-violations/--no-fail-on-violations"
    ),
) -> None:
    """Run the dataflow audit and emit synthesis outputs (prototype)."""
    with _cli_deadline_scope():
        result, paths_out, timestamp = _run_synth(
            paths=paths,
            root=root,
            out_dir=out_dir,
            no_timestamp=no_timestamp,
            config=config,
            exclude=exclude,
            ignore_params_csv=ignore_params_csv,
            transparent_decorators_csv=transparent_decorators_csv,
            allow_external=allow_external,
            strictness=strictness,
            no_recursive=no_recursive,
            max_components=max_components,
            type_audit_report=type_audit_report,
            type_audit_max=type_audit_max,
            synthesis_max_tier=synthesis_max_tier,
            synthesis_min_bundle_size=synthesis_min_bundle_size,
            synthesis_allow_singletons=synthesis_allow_singletons,
            synthesis_protocols_kind=synthesis_protocols_kind,
            refactor_plan=refactor_plan,
            fail_on_violations=fail_on_violations,
        )
        _emit_synth_outputs(
            paths_out=paths_out,
            timestamp=timestamp,
            refactor_plan=refactor_plan,
        )
        raise typer.Exit(code=int(result.get("exit_code", 0)))


@app.command("synthesis-plan")
def synthesis_plan(
    input_path: Optional[Path] = typer.Option(
        None, "--input", help="JSON payload describing bundles and synthesis settings."
    ),
    output_path: Optional[Path] = typer.Option(
        None, "--output", help="Write synthesis plan JSON to this path."
    ),
) -> None:
    """Generate a synthesis plan from a JSON payload (prototype)."""
    with _cli_deadline_scope():
        _run_synthesis_plan(input_path=input_path, output_path=output_path)


def _run_synthesis_plan(
    *,
    input_path: Optional[Path],
    output_path: Optional[Path],
    runner: Runner = run_command,
) -> None:
    """Generate a synthesis plan from a JSON payload (prototype)."""
    payload: JSONObject = {}
    if input_path is not None:
        try:
            loaded = json.loads(input_path.read_text())
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"Invalid JSON payload: {exc}") from exc
        if not isinstance(loaded, dict):
            raise typer.BadParameter("Synthesis payload must be a JSON object.")
        payload = loaded
    result = dispatch_command(
        command=SYNTHESIS_COMMAND,
        payload=payload,
        root=Path("."),
        runner=runner,
    )
    output = json.dumps(result, indent=2, sort_keys=True)
    if output_path is None:
        typer.echo(output)
    else:
        output_path.write_text(output)


def _emit_synth_outputs(
    *,
    paths_out: dict[str, Path],
    timestamp: Path | None,
    refactor_plan: bool,
) -> None:
    if timestamp:
        typer.echo(f"Snapshot: {paths_out['output_root']}")
    typer.echo(f"- {paths_out['report']}")
    typer.echo(f"- {paths_out['dot']}")
    typer.echo(f"- {paths_out['plan']}")
    typer.echo(f"- {paths_out['protocol']}")
    if paths_out["fingerprint_synth"].exists():
        typer.echo(f"- {paths_out['fingerprint_synth']}")
    if paths_out["fingerprint_provenance"].exists():
        typer.echo(f"- {paths_out['fingerprint_provenance']}")
    if paths_out["fingerprint_coherence"].exists():
        typer.echo(f"- {paths_out['fingerprint_coherence']}")
    if paths_out["fingerprint_rewrite_plans"].exists():
        typer.echo(f"- {paths_out['fingerprint_rewrite_plans']}")
    if paths_out["fingerprint_exception_obligations"].exists():
        typer.echo(f"- {paths_out['fingerprint_exception_obligations']}")
    if paths_out["fingerprint_handledness"].exists():
        typer.echo(f"- {paths_out['fingerprint_handledness']}")
    if refactor_plan:
        typer.echo(f"- {paths_out['refactor']}")


def run_structure_diff(
    *,
    request: SnapshotDiffRequest,
    root: Path | None = None,
    runner: Runner | None = None,
) -> JSONObject:
    payload = request.to_payload()
    runner = runner or DEFAULT_RUNNER
    root_path = root or Path(".")
    return dispatch_command(
        command=STRUCTURE_DIFF_COMMAND,
        payload=payload,
        root=root_path,
        runner=runner,
    )


def run_decision_diff(
    *,
    request: SnapshotDiffRequest,
    root: Path | None = None,
    runner: Runner | None = None,
) -> JSONObject:
    payload = request.to_payload()
    runner = runner or DEFAULT_RUNNER
    root_path = root or Path(".")
    return dispatch_command(
        command=DECISION_DIFF_COMMAND,
        payload=payload,
        root=root_path,
        runner=runner,
    )


def run_structure_reuse(
    *,
    snapshot: Path,
    min_count: int = 2,
    lemma_stubs: Path | None = None,
    root: Path | None = None,
    runner: Runner | None = None,
) -> JSONObject:
    payload = {"snapshot": str(snapshot), "min_count": int(min_count)}
    if lemma_stubs is not None:
        payload["lemma_stubs"] = str(lemma_stubs)
    runner = runner or DEFAULT_RUNNER
    root_path = root or Path(".")
    return dispatch_command(
        command=STRUCTURE_REUSE_COMMAND,
        payload=payload,
        root=root_path,
        runner=runner,
    )


def _emit_structure_diff(result: JSONObject) -> None:
    check_deadline()
    errors = result.get("errors")
    exit_code = int(result.get("exit_code", 0))
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        for error in errors:
            check_deadline()
            typer.secho(str(error), err=True, fg=typer.colors.RED)
    if exit_code:
        raise typer.Exit(code=exit_code)


def _emit_decision_diff(result: JSONObject) -> None:
    check_deadline()
    errors = result.get("errors")
    exit_code = int(result.get("exit_code", 0))
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        for error in errors:
            check_deadline()
            typer.secho(str(error), err=True, fg=typer.colors.RED)
    if exit_code:
        raise typer.Exit(code=exit_code)


def _emit_structure_reuse(result: JSONObject) -> None:
    check_deadline()
    errors = result.get("errors")
    exit_code = int(result.get("exit_code", 0))
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        for error in errors:
            check_deadline()
            typer.secho(str(error), err=True, fg=typer.colors.RED)
    if exit_code:
        raise typer.Exit(code=exit_code)


@app.command("structure-diff")
def structure_diff(
    baseline: Path = typer.Option(..., "--baseline"),
    current: Path = typer.Option(..., "--current"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Compare two structure snapshots and emit a JSON diff."""
    with _cli_deadline_scope():
        request = SnapshotDiffRequest(baseline=baseline, current=current)
        result = run_structure_diff(request=request, root=root)
        _emit_structure_diff(result)


@app.command("decision-diff")
def decision_diff(
    baseline: Path = typer.Option(..., "--baseline"),
    current: Path = typer.Option(..., "--current"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Compare two decision surface snapshots and emit a JSON diff."""
    with _cli_deadline_scope():
        request = SnapshotDiffRequest(baseline=baseline, current=current)
        result = run_decision_diff(request=request, root=root)
        _emit_decision_diff(result)


@app.command("structure-reuse")
def structure_reuse(
    snapshot: Path = typer.Option(..., "--snapshot"),
    min_count: int = typer.Option(2, "--min-count"),
    lemma_stubs: Optional[Path] = typer.Option(
        None, "--lemma-stubs", help="Write lemma stubs to file or '-' for stdout."
    ),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Detect repeated subtrees in a structure snapshot."""
    with _cli_deadline_scope():
        result = run_structure_reuse(
            snapshot=snapshot,
            min_count=min_count,
            lemma_stubs=lemma_stubs,
            root=root,
        )
        _emit_structure_reuse(result)


@app.command("refactor-protocol")
def refactor_protocol(
    input_path: Optional[Path] = typer.Option(
        None, "--input", help="JSON payload describing the refactor request."
    ),
    output_path: Optional[Path] = typer.Option(
        None, "--output", help="Write refactor response JSON to this path."
    ),
    protocol_name: Optional[str] = typer.Option(None, "--protocol-name"),
    bundle: Optional[List[str]] = typer.Option(None, "--bundle"),
    field: Optional[List[str]] = typer.Option(
        None,
        "--field",
        help="Field spec in 'name:type' form (repeatable).",
    ),
    target_path: Optional[Path] = typer.Option(None, "--target-path"),
    target_functions: Optional[List[str]] = typer.Option(None, "--target-function"),
    compatibility_shim: bool = typer.Option(
        False, "--compat-shim/--no-compat-shim"
    ),
    rationale: Optional[str] = typer.Option(None, "--rationale"),
) -> None:
    """Generate protocol refactor edits from a JSON payload (prototype)."""
    with _cli_deadline_scope():
        _run_refactor_protocol(
            input_path=input_path,
            output_path=output_path,
            protocol_name=protocol_name,
            bundle=bundle,
            field=field,
            target_path=target_path,
            target_functions=target_functions,
            compatibility_shim=compatibility_shim,
            rationale=rationale,
        )


def _run_refactor_protocol(
    *,
    input_path: Optional[Path],
    output_path: Optional[Path],
    protocol_name: Optional[str],
    bundle: Optional[List[str]],
    field: Optional[List[str]],
    target_path: Optional[Path],
    target_functions: Optional[List[str]],
    compatibility_shim: bool,
    rationale: Optional[str],
    runner: Runner = run_command,
) -> None:
    """Generate protocol refactor edits from a JSON payload (prototype)."""
    input_payload: JSONObject | None = None
    if input_path is not None:
        try:
            loaded = json.loads(input_path.read_text())
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"Invalid JSON payload: {exc}") from exc
        if not isinstance(loaded, dict):
            raise typer.BadParameter("Refactor payload must be a JSON object.")
        input_payload = loaded
    payload = build_refactor_payload(
        input_payload=input_payload,
        protocol_name=protocol_name,
        bundle=bundle,
        field=field,
        target_path=target_path,
        target_functions=target_functions,
        compatibility_shim=compatibility_shim,
        rationale=rationale,
    )
    result = dispatch_command(
        command=REFACTOR_COMMAND,
        payload=payload,
        root=None,
        runner=runner,
    )
    output = json.dumps(result, indent=2, sort_keys=True)
    if output_path is None:
        typer.echo(output)
    else:
        output_path.write_text(output)
