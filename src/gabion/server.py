from __future__ import annotations

import json
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlparse

from pygls.lsp.server import LanguageServer
from lsprotocol.types import (
    TEXT_DOCUMENT_DID_OPEN,
    TEXT_DOCUMENT_DID_SAVE,
    TEXT_DOCUMENT_CODE_ACTION,
    CodeAction,
    CodeActionKind,
    CodeActionParams,
    Command,
    Diagnostic,
    DiagnosticSeverity,
    Position,
    Range,
    WorkspaceEdit,
)

from gabion.analysis import (
    AuditConfig,
    analyze_paths,
    apply_baseline,
    compute_structure_metrics,
    compute_violations,
    build_refactor_plan,
    build_synthesis_plan,
    diff_structure_snapshots,
    load_structure_snapshot,
    load_baseline,
    render_dot,
    render_structure_snapshot,
    render_protocol_stubs,
    render_refactor_plan,
    render_report,
    render_synthesis_section,
    resolve_baseline_path,
    write_baseline,
)
from gabion.config import dataflow_defaults, merge_payload
from gabion.refactor import (
    FieldSpec,
    RefactorEngine,
    RefactorRequest as RefactorRequestModel,
)
from gabion.schema import (
    RefactorRequest,
    RefactorResponse,
    SynthesisResponse,
    SynthesisRequest,
    TextEditDTO,
)
from gabion.synthesis import NamingContext, SynthesisConfig, Synthesizer

server = LanguageServer("gabion", "0.1.0")
DATAFLOW_COMMAND = "gabion.dataflowAudit"
SYNTHESIS_COMMAND = "gabion.synthesisPlan"
REFACTOR_COMMAND = "gabion.refactorProtocol"
STRUCTURE_DIFF_COMMAND = "gabion.structureDiff"


def _uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return Path(uri)


def _normalize_transparent_decorators(value: object) -> set[str] | None:
    if value is None:
        return None
    items: list[str] = []
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, str):
                items.extend([part.strip() for part in item.split(",") if part.strip()])
    if not items:
        return None
    return set(items)


def _diagnostics_for_path(path_str: str, project_root: Path | None) -> list[Diagnostic]:
    result = analyze_paths(
        [Path(path_str)],
        recursive=True,
        type_audit=False,
        type_audit_report=False,
        type_audit_max=0,
        include_constant_smells=False,
        include_unused_arg_smells=False,
        config=AuditConfig(project_root=project_root),
    )
    diagnostics: list[Diagnostic] = []
    for path, bundles in result.groups_by_path.items():
        span_map = result.param_spans_by_path.get(path, {})
        for fn_name, group_list in bundles.items():
            param_spans = span_map.get(fn_name, {})
            for bundle in group_list:
                message = f"Implicit bundle detected: {', '.join(sorted(bundle))}"
                for name in sorted(bundle):
                    span = param_spans.get(name)
                    if span is None:  # pragma: no cover - spans are derived from parsed params
                        start = Position(line=0, character=0)  # pragma: no cover
                        end = Position(line=0, character=1)  # pragma: no cover
                    else:
                        start_line, start_col, end_line, end_col = span
                        start = Position(line=start_line, character=start_col)
                        end = Position(line=end_line, character=end_col)
                    diagnostics.append(
                        Diagnostic(
                            range=Range(start=start, end=end),
                            message=message,
                            severity=DiagnosticSeverity.Information,
                            source="gabion",
                        )
                    )
    return diagnostics


@server.command(DATAFLOW_COMMAND)
def execute_command(ls: LanguageServer, payload: dict | None = None) -> dict:
    if payload is None:
        payload = {}
    root = payload.get("root") or ls.workspace.root_path or "."
    config_path = payload.get("config")
    defaults = dataflow_defaults(
        Path(root), Path(config_path) if config_path else None
    )
    payload = merge_payload(payload, defaults)

    raw_paths = payload.get("paths") or []
    if raw_paths:
        paths = [Path(p) for p in raw_paths]
    else:
        paths = [Path(root)]
    root = payload.get("root") or root
    report_path = payload.get("report")
    dot_path = payload.get("dot")
    fail_on_violations = payload.get("fail_on_violations", False)
    no_recursive = payload.get("no_recursive", False)
    max_components = payload.get("max_components", 10)
    type_audit = payload.get("type_audit", False)
    type_audit_report = payload.get("type_audit_report", False)
    type_audit_max = payload.get("type_audit_max", 50)
    fail_on_type_ambiguities = payload.get("fail_on_type_ambiguities", False)
    exclude_dirs = set(payload.get("exclude", []))
    ignore_params = set(payload.get("ignore_params", []))
    allow_external = payload.get("allow_external", False)
    strictness = payload.get("strictness", "high")
    transparent_decorators = _normalize_transparent_decorators(
        payload.get("transparent_decorators")
    )
    baseline_path = resolve_baseline_path(payload.get("baseline"), Path(root))
    baseline_write = bool(payload.get("baseline_write", False)) and baseline_path is not None
    synthesis_plan_path = payload.get("synthesis_plan")
    synthesis_report = payload.get("synthesis_report", False)
    structure_tree_path = payload.get("structure_tree")
    structure_metrics_path = payload.get("structure_metrics")
    synthesis_max_tier = payload.get("synthesis_max_tier", 2)
    synthesis_min_bundle_size = payload.get("synthesis_min_bundle_size", 2)
    synthesis_allow_singletons = payload.get("synthesis_allow_singletons", False)
    synthesis_protocols_path = payload.get("synthesis_protocols")
    synthesis_protocols_kind = payload.get("synthesis_protocols_kind", "dataclass")
    refactor_plan = payload.get("refactor_plan", False)
    refactor_plan_json = payload.get("refactor_plan_json")

    config = AuditConfig(
        project_root=Path(root),
        exclude_dirs=exclude_dirs,
        ignore_params=ignore_params,
        external_filter=not allow_external,
        strictness=strictness,
        transparent_decorators=transparent_decorators,
    )
    if fail_on_type_ambiguities:
        type_audit = True
    analysis = analyze_paths(
        paths,
        recursive=not no_recursive,
        type_audit=type_audit or type_audit_report,
        type_audit_report=type_audit_report,
        type_audit_max=type_audit_max,
        include_constant_smells=bool(report_path),
        include_unused_arg_smells=bool(report_path),
        config=config,
    )

    response: dict = {
        "type_suggestions": analysis.type_suggestions,
        "type_ambiguities": analysis.type_ambiguities,
        "unused_arg_smells": analysis.unused_arg_smells,
    }

    synthesis_plan: dict[str, object] | None = None
    if synthesis_plan_path or synthesis_report or synthesis_protocols_path:
        synthesis_plan = build_synthesis_plan(
            analysis.groups_by_path,
            project_root=Path(root),
            max_tier=int(synthesis_max_tier),
            min_bundle_size=int(synthesis_min_bundle_size),
            allow_singletons=bool(synthesis_allow_singletons),
            config=config,
        )
        if synthesis_plan_path:
            payload_json = json.dumps(synthesis_plan, indent=2, sort_keys=True)
            if synthesis_plan_path == "-":
                response["synthesis_plan"] = synthesis_plan
            else:
                Path(synthesis_plan_path).write_text(payload_json)
        if synthesis_protocols_path:
            stubs = render_protocol_stubs(
                synthesis_plan, kind=str(synthesis_protocols_kind)
            )
            if synthesis_protocols_path == "-":
                response["synthesis_protocols"] = stubs
            else:
                Path(synthesis_protocols_path).write_text(stubs)

    refactor_plan_payload: dict[str, object] | None = None
    if refactor_plan or refactor_plan_json:
        refactor_plan_payload = build_refactor_plan(
            analysis.groups_by_path,
            paths,
            config=config,
        )
        if refactor_plan_json:
            payload_json = json.dumps(refactor_plan_payload, indent=2, sort_keys=True)
            if refactor_plan_json == "-":
                response["refactor_plan"] = refactor_plan_payload
            else:
                Path(refactor_plan_json).write_text(payload_json)

    if dot_path:
        dot = render_dot(analysis.groups_by_path)
        if dot_path == "-":
            response["dot"] = dot
        else:
            Path(dot_path).write_text(dot)
    if structure_tree_path:
        snapshot = render_structure_snapshot(
            analysis.groups_by_path,
            project_root=config.project_root,
        )
        payload_json = json.dumps(snapshot, indent=2, sort_keys=True)
        if structure_tree_path == "-":
            response["structure_tree"] = snapshot
        else:
            Path(structure_tree_path).write_text(payload_json)
    if structure_metrics_path:
        metrics = compute_structure_metrics(analysis.groups_by_path)
        payload_json = json.dumps(metrics, indent=2, sort_keys=True)
        if structure_metrics_path == "-":
            response["structure_metrics"] = metrics
        else:
            Path(structure_metrics_path).write_text(payload_json)

    violations: list[str] = []
    effective_violations: list[str] | None = None
    if report_path:
        report, violations = render_report(
            analysis.groups_by_path,
            max_components,
            type_suggestions=analysis.type_suggestions if type_audit_report else None,
            type_ambiguities=analysis.type_ambiguities if type_audit_report else None,
            constant_smells=analysis.constant_smells,
            unused_arg_smells=analysis.unused_arg_smells,
        )
        if baseline_path is not None:
            baseline_entries = load_baseline(baseline_path)
            if baseline_write:
                write_baseline(baseline_path, violations)
                baseline_entries = set(violations)
                effective_violations = []
            else:
                effective_violations, _ = apply_baseline(violations, baseline_entries)
            report = (
                report
                + "\n\nBaseline/Ratchet:\n```\n"
                + f"Baseline: {baseline_path}\n"
                + f"Baseline entries: {len(baseline_entries)}\n"
                + f"New violations: {len(effective_violations)}\n"
                + "```\n"
            )
        if synthesis_plan and (
            synthesis_report or synthesis_plan_path or synthesis_protocols_path
        ):
            report = report + render_synthesis_section(synthesis_plan)
        if refactor_plan_payload and (refactor_plan or refactor_plan_json):
            report = report + render_refactor_plan(refactor_plan_payload)
        Path(report_path).write_text(report)
    else:
        violations = compute_violations(
            analysis.groups_by_path,
            max_components,
            type_suggestions=analysis.type_suggestions if type_audit_report else None,
            type_ambiguities=analysis.type_ambiguities if type_audit_report else None,
        )
        if baseline_path is not None:
            baseline_entries = load_baseline(baseline_path)
            if baseline_write:
                write_baseline(baseline_path, violations)
                effective_violations = []
            else:
                effective_violations, _ = apply_baseline(violations, baseline_entries)

    if effective_violations is None:
        effective_violations = violations
    response["violations"] = len(effective_violations)
    if baseline_path is not None:
        response["baseline_path"] = str(baseline_path)
        response["baseline_written"] = bool(baseline_write)
    if fail_on_type_ambiguities and analysis.type_ambiguities:
        response["exit_code"] = 1
    else:
        if baseline_write:
            response["exit_code"] = 0
        else:
            response["exit_code"] = 1 if (fail_on_violations and effective_violations) else 0
    return response


@server.command(SYNTHESIS_COMMAND)
def execute_synthesis(ls: LanguageServer, payload: dict | None = None) -> dict:
    if payload is None:
        payload = {}
    try:
        request = SynthesisRequest.model_validate(payload)
    except Exception as exc:  # pydantic validation
        return {"protocols": [], "warnings": [], "errors": [str(exc)]}

    bundle_tiers: dict[frozenset[str], int] = {}
    for entry in request.bundles:
        bundle = entry.bundle
        if not bundle:
            continue
        bundle_tiers[frozenset(bundle)] = entry.tier

    field_types = request.field_types or {}
    config = SynthesisConfig(
        max_tier=request.max_tier,
        min_bundle_size=request.min_bundle_size,
        allow_singletons=request.allow_singletons,
        merge_overlap_threshold=request.merge_overlap_threshold,
    )
    naming_context = NamingContext(
        existing_names=set(request.existing_names),
        frequency=request.frequency or {},
        fallback_prefix=request.fallback_prefix,
    )
    plan = Synthesizer(config=config).plan(
        bundle_tiers=bundle_tiers,
        field_types=field_types,
        naming_context=naming_context,
    )
    response = SynthesisResponse(
        protocols=[
            {
                "name": spec.name,
                "fields": [
                    {
                        "name": field.name,
                        "type_hint": field.type_hint,
                        "source_params": sorted(field.source_params),
                    }
                    for field in spec.fields
                ],
                "bundle": sorted(spec.bundle),
                "tier": spec.tier,
                "rationale": spec.rationale,
            }
            for spec in plan.protocols
        ],
        warnings=plan.warnings,
        errors=plan.errors,
    )
    return response.model_dump()


@server.command(REFACTOR_COMMAND)
def execute_refactor(ls: LanguageServer, payload: dict | None = None) -> dict:
    if payload is None:
        payload = {}
    try:
        request = RefactorRequest.model_validate(payload)
    except Exception as exc:  # pydantic validation
        return RefactorResponse(errors=[str(exc)]).model_dump()

    project_root = None
    if ls.workspace.root_path:
        project_root = Path(ls.workspace.root_path)
    engine = RefactorEngine(project_root=project_root)
    plan = engine.plan_protocol_extraction(
        RefactorRequestModel(
            protocol_name=request.protocol_name,
            bundle=request.bundle,
            fields=[
                FieldSpec(name=field.name, type_hint=field.type_hint)
                for field in request.fields or []
            ],
            target_path=request.target_path,
            target_functions=request.target_functions,
            compatibility_shim=request.compatibility_shim,
            rationale=request.rationale,
        )
    )
    edits = [
        TextEditDTO(
            path=edit.path,
            start=edit.start,
            end=edit.end,
            replacement=edit.replacement,
        )
        for edit in plan.edits
    ]
    response = RefactorResponse(
        edits=edits,
        warnings=plan.warnings,
        errors=plan.errors,
    )
    return response.model_dump()


@server.command(STRUCTURE_DIFF_COMMAND)
def execute_structure_diff(ls: LanguageServer, payload: dict | None = None) -> dict:
    if payload is None:
        payload = {}
    baseline_path = payload.get("baseline")
    current_path = payload.get("current")
    if not baseline_path or not current_path:
        return {
            "exit_code": 2,
            "errors": ["baseline and current snapshot paths are required"],
        }
    try:
        baseline = load_structure_snapshot(Path(baseline_path))
        current = load_structure_snapshot(Path(current_path))
    except ValueError as exc:
        return {"exit_code": 2, "errors": [str(exc)]}
    return {"exit_code": 0, "diff": diff_structure_snapshots(baseline, current)}


@server.feature(TEXT_DOCUMENT_CODE_ACTION)
def code_action(ls: LanguageServer, params: CodeActionParams) -> list[CodeAction]:
    path = _uri_to_path(params.text_document.uri)
    payload = {
        "protocol_name": "TODO_Bundle",
        "bundle": [],
        "target_path": str(path),
        "target_functions": [],
        "rationale": "Stub code action; populate bundle details manually.",
    }
    title = "Gabion: Extract Protocol (stub)"
    return [
        CodeAction(
            title=title,
            kind=CodeActionKind.RefactorExtract,
            command=Command(title=title, command=REFACTOR_COMMAND, arguments=[payload]),
            edit=WorkspaceEdit(changes={}),
        )
    ]


@server.feature(TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: LanguageServer, params) -> None:
    uri = params.text_document.uri
    doc = ls.workspace.get_document(uri)
    root = Path(ls.workspace.root_path) if ls.workspace.root_path else None
    diagnostics = _diagnostics_for_path(doc.path, root)
    ls.publish_diagnostics(uri, diagnostics)


@server.feature(TEXT_DOCUMENT_DID_SAVE)
def did_save(ls: LanguageServer, params) -> None:
    uri = params.text_document.uri
    doc = ls.workspace.get_document(uri)
    root = Path(ls.workspace.root_path) if ls.workspace.root_path else None
    diagnostics = _diagnostics_for_path(doc.path, root)
    ls.publish_diagnostics(uri, diagnostics)


def start(start_fn: Callable[[], None] | None = None) -> None:
    """Start the language server (stub)."""
    (start_fn or server.start_io)()


if __name__ == "__main__":  # pragma: no cover
    start()  # pragma: no cover
