from __future__ import annotations

from pathlib import Path

from pygls.lsp.server import LanguageServer
from lsprotocol.types import (
    TEXT_DOCUMENT_DID_OPEN,
    TEXT_DOCUMENT_DID_SAVE,
    Diagnostic,
    DiagnosticSeverity,
    Position,
    Range,
)

from gabion.analysis import (
    AuditConfig,
    analyze_paths,
    compute_violations,
    render_dot,
    render_report,
)
from gabion.config import dataflow_defaults, merge_payload
from gabion.schema import SynthesisRequest, SynthesisResponse
from gabion.synthesis import NamingContext, SynthesisConfig, Synthesizer

server = LanguageServer("gabion", "0.1.0")
DATAFLOW_COMMAND = "gabion.dataflowAudit"
SYNTHESIS_COMMAND = "gabion.synthesisPlan"


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
    for bundles in result.groups_by_path.values():
        for _, group_list in bundles.items():
            for bundle in group_list:
                message = f"Implicit bundle detected: {', '.join(sorted(bundle))}"
                diagnostics.append(
                    Diagnostic(
                        range=Range(
                            start=Position(line=0, character=0),
                            end=Position(line=0, character=1),
                        ),
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
    exclude_dirs = set(payload.get("exclude", []))
    ignore_params = set(payload.get("ignore_params", []))
    allow_external = payload.get("allow_external", False)
    strictness = payload.get("strictness", "high")

    config = AuditConfig(
        project_root=Path(root),
        exclude_dirs=exclude_dirs,
        ignore_params=ignore_params,
        external_filter=not allow_external,
        strictness=strictness,
    )
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

    if dot_path:
        dot = render_dot(analysis.groups_by_path)
        if dot_path == "-":
            response["dot"] = dot
        else:
            Path(dot_path).write_text(dot)

    violations: list[str] = []
    if report_path:
        report, violations = render_report(
            analysis.groups_by_path,
            max_components,
            type_suggestions=analysis.type_suggestions if type_audit_report else None,
            type_ambiguities=analysis.type_ambiguities if type_audit_report else None,
            constant_smells=analysis.constant_smells,
            unused_arg_smells=analysis.unused_arg_smells,
        )
        Path(report_path).write_text(report)
    else:
        violations = compute_violations(
            analysis.groups_by_path,
            max_components,
            type_suggestions=analysis.type_suggestions if type_audit_report else None,
            type_ambiguities=analysis.type_ambiguities if type_audit_report else None,
        )

    response["violations"] = len(violations)
    response["exit_code"] = 1 if (fail_on_violations and violations) else 0
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


def start() -> None:
    """Start the language server (stub)."""
    server.start_io()


if __name__ == "__main__":
    start()
