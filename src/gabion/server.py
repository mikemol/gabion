from __future__ import annotations

import json
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlparse

from pygls.lsp.server import LanguageServer
from pydantic import ValidationError
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

from gabion.json_types import JSONObject, JSONValue

from gabion.analysis import (
    AnalysisResult,
    AuditConfig,
    ReportCarrier,
    analyze_paths,
    apply_baseline,
    compute_structure_metrics,
    compute_structure_reuse,
    render_reuse_lemma_stubs,
    compute_violations,
    build_refactor_plan,
    build_synthesis_plan,
    diff_structure_snapshots,
    diff_decision_snapshots,
    load_structure_snapshot,
    load_decision_snapshot,
    load_baseline,
    render_dot,
    render_structure_snapshot,
    render_decision_snapshot,
    render_protocol_stubs,
    render_refactor_plan,
    render_report,
    render_synthesis_section,
    resolve_baseline_path,
    write_baseline,
)
from gabion.analysis.aspf import Forest
from gabion.analysis import ambiguity_delta
from gabion.analysis import ambiguity_state
from gabion.analysis import call_cluster_consolidation
from gabion.analysis import call_clusters
from gabion.analysis import test_annotation_drift
from gabion.analysis import test_annotation_drift_delta
from gabion.analysis import test_obsolescence
from gabion.analysis import test_obsolescence_delta
from gabion.analysis import test_obsolescence_state
from gabion.analysis import test_evidence_suggestions
from gabion.analysis.timeout_context import (
    Deadline,
    TimeoutExceeded,
    check_deadline,
    forest_scope,
    reset_forest,
    set_forest,
    deadline_profile_scope,
    reset_deadline_profile,
    reset_deadline,
    set_deadline_profile,
    set_deadline,
)
from gabion.invariants import never
from gabion.config import (
    dataflow_defaults,
    dataflow_deadline_roots,
    decision_defaults,
    decision_ignore_list,
    decision_require_tiers,
    decision_tier_map,
    exception_defaults,
    exception_never_list,
    fingerprint_defaults,
    merge_payload,
)
from gabion.analysis.type_fingerprints import (
    Fingerprint,
    PrimeRegistry,
    TypeConstructorRegistry,
    build_fingerprint_registry,
)
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
STRUCTURE_REUSE_COMMAND = "gabion.structureReuse"
DECISION_DIFF_COMMAND = "gabion.decisionDiff"

_SERVER_DEADLINE_OVERHEAD_MIN_NS = 10_000_000
_SERVER_DEADLINE_OVERHEAD_MAX_NS = 200_000_000
_SERVER_DEADLINE_OVERHEAD_DIVISOR = 20


def _require_payload(payload: object, *, command: str) -> dict[str, object]:
    if payload is None:
        never("missing command payload", command=command)
    if not isinstance(payload, dict):
        never(
            "invalid command payload type",
            command=command,
            payload_type=type(payload).__name__,
        )
    return payload


def _truthy_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def _server_deadline_overhead_ns(total_ns: int) -> int:
    if total_ns <= 0:
        return 0
    overhead = total_ns // _SERVER_DEADLINE_OVERHEAD_DIVISOR
    if overhead < _SERVER_DEADLINE_OVERHEAD_MIN_NS:
        overhead = _SERVER_DEADLINE_OVERHEAD_MIN_NS
    if overhead > _SERVER_DEADLINE_OVERHEAD_MAX_NS:
        overhead = _SERVER_DEADLINE_OVERHEAD_MAX_NS
    if overhead >= total_ns:
        overhead = max(0, total_ns - 1)
    return overhead
def _deadline_from_payload(payload: dict) -> Deadline:
    timeout_ticks = payload.get("analysis_timeout_ticks")
    timeout_tick_ns = payload.get("analysis_timeout_tick_ns")
    timeout_ms = payload.get("analysis_timeout_ms")
    timeout_seconds = payload.get("analysis_timeout_seconds")
    if timeout_ticks not in (None, ""):
        try:
            ticks_value = int(timeout_ticks)
        except (TypeError, ValueError):
            never("invalid analysis timeout ticks", ticks=timeout_ticks)
        if ticks_value <= 0:
            never("invalid analysis timeout ticks", ticks=timeout_ticks)
        if timeout_tick_ns in (None, ""):
            never("missing analysis timeout tick_ns", ticks=ticks_value)
        try:
            tick_ns_value = int(timeout_tick_ns)
        except (TypeError, ValueError):
            never("invalid analysis timeout tick_ns", tick_ns=timeout_tick_ns)
        if tick_ns_value <= 0:
            never("invalid analysis timeout tick_ns", tick_ns=timeout_tick_ns)
        total_ns = ticks_value * tick_ns_value
        overhead_ns = _server_deadline_overhead_ns(total_ns)
        remaining_ns = max(1, total_ns - overhead_ns)
        if remaining_ns < tick_ns_value:
            tick_ns_value = max(1, remaining_ns)
        ticks_value = max(1, remaining_ns // tick_ns_value)
        return Deadline.from_timeout_ticks(ticks_value, tick_ns_value)
    if timeout_ms not in (None, ""):
        try:
            ms_value = int(timeout_ms)
        except (TypeError, ValueError):
            never("invalid analysis timeout ms", ms=timeout_ms)
        if ms_value <= 0:
            never("invalid analysis timeout ms", ms=timeout_ms)
        total_ns = ms_value * 1_000_000
        overhead_ns = _server_deadline_overhead_ns(total_ns)
        remaining_ns = max(1, total_ns - overhead_ns)
        tick_ns_value = min(1_000_000, remaining_ns)
        ticks_value = max(1, remaining_ns // tick_ns_value)
        return Deadline.from_timeout_ticks(ticks_value, tick_ns_value)
    if timeout_seconds not in (None, ""):
        # Deprecated: prefer analysis_timeout_ticks / analysis_timeout_tick_ns.
        try:
            seconds_value = Decimal(str(timeout_seconds))
        except (InvalidOperation, ValueError):
            never("invalid analysis timeout seconds", seconds=timeout_seconds)
        if seconds_value <= 0:
            never("invalid analysis timeout seconds", seconds=timeout_seconds)
        ms_value = int(seconds_value * Decimal(1000))
        if ms_value <= 0:
            never("invalid analysis timeout seconds", seconds=timeout_seconds)
        total_ns = ms_value * 1_000_000
        overhead_ns = _server_deadline_overhead_ns(total_ns)
        remaining_ns = max(1, total_ns - overhead_ns)
        tick_ns_value = min(1_000_000, remaining_ns)
        ticks_value = max(1, remaining_ns // tick_ns_value)
        return Deadline.from_timeout_ticks(ticks_value, tick_ns_value)
    never("missing analysis timeout", payload_keys=sorted(payload.keys()))


@contextmanager
def _deadline_scope_from_payload(payload: object):
    normalized_payload = _require_payload(payload, command="deadline_scope")
    deadline = _deadline_from_payload(normalized_payload)
    profile_enabled = _truthy_flag(normalized_payload.get("deadline_profile"))
    root_value = normalized_payload.get("root")
    profile_root = Path(str(root_value)).resolve() if root_value not in (None, "") else None
    with deadline_profile_scope(
        project_root=profile_root,
        enabled=profile_enabled,
    ):
        with forest_scope(Forest()):
            token = set_deadline(deadline)
            try:
                yield
            finally:
                reset_deadline(token)


def _output_dirs(report_root: Path) -> tuple[Path, Path]:
    out_dir = report_root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = report_root / "artifacts" / "out"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, artifact_dir


def _uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return Path(uri)


def _normalize_transparent_decorators(value: object) -> set[str] | None:
    check_deadline()
    if value is None:
        return None
    items: list[str] = []
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            check_deadline()
            if isinstance(item, str):
                items.extend([part.strip() for part in item.split(",") if part.strip()])
    if not items:
        return None
    return set(items)


def _diagnostics_for_path(path_str: str, project_root: Path | None) -> list[Diagnostic]:
    forest = Forest()
    with forest_scope(forest):
        check_deadline()
        result = analyze_paths(
            [Path(path_str)],
            forest=forest,
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
        check_deadline()
        span_map = result.param_spans_by_path.get(path, {})
        for fn_name, group_list in bundles.items():
            check_deadline()
            param_spans = span_map.get(fn_name, {})
            for bundle in group_list:
                check_deadline()
                message = f"Implicit bundle detected: {', '.join(sorted(bundle))}"
                for name in sorted(bundle):
                    check_deadline()
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
    payload = _require_payload(payload, command=DATAFLOW_COMMAND)
    profile_enabled = _truthy_flag(payload.get("deadline_profile"))
    profile_root_value = payload.get("root") or ls.workspace.root_path or "."
    profile_token = set_deadline_profile(
        project_root=Path(str(profile_root_value)),
        enabled=profile_enabled,
    )
    deadline = _deadline_from_payload(payload)
    deadline_token = set_deadline(deadline)
    forest = Forest()
    forest_token = set_forest(forest)
    try:
        root = payload.get("root") or ls.workspace.root_path or "."
        config_path = payload.get("config")
        defaults = dataflow_defaults(
            Path(root), Path(config_path) if config_path else None
        )
        deadline_roots = set(dataflow_deadline_roots(defaults))
        decision_section = decision_defaults(
            Path(root), Path(config_path) if config_path else None
        )
        decision_tiers = decision_tier_map(decision_section)
        decision_require = decision_require_tiers(decision_section)
        exception_section = exception_defaults(
            Path(root), Path(config_path) if config_path else None
        )
        never_exceptions = set(exception_never_list(exception_section))
        fingerprint_section = fingerprint_defaults(
            Path(root), Path(config_path) if config_path else None
        )
        synth_min_occurrences = 0
        synth_version = "synth@1"
        if isinstance(fingerprint_section, dict):
            try:
                synth_min_occurrences = int(
                    fingerprint_section.get("synth_min_occurrences", 0) or 0
                )
            except (TypeError, ValueError):
                synth_min_occurrences = 0
            synth_version = str(
                fingerprint_section.get("synth_version", synth_version) or synth_version
            )
        fingerprint_registry: PrimeRegistry | None = None
        fingerprint_index: dict[Fingerprint, set[str]] = {}
        constructor_registry: TypeConstructorRegistry | None = None
        fingerprint_spec: dict[str, JSONValue] = {}
        if isinstance(fingerprint_section, dict):
            fingerprint_spec = {
                key: value
                for key, value in fingerprint_section.items()
                if not str(key).startswith("synth_")
            }
        if fingerprint_spec:
            registry, index = build_fingerprint_registry(fingerprint_spec)
            if index:
                fingerprint_registry = registry
                fingerprint_index = index
                constructor_registry = TypeConstructorRegistry(registry)
        payload = merge_payload(payload, defaults)
        deadline_roots = set(payload.get("deadline_roots", deadline_roots))

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
        lint = bool(payload.get("lint", False))
        exclude_dirs = set(payload.get("exclude", []))
        ignore_params = set(payload.get("ignore_params", []))
        decision_ignore_params = set(ignore_params)
        decision_ignore_params.update(decision_ignore_list(decision_section))
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
        decision_snapshot_path = payload.get("decision_snapshot")
        emit_test_obsolescence = bool(payload.get("emit_test_obsolescence", False))
        emit_test_obsolescence_state = bool(
            payload.get("emit_test_obsolescence_state", False)
        )
        test_obsolescence_state_path = payload.get("test_obsolescence_state")
        emit_test_obsolescence_delta = bool(
            payload.get("emit_test_obsolescence_delta", False)
        )
        write_test_obsolescence_baseline = bool(
            payload.get("write_test_obsolescence_baseline", False)
        )
        emit_test_evidence_suggestions = bool(
            payload.get("emit_test_evidence_suggestions", False)
        )
        emit_call_clusters = bool(payload.get("emit_call_clusters", False))
        emit_call_cluster_consolidation = bool(
            payload.get("emit_call_cluster_consolidation", False)
        )
        emit_test_annotation_drift = bool(
            payload.get("emit_test_annotation_drift", False)
        )
        test_annotation_drift_state_path = payload.get("test_annotation_drift_state")
        emit_test_annotation_drift_delta = bool(
            payload.get("emit_test_annotation_drift_delta", False)
        )
        write_test_annotation_drift_baseline = bool(
            payload.get("write_test_annotation_drift_baseline", False)
        )
        emit_ambiguity_delta = bool(payload.get("emit_ambiguity_delta", False))
        emit_ambiguity_state = bool(payload.get("emit_ambiguity_state", False))
        ambiguity_state_path = payload.get("ambiguity_state")
        write_ambiguity_baseline = bool(payload.get("write_ambiguity_baseline", False))
        synthesis_max_tier = payload.get("synthesis_max_tier", 2)
        synthesis_min_bundle_size = payload.get("synthesis_min_bundle_size", 2)
        synthesis_allow_singletons = payload.get("synthesis_allow_singletons", False)
        synthesis_protocols_path = payload.get("synthesis_protocols")
        synthesis_protocols_kind = payload.get("synthesis_protocols_kind", "dataclass")
        refactor_plan = payload.get("refactor_plan", False)
        refactor_plan_json = payload.get("refactor_plan_json")
        fingerprint_synth_json = payload.get("fingerprint_synth_json")
        fingerprint_provenance_json = payload.get("fingerprint_provenance_json")
        fingerprint_deadness_json = payload.get("fingerprint_deadness_json")
        fingerprint_coherence_json = payload.get("fingerprint_coherence_json")
        fingerprint_rewrite_plans_json = payload.get("fingerprint_rewrite_plans_json")
        fingerprint_exception_obligations_json = payload.get(
            "fingerprint_exception_obligations_json"
        )
        fingerprint_handledness_json = payload.get("fingerprint_handledness_json")

        config = AuditConfig(
            project_root=Path(root),
            exclude_dirs=exclude_dirs,
            ignore_params=ignore_params,
            decision_ignore_params=decision_ignore_params,
            external_filter=not allow_external,
            strictness=strictness,
            transparent_decorators=transparent_decorators,
            decision_tiers=decision_tiers,
            decision_require_tiers=decision_require,
            never_exceptions=never_exceptions,
            deadline_roots=deadline_roots,
            fingerprint_registry=fingerprint_registry,
            fingerprint_index=fingerprint_index,
            constructor_registry=constructor_registry,
            fingerprint_synth_min_occurrences=synth_min_occurrences,
            fingerprint_synth_version=synth_version,
        )
        if fail_on_type_ambiguities:
            type_audit = True
        include_decisions = bool(report_path) or bool(decision_snapshot_path) or bool(
            fail_on_violations
        )
        if decision_tiers:
            include_decisions = True
        include_rewrite_plans = bool(report_path) or bool(fingerprint_rewrite_plans_json)
        include_exception_obligations = bool(report_path) or bool(
            fingerprint_exception_obligations_json
        )
        include_handledness_witnesses = bool(report_path) or bool(
            fingerprint_handledness_json
        )
        include_never_invariants = bool(report_path)
        include_ambiguities = bool(report_path) or lint or emit_ambiguity_state
        if (emit_ambiguity_delta or write_ambiguity_baseline) and not ambiguity_state_path:
            include_ambiguities = True
        include_coherence = (
            bool(report_path) or bool(fingerprint_coherence_json) or include_rewrite_plans
        )
        needs_analysis = (
            bool(report_path)
            or bool(dot_path)
            or bool(structure_tree_path)
            or bool(structure_metrics_path)
            or bool(decision_snapshot_path)
            or bool(synthesis_plan_path)
            or bool(synthesis_report)
            or bool(synthesis_protocols_path)
            or bool(refactor_plan)
            or bool(refactor_plan_json)
            or bool(fingerprint_synth_json)
            or bool(fingerprint_provenance_json)
            or bool(fingerprint_deadness_json)
            or bool(fingerprint_coherence_json)
            or bool(fingerprint_rewrite_plans_json)
            or bool(fingerprint_exception_obligations_json)
            or bool(fingerprint_handledness_json)
            or bool(type_audit)
            or bool(type_audit_report)
            or bool(fail_on_type_ambiguities)
            or bool(fail_on_violations)
            or baseline_path is not None
            or bool(lint)
            or bool(emit_test_evidence_suggestions)
            or bool(include_ambiguities)
        )
        if needs_analysis:
            analysis = analyze_paths(
                paths,
                forest=forest,
                recursive=not no_recursive,
                type_audit=type_audit or type_audit_report,
                type_audit_report=type_audit_report,
                type_audit_max=type_audit_max,
                include_constant_smells=bool(report_path),
                include_unused_arg_smells=bool(report_path),
                include_deadness_witnesses=bool(report_path)
                or bool(fingerprint_deadness_json),
                include_coherence_witnesses=include_coherence,
                include_rewrite_plans=include_rewrite_plans,
                include_exception_obligations=include_exception_obligations,
                include_handledness_witnesses=include_handledness_witnesses,
                include_never_invariants=include_never_invariants,
                include_deadline_obligations=bool(report_path) or lint,
                include_decision_surfaces=include_decisions,
                include_value_decision_surfaces=include_decisions,
                include_invariant_propositions=bool(report_path),
                include_lint_lines=lint,
                include_ambiguities=include_ambiguities,
                include_bundle_forest=True,
                config=config,
            )
        else:
            analysis = AnalysisResult(
                groups_by_path={},
                param_spans_by_path={},
                bundle_sites_by_path={},
                type_suggestions=[],
                type_ambiguities=[],
                type_callsite_evidence=[],
                constant_smells=[],
                unused_arg_smells=[],
                forest=forest,
            )

        response: dict = {
            "type_suggestions": analysis.type_suggestions,
            "type_ambiguities": analysis.type_ambiguities,
            "type_callsite_evidence": analysis.type_callsite_evidence,
            "unused_arg_smells": analysis.unused_arg_smells,
            "decision_surfaces": analysis.decision_surfaces,
            "value_decision_surfaces": analysis.value_decision_surfaces,
            "value_decision_rewrites": analysis.value_decision_rewrites,
            "decision_warnings": analysis.decision_warnings,
            "fingerprint_warnings": analysis.fingerprint_warnings,
            "fingerprint_matches": analysis.fingerprint_matches,
            "fingerprint_synth": analysis.fingerprint_synth,
            "fingerprint_synth_registry": analysis.fingerprint_synth_registry,
            "fingerprint_provenance": analysis.fingerprint_provenance,
            "fingerprint_deadness": analysis.deadness_witnesses,
            "fingerprint_coherence": analysis.coherence_witnesses,
            "fingerprint_rewrite_plans": analysis.rewrite_plans,
            "fingerprint_exception_obligations": analysis.exception_obligations,
            "fingerprint_handledness": analysis.handledness_witnesses,
            "never_invariants": analysis.never_invariants,
            "deadline_obligations": analysis.deadline_obligations,
            "ambiguity_witnesses": analysis.ambiguity_witnesses,
            "invariant_propositions": [
                prop.as_dict() for prop in analysis.invariant_propositions
            ],
            "context_suggestions": analysis.context_suggestions,
        }
        if lint:
            response["lint_lines"] = analysis.lint_lines

        synthesis_plan: JSONObject | None = None
        if synthesis_plan_path or synthesis_report or synthesis_protocols_path:
            try:
                synthesis_plan = build_synthesis_plan(
                    analysis.groups_by_path,
                    project_root=Path(root),
                    max_tier=synthesis_max_tier,
                    min_bundle_size=synthesis_min_bundle_size,
                    allow_singletons=synthesis_allow_singletons,
                    merge_overlap_threshold=payload.get("merge_overlap_threshold", None),
                    config=config,
                )
            except (TypeError, ValueError, OSError) as exc:
                response.setdefault("synthesis_errors", []).append(str(exc))
            if synthesis_plan is not None:
                if synthesis_plan_path:
                    output = json.dumps(synthesis_plan, indent=2, sort_keys=True)
                    if synthesis_plan_path == "-":
                        response["synthesis_plan"] = synthesis_plan
                    else:
                        Path(synthesis_plan_path).write_text(output)
                if synthesis_report:
                    response["synthesis_plan"] = synthesis_plan
        if synthesis_protocols_path and synthesis_plan is not None:
            output = render_protocol_stubs(
                synthesis_plan,
                kind=synthesis_protocols_kind,
            )
            if synthesis_protocols_path == "-":
                response["synthesis_protocols"] = output
            else:
                Path(synthesis_protocols_path).write_text(output)
        if refactor_plan or refactor_plan_json:
            plan_payload = build_refactor_plan(
                analysis.groups_by_path,
                paths,
                config=config,
            )
            if refactor_plan_json:
                if refactor_plan_json == "-":
                    response["refactor_plan"] = plan_payload
                else:
                    Path(refactor_plan_json).write_text(
                        json.dumps(plan_payload, indent=2, sort_keys=True)
                    )
            if refactor_plan:
                response["refactor_plan"] = plan_payload

        if decision_snapshot_path is not None:
            payload_value = render_decision_snapshot(
                decision_surfaces=analysis.decision_surfaces,
                value_decision_surfaces=analysis.value_decision_surfaces,
                forest=analysis.forest,
                project_root=Path(root),
            )
            if decision_snapshot_path == "-":
                response["decision_snapshot"] = payload_value
            else:
                Path(decision_snapshot_path).write_text(
                    json.dumps(payload_value, indent=2, sort_keys=True)
                )

        if structure_tree_path is not None:
            payload_value = render_structure_snapshot(
                analysis.groups_by_path,
                forest=analysis.forest,
                project_root=Path(root),
                invariant_propositions=analysis.invariant_propositions,
            )
            if structure_tree_path == "-":
                response["structure_tree"] = payload_value
            else:
                Path(structure_tree_path).write_text(
                    json.dumps(payload_value, indent=2, sort_keys=True)
                )

        if structure_metrics_path is not None:
            metrics = compute_structure_metrics(
                analysis.groups_by_path,
                forest=analysis.forest,
            )
            if structure_metrics_path == "-":
                response["structure_metrics"] = metrics
            else:
                Path(structure_metrics_path).write_text(
                    json.dumps(metrics, indent=2, sort_keys=True)
                )

        if emit_test_obsolescence_delta and write_test_obsolescence_baseline:
            never(
                "conflicting obsolescence flags",
                emit_test_obsolescence_delta=emit_test_obsolescence_delta,
                write_test_obsolescence_baseline=write_test_obsolescence_baseline,
            )
        if emit_test_annotation_drift_delta and write_test_annotation_drift_baseline:
            never(
                "conflicting annotation drift flags",
                emit_test_annotation_drift_delta=emit_test_annotation_drift_delta,
                write_test_annotation_drift_baseline=write_test_annotation_drift_baseline,
            )
        if emit_ambiguity_delta and write_ambiguity_baseline:
            never(
                "conflicting ambiguity flags",
                emit_ambiguity_delta=emit_ambiguity_delta,
                write_ambiguity_baseline=write_ambiguity_baseline,
            )
        if emit_test_obsolescence_state and test_obsolescence_state_path:
            never(
                "conflicting obsolescence state flags",
                emit_test_obsolescence_state=emit_test_obsolescence_state,
                test_obsolescence_state_path=test_obsolescence_state_path,
            )
        if emit_ambiguity_state and ambiguity_state_path:
            never(
                "conflicting ambiguity state flags",
                emit_ambiguity_state=emit_ambiguity_state,
                ambiguity_state_path=ambiguity_state_path,
            )

        if emit_test_evidence_suggestions:
            report_root = Path(root)
            evidence_path = report_root / "out" / "test_evidence.json"
            entries = test_evidence_suggestions.load_test_evidence(str(evidence_path))
            suggestions, summary = test_evidence_suggestions.suggest_evidence(
                entries,
                root=report_root,
                paths=paths,
                forest=analysis.forest,
                config=config,
            )
            suggestions_payload = test_evidence_suggestions.render_json_payload(
                suggestions, summary
            )
            report_md = test_evidence_suggestions.render_markdown(suggestions, summary)
            out_dir, artifact_dir = _output_dirs(report_root)
            report_json = json.dumps(suggestions_payload, indent=2, sort_keys=True) + "\n"
            (artifact_dir / "test_evidence_suggestions.json").write_text(report_json)
            (out_dir / "test_evidence_suggestions.md").write_text(report_md)
            response["test_evidence_suggestions_summary"] = (
                suggestions_payload.get("summary", {})
            )
        if emit_call_clusters:
            report_root = Path(root)
            evidence_path = report_root / "out" / "test_evidence.json"
            clusters_payload = call_clusters.build_call_clusters_payload(
                paths,
                root=report_root,
                evidence_path=evidence_path,
                config=config,
            )
            report_md = call_clusters.render_markdown(clusters_payload)
            out_dir, artifact_dir = _output_dirs(report_root)
            report_json = json.dumps(clusters_payload, indent=2, sort_keys=True) + "\n"
            (artifact_dir / "call_clusters.json").write_text(report_json)
            (out_dir / "call_clusters.md").write_text(report_md)
            response["call_clusters_summary"] = clusters_payload.get("summary", {})
        if emit_call_cluster_consolidation:
            report_root = Path(root)
            evidence_path = report_root / "out" / "test_evidence.json"
            consolidation_payload = (
                call_cluster_consolidation.build_call_cluster_consolidation_payload(
                    evidence_path=evidence_path,
                )
            )
            report_md = call_cluster_consolidation.render_markdown(consolidation_payload)
            out_dir, artifact_dir = _output_dirs(report_root)
            report_json = (
                json.dumps(consolidation_payload, indent=2, sort_keys=True) + "\n"
            )
            (artifact_dir / "call_cluster_consolidation.json").write_text(report_json)
            (out_dir / "call_cluster_consolidation.md").write_text(report_md)
            response["call_cluster_consolidation_summary"] = consolidation_payload.get(
                "summary", {}
            )
        drift_payload = None
        if test_annotation_drift_state_path:
            state_path = Path(test_annotation_drift_state_path)
            if not state_path.exists():
                never("annotation drift state not found", path=str(state_path))
            payload_value = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(payload_value, dict):
                never("annotation drift state must be a JSON object")
            drift_payload = payload_value
        elif (
            emit_test_annotation_drift
            or emit_test_annotation_drift_delta
            or write_test_annotation_drift_baseline
        ):
            report_root = Path(root)
            evidence_path = report_root / "out" / "test_evidence.json"
            drift_payload = test_annotation_drift.build_annotation_drift_payload(
                paths,
                root=report_root,
                evidence_path=evidence_path,
            )
        if drift_payload is not None and (
            emit_test_annotation_drift
            or emit_test_annotation_drift_delta
            or write_test_annotation_drift_baseline
        ):
            report_root = Path(root)
            out_dir, artifact_dir = _output_dirs(report_root)
            if emit_test_annotation_drift:
                report_json = json.dumps(drift_payload, indent=2, sort_keys=True) + "\n"
                report_md = test_annotation_drift.render_markdown(drift_payload)
                (artifact_dir / "test_annotation_drift.json").write_text(report_json)
                (out_dir / "test_annotation_drift.md").write_text(report_md)
                response["test_annotation_drift_summary"] = drift_payload.get(
                    "summary", {}
                )
            if emit_test_annotation_drift_delta or write_test_annotation_drift_baseline:
                summary = drift_payload.get("summary", {})
                baseline_payload = test_annotation_drift_delta.build_baseline_payload(
                    summary if isinstance(summary, dict) else {}
                )
                baseline_path = test_annotation_drift_delta.resolve_baseline_path(
                    report_root
                )
                response["test_annotation_drift_baseline_path"] = str(baseline_path)
                if write_test_annotation_drift_baseline:
                    baseline_path.parent.mkdir(parents=True, exist_ok=True)
                    test_annotation_drift_delta.write_baseline(
                        str(baseline_path), baseline_payload
                    )
                    response["test_annotation_drift_baseline_written"] = True
                else:
                    response["test_annotation_drift_baseline_written"] = False
                if emit_test_annotation_drift_delta:
                    if not baseline_path.exists():
                        never("annotation drift baseline not found", path=str(baseline_path))
                    baseline = test_annotation_drift_delta.load_baseline(
                        str(baseline_path)
                    )
                    current = test_annotation_drift_delta.parse_baseline_payload(
                        baseline_payload
                    )
                    delta_payload = test_annotation_drift_delta.build_delta_payload(
                        baseline, current, baseline_path=str(baseline_path)
                    )
                    report_json = json.dumps(delta_payload, indent=2, sort_keys=True) + "\n"
                    report_md = test_annotation_drift_delta.render_markdown(delta_payload)
                    (artifact_dir / "test_annotation_drift_delta.json").write_text(
                        report_json
                    )
                    (out_dir / "test_annotation_drift_delta.md").write_text(report_md)
                    response["test_annotation_drift_delta_summary"] = delta_payload.get(
                        "summary", {}
                    )
        obsolescence_candidates: list[dict[str, object]] | None = None
        obsolescence_summary: dict[str, int] | None = None
        obsolescence_baseline_payload: dict[str, object] | None = None
        obsolescence_baseline: test_obsolescence_delta.ObsolescenceBaseline | None = None
        if test_obsolescence_state_path:
            state_path = Path(test_obsolescence_state_path)
            if not state_path.exists():
                never("test obsolescence state not found", path=str(state_path))
            state = test_obsolescence_state.load_state(str(state_path))
            obsolescence_candidates = [
                {str(k): entry[k] for k in entry} for entry in state.candidates
            ]
            obsolescence_summary = state.baseline.summary
            obsolescence_baseline_payload = {
                str(k): state.baseline_payload[k] for k in state.baseline_payload
            }
            obsolescence_baseline = state.baseline
        elif (
            emit_test_obsolescence
            or emit_test_obsolescence_delta
            or write_test_obsolescence_baseline
            or emit_test_obsolescence_state
        ):
            report_root = Path(root)
            evidence_path = report_root / "out" / "test_evidence.json"
            risk_registry_path = report_root / "out" / "evidence_risk_registry.json"
            evidence_by_test, status_by_test = test_obsolescence.load_test_evidence(
                str(evidence_path)
            )
            risk_registry = test_obsolescence.load_risk_registry(str(risk_registry_path))
            candidates, summary_counts = test_obsolescence.classify_candidates(
                evidence_by_test, status_by_test, risk_registry
            )
            obsolescence_candidates = candidates
            obsolescence_summary = summary_counts
            obsolescence_baseline_payload = test_obsolescence_delta.build_baseline_payload(
                evidence_by_test, status_by_test, candidates, summary_counts
            )
            obsolescence_baseline = test_obsolescence_delta.parse_baseline_payload(
                obsolescence_baseline_payload
            )
            if emit_test_obsolescence_state:
                out_dir, artifact_dir = _output_dirs(report_root)
                state_payload = test_obsolescence_state.build_state_payload(
                    evidence_by_test,
                    status_by_test,
                    candidates,
                    summary_counts,
                )
                (artifact_dir / "test_obsolescence_state.json").write_text(
                    json.dumps(state_payload, indent=2, sort_keys=True) + "\n"
                )

        if emit_test_obsolescence and obsolescence_candidates is not None:
            report_root = Path(root)
            report_payload = test_obsolescence.render_json_payload(
                obsolescence_candidates, obsolescence_summary or {}
            )
            out_dir, artifact_dir = _output_dirs(report_root)
            report_json = json.dumps(report_payload, indent=2, sort_keys=True) + "\n"
            report_md = test_obsolescence.render_markdown(
                obsolescence_candidates, obsolescence_summary or {}
            )
            (artifact_dir / "test_obsolescence_report.json").write_text(report_json)
            (out_dir / "test_obsolescence_report.md").write_text(report_md)
            response["test_obsolescence_summary"] = obsolescence_summary or {}

        if (
            emit_test_obsolescence_delta or write_test_obsolescence_baseline
        ) and obsolescence_baseline_payload is not None:
            report_root = Path(root)
            baseline_path = test_obsolescence_delta.resolve_baseline_path(report_root)
            response["test_obsolescence_baseline_path"] = str(baseline_path)
            if write_test_obsolescence_baseline:
                baseline_path.parent.mkdir(parents=True, exist_ok=True)
                test_obsolescence_delta.write_baseline(
                    str(baseline_path), obsolescence_baseline_payload
                )
                response["test_obsolescence_baseline_written"] = True
            else:
                response["test_obsolescence_baseline_written"] = False
            if emit_test_obsolescence_delta:
                if not baseline_path.exists():
                    never("test obsolescence baseline not found", path=str(baseline_path))
                baseline = test_obsolescence_delta.load_baseline(str(baseline_path))
                current = (
                    obsolescence_baseline
                    if obsolescence_baseline is not None
                    else test_obsolescence_delta.parse_baseline_payload(
                        obsolescence_baseline_payload
                    )
                )
                delta_payload = test_obsolescence_delta.build_delta_payload(
                    baseline, current, baseline_path=str(baseline_path)
                )
                out_dir, artifact_dir = _output_dirs(report_root)
                report_json = json.dumps(delta_payload, indent=2, sort_keys=True) + "\n"
                report_md = test_obsolescence_delta.render_markdown(delta_payload)
                (artifact_dir / "test_obsolescence_delta.json").write_text(report_json)
                (out_dir / "test_obsolescence_delta.md").write_text(report_md)
                response["test_obsolescence_delta_summary"] = delta_payload.get(
                    "summary", {}
                )

        ambiguity_witnesses: list[dict[str, object]] | None = None
        ambiguity_baseline_payload: dict[str, object] | None = None
        ambiguity_baseline: ambiguity_delta.AmbiguityBaseline | None = None
        if ambiguity_state_path:
            state_path = Path(ambiguity_state_path)
            if not state_path.exists():
                never("ambiguity state not found", path=str(state_path))
            state = ambiguity_state.load_state(
                str(state_path),
            )
            ambiguity_witnesses = [
                {str(k): entry[k] for k in entry} for entry in state.witnesses
            ]
            ambiguity_baseline_payload = ambiguity_delta.build_baseline_payload(
                ambiguity_witnesses,
            )
            ambiguity_baseline = state.baseline
        elif emit_ambiguity_delta or write_ambiguity_baseline or emit_ambiguity_state:
            ambiguity_witnesses = [
                {str(k): entry[k] for k in entry} for entry in analysis.ambiguity_witnesses
            ]
            if emit_ambiguity_state:
                report_root = Path(root)
                out_dir, artifact_dir = _output_dirs(report_root)
                state_payload = ambiguity_state.build_state_payload(
                    ambiguity_witnesses,
                )
                (artifact_dir / "ambiguity_state.json").write_text(
                    json.dumps(state_payload, indent=2, sort_keys=True) + "\n"
                )
            ambiguity_baseline_payload = ambiguity_delta.build_baseline_payload(
                ambiguity_witnesses,
            )
            ambiguity_baseline = ambiguity_delta.parse_baseline_payload(
                ambiguity_baseline_payload,
            )
        if (
            emit_ambiguity_delta or write_ambiguity_baseline
        ) and ambiguity_baseline_payload is not None:
            report_root = Path(root)
            baseline_path = ambiguity_delta.resolve_baseline_path(report_root)
            response["ambiguity_baseline_path"] = str(baseline_path)
            if write_ambiguity_baseline:
                baseline_path.parent.mkdir(parents=True, exist_ok=True)
                ambiguity_delta.write_baseline(
                    str(baseline_path), ambiguity_baseline_payload
                )
                response["ambiguity_baseline_written"] = True
            else:
                response["ambiguity_baseline_written"] = False
            if emit_ambiguity_delta:
                if not baseline_path.exists():
                    never("ambiguity baseline not found", path=str(baseline_path))
                baseline = ambiguity_delta.load_baseline(str(baseline_path))
                current = (
                    ambiguity_baseline
                    if ambiguity_baseline is not None
                    else ambiguity_delta.parse_baseline_payload(
                        ambiguity_baseline_payload
                    )
                )
                delta_payload = ambiguity_delta.build_delta_payload(
                    baseline, current, baseline_path=str(baseline_path)
                )
                out_dir, artifact_dir = _output_dirs(report_root)
                report_json = json.dumps(delta_payload, indent=2, sort_keys=True) + "\n"
                report_md = ambiguity_delta.render_markdown(delta_payload)
                (artifact_dir / "ambiguity_delta.json").write_text(report_json)
                (out_dir / "ambiguity_delta.md").write_text(report_md)
                response["ambiguity_delta_summary"] = delta_payload.get("summary", {})

        report: str | None = None
        violations: list[str] = []
        effective_violations: list[str] | None = None
        baseline_entries: list[str] = []
        if report_path:
            report_carrier = ReportCarrier.from_analysis_result(analysis)
            report, violations = render_report(
                analysis.groups_by_path,
                max_components,
                report=report_carrier,
            )
            if baseline_path is not None:
                baseline_entries = load_baseline(baseline_path)
                if baseline_write:
                    write_baseline(baseline_path, violations)
                    effective_violations = []
                else:
                    effective_violations, _ = apply_baseline(violations, baseline_entries)
            if report_path:
                if decision_snapshot_path:
                    decision_payload = render_decision_snapshot(
                        decision_surfaces=analysis.decision_surfaces,
                        value_decision_surfaces=analysis.value_decision_surfaces,
                        forest=analysis.forest,
                        project_root=Path(root),
                    )
                    report = report + "\n" + json.dumps(
                        decision_payload, indent=2, sort_keys=True
                    )
                if structure_tree_path:
                    structure_payload = render_structure_snapshot(
                        analysis.groups_by_path,
                        forest=analysis.forest,
                        project_root=Path(root),
                        invariant_propositions=analysis.invariant_propositions,
                    )
                    report = report + "\n" + json.dumps(
                        structure_payload, indent=2, sort_keys=True
                    )
                if structure_metrics_path:
                    report = report + "\n" + json.dumps(metrics, indent=2, sort_keys=True)
                if synthesis_plan and (
                    synthesis_report or synthesis_plan_path or synthesis_protocols_path
                ):
                    report = report + render_synthesis_section(synthesis_plan)
                if refactor_plan and (refactor_plan or refactor_plan_json):
                    report = report + render_refactor_plan(plan_payload)
                Path(report_path).write_text(report)
        else:
            violation_carrier = ReportCarrier(
                forest=analysis.forest,
                type_suggestions=analysis.type_suggestions if type_audit_report else [],
                type_ambiguities=analysis.type_ambiguities if type_audit_report else [],
                decision_warnings=analysis.decision_warnings,
                fingerprint_warnings=analysis.fingerprint_warnings,
                parse_failure_witnesses=analysis.parse_failure_witnesses,
            )
            violations = compute_violations(
                analysis.groups_by_path,
                max_components,
                report=violation_carrier,
            )
            if baseline_path is not None:
                baseline_entries = load_baseline(baseline_path)
                if baseline_write:
                    write_baseline(baseline_path, violations)
                    effective_violations = []
                else:
                    effective_violations, _ = apply_baseline(violations, baseline_entries)

        if dot_path and analysis.forest is not None:
            dot_payload = render_dot(analysis.forest)
            if dot_path == "-":
                response["dot"] = dot_payload
                if report is not None:
                    report = report + "\n" + dot_payload
            else:
                Path(dot_path).write_text(dot_payload)

        if effective_violations is None:
            effective_violations = violations
        response["violations"] = len(effective_violations)
        if fingerprint_synth_json and analysis.fingerprint_synth_registry:
            payload_json = json.dumps(
                analysis.fingerprint_synth_registry, indent=2, sort_keys=True
            )
            if fingerprint_synth_json == "-":
                response["fingerprint_synth_registry"] = analysis.fingerprint_synth_registry
            else:
                Path(fingerprint_synth_json).write_text(payload_json)
        if fingerprint_provenance_json and analysis.fingerprint_provenance:
            payload_json = json.dumps(
                analysis.fingerprint_provenance, indent=2, sort_keys=True
            )
            if fingerprint_provenance_json == "-":
                response["fingerprint_provenance"] = analysis.fingerprint_provenance
            else:
                Path(fingerprint_provenance_json).write_text(payload_json)
        if fingerprint_deadness_json is not None:
            payload_json = json.dumps(
                analysis.deadness_witnesses, indent=2, sort_keys=True
            )
            if fingerprint_deadness_json == "-":
                response["fingerprint_deadness"] = analysis.deadness_witnesses
            else:
                Path(fingerprint_deadness_json).write_text(payload_json)
        if fingerprint_coherence_json is not None:
            payload_json = json.dumps(
                analysis.coherence_witnesses, indent=2, sort_keys=True
            )
            if fingerprint_coherence_json == "-":
                response["fingerprint_coherence"] = analysis.coherence_witnesses
            else:
                Path(fingerprint_coherence_json).write_text(payload_json)
        if fingerprint_rewrite_plans_json is not None:
            payload_json = json.dumps(analysis.rewrite_plans, indent=2, sort_keys=True)
            if fingerprint_rewrite_plans_json == "-":
                response["fingerprint_rewrite_plans"] = analysis.rewrite_plans
            else:
                Path(fingerprint_rewrite_plans_json).write_text(payload_json)
        if fingerprint_exception_obligations_json is not None:
            payload_json = json.dumps(
                analysis.exception_obligations, indent=2, sort_keys=True
            )
            if fingerprint_exception_obligations_json == "-":
                response["fingerprint_exception_obligations"] = (
                    analysis.exception_obligations
                )
            else:
                Path(fingerprint_exception_obligations_json).write_text(payload_json)
        if fingerprint_handledness_json is not None:
            payload_json = json.dumps(
                analysis.handledness_witnesses, indent=2, sort_keys=True
            )
            if fingerprint_handledness_json == "-":
                response["fingerprint_handledness"] = analysis.handledness_witnesses
            else:
                Path(fingerprint_handledness_json).write_text(payload_json)
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
    except TimeoutExceeded as exc:
        return {
            "exit_code": 2,
            "timeout": True,
            "timeout_context": exc.context.as_payload(),
        }
    finally:
        reset_forest(forest_token)
        reset_deadline(deadline_token)
        reset_deadline_profile(profile_token)


@server.command(SYNTHESIS_COMMAND)

def execute_synthesis(ls: LanguageServer, payload: dict | None = None) -> dict:
    payload = _require_payload(payload, command=SYNTHESIS_COMMAND)
    with _deadline_scope_from_payload(payload):
        check_deadline()
        try:
            request = SynthesisRequest.model_validate(payload)
        except ValidationError as exc:
            return {"protocols": [], "warnings": [], "errors": [str(exc)]}

        bundle_tiers: dict[frozenset[str], int] = {}
        for entry in request.bundles:
            check_deadline()
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
    payload = _require_payload(payload, command=REFACTOR_COMMAND)
    with _deadline_scope_from_payload(payload):
        try:
            request = RefactorRequest.model_validate(payload)
        except ValidationError as exc:
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
    payload = _require_payload(payload, command=STRUCTURE_DIFF_COMMAND)
    with _deadline_scope_from_payload(payload):
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


@server.command(STRUCTURE_REUSE_COMMAND)
def execute_structure_reuse(ls: LanguageServer, payload: dict | None = None) -> dict:
    payload = _require_payload(payload, command=STRUCTURE_REUSE_COMMAND)
    with _deadline_scope_from_payload(payload):
        snapshot_path = payload.get("snapshot")
        lemma_stubs_path = payload.get("lemma_stubs")
        min_count = payload.get("min_count", 2)
        if not snapshot_path:
            return {"exit_code": 2, "errors": ["snapshot path is required"]}
        try:
            snapshot = load_structure_snapshot(Path(snapshot_path))
        except ValueError as exc:
            return {"exit_code": 2, "errors": [str(exc)]}
        try:
            min_count_int = int(min_count)
        except (TypeError, ValueError):
            return {"exit_code": 2, "errors": ["min_count must be an integer"]}
        if min_count_int <= 0:
            return {"exit_code": 2, "errors": ["min_count must be positive"]}
        reuse = compute_structure_reuse(snapshot, min_count=min_count_int)
        response: JSONObject = {"exit_code": 0, "reuse": reuse}
        if lemma_stubs_path:
            stubs = render_reuse_lemma_stubs(reuse)
            if lemma_stubs_path == "-":
                response["lemma_stubs"] = stubs
            else:
                Path(lemma_stubs_path).write_text(stubs)
        return response


@server.command(DECISION_DIFF_COMMAND)
def execute_decision_diff(ls: LanguageServer, payload: dict | None = None) -> dict:
    payload = _require_payload(payload, command=DECISION_DIFF_COMMAND)
    with _deadline_scope_from_payload(payload):
        baseline_path = payload.get("baseline")
        current_path = payload.get("current")
        if not baseline_path or not current_path:
            return {
                "exit_code": 2,
                "errors": ["baseline and current decision snapshot paths are required"],
            }
        try:
            baseline = load_decision_snapshot(Path(baseline_path))
            current = load_decision_snapshot(Path(current_path))
        except ValueError as exc:
            return {"exit_code": 2, "errors": [str(exc)]}
        return {"exit_code": 0, "diff": diff_decision_snapshots(baseline, current)}


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
