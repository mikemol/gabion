# gabion:boundary_normalization_module
from __future__ import annotations
# gabion:decision_protocol_module

from itertools import zip_longest
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, cast

from gabion.ingest.adapter_contract import NormalizedIngestBundle
from gabion.ingest.registry import resolve_adapter
from gabion.server_core.command_contract import CommandRuntimeInput, CommandRuntimeState
from gabion.server_core.command_effects import CommandEffects
from gabion.server_core.command_reducers import (
    initial_collection_progress,
    initial_paths_count,
    normalize_paths,
    normalize_timeout_total_ticks,
)

if TYPE_CHECKING:
    from gabion.server import ExecuteCommandDeps


_BOUND = False
_DURATION_TIMEOUT_CLOCK_MULTIPLIER = 16


def _bind_server_symbols() -> None:
    global _BOUND
    if _BOUND:
        return
    from gabion import server as _server

    module_globals = globals()
    for name, value in _server.__dict__.items():
        module_globals.setdefault(name, value)
    _BOUND = True


def _record_trace_1cell(
    *,
    execute_deps: CommandEffects,
    state: object | None,
    kind: str,
    source_label: str,
    target_label: str,
    representative: str,
    basis_path: tuple[str, ...],
    surface: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    if state is None:
        return
    execute_deps.record_1cell_fn(
        state,
        kind=kind,
        source_label=source_label,
        target_label=target_label,
        representative=representative,
        basis_path=basis_path,
        surface=surface,
        metadata=metadata,
    )


def _reject_removed_legacy_payload_keys(payload: dict[str, object]) -> None:
    removed_keys = {
        "resume_checkpoint": "--resume-checkpoint",
        "resume_on_timeout": "--resume-on-timeout",
        "emit_timeout_progress_report": "--emit-timeout-progress-report",
        "emit_checkpoint_intro_timeline": "--emit-checkpoint-intro-timeline",
    }
    present_flags: list[str] = []
    present_payload_keys: list[str] = []
    for payload_key, cli_flag in removed_keys.items():
        if payload_key not in payload:
            continue
        present_flags.append(cli_flag)
        present_payload_keys.append(payload_key)
    if not present_flags:
        return
    never(
        "removed legacy check timeout/resume flags",
        flags=present_flags,
        payload_keys=present_payload_keys,
        guidance=(
            "Use ASPF state handoff via aspf_state_json/aspf_import_state and "
            "optional aspf_delta_jsonl."
        ),
    )




@dataclass(frozen=True)
class _DataflowCapabilityAnnotations:
    selected_adapter: str
    supported_analysis_surfaces: list[str]
    disabled_surface_reasons: dict[str, str]


@dataclass(frozen=True)
class _TimeoutIngressCarrier:
    has_tick_timeout: bool
    has_duration_timeout: bool


@dataclass(frozen=True)
class _AuxOperationIngressCarrier:
    domain: str
    action: str
    state_in: object
    baseline_path: Path | None


@dataclass(frozen=True)
class _CommandPayloadIngressCarrier:
    payload: dict[str, object]
    dataflow_capabilities: _DataflowCapabilityAnnotations
    timeout: _TimeoutIngressCarrier
    aux_operation: _AuxOperationIngressCarrier | None


def _normalize_command_payload_ingress(
    *,
    payload: dict[str, object],
    root: Path,
) -> _CommandPayloadIngressCarrier:
    normalized_payload, dataflow_capabilities = _normalize_dataflow_format_controls(payload)
    has_tick_timeout = (
        normalized_payload.get("analysis_timeout_ticks") not in (None, "")
        or normalized_payload.get("analysis_timeout_tick_ns") not in (None, "")
    )
    has_duration_timeout = (
        normalized_payload.get("analysis_timeout_ms") not in (None, "")
        or normalized_payload.get("analysis_timeout_seconds") not in (None, "")
    )
    aux_operation_raw = normalized_payload.get("aux_operation")
    aux_operation: _AuxOperationIngressCarrier | None = None
    if aux_operation_raw is not None:
        if not isinstance(aux_operation_raw, dict):
            never("invalid aux operation payload", payload_type=type(aux_operation_raw).__name__)
        aux_domain = str(aux_operation_raw.get("domain", "")).strip().lower()
        aux_action = str(aux_operation_raw.get("action", "")).strip().lower()
        aux_state_in = aux_operation_raw.get("state_in")
        aux_baseline_path = resolve_baseline_path(
            aux_operation_raw.get("baseline_path"),
            root,
        )
        if aux_domain not in {"obsolescence", "annotation-drift", "ambiguity"}:
            never(
                "invalid aux operation domain",
                domain=aux_domain,
                action=aux_action,
            )
        if aux_action in {"delta", "baseline-write"} and aux_baseline_path is None:
            never(
                "aux operation missing baseline path",
                domain=aux_domain,
                action=aux_action,
            )
        aux_operation = _AuxOperationIngressCarrier(
            domain=aux_domain,
            action=aux_action,
            state_in=aux_state_in,
            baseline_path=aux_baseline_path,
        )
    return _CommandPayloadIngressCarrier(
        payload=normalized_payload,
        dataflow_capabilities=dataflow_capabilities,
        timeout=_TimeoutIngressCarrier(
            has_tick_timeout=has_tick_timeout,
            has_duration_timeout=has_duration_timeout,
        ),
        aux_operation=aux_operation,
    )


def _normalize_dataflow_format_controls(
    payload: dict[str, object],
) -> tuple[dict[str, object], _DataflowCapabilityAnnotations]:
    supported_surfaces = {
        "decision_surfaces": "Decision-surface extraction and reporting.",
        "value_decision_surfaces": "Value-encoded decision-surface extraction.",
        "type_ambiguities": "Type-ambiguity detection and reporting.",
        "rewrite_plans": "Fingerprint rewrite-plan analysis and projection.",
    }
    profile_matrix: dict[str, tuple[list[str], dict[str, str]]] = {
        "default": (
            [
                "decision_surfaces",
                "value_decision_surfaces",
                "type_ambiguities",
                "rewrite_plans",
            ],
            {},
        ),
        "syntax-only": (
            [],
            {
                surface: "disabled by ingest profile syntax-only"
                for surface in supported_surfaces
            },
        ),
    }
    raw_language = payload.get("language")
    normalized_language = (
        str(raw_language).strip().lower() if raw_language is not None else "python"
    )
    if not normalized_language:
        never("empty dataflow language")  # pragma: no cover - invariant sink
    if normalized_language != "python":
        never(  # pragma: no cover - invariant sink
            "unsupported dataflow language",
            language=normalized_language,
            supported_languages=["python"],
        )
    raw_ingest_profile = payload.get("ingest_profile")
    normalized_ingest_profile = (
        str(raw_ingest_profile).strip().lower()
        if raw_ingest_profile is not None
        else "default"
    )
    if not normalized_ingest_profile:
        never("empty dataflow ingest profile", language=normalized_language)  # pragma: no cover - invariant sink
    if normalized_ingest_profile not in profile_matrix:
        never(
            "unsupported dataflow ingest profile",
            language=normalized_language,
            ingest_profile=normalized_ingest_profile,
            supported_ingest_profiles=list(profile_matrix),
        )
    surfaces, disabled_surface_reasons = profile_matrix[normalized_ingest_profile]
    selected_adapter = f"{normalized_language}:{normalized_ingest_profile}"
    normalized_payload = boundary_order.apply_boundary_updates_once(
        payload,
        {
            "language": normalized_language,
            "ingest_profile": normalized_ingest_profile,
            "selected_adapter": selected_adapter,
            "supported_analysis_surfaces": sort_once(
                list(surfaces),
                source="server_core.command_orchestrator._normalize_dataflow_format_controls.supported_analysis_surfaces",
            ),
            "disabled_surface_reasons": {
                surface: disabled_surface_reasons[surface]
                for surface in sort_once(
                    disabled_surface_reasons,
                    source="server_core.command_orchestrator._normalize_dataflow_format_controls.disabled_surface_keys",
                )
            },
        },
        source="server_core.command_orchestrator._normalize_dataflow_format_controls.payload",
    )
    return normalized_payload, _DataflowCapabilityAnnotations(
        selected_adapter=selected_adapter,
        supported_analysis_surfaces=list(normalized_payload["supported_analysis_surfaces"]),
        disabled_surface_reasons=dict(
            cast(dict[str, str], normalized_payload["disabled_surface_reasons"])
        ),
    )


def _normalize_duration_timeout_clock_ticks(
    *,
    timeout: _TimeoutIngressCarrier,
    total_ticks: int,
) -> int:
    if timeout.has_tick_timeout:
        return total_ticks
    if not timeout.has_duration_timeout:
        return total_ticks
    return max(1, total_ticks * _DURATION_TIMEOUT_CLOCK_MULTIPLIER)


@dataclass(frozen=True)
class _AuxiliaryMode:
    domain: str
    kind: str
    state_path: object
    baseline_path_override: Path | None = None

    @property
    def emit_report(self) -> bool:
        return self.kind == "report"

    @property
    def emit_state(self) -> bool:
        return self.kind in {"state", "delta"}

    @property
    def emit_delta(self) -> bool:
        return self.kind == "delta"

    @property
    def write_baseline(self) -> bool:
        return self.kind == "baseline-write"


@dataclass(frozen=True)
class _AuxiliaryModeSelection:
    obsolescence: _AuxiliaryMode
    annotation_drift: _AuxiliaryMode
    ambiguity: _AuxiliaryMode


def _auxiliary_mode_from_payload(
    *,
    payload: dict[str, object],
    mode_key: str,
    state_key: str,
    emit_state_key: str,
    emit_delta_key: str,
    write_baseline_key: str,
    emit_report_key: str | None,
    domain: str,
    allow_report: bool,
) -> _AuxiliaryMode:
    raw_mode = payload.get(mode_key)
    if isinstance(raw_mode, Mapping):
        kind = str(raw_mode.get("kind", "off") or "off").strip().lower()
        state_path = raw_mode.get("state_path")
    else:
        emit_report = (
            bool(payload.get(emit_report_key, False)) if emit_report_key is not None else False
        )
        emit_state = bool(payload.get(emit_state_key, False))
        emit_delta = bool(payload.get(emit_delta_key, False))
        write_baseline = bool(payload.get(write_baseline_key, False))
        state_path = payload.get(state_key)
        enabled = sum(1 for flag in (emit_report, emit_state, emit_delta, write_baseline) if flag)
        if enabled > 1:
            never(
                "conflicting auxiliary mode flags",
                domain=domain,
                emit_report=emit_report,
                emit_state=emit_state,
                emit_delta=emit_delta,
                write_baseline=write_baseline,
            )
        if emit_report:
            kind = "report"
        elif emit_state:
            kind = "state"
        elif emit_delta:
            kind = "delta"
        elif write_baseline:
            kind = "baseline-write"
        else:
            kind = "off"
    allowed = {"off", "state", "delta", "baseline-write"}
    if allow_report:
        allowed.add("report")
    if kind not in allowed:
        never("invalid auxiliary mode", domain=domain, kind=kind, allowed=sorted(allowed))
    if kind == "state" and state_path not in (None, ""):
        never("state mode does not accept state path", domain=domain, state_path=state_path)
    return _AuxiliaryMode(domain=domain, kind=kind, state_path=state_path)


def _emit_annotation_drift_outputs(
    *,
    response: dict[str, object],
    root: str,
    paths: list[Path],
    test_annotation_drift_state_path: object,
    emit_test_annotation_drift: bool,
    emit_test_annotation_drift_delta: bool,
    write_test_annotation_drift_baseline: bool,
    annotation_drift_baseline_path: Path | None = None,
) -> None:
    drift_payload = None
    if test_annotation_drift_state_path:
        state_path = Path(str(test_annotation_drift_state_path))
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
            report_json = json.dumps(drift_payload, indent=2, sort_keys=False) + "\n"
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
            baseline_path = (
                annotation_drift_baseline_path
                if annotation_drift_baseline_path is not None
                else test_annotation_drift_delta.resolve_baseline_path(report_root)
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
                report_json = json.dumps(delta_payload, indent=2, sort_keys=False) + "\n"
                report_md = test_annotation_drift_delta.render_markdown(delta_payload)
                (artifact_dir / "test_annotation_drift_delta.json").write_text(
                    report_json
                )
                (out_dir / "test_annotation_drift_delta.md").write_text(report_md)
                response["test_annotation_drift_delta_summary"] = delta_payload.get(
                    "summary", {}
                )


def _emit_test_obsolescence_outputs(
    *,
    response: dict[str, object],
    root: str,
    emit_test_obsolescence: bool,
    emit_test_obsolescence_state: bool,
    test_obsolescence_state_path: object,
    emit_test_obsolescence_delta: bool,
    write_test_obsolescence_baseline: bool,
    obsolescence_baseline_path: Path | None = None,
) -> None:
    obsolescence_candidates: list[dict[str, object]] | None = None
    obsolescence_summary: dict[str, int] | None = None
    obsolescence_active_summary: dict[str, int] | None = None
    obsolescence_baseline_payload: dict[str, object] | None = None
    obsolescence_baseline: test_obsolescence_delta.ObsolescenceBaseline | None = None
    if test_obsolescence_state_path:
        state_path = Path(str(test_obsolescence_state_path))
        if not state_path.exists():
            never("test obsolescence state not found", path=str(state_path))
        state = test_obsolescence_state.load_state(str(state_path))
        obsolescence_candidates = [
            {str(k): entry[k] for k in entry} for entry in state.candidates
        ]
        obsolescence_summary = state.baseline.summary
        active_payload = state.baseline.active
        active_summary_value = active_payload.get("summary", {})
        if isinstance(active_summary_value, dict):
            obsolescence_active_summary = {
                str(key): int(value) if isinstance(value, int) else 0
                for key, value in active_summary_value.items()
                if isinstance(key, str)
            }
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
        classification = test_obsolescence.classify_candidates(
            evidence_by_test, status_by_test, risk_registry
        )
        obsolescence_candidates = classification.stale_candidates
        obsolescence_summary = classification.stale_summary
        obsolescence_active_summary = classification.active_summary
        obsolescence_baseline_payload = test_obsolescence_delta.build_baseline_payload(
            evidence_by_test,
            status_by_test,
            classification.stale_candidates,
            classification.stale_summary,
            active_tests=classification.active_tests,
            active_summary=classification.active_summary,
        )
        obsolescence_baseline = test_obsolescence_delta.parse_baseline_payload(
            obsolescence_baseline_payload
        )
        if emit_test_obsolescence_state:
            _out_dir, artifact_dir = _output_dirs(report_root)
            state_payload = test_obsolescence_state.build_state_payload(
                evidence_by_test,
                status_by_test,
                classification.stale_candidates,
                classification.stale_summary,
                active_tests=classification.active_tests,
                active_summary=classification.active_summary,
            )
            (artifact_dir / "test_obsolescence_state.json").write_text(
                json.dumps(state_payload, indent=2, sort_keys=False) + "\n"
            )

    if emit_test_obsolescence and obsolescence_candidates is not None:
        report_root = Path(root)
        report_payload = test_obsolescence.render_json_payload(
            obsolescence_candidates, obsolescence_summary or {}
        )
        out_dir, artifact_dir = _output_dirs(report_root)
        report_json = json.dumps(report_payload, indent=2, sort_keys=False) + "\n"
        report_md = test_obsolescence.render_markdown(
            obsolescence_candidates, obsolescence_summary or {}
        )
        (artifact_dir / "test_obsolescence_report.json").write_text(report_json)
        (out_dir / "test_obsolescence_report.md").write_text(report_md)
        response["test_obsolescence_summary"] = obsolescence_summary or {}
        response["test_obsolescence_active_summary"] = obsolescence_active_summary or {}

    if (
        emit_test_obsolescence_delta or write_test_obsolescence_baseline
    ) and obsolescence_baseline_payload is not None:
        report_root = Path(root)
        baseline_path = (
            obsolescence_baseline_path
            if obsolescence_baseline_path is not None
            else test_obsolescence_delta.resolve_baseline_path(report_root)
        )
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
            report_json = json.dumps(delta_payload, indent=2, sort_keys=False) + "\n"
            report_md = test_obsolescence_delta.render_markdown(delta_payload)
            (artifact_dir / "test_obsolescence_delta.json").write_text(report_json)
            (out_dir / "test_obsolescence_delta.md").write_text(report_md)
            response["test_obsolescence_delta_summary"] = delta_payload.get(
                "summary", {}
            )


def _emit_ambiguity_outputs(
    *,
    response: dict[str, object],
    analysis: AnalysisResult,
    root: str,
    ambiguity_state_path: object,
    emit_ambiguity_delta: bool,
    emit_ambiguity_state: bool,
    write_ambiguity_baseline: bool,
    ambiguity_baseline_path: Path | None = None,
) -> None:
    ambiguity_witnesses: list[dict[str, object]] | None = None
    ambiguity_baseline_payload: dict[str, object] | None = None
    ambiguity_baseline: ambiguity_delta.AmbiguityBaseline | None = None
    if ambiguity_state_path:
        state_path = Path(str(ambiguity_state_path))
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
            _out_dir, artifact_dir = _output_dirs(report_root)
            state_payload = ambiguity_state.build_state_payload(
                ambiguity_witnesses,
            )
            (artifact_dir / "ambiguity_state.json").write_text(
                json.dumps(state_payload, indent=2, sort_keys=False) + "\n"
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
        baseline_path = (
            ambiguity_baseline_path
            if ambiguity_baseline_path is not None
            else ambiguity_delta.resolve_baseline_path(report_root)
        )
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
            report_json = json.dumps(delta_payload, indent=2, sort_keys=False) + "\n"
            report_md = ambiguity_delta.render_markdown(delta_payload)
            (artifact_dir / "ambiguity_delta.json").write_text(report_json)
            (out_dir / "ambiguity_delta.md").write_text(report_md)
            response["ambiguity_delta_summary"] = delta_payload.get("summary", {})


def _apply_auxiliary_artifact_outputs(
    *,
    response: dict[str, object],
    analysis: AnalysisResult,
    root: str,
    paths: list[Path],
    config: AuditConfig,
    name_filter_bundle: DataflowNameFilterBundle,
    emit_test_obsolescence: bool,
    emit_test_obsolescence_state: bool,
    test_obsolescence_state_path: object,
    emit_test_obsolescence_delta: bool,
    write_test_obsolescence_baseline: bool,
    emit_test_evidence_suggestions: bool,
    emit_call_clusters: bool,
    emit_call_cluster_consolidation: bool,
    emit_test_annotation_drift: bool,
    emit_semantic_coverage_map: bool,
    test_annotation_drift_state_path: object,
    semantic_coverage_mapping_path: object,
    emit_test_annotation_drift_delta: bool,
    write_test_annotation_drift_baseline: bool,
    emit_ambiguity_delta: bool,
    emit_ambiguity_state: bool,
    ambiguity_state_path: object,
    write_ambiguity_baseline: bool,
    obsolescence_baseline_path: Path | None,
    annotation_drift_baseline_path: Path | None,
    ambiguity_baseline_path: Path | None,
) -> None:
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
        report_json = json.dumps(suggestions_payload, indent=2, sort_keys=False) + "\n"
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
        report_json = json.dumps(clusters_payload, indent=2, sort_keys=False) + "\n"
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
            json.dumps(consolidation_payload, indent=2, sort_keys=False) + "\n"
        )
        (artifact_dir / "call_cluster_consolidation.json").write_text(report_json)
        (out_dir / "call_cluster_consolidation.md").write_text(report_md)
        response["call_cluster_consolidation_summary"] = consolidation_payload.get(
            "summary", {}
        )
    _emit_annotation_drift_outputs(
        response=response,
        root=root,
        paths=paths,
        test_annotation_drift_state_path=test_annotation_drift_state_path,
        emit_test_annotation_drift=emit_test_annotation_drift,
        emit_test_annotation_drift_delta=emit_test_annotation_drift_delta,
        write_test_annotation_drift_baseline=write_test_annotation_drift_baseline,
        annotation_drift_baseline_path=annotation_drift_baseline_path,
    )

    if emit_semantic_coverage_map:
        report_root = Path(root)
        _out_dir, artifact_dir = _output_dirs(report_root)
        mapping_path = (
            Path(str(semantic_coverage_mapping_path))
            if semantic_coverage_mapping_path
            else report_root / "out" / "semantic_coverage_mapping.json"
        )
        evidence_path = report_root / "out" / "test_evidence.json"
        semantic_payload = semantic_coverage_map.build_semantic_coverage_payload(
            paths=paths,
            root=Path(root),
            mapping_path=mapping_path,
            evidence_path=evidence_path,
            exclude=name_filter_bundle.exclude_dirs,
        )
        report_md = semantic_coverage_map.render_markdown(semantic_payload)
        semantic_coverage_map.write_semantic_coverage(
            semantic_payload,
            output_path=artifact_dir / "semantic_coverage_map.json",
        )
        (report_root / "artifacts" / "audit_reports").mkdir(parents=True, exist_ok=True)
        (report_root / "artifacts" / "audit_reports" / "semantic_coverage_map.md").write_text(report_md)
        response["semantic_coverage_map_summary"] = semantic_payload.get("summary", {})

    _emit_test_obsolescence_outputs(
        response=response,
        root=root,
        emit_test_obsolescence=emit_test_obsolescence,
        emit_test_obsolescence_state=emit_test_obsolescence_state,
        test_obsolescence_state_path=test_obsolescence_state_path,
        emit_test_obsolescence_delta=emit_test_obsolescence_delta,
        write_test_obsolescence_baseline=write_test_obsolescence_baseline,
        obsolescence_baseline_path=obsolescence_baseline_path,
    )
    _emit_ambiguity_outputs(
        response=response,
        analysis=analysis,
        root=root,
        ambiguity_state_path=ambiguity_state_path,
        emit_ambiguity_delta=emit_ambiguity_delta,
        emit_ambiguity_state=emit_ambiguity_state,
        write_ambiguity_baseline=write_ambiguity_baseline,
        ambiguity_baseline_path=ambiguity_baseline_path,
    )


@dataclass(frozen=True)
class _PrimaryOutputContext:
    analysis: AnalysisResult
    root: str
    paths: list[Path]
    payload: dict[str, object]
    config: AuditConfig
    synthesis_plan_path: object
    synthesis_report: bool
    synthesis_protocols_path: object
    synthesis_protocols_kind: object
    synthesis_max_tier: int
    synthesis_min_bundle_size: int
    synthesis_allow_singletons: bool
    refactor_plan: bool
    refactor_plan_json: object
    decision_snapshot_path: object
    structure_tree_path: object
    structure_metrics_path: object


@dataclass(frozen=True)
class _PrimaryOutputArtifacts:
    synthesis_plan: JSONObject | None
    refactor_plan_payload: JSONObject | None
    structure_metrics_payload: JSONObject | None


@dataclass(frozen=True)
class _ReportFinalizationContext:
    analysis: AnalysisResult
    root: str
    max_components: int
    report_path: object
    report_output_path: Path | None
    projection_rows: list[JSONObject]
    report_section_journal_path: Path
    report_section_witness_digest: str | None
    report_phase_checkpoint_path: Path | None
    analysis_resume_state_path: Path | None
    analysis_resume_reused_files: int
    type_audit_report: bool
    baseline_path: Path | None
    baseline_write: bool
    decision_snapshot_path: object
    structure_tree_path: object
    structure_metrics_path: object
    structure_metrics_payload: JSONObject | None
    synthesis_plan: JSONObject | None
    synthesis_plan_path: object
    synthesis_report: bool
    synthesis_protocols_path: object
    refactor_plan: bool
    refactor_plan_json: object
    refactor_plan_payload: JSONObject | None


@dataclass(frozen=True)
class _ReportFinalizationOutcome:
    report: str | None
    violations: list[str]
    effective_violations: list[str]
    phase_checkpoint_state: JSONObject


@dataclass(frozen=True)
class _TimeoutCleanupContext:
    timeout_hard_deadline_ns: int
    cleanup_grace_ns: int
    timeout_total_ns: int
    analysis_window_ns: int
    analysis_resume_state_path: Path | None
    analysis_resume_input_manifest_digest: str | None
    last_collection_resume_payload: JSONObject | None
    execute_deps: CommandEffects
    analysis_resume_input_witness: JSONObject | None
    emit_phase_timeline: bool
    phase_timeline_path: Path
    analysis_resume_total_files: int
    analysis_resume_state_status: str | None
    analysis_resume_reused_files: int
    profile_enabled: bool
    latest_collection_progress: JSONObject
    semantic_progress_cumulative: JSONObject | None
    report_output_path: Path | None
    projection_rows: list[JSONObject]
    report_phase_checkpoint_path: Path | None
    report_section_journal_path: Path
    report_section_witness_digest: str | None
    phase_checkpoint_state: JSONObject
    enable_phase_projection_checkpoints: bool
    forest: Forest
    analysis_resume_intro_payload: JSONObject | None
    runtime_root: Path
    initial_paths_count_value: int
    execution_plan: ExecutionPlan
    aspf_trace_state: object | None
    ensure_report_sections_cache_fn: object
    emit_lsp_progress_fn: object
    dataflow_capabilities: _DataflowCapabilityAnnotations = field(
        default_factory=lambda: _DataflowCapabilityAnnotations(
            selected_adapter="python:default",
            supported_analysis_surfaces=[
                "decision_surfaces",
                "value_decision_surfaces",
                "type_ambiguities",
                "rewrite_plans",
            ],
            disabled_surface_reasons={},
        )
    )
    analysis_resume_source: str = "cold_start"
    analysis_resume_state_compatibility_status: str | None = None


@dataclass(frozen=True)
class _ProgressEmitter:
    emit: object
    stop: object
    emit_phase_progress_events: bool


@dataclass(frozen=True)
class _AnalysisExecutionContext:
    execute_deps: CommandEffects
    aspf_trace_state: object | None
    runtime_state: CommandRuntimeState
    forest: Forest
    paths: list[Path]
    no_recursive: bool
    type_audit: bool
    type_audit_report: bool
    type_audit_max: int
    report_path: object
    include_coherence: bool
    include_rewrite_plans: bool
    include_exception_obligations: bool
    include_handledness_witnesses: bool
    include_never_invariants: bool
    include_wl_refinement: bool
    include_decisions: bool
    lint: bool
    include_ambiguities: bool
    config: AuditConfig
    needs_analysis: bool
    file_paths_for_run: list[Path] | None
    analysis_resume_intro_payload: JSONObject | None
    analysis_resume_reused_files: int
    analysis_resume_total_files: int
    analysis_resume_state_path: Path | None
    analysis_resume_state_status: str | None
    analysis_resume_input_manifest_digest: str | None
    analysis_resume_input_witness: JSONObject | None
    analysis_resume_intro_timeline_header: str | None
    analysis_resume_intro_timeline_row: str | None
    phase_timeline_path: Path
    emit_phase_timeline: bool
    enable_phase_projection_checkpoints: bool
    report_output_path: Path | None
    projection_rows: list[JSONObject]
    report_section_journal_path: Path
    report_section_witness_digest: str | None
    report_phase_checkpoint_path: Path | None
    phase_checkpoint_state: JSONObject
    profile_enabled: bool
    emit_phase_progress_events: bool
    fingerprint_deadness_json: object
    emit_lsp_progress_fn: object
    ensure_report_sections_cache_fn: object
    clear_report_sections_cache_reason_fn: object
    check_deadline_fn: object
    profiling_stage_ns: dict[str, int]
    profiling_counters: dict[str, int]


@dataclass(frozen=True)
class _AnalysisExecutionOutcome:
    analysis: AnalysisResult
    last_collection_resume_payload: JSONObject | None
    semantic_progress_cumulative: JSONObject | None
    latest_collection_progress: JSONObject


@dataclass
class _AnalysisExecutionMutableState:
    last_collection_resume_payload: JSONObject | None
    semantic_progress_cumulative: JSONObject | None
    latest_collection_progress: JSONObject


@dataclass
class _AnalysisResumePreparationState:
    analysis_resume_state_path: Path | None
    analysis_resume_input_witness: JSONObject | None
    analysis_resume_input_manifest_digest: str | None
    analysis_resume_total_files: int
    analysis_resume_reused_files: int
    analysis_resume_state_status: str | None
    analysis_resume_state_compatibility_status: str | None
    analysis_resume_intro_payload: JSONObject | None
    analysis_resume_intro_timeline_header: str | None
    analysis_resume_intro_timeline_row: str | None
    report_section_witness_digest: str | None
    phase_checkpoint_state: JSONObject
    semantic_progress_cumulative: JSONObject | None
    last_collection_resume_payload: JSONObject | None
    analysis_resume_source: str = "cold_start"


def _aspf_import_state_paths(
    payload_value: list[str] | None,
    *,
    root: Path,
) -> tuple[Path, ...]:
    if payload_value is None:
        return ()
    resolved: list[Path] = []
    for text in payload_value:
        candidate = Path(text)
        resolved.append(candidate if candidate.is_absolute() else (root / candidate))
    return tuple(resolved)


def _prepare_analysis_resume_state(
    *,
    execute_deps: CommandEffects,
    aspf_trace_state: object | None,
    needs_analysis: bool,
    normalized_ingest: NormalizedIngestBundle,
    root: str,
    payload: Mapping[str, object],
    aspf_import_state: list[str] | None = None,
    no_recursive: bool,
    report_path: object,
    include_wl_refinement: bool,
    config: AuditConfig,
    report_output_path: Path | None,
    state: _AnalysisResumePreparationState,
    runtime_state: CommandRuntimeState,
) -> tuple[list[Path] | None, JSONObject | None]:
    file_paths_for_run: list[Path] | None = None
    collection_resume_payload: JSONObject | None = None
    if needs_analysis:
        resolved_root = Path(root)
        file_paths_for_run = list(normalized_ingest.file_paths)
        state.analysis_resume_total_files = len(file_paths_for_run)
        # Legacy checkpoint payload ingress is hard-rejected; ASPF import-state is the
        # only supported continuation path for active orchestration.
        state.analysis_resume_state_path = None
        input_manifest = execute_deps.analysis_input_manifest_fn(
            root=resolved_root,
            file_paths=file_paths_for_run,
            recursive=not no_recursive,
            include_invariant_propositions=bool(report_path),
            include_wl_refinement=include_wl_refinement,
            config=config,
        )
        state.analysis_resume_input_manifest_digest = (
            execute_deps.analysis_input_manifest_digest_fn(input_manifest)
        )
        import_state_paths = _aspf_import_state_paths(
            aspf_import_state,
            root=resolved_root,
        )
        if import_state_paths:
            aspf_resume_payload = cast(
                Mapping[str, object],
                execute_deps.load_aspf_resume_state_fn(
                    import_state_paths=import_state_paths
                )
                or {},
            )
            imported_manifest_digest_raw = aspf_resume_payload.get(
                "analysis_manifest_digest"
            )
            imported_manifest_digest = (
                str(imported_manifest_digest_raw)
                if isinstance(imported_manifest_digest_raw, str)
                else None
            )
            resume_projection = cast(
                Mapping[str, object],
                aspf_resume_payload.get("resume_projection", {}),
            )
            raw_collection_resume = cast(
                Mapping[str, object],
                resume_projection.get("collection_resume", {}),
            )
            imported_collection_resume: JSONObject = {
                str(key): raw_collection_resume[key]
                for key in raw_collection_resume
            }
            current_manifest_digest = state.analysis_resume_input_manifest_digest
            resume_available = bool(imported_collection_resume)
            manifest_available = isinstance(imported_manifest_digest, str)
            current_manifest_available = isinstance(current_manifest_digest, str)
            manifest_match = (
                manifest_available
                and current_manifest_available
                and imported_manifest_digest == current_manifest_digest
            )
            compatibility_status = {
                (False, False, False, False): "aspf_state_missing_collection_resume",
                (False, False, False, True): "aspf_state_missing_collection_resume",
                (False, False, True, False): "aspf_state_missing_collection_resume",
                (False, False, True, True): "aspf_state_missing_collection_resume",
                (False, True, False, False): "aspf_state_missing_collection_resume",
                (False, True, False, True): "aspf_state_missing_collection_resume",
                (False, True, True, False): "aspf_state_missing_collection_resume",
                (False, True, True, True): "aspf_state_missing_collection_resume",
                (True, False, False, False): "aspf_state_missing_manifest_digest",
                (True, False, False, True): "aspf_state_missing_manifest_digest",
                (True, False, True, False): "aspf_state_missing_manifest_digest",
                (True, False, True, True): "aspf_state_missing_manifest_digest",
                (True, True, False, False): "aspf_state_missing_current_manifest_digest",
                (True, True, False, True): "aspf_state_missing_current_manifest_digest",
                (True, True, True, False): "aspf_state_manifest_mismatch",
                (True, True, True, True): "aspf_state_compatible",
            }[
                (
                    resume_available,
                    manifest_available,
                    current_manifest_available,
                    manifest_match,
                )
            ]
            state.analysis_resume_state_status = {
                "aspf_state_compatible": "aspf_state_loaded",
            }.get(compatibility_status, "aspf_state_skipped")
            collection_resume_payload = {
                "aspf_state_compatible": imported_collection_resume,
            }.get(compatibility_status)
            state.analysis_resume_reused_files = _analysis_resume_progress(
                collection_resume=collection_resume_payload or {},
                total_files=state.analysis_resume_total_files,
            )["completed_files"]
            state.analysis_resume_source = {
                "aspf_state_loaded": "aspf_state",
            }.get(state.analysis_resume_state_status, "cold_start")
            state.analysis_resume_state_compatibility_status = compatibility_status
            _record_trace_1cell(
                execute_deps=execute_deps,
                state=aspf_trace_state,
                kind="resume_load",
                source_label="runtime:aspf_state",
                target_label="analysis:resume_seed",
                representative=str(state.analysis_resume_state_status),
                basis_path=("resume", "load", "aspf_state"),
                surface="delta_state",
                metadata={
                    "import_state_paths": [str(path) for path in import_state_paths],
                    "status": state.analysis_resume_state_status,
                    "compatibility_status": compatibility_status,
                    "import_manifest_digest": imported_manifest_digest,
                    "current_manifest_digest": state.analysis_resume_input_manifest_digest,
                },
            )
        if state.analysis_resume_state_status is None:
            state.analysis_resume_state_status = "cold_start"
        if state.analysis_resume_state_compatibility_status is None:
            state.analysis_resume_state_compatibility_status = "cold_start"
        state.report_section_witness_digest = _report_witness_digest(
            input_witness=state.analysis_resume_input_witness,
            manifest_digest=state.analysis_resume_input_manifest_digest,
        )
        if report_output_path is not None:
            state.phase_checkpoint_state = {}
    state.last_collection_resume_payload = collection_resume_payload
    collection_resume = cast(Mapping[str, object], collection_resume_payload or {})
    raw_semantic_progress = cast(
        Mapping[str, object], collection_resume.get("semantic_progress", {})
    )
    state.semantic_progress_cumulative = {
        str(key): raw_semantic_progress[key] for key in raw_semantic_progress
    }
    runtime_state.semantic_progress_cumulative = dict(state.semantic_progress_cumulative)
    return file_paths_for_run, collection_resume_payload


def _run_analysis_with_progress(
    *,
    context: _AnalysisExecutionContext,
    state: _AnalysisExecutionMutableState,
    collection_resume_payload: JSONObject | None,
) -> _AnalysisExecutionOutcome:
    last_collection_intro_signature: tuple[int, int, int, int] | None = None
    last_collection_semantic_witness_digest: str | None = None
    last_collection_checkpoint_flush_ns = 0
    last_analysis_index_resume_signature: tuple[
        int,
        str,
        int,
        int,
        str,
        str,
    ] | None = None
    last_collection_report_flush_ns = 0
    last_collection_report_flush_completed = -1
    phase_progress_signatures: dict[str, tuple[object, ...]] = {}
    phase_progress_last_flush_ns: dict[str, int] = {}

    if context.emit_phase_progress_events:
        context.emit_lsp_progress_fn(
            phase="collection",
            collection_progress={
                "completed_files": context.analysis_resume_reused_files,
                "in_progress_files": 0,
                "remaining_files": max(
                    context.analysis_resume_total_files
                    - context.analysis_resume_reused_files,
                    0,
                ),
                "total_files": context.analysis_resume_total_files,
            },
            semantic_progress=None,
            work_done=context.analysis_resume_reused_files,
            work_total=context.analysis_resume_total_files,
            include_timing=context.profile_enabled,
            analysis_state="analysis_collection_bootstrap",
            classification="aspf_resume_state_detected",
            event_kind="checkpoint",
        )

    def _persist_collection_resume(progress_payload: JSONObject) -> None:
        nonlocal last_collection_intro_signature
        nonlocal last_collection_semantic_witness_digest
        nonlocal last_collection_checkpoint_flush_ns
        nonlocal last_analysis_index_resume_signature
        nonlocal last_collection_report_flush_ns
        nonlocal last_collection_report_flush_completed
        context.profiling_counters["server.collection_resume_persist_calls"] += 1
        semantic_progress = context.execute_deps.collection_semantic_progress_fn(
            previous_collection_resume=state.last_collection_resume_payload,
            collection_resume=progress_payload,
            total_files=context.analysis_resume_total_files,
            cumulative=state.semantic_progress_cumulative,
        )
        state.semantic_progress_cumulative = semantic_progress
        context.runtime_state.semantic_progress_cumulative = dict(semantic_progress)
        persisted_progress_payload: JSONObject = {
            str(key): progress_payload[key] for key in progress_payload
        }
        persisted_progress_payload["semantic_progress"] = semantic_progress
        state.last_collection_resume_payload = persisted_progress_payload
        collection_progress = _analysis_resume_progress(
            collection_resume=persisted_progress_payload,
            total_files=context.analysis_resume_total_files,
        )
        state.latest_collection_progress = dict(collection_progress)
        context.runtime_state.latest_collection_progress = dict(collection_progress)
        context.emit_lsp_progress_fn(
            phase="collection",
            collection_progress=collection_progress,
            semantic_progress=semantic_progress,
            work_done=collection_progress.get("completed_files"),
            work_total=collection_progress.get("total_files"),
            include_timing=context.profile_enabled,
        )
        collection_intro_signature = (
            collection_progress["completed_files"],
            collection_progress["in_progress_files"],
            collection_progress["remaining_files"],
            _analysis_index_resume_hydrated_count(persisted_progress_payload),
        )
        semantic_witness_digest = semantic_progress.get("current_witness_digest")
        if not isinstance(semantic_witness_digest, str):
            semantic_witness_digest = None
        analysis_index_signature = _analysis_index_resume_signature(
            persisted_progress_payload
        )
        intro_changed = collection_intro_signature != last_collection_intro_signature
        semantic_changed = (
            semantic_witness_digest != last_collection_semantic_witness_digest
        )
        analysis_index_changed = (
            analysis_index_signature != last_analysis_index_resume_signature
        )
        raw_substantive_progress = semantic_progress.get("substantive_progress")
        semantic_substantive_progress = (
            raw_substantive_progress
            if isinstance(raw_substantive_progress, bool)
            else False
        )
        now_ns = time.monotonic_ns()
        if intro_changed and context.execute_deps.collection_checkpoint_flush_due_fn(
            intro_changed=True,
            remaining_files=collection_progress["remaining_files"],
            semantic_substantive_progress=semantic_substantive_progress,
            now_ns=now_ns,
            last_flush_ns=last_collection_checkpoint_flush_ns,
        ):
            last_collection_checkpoint_flush_ns = now_ns
        last_collection_semantic_witness_digest = semantic_witness_digest
        last_analysis_index_resume_signature = analysis_index_signature
        if not intro_changed:
            return
        last_collection_intro_signature = collection_intro_signature
        if not context.report_output_path or not context.projection_rows:
            return
        completed_files = collection_progress["completed_files"]
        _ = _collection_report_flush_due(
            completed_files=completed_files,
            remaining_files=collection_progress["remaining_files"],
            now_ns=now_ns,
            last_flush_ns=last_collection_report_flush_ns,
            last_flush_completed=last_collection_report_flush_completed,
        )
        if True:
            last_collection_report_flush_ns = now_ns
            last_collection_report_flush_completed = completed_files
            sections, journal_reason = context.ensure_report_sections_cache_fn()
            sections["intro"] = _collection_progress_intro_lines(
                collection_resume=persisted_progress_payload,
                total_files=context.analysis_resume_total_files,
                resume_state_intro=context.analysis_resume_intro_payload,
            )
            preview_groups_by_path = _groups_by_path_from_collection_resume(
                persisted_progress_payload
            )
            preview_report = ReportCarrier(
                forest=context.forest,
                parse_failure_witnesses=[],
            )
            preview_sections = context.execute_deps.project_report_sections_fn(
                preview_groups_by_path,
                preview_report,
                max_phase="post",
                include_previews=True,
                preview_only=True,
            )
            if True:
                sections.update(preview_sections)
            sections.setdefault(
                "components",
                _collection_components_preview_lines(
                    collection_resume=persisted_progress_payload
                ),
            )
            partial_report, pending_reasons = _render_incremental_report(
                analysis_state="analysis_collection_in_progress",
                progress_payload=persisted_progress_payload,
                projection_rows=context.projection_rows,
                sections=sections,
            )
            pending_reasons.pop("intro", None)
            _apply_journal_pending_reason(
                projection_rows=context.projection_rows,
                sections=sections,
                pending_reasons=pending_reasons,
                journal_reason=journal_reason,
            )
            context.report_output_path.parent.mkdir(parents=True, exist_ok=True)
            _write_text_profiled(
                context.report_output_path,
                partial_report,
                io_name="report_markdown.write",
            )
            _write_report_section_journal(
                path=context.report_section_journal_path,
                witness_digest=context.report_section_witness_digest,
                projection_rows=context.projection_rows,
                sections=sections,
                pending_reasons=pending_reasons,
            )
            context.clear_report_sections_cache_reason_fn()
            context.phase_checkpoint_state["collection"] = {
                "status": "checkpointed",
                "work_done": collection_progress["completed_files"],
                "work_total": collection_progress["total_files"],
                "completed_files": collection_progress["completed_files"],
                "in_progress_files": collection_progress["in_progress_files"],
                "remaining_files": collection_progress["remaining_files"],
                "total_files": collection_progress["total_files"],
                "section_ids": sort_once(sections, source="src/gabion/server.py:4603"),
            }

    def _projection_phase_signature(
        phase: Literal["collection", "forest", "edge", "post"],
        groups_by_path: Mapping[Path, dict[str, list[set[str]]]],
        report_carrier: ReportCarrier,
    ) -> tuple[int, ...]:
        context.check_deadline_fn()
        return (
            len(groups_by_path),
            len(report_carrier.forest.nodes),
            len(report_carrier.forest.alts),
            len(report_carrier.bundle_sites_by_path),
            len(report_carrier.type_suggestions),
            len(report_carrier.type_ambiguities),
            len(report_carrier.type_callsite_evidence),
            len(report_carrier.constant_smells),
            len(report_carrier.unused_arg_smells),
            len(report_carrier.deadness_witnesses),
            len(report_carrier.coherence_witnesses),
            len(report_carrier.rewrite_plans),
            len(report_carrier.exception_obligations),
            len(report_carrier.never_invariants),
            len(report_carrier.ambiguity_witnesses),
            len(report_carrier.handledness_witnesses),
            len(report_carrier.decision_surfaces),
            len(report_carrier.value_decision_surfaces),
            len(report_carrier.decision_warnings),
            len(report_carrier.fingerprint_warnings),
            len(report_carrier.fingerprint_matches),
            len(report_carrier.fingerprint_synth),
            len(report_carrier.fingerprint_provenance),
            len(report_carrier.context_suggestions),
            len(report_carrier.invariant_propositions),
            len(report_carrier.value_decision_rewrites),
            len(report_carrier.deadline_obligations),
            len(report_carrier.parse_failure_witnesses),
            report_projection_phase_rank(phase),
        )

    def _persist_projection_phase(
        phase: Literal["collection", "forest", "edge", "post"],
        groups_by_path: dict[Path, dict[str, list[set[str]]]],
        report_carrier: ReportCarrier,
        work_done: int,
        work_total: int,
    ) -> None:
        progress_marker = str(getattr(report_carrier, "progress_marker", "") or "")
        progress_analysis_state = f"analysis_{phase}_in_progress"
        phase_progress_v2 = (
            report_carrier.phase_progress_v2
            if isinstance(report_carrier.phase_progress_v2, Mapping)
            else None
        )
        context.emit_lsp_progress_fn(
            phase=phase,
            collection_progress=state.latest_collection_progress,
            semantic_progress=state.semantic_progress_cumulative,
            work_done=work_done,
            work_total=work_total,
            include_timing=context.profile_enabled,
            analysis_state=progress_analysis_state,
            phase_progress_v2=phase_progress_v2,
            progress_marker=progress_marker,
        )
        if not context.report_output_path or not context.projection_rows:
            return
        projection_started_ns = time.monotonic_ns()
        phase_signature = _projection_phase_signature(
            phase,
            groups_by_path,
            report_carrier,
        ) + (int(work_done), int(work_total), progress_marker)
        if phase_progress_signatures.get(phase) == phase_signature:
            return
        phase_progress_signatures[phase] = phase_signature
        now_ns = time.monotonic_ns()
        last_flush_ns = phase_progress_last_flush_ns.get(phase, 0)
        if not _projection_phase_flush_due(
            phase=phase,
            now_ns=now_ns,
            last_flush_ns=last_flush_ns,
        ):
            return
        phase_progress_last_flush_ns[phase] = now_ns
        available_sections = context.execute_deps.project_report_sections_fn(
            groups_by_path,
            report_carrier,
            max_phase="post",
            include_previews=True,
            preview_only=True,
        )
        _record_trace_1cell(
            execute_deps=context.execute_deps,
            state=context.aspf_trace_state,
            kind="report_projection",
            source_label="analysis:groups_by_path",
            target_label="report:sections",
            representative=f"projection:{phase}",
            basis_path=("report", "projection", str(phase)),
            surface="groups_by_path",
            metadata={
                "phase": phase,
                "section_count": len(available_sections),
            },
        )
        sections, journal_reason = context.ensure_report_sections_cache_fn()
        sections.update(available_sections)
        partial_report, pending_reasons = _render_incremental_report(
            analysis_state=progress_analysis_state,
            progress_payload={
                "phase": phase,
                "work_done": int(work_done),
                "work_total": int(work_total),
                "event_kind": "progress",
                "progress_marker": progress_marker,
                "phase_progress_v2": (
                    {str(key): phase_progress_v2[key] for key in phase_progress_v2}
                    if isinstance(phase_progress_v2, Mapping)
                    else None
                ),
            },
            projection_rows=context.projection_rows,
            sections=sections,
        )
        _apply_journal_pending_reason(
            projection_rows=context.projection_rows,
            sections=sections,
            pending_reasons=pending_reasons,
            journal_reason=journal_reason,
        )
        context.report_output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_text_profiled(
            context.report_output_path,
            partial_report,
            io_name="report_markdown.write",
        )
        _write_report_section_journal(
            path=context.report_section_journal_path,
            witness_digest=context.report_section_witness_digest,
            projection_rows=context.projection_rows,
            sections=sections,
            pending_reasons=pending_reasons,
        )
        context.clear_report_sections_cache_reason_fn()
        context.phase_checkpoint_state[phase] = {
            "status": "checkpointed",
            "work_done": int(work_done),
            "work_total": int(work_total),
            "section_ids": sort_once(sections, source="src/gabion/server.py:4748"),
            "resolved_sections": len(sections),
        }
        context.profiling_stage_ns["server.projection_emit"] += (
            time.monotonic_ns() - projection_started_ns
        )
        context.profiling_counters["server.projection_emit_calls"] += 1

    if context.needs_analysis and context.file_paths_for_run is not None:
        bootstrap_collection_resume = collection_resume_payload
        if bootstrap_collection_resume is None:
            seed_paths = context.file_paths_for_run[:1] if context.file_paths_for_run else []
            bootstrap_collection_resume = (
                context.execute_deps.build_analysis_collection_resume_seed_fn(
                    in_progress_paths=seed_paths
                )
            )
        _persist_collection_resume(bootstrap_collection_resume)

    if context.needs_analysis:
        analysis_started_ns = time.monotonic_ns()
        _record_trace_1cell(
            execute_deps=context.execute_deps,
            state=context.aspf_trace_state,
            kind="analysis_call_start",
            source_label="runtime:inputs",
            target_label="analysis:engine",
            representative="analyze_paths.start",
            basis_path=("analysis", "call", "start"),
        )
        analysis = context.execute_deps.analyze_paths_fn(
            context.paths,
            forest=context.forest,
            recursive=not context.no_recursive,
            type_audit=context.type_audit or context.type_audit_report,
            type_audit_report=context.type_audit_report,
            type_audit_max=context.type_audit_max,
            include_constant_smells=bool(context.report_path),
            include_unused_arg_smells=bool(context.report_path),
            include_deadness_witnesses=bool(context.report_path)
            or bool(context.fingerprint_deadness_json),
            include_coherence_witnesses=context.include_coherence,
            include_rewrite_plans=context.include_rewrite_plans,
            include_exception_obligations=context.include_exception_obligations,
            include_handledness_witnesses=context.include_handledness_witnesses,
            include_never_invariants=context.include_never_invariants,
            include_wl_refinement=context.include_wl_refinement,
            include_deadline_obligations=bool(context.report_path) or context.lint,
            include_decision_surfaces=context.include_decisions,
            include_value_decision_surfaces=context.include_decisions,
            include_invariant_propositions=bool(context.report_path),
            include_lint_lines=context.lint,
            include_ambiguities=context.include_ambiguities,
            include_bundle_forest=True,
            config=context.config,
            file_paths_override=context.file_paths_for_run,
            collection_resume=collection_resume_payload,
            on_collection_progress=_persist_collection_resume,
            on_phase_progress=(
                _persist_projection_phase
                if context.emit_phase_progress_events
                or context.enable_phase_projection_checkpoints
                else None
            ),
        )
        _record_trace_1cell(
            execute_deps=context.execute_deps,
            state=context.aspf_trace_state,
            kind="analysis_call_end",
            source_label="analysis:engine",
            target_label="analysis:result",
            representative="analyze_paths.end",
            basis_path=("analysis", "call", "end"),
        )
        context.profiling_stage_ns["server.analysis_call"] += (
            time.monotonic_ns() - analysis_started_ns
        )
    else:
        _record_trace_1cell(
            execute_deps=context.execute_deps,
            state=context.aspf_trace_state,
            kind="analysis_call_skipped",
            source_label="runtime:inputs",
            target_label="analysis:result",
            representative="analyze_paths.skipped",
            basis_path=("analysis", "call", "skipped"),
        )
        analysis = AnalysisResult(
            groups_by_path={},
            param_spans_by_path={},
            bundle_sites_by_path={},
            type_suggestions=[],
            type_ambiguities=[],
            type_callsite_evidence=[],
            constant_smells=[],
            unused_arg_smells=[],
            forest=context.forest,
        )
    return _AnalysisExecutionOutcome(
        analysis=analysis,
        last_collection_resume_payload=state.last_collection_resume_payload,
        semantic_progress_cumulative=state.semantic_progress_cumulative,
        latest_collection_progress=dict(state.latest_collection_progress),
    )


def _write_json_output_or_response(
    *,
    response: dict[str, object],
    target: object,
    response_key: str,
    payload: object,
) -> None:
    payload_json = json.dumps(payload, indent=2, sort_keys=False)
    if _is_stdout_target(target):
        response[response_key] = payload
    else:
        Path(str(target)).write_text(payload_json)


@dataclass(frozen=True)
class _NotificationRuntime:
    send_notification_fn: object
    emit_phase_progress_events: bool


def _notification_runtime(send_notification: object) -> _NotificationRuntime:
    if send_notification is None:
        return _NotificationRuntime(
            send_notification_fn=lambda _method, _params: None,
            emit_phase_progress_events=False,
        )
    if callable(send_notification):
        return _NotificationRuntime(
            send_notification_fn=send_notification,
            emit_phase_progress_events=True,
        )
    never(
        "invalid send_notification callback",
        callback_type=type(send_notification).__name__,
    )
    return _NotificationRuntime(
        send_notification_fn=lambda _method, _params: None,
        emit_phase_progress_events=False,
    )


def _create_progress_emitter(
    *,
    notification_runtime: _NotificationRuntime,
    phase_timeline_markdown_path: Path,
    phase_timeline_jsonl_path: Path,
    progress_heartbeat_seconds: float,
    profiling_stage_ns: dict[str, int],
    profiling_counters: dict[str, int],
) -> _ProgressEmitter:
    progress_event_seq = 0
    progress_state_lock = threading.Lock()
    last_progress_template: dict[str, object] | None = None
    last_progress_change_ns = 0
    last_progress_notification_ns = 0
    phase_timeline_header_emitted = False
    heartbeat_stop_event = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    send_notification = notification_runtime.send_notification_fn
    emit_phase_progress_events = notification_runtime.emit_phase_progress_events

    def emit_lsp_progress(
        *,
        phase: Literal["collection", "forest", "edge", "post"],
        collection_progress: Mapping[str, JSONValue],
        semantic_progress: Mapping[str, JSONValue] | None,
        work_done: int | None = None,
        work_total: int | None = None,
        include_timing: bool = False,
        done: bool = False,
        analysis_state: str | None = None,
        classification: str | None = None,
        phase_progress_v2: Mapping[str, JSONValue] | None = None,
        progress_marker: str | None = None,
        event_kind: Literal["progress", "heartbeat", "terminal", "checkpoint"] = "progress",
        stale_for_s: float | None = None,
        record_for_heartbeat: bool = True,
        update_notification_clock: bool = True,
    ) -> None:
        nonlocal progress_event_seq
        nonlocal last_progress_template
        nonlocal last_progress_change_ns
        nonlocal last_progress_notification_ns
        nonlocal phase_timeline_header_emitted
        semantic_payload: JSONObject = {}
        if isinstance(semantic_progress, Mapping):
            for raw_key, raw_value in semantic_progress.items():
                if not isinstance(raw_key, str):
                    continue
                if raw_key == "substantive_progress" or raw_key.startswith(
                    "cumulative_"
                ):
                    semantic_payload[raw_key] = raw_value
        if "substantive_progress" not in semantic_payload:
            semantic_payload["substantive_progress"] = False
        progress_value: JSONObject = {
            "format_version": 1,
            "schema": "gabion/dataflow_progress_v1",
            "phase": phase,
            "completed_files": int(collection_progress.get("completed_files", 0)),
            "in_progress_files": int(collection_progress.get("in_progress_files", 0)),
            "remaining_files": int(collection_progress.get("remaining_files", 0)),
            "total_files": int(collection_progress.get("total_files", 0)),
            "semantic_deltas": semantic_payload,
        }
        normalized_phase_progress_v2, primary_done, primary_total = _build_phase_progress_v2(
            phase=phase,
            collection_progress=collection_progress,
            semantic_progress=semantic_progress,
            work_done=work_done,
            work_total=work_total,
            phase_progress_v2=phase_progress_v2,
        )
        progress_value["phase_progress_v2"] = normalized_phase_progress_v2
        progress_value["work_done"] = primary_done
        progress_value["work_total"] = primary_total
        progress_value["telemetry_semantics_version"] = 2
        progress_value["event_kind"] = event_kind
        progress_value["ts_utc"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z")
        if isinstance(progress_marker, str) and progress_marker:
            progress_value["progress_marker"] = progress_marker
        if isinstance(stale_for_s, (int, float)):
            progress_value["stale_for_s"] = max(float(stale_for_s), 0.0)
        if include_timing:
            progress_value["profiling_v1"] = {
                "format_version": 1,
                "server": {
                    "stage_ns": dict(profiling_stage_ns),
                    "counters": dict(profiling_counters),
                },
            }
        if done:
            progress_value["done"] = True
        if isinstance(analysis_state, str) and analysis_state:
            progress_value["analysis_state"] = analysis_state
        if isinstance(classification, str) and classification:
            progress_value["classification"] = classification
        now_ns = time.monotonic_ns()
        with progress_state_lock:
            progress_event_seq += 1
            progress_value["event_seq"] = progress_event_seq
            phase_timeline_header, phase_timeline_row = _append_phase_timeline_event(
                markdown_path=phase_timeline_markdown_path,
                jsonl_path=phase_timeline_jsonl_path,
                progress_value=progress_value,
            )
            if not phase_timeline_header_emitted:
                progress_value["phase_timeline_header"] = (
                    phase_timeline_header
                    if isinstance(phase_timeline_header, str) and phase_timeline_header
                    else _phase_timeline_header_block()
                )
                phase_timeline_header_emitted = True
            progress_value["phase_timeline_row"] = phase_timeline_row
            ordered_progress_value = boundary_order.canonicalize_boundary_mapping(
                progress_value,
                source=(
                    "server._emit_lsp_progress."
                    f"{phase}.{event_kind}.progress_value"
                ),
            )
            send_notification(
                _LSP_PROGRESS_NOTIFICATION_METHOD,
                {
                    "token": _LSP_PROGRESS_TOKEN,
                    "value": ordered_progress_value,
                },
            )
            if update_notification_clock:
                last_progress_notification_ns = now_ns
            if done:
                last_progress_template = None
            elif record_for_heartbeat and event_kind != "heartbeat":
                last_progress_template = {
                    "phase": phase,
                    "collection_progress": {
                        str(key): collection_progress[key] for key in collection_progress
                    },
                    "semantic_progress": (
                        {str(key): semantic_progress[key] for key in semantic_progress}
                        if isinstance(semantic_progress, Mapping)
                        else None
                    ),
                    "work_done": primary_done,
                    "work_total": primary_total,
                    "include_timing": include_timing,
                    "analysis_state": analysis_state,
                    "classification": classification,
                    "phase_progress_v2": {
                        str(key): normalized_phase_progress_v2[key]
                        for key in normalized_phase_progress_v2
                    },
                    "progress_marker": progress_marker,
                }
                last_progress_change_ns = now_ns

    def _progress_heartbeat_loop() -> None:
        heartbeat_interval_ns = int(progress_heartbeat_seconds * 1_000_000_000)
        deadline_flush_ns = int(_PROGRESS_DEADLINE_FLUSH_SECONDS * 1_000_000_000)
        deadline_flush_margin_ns = int(
            _PROGRESS_DEADLINE_FLUSH_MARGIN_SECONDS * 1_000_000_000
        )
        deadline_emit_ns = max(1, deadline_flush_ns - deadline_flush_margin_ns)
        watchdog_interval_ns = int(
            _PROGRESS_DEADLINE_WATCHDOG_SECONDS * 1_000_000_000
        )
        last_watchdog_flush_ns = 0
        last_heartbeat_emit_ns = 0
        while not heartbeat_stop_event.wait(_PROGRESS_HEARTBEAT_POLL_SECONDS):
            with progress_state_lock:
                template = (
                    dict(last_progress_template)
                    if isinstance(last_progress_template, dict)
                    else {}
                )
                notification_ns = int(last_progress_notification_ns)
                change_ns = int(last_progress_change_ns)
            now_ns = time.monotonic_ns()
            progress_ready = bool(template) & (notification_ns > 0) & (change_ns > 0)
            stale_notification_ns = now_ns - notification_ns if progress_ready else 0
            stale_seconds = (
                max(0.0, (now_ns - change_ns) / 1_000_000_000.0)
                if progress_ready
                else 0.0
            )
            heartbeat_due = bool(
                progress_ready
                & (heartbeat_interval_ns > 0)
                & (stale_notification_ns >= heartbeat_interval_ns)
                & (now_ns - last_heartbeat_emit_ns >= heartbeat_interval_ns)
            )
            deadline_due = bool(
                progress_ready
                & (stale_notification_ns >= deadline_emit_ns)
                & (now_ns - last_watchdog_flush_ns >= deadline_emit_ns)
            )
            watchdog_due = bool(
                progress_ready
                & (stale_notification_ns >= watchdog_interval_ns)
                & (now_ns - last_watchdog_flush_ns >= watchdog_interval_ns)
            )
            if deadline_due or watchdog_due:
                emit_lsp_progress(
                    phase=cast(
                        Literal["collection", "forest", "edge", "post"],
                        str(template.get("phase", "collection")),
                    ),
                    collection_progress=cast(
                        Mapping[str, JSONValue],
                        template.get("collection_progress", {}),
                    ),
                    semantic_progress=cast(
                        Mapping[str, JSONValue] | None,
                        template.get("semantic_progress"),
                    ),
                    work_done=cast(int | None, template.get("work_done")),
                    work_total=cast(int | None, template.get("work_total")),
                    include_timing=bool(template.get("include_timing", False)),
                    done=False,
                    analysis_state=cast(str | None, template.get("analysis_state")),
                    classification=cast(str | None, template.get("classification")),
                    phase_progress_v2=cast(
                        Mapping[str, JSONValue] | None,
                        template.get("phase_progress_v2"),
                    ),
                    progress_marker=cast(str | None, template.get("progress_marker")),
                    event_kind="progress",
                    stale_for_s=stale_seconds,
                    record_for_heartbeat=False,
                    update_notification_clock=False,
                )
                last_watchdog_flush_ns = now_ns
            if heartbeat_due:
                emit_lsp_progress(
                    phase=cast(
                        Literal["collection", "forest", "edge", "post"],
                        str(template.get("phase", "collection")),
                    ),
                    collection_progress=cast(
                        Mapping[str, JSONValue],
                        template.get("collection_progress", {}),
                    ),
                    semantic_progress=cast(
                        Mapping[str, JSONValue] | None,
                        template.get("semantic_progress"),
                    ),
                    work_done=cast(int | None, template.get("work_done")),
                    work_total=cast(int | None, template.get("work_total")),
                    include_timing=bool(template.get("include_timing", False)),
                    done=False,
                    analysis_state=cast(str | None, template.get("analysis_state")),
                    classification=cast(str | None, template.get("classification")),
                    phase_progress_v2=cast(
                        Mapping[str, JSONValue] | None,
                        template.get("phase_progress_v2"),
                    ),
                    progress_marker=cast(str | None, template.get("progress_marker")),
                    event_kind="heartbeat",
                    stale_for_s=stale_seconds,
                    record_for_heartbeat=False,
                )
                last_heartbeat_emit_ns = now_ns

    if emit_phase_progress_events and progress_heartbeat_seconds > 0:
        heartbeat_thread = threading.Thread(
            target=_progress_heartbeat_loop,
            name="gabion-progress-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()

    def stop() -> None:
        heartbeat_stop_event.set()
        if isinstance(heartbeat_thread, threading.Thread):
            heartbeat_thread.join(timeout=1.5)

    return _ProgressEmitter(
        emit=emit_lsp_progress,
        stop=stop,
        emit_phase_progress_events=emit_phase_progress_events,
    )


def _emit_primary_outputs(
    *,
    response: dict[str, object],
    context: _PrimaryOutputContext,
) -> _PrimaryOutputArtifacts:
    synthesis_plan: JSONObject | None = None
    if (
        context.synthesis_plan_path
        or context.synthesis_report
        or context.synthesis_protocols_path
    ):
        try:
            synthesis_plan = build_synthesis_plan(
                context.analysis.groups_by_path,
                project_root=Path(context.root),
                max_tier=context.synthesis_max_tier,
                min_bundle_size=context.synthesis_min_bundle_size,
                allow_singletons=context.synthesis_allow_singletons,
                merge_overlap_threshold=context.payload.get("merge_overlap_threshold", None),
                config=context.config,
            )
        except (TypeError, ValueError, OSError) as exc:
            response.setdefault("synthesis_errors", []).append(str(exc))
        if synthesis_plan is not None:
            if context.synthesis_plan_path:
                _write_json_output_or_response(
                    response=response,
                    target=context.synthesis_plan_path,
                    response_key="synthesis_plan",
                    payload=synthesis_plan,
                )
            if context.synthesis_report:
                response["synthesis_plan"] = synthesis_plan
    if context.synthesis_protocols_path and synthesis_plan is not None:
        output = render_protocol_stubs(
            synthesis_plan,
            kind=context.synthesis_protocols_kind,
        )
        if _is_stdout_target(context.synthesis_protocols_path):
            response["synthesis_protocols"] = output
        else:
            Path(str(context.synthesis_protocols_path)).write_text(output)
    refactor_plan_payload: JSONObject | None = None
    if context.refactor_plan or context.refactor_plan_json:
        refactor_plan_payload = build_refactor_plan(
            context.analysis.groups_by_path,
            context.paths,
            config=context.config,
        )
        if context.refactor_plan_json:
            _write_json_output_or_response(
                response=response,
                target=context.refactor_plan_json,
                response_key="refactor_plan",
                payload=refactor_plan_payload,
            )
        if context.refactor_plan:
            response["refactor_plan"] = refactor_plan_payload
    if context.decision_snapshot_path is not None:
        payload_value = render_decision_snapshot(
            surfaces=DecisionSnapshotSurfaces(
                decision_surfaces=context.analysis.decision_surfaces,
                value_decision_surfaces=context.analysis.value_decision_surfaces,
            ),
            forest=context.analysis.forest,
            project_root=Path(context.root),
            groups_by_path=context.analysis.groups_by_path,
        )
        _write_json_output_or_response(
            response=response,
            target=context.decision_snapshot_path,
            response_key="decision_snapshot",
            payload=payload_value,
        )
    if context.structure_tree_path is not None:
        payload_value = render_structure_snapshot(
            context.analysis.groups_by_path,
            forest=context.analysis.forest,
            project_root=Path(context.root),
            invariant_propositions=context.analysis.invariant_propositions,
        )
        _write_json_output_or_response(
            response=response,
            target=context.structure_tree_path,
            response_key="structure_tree",
            payload=payload_value,
        )
    structure_metrics_payload: JSONObject | None = None
    if context.structure_metrics_path is not None:
        structure_metrics_payload = compute_structure_metrics(
            context.analysis.groups_by_path,
            forest=context.analysis.forest,
        )
        _write_json_output_or_response(
            response=response,
            target=context.structure_metrics_path,
            response_key="structure_metrics",
            payload=structure_metrics_payload,
        )
    return _PrimaryOutputArtifacts(
        synthesis_plan=synthesis_plan,
        refactor_plan_payload=refactor_plan_payload,
        structure_metrics_payload=structure_metrics_payload,
    )


def _emit_fingerprint_artifact_outputs(
    *,
    response: dict[str, object],
    analysis: AnalysisResult,
    fingerprint_synth_json: object,
    fingerprint_provenance_json: object,
    fingerprint_deadness_json: object,
    fingerprint_coherence_json: object,
    fingerprint_rewrite_plans_json: object,
    fingerprint_exception_obligations_json: object,
    fingerprint_handledness_json: object,
) -> None:
    if fingerprint_synth_json and analysis.fingerprint_synth_registry:
        _write_json_output_or_response(
            response=response,
            target=fingerprint_synth_json,
            response_key="fingerprint_synth_registry",
            payload=analysis.fingerprint_synth_registry,
        )
    if fingerprint_provenance_json and analysis.fingerprint_provenance:
        _write_json_output_or_response(
            response=response,
            target=fingerprint_provenance_json,
            response_key="fingerprint_provenance",
            payload=analysis.fingerprint_provenance,
        )
    if fingerprint_deadness_json is not None:
        _write_json_output_or_response(
            response=response,
            target=fingerprint_deadness_json,
            response_key="fingerprint_deadness",
            payload=analysis.deadness_witnesses,
        )
    if fingerprint_coherence_json is not None:
        _write_json_output_or_response(
            response=response,
            target=fingerprint_coherence_json,
            response_key="fingerprint_coherence",
            payload=analysis.coherence_witnesses,
        )
    if fingerprint_rewrite_plans_json is not None:
        _write_json_output_or_response(
            response=response,
            target=fingerprint_rewrite_plans_json,
            response_key="fingerprint_rewrite_plans",
            payload=analysis.rewrite_plans,
        )
    if fingerprint_exception_obligations_json is not None:
        _write_json_output_or_response(
            response=response,
            target=fingerprint_exception_obligations_json,
            response_key="fingerprint_exception_obligations",
            payload=analysis.exception_obligations,
        )
    if fingerprint_handledness_json is not None:
        _write_json_output_or_response(
            response=response,
            target=fingerprint_handledness_json,
            response_key="fingerprint_handledness",
            payload=analysis.handledness_witnesses,
        )


def _finalize_report_and_violations(
    *,
    context: _ReportFinalizationContext,
    phase_checkpoint_state: JSONObject,
) -> _ReportFinalizationOutcome:
    report = None
    violations: list[str] = []
    effective_violations: list[str] | None = None
    if context.report_path:
        report_carrier = ReportCarrier.from_analysis_result(context.analysis)
        report_markdown, _ = render_report(
            context.analysis.groups_by_path,
            context.max_components,
            report=report_carrier,
        )
        resolved_sections_for_obligations = extract_report_sections(report_markdown)
        pending_projection_reasons: dict[str, str] = {}
        for row in context.projection_rows:
            check_deadline()
            section_id = str(row.get("section_id", "") or "")
            if not section_id or section_id in resolved_sections_for_obligations:
                continue
            pending_projection_reasons[section_id] = "missing_dep"
        success_progress_payload: JSONObject = {
            "classification": "succeeded",
            "resume_supported": context.analysis_resume_reused_files > 0,
        }
        runtime_obligations = _incremental_progress_obligations(
            analysis_state="succeeded",
            progress_payload=success_progress_payload,
            resume_payload_available=context.analysis_resume_reused_files > 0,
            partial_report_written=False,
            report_requested=bool(context.report_path),
            projection_rows=context.projection_rows,
            sections=resolved_sections_for_obligations,
            pending_reasons=pending_projection_reasons,
        )
        (
            report_carrier.resumability_obligations,
            report_carrier.incremental_report_obligations,
        ) = _split_incremental_obligations(runtime_obligations)
        report_markdown, violations = render_report(
            context.analysis.groups_by_path,
            context.max_components,
            report=report_carrier,
        )
        report = report_markdown
        if context.baseline_path is not None:
            baseline_entries = load_baseline(context.baseline_path)
            if context.baseline_write:
                write_baseline(context.baseline_path, violations)
                effective_violations = []
            else:
                effective_violations, _ = apply_baseline(violations, baseline_entries)
        if context.report_output_path and context.projection_rows:
            resolved_sections = extract_report_sections(report_markdown)
            _write_report_section_journal(
                path=context.report_section_journal_path,
                witness_digest=context.report_section_witness_digest,
                projection_rows=context.projection_rows,
                sections=resolved_sections,
            )
            phase_checkpoint_state["post"] = {
                "status": "final",
                "work_done": 1,
                "work_total": 1,
                "section_ids": sort_once(resolved_sections, source = 'src/gabion/server.py:5395'),
                "resolved_sections": len(resolved_sections),
            }
        if context.decision_snapshot_path:
            decision_payload = render_decision_snapshot(
                surfaces=DecisionSnapshotSurfaces(
                    decision_surfaces=context.analysis.decision_surfaces,
                    value_decision_surfaces=context.analysis.value_decision_surfaces,
                ),
                forest=context.analysis.forest,
                project_root=Path(context.root),
                groups_by_path=context.analysis.groups_by_path,
            )
            report = report + "\n" + json.dumps(
                decision_payload, indent=2, sort_keys=False
            )
        if context.structure_tree_path:
            structure_payload = render_structure_snapshot(
                context.analysis.groups_by_path,
                forest=context.analysis.forest,
                project_root=Path(context.root),
                invariant_propositions=context.analysis.invariant_propositions,
            )
            report = report + "\n" + json.dumps(
                structure_payload, indent=2, sort_keys=False
            )
        if (
            context.structure_metrics_path
            and context.structure_metrics_payload is not None
        ):
            report = report + "\n" + json.dumps(
                context.structure_metrics_payload, indent=2, sort_keys=False
            )
        if context.synthesis_plan and (
            context.synthesis_report
            or context.synthesis_plan_path
            or context.synthesis_protocols_path
        ):
            report = report + render_synthesis_section(context.synthesis_plan)
        if context.refactor_plan and (
            context.refactor_plan or context.refactor_plan_json
        ):
            if context.refactor_plan_payload is not None:
                report = report + render_refactor_plan(context.refactor_plan_payload)
        if context.report_output_path is not None:
            context.report_output_path.parent.mkdir(parents=True, exist_ok=True)
            _write_text_profiled(
                context.report_output_path,
                report,
                io_name="report_markdown.write",
            )
    else:
        violation_carrier = ReportCarrier(
            forest=context.analysis.forest,
            type_suggestions=(
                context.analysis.type_suggestions
                if context.type_audit_report
                else []
            ),
            type_ambiguities=(
                context.analysis.type_ambiguities
                if context.type_audit_report
                else []
            ),
            decision_warnings=context.analysis.decision_warnings,
            fingerprint_warnings=context.analysis.fingerprint_warnings,
            parse_failure_witnesses=context.analysis.parse_failure_witnesses,
        )
        violations = compute_violations(
            context.analysis.groups_by_path,
            context.max_components,
            report=violation_carrier,
        )
        if context.baseline_path is not None:
            baseline_entries = load_baseline(context.baseline_path)
            if context.baseline_write:
                write_baseline(context.baseline_path, violations)
                effective_violations = []
            else:
                effective_violations, _ = apply_baseline(violations, baseline_entries)
    if effective_violations is None:
        effective_violations = violations
    return _ReportFinalizationOutcome(
        report=report,
        violations=violations,
        effective_violations=effective_violations,
        phase_checkpoint_state=phase_checkpoint_state,
    )


@dataclass(frozen=True)
class _TimeoutReportOutcome:
    partial_report_written: bool
    resolved_sections: dict[str, list[str]]
    pending_reasons: dict[str, str]
    phase_checkpoint_state: JSONObject


def _initialize_timeout_payload(
    *,
    exc: TimeoutExceeded,
    context: _TimeoutCleanupContext,
    mark_cleanup_timeout_fn: Callable[[str], None],
) -> tuple[JSONObject, JSONObject]:
    try:
        timeout_payload = _timeout_context_payload(exc)
    except TimeoutExceeded:
        mark_cleanup_timeout_fn("timeout_context_payload")
        timeout_payload = {
            "summary": "Analysis timed out.",
            "progress": {"classification": "timed_out_no_progress"},
        }
    progress_payload = dict(cast(Mapping[str, object], timeout_payload.get("progress", {})))
    progress_payload["classification"] = str(
        progress_payload.get("classification", "timed_out_no_progress")
        or "timed_out_no_progress"
    )
    timeout_payload["progress"] = progress_payload
    progress_payload.setdefault(
        "timeout_budget",
        {
            "total_timeout_ns": context.timeout_total_ns,
            "analysis_window_ns": context.analysis_window_ns,
            "cleanup_grace_ns": context.cleanup_grace_ns,
            "hard_deadline_ns": context.timeout_hard_deadline_ns,
        },
    )
    progress_payload["resume_supported"] = bool(
        progress_payload.get("resume_supported", False)
    )
    return timeout_payload, progress_payload


def _copy_json_mapping(payload: Mapping[str, object]) -> JSONObject:
    return {str(key): payload[key] for key in payload}


def _emit_trace_artifacts_payloads(
    *,
    response: dict[str, object],
    trace_artifacts: object | None,
) -> None:
    if trace_artifacts is None:
        return
    response["aspf_trace"] = _copy_json_mapping(trace_artifacts.trace_payload)
    response["aspf_equivalence"] = _copy_json_mapping(trace_artifacts.equivalence_payload)
    response["aspf_opportunities"] = _copy_json_mapping(trace_artifacts.opportunities_payload)
    response["aspf_delta_ledger"] = _copy_json_mapping(trace_artifacts.delta_ledger_payload)
    response["aspf_state"] = _copy_json_mapping(
        trace_artifacts.state_payload
        if trace_artifacts.state_payload is not None
        else {}
    )


def _persist_timeout_resume_state(
    *,
    context: _TimeoutCleanupContext,
    timeout_collection_resume_payload: JSONObject | None,
    mark_cleanup_timeout_fn: Callable[[str], None],
    emit_lsp_progress_fn: object,
) -> JSONObject | None:
    _ = (mark_cleanup_timeout_fn, emit_lsp_progress_fn)
    if isinstance(context.last_collection_resume_payload, Mapping):
        return {
            str(key): context.last_collection_resume_payload[key]
            for key in context.last_collection_resume_payload
        }
    return timeout_collection_resume_payload


def _load_timeout_resume_progress(
    *,
    context: _TimeoutCleanupContext,
    progress_payload: JSONObject,
    timeout_collection_resume_payload: JSONObject | None,
    mark_cleanup_timeout_fn: Callable[[str], None],
) -> JSONObject | None:
    _ = mark_cleanup_timeout_fn
    collection_resume: JSONObject | None = timeout_collection_resume_payload
    if collection_resume is None and isinstance(
        context.last_collection_resume_payload, Mapping
    ):
        collection_resume = {
            str(key): context.last_collection_resume_payload[key]
            for key in context.last_collection_resume_payload
        }
    if collection_resume is None:
        return timeout_collection_resume_payload
    timeout_collection_resume_payload = collection_resume
    resume_progress = _analysis_resume_progress(
        collection_resume=collection_resume,
        total_files=context.analysis_resume_total_files,
    )
    progress_payload["completed_files"] = resume_progress["completed_files"]
    progress_payload["in_progress_files"] = resume_progress["in_progress_files"]
    progress_payload["remaining_files"] = resume_progress["remaining_files"]
    progress_payload["total_files"] = resume_progress["total_files"]
    resume_supported = (
        resume_progress["completed_files"] > 0
        or resume_progress.get("in_progress_files", 0) > 0
    )
    progress_payload["resume_supported"] = resume_supported
    semantic_progress = dict(
        cast(Mapping[str, object], collection_resume.get("semantic_progress", {}))
    )
    progress_payload["semantic_progress"] = {
        str(key): semantic_progress[key] for key in semantic_progress
    }
    raw_semantic_substantive = semantic_progress.get("substantive_progress")
    semantic_substantive_progress = {
        True: True,
        False: False,
    }.get(raw_semantic_substantive)
    resume_token: JSONObject = {
        "phase": "analysis_collection",
        "carrier_refs": {
            "collection_resume": True,
        },
        **resume_progress,
    }
    resume_payload: JSONObject = {"resume_token": resume_token}
    resume_payload["input_witness"] = context.analysis_resume_input_witness
    progress_payload["resume"] = resume_payload
    classification = str(progress_payload.get("classification", "") or "")
    progress_payload["classification"] = (
        "timed_out_progress_resume"
        if (
            resume_supported
            and classification == "timed_out_no_progress"
            and (
                semantic_substantive_progress is None
                or semantic_substantive_progress
            )
        )
        else classification
    )
    return timeout_collection_resume_payload


def _derive_timeout_analysis_state(*, progress_payload: JSONObject) -> str:
    analysis_state = str(
        progress_payload.get("classification", "timed_out_no_progress")
        or "timed_out_no_progress"
    )
    progress_payload["resume_supported"] = bool(
        progress_payload.get("resume_supported")
    ) or analysis_state == "timed_out_progress_resume"
    return analysis_state


def _render_timeout_partial_report(
    *,
    context: _TimeoutCleanupContext,
    analysis_state: str,
    progress_payload: JSONObject,
    timeout_collection_resume_payload: JSONObject | None,
    phase_checkpoint_state: JSONObject,
    mark_cleanup_timeout_fn: Callable[[str], None],
) -> _TimeoutReportOutcome:
    partial_report_written = False
    resolved_sections: dict[str, list[str]] = {}
    pending_reasons: dict[str, str] = {}
    if context.report_output_path is not None and context.projection_rows:
        try:
            phase_checkpoint_state = phase_checkpoint_state or {}
            ensure_report_sections_cache = context.ensure_report_sections_cache_fn
            if callable(ensure_report_sections_cache):
                resolved_sections, journal_reason = ensure_report_sections_cache()
            else:
                resolved_sections, journal_reason = ({}, None)
            resolved_sections.setdefault(
                "components",
                _collection_components_preview_lines(
                    collection_resume=timeout_collection_resume_payload or {},
                ),
            )
            if (
                context.enable_phase_projection_checkpoints
                and timeout_collection_resume_payload is not None
            ):
                preview_groups_by_path = _groups_by_path_from_collection_resume(
                    timeout_collection_resume_payload
                )
                preview_report = ReportCarrier(
                    forest=context.forest,
                    parse_failure_witnesses=[],
                )
                preview_sections = context.execute_deps.project_report_sections_fn(
                    preview_groups_by_path,
                    preview_report,
                    max_phase="post",
                    include_previews=True,
                    preview_only=True,
                )
                for section_id, section_lines in preview_sections.items():
                    check_deadline()
                    resolved_sections.setdefault(section_id, section_lines)
            intro_lines = (
                _collection_progress_intro_lines(
                    collection_resume=timeout_collection_resume_payload,
                    total_files=context.analysis_resume_total_files,
                    resume_state_intro=context.analysis_resume_intro_payload,
                )
                if timeout_collection_resume_payload is not None
                else [
                    "Collection bootstrap checkpoint (provisional).",
                    f"- `root`: `{context.runtime_root}`",
                    f"- `paths_requested`: `{context.initial_paths_count_value}`",
                ]
            )
            resolved_sections.setdefault("intro", intro_lines)
            partial_report, pending_reasons = _render_incremental_report(
                analysis_state=analysis_state,
                progress_payload=progress_payload,
                projection_rows=context.projection_rows,
                sections=resolved_sections,
            )
            _apply_journal_pending_reason(
                projection_rows=context.projection_rows,
                sections=resolved_sections,
                pending_reasons=pending_reasons,
                journal_reason=journal_reason,
            )
            context.report_output_path.parent.mkdir(parents=True, exist_ok=True)
            _write_text_profiled(
                context.report_output_path,
                partial_report,
                io_name="report_markdown.write",
            )
            _write_report_section_journal(
                path=context.report_section_journal_path,
                witness_digest=context.report_section_witness_digest,
                projection_rows=context.projection_rows,
                sections=resolved_sections,
                pending_reasons=pending_reasons,
            )
            phase_checkpoint_state["timeout"] = {
                "status": "timed_out",
                "analysis_state": analysis_state,
                "section_ids": sort_once(resolved_sections, source = 'src/gabion/server.py:5848'),
                "resolved_sections": len(resolved_sections),
                "completed_phase": _latest_report_phase(phase_checkpoint_state),
            }
            partial_report_written = True
        except TimeoutExceeded:
            mark_cleanup_timeout_fn("render_timeout_report")
    return _TimeoutReportOutcome(
        partial_report_written=partial_report_written,
        resolved_sections=resolved_sections,
        pending_reasons=pending_reasons,
        phase_checkpoint_state=phase_checkpoint_state,
    )


def _handle_timeout_cleanup(
    *,
    exc: TimeoutExceeded,
    context: _TimeoutCleanupContext,
) -> dict:
    cleanup_now_ns = time.monotonic_ns()
    cleanup_remaining_ns = max(0, context.timeout_hard_deadline_ns - cleanup_now_ns)
    cleanup_window_ns = min(context.cleanup_grace_ns, cleanup_remaining_ns)
    cleanup_deadline_token = set_deadline(
        Deadline(deadline_ns=cleanup_now_ns + max(1, cleanup_window_ns))
    )
    cleanup_timeout_steps: list[str] = []
    phase_checkpoint_state = context.phase_checkpoint_state
    ensure_report_sections_cache = context.ensure_report_sections_cache_fn
    emit_lsp_progress = context.emit_lsp_progress_fn

    def _mark_cleanup_timeout(step: str) -> None:
        cleanup_timeout_steps.append(step)

    try:
        timeout_payload, progress_payload = _initialize_timeout_payload(
            exc=exc,
            context=context,
            mark_cleanup_timeout_fn=_mark_cleanup_timeout,
        )
        timeout_collection_resume_payload: JSONObject | None = None
        timeout_collection_resume_payload = _persist_timeout_resume_state(
            context=context,
            timeout_collection_resume_payload=timeout_collection_resume_payload,
            mark_cleanup_timeout_fn=_mark_cleanup_timeout,
            emit_lsp_progress_fn=emit_lsp_progress,
        )
        timeout_collection_resume_payload = _load_timeout_resume_progress(
            context=context,
            progress_payload=progress_payload,
            timeout_collection_resume_payload=timeout_collection_resume_payload,
            mark_cleanup_timeout_fn=_mark_cleanup_timeout,
        )
        analysis_state = _derive_timeout_analysis_state(
            progress_payload=progress_payload
        )
        timeout_report_outcome = _render_timeout_partial_report(
            context=context,
            analysis_state=analysis_state,
            progress_payload=progress_payload,
            timeout_collection_resume_payload=timeout_collection_resume_payload,
            phase_checkpoint_state=phase_checkpoint_state,
            mark_cleanup_timeout_fn=_mark_cleanup_timeout,
        )
        partial_report_written = timeout_report_outcome.partial_report_written
        resolved_sections = timeout_report_outcome.resolved_sections
        pending_reasons = timeout_report_outcome.pending_reasons
        phase_checkpoint_state = timeout_report_outcome.phase_checkpoint_state
        try:
            obligations = _incremental_progress_obligations(
                analysis_state=analysis_state,
                progress_payload=progress_payload,
                resume_payload_available=timeout_collection_resume_payload is not None,
                partial_report_written=partial_report_written,
                report_requested=context.report_output_path is not None,
                projection_rows=context.projection_rows,
                sections=resolved_sections,
                pending_reasons=pending_reasons,
            )
        except TimeoutExceeded:
            _mark_cleanup_timeout("incremental_obligations")
            obligations = []
        progress_payload["incremental_obligations"] = obligations
        if cleanup_timeout_steps:
            progress_payload["cleanup_truncated"] = True
            progress_payload["cleanup_timeout_steps"] = cleanup_timeout_steps
        timeout_classification = progress_payload.get("classification")
        emit_lsp_progress(
            phase="post",
            collection_progress=context.latest_collection_progress,
            semantic_progress=context.semantic_progress_cumulative,
            include_timing=True,
            done=True,
            analysis_state=analysis_state,
            classification=(
                timeout_classification
                if isinstance(timeout_classification, str)
                else "timed_out_no_progress"
            ),
            event_kind="terminal",
        )
        _record_trace_1cell(
            execute_deps=context.execute_deps,
            state=context.aspf_trace_state,
            kind="timeout_cleanup",
            source_label="analysis:engine",
            target_label="runtime:timeout_context",
            representative="analysis.timeout.cleanup",
            basis_path=("timeout", "cleanup"),
            surface="violation_summary",
        )
        timeout_response: dict[str, object] = {
            "exit_code": 2,
            "timeout": True,
            "analysis_state": analysis_state,
            "execution_plan": context.execution_plan.as_json_dict(),
            "timeout_context": timeout_payload,
            "selected_adapter": context.dataflow_capabilities.selected_adapter,
            "supported_analysis_surfaces": list(
                context.dataflow_capabilities.supported_analysis_surfaces
            ),
            "disabled_surface_reasons": dict(
                context.dataflow_capabilities.disabled_surface_reasons
            ),
        }
        trace_artifacts = context.execute_deps.finalize_trace_fn(
            state=context.aspf_trace_state,
            root=context.runtime_root,
            semantic_surface_payloads={
                "groups_by_path": {},
                "decision_surfaces": [],
                "rewrite_plans": [],
                "synthesis_plan": [],
                "delta_state": progress_payload,
                "delta_payload": timeout_payload,
                "violation_summary": {"timeout": True, "analysis_state": analysis_state},
                "_resume_collection": (
                    timeout_collection_resume_payload
                    if isinstance(timeout_collection_resume_payload, Mapping)
                    else {}
                ),
                "_latest_collection_progress": context.latest_collection_progress,
                "_semantic_progress": (
                    context.semantic_progress_cumulative
                    if isinstance(context.semantic_progress_cumulative, Mapping)
                    else {}
                ),
                "_analysis_manifest_digest": context.analysis_resume_input_manifest_digest,
                "_resume_source": context.analysis_resume_source,
                "_resume_compatibility_status": (
                    context.analysis_resume_state_compatibility_status
                ),
            },
            exit_code=2,
            analysis_state=analysis_state,
        )
        _emit_trace_artifacts_payloads(
            response=timeout_response,
            trace_artifacts=trace_artifacts,
        )
        return _normalize_dataflow_response(timeout_response)
    finally:
        reset_deadline(cleanup_deadline_token)


@dataclass(frozen=True)
class _ExecutionPayloadOptions:
    emit_phase_timeline: bool
    progress_heartbeat_seconds: float
    dot_path: object
    fail_on_violations: object
    no_recursive: object
    max_components: object
    type_audit: object
    type_audit_report: object
    type_audit_max: object
    fail_on_type_ambiguities: object
    lint: bool
    allow_external: object
    strictness: str
    baseline_path: Path | None
    baseline_write: bool
    synthesis_plan_path: object
    synthesis_report: object
    structure_tree_path: object
    structure_metrics_path: object
    decision_snapshot_path: object
    obsolescence_mode: _AuxiliaryMode
    annotation_drift_mode: _AuxiliaryMode
    ambiguity_mode: _AuxiliaryMode
    emit_test_evidence_suggestions: bool
    emit_call_clusters: bool
    emit_call_cluster_consolidation: bool
    emit_semantic_coverage_map: bool
    semantic_coverage_mapping_path: object
    synthesis_max_tier: object
    synthesis_min_bundle_size: object
    synthesis_allow_singletons: object
    synthesis_protocols_path: object
    synthesis_protocols_kind: object
    refactor_plan: object
    refactor_plan_json: object
    fingerprint_synth_json: object
    fingerprint_provenance_json: object
    fingerprint_deadness_json: object
    fingerprint_coherence_json: object
    fingerprint_rewrite_plans_json: object
    fingerprint_exception_obligations_json: object
    fingerprint_handledness_json: object
    include_wl_refinement: bool
    aspf_trace_json: object
    aspf_import_trace: object
    aspf_equivalence_against: object
    aspf_opportunities_json: object
    aspf_state_json: object
    aspf_import_state: list[str] | None
    aspf_delta_jsonl: object
    aspf_semantic_surface: object


    @property
    def emit_test_obsolescence(self) -> bool:
        return self.obsolescence_mode.emit_report

    @property
    def emit_test_obsolescence_state(self) -> bool:
        return self.obsolescence_mode.emit_state

    @property
    def test_obsolescence_state_path(self) -> object:
        return self.obsolescence_mode.state_path

    @property
    def emit_test_obsolescence_delta(self) -> bool:
        return self.obsolescence_mode.emit_delta

    @property
    def write_test_obsolescence_baseline(self) -> bool:
        return self.obsolescence_mode.write_baseline

    @property
    def obsolescence_baseline_path_override(self) -> Path | None:
        return self.obsolescence_mode.baseline_path_override

    @property
    def emit_test_annotation_drift(self) -> bool:
        return self.annotation_drift_mode.kind in {"report", "state"}

    @property
    def test_annotation_drift_state_path(self) -> object:
        return self.annotation_drift_mode.state_path

    @property
    def emit_test_annotation_drift_delta(self) -> bool:
        return self.annotation_drift_mode.emit_delta

    @property
    def write_test_annotation_drift_baseline(self) -> bool:
        return self.annotation_drift_mode.write_baseline

    @property
    def annotation_drift_baseline_path_override(self) -> Path | None:
        return self.annotation_drift_mode.baseline_path_override

    @property
    def emit_ambiguity_delta(self) -> bool:
        return self.ambiguity_mode.emit_delta

    @property
    def emit_ambiguity_state(self) -> bool:
        return self.ambiguity_mode.emit_state

    @property
    def ambiguity_state_path(self) -> object:
        return self.ambiguity_mode.state_path

    @property
    def write_ambiguity_baseline(self) -> bool:
        return self.ambiguity_mode.write_baseline

    @property
    def ambiguity_baseline_path_override(self) -> Path | None:
        return self.ambiguity_mode.baseline_path_override


@dataclass(frozen=True)
class _AnalysisInclusionFlags:
    type_audit: bool
    include_decisions: bool
    include_rewrite_plans: bool
    include_exception_obligations: bool
    include_handledness_witnesses: bool
    include_never_invariants: bool
    include_wl_refinement: bool
    include_ambiguities: bool
    include_coherence: bool
    needs_analysis: bool


def _select_auxiliary_mode_selection(
    *,
    payload: dict[str, object],
    aux_operation: _AuxOperationIngressCarrier | None,
) -> _AuxiliaryModeSelection:
    obsolescence_mode = _auxiliary_mode_from_payload(
        payload=payload,
        mode_key="obsolescence_mode",
        state_key="test_obsolescence_state",
        emit_state_key="emit_test_obsolescence_state",
        emit_delta_key="emit_test_obsolescence_delta",
        write_baseline_key="write_test_obsolescence_baseline",
        emit_report_key="emit_test_obsolescence",
        domain="obsolescence",
        allow_report=True,
    )
    annotation_drift_mode = _auxiliary_mode_from_payload(
        payload=payload,
        mode_key="annotation_drift_mode",
        state_key="test_annotation_drift_state",
        emit_state_key="emit_test_annotation_drift_state",
        emit_delta_key="emit_test_annotation_drift_delta",
        write_baseline_key="write_test_annotation_drift_baseline",
        emit_report_key="emit_test_annotation_drift",
        domain="annotation-drift",
        allow_report=True,
    )
    ambiguity_mode = _auxiliary_mode_from_payload(
        payload=payload,
        mode_key="ambiguity_mode",
        state_key="ambiguity_state",
        emit_state_key="emit_ambiguity_state",
        emit_delta_key="emit_ambiguity_delta",
        write_baseline_key="write_ambiguity_baseline",
        emit_report_key=None,
        domain="ambiguity",
        allow_report=False,
    )
    if aux_operation is not None:
        aux_domain = aux_operation.domain
        aux_action = aux_operation.action
        allowed_actions = {
            "obsolescence": {"report", "state", "delta", "baseline-write"},
            "annotation-drift": {"report", "state", "delta", "baseline-write"},
            "ambiguity": {"state", "delta", "baseline-write"},
        }
        if aux_action not in allowed_actions.get(aux_domain, set()):
            never("invalid aux operation action", domain=aux_domain, action=aux_action)
        aux_mode = _AuxiliaryMode(
            domain=aux_domain,
            kind=aux_action,
            state_path=aux_operation.state_in,
            baseline_path_override=aux_operation.baseline_path,
        )
        if aux_domain == "obsolescence":
            obsolescence_mode = aux_mode
            annotation_drift_mode = _AuxiliaryMode(domain="annotation-drift", kind="off", state_path=None)
            ambiguity_mode = _AuxiliaryMode(domain="ambiguity", kind="off", state_path=None)
        elif aux_domain == "annotation-drift":
            annotation_drift_mode = aux_mode
            obsolescence_mode = _AuxiliaryMode(domain="obsolescence", kind="off", state_path=None)
            ambiguity_mode = _AuxiliaryMode(domain="ambiguity", kind="off", state_path=None)
        elif aux_domain == "ambiguity":
            ambiguity_mode = aux_mode
            obsolescence_mode = _AuxiliaryMode(domain="obsolescence", kind="off", state_path=None)
            annotation_drift_mode = _AuxiliaryMode(domain="annotation-drift", kind="off", state_path=None)
        else:
            never("invalid aux operation domain", domain=aux_domain, action=aux_action)
    return _AuxiliaryModeSelection(
        obsolescence=obsolescence_mode,
        annotation_drift=annotation_drift_mode,
        ambiguity=ambiguity_mode,
    )


def _parse_execution_payload_options(
    *,
    payload: dict[str, object],
    root: Path,
    aux_operation: _AuxOperationIngressCarrier | None = None,
) -> _ExecutionPayloadOptions:
    strictness = str(payload.get("strictness", "high"))
    if strictness not in {"high", "low"}:
        never("invalid strictness", strictness=str(strictness))
    baseline_path = resolve_baseline_path(payload.get("baseline"), root)
    aux_mode_selection = _select_auxiliary_mode_selection(
        payload=payload,
        aux_operation=aux_operation,
    )
    return _ExecutionPayloadOptions(
        emit_phase_timeline=False,
        progress_heartbeat_seconds=_progress_heartbeat_seconds(payload),
        dot_path=payload.get("dot"),
        fail_on_violations=payload.get("fail_on_violations", False),
        no_recursive=payload.get("no_recursive", False),
        max_components=payload.get("max_components", 10),
        type_audit=payload.get("type_audit", False),
        type_audit_report=payload.get("type_audit_report", False),
        type_audit_max=payload.get("type_audit_max", 50),
        fail_on_type_ambiguities=payload.get("fail_on_type_ambiguities", False),
        lint=bool(payload.get("lint", False)),
        allow_external=payload.get("allow_external", False),
        strictness=strictness,
        baseline_path=baseline_path,
        baseline_write=bool(payload.get("baseline_write", False))
        and baseline_path is not None,
        synthesis_plan_path=payload.get("synthesis_plan"),
        synthesis_report=payload.get("synthesis_report", False),
        structure_tree_path=payload.get("structure_tree"),
        structure_metrics_path=payload.get("structure_metrics"),
        decision_snapshot_path=payload.get("decision_snapshot"),
        obsolescence_mode=aux_mode_selection.obsolescence,
        emit_test_evidence_suggestions=bool(
            payload.get("emit_test_evidence_suggestions", False)
        ),
        emit_call_clusters=bool(payload.get("emit_call_clusters", False)),
        emit_call_cluster_consolidation=bool(
            payload.get("emit_call_cluster_consolidation", False)
        ),
        annotation_drift_mode=aux_mode_selection.annotation_drift,
        ambiguity_mode=aux_mode_selection.ambiguity,
        emit_semantic_coverage_map=bool(payload.get("emit_semantic_coverage_map", False)),
        semantic_coverage_mapping_path=payload.get("semantic_coverage_mapping"),
        synthesis_max_tier=payload.get("synthesis_max_tier", 2),
        synthesis_min_bundle_size=payload.get("synthesis_min_bundle_size", 2),
        synthesis_allow_singletons=payload.get("synthesis_allow_singletons", False),
        synthesis_protocols_path=payload.get("synthesis_protocols"),
        synthesis_protocols_kind=payload.get("synthesis_protocols_kind", "dataclass"),
        refactor_plan=payload.get("refactor_plan", False),
        refactor_plan_json=payload.get("refactor_plan_json"),
        fingerprint_synth_json=payload.get("fingerprint_synth_json"),
        fingerprint_provenance_json=payload.get("fingerprint_provenance_json"),
        fingerprint_deadness_json=payload.get("fingerprint_deadness_json"),
        fingerprint_coherence_json=payload.get("fingerprint_coherence_json"),
        fingerprint_rewrite_plans_json=payload.get("fingerprint_rewrite_plans_json"),
        fingerprint_exception_obligations_json=payload.get(
            "fingerprint_exception_obligations_json"
        ),
        fingerprint_handledness_json=payload.get("fingerprint_handledness_json"),
        include_wl_refinement=_truthy_flag(payload.get("include_wl_refinement")),
        aspf_trace_json=payload.get("aspf_trace_json"),
        aspf_import_trace=payload.get("aspf_import_trace"),
        aspf_equivalence_against=payload.get("aspf_equivalence_against"),
        aspf_opportunities_json=payload.get("aspf_opportunities_json"),
        aspf_state_json=payload.get("aspf_state_json"),
        aspf_import_state=cast(list[str] | None, payload.get("aspf_import_state")),
        aspf_delta_jsonl=payload.get("aspf_delta_jsonl"),
        aspf_semantic_surface=payload.get("aspf_semantic_surface"),
    )


def _compute_analysis_inclusion_flags(
    *,
    options: _ExecutionPayloadOptions,
    report_path: object,
    decision_tiers: dict[str, int],
) -> _AnalysisInclusionFlags:
    type_audit = bool(options.type_audit)
    if options.fail_on_type_ambiguities:
        type_audit = True
    include_decisions = bool(report_path) or bool(options.decision_snapshot_path) or bool(
        options.fail_on_violations
    )
    if decision_tiers:
        include_decisions = True
    include_rewrite_plans = bool(report_path) or bool(options.fingerprint_rewrite_plans_json)
    include_exception_obligations = bool(report_path) or bool(
        options.fingerprint_exception_obligations_json
    )
    include_handledness_witnesses = bool(report_path) or bool(
        options.fingerprint_handledness_json
    )
    include_never_invariants = bool(report_path)
    include_ambiguities = bool(report_path) or options.lint or options.emit_ambiguity_state
    if (options.emit_ambiguity_delta or options.write_ambiguity_baseline) and not options.ambiguity_state_path:
        include_ambiguities = True
    include_coherence = (
        bool(report_path)
        or bool(options.fingerprint_coherence_json)
        or include_rewrite_plans
    )
    needs_analysis = (
        bool(report_path)
        or bool(options.dot_path)
        or bool(options.structure_tree_path)
        or bool(options.structure_metrics_path)
        or bool(options.decision_snapshot_path)
        or bool(options.synthesis_plan_path)
        or bool(options.synthesis_report)
        or bool(options.synthesis_protocols_path)
        or bool(options.refactor_plan)
        or bool(options.refactor_plan_json)
        or bool(options.fingerprint_synth_json)
        or bool(options.fingerprint_provenance_json)
        or bool(options.fingerprint_deadness_json)
        or bool(options.fingerprint_coherence_json)
        or bool(options.fingerprint_rewrite_plans_json)
        or bool(options.fingerprint_exception_obligations_json)
        or bool(options.fingerprint_handledness_json)
        or bool(type_audit)
        or bool(options.type_audit_report)
        or bool(options.fail_on_type_ambiguities)
        or bool(options.fail_on_violations)
        or options.baseline_path is not None
        or bool(options.lint)
        or bool(options.emit_test_evidence_suggestions)
        or bool(include_ambiguities)
        or bool(options.aspf_trace_json)
        or bool(options.aspf_import_trace)
        or bool(options.aspf_equivalence_against)
        or bool(options.aspf_opportunities_json)
        or bool(options.aspf_state_json)
        or bool(options.aspf_import_state)
        or bool(options.aspf_delta_jsonl)
    )
    return _AnalysisInclusionFlags(
        type_audit=type_audit,
        include_decisions=include_decisions,
        include_rewrite_plans=include_rewrite_plans,
        include_exception_obligations=include_exception_obligations,
        include_handledness_witnesses=include_handledness_witnesses,
        include_never_invariants=include_never_invariants,
        include_wl_refinement=options.include_wl_refinement,
        include_ambiguities=include_ambiguities,
        include_coherence=include_coherence,
        needs_analysis=needs_analysis,
    )


@dataclass(frozen=True)
class _SuccessResponseContext:
    execute_deps: CommandEffects
    aspf_trace_state: object | None
    analysis: AnalysisResult
    root: str
    paths: list[Path]
    payload: dict[str, object]
    config: AuditConfig
    options: _ExecutionPayloadOptions
    name_filter_bundle: DataflowNameFilterBundle
    report_path: object
    report_output_path: Path | None
    report_section_journal_path: Path
    report_section_witness_digest: str | None
    report_phase_checkpoint_path: Path | None
    projection_rows: list[JSONObject]
    analysis_resume_state_path: Path | None
    analysis_resume_source: str
    analysis_resume_state_status: str | None
    analysis_resume_state_compatibility_status: str | None
    analysis_resume_manifest_digest: str | None
    analysis_resume_reused_files: int
    analysis_resume_total_files: int
    profiling_stage_ns: dict[str, int]
    profiling_counters: dict[str, int]
    phase_checkpoint_state: JSONObject
    execution_plan: ExecutionPlan
    last_collection_resume_payload: JSONObject | None
    semantic_progress_cumulative: JSONObject | None
    latest_collection_progress: JSONObject
    emit_lsp_progress_fn: Callable[..., None]
    dataflow_capabilities: _DataflowCapabilityAnnotations


@dataclass(frozen=True)
class _SuccessResponseOutcome:
    response: dict
    phase_checkpoint_state: JSONObject


def _build_success_response(
    *,
    context: _SuccessResponseContext,
) -> _SuccessResponseOutcome:
    analysis = context.analysis
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
    if (
        context.analysis_resume_state_path is not None
        or context.analysis_resume_source != "cold_start"
        or context.analysis_resume_reused_files > 0
    ):
        cache_verdict = _analysis_resume_cache_verdict(
            status=context.analysis_resume_state_status,
            reused_files=context.analysis_resume_reused_files,
            compatibility_status=context.analysis_resume_state_compatibility_status,
        )
        response["analysis_resume"] = {
            "checkpoint_path": (
                str(context.analysis_resume_state_path)
                if context.analysis_resume_state_path is not None
                else None
            ),
            "source": context.analysis_resume_source,
            "manifest_digest": context.analysis_resume_manifest_digest,
            "reused_files": context.analysis_resume_reused_files,
            "total_files": context.analysis_resume_total_files,
            "remaining_files": max(
                context.analysis_resume_total_files
                - context.analysis_resume_reused_files,
                0,
            ),
            "status": context.analysis_resume_state_status,
            "compatibility_status": context.analysis_resume_state_compatibility_status,
            "cache_verdict": cache_verdict,
        }
    profiling_v1: JSONObject = {
        "format_version": 1,
        "server": {
            "stage_ns": context.profiling_stage_ns,
            "counters": context.profiling_counters,
        },
    }
    if isinstance(analysis.profiling_v1, Mapping):
        profiling_v1["analysis"] = {
            str(key): analysis.profiling_v1[key] for key in analysis.profiling_v1
        }
    response["profiling_v1"] = profiling_v1
    if context.options.lint:
        response["lint_lines"] = analysis.lint_lines

    primary_outputs = _emit_primary_outputs(
        response=response,
        context=_PrimaryOutputContext(
            analysis=analysis,
            root=context.root,
            paths=context.paths,
            payload=context.payload,
            config=context.config,
            synthesis_plan_path=context.options.synthesis_plan_path,
            synthesis_report=context.options.synthesis_report,
            synthesis_protocols_path=context.options.synthesis_protocols_path,
            synthesis_protocols_kind=context.options.synthesis_protocols_kind,
            synthesis_max_tier=context.options.synthesis_max_tier,
            synthesis_min_bundle_size=context.options.synthesis_min_bundle_size,
            synthesis_allow_singletons=context.options.synthesis_allow_singletons,
            refactor_plan=context.options.refactor_plan,
            refactor_plan_json=context.options.refactor_plan_json,
            decision_snapshot_path=context.options.decision_snapshot_path,
            structure_tree_path=context.options.structure_tree_path,
            structure_metrics_path=context.options.structure_metrics_path,
        ),
    )
    synthesis_plan = primary_outputs.synthesis_plan
    plan_payload = primary_outputs.refactor_plan_payload
    metrics = primary_outputs.structure_metrics_payload
    if synthesis_plan is not None:
        _record_trace_1cell(
            execute_deps=context.execute_deps,
            state=context.aspf_trace_state,
            kind="artifact_emit",
            source_label="analysis:result",
            target_label="artifact:synthesis_plan",
            representative="emit:synthesis_plan",
            basis_path=("artifact", "emit", "synthesis_plan"),
            surface="synthesis_plan",
        )
    if plan_payload is not None:
        _record_trace_1cell(
            execute_deps=context.execute_deps,
            state=context.aspf_trace_state,
            kind="artifact_emit",
            source_label="analysis:result",
            target_label="artifact:refactor_plan",
            representative="emit:refactor_plan",
            basis_path=("artifact", "emit", "refactor_plan"),
            surface="rewrite_plans",
        )

    _apply_auxiliary_artifact_outputs(
        response=response,
        analysis=analysis,
        root=context.root,
        paths=context.paths,
        config=context.config,
        name_filter_bundle=context.name_filter_bundle,
        emit_test_obsolescence=context.options.emit_test_obsolescence,
        emit_test_obsolescence_state=context.options.emit_test_obsolescence_state,
        test_obsolescence_state_path=context.options.test_obsolescence_state_path,
        emit_test_obsolescence_delta=context.options.emit_test_obsolescence_delta,
        write_test_obsolescence_baseline=context.options.write_test_obsolescence_baseline,
        emit_test_evidence_suggestions=context.options.emit_test_evidence_suggestions,
        emit_call_clusters=context.options.emit_call_clusters,
        emit_call_cluster_consolidation=context.options.emit_call_cluster_consolidation,
        emit_test_annotation_drift=context.options.emit_test_annotation_drift,
        emit_semantic_coverage_map=context.options.emit_semantic_coverage_map,
        test_annotation_drift_state_path=context.options.test_annotation_drift_state_path,
        semantic_coverage_mapping_path=context.options.semantic_coverage_mapping_path,
        emit_test_annotation_drift_delta=context.options.emit_test_annotation_drift_delta,
        write_test_annotation_drift_baseline=(
            context.options.write_test_annotation_drift_baseline
        ),
        emit_ambiguity_delta=context.options.emit_ambiguity_delta,
        emit_ambiguity_state=context.options.emit_ambiguity_state,
        ambiguity_state_path=context.options.ambiguity_state_path,
        write_ambiguity_baseline=context.options.write_ambiguity_baseline,
        obsolescence_baseline_path=context.options.obsolescence_baseline_path_override,
        annotation_drift_baseline_path=(
            context.options.annotation_drift_baseline_path_override
        ),
        ambiguity_baseline_path=context.options.ambiguity_baseline_path_override,
    )
    for artifact_key, representative in (
        ("test_obsolescence_delta_summary", "emit:test_obsolescence_delta"),
        ("test_annotation_drift_delta_summary", "emit:test_annotation_drift_delta"),
        ("ambiguity_delta_summary", "emit:ambiguity_delta"),
        ("test_obsolescence_summary", "emit:test_obsolescence_state"),
        ("test_annotation_drift_summary", "emit:test_annotation_drift_state"),
        ("ambiguity_state_summary", "emit:ambiguity_state"),
    ):
        if artifact_key not in response:
            continue
        _record_trace_1cell(
            execute_deps=context.execute_deps,
            state=context.aspf_trace_state,
            kind="artifact_emit",
            source_label="analysis:result",
            target_label=f"artifact:{artifact_key}",
            representative=representative,
            basis_path=("artifact", "emit", artifact_key),
            surface="delta_payload"
            if "delta" in artifact_key
            else "delta_state",
        )
    if context.aspf_trace_state is not None:
        materialized_one_cells: list[JSONObject] = []
        for cell, metadata in zip_longest(
            context.aspf_trace_state.one_cells,
            context.aspf_trace_state.one_cell_metadata,
            fillvalue={},
        ):
            payload = cell.as_dict()
            payload["kind"] = str(metadata.get("kind", ""))
            payload["surface"] = str(metadata.get("surface", ""))
            materialized_one_cells.append(payload)
        analysis.aspf_one_cells = materialized_one_cells
        analysis.aspf_two_cell_witnesses = [
            witness.as_dict() for witness in context.aspf_trace_state.two_cell_witnesses
        ]
        analysis.aspf_cofibration_witnesses = [
            carrier.as_dict() for carrier in context.aspf_trace_state.cofibrations
        ]
        analysis.aspf_surface_representatives = {
            str(key): str(context.aspf_trace_state.surface_representatives[key])
            for key in context.aspf_trace_state.surface_representatives
        }
    report_outcome = _finalize_report_and_violations(
        context=_ReportFinalizationContext(
            analysis=analysis,
            root=context.root,
            max_components=context.options.max_components,
            report_path=context.report_path,
            report_output_path=context.report_output_path,
            projection_rows=context.projection_rows,
            report_section_journal_path=context.report_section_journal_path,
            report_section_witness_digest=context.report_section_witness_digest,
            report_phase_checkpoint_path=context.report_phase_checkpoint_path,
            analysis_resume_state_path=context.analysis_resume_state_path,
            analysis_resume_reused_files=context.analysis_resume_reused_files,
            type_audit_report=context.options.type_audit_report,
            baseline_path=context.options.baseline_path,
            baseline_write=context.options.baseline_write,
            decision_snapshot_path=context.options.decision_snapshot_path,
            structure_tree_path=context.options.structure_tree_path,
            structure_metrics_path=context.options.structure_metrics_path,
            structure_metrics_payload=metrics,
            synthesis_plan=synthesis_plan,
            synthesis_plan_path=context.options.synthesis_plan_path,
            synthesis_report=context.options.synthesis_report,
            synthesis_protocols_path=context.options.synthesis_protocols_path,
            refactor_plan=context.options.refactor_plan,
            refactor_plan_json=context.options.refactor_plan_json,
            refactor_plan_payload=plan_payload,
        ),
        phase_checkpoint_state=context.phase_checkpoint_state,
    )
    report = report_outcome.report
    effective_violations = report_outcome.effective_violations
    phase_checkpoint_state = report_outcome.phase_checkpoint_state
    if context.options.dot_path and analysis.forest is not None:
        dot_payload = render_dot(analysis.forest)
        if _is_stdout_target(context.options.dot_path):
            response["dot"] = dot_payload
            report = (f"{report}\n" if report is not None else "") + dot_payload
        else:
            Path(context.options.dot_path).write_text(dot_payload)
    response["violations"] = len(effective_violations)
    _emit_fingerprint_artifact_outputs(
        response=response,
        analysis=analysis,
        fingerprint_synth_json=context.options.fingerprint_synth_json,
        fingerprint_provenance_json=context.options.fingerprint_provenance_json,
        fingerprint_deadness_json=context.options.fingerprint_deadness_json,
        fingerprint_coherence_json=context.options.fingerprint_coherence_json,
        fingerprint_rewrite_plans_json=context.options.fingerprint_rewrite_plans_json,
        fingerprint_exception_obligations_json=(
            context.options.fingerprint_exception_obligations_json
        ),
        fingerprint_handledness_json=context.options.fingerprint_handledness_json,
    )
    if context.options.baseline_path is not None:
        response["baseline_path"] = str(context.options.baseline_path)
        response["baseline_written"] = bool(context.options.baseline_write)
    if context.options.fail_on_type_ambiguities and analysis.type_ambiguities:
        response["exit_code"] = 1
    else:
        if context.options.baseline_write:
            response["exit_code"] = 0
        else:
            response["exit_code"] = (
                1 if (context.options.fail_on_violations and effective_violations) else 0
            )
    response["analysis_state"] = "succeeded"
    response["execution_plan"] = context.execution_plan.as_json_dict()
    response["selected_adapter"] = context.dataflow_capabilities.selected_adapter
    response["supported_analysis_surfaces"] = list(
        context.dataflow_capabilities.supported_analysis_surfaces
    )
    response["disabled_surface_reasons"] = dict(
        context.dataflow_capabilities.disabled_surface_reasons
    )
    trace_artifacts = context.execute_deps.finalize_trace_fn(
        state=context.aspf_trace_state,
        root=Path(context.root),
        semantic_surface_payloads={
            "groups_by_path": analysis.groups_by_path,
            "decision_surfaces": analysis.decision_surfaces,
            "rewrite_plans": analysis.rewrite_plans,
            "synthesis_plan": synthesis_plan if synthesis_plan is not None else [],
            "delta_state": {
                "test_obsolescence": response.get("test_obsolescence_summary"),
                "test_annotation_drift": response.get("test_annotation_drift_summary"),
                "ambiguity": response.get("ambiguity_state_summary"),
            },
            "delta_payload": {
                "test_obsolescence_delta": response.get("test_obsolescence_delta_summary"),
                "test_annotation_drift_delta": response.get(
                    "test_annotation_drift_delta_summary"
                ),
                "ambiguity_delta": response.get("ambiguity_delta_summary"),
            },
            "violation_summary": {
                "violations": len(effective_violations),
                "decision_warnings": analysis.decision_warnings,
                "errors": response.get("errors", []),
            },
            "_resume_collection": (
                context.last_collection_resume_payload
                if isinstance(context.last_collection_resume_payload, Mapping)
                else {}
            ),
            "_latest_collection_progress": context.latest_collection_progress,
            "_semantic_progress": (
                context.semantic_progress_cumulative
                if isinstance(context.semantic_progress_cumulative, Mapping)
                else {}
            ),
            "_analysis_manifest_digest": context.analysis_resume_manifest_digest,
            "_resume_source": context.analysis_resume_source,
            "_resume_compatibility_status": (
                context.analysis_resume_state_compatibility_status
            ),
        },
        exit_code=int(response.get("exit_code", 0) or 0),
        analysis_state=(
            str(response.get("analysis_state"))
            if response.get("analysis_state") is not None
            else None
        ),
    )
    _emit_trace_artifacts_payloads(
        response=response,
        trace_artifacts=trace_artifacts,
    )
    emit_lsp_progress = context.emit_lsp_progress_fn
    emit_lsp_progress(
        phase="post",
        collection_progress=context.latest_collection_progress,
        semantic_progress=context.semantic_progress_cumulative,
        include_timing=True,
        done=True,
        analysis_state="succeeded",
        classification="succeeded",
        event_kind="terminal",
    )
    return _SuccessResponseOutcome(
        response=_normalize_dataflow_response(response),
        phase_checkpoint_state=phase_checkpoint_state,
    )


def execute_command_total(
    ls: LanguageServer,
    payload: dict[str, object],
    *,
    deps: ExecuteCommandDeps | None = None,
) -> dict:
    _bind_server_symbols()
    execute_deps: CommandEffects = deps or _default_execute_command_deps()
    execution_plan = _materialize_execution_plan(payload)
    payload = dict(execution_plan.inputs)
    _reject_removed_legacy_payload_keys(payload)
    write_execution_plan_artifact(
        execution_plan,
        root=Path(str(payload.get("root") or ls.workspace.root_path or ".")),
    )
    profile_enabled = _truthy_flag(payload.get("deadline_profile"))
    profile_root_value = payload.get("root") or ls.workspace.root_path or "."
    initial_root = Path(str(profile_root_value))
    initial_report_path = payload.get("report")
    initial_report_path_text = (
        str(initial_report_path) if isinstance(initial_report_path, str) else None
    )
    timeout_total_ns, analysis_window_ns, cleanup_grace_ns = _analysis_timeout_budget_ns(
        payload
    )
    ingress = _normalize_command_payload_ingress(
        payload=payload,
        root=initial_root,
    )
    payload = ingress.payload
    dataflow_capabilities = ingress.dataflow_capabilities
    normalized_timeout_total_ticks = _normalize_duration_timeout_clock_ticks(
        timeout=ingress.timeout,
        total_ticks=_analysis_timeout_total_ticks(payload),
    )
    timeout_total_ticks = normalize_timeout_total_ticks(
        payload,
        default_ticks=normalized_timeout_total_ticks,
        never_fn=never,
    )
    runtime_input = CommandRuntimeInput(
        payload=payload,
        root=initial_root,
        report_path_text=initial_report_path_text,
        timeout_total_ticks=timeout_total_ticks,
    )
    timeout_start_ns = time.monotonic_ns()
    timeout_hard_deadline_ns = timeout_start_ns + timeout_total_ns
    analysis_deadline_ns = timeout_start_ns + analysis_window_ns
    deadline_token = set_deadline(Deadline(deadline_ns=analysis_deadline_ns))
    deadline_clock_token = set_deadline_clock(GasMeter(limit=timeout_total_ticks))
    profile_token = set_deadline_profile(
        project_root=runtime_input.root,
        enabled=profile_enabled,
        sample_interval=_deadline_profile_sample_interval(payload),
    )
    forest = Forest()
    forest_token = set_forest(forest)
    explicit_resume_state = False
    analysis_resume_state_path: Path | None = None
    analysis_resume_input_witness: JSONObject | None = None
    analysis_resume_input_manifest_digest: str | None = None
    analysis_resume_total_files = 0
    analysis_resume_reused_files = 0
    analysis_resume_state_status: str | None = None
    analysis_resume_state_compatibility_status: str | None = None
    analysis_resume_source = "cold_start"
    analysis_resume_intro_payload: JSONObject | None = None
    analysis_resume_intro_timeline_header: str | None = None
    analysis_resume_intro_timeline_row: str | None = None
    report_section_witness_digest: str | None = None
    report_output_path = _resolve_report_output_path(
        root=runtime_input.root,
        report_path=runtime_input.report_path_text,
    )
    report_section_journal_path = _resolve_report_section_journal_path(
        root=runtime_input.root,
        report_path=runtime_input.report_path_text,
    )
    report_phase_checkpoint_path: Path | None = None
    projection_rows: list[JSONObject] = (
        execute_deps.report_projection_spec_rows_fn() if report_output_path else []
    )
    enable_phase_projection_checkpoints = False
    phase_checkpoint_state: JSONObject = {}
    last_collection_resume_payload: JSONObject | None = None
    report_sections_cache: dict[str, list[str]] = {}
    report_sections_cache_reason: str | None = None
    report_sections_cache_loaded = False
    runtime_state = CommandRuntimeState(
        latest_collection_progress=initial_collection_progress(
            total_files=analysis_resume_total_files
        )
    )
    semantic_progress_cumulative: JSONObject | None = runtime_state.semantic_progress_cumulative
    latest_collection_progress: JSONObject = dict(runtime_state.latest_collection_progress)
    emit_phase_timeline = False
    phase_timeline_path = runtime_input.root / "_unused_phase_timeline.md"
    phase_timeline_markdown_path = _phase_timeline_md_path(root=runtime_input.root)
    phase_timeline_jsonl_path = _phase_timeline_jsonl_path(root=runtime_input.root)
    progress_heartbeat_seconds = _progress_heartbeat_seconds(payload)
    dot_path = payload.get("dot")
    aspf_trace_state: object | None = None
    progress_emitter: _ProgressEmitter | None = None
    emit_phase_progress_events = False

    def _emit_lsp_progress(**_kwargs: object) -> None:
        return

    def _ensure_report_sections_cache() -> tuple[dict[str, list[str]], str | None]:
        nonlocal report_sections_cache
        nonlocal report_sections_cache_reason
        nonlocal report_sections_cache_loaded
        if not report_sections_cache_loaded:
            report_sections_cache, report_sections_cache_reason = (
                execute_deps.load_report_section_journal_fn(
                path=report_section_journal_path,
                witness_digest=report_section_witness_digest,
                )
            )
            report_sections_cache_loaded = True
        return report_sections_cache, report_sections_cache_reason

    def _clear_report_sections_cache_reason() -> None:
        nonlocal report_sections_cache_reason
        report_sections_cache_reason = None

    raw_initial_paths = payload.get("paths")
    initial_paths_count_value = initial_paths_count(raw_initial_paths)
    execute_deps.write_bootstrap_incremental_artifacts_fn(
        report_output_path=report_output_path,
        report_section_journal_path=report_section_journal_path,
        report_phase_checkpoint_path=report_phase_checkpoint_path,
        witness_digest=report_section_witness_digest,
        root=runtime_input.root,
        paths_requested=initial_paths_count_value,
        projection_rows=projection_rows,
        phase_checkpoint_state=phase_checkpoint_state,
    )
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
        fingerprint_spec: dict[str, JSONValue] = {
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

        raw_paths = payload.get("paths")
        paths = normalize_paths(raw_paths, root=Path(str(root)))
        requested_language_raw = payload.get("language_id", payload.get("language"))
        requested_language = (
            str(requested_language_raw).strip().lower()
            if isinstance(requested_language_raw, str) and requested_language_raw.strip()
            else None
        )
        root = payload.get("root") or root
        report_path = payload.get("report")
        report_path_text = str(report_path) if isinstance(report_path, str) else None
        report_output_path = _resolve_report_output_path(
            root=Path(root),
            report_path=report_path_text,
        )
        report_section_journal_path = _resolve_report_section_journal_path(
            root=Path(root),
            report_path=report_path_text,
        )
        report_phase_checkpoint_path = None
        projection_rows = (
            execute_deps.report_projection_spec_rows_fn() if report_output_path else []
        )
        options = _parse_execution_payload_options(
            payload=payload,
            root=Path(root),
            aux_operation=ingress.aux_operation,
        )
        aspf_trace_state = execute_deps.start_trace_fn(
            root=Path(root),
            payload=payload,
        )
        _record_trace_1cell(
            execute_deps=execute_deps,
            state=aspf_trace_state,
            kind="command_start",
            source_label="runtime:command",
            target_label="analysis:entry",
            representative="gabion.dataflow.start",
            basis_path=("command", "start"),
        )
        enable_phase_projection_checkpoints = bool(report_output_path)
        emit_phase_timeline = False
        phase_timeline_path = Path(root) / "_unused_phase_timeline.md"
        phase_timeline_markdown_path = _phase_timeline_md_path(root=Path(root))
        phase_timeline_jsonl_path = _phase_timeline_jsonl_path(root=Path(root))
        progress_heartbeat_seconds = options.progress_heartbeat_seconds
        dot_path = options.dot_path
        fail_on_violations = options.fail_on_violations
        no_recursive = options.no_recursive
        max_components = options.max_components
        type_audit = options.type_audit
        type_audit_report = options.type_audit_report
        type_audit_max = options.type_audit_max
        fail_on_type_ambiguities = options.fail_on_type_ambiguities
        lint = options.lint
        name_filter_bundle = DataflowNameFilterBundle.from_payload(
            payload=payload,
            defaults=defaults,
            decision_section=decision_section,
        )
        allow_external = options.allow_external
        strictness = options.strictness
        baseline_path = options.baseline_path
        baseline_write = options.baseline_write
        synthesis_plan_path = options.synthesis_plan_path
        synthesis_report = options.synthesis_report
        structure_tree_path = options.structure_tree_path
        structure_metrics_path = options.structure_metrics_path
        decision_snapshot_path = options.decision_snapshot_path
        emit_test_obsolescence = options.emit_test_obsolescence
        emit_test_obsolescence_state = options.emit_test_obsolescence_state
        test_obsolescence_state_path = options.test_obsolescence_state_path
        emit_test_obsolescence_delta = options.emit_test_obsolescence_delta
        write_test_obsolescence_baseline = options.write_test_obsolescence_baseline
        emit_test_evidence_suggestions = options.emit_test_evidence_suggestions
        emit_call_clusters = options.emit_call_clusters
        emit_call_cluster_consolidation = options.emit_call_cluster_consolidation
        emit_test_annotation_drift = options.emit_test_annotation_drift
        emit_semantic_coverage_map = options.emit_semantic_coverage_map
        test_annotation_drift_state_path = options.test_annotation_drift_state_path
        semantic_coverage_mapping_path = options.semantic_coverage_mapping_path
        emit_test_annotation_drift_delta = options.emit_test_annotation_drift_delta
        write_test_annotation_drift_baseline = options.write_test_annotation_drift_baseline
        emit_ambiguity_delta = options.emit_ambiguity_delta
        emit_ambiguity_state = options.emit_ambiguity_state
        ambiguity_state_path = options.ambiguity_state_path
        write_ambiguity_baseline = options.write_ambiguity_baseline
        synthesis_max_tier = options.synthesis_max_tier
        synthesis_min_bundle_size = options.synthesis_min_bundle_size
        synthesis_allow_singletons = options.synthesis_allow_singletons
        synthesis_protocols_path = options.synthesis_protocols_path
        synthesis_protocols_kind = options.synthesis_protocols_kind
        refactor_plan = options.refactor_plan
        refactor_plan_json = options.refactor_plan_json
        fingerprint_synth_json = options.fingerprint_synth_json
        fingerprint_provenance_json = options.fingerprint_provenance_json
        fingerprint_deadness_json = options.fingerprint_deadness_json
        fingerprint_coherence_json = options.fingerprint_coherence_json
        fingerprint_rewrite_plans_json = options.fingerprint_rewrite_plans_json
        fingerprint_exception_obligations_json = (
            options.fingerprint_exception_obligations_json
        )
        fingerprint_handledness_json = options.fingerprint_handledness_json

        config = AuditConfig(
            project_root=Path(root),
            exclude_dirs=name_filter_bundle.exclude_dirs,
            ignore_params=name_filter_bundle.ignore_params,
            decision_ignore_params=name_filter_bundle.decision_ignore_params,
            external_filter=not allow_external,
            strictness=strictness,
            transparent_decorators=name_filter_bundle.transparent_decorators,
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
        ingest_adapter = resolve_adapter(
            paths=paths,
            language_id=requested_language,
        )
        normalized_ingest = ingest_adapter.normalize(paths, config=config)
        inclusion_flags = _compute_analysis_inclusion_flags(
            options=options,
            report_path=report_path,
            decision_tiers=decision_tiers,
        )
        type_audit = inclusion_flags.type_audit
        include_decisions = inclusion_flags.include_decisions
        include_rewrite_plans = inclusion_flags.include_rewrite_plans
        include_exception_obligations = (
            inclusion_flags.include_exception_obligations
        )
        include_handledness_witnesses = (
            inclusion_flags.include_handledness_witnesses
        )
        include_never_invariants = inclusion_flags.include_never_invariants
        include_wl_refinement = inclusion_flags.include_wl_refinement
        include_ambiguities = inclusion_flags.include_ambiguities
        include_coherence = inclusion_flags.include_coherence
        needs_analysis = inclusion_flags.needs_analysis
        analysis_resume_state = _AnalysisResumePreparationState(
            analysis_resume_state_path=analysis_resume_state_path,
            analysis_resume_input_witness=analysis_resume_input_witness,
            analysis_resume_input_manifest_digest=analysis_resume_input_manifest_digest,
            analysis_resume_total_files=analysis_resume_total_files,
            analysis_resume_reused_files=analysis_resume_reused_files,
            analysis_resume_state_status=analysis_resume_state_status,
            analysis_resume_state_compatibility_status=(
                analysis_resume_state_compatibility_status
            ),
            analysis_resume_intro_payload=analysis_resume_intro_payload,
            analysis_resume_intro_timeline_header=analysis_resume_intro_timeline_header,
            analysis_resume_intro_timeline_row=analysis_resume_intro_timeline_row,
            report_section_witness_digest=report_section_witness_digest,
            phase_checkpoint_state=phase_checkpoint_state,
            semantic_progress_cumulative=semantic_progress_cumulative,
            last_collection_resume_payload=last_collection_resume_payload,
            analysis_resume_source=analysis_resume_source,
        )
        file_paths_for_run, collection_resume_payload = _prepare_analysis_resume_state(
            execute_deps=execute_deps,
            aspf_trace_state=aspf_trace_state,
            needs_analysis=needs_analysis,
            normalized_ingest=normalized_ingest,
            root=str(root),
            payload=payload,
            aspf_import_state=options.aspf_import_state,
            no_recursive=bool(no_recursive),
            report_path=report_path,
            include_wl_refinement=include_wl_refinement,
            config=config,
            report_output_path=report_output_path,
            state=analysis_resume_state,
            runtime_state=runtime_state,
        )
        analysis_resume_state_path = analysis_resume_state.analysis_resume_state_path
        analysis_resume_input_witness = analysis_resume_state.analysis_resume_input_witness
        analysis_resume_input_manifest_digest = (
            analysis_resume_state.analysis_resume_input_manifest_digest
        )
        analysis_resume_total_files = analysis_resume_state.analysis_resume_total_files
        analysis_resume_reused_files = analysis_resume_state.analysis_resume_reused_files
        analysis_resume_state_status = analysis_resume_state.analysis_resume_state_status
        analysis_resume_state_compatibility_status = (
            analysis_resume_state.analysis_resume_state_compatibility_status
        )
        analysis_resume_source = analysis_resume_state.analysis_resume_source
        analysis_resume_intro_payload = analysis_resume_state.analysis_resume_intro_payload
        analysis_resume_intro_timeline_header = (
            analysis_resume_state.analysis_resume_intro_timeline_header
        )
        analysis_resume_intro_timeline_row = (
            analysis_resume_state.analysis_resume_intro_timeline_row
        )
        report_section_witness_digest = analysis_resume_state.report_section_witness_digest
        phase_checkpoint_state = analysis_resume_state.phase_checkpoint_state
        semantic_progress_cumulative = analysis_resume_state.semantic_progress_cumulative
        last_collection_resume_payload = analysis_resume_state.last_collection_resume_payload
        last_collection_intro_signature: tuple[int, int, int, int] | None = None
        last_collection_semantic_witness_digest: str | None = None
        last_collection_checkpoint_flush_ns = 0
        last_analysis_index_resume_signature: tuple[
            int, str, int, int, str, str
        ] | None = None
        last_collection_report_flush_ns = 0
        last_collection_report_flush_completed = -1
        phase_progress_signatures: dict[str, tuple[object, ...]] = {}
        phase_progress_last_flush_ns: dict[str, int] = {}
        profiling_stage_ns: dict[str, int] = {
            "server.analysis_call": 0,
            "server.projection_emit": 0,
        }
        profiling_counters: dict[str, int] = {
            "server.collection_resume_persist_calls": 0,
            "server.projection_emit_calls": 0,
        }
        progress_emitter = _create_progress_emitter(
            notification_runtime=_notification_runtime(
                getattr(ls, "send_notification", None)
            ),
            phase_timeline_markdown_path=phase_timeline_markdown_path,
            phase_timeline_jsonl_path=phase_timeline_jsonl_path,
            progress_heartbeat_seconds=progress_heartbeat_seconds,
            profiling_stage_ns=profiling_stage_ns,
            profiling_counters=profiling_counters,
        )
        _emit_lsp_progress = progress_emitter.emit
        emit_phase_progress_events = progress_emitter.emit_phase_progress_events

        analysis_execution_state = _AnalysisExecutionMutableState(
            last_collection_resume_payload=last_collection_resume_payload,
            semantic_progress_cumulative=semantic_progress_cumulative,
            latest_collection_progress=dict(latest_collection_progress),
        )
        try:
            analysis_outcome = _run_analysis_with_progress(
                context=_AnalysisExecutionContext(
                    execute_deps=execute_deps,
                    aspf_trace_state=aspf_trace_state,
                    runtime_state=runtime_state,
                    forest=forest,
                    paths=paths,
                    no_recursive=bool(no_recursive),
                    type_audit=bool(type_audit),
                    type_audit_report=bool(type_audit_report),
                    type_audit_max=int(type_audit_max),
                    report_path=report_path,
                    include_coherence=include_coherence,
                    include_rewrite_plans=include_rewrite_plans,
                    include_exception_obligations=include_exception_obligations,
                    include_handledness_witnesses=include_handledness_witnesses,
                    include_never_invariants=include_never_invariants,
                    include_wl_refinement=include_wl_refinement,
                    include_decisions=include_decisions,
                    lint=lint,
                    include_ambiguities=include_ambiguities,
                    config=config,
                    needs_analysis=needs_analysis,
                    file_paths_for_run=file_paths_for_run,
                    analysis_resume_intro_payload=analysis_resume_intro_payload,
                    analysis_resume_reused_files=analysis_resume_reused_files,
                    analysis_resume_total_files=analysis_resume_total_files,
                    analysis_resume_state_path=analysis_resume_state_path,
                    analysis_resume_state_status=analysis_resume_state_status,
                    analysis_resume_input_manifest_digest=analysis_resume_input_manifest_digest,
                    analysis_resume_input_witness=analysis_resume_input_witness,
                    analysis_resume_intro_timeline_header=analysis_resume_intro_timeline_header,
                    analysis_resume_intro_timeline_row=analysis_resume_intro_timeline_row,
                    phase_timeline_path=phase_timeline_path,
                    emit_phase_timeline=emit_phase_timeline,
                    enable_phase_projection_checkpoints=enable_phase_projection_checkpoints,
                    report_output_path=report_output_path,
                    projection_rows=projection_rows,
                    report_section_journal_path=report_section_journal_path,
                    report_section_witness_digest=report_section_witness_digest,
                    report_phase_checkpoint_path=report_phase_checkpoint_path,
                    phase_checkpoint_state=phase_checkpoint_state,
                    profile_enabled=profile_enabled,
                    emit_phase_progress_events=emit_phase_progress_events,
                    fingerprint_deadness_json=fingerprint_deadness_json,
                    emit_lsp_progress_fn=_emit_lsp_progress,
                    ensure_report_sections_cache_fn=_ensure_report_sections_cache,
                    clear_report_sections_cache_reason_fn=_clear_report_sections_cache_reason,
                    check_deadline_fn=check_deadline,
                    profiling_stage_ns=profiling_stage_ns,
                    profiling_counters=profiling_counters,
                ),
                state=analysis_execution_state,
                collection_resume_payload=collection_resume_payload,
            )
        except TimeoutExceeded:
            last_collection_resume_payload = (
                analysis_execution_state.last_collection_resume_payload
            )
            semantic_progress_cumulative = (
                analysis_execution_state.semantic_progress_cumulative
            )
            latest_collection_progress = dict(
                analysis_execution_state.latest_collection_progress
            )
            raise
        analysis = analysis_outcome.analysis
        last_collection_resume_payload = analysis_outcome.last_collection_resume_payload
        semantic_progress_cumulative = analysis_outcome.semantic_progress_cumulative
        latest_collection_progress = analysis_outcome.latest_collection_progress
        success_outcome = _build_success_response(
            context=_SuccessResponseContext(
                execute_deps=execute_deps,
                aspf_trace_state=aspf_trace_state,
                analysis=analysis,
                root=str(root),
                paths=paths,
                payload=payload,
                config=config,
                options=options,
                name_filter_bundle=name_filter_bundle,
                report_path=report_path,
                report_output_path=report_output_path,
                report_section_journal_path=report_section_journal_path,
                report_section_witness_digest=report_section_witness_digest,
                report_phase_checkpoint_path=report_phase_checkpoint_path,
                projection_rows=projection_rows,
                analysis_resume_state_path=analysis_resume_state_path,
                analysis_resume_source=analysis_resume_source,
                analysis_resume_state_status=analysis_resume_state_status,
                analysis_resume_state_compatibility_status=(
                    analysis_resume_state_compatibility_status
                ),
                analysis_resume_manifest_digest=analysis_resume_input_manifest_digest,
                analysis_resume_reused_files=analysis_resume_reused_files,
                analysis_resume_total_files=analysis_resume_total_files,
                profiling_stage_ns=profiling_stage_ns,
                profiling_counters=profiling_counters,
                phase_checkpoint_state=phase_checkpoint_state,
                execution_plan=execution_plan,
                last_collection_resume_payload=last_collection_resume_payload,
                semantic_progress_cumulative=semantic_progress_cumulative,
                latest_collection_progress=latest_collection_progress,
                emit_lsp_progress_fn=_emit_lsp_progress,
                dataflow_capabilities=dataflow_capabilities,
            )
        )
        phase_checkpoint_state = success_outcome.phase_checkpoint_state
        return success_outcome.response
    except TimeoutExceeded as exc:
        return _handle_timeout_cleanup(
            exc=exc,
            context=_TimeoutCleanupContext(
                timeout_hard_deadline_ns=timeout_hard_deadline_ns,
                cleanup_grace_ns=cleanup_grace_ns,
                timeout_total_ns=timeout_total_ns,
                analysis_window_ns=analysis_window_ns,
                analysis_resume_state_path=analysis_resume_state_path,
                analysis_resume_input_manifest_digest=analysis_resume_input_manifest_digest,
                last_collection_resume_payload=last_collection_resume_payload,
                execute_deps=execute_deps,
                analysis_resume_input_witness=analysis_resume_input_witness,
                emit_phase_timeline=emit_phase_timeline,
                phase_timeline_path=phase_timeline_path,
                analysis_resume_total_files=analysis_resume_total_files,
                analysis_resume_source=analysis_resume_source,
                analysis_resume_state_status=analysis_resume_state_status,
                analysis_resume_state_compatibility_status=(
                    analysis_resume_state_compatibility_status
                ),
                analysis_resume_reused_files=analysis_resume_reused_files,
                profile_enabled=profile_enabled,
                latest_collection_progress=latest_collection_progress,
                semantic_progress_cumulative=semantic_progress_cumulative,
                report_output_path=report_output_path,
                projection_rows=projection_rows,
                report_phase_checkpoint_path=report_phase_checkpoint_path,
                report_section_journal_path=report_section_journal_path,
                report_section_witness_digest=report_section_witness_digest,
                phase_checkpoint_state=phase_checkpoint_state,
                enable_phase_projection_checkpoints=enable_phase_projection_checkpoints,
                forest=forest,
                analysis_resume_intro_payload=analysis_resume_intro_payload,
                runtime_root=runtime_input.root,
                initial_paths_count_value=initial_paths_count_value,
                execution_plan=execution_plan,
                aspf_trace_state=aspf_trace_state,
                ensure_report_sections_cache_fn=_ensure_report_sections_cache,
                emit_lsp_progress_fn=_emit_lsp_progress,
                dataflow_capabilities=dataflow_capabilities,
            ),
        )
    except Exception:
        _emit_lsp_progress(
            phase="post",
            collection_progress=latest_collection_progress,
            semantic_progress=semantic_progress_cumulative,
            include_timing=True,
            done=True,
            analysis_state="failed",
            classification="failed",
            event_kind="terminal",
        )
        raise
    finally:
        if progress_emitter is not None:
            progress_emitter.stop()
        reset_forest(forest_token)
        reset_deadline_clock(deadline_clock_token)
        reset_deadline(deadline_token)
        reset_deadline_profile(profile_token)
