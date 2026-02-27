from __future__ import annotations
# gabion:decision_protocol_module
# gabion:boundary_normalization_module

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

_SURFACE_BUNDLE_INFERENCE = "bundle-inference"
_SURFACE_DECISION_SURFACES = "decision-surfaces"
_SURFACE_TYPE_FLOW = "type-flow"
_SURFACE_EXCEPTION_OBLIGATIONS = "exception-obligations"
_SURFACE_REWRITE_PLAN_SUPPORT = "rewrite-plan-support"


def _capability_enabled(adapter_contract: object, capability_name: str) -> bool:
    if type(adapter_contract) is not dict:
        return True
    capabilities = cast(dict[object, object], adapter_contract).get("capabilities")
    if type(capabilities) is not dict:
        return True
    value = cast(dict[object, object], capabilities).get(capability_name)
    return bool(value) if type(value) is bool else True


def _unsupported_surface_diagnostic(
    *,
    surface: str,
    capability_name: str,
    runtime_config: AuditConfig,
) -> JSONObject:
    required = surface in runtime_config.required_analysis_surfaces
    adapter_contract = runtime_config.adapter_contract
    adapter_name = "native"
    if type(adapter_contract) is dict:
        adapter_name = str(cast(dict[object, object], adapter_contract).get("name", "native") or "native")
    return {
        "kind": "unsupported_by_adapter",
        "surface": surface,
        "capability": capability_name,
        "adapter": adapter_name,
        "required_by_policy": required,
    }

_BOUND = False


def _bind_audit_symbols() -> None:
    global _BOUND
    if _BOUND:
        return
    from gabion.analysis import dataflow_audit as _audit

    module_globals = globals()
    for name, value in _audit.__dict__.items():
        module_globals.setdefault(name, value)
    _BOUND = True


@dataclass(frozen=True)
class _ForestPhaseResult:
    forest_spec: object
    ambiguity_witnesses: list[JSONObject]


@dataclass(frozen=True)
class _EdgePhaseResult:
    type_suggestions: list[str]
    type_ambiguities: list[str]
    type_callsite_evidence: list[str]
    constant_smells: list[str]
    deadness_witnesses: list[JSONObject]
    unused_arg_smells: list[str]


@dataclass(frozen=True)
class _PostPhaseResult:
    deadline_obligations: list[JSONObject]
    decision_surfaces: list[str]
    decision_warnings: list[str]
    value_decision_surfaces: list[str]
    value_decision_rewrites: list[str]
    fingerprint_warnings: list[str]
    fingerprint_matches: list[str]
    fingerprint_synth: list[str]
    fingerprint_synth_registry: object
    fingerprint_provenance: list[JSONObject]
    coherence_witnesses: list[JSONObject]
    rewrite_plans: list[JSONObject]
    exception_obligations: list[JSONObject]
    never_invariants: list[JSONObject]
    handledness_witnesses: list[JSONObject]
    context_suggestions: list[str]
    lint_lines: list[str]


def _normalized_dimension_payload(
    raw_dimensions: Mapping[str, object],
) -> dict[str, JSONObject]:
    normalized: dict[str, JSONObject] = {}
    for raw_name, raw_payload in raw_dimensions.items():
        check_deadline()
        match raw_name:
            case str() as dim_name:
                match raw_payload:
                    case Mapping() as dimension_payload:
                        raw_done = dimension_payload.get("done")
                        raw_total = dimension_payload.get("total")
                        match (raw_done, raw_total):
                            case (done_value, total_value) if (
                                type(done_value) is int and type(total_value) is int
                            ):
                                done = max(int(done_value), 0)
                                total = max(int(total_value), 0)
                                if total:
                                    done = min(done, total)
                                normalized[dim_name] = {"done": done, "total": total}
                            case _:
                                pass
                    case _:
                        pass
            case _:
                pass
    return normalized


def _apply_forest_progress_delta(
    progress_delta: object,
    *,
    forest_mutable_progress_done: int,
    forest_mutable_progress_total: int,
    forest_progress_marker: str,
    forest_dimensions: dict[str, JSONObject],
) -> tuple[int, int, str, dict[str, JSONObject], bool]:
    match progress_delta:
        case Mapping() as progress_delta_payload:
            raw_done = progress_delta_payload.get("primary_done")
            raw_total = progress_delta_payload.get("primary_total")
            if type(raw_done) is int:
                forest_mutable_progress_done = max(int(raw_done), 0)
            if type(raw_total) is int:
                forest_mutable_progress_total = max(int(raw_total), 0)
            forest_mutable_progress_total = max(
                forest_mutable_progress_total,
                forest_mutable_progress_done,
            )
            raw_marker = progress_delta_payload.get("marker")
            match raw_marker:
                case str() as marker_text if marker_text:
                    forest_progress_marker = marker_text
                case _:
                    pass
            raw_dimensions = progress_delta_payload.get("dimensions")
            match raw_dimensions:
                case Mapping() as dimensions_payload:
                    forest_dimensions = _normalized_dimension_payload(dimensions_payload)
                case _:
                    pass
            return (
                forest_mutable_progress_done,
                forest_mutable_progress_total,
                forest_progress_marker,
                forest_dimensions,
                True,
            )
        case _:
            return (
                forest_mutable_progress_done,
                forest_mutable_progress_total,
                forest_progress_marker,
                forest_dimensions,
                False,
            )


def _run_forest_phase(
    *,
    file_paths: list[Path],
    forest: Forest,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    bundle_sites_by_path: dict[Path, dict[str, list[set[str]]]],
    invariant_propositions: list[InvariantProposition],
    parse_failure_witnesses: list[JSONObject],
    analysis_index_for_progress: object,
    config: AuditConfig,
    include_bundle_forest: bool,
    include_decision_surfaces: bool,
    include_value_decision_surfaces: bool,
    include_lint_lines: bool,
    include_never_invariants: bool,
    include_wl_refinement: bool,
    include_deadline_obligations: bool,
    include_ambiguities: bool,
    require_analysis_index_fn: Callable[[], AnalysisIndex],
    emit_phase_progress_fn: Callable[..., None],
    deadline_check_fn: Callable[..., None],
    analysis_profile_stage_ns: dict[str, int],
) -> _ForestPhaseResult:
    check_deadline()
    if analysis_index_for_progress is None and (
        include_bundle_forest
        or include_decision_surfaces
        or include_value_decision_surfaces
        or include_lint_lines
        or include_never_invariants
        or include_wl_refinement
        or include_deadline_obligations
        or include_ambiguities
    ):
        analysis_index_for_progress = require_analysis_index_fn()
    forest_inventory_files_total = len(file_paths)
    forest_inventory_callsites_total = 0
    if analysis_index_for_progress is not None:
        forest_inventory_callsites_total = sum(
            len(info.calls) for info in analysis_index_for_progress.by_qual.values()
        )
    forest_mutable_progress_done = 0
    forest_mutable_progress_total = 0
    forest_ambiguity_total = 1 if include_ambiguities else 0
    forest_ambiguity_done = 0
    forest_progress_marker = "start"
    forest_dimensions: dict[str, JSONObject] = {}
    forest_spec: object = None
    ambiguity_witnesses: list[JSONObject] = []

    def _forest_phase_progress_v2() -> JSONObject:
        primary_done = forest_mutable_progress_done + forest_ambiguity_done
        primary_total = forest_mutable_progress_total + forest_ambiguity_total
        dimensions: JSONObject = {
            "forest_mutable_steps": {
                "done": forest_mutable_progress_done,
                "total": forest_mutable_progress_total,
            },
            "ambiguity_pass": {
                "done": forest_ambiguity_done,
                "total": forest_ambiguity_total,
            },
            "callsite_inventory": {
                "done": forest_inventory_callsites_total,
                "total": forest_inventory_callsites_total,
            },
        }
        for dim_name, dim_payload in forest_dimensions.items():
            check_deadline()
            dimensions[dim_name] = {
                "done": int(dim_payload.get("done", 0)),
                "total": int(dim_payload.get("total", 0)),
            }
        return {
            "format_version": 1,
            "schema": "gabion/phase_progress_v2",
            "primary_unit": "forest_mutable_steps",
            "primary_done": primary_done,
            "primary_total": primary_total,
            "dimensions": dimensions,
            "inventory": {
                "callsites_total": forest_inventory_callsites_total,
                "input_file_paths_total": forest_inventory_files_total,
            },
        }

    def _emit_forest_phase_progress() -> None:
        forest_phase_progress_v2 = _forest_phase_progress_v2()
        raw_primary_done = forest_phase_progress_v2.get("primary_done")
        raw_primary_total = forest_phase_progress_v2.get("primary_total")
        primary_done = int(raw_primary_done) if type(raw_primary_done) is int else 0
        primary_total = int(raw_primary_total) if type(raw_primary_total) is int else 0
        emit_phase_progress_fn(
            "forest",
            report_carrier=ReportCarrier(
                forest=forest,
                bundle_sites_by_path=bundle_sites_by_path,
                ambiguity_witnesses=ambiguity_witnesses,
                invariant_propositions=invariant_propositions,
                parse_failure_witnesses=parse_failure_witnesses,
            ),
            work_progress=_phase_work_progress(
                work_done=primary_done,
                work_total=primary_total,
            ),
            phase_progress_v2=forest_phase_progress_v2,
            progress_marker=forest_progress_marker,
        )

    def _on_forest_progress(progress_delta: object = 0) -> None:
        nonlocal forest_mutable_progress_done
        nonlocal forest_mutable_progress_total
        nonlocal forest_progress_marker
        nonlocal forest_dimensions
        (
            forest_mutable_progress_done,
            forest_mutable_progress_total,
            forest_progress_marker,
            forest_dimensions,
            _changed,
        ) = _apply_forest_progress_delta(
            progress_delta,
            forest_mutable_progress_done=forest_mutable_progress_done,
            forest_mutable_progress_total=forest_mutable_progress_total,
            forest_progress_marker=forest_progress_marker,
            forest_dimensions=forest_dimensions,
        )
        _emit_forest_phase_progress()

    _emit_forest_phase_progress()
    forest_started_ns = time.monotonic_ns()
    if (
        include_bundle_forest
        or include_decision_surfaces
        or include_value_decision_surfaces
        or include_lint_lines
        or include_never_invariants
        or include_wl_refinement
        or include_deadline_obligations
        or include_ambiguities
    ):
        _populate_bundle_forest(
            forest,
            groups_by_path=groups_by_path,
            file_paths=file_paths,
            project_root=config.project_root,
            include_all_sites=True,
            ignore_params=config.ignore_params,
            strictness=config.strictness,
            transparent_decorators=config.transparent_decorators,
            parse_failure_witnesses=parse_failure_witnesses,
            analysis_index=require_analysis_index_fn(),
            on_progress=_on_forest_progress,
        )
        forest_progress_marker = "forest_spec_materialization"
        forest_spec = build_forest_spec(
            include_bundle_forest=True,
            include_decision_surfaces=include_decision_surfaces,
            include_value_decision_surfaces=include_value_decision_surfaces,
            include_never_invariants=include_never_invariants,
            include_wl_refinement=include_wl_refinement,
            include_ambiguities=include_ambiguities,
            include_deadline_obligations=include_deadline_obligations,
            include_lint_findings=include_lint_lines,
            include_all_sites=True,
            ignore_params=config.ignore_params,
            decision_ignore_params=config.decision_ignore_params or config.ignore_params,
            transparent_decorators=config.transparent_decorators,
            strictness=config.strictness,
            decision_tiers=config.decision_tiers,
            require_tiers=config.decision_require_tiers,
            external_filter=config.external_filter,
        )
        deadline_check_fn(allow_frame_fallback=False)
        if include_wl_refinement:
            emit_wl_refinement_facets(
                forest=forest,
                spec=WL_REFINEMENT_SPEC,
            )
            forest_progress_marker = "wl_refinement"
            _emit_forest_phase_progress()

    _emit_forest_phase_progress()

    if include_ambiguities:
        forest_progress_marker = "ambiguity:start"
        deadline_check_fn(allow_frame_fallback=False)
        call_ambiguities = _collect_call_ambiguities(
            file_paths,
            project_root=config.project_root,
            ignore_params=config.ignore_params,
            strictness=config.strictness,
            external_filter=config.external_filter,
            transparent_decorators=config.transparent_decorators,
            parse_failure_witnesses=parse_failure_witnesses,
            analysis_index=require_analysis_index_fn(),
        )
        ambiguity_witnesses = _emit_call_ambiguities(
            call_ambiguities,
            project_root=config.project_root,
            forest=forest,
        )
        _materialize_ambiguity_suite_agg_spec(forest=forest)
        _materialize_ambiguity_virtual_set_spec(forest=forest)
        forest_ambiguity_done = 1
        forest_progress_marker = "ambiguity:done"
        _emit_forest_phase_progress()

    analysis_profile_stage_ns["analysis.forest"] += time.monotonic_ns() - forest_started_ns
    return _ForestPhaseResult(
        forest_spec=forest_spec,
        ambiguity_witnesses=ambiguity_witnesses,
    )


def _run_edge_phase(
    *,
    file_paths: list[Path],
    forest: Forest,
    bundle_sites_by_path: dict[Path, dict[str, list[set[str]]]],
    ambiguity_witnesses: list[JSONObject],
    invariant_propositions: list[InvariantProposition],
    parse_failure_witnesses: list[JSONObject],
    config: AuditConfig,
    type_audit: bool,
    type_audit_report: bool,
    type_audit_max: int,
    include_constant_smells: bool,
    include_deadness_witnesses: bool,
    include_unused_arg_smells: bool,
    require_analysis_index_fn: Callable[[], AnalysisIndex],
    emit_phase_progress_fn: Callable[..., None],
    deadline_check_fn: Callable[..., None],
    analysis_profile_stage_ns: dict[str, int],
) -> _EdgePhaseResult:
    type_suggestions: list[str] = []
    type_ambiguities: list[str] = []
    type_callsite_evidence: list[str] = []
    constant_smells: list[str] = []
    deadness_witnesses: list[JSONObject] = []
    unused_arg_smells: list[str] = []
    edge_work_total = 0
    if type_audit or type_audit_report:
        edge_work_total += 1
    if include_constant_smells or include_deadness_witnesses:
        edge_work_total += 1
    if include_unused_arg_smells:
        edge_work_total += 1
    edge_work_done = 0

    def _emit_edge_phase_progress() -> None:
        emit_phase_progress_fn(
            "edge",
            report_carrier=ReportCarrier(
                forest=forest,
                bundle_sites_by_path=bundle_sites_by_path,
                type_suggestions=type_suggestions,
                type_ambiguities=type_ambiguities,
                type_callsite_evidence=type_callsite_evidence,
                constant_smells=constant_smells,
                unused_arg_smells=unused_arg_smells,
                deadness_witnesses=deadness_witnesses,
                ambiguity_witnesses=ambiguity_witnesses,
                invariant_propositions=invariant_propositions,
                parse_failure_witnesses=parse_failure_witnesses,
            ),
            work_progress=_phase_work_progress(
                work_done=edge_work_done,
                work_total=edge_work_total,
            ),
        )

    _emit_edge_phase_progress()
    edge_started_ns = time.monotonic_ns()
    if type_audit or type_audit_report:
        deadline_check_fn(allow_frame_fallback=False)
        type_suggestions, type_ambiguities, type_callsite_evidence = (
            analyze_type_flow_repo_with_evidence(
                file_paths,
                project_root=config.project_root,
                ignore_params=config.ignore_params,
                strictness=config.strictness,
                external_filter=config.external_filter,
                transparent_decorators=config.transparent_decorators,
                parse_failure_witnesses=parse_failure_witnesses,
                analysis_index=require_analysis_index_fn(),
            )
        )
        if type_audit_report:
            type_suggestions = type_suggestions[:type_audit_max]
            type_ambiguities = type_ambiguities[:type_audit_max]
            type_callsite_evidence = type_callsite_evidence[:type_audit_max]
        edge_work_done += 1
        _emit_edge_phase_progress()

    if include_constant_smells or include_deadness_witnesses:
        constant_details = _collect_constant_flow_details(
            file_paths,
            project_root=config.project_root,
            ignore_params=config.ignore_params,
            strictness=config.strictness,
            external_filter=config.external_filter,
            transparent_decorators=config.transparent_decorators,
            parse_failure_witnesses=parse_failure_witnesses,
            analysis_index=require_analysis_index_fn(),
        )
        if include_constant_smells:
            constant_smells = _constant_smells_from_details(constant_details)
        if include_deadness_witnesses:
            deadness_witnesses = _deadness_witnesses_from_constant_details(
                constant_details,
                project_root=config.project_root,
            )
        edge_work_done += 1
        _emit_edge_phase_progress()

    if include_unused_arg_smells:
        unused_arg_smells = analyze_unused_arg_flow_repo(
            file_paths,
            project_root=config.project_root,
            ignore_params=config.ignore_params,
            strictness=config.strictness,
            external_filter=config.external_filter,
            transparent_decorators=config.transparent_decorators,
            parse_failure_witnesses=parse_failure_witnesses,
            analysis_index=require_analysis_index_fn(),
        )
        edge_work_done += 1
        _emit_edge_phase_progress()

    _emit_edge_phase_progress()
    analysis_profile_stage_ns["analysis.edge"] += time.monotonic_ns() - edge_started_ns
    return _EdgePhaseResult(
        type_suggestions=type_suggestions,
        type_ambiguities=type_ambiguities,
        type_callsite_evidence=type_callsite_evidence,
        constant_smells=constant_smells,
        deadness_witnesses=deadness_witnesses,
        unused_arg_smells=unused_arg_smells,
    )


def _run_post_phase(
    *,
    file_paths: list[Path],
    forest: Forest,
    groups_by_path: dict[Path, dict[str, list[set[str]]]],
    bundle_sites_by_path: dict[Path, dict[str, list[set[str]]]],
    type_suggestions: list[str],
    type_ambiguities: list[str],
    type_callsite_evidence: list[str],
    constant_smells: list[str],
    unused_arg_smells: list[str],
    deadness_witnesses: list[JSONObject],
    ambiguity_witnesses: list[JSONObject],
    invariant_propositions: list[InvariantProposition],
    parse_failure_witnesses: list[JSONObject],
    config: AuditConfig,
    include_deadline_obligations: bool,
    include_decision_surfaces: bool,
    include_value_decision_surfaces: bool,
    include_exception_obligations: bool,
    include_handledness_witnesses: bool,
    include_never_invariants: bool,
    include_lint_lines: bool,
    include_coherence_witnesses: bool,
    include_rewrite_plans: bool,
    require_analysis_index_fn: Callable[[], AnalysisIndex],
    emit_phase_progress_fn: Callable[..., None],
    analysis_profile_stage_ns: dict[str, int],
) -> _PostPhaseResult:
    deadline_obligations: list[JSONObject] = []
    decision_surfaces: list[str] = []
    decision_warnings: list[str] = []
    decision_lint_lines: list[str] = []
    value_decision_surfaces: list[str] = []
    value_decision_rewrites: list[str] = []
    fingerprint_warnings: list[str] = []
    fingerprint_matches: list[str] = []
    fingerprint_synth: list[str] = []
    fingerprint_synth_registry: object = None
    fingerprint_provenance: list[JSONObject] = []
    coherence_witnesses: list[JSONObject] = []
    rewrite_plans: list[JSONObject] = []
    exception_obligations: list[JSONObject] = []
    never_invariants: list[JSONObject] = []
    handledness_witnesses: list[JSONObject] = []
    context_suggestions: list[str] = []
    lint_lines: list[str] = []
    post_task_flags = [
        include_deadline_obligations,
        include_decision_surfaces,
        include_value_decision_surfaces,
        include_exception_obligations
        or (include_lint_lines and bool(config.never_exceptions)),
        include_never_invariants,
        config.fingerprint_registry is not None and bool(config.fingerprint_index),
        include_lint_lines,
    ]
    post_work_total = sum(1 for enabled in post_task_flags if enabled)
    post_work_done = 0
    post_progress_marker = "enter"

    def _emit_post_phase_progress(*, marker: str = "") -> None:
        nonlocal post_progress_marker
        if marker:
            post_progress_marker = marker
        emit_phase_progress_fn(
            "post",
            report_carrier=ReportCarrier(
                forest=forest,
                bundle_sites_by_path=bundle_sites_by_path,
                type_suggestions=type_suggestions,
                type_ambiguities=type_ambiguities,
                type_callsite_evidence=type_callsite_evidence,
                constant_smells=constant_smells,
                unused_arg_smells=unused_arg_smells,
                deadness_witnesses=deadness_witnesses,
                coherence_witnesses=coherence_witnesses,
                rewrite_plans=rewrite_plans,
                exception_obligations=exception_obligations,
                never_invariants=never_invariants,
                ambiguity_witnesses=ambiguity_witnesses,
                handledness_witnesses=handledness_witnesses,
                decision_surfaces=decision_surfaces,
                value_decision_surfaces=value_decision_surfaces,
                decision_warnings=decision_warnings,
                fingerprint_warnings=fingerprint_warnings,
                fingerprint_matches=fingerprint_matches,
                fingerprint_synth=fingerprint_synth,
                fingerprint_provenance=fingerprint_provenance,
                context_suggestions=context_suggestions,
                invariant_propositions=invariant_propositions,
                value_decision_rewrites=value_decision_rewrites,
                deadline_obligations=deadline_obligations,
                parse_failure_witnesses=parse_failure_witnesses,
                progress_marker=post_progress_marker,
            ),
            work_progress=_phase_work_progress(
                work_done=post_work_done,
                work_total=post_work_total,
            ),
            progress_marker=post_progress_marker,
        )

    _emit_post_phase_progress()
    post_started_ns = time.monotonic_ns()
    if include_deadline_obligations:
        _emit_post_phase_progress(marker="deadline_obligations:start")

        def _on_deadline_obligation_progress(marker: str) -> None:
            _emit_post_phase_progress(
                marker=f"deadline_obligations:{str(marker or 'in_progress')}"
            )

        deadline_obligations = _collect_deadline_obligations(
            file_paths,
            project_root=config.project_root,
            config=config,
            forest=forest,
            parse_failure_witnesses=parse_failure_witnesses,
            analysis_index=require_analysis_index_fn(),
            on_progress=_on_deadline_obligation_progress,
        )
        _emit_post_phase_progress(marker="deadline_obligations:suite_order")
        _materialize_suite_order_spec(forest=forest)
        post_work_done += 1
        _emit_post_phase_progress(marker="deadline_obligations:done")

    if include_decision_surfaces:
        _emit_post_phase_progress(marker="decision_surfaces:start")
        decision_surfaces, decision_warnings, decision_lint_lines = (
            analyze_decision_surfaces_repo(
                file_paths,
                project_root=config.project_root,
                ignore_params=config.decision_ignore_params or config.ignore_params,
                strictness=config.strictness,
                external_filter=config.external_filter,
                transparent_decorators=config.transparent_decorators,
                decision_tiers=config.decision_tiers,
                require_tiers=config.decision_require_tiers,
                forest=forest,
                parse_failure_witnesses=parse_failure_witnesses,
                analysis_index=require_analysis_index_fn(),
            )
        )
        post_work_done += 1
        _emit_post_phase_progress(marker="decision_surfaces:done")

    if include_value_decision_surfaces:
        _emit_post_phase_progress(marker="value_decisions:start")
        (
            value_decision_surfaces,
            value_warnings,
            value_decision_rewrites,
            value_lint_lines,
        ) = analyze_value_encoded_decisions_repo(
            file_paths,
            project_root=config.project_root,
            ignore_params=config.decision_ignore_params or config.ignore_params,
            strictness=config.strictness,
            external_filter=config.external_filter,
            transparent_decorators=config.transparent_decorators,
            decision_tiers=config.decision_tiers,
            require_tiers=config.decision_require_tiers,
            forest=forest,
            parse_failure_witnesses=parse_failure_witnesses,
            analysis_index=require_analysis_index_fn(),
        )
        decision_warnings.extend(value_warnings)
        decision_lint_lines.extend(value_lint_lines)
        post_work_done += 1
        _emit_post_phase_progress(marker="value_decisions:done")

    need_exception_obligations = include_exception_obligations or (
        include_lint_lines and bool(config.never_exceptions)
    )
    if need_exception_obligations or include_handledness_witnesses:
        handledness_witnesses = _collect_handledness_witnesses(
            file_paths,
            project_root=config.project_root,
            ignore_params=config.ignore_params,
        )
    if need_exception_obligations:
        _emit_post_phase_progress(marker="exception_obligations:start")
        exception_obligations = _collect_exception_obligations(
            file_paths,
            project_root=config.project_root,
            ignore_params=config.ignore_params,
            handledness_witnesses=handledness_witnesses,
            deadness_witnesses=deadness_witnesses,
            never_exceptions=config.never_exceptions,
        )
        post_work_done += 1
        _emit_post_phase_progress(marker="exception_obligations:done")
    if include_never_invariants:
        _emit_post_phase_progress(marker="never_invariants:start")
        never_invariants = _collect_never_invariants(
            file_paths,
            project_root=config.project_root,
            ignore_params=config.ignore_params,
            forest=forest,
            deadness_witnesses=deadness_witnesses,
        )
        post_work_done += 1
        _emit_post_phase_progress(marker="never_invariants:done")
    if config.fingerprint_registry is not None and config.fingerprint_index:
        _emit_post_phase_progress(marker="fingerprint:start")
        annotations_by_path: dict[Path, dict[str, dict[str, object]]] = {}
        annotation_total = len(file_paths)
        if annotation_total:
            for annotation_index, annotation_path in enumerate(file_paths, start=1):
                check_deadline()
                annotations_by_path.update(
                    _param_annotations_by_path(
                        [annotation_path],
                        ignore_params=config.ignore_params,
                        parse_failure_witnesses=parse_failure_witnesses,
                    )
                )
                _emit_post_phase_progress(
                    marker=f"fingerprint:annotations:{annotation_index}/{annotation_total}"
                )
        else:
            _emit_post_phase_progress(marker="fingerprint:annotations:0/0")
        base_keys, ctor_keys = _collect_fingerprint_atom_keys(
            groups_by_path,
            annotations_by_path,
        )
        for key in base_keys:
            check_deadline()
            config.fingerprint_registry.get_or_assign(key)
        if config.constructor_registry is not None:
            for key in ctor_keys:
                check_deadline()
                config.constructor_registry.get_or_assign(key)
        _emit_post_phase_progress(marker="fingerprint:normalize")
        fingerprint_warnings = _compute_fingerprint_warnings(
            groups_by_path,
            annotations_by_path,
            registry=config.fingerprint_registry,
            index=config.fingerprint_index,
            ctor_registry=config.constructor_registry,
        )
        _emit_post_phase_progress(marker="fingerprint:warnings")
        fingerprint_matches = _compute_fingerprint_matches(
            groups_by_path,
            annotations_by_path,
            registry=config.fingerprint_registry,
            index=config.fingerprint_index,
            ctor_registry=config.constructor_registry,
        )
        _emit_post_phase_progress(marker="fingerprint:matches")
        fingerprint_provenance = _compute_fingerprint_provenance(
            groups_by_path,
            annotations_by_path,
            registry=config.fingerprint_registry,
            project_root=config.project_root,
            index=config.fingerprint_index,
            ctor_registry=config.constructor_registry,
        )
        _emit_post_phase_progress(marker="fingerprint:provenance")
        fingerprint_synth, fingerprint_synth_registry = _compute_fingerprint_synth(
            groups_by_path,
            annotations_by_path,
            registry=config.fingerprint_registry,
            ctor_registry=config.constructor_registry,
            min_occurrences=config.fingerprint_synth_min_occurrences,
            version=config.fingerprint_synth_version,
            existing=config.fingerprint_synth_registry,
        )
        _emit_post_phase_progress(marker="fingerprint:synth")
        if include_coherence_witnesses:
            coherence_witnesses = _compute_fingerprint_coherence(
                fingerprint_provenance,
                synth_version=config.fingerprint_synth_version,
            )
            _emit_post_phase_progress(marker="fingerprint:coherence")
        if include_rewrite_plans:
            rewrite_plans = _compute_fingerprint_rewrite_plans(
                fingerprint_provenance,
                coherence_witnesses,
                synth_version=config.fingerprint_synth_version,
                exception_obligations=(
                    exception_obligations if include_exception_obligations else None
                ),
            )
            _emit_post_phase_progress(marker="fingerprint:rewrite_plans")
        post_work_done += 1
        _emit_post_phase_progress(marker="fingerprint:done")

    if decision_surfaces:
        for entry in decision_surfaces:
            check_deadline()
            if "(internal callers" in entry:
                context_suggestions.append(f"Consider contextvar for {entry}")
        _emit_post_phase_progress(marker="context_suggestions")

    if include_lint_lines:
        _emit_post_phase_progress(marker="lint:start")
        broad_type_lint_lines = _internal_broad_type_lint_lines(
            file_paths,
            project_root=config.project_root,
            ignore_params=config.ignore_params,
            strictness=config.strictness,
            external_filter=config.external_filter,
            transparent_decorators=config.transparent_decorators,
            parse_failure_witnesses=parse_failure_witnesses,
            analysis_index=require_analysis_index_fn(),
        )
        lint_lines = _compute_lint_lines(
            forest=forest,
            groups_by_path=groups_by_path,
            bundle_sites_by_path=bundle_sites_by_path,
            type_callsite_evidence=type_callsite_evidence,
            ambiguity_witnesses=ambiguity_witnesses,
            exception_obligations=exception_obligations,
            never_invariants=never_invariants,
            deadline_obligations=deadline_obligations,
            decision_lint_lines=decision_lint_lines,
            broad_type_lint_lines=broad_type_lint_lines,
            constant_smells=constant_smells,
            unused_arg_smells=unused_arg_smells,
        )
        post_work_done += 1
        _emit_post_phase_progress(marker="lint:done")

    _emit_post_phase_progress(marker="complete")
    analysis_profile_stage_ns["analysis.post"] += time.monotonic_ns() - post_started_ns
    return _PostPhaseResult(
        deadline_obligations=deadline_obligations,
        decision_surfaces=decision_surfaces,
        decision_warnings=decision_warnings,
        value_decision_surfaces=value_decision_surfaces,
        value_decision_rewrites=value_decision_rewrites,
        fingerprint_warnings=fingerprint_warnings,
        fingerprint_matches=fingerprint_matches,
        fingerprint_synth=fingerprint_synth,
        fingerprint_synth_registry=fingerprint_synth_registry,
        fingerprint_provenance=fingerprint_provenance,
        coherence_witnesses=coherence_witnesses,
        rewrite_plans=rewrite_plans,
        exception_obligations=exception_obligations,
        never_invariants=never_invariants,
        handledness_witnesses=handledness_witnesses,
        context_suggestions=context_suggestions,
        lint_lines=lint_lines,
    )


def analyze_paths(
    paths: list[Path],
    *,
    forest: Forest,
    recursive: bool,
    type_audit: bool,
    type_audit_report: bool,
    type_audit_max: int,
    include_constant_smells: bool,
    include_unused_arg_smells: bool,
    include_deadness_witnesses: bool = False,
    include_coherence_witnesses: bool = False,
    include_rewrite_plans: bool = False,
    include_exception_obligations: bool = False,
    include_handledness_witnesses: bool = False,
    include_never_invariants: bool = False,
    include_wl_refinement: bool = False,
    include_decision_surfaces: bool = False,
    include_value_decision_surfaces: bool = False,
    include_invariant_propositions: bool = False,
    include_lint_lines: bool = False,
    include_ambiguities: bool = False,
    include_bundle_forest: bool = False,
    include_deadline_obligations: bool = False,
    config: object = None,
    file_paths_override: object = None,
    collection_resume: object = None,
    on_collection_progress: object = None,
    on_phase_progress: object = None,
) -> AnalysisResult:
    _bind_audit_symbols()
    check_deadline()
    forest_token = set_forest(forest)

    def _invalid_progress_callback(*, callback_name: str, callback_value: object) -> None:
        never(
            f"invalid {callback_name} callback",
            callback_type=type(callback_value).__name__,
        )

    if on_collection_progress is None:
        emit_collection_progress_enabled = False
        _collection_progress_callback = lambda _payload: None
    elif callable(on_collection_progress):
        emit_collection_progress_enabled = True
        _collection_progress_callback = on_collection_progress
    else:
        _invalid_progress_callback(
            callback_name="on_collection_progress",
            callback_value=on_collection_progress,
        )
    if on_phase_progress is None:
        emit_phase_progress_enabled = False
        _phase_progress_callback = (
            lambda _phase,
            _groups_by_path,
            _report_carrier,
            _work_done,
            _work_total: None
        )
    elif callable(on_phase_progress):
        emit_phase_progress_enabled = True
        _phase_progress_callback = on_phase_progress
    else:
        _invalid_progress_callback(
            callback_name="on_phase_progress",
            callback_value=on_phase_progress,
        )

    def _emit_collection_progress(*, force: bool = False) -> None:
        _ = force

    def _best_effort_timeout_flush(callback: Callable[[], None]) -> None:
        try:
            callback()
        except TimeoutExceeded:
            return

    try:
        if config is None:
            config = AuditConfig()
        runtime_config = cast(AuditConfig, config)
        unsupported_by_adapter: list[JSONObject] = []
        if include_bundle_forest and not _capability_enabled(
            runtime_config.adapter_contract,
            "bundle_inference",
        ):
            include_bundle_forest = False
            unsupported_by_adapter.append(
                _unsupported_surface_diagnostic(
                    surface=_SURFACE_BUNDLE_INFERENCE,
                    capability_name="bundle_inference",
                    runtime_config=runtime_config,
                )
            )
        if (include_decision_surfaces or include_value_decision_surfaces) and not _capability_enabled(
            runtime_config.adapter_contract,
            "decision_surfaces",
        ):
            include_decision_surfaces = False
            include_value_decision_surfaces = False
            unsupported_by_adapter.append(
                _unsupported_surface_diagnostic(
                    surface=_SURFACE_DECISION_SURFACES,
                    capability_name="decision_surfaces",
                    runtime_config=runtime_config,
                )
            )
        if (type_audit or type_audit_report) and not _capability_enabled(
            runtime_config.adapter_contract,
            "type_flow",
        ):
            type_audit = False
            type_audit_report = False
            unsupported_by_adapter.append(
                _unsupported_surface_diagnostic(
                    surface=_SURFACE_TYPE_FLOW,
                    capability_name="type_flow",
                    runtime_config=runtime_config,
                )
            )
        if (include_exception_obligations or include_handledness_witnesses) and not _capability_enabled(
            runtime_config.adapter_contract,
            "exception_obligations",
        ):
            include_exception_obligations = False
            include_handledness_witnesses = False
            unsupported_by_adapter.append(
                _unsupported_surface_diagnostic(
                    surface=_SURFACE_EXCEPTION_OBLIGATIONS,
                    capability_name="exception_obligations",
                    runtime_config=runtime_config,
                )
            )
        if include_rewrite_plans and not _capability_enabled(
            runtime_config.adapter_contract,
            "rewrite_plan_support",
        ):
            include_rewrite_plans = False
            unsupported_by_adapter.append(
                _unsupported_surface_diagnostic(
                    surface=_SURFACE_REWRITE_PLAN_SUPPORT,
                    capability_name="rewrite_plan_support",
                    runtime_config=runtime_config,
                )
            )
        if file_paths_override is None:
            file_paths = resolve_analysis_paths(paths, config=runtime_config)
        else:
            file_paths = _iter_monotonic_paths(
                file_paths_override,
                source="analyze_paths.file_paths_override",
            )
        (
            groups_by_path,
            param_spans_by_path,
            bundle_sites_by_path,
            invariant_propositions,
            completed_paths,
            in_progress_scan_by_path,
            analysis_index_resume_payload,
        ) = _load_analysis_collection_resume_payload(
            payload=collection_resume,
            file_paths=file_paths,
            include_invariant_propositions=include_invariant_propositions,
        )
        file_stage_timings_v1_by_path: dict[Path, JSONObject] = {}
        match collection_resume:
            case Mapping() as collection_resume_payload:
                raw_stage_timings = collection_resume_payload.get(
                    "file_stage_timings_v1_by_path"
                )
                match raw_stage_timings:
                    case Mapping() as stage_timings_payload:
                        for path in file_paths:
                            check_deadline()
                            path_key = _analysis_collection_resume_path_key(path)
                            raw_entry = stage_timings_payload.get(path_key)
                            match raw_entry:
                                case Mapping() as stage_entry_payload:
                                    normalized_entry: JSONObject = {}
                                    for key, value in stage_entry_payload.items():
                                        check_deadline()
                                        normalized_entry[str(key)] = value
                                    file_stage_timings_v1_by_path[path] = normalized_entry
                                case _:
                                    pass
                    case _:
                        pass
            case _:
                pass
        forest_spec: object = None
        planned_forest_spec_id: object = None
        ambiguity_witnesses: list[JSONObject] = []
        parse_failure_witnesses: list[JSONObject] = []
        analysis_index: object = None
        analysis_profile_stage_ns: dict[str, int] = {
            "analysis.collection": 0,
            "analysis.analysis_index": 0,
            "analysis.forest": 0,
            "analysis.edge": 0,
            "analysis.post": 0,
        }
        analysis_profile_counters: Counter[str] = Counter(
            {
                "analysis.files_total": len(file_paths),
                "analysis.files_completed": len(completed_paths),
                "analysis.collection_progress_emits": 0,
                "analysis.phase_progress_emits": 0,
            }
        )

        def _deadline_check(*, allow_frame_fallback: bool) -> None:
            forest_spec_id = None
            if forest_spec is not None:
                forest_spec_id = forest_spec_metadata(forest_spec).get(
                    "generated_by_forest_spec_id"
                )
            check_deadline(
                    project_root=runtime_config.project_root,
                    forest_spec_id=str(forest_spec_id) if forest_spec_id else None,
                    allow_frame_fallback=allow_frame_fallback,
                )

        if (
            include_bundle_forest
            or include_decision_surfaces
            or include_value_decision_surfaces
            or include_lint_lines
            or include_never_invariants
            or include_wl_refinement
            or include_deadline_obligations
            or include_ambiguities
        ):
            planned_spec = build_forest_spec(
                include_bundle_forest=True,
                include_decision_surfaces=include_decision_surfaces,
                include_value_decision_surfaces=include_value_decision_surfaces,
                include_never_invariants=include_never_invariants,
                include_wl_refinement=include_wl_refinement,
                include_ambiguities=include_ambiguities,
                include_deadline_obligations=include_deadline_obligations,
                include_lint_findings=include_lint_lines,
                include_all_sites=True,
                ignore_params=runtime_config.ignore_params,
                decision_ignore_params=runtime_config.decision_ignore_params
                or runtime_config.ignore_params,
                transparent_decorators=runtime_config.transparent_decorators,
                strictness=runtime_config.strictness,
                decision_tiers=runtime_config.decision_tiers,
                require_tiers=runtime_config.decision_require_tiers,
                external_filter=runtime_config.external_filter,
            )
            planned_forest_spec_id = str(
                forest_spec_metadata(planned_spec).get("generated_by_forest_spec_id", "")
                or ""
            )

        def _require_analysis_index() -> AnalysisIndex:
            nonlocal analysis_index
            nonlocal analysis_index_resume_payload
            if analysis_index is None:
                index_started_ns = time.monotonic_ns()

                def _on_analysis_index_progress(progress_payload: JSONObject) -> None:
                    nonlocal analysis_index_resume_payload
                    analysis_index_resume_payload = {
                        str(key): progress_payload[key] for key in progress_payload
                    }
                    _emit_collection_progress(force=True)

                analysis_index = _build_analysis_index(
                    file_paths,
                    project_root=runtime_config.project_root,
                    ignore_params=runtime_config.ignore_params,
                    strictness=runtime_config.strictness,
                    external_filter=runtime_config.external_filter,
                    transparent_decorators=runtime_config.transparent_decorators,
                    parse_failure_witnesses=parse_failure_witnesses,
                    resume_payload=analysis_index_resume_payload,
                    on_progress=(
                        _on_analysis_index_progress
                        if emit_collection_progress_enabled
                        else None
                    ),
                    forest_spec_id=str(planned_forest_spec_id)
                    if planned_forest_spec_id
                    else None,
                    fingerprint_seed_revision=runtime_config.fingerprint_seed_revision,
                    decision_ignore_params=runtime_config.decision_ignore_params,
                    decision_require_tiers=runtime_config.decision_require_tiers,
                )
                analysis_profile_stage_ns["analysis.analysis_index"] += (
                    time.monotonic_ns() - index_started_ns
                )
            return cast(AnalysisIndex, analysis_index)

        collection_progress_last_emit_monotonic = 0.0
        collection_progress_has_emitted = False

        def _emit_collection_progress(*, force: bool = False) -> None:
            nonlocal collection_progress_last_emit_monotonic
            nonlocal collection_progress_has_emitted
            if not emit_collection_progress_enabled:
                return
            now = time.monotonic()
            should_skip_emit = (
                not force
                and collection_progress_has_emitted
                and now - collection_progress_last_emit_monotonic
                < _PROGRESS_EMIT_MIN_INTERVAL_SECONDS
            )
            if should_skip_emit:
                collection_progress_last_emit_monotonic = collection_progress_last_emit_monotonic
            else:
                collection_progress_last_emit_monotonic = now
                collection_progress_has_emitted = True
                analysis_profile_counters["analysis.collection_progress_emits"] += 1
                _collection_progress_callback(
                    _build_analysis_collection_resume_payload(
                        groups_by_path=groups_by_path,
                        param_spans_by_path=param_spans_by_path,
                        bundle_sites_by_path=bundle_sites_by_path,
                        invariant_propositions=invariant_propositions,
                        completed_paths=completed_paths,
                        in_progress_scan_by_path=in_progress_scan_by_path,
                        analysis_index_resume=analysis_index_resume_payload,
                        file_stage_timings_v1_by_path=file_stage_timings_v1_by_path,
                    )
                )

        def _emit_phase_progress(
            phase: ReportProjectionPhase,
            *,
            report_carrier: ReportCarrier,
            work_progress: _PhaseWorkProgress,
            phase_progress_v2: object = None,
            progress_marker: str = "",
        ) -> None:
            if not emit_phase_progress_enabled:
                return
            match phase_progress_v2:
                case Mapping() as phase_progress_map:
                    report_carrier.phase_progress_v2 = {
                        str(key): phase_progress_map[key] for key in phase_progress_map
                    }
                case _:
                    report_carrier.phase_progress_v2 = None
            report_carrier.progress_marker = progress_marker
            analysis_profile_counters["analysis.phase_progress_emits"] += 1
            _phase_progress_callback(
                phase,
                groups_by_path,
                report_carrier,
                work_progress.work_done,
                work_progress.work_total,
            )

        collection_started_ns = time.monotonic_ns()
        for path in file_paths:
            check_deadline()
            if path in completed_paths:
                continue
            in_progress_scan_by_path[path] = {"phase": "scan_pending"}
            _emit_collection_progress(force=True)
            _deadline_check(allow_frame_fallback=True)

            def _on_file_scan_progress(progress_state: JSONObject) -> None:
                in_progress_scan_by_path[path] = progress_state
                _emit_collection_progress()

            def _on_file_scan_profile(profile_payload: JSONObject) -> None:
                file_stage_timings_v1_by_path[path] = {
                    str(key): profile_payload[key] for key in profile_payload
                }

            groups, spans, sites = _analyze_file_internal(
                path,
                recursive=recursive,
                config=runtime_config,
                resume_state=in_progress_scan_by_path.get(path),
                on_progress=_on_file_scan_progress,
                on_profile=_on_file_scan_profile,
            )
            groups_by_path[path] = groups
            param_spans_by_path[path] = spans
            bundle_sites_by_path[path] = sites
            in_progress_scan_by_path.pop(path, None)
            if include_invariant_propositions:
                invariant_propositions.extend(
                    _collect_invariant_propositions(
                        path,
                        ignore_params=runtime_config.ignore_params,
                        project_root=runtime_config.project_root,
                        emitters=runtime_config.invariant_emitters,
                    )
                )
            completed_paths.add(path)
            analysis_profile_counters["analysis.files_completed"] = len(completed_paths)
            _emit_collection_progress(force=True)

        analysis_profile_stage_ns["analysis.collection"] += (
            time.monotonic_ns() - collection_started_ns
        )
        _emit_phase_progress(
            "collection",
            report_carrier=ReportCarrier(
                forest=forest,
                bundle_sites_by_path=bundle_sites_by_path,
                invariant_propositions=invariant_propositions,
                parse_failure_witnesses=parse_failure_witnesses,
            ),
            work_progress=_phase_work_progress(work_done=1, work_total=1),
            phase_progress_v2={
                "format_version": 1,
                "schema": "gabion/phase_progress_v2",
                "primary_unit": "collection_files",
                "primary_done": 1,
                "primary_total": 1,
                "dimensions": {
                    "collection_files": {"done": 1, "total": 1},
                },
                "inventory": {},
            },
        )

        forest_phase = _run_forest_phase(
            file_paths=file_paths,
            forest=forest,
            groups_by_path=groups_by_path,
            bundle_sites_by_path=bundle_sites_by_path,
            invariant_propositions=invariant_propositions,
            parse_failure_witnesses=parse_failure_witnesses,
            analysis_index_for_progress=analysis_index,
            config=config,
            include_bundle_forest=include_bundle_forest,
            include_decision_surfaces=include_decision_surfaces,
            include_value_decision_surfaces=include_value_decision_surfaces,
            include_lint_lines=include_lint_lines,
            include_never_invariants=include_never_invariants,
            include_wl_refinement=include_wl_refinement,
            include_deadline_obligations=include_deadline_obligations,
            include_ambiguities=include_ambiguities,
            require_analysis_index_fn=_require_analysis_index,
            emit_phase_progress_fn=_emit_phase_progress,
            deadline_check_fn=_deadline_check,
            analysis_profile_stage_ns=analysis_profile_stage_ns,
        )
        forest_spec = forest_phase.forest_spec
        ambiguity_witnesses = forest_phase.ambiguity_witnesses

        edge_phase = _run_edge_phase(
            file_paths=file_paths,
            forest=forest,
            bundle_sites_by_path=bundle_sites_by_path,
            ambiguity_witnesses=ambiguity_witnesses,
            invariant_propositions=invariant_propositions,
            parse_failure_witnesses=parse_failure_witnesses,
            config=config,
            type_audit=type_audit,
            type_audit_report=type_audit_report,
            type_audit_max=type_audit_max,
            include_constant_smells=include_constant_smells,
            include_deadness_witnesses=include_deadness_witnesses,
            include_unused_arg_smells=include_unused_arg_smells,
            require_analysis_index_fn=_require_analysis_index,
            emit_phase_progress_fn=_emit_phase_progress,
            deadline_check_fn=_deadline_check,
            analysis_profile_stage_ns=analysis_profile_stage_ns,
        )
        type_suggestions = edge_phase.type_suggestions
        type_ambiguities = edge_phase.type_ambiguities
        type_callsite_evidence = edge_phase.type_callsite_evidence
        constant_smells = edge_phase.constant_smells
        deadness_witnesses = edge_phase.deadness_witnesses
        unused_arg_smells = edge_phase.unused_arg_smells

        post_phase = _run_post_phase(
            file_paths=file_paths,
            forest=forest,
            groups_by_path=groups_by_path,
            bundle_sites_by_path=bundle_sites_by_path,
            type_suggestions=type_suggestions,
            type_ambiguities=type_ambiguities,
            type_callsite_evidence=type_callsite_evidence,
            constant_smells=constant_smells,
            unused_arg_smells=unused_arg_smells,
            deadness_witnesses=deadness_witnesses,
            ambiguity_witnesses=ambiguity_witnesses,
            invariant_propositions=invariant_propositions,
            parse_failure_witnesses=parse_failure_witnesses,
            config=config,
            include_deadline_obligations=include_deadline_obligations,
            include_decision_surfaces=include_decision_surfaces,
            include_value_decision_surfaces=include_value_decision_surfaces,
            include_exception_obligations=include_exception_obligations,
            include_handledness_witnesses=include_handledness_witnesses,
            include_never_invariants=include_never_invariants,
            include_lint_lines=include_lint_lines,
            include_coherence_witnesses=include_coherence_witnesses,
            include_rewrite_plans=include_rewrite_plans,
            require_analysis_index_fn=_require_analysis_index,
            emit_phase_progress_fn=_emit_phase_progress,
            analysis_profile_stage_ns=analysis_profile_stage_ns,
        )
        deadline_obligations = post_phase.deadline_obligations
        decision_surfaces = post_phase.decision_surfaces
        decision_warnings = post_phase.decision_warnings
        value_decision_surfaces = post_phase.value_decision_surfaces
        value_decision_rewrites = post_phase.value_decision_rewrites
        fingerprint_warnings = post_phase.fingerprint_warnings
        fingerprint_matches = post_phase.fingerprint_matches
        fingerprint_synth = post_phase.fingerprint_synth
        fingerprint_synth_registry = post_phase.fingerprint_synth_registry
        fingerprint_provenance = post_phase.fingerprint_provenance
        coherence_witnesses = post_phase.coherence_witnesses
        rewrite_plans = post_phase.rewrite_plans
        exception_obligations = post_phase.exception_obligations
        never_invariants = post_phase.never_invariants
        handledness_witnesses = post_phase.handledness_witnesses
        context_suggestions = post_phase.context_suggestions
        lint_lines = post_phase.lint_lines
        profiling_v1 = _profiling_v1_payload(
            stage_ns=analysis_profile_stage_ns,
            counters=analysis_profile_counters,
        )
        profiling_v1["file_stage_timings_v1_by_path"] = {
            _analysis_collection_resume_path_key(path): file_stage_timings_v1_by_path[path]
            for path in sort_once(
                file_stage_timings_v1_by_path,
                source="analyze_paths.file_stage_timings_v1_by_path",
                key=_analysis_collection_resume_path_key,
            )
        }

        return AnalysisResult(
            groups_by_path=groups_by_path,
            param_spans_by_path=param_spans_by_path,
            bundle_sites_by_path=bundle_sites_by_path,
            type_suggestions=type_suggestions,
            type_ambiguities=type_ambiguities,
            type_callsite_evidence=type_callsite_evidence,
            constant_smells=constant_smells,
            unused_arg_smells=unused_arg_smells,
            forest=forest,
            lint_lines=lint_lines,
            deadness_witnesses=deadness_witnesses,
            decision_surfaces=decision_surfaces,
            value_decision_surfaces=value_decision_surfaces,
            decision_warnings=sort_once(
                set(decision_warnings),
                source="analyze_paths.decision_warnings",
            ),
            fingerprint_warnings=fingerprint_warnings,
            fingerprint_matches=fingerprint_matches,
            fingerprint_synth=fingerprint_synth,
            fingerprint_synth_registry=fingerprint_synth_registry,
            fingerprint_provenance=fingerprint_provenance,
            coherence_witnesses=coherence_witnesses,
            rewrite_plans=rewrite_plans,
            exception_obligations=exception_obligations,
            never_invariants=never_invariants,
            handledness_witnesses=handledness_witnesses,
            context_suggestions=context_suggestions,
            invariant_propositions=invariant_propositions,
            value_decision_rewrites=value_decision_rewrites,
            ambiguity_witnesses=ambiguity_witnesses,
            deadline_obligations=deadline_obligations,
            parse_failure_witnesses=parse_failure_witnesses,
            forest_spec=forest_spec,
            profiling_v1=profiling_v1,
            unsupported_by_adapter=unsupported_by_adapter,
        )
    except TimeoutExceeded:
        _best_effort_timeout_flush(lambda: _emit_collection_progress(force=True))
        raise
    finally:
        reset_forest(forest_token)
