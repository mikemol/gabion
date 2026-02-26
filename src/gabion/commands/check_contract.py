# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import typer

from gabion.json_types import JSONObject

SplitCsvEntriesFn = Callable[[list[str]], list[str]]
SplitCsvFn = Callable[[str], list[str]]


def split_csv_entries(entries: list[str]) -> list[str]:
    merged: list[str] = []
    for entry in entries:
        merged.extend([part.strip() for part in entry.split(",") if part.strip()])
    return merged


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


@dataclass(frozen=True)
class CheckArtifactFlags:
    emit_test_obsolescence: bool
    emit_test_evidence_suggestions: bool
    emit_call_clusters: bool
    emit_call_cluster_consolidation: bool
    emit_test_annotation_drift: bool
    emit_semantic_coverage_map: bool = False


@dataclass(frozen=True)
class CheckPolicyFlags:
    fail_on_violations: bool
    fail_on_type_ambiguities: bool
    lint: bool


_CHECK_AUX_DOMAIN_ACTIONS: dict[str, tuple[str, ...]] = {
    "obsolescence": ("report", "state", "delta", "baseline-write"),
    "annotation-drift": ("report", "state", "delta", "baseline-write"),
    "ambiguity": ("state", "delta", "baseline-write"),
}


@dataclass(frozen=True)
class CheckAuxOperation:
    domain: str
    action: str
    baseline_path: Path | None = None
    state_in_path: Path | None = None
    out_json: Path | None = None
    out_md: Path | None = None

    def validate(self) -> None:
        domain = self.domain.strip().lower()
        action = self.action.strip().lower()
        allowed = _CHECK_AUX_DOMAIN_ACTIONS.get(domain)
        if allowed is None:
            raise typer.BadParameter(
                "aux_operation domain must be one of: obsolescence, annotation-drift, ambiguity."
            )
        if action not in allowed:
            raise typer.BadParameter(
                f"aux_operation action '{action}' is not valid for domain '{domain}'."
            )
        if action in {"delta", "baseline-write"} and self.baseline_path is None:
            raise typer.BadParameter(
                "aux_operation requires baseline_path for delta and baseline-write actions."
            )

    def to_payload(self) -> JSONObject:
        self.validate()
        return {
            "domain": self.domain.strip().lower(),
            "action": self.action.strip().lower(),
            "baseline_path": str(self.baseline_path)
            if self.baseline_path is not None
            else None,
            "state_in": str(self.state_in_path)
            if self.state_in_path is not None
            else None,
            "out_json": str(self.out_json) if self.out_json is not None else None,
            "out_md": str(self.out_md) if self.out_md is not None else None,
        }


@dataclass(frozen=True)
class DataflowFilterBundle:
    ignore_params_csv: str | None
    transparent_decorators_csv: str | None

    def to_payload_lists(
        self,
        *,
        split_csv_fn: SplitCsvFn = split_csv,
    ) -> tuple[list[str] | None, list[str] | None]:
        ignore_list = (
            split_csv_fn(self.ignore_params_csv)
            if self.ignore_params_csv is not None
            else None
        )
        transparent_list = (
            split_csv_fn(self.transparent_decorators_csv)
            if self.transparent_decorators_csv is not None
            else None
        )
        return ignore_list, transparent_list


@dataclass(frozen=True)
class CheckDeltaOptions:
    emit_test_obsolescence_state: bool
    test_obsolescence_state: Path | None
    emit_test_obsolescence_delta: bool
    test_annotation_drift_state: Path | None
    emit_test_annotation_drift_delta: bool
    write_test_annotation_drift_baseline: bool
    write_test_obsolescence_baseline: bool
    emit_ambiguity_delta: bool
    emit_ambiguity_state: bool
    ambiguity_state: Path | None
    write_ambiguity_baseline: bool
    semantic_coverage_mapping: Path | None = None

    def validate(self) -> None:
        if self.emit_test_obsolescence_delta and self.write_test_obsolescence_baseline:
            raise typer.BadParameter(
                "Use --emit-test-obsolescence-delta or --write-test-obsolescence-baseline, not both."
            )
        if self.emit_test_obsolescence_state and self.test_obsolescence_state is not None:
            raise typer.BadParameter(
                "Use --emit-test-obsolescence-state or --test-obsolescence-state, not both."
            )
        if (
            self.emit_test_annotation_drift_delta
            and self.write_test_annotation_drift_baseline
        ):
            raise typer.BadParameter(
                "Use --emit-test-annotation-drift-delta or --write-test-annotation-drift-baseline, not both."
            )
        if self.emit_ambiguity_delta and self.write_ambiguity_baseline:
            raise typer.BadParameter(
                "Use --emit-ambiguity-delta or --write-ambiguity-baseline, not both."
            )
        if self.emit_ambiguity_state and self.ambiguity_state is not None:
            raise typer.BadParameter(
                "Use --emit-ambiguity-state or --ambiguity-state, not both."
            )

    def to_payload(self) -> JSONObject:
        return {
            "emit_test_obsolescence_state": self.emit_test_obsolescence_state,
            "test_obsolescence_state": str(self.test_obsolescence_state)
            if self.test_obsolescence_state is not None
            else None,
            "emit_test_obsolescence_delta": self.emit_test_obsolescence_delta,
            "test_annotation_drift_state": str(self.test_annotation_drift_state)
            if self.test_annotation_drift_state is not None
            else None,
            "emit_test_annotation_drift_delta": self.emit_test_annotation_drift_delta,
            "write_test_annotation_drift_baseline": self.write_test_annotation_drift_baseline,
            "semantic_coverage_mapping": str(self.semantic_coverage_mapping)
            if self.semantic_coverage_mapping is not None
            else None,
            "write_test_obsolescence_baseline": self.write_test_obsolescence_baseline,
            "emit_ambiguity_delta": self.emit_ambiguity_delta,
            "emit_ambiguity_state": self.emit_ambiguity_state,
            "ambiguity_state": str(self.ambiguity_state)
            if self.ambiguity_state is not None
            else None,
            "write_ambiguity_baseline": self.write_ambiguity_baseline,
        }


@dataclass(frozen=True)
class DataflowPayloadCommonOptions:
    paths: list[Path]
    root: Path
    config: Path | None
    report: Path | None
    fail_on_violations: bool
    fail_on_type_ambiguities: bool
    baseline: Path | None
    baseline_write: bool | None
    decision_snapshot: Path | None
    exclude: list[str] | None
    filter_bundle: DataflowFilterBundle
    allow_external: bool | None
    strictness: str | None
    lint: bool
    deadline_profile: bool = True
    aspf_trace_json: Path | None = None
    aspf_import_trace: list[Path] | None = None
    aspf_equivalence_against: list[Path] | None = None
    aspf_opportunities_json: Path | None = None
    aspf_state_json: Path | None = None
    aspf_import_state: list[Path] | None = None
    aspf_delta_jsonl: Path | None = None
    aspf_action_plan_json: Path | None = None
    aspf_action_plan_md: Path | None = None
    aspf_semantic_surface: list[str] | None = None


def delta_bundle_artifact_flags() -> CheckArtifactFlags:
    return CheckArtifactFlags(
        emit_test_obsolescence=False,
        emit_test_evidence_suggestions=False,
        emit_call_clusters=False,
        emit_call_cluster_consolidation=False,
        emit_test_annotation_drift=True,
        emit_semantic_coverage_map=False,
    )


def delta_bundle_delta_options() -> CheckDeltaOptions:
    return CheckDeltaOptions(
        emit_test_obsolescence_state=True,
        test_obsolescence_state=None,
        emit_test_obsolescence_delta=True,
        test_annotation_drift_state=None,
        emit_test_annotation_drift_delta=True,
        write_test_annotation_drift_baseline=False,
        write_test_obsolescence_baseline=False,
        emit_ambiguity_delta=True,
        emit_ambiguity_state=True,
        ambiguity_state=None,
        write_ambiguity_baseline=False,
        semantic_coverage_mapping=None,
    )


def build_dataflow_payload_common(
    *,
    options: DataflowPayloadCommonOptions,
    split_csv_entries_fn: SplitCsvEntriesFn = split_csv_entries,
    split_csv_fn: SplitCsvFn = split_csv,
) -> JSONObject:
    exclude_dirs = (
        split_csv_entries_fn(options.exclude)
        if options.exclude is not None
        else None
    )
    ignore_list, transparent_list = options.filter_bundle.to_payload_lists(
        split_csv_fn=split_csv_fn
    )
    return {
        "paths": [str(p) for p in options.paths],
        "root": str(options.root),
        "config": str(options.config) if options.config is not None else None,
        "report": str(options.report) if options.report is not None else None,
        "fail_on_violations": options.fail_on_violations,
        "fail_on_type_ambiguities": options.fail_on_type_ambiguities,
        "baseline": str(options.baseline) if options.baseline is not None else None,
        "baseline_write": options.baseline_write,
        "decision_snapshot": str(options.decision_snapshot)
        if options.decision_snapshot is not None
        else None,
        "exclude": exclude_dirs,
        "ignore_params": ignore_list,
        "transparent_decorators": transparent_list,
        "allow_external": options.allow_external,
        "strictness": options.strictness,
        "lint": options.lint,
        "deadline_profile": bool(options.deadline_profile),
        "aspf_trace_json": str(options.aspf_trace_json)
        if options.aspf_trace_json is not None
        else None,
        "aspf_import_trace": [str(path) for path in (options.aspf_import_trace or [])],
        "aspf_equivalence_against": [
            str(path) for path in (options.aspf_equivalence_against or [])
        ],
        "aspf_opportunities_json": str(options.aspf_opportunities_json)
        if options.aspf_opportunities_json is not None
        else None,
        "aspf_state_json": str(options.aspf_state_json)
        if options.aspf_state_json is not None
        else None,
        "aspf_import_state": [str(path) for path in (options.aspf_import_state or [])],
        "aspf_delta_jsonl": str(options.aspf_delta_jsonl)
        if options.aspf_delta_jsonl is not None
        else None,
        "aspf_action_plan_json": str(options.aspf_action_plan_json)
        if options.aspf_action_plan_json is not None
        else None,
        "aspf_action_plan_md": str(options.aspf_action_plan_md)
        if options.aspf_action_plan_md is not None
        else None,
        "aspf_semantic_surface": [
            str(surface) for surface in (options.aspf_semantic_surface or [])
        ],
    }


def build_check_payload(
    *,
    paths: list[Path] | None,
    report: Path | None,
    fail_on_violations: bool,
    root: Path,
    config: Path | None,
    baseline: Path | None,
    baseline_write: bool,
    decision_snapshot: Path | None,
    artifact_flags: CheckArtifactFlags,
    delta_options: CheckDeltaOptions,
    exclude: list[str] | None,
    filter_bundle: DataflowFilterBundle | None,
    allow_external: bool | None,
    strictness: str | None,
    fail_on_type_ambiguities: bool,
    lint: bool,
    analysis_tick_limit: int | None = None,
    aux_operation: CheckAuxOperation | None = None,
    aspf_trace_json: Path | None = None,
    aspf_import_trace: list[Path] | None = None,
    aspf_equivalence_against: list[Path] | None = None,
    aspf_opportunities_json: Path | None = None,
    aspf_state_json: Path | None = None,
    aspf_import_state: list[Path] | None = None,
    aspf_delta_jsonl: Path | None = None,
    aspf_action_plan_json: Path | None = None,
    aspf_action_plan_md: Path | None = None,
    aspf_semantic_surface: list[str] | None = None,
    split_csv_entries_fn: SplitCsvEntriesFn = split_csv_entries,
    split_csv_fn: SplitCsvFn = split_csv,
) -> JSONObject:
    resolved_filter_bundle = filter_bundle or DataflowFilterBundle(None, None)
    resolved_paths = paths or [Path(".")]
    delta_options.validate()
    baseline_write_value = bool(baseline is not None and baseline_write)
    payload = build_dataflow_payload_common(
        options=DataflowPayloadCommonOptions(
            paths=resolved_paths,
            root=root,
            config=config,
            report=report,
            fail_on_violations=fail_on_violations,
            fail_on_type_ambiguities=fail_on_type_ambiguities,
            baseline=baseline,
            baseline_write=baseline_write_value,
            decision_snapshot=decision_snapshot,
            exclude=exclude,
            filter_bundle=resolved_filter_bundle,
            allow_external=allow_external,
            strictness=strictness,
            lint=lint,
            aspf_trace_json=aspf_trace_json,
            aspf_import_trace=aspf_import_trace,
            aspf_equivalence_against=aspf_equivalence_against,
            aspf_opportunities_json=aspf_opportunities_json,
            aspf_state_json=aspf_state_json,
            aspf_import_state=aspf_import_state,
            aspf_delta_jsonl=aspf_delta_jsonl,
            aspf_action_plan_json=aspf_action_plan_json,
            aspf_action_plan_md=aspf_action_plan_md,
            aspf_semantic_surface=aspf_semantic_surface,
        ),
        split_csv_entries_fn=split_csv_entries_fn,
        split_csv_fn=split_csv_fn,
    )
    payload.update(
        {
            "emit_test_obsolescence": artifact_flags.emit_test_obsolescence,
            "emit_test_evidence_suggestions": artifact_flags.emit_test_evidence_suggestions,
            "emit_call_clusters": artifact_flags.emit_call_clusters,
            "emit_call_cluster_consolidation": artifact_flags.emit_call_cluster_consolidation,
            "emit_test_annotation_drift": artifact_flags.emit_test_annotation_drift,
            "emit_semantic_coverage_map": artifact_flags.emit_semantic_coverage_map,
            "type_audit": True if fail_on_type_ambiguities else None,
            "semantic_coverage_mapping": str(delta_options.semantic_coverage_mapping)
            if delta_options.semantic_coverage_mapping is not None
            else None,
            "analysis_tick_limit": int(analysis_tick_limit)
            if analysis_tick_limit is not None
            else None,
        }
    )
    payload.update(delta_options.to_payload())
    if aux_operation is not None:
        payload["aux_operation"] = aux_operation.to_payload()
    return payload
