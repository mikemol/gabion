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
    resume_checkpoint: Path | None
    emit_timeout_progress_report: bool
    resume_on_timeout: int
    deadline_profile: bool = True


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
        "resume_checkpoint": str(options.resume_checkpoint)
        if options.resume_checkpoint is not None
        else None,
        "emit_timeout_progress_report": bool(options.emit_timeout_progress_report),
        "resume_on_timeout": int(options.resume_on_timeout),
        "deadline_profile": bool(options.deadline_profile),
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
    resume_checkpoint: Path | None = None,
    emit_timeout_progress_report: bool = False,
    resume_on_timeout: int = 0,
    analysis_tick_limit: int | None = None,
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
            resume_checkpoint=resume_checkpoint,
            emit_timeout_progress_report=emit_timeout_progress_report,
            resume_on_timeout=resume_on_timeout,
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
    return payload

