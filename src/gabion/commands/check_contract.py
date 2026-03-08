from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal, Mapping

import typer

from gabion.commands import aux_operation_contract
from gabion.json_types import JSONObject
from gabion.runtime_shape_dispatch import json_list_optional

SplitCsvEntriesFn = Callable[[list[str]], list[str]]
SplitCsvFn = Callable[[str], list[str]]
ParseLintEntryFn = Callable[[str], object | None]

LintResolutionKind = Literal[
    "provided_entries",
    "derive_from_lines",
    "empty",
]


def split_csv_entries(entries: list[str]) -> list[str]:
    merged: list[str] = []
    for entry in entries:
        merged.extend([part.strip() for part in entry.split(",") if part.strip()])
    return merged


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


PayloadRecord = tuple[str, object]
OptionalPayloadRecord = tuple[str, object | None]


def _record_is_present(record: OptionalPayloadRecord) -> bool:
    return record[1] is not None


def _payload_from_records(records: Iterable[PayloadRecord]) -> JSONObject:
    return dict(records)


def _path_payload(pairs: Iterable[tuple[str, Path | None]]) -> JSONObject:
    return _payload_from_records(
        map(
            lambda pair: (pair[0], str(pair[1])),
            filter(_record_is_present, pairs),
        )
    )


def _split_csv_payload(
    pairs: Iterable[tuple[str, str | None]],
    *,
    split_csv_fn: SplitCsvFn,
) -> JSONObject:
    return _payload_from_records(
        map(
            lambda pair: (pair[0], split_csv_fn(pair[1])),
            filter(_record_is_present, pairs),
        )
    )


def _split_csv_entries_payload(
    pairs: Iterable[tuple[str, list[str] | None]],
    *,
    split_csv_entries_fn: SplitCsvEntriesFn,
) -> JSONObject:
    return _payload_from_records(
        map(
            lambda pair: (pair[0], split_csv_entries_fn(pair[1])),
            filter(_record_is_present, pairs),
        )
    )


def _int_payload(pairs: Iterable[tuple[str, int | None]]) -> JSONObject:
    return _payload_from_records(
        map(
            lambda pair: (pair[0], int(pair[1])),
            filter(_record_is_present, pairs),
        )
    )


def _enabled_flag_payload(*, key: str, value: bool) -> JSONObject:
    match value:
        case True:
            return {key: True}
        case False:
            return {}


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
class CheckAuxOperation:
    domain: str
    action: str
    baseline_path: Path | None = None
    state_in_path: Path | None = None
    out_json: Path | None = None
    out_md: Path | None = None

    def validate(self) -> None:
        aux_operation_contract.validate_aux_operation_for_typer(
            domain=self.domain,
            action=self.action,
            baseline_path=self.baseline_path,
        )

    def to_payload(self) -> JSONObject:
        decision = aux_operation_contract.validate_aux_operation_for_typer(
            domain=self.domain,
            action=self.action,
            baseline_path=self.baseline_path,
        )
        return {
            "domain": decision.domain,
            "action": decision.action,
            **_path_payload(
                (
                    ("baseline_path", self.baseline_path),
                    ("state_in", self.state_in_path),
                    ("out_json", self.out_json),
                    ("out_md", self.out_md),
                )
            ),
        }


@dataclass(frozen=True)
class DataflowFilterBundle:
    ignore_params_csv: str | None
    transparent_decorators_csv: str | None

    def to_payload_lists(
        self,
        *,
        split_csv_fn: SplitCsvFn = split_csv,
    ) -> tuple[list[str], list[str]]:
        payload = _split_csv_payload(
            (
                ("ignore_params", self.ignore_params_csv),
                ("transparent_decorators", self.transparent_decorators_csv),
            ),
            split_csv_fn=split_csv_fn,
        )
        ignore_list = payload.get("ignore_params", [])
        transparent_list = payload.get("transparent_decorators", [])
        return ignore_list, transparent_list


@dataclass(frozen=True)
class LintEntriesDecision:
    """Decision-protocol surface for lint-entry trichotomy normalization."""

    kind: LintResolutionKind
    lint_lines: tuple[str, ...]
    lint_entries_payload: tuple[object, ...]

    @classmethod
    def from_response(cls, response: Mapping[str, object]) -> "LintEntriesDecision":
        lint_lines_raw = json_list_optional(response.get("lint_lines"))
        lint_lines = (
            tuple(str(line) for line in lint_lines_raw)
            if lint_lines_raw is not None
            else ()
        )
        lint_entries_raw = json_list_optional(response.get("lint_entries"))
        if lint_entries_raw is not None:
            return cls(
                kind="provided_entries",
                lint_lines=lint_lines,
                lint_entries_payload=tuple(lint_entries_raw),
            )
        if lint_lines:
            return cls(
                kind="derive_from_lines",
                lint_lines=lint_lines,
                lint_entries_payload=(),
            )
        return cls(
            kind="empty",
            lint_lines=(),
            lint_entries_payload=(),
        )

    def normalize_entries(self, *, parse_lint_entry_fn: ParseLintEntryFn) -> list[object]:
        if self.kind == "provided_entries":
            return list(self.lint_entries_payload)
        if self.kind == "derive_from_lines":
            entries: list[object] = []
            for line in self.lint_lines:
                parsed = parse_lint_entry_fn(line)
                if parsed is not None:
                    entries.append(parsed)
            return entries
        return []


@dataclass(frozen=True)
class CheckAuxMode:
    kind: Literal["off", "report", "state", "delta", "baseline-write"]
    state_path: Path | None = None

    def validate(self, *, domain: str, allow_report: bool) -> None:
        allowed = {"off", "state", "delta", "baseline-write"}
        if allow_report:
            allowed.add("report")
        if self.kind not in allowed:
            raise typer.BadParameter(
                f"{domain} mode must be one of: {', '.join(sorted(allowed))}."
            )
        if self.kind == "state" and self.state_path is not None:
            raise typer.BadParameter(
                f"{domain} state mode does not accept a state path override."
            )

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
class CheckDeltaOptions:
    obsolescence_mode: CheckAuxMode
    annotation_drift_mode: CheckAuxMode
    ambiguity_mode: CheckAuxMode
    semantic_coverage_mapping: Path | None = None

    def validate(self) -> None:
        self.obsolescence_mode.validate(domain="obsolescence", allow_report=True)
        self.annotation_drift_mode.validate(
            domain="annotation-drift", allow_report=True
        )
        self.ambiguity_mode.validate(domain="ambiguity", allow_report=False)
        return

    def to_payload(self) -> JSONObject:
        payload: JSONObject = {
            "obsolescence_mode": {
                "kind": self.obsolescence_mode.kind,
                **_path_payload((("state_path", self.obsolescence_mode.state_path),)),
            },
            "annotation_drift_mode": {
                "kind": self.annotation_drift_mode.kind,
                **_path_payload((("state_path", self.annotation_drift_mode.state_path),)),
            },
            "ambiguity_mode": {
                "kind": self.ambiguity_mode.kind,
                **_path_payload((("state_path", self.ambiguity_mode.state_path),)),
            },
        }
        payload.update(
            _path_payload((("semantic_coverage_mapping", self.semantic_coverage_mapping),))
        )
        return payload

    @property
    def emit_test_obsolescence_state(self) -> bool:
        return self.obsolescence_mode.emit_state

    @property
    def test_obsolescence_state(self) -> Path | None:
        return self.obsolescence_mode.state_path

    @property
    def emit_test_obsolescence_delta(self) -> bool:
        return self.obsolescence_mode.emit_delta

    @property
    def write_test_obsolescence_baseline(self) -> bool:
        return self.obsolescence_mode.write_baseline

    @property
    def test_annotation_drift_state(self) -> Path | None:
        return self.annotation_drift_mode.state_path

    @property
    def emit_test_annotation_drift_delta(self) -> bool:
        return self.annotation_drift_mode.emit_delta

    @property
    def write_test_annotation_drift_baseline(self) -> bool:
        return self.annotation_drift_mode.write_baseline

    @property
    def emit_ambiguity_delta(self) -> bool:
        return self.ambiguity_mode.emit_delta

    @property
    def emit_ambiguity_state(self) -> bool:
        return self.ambiguity_mode.emit_state

    @property
    def ambiguity_state(self) -> Path | None:
        return self.ambiguity_mode.state_path

    @property
    def write_ambiguity_baseline(self) -> bool:
        return self.ambiguity_mode.write_baseline


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
    language: str | None = None
    ingest_profile: str | None = None
    deadline_profile: bool = True
    aspf_trace_json: Path | None = None
    aspf_import_trace: list[Path] | None = None
    aspf_equivalence_against: list[Path] | None = None
    aspf_opportunities_json: Path | None = None
    aspf_state_json: Path | None = None
    aspf_import_state: list[Path] | None = None
    aspf_delta_jsonl: Path | None = None
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
        obsolescence_mode=CheckAuxMode(kind="delta"),
        annotation_drift_mode=CheckAuxMode(kind="delta"),
        ambiguity_mode=CheckAuxMode(kind="delta"),
        semantic_coverage_mapping=None,
    )


def build_dataflow_payload_common(
    *,
    options: DataflowPayloadCommonOptions,
    split_csv_entries_fn: SplitCsvEntriesFn = split_csv_entries,
    split_csv_fn: SplitCsvFn = split_csv,
) -> JSONObject:
    exclude_payload = _split_csv_entries_payload(
        (("exclude", options.exclude),),
        split_csv_entries_fn=split_csv_entries_fn,
    )
    ignore_list, transparent_list = options.filter_bundle.to_payload_lists(split_csv_fn=split_csv_fn)
    return {
        "paths": [str(p) for p in options.paths],
        "root": str(options.root),
        **_path_payload(
            (
                ("config", options.config),
                ("report", options.report),
                ("baseline", options.baseline),
                ("decision_snapshot", options.decision_snapshot),
                ("aspf_trace_json", options.aspf_trace_json),
                ("aspf_opportunities_json", options.aspf_opportunities_json),
                ("aspf_state_json", options.aspf_state_json),
                ("aspf_delta_jsonl", options.aspf_delta_jsonl),
            )
        ),
        **_payload_from_records(
            filter(
                _record_is_present,
                (
                    ("baseline_write", options.baseline_write),
                    ("allow_external", options.allow_external),
                    ("strictness", options.strictness),
                    ("language", options.language),
                    ("ingest_profile", options.ingest_profile),
                ),
            )
        ),
        "fail_on_violations": options.fail_on_violations,
        "fail_on_type_ambiguities": options.fail_on_type_ambiguities,
        "exclude": exclude_payload.get("exclude", []),
        "ignore_params": ignore_list,
        "transparent_decorators": transparent_list,
        "lint": options.lint,
        "deadline_profile": bool(options.deadline_profile),
        "aspf_import_trace": [str(path) for path in (options.aspf_import_trace or [])],
        "aspf_equivalence_against": [
            str(path) for path in (options.aspf_equivalence_against or [])
        ],
        "aspf_import_state": [str(path) for path in (options.aspf_import_state or [])],
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
            **_enabled_flag_payload(
                key="type_audit",
                value=fail_on_type_ambiguities,
            ),
            **_path_payload((("semantic_coverage_mapping", delta_options.semantic_coverage_mapping),)),
            **_int_payload((("analysis_tick_limit", analysis_tick_limit),)),
        }
    )
    payload.update(delta_options.to_payload())
    if aux_operation is not None:
        payload["aux_operation"] = aux_operation.to_payload()
    return payload
