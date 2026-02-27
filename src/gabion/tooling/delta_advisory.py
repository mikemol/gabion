# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping

from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import sort_once
import json

from gabion.runtime import env_policy, json_io
from gabion.tooling import advisory_evidence
from gabion.tooling.deadline_runtime import DeadlineBudget, deadline_scope_from_lsp_env

AdvisoryId = Literal["obsolescence", "annotation_drift", "ambiguity", "docflow"]

OBSOLESCENCE_ENV_FLAG = "GABION_GATE_UNMAPPED_DELTA"
ANNOTATION_DRIFT_ENV_FLAG = "GABION_GATE_ORPHANED_DELTA"
AMBIGUITY_ENV_FLAG = "GABION_GATE_AMBIGUITY_DELTA"

_DEFAULT_ADVISORY_TIMEOUT_TICKS = 120_000
_DEFAULT_ADVISORY_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_ADVISORY_TIMEOUT_BUDGET = DeadlineBudget(
    ticks=_DEFAULT_ADVISORY_TIMEOUT_TICKS,
    tick_ns=_DEFAULT_ADVISORY_TIMEOUT_TICK_NS,
)


@dataclass(frozen=True)
class AdvisoryMetric:
    key: str
    baseline: int
    current: int
    delta: int


@dataclass(frozen=True)
class AdvisoryNormalizedSummary:
    heading: str
    metrics: tuple[AdvisoryMetric, ...]


@dataclass(frozen=True)
class AdvisoryConfig:
    id: AdvisoryId
    delta_path: Path
    artifact_path: Path
    missing_message: str
    error_prefix: str
    summary_builder: Callable[[Mapping[str, object]], AdvisoryNormalizedSummary]
    env_flag: str | None = None
    skip_message: str | None = None


def _deadline_scope():
    return deadline_scope_from_lsp_env(
        default_budget=_DEFAULT_ADVISORY_TIMEOUT_BUDGET,
    )


def _enabled(env_flag: str) -> bool:
    return env_policy.env_enabled_flag(env_flag)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _count(value: object) -> int:
    return int(value) if isinstance(value, int) else 0


def _metric_entry(
    key: str,
    baseline: Mapping[str, object],
    current: Mapping[str, object],
    delta: Mapping[str, object],
) -> AdvisoryMetric:
    return AdvisoryMetric(
        key=key,
        baseline=_count(baseline.get(key)),
        current=_count(current.get(key)),
        delta=_count(delta.get(key)),
    )


def _render_summary(summary: AdvisoryNormalizedSummary, print_fn: Callable[[str], None]) -> None:
    print_fn(summary.heading)
    for entry in summary.metrics:
        check_deadline()
        print_fn(f"- {entry.key}: {entry.baseline} -> {entry.current} ({entry.delta})")


def _obsolescence_summary(payload: Mapping[str, object]) -> AdvisoryNormalizedSummary:
    summary = _mapping(payload.get("summary"))
    counts = _mapping(summary.get("counts"))
    baseline = _mapping(counts.get("baseline"))
    current = _mapping(counts.get("current"))
    delta = _mapping(counts.get("delta"))
    opaque = _mapping(summary.get("opaque_evidence"))
    keys = [
        "redundant_by_evidence",
        "equivalent_witness",
        "obsolete_candidate",
        "unmapped",
    ]
    metrics = [
        _metric_entry(key, baseline, current, delta)
        for key in keys
    ]
    metrics.append(
        AdvisoryMetric(
            key="opaque_evidence_count",
            baseline=_count(opaque.get("baseline")),
            current=_count(opaque.get("current")),
            delta=_count(opaque.get("delta")),
        )
    )
    return AdvisoryNormalizedSummary(
        heading="Test obsolescence delta summary (advisory):",
        metrics=tuple(metrics),
    )


def _annotation_drift_summary(payload: Mapping[str, object]) -> AdvisoryNormalizedSummary:
    summary = _mapping(payload.get("summary"))
    baseline = _mapping(summary.get("baseline"))
    current = _mapping(summary.get("current"))
    delta = _mapping(summary.get("delta"))
    keys = sort_once(
        {*baseline.keys(), *current.keys(), *delta.keys()},
        source="gabion.tooling.delta_advisory.annotation_summary_keys",
    )
    return AdvisoryNormalizedSummary(
        heading="Annotation drift delta summary (advisory):",
        metrics=tuple(_metric_entry(str(key), baseline, current, delta) for key in keys),
    )


def _ambiguity_summary(payload: Mapping[str, object]) -> AdvisoryNormalizedSummary:
    summary = _mapping(payload.get("summary"))
    total = _mapping(summary.get("total"))
    by_kind = _mapping(summary.get("by_kind"))
    baseline = _mapping(by_kind.get("baseline"))
    current = _mapping(by_kind.get("current"))
    delta = _mapping(by_kind.get("delta"))
    keys = sort_once(
        {*baseline.keys(), *current.keys(), *delta.keys()},
        source="gabion.tooling.delta_advisory.ambiguity_by_kind_keys",
    )
    by_kind_metrics = [_metric_entry(str(key), baseline, current, delta) for key in keys]
    return AdvisoryNormalizedSummary(
        heading="Ambiguity delta summary (advisory):",
        metrics=(
            AdvisoryMetric(
                key="total",
                baseline=_count(total.get("baseline")),
                current=_count(total.get("current")),
                delta=_count(total.get("delta")),
            ),
            *by_kind_metrics,
        ),
    )


def _docflow_summary(payload: Mapping[str, object]) -> AdvisoryNormalizedSummary:
    summary = _mapping(payload.get("summary"))
    baseline = _mapping(summary.get("baseline"))
    current = _mapping(summary.get("current"))
    delta = _mapping(summary.get("delta"))
    keys = ["compliant", "contradicts", "excess", "proposed"]
    return AdvisoryNormalizedSummary(
        heading="Docflow compliance delta summary (advisory):",
        metrics=tuple(_metric_entry(key, baseline, current, delta) for key in keys),
    )


def _evidence_payload(
    *,
    config: AdvisoryConfig,
    normalized: AdvisoryNormalizedSummary,
    timestamp: str,
    threshold_class: str = "telemetry_non_blocking",
) -> advisory_evidence.AdvisoryEvidencePayload:
    entries = tuple(
        advisory_evidence.AdvisoryEvidenceEntry(
            domain=config.id,
            key=entry.key,
            baseline=entry.baseline,
            current=entry.current,
            delta=entry.delta,
            threshold_class=threshold_class,
            message=f"{config.id}:{entry.key} delta={entry.delta}",
            timestamp=timestamp,
        )
        for entry in normalized.metrics
    )
    return advisory_evidence.AdvisoryEvidencePayload(
        domain=config.id,
        source_delta_path=str(config.delta_path),
        generated_at=timestamp,
        entries=entries,
    )


# gabion:boundary_normalization
def _write_aggregate_with_domain(payload: advisory_evidence.AdvisoryEvidencePayload) -> None:
    existing = json_io.load_json_object_path(advisory_evidence.DEFAULT_ADVISORY_AGGREGATE_PATH)
    advisories_raw = _mapping(existing.get("advisories"))
    domain_payloads: dict[str, advisory_evidence.AdvisoryEvidencePayload] = {}
    for raw_domain, raw_payload in advisories_raw.items():
        if raw_domain == payload.domain:
            continue
        item = _mapping(raw_payload)
        entries_raw = item.get("entries")
        if not isinstance(entries_raw, list):
            continue
        entries = tuple(
            advisory_evidence.AdvisoryEvidenceEntry(
                domain=str(raw_domain),
                key=str(_mapping(raw_entry).get("key", "")),
                baseline=_count(_mapping(raw_entry).get("baseline")),
                current=_count(_mapping(raw_entry).get("current")),
                delta=_count(_mapping(raw_entry).get("delta")),
                threshold_class=str(_mapping(raw_entry).get("threshold_class", "telemetry_non_blocking")),
                message=str(_mapping(raw_entry).get("message", "")),
                timestamp=str(_mapping(raw_entry).get("timestamp", payload.generated_at)),
            )
            for raw_entry in entries_raw
            if isinstance(raw_entry, Mapping)
        )
        domain_payloads[str(raw_domain)] = advisory_evidence.AdvisoryEvidencePayload(
            domain=str(raw_domain),
            source_delta_path=str(item.get("source_delta_path", "")),
            generated_at=str(item.get("generated_at", payload.generated_at)),
            entries=entries,
        )
    domain_payloads[payload.domain] = payload
    advisory_evidence.write_aggregate(
        domain_payloads,
        generated_at=payload.generated_at,
    )


_ADVISORY_CONFIGS: dict[AdvisoryId, AdvisoryConfig] = {
    "obsolescence": AdvisoryConfig(
        id="obsolescence",
        delta_path=Path("artifacts/out/test_obsolescence_delta.json"),
        artifact_path=Path("artifacts/out/obsolescence_advisory.json"),
        missing_message="Test obsolescence delta missing (advisory).",
        error_prefix="Test obsolescence delta advisory error",
        summary_builder=_obsolescence_summary,
        env_flag=OBSOLESCENCE_ENV_FLAG,
        skip_message=(
            "Test obsolescence delta advisory skipped; "
            f"{OBSOLESCENCE_ENV_FLAG}=1 enables the gate."
        ),
    ),
    "annotation_drift": AdvisoryConfig(
        id="annotation_drift",
        delta_path=Path("artifacts/out/test_annotation_drift_delta.json"),
        artifact_path=Path("artifacts/out/annotation_drift_advisory.json"),
        missing_message="Annotation drift delta missing (advisory).",
        error_prefix="Annotation drift delta advisory error",
        summary_builder=_annotation_drift_summary,
        env_flag=ANNOTATION_DRIFT_ENV_FLAG,
        skip_message=(
            "Annotation drift delta advisory skipped; "
            f"{ANNOTATION_DRIFT_ENV_FLAG}=1 enables the gate."
        ),
    ),
    "ambiguity": AdvisoryConfig(
        id="ambiguity",
        delta_path=Path("artifacts/out/ambiguity_delta.json"),
        artifact_path=Path("artifacts/out/ambiguity_advisory.json"),
        missing_message="Ambiguity delta missing (advisory).",
        error_prefix="Ambiguity delta advisory error",
        summary_builder=_ambiguity_summary,
        env_flag=AMBIGUITY_ENV_FLAG,
        skip_message=(
            "Ambiguity delta advisory skipped; "
            f"{AMBIGUITY_ENV_FLAG}=1 enables the gate."
        ),
    ),
    "docflow": AdvisoryConfig(
        id="docflow",
        delta_path=Path("artifacts/out/docflow_compliance_delta.json"),
        artifact_path=Path("artifacts/out/docflow_advisory.json"),
        missing_message="Docflow compliance delta missing (advisory).",
        error_prefix="Docflow compliance delta advisory error",
        summary_builder=_docflow_summary,
    ),
}


def main_for_advisory(
    advisory_id: AdvisoryId,
    *,
    delta_path: Path | None = None,
    print_fn: Callable[[str], None] = print,
    timestamp_fn: Callable[[], str] = advisory_evidence.advisory_timestamp,
) -> int:
    config = _ADVISORY_CONFIGS[advisory_id]
    with _deadline_scope():
        try:
            if isinstance(config.env_flag, str) and config.env_flag:
                if _enabled(config.env_flag):
                    if isinstance(config.skip_message, str) and config.skip_message:
                        print_fn(config.skip_message)
                    return 0
            target_path = delta_path if isinstance(delta_path, Path) else config.delta_path
            if not target_path.exists():
                print_fn(config.missing_message)
                return 0
            payload = json.loads(target_path.read_text(encoding="utf-8"))
            normalized = config.summary_builder(_mapping(payload))
            _render_summary(normalized, print_fn)
            timestamp = timestamp_fn()
            evidence_payload = _evidence_payload(
                config=config,
                normalized=normalized,
                timestamp=timestamp,
            )
            advisory_evidence.write_payload(config.artifact_path, evidence_payload)
            _write_aggregate_with_domain(evidence_payload)
        except Exception as exc:  # advisory only; keep CI green
            print_fn(f"{config.error_prefix}: {exc}")
    return 0


def telemetry_main(*, print_fn: Callable[[str], None] = print) -> int:
    for advisory_id in sort_once(
        _ADVISORY_CONFIGS.keys(),
        source="gabion.tooling.delta_advisory.telemetry_main",
    ):
        check_deadline()
        main_for_advisory(advisory_id, print_fn=print_fn)
    return 0


def obsolescence_main(*, delta_path: Path | None = None) -> int:
    return main_for_advisory("obsolescence", delta_path=delta_path)


def annotation_drift_main(*, delta_path: Path | None = None) -> int:
    return main_for_advisory("annotation_drift", delta_path=delta_path)


def ambiguity_main(*, delta_path: Path | None = None) -> int:
    return main_for_advisory("ambiguity", delta_path=delta_path)


def docflow_main(*, delta_path: Path | None = None) -> int:
    return main_for_advisory("docflow", delta_path=delta_path)
