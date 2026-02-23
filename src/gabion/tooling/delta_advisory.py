# gabion:decision_protocol_module
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping

from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import sort_once
from gabion.runtime import env_policy
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
class AdvisoryConfig:
    id: AdvisoryId
    delta_path: Path
    missing_message: str
    error_prefix: str
    summary_renderer: Callable[[Mapping[str, object], Callable[[str], None]], None]
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


def _obsolescence_summary(
    payload: Mapping[str, object],
    print_fn: Callable[[str], None],
) -> None:
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
    print_fn("Test obsolescence delta summary (advisory):")
    for key in keys:
        check_deadline()
        print_fn(
            f"- {key}: {baseline.get(key, 0)} -> {current.get(key, 0)} ({delta.get(key, 0)})"
        )
    print_fn(
        "- opaque_evidence_count: "
        f"{opaque.get('baseline', 0)} -> {opaque.get('current', 0)} ({opaque.get('delta', 0)})"
    )


def _annotation_drift_summary(
    payload: Mapping[str, object],
    print_fn: Callable[[str], None],
) -> None:
    summary = _mapping(payload.get("summary"))
    baseline = _mapping(summary.get("baseline"))
    current = _mapping(summary.get("current"))
    delta = _mapping(summary.get("delta"))
    keys = sort_once(
        {*baseline.keys(), *current.keys(), *delta.keys()},
        source="gabion.tooling.delta_advisory.annotation_summary_keys",
    )
    print_fn("Annotation drift delta summary (advisory):")
    for key in keys:
        check_deadline()
        print_fn(
            f"- {key}: {baseline.get(key, 0)} -> {current.get(key, 0)} ({delta.get(key, 0)})"
        )


def _ambiguity_summary(
    payload: Mapping[str, object],
    print_fn: Callable[[str], None],
) -> None:
    summary = _mapping(payload.get("summary"))
    total = _mapping(summary.get("total"))
    by_kind = _mapping(summary.get("by_kind"))
    baseline = _mapping(by_kind.get("baseline"))
    current = _mapping(by_kind.get("current"))
    delta = _mapping(by_kind.get("delta"))
    print_fn("Ambiguity delta summary (advisory):")
    print_fn(
        "- total: "
        f"{total.get('baseline', 0)} -> {total.get('current', 0)} ({total.get('delta', 0)})"
    )
    keys = sort_once(
        {*baseline.keys(), *current.keys(), *delta.keys()},
        source="gabion.tooling.delta_advisory.ambiguity_by_kind_keys",
    )
    for key in keys:
        check_deadline()
        print_fn(
            f"- {key}: {baseline.get(key, 0)} -> {current.get(key, 0)} ({delta.get(key, 0)})"
        )


def _docflow_summary(
    payload: Mapping[str, object],
    print_fn: Callable[[str], None],
) -> None:
    summary = _mapping(payload.get("summary"))
    baseline = _mapping(summary.get("baseline"))
    current = _mapping(summary.get("current"))
    delta = _mapping(summary.get("delta"))
    keys = ["compliant", "contradicts", "excess", "proposed"]
    print_fn("Docflow compliance delta summary (advisory):")
    for key in keys:
        check_deadline()
        print_fn(
            f"- {key}: {baseline.get(key, 0)} -> {current.get(key, 0)} ({delta.get(key, 0)})"
        )


_ADVISORY_CONFIGS: dict[AdvisoryId, AdvisoryConfig] = {
    "obsolescence": AdvisoryConfig(
        id="obsolescence",
        delta_path=Path("artifacts/out/test_obsolescence_delta.json"),
        missing_message="Test obsolescence delta missing (advisory).",
        error_prefix="Test obsolescence delta advisory error",
        summary_renderer=_obsolescence_summary,
        env_flag=OBSOLESCENCE_ENV_FLAG,
        skip_message=(
            "Test obsolescence delta advisory skipped; "
            f"{OBSOLESCENCE_ENV_FLAG}=1 enables the gate."
        ),
    ),
    "annotation_drift": AdvisoryConfig(
        id="annotation_drift",
        delta_path=Path("artifacts/out/test_annotation_drift_delta.json"),
        missing_message="Annotation drift delta missing (advisory).",
        error_prefix="Annotation drift delta advisory error",
        summary_renderer=_annotation_drift_summary,
        env_flag=ANNOTATION_DRIFT_ENV_FLAG,
        skip_message=(
            "Annotation drift delta advisory skipped; "
            f"{ANNOTATION_DRIFT_ENV_FLAG}=1 enables the gate."
        ),
    ),
    "ambiguity": AdvisoryConfig(
        id="ambiguity",
        delta_path=Path("artifacts/out/ambiguity_delta.json"),
        missing_message="Ambiguity delta missing (advisory).",
        error_prefix="Ambiguity delta advisory error",
        summary_renderer=_ambiguity_summary,
        env_flag=AMBIGUITY_ENV_FLAG,
        skip_message=(
            "Ambiguity delta advisory skipped; "
            f"{AMBIGUITY_ENV_FLAG}=1 enables the gate."
        ),
    ),
    "docflow": AdvisoryConfig(
        id="docflow",
        delta_path=Path("artifacts/out/docflow_compliance_delta.json"),
        missing_message="Docflow compliance delta missing (advisory).",
        error_prefix="Docflow compliance delta advisory error",
        summary_renderer=_docflow_summary,
    ),
}


def main_for_advisory(
    advisory_id: AdvisoryId,
    *,
    delta_path: Path | None = None,
    print_fn: Callable[[str], None] = print,
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
            config.summary_renderer(_mapping(payload), print_fn)
        except Exception as exc:  # advisory only; keep CI green
            print_fn(f"{config.error_prefix}: {exc}")
    return 0


def obsolescence_main(*, delta_path: Path | None = None) -> int:
    return main_for_advisory("obsolescence", delta_path=delta_path)


def annotation_drift_main(*, delta_path: Path | None = None) -> int:
    return main_for_advisory("annotation_drift", delta_path=delta_path)


def ambiguity_main(*, delta_path: Path | None = None) -> int:
    return main_for_advisory("ambiguity", delta_path=delta_path)


def docflow_main(*, delta_path: Path | None = None) -> int:
    return main_for_advisory("docflow", delta_path=delta_path)
