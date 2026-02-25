# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from gabion.analysis import test_obsolescence_delta
from gabion.analysis.baseline_io import (
    attach_spec_metadata,
    load_json,
    parse_spec_metadata,
    parse_version,
)
from gabion.analysis.projection_registry import (
    TEST_OBSOLESCENCE_STATE_SPEC,
)
from gabion.json_types import JSONValue
from gabion.analysis.timeout_context import check_deadline

STATE_VERSION = 1


@dataclass(frozen=True)
class ObsolescenceState:
    candidates: list[dict[str, JSONValue]]
    baseline: test_obsolescence_delta.ObsolescenceBaseline
    baseline_payload: dict[str, JSONValue]
    generated_by_spec_id: str
    generated_by_spec: dict[str, JSONValue]


def build_state_payload(
    evidence_by_test: Mapping[str, Iterable[object]],
    status_by_test: Mapping[str, str],
    candidates: Iterable[Mapping[str, object]],
    summary_counts: Mapping[str, int],
    *,
    active_tests: Iterable[str] | None = None,
    active_summary: Mapping[str, int] | None = None,
) -> dict[str, JSONValue]:
    # dataflow-bundle: evidence_by_test, status_by_test, candidates, summary_counts
    baseline_payload = test_obsolescence_delta.build_baseline_payload(
        evidence_by_test,
        status_by_test,
        candidates,
        summary_counts,
        active_tests=active_tests,
        active_summary=active_summary,
    )
    payload: dict[str, JSONValue] = {
        "version": STATE_VERSION,
        "baseline": baseline_payload,
        "candidates": list(candidates),
    }
    return attach_spec_metadata(payload, spec=TEST_OBSOLESCENCE_STATE_SPEC)


def parse_state_payload(payload: Mapping[str, JSONValue]) -> ObsolescenceState:
    check_deadline()
    parse_version(
        payload,
        expected=STATE_VERSION,
        error_context="test obsolescence state",
    )
    baseline_payload = payload.get("baseline", {})
    if not isinstance(baseline_payload, Mapping):
        raise ValueError("Test obsolescence state baseline must be an object.")
    baseline = test_obsolescence_delta.parse_baseline_payload(baseline_payload)
    candidates_payload = payload.get("candidates", [])
    candidates: list[dict[str, JSONValue]] = []
    if isinstance(candidates_payload, list):
        for entry in candidates_payload:
            if not isinstance(entry, Mapping):
                continue
            candidates.append({str(k): entry[k] for k in entry})
    spec_id, spec = parse_spec_metadata(payload)
    return ObsolescenceState(
        candidates=candidates,
        baseline=baseline,
        baseline_payload={str(k): baseline_payload[k] for k in baseline_payload},
        generated_by_spec_id=spec_id,
        generated_by_spec=spec,
    )


def load_state(path: str) -> ObsolescenceState:
    return parse_state_payload(load_json(path))
