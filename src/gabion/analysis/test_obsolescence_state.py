from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from gabion.analysis import test_obsolescence_delta
from gabion.analysis.projection_registry import (
    TEST_OBSOLESCENCE_STATE_SPEC,
    spec_metadata_payload,
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
) -> dict[str, JSONValue]:
    # dataflow-bundle: evidence_by_test, status_by_test, candidates, summary_counts
    baseline_payload = test_obsolescence_delta.build_baseline_payload(
        evidence_by_test,
        status_by_test,
        candidates,
        summary_counts,
    )
    payload: dict[str, JSONValue] = {
        "version": STATE_VERSION,
        "baseline": baseline_payload,
        "candidates": list(candidates),
    }
    payload.update(spec_metadata_payload(TEST_OBSOLESCENCE_STATE_SPEC))
    return payload


def parse_state_payload(payload: Mapping[str, JSONValue]) -> ObsolescenceState:
    check_deadline()
    version = payload.get("version", STATE_VERSION)
    try:
        version_value = int(version) if version is not None else STATE_VERSION
    except (TypeError, ValueError):
        version_value = -1
    if version_value != STATE_VERSION:
        raise ValueError(
            "Unsupported test obsolescence state "
            f"version={version!r}; expected {STATE_VERSION}"
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
    spec_id = str(payload.get("generated_by_spec_id", "") or "")
    spec_payload = payload.get("generated_by_spec", {})
    spec: dict[str, JSONValue] = {}
    if isinstance(spec_payload, Mapping):
        spec = {str(key): spec_payload[key] for key in spec_payload}
    return ObsolescenceState(
        candidates=candidates,
        baseline=baseline,
        baseline_payload={str(k): baseline_payload[k] for k in baseline_payload},
        generated_by_spec_id=spec_id,
        generated_by_spec=spec,
    )


def load_state(path: str) -> ObsolescenceState:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Test obsolescence state must be a JSON object.")
    return parse_state_payload(payload)
