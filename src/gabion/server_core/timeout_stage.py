from __future__ import annotations

from typing import Mapping, TypeVar

from gabion.runtime_shape_dispatch import str_optional
from gabion.server_core.stage_contracts import JSONObject, StageTimeoutResult, TimeoutCleanupHandler

_TimeoutCleanupContextT = TypeVar("_TimeoutCleanupContextT")


# gabion:decision_protocol
def timeout_classification_decision(*, progress_payload: JSONObject) -> str:
    timeout_classification = progress_payload.get("classification")
    normalized_timeout_classification = str_optional(timeout_classification)
    if normalized_timeout_classification:
        return normalized_timeout_classification
    return "timed_out_no_progress"


def run_timeout_stage(
    *,
    exc: BaseException,
    context: _TimeoutCleanupContextT,
    cleanup_handler: TimeoutCleanupHandler[_TimeoutCleanupContextT],
) -> StageTimeoutResult:
    return StageTimeoutResult(response=cleanup_handler(exc=exc, context=context))


def render_timeout_payload(*, base_payload: Mapping[str, object], classification: str) -> dict[str, object]:
    rendered = dict(base_payload)
    rendered["classification"] = classification
    return rendered
