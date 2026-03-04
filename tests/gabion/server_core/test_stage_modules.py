from __future__ import annotations

from pathlib import Path

from gabion.server_core.analysis_stage import run_analysis_stage
from gabion.server_core.ingress_stage import default_mode_selector, run_ingress_stage
from gabion.server_core.output_stage import run_output_stage
from gabion.server_core.timeout_stage import (
    render_timeout_payload,
    run_timeout_stage,
    timeout_classification_decision,
)


class _Outcome:
    def __init__(self) -> None:
        self.semantic_progress_cumulative = {"done": 1}
        self.latest_collection_progress = {"done": 1, "total": 1}
        self.last_collection_resume_payload = {"cursor": "next"}


def test_ingress_stage_normalizes_options_and_mode() -> None:
    stage = run_ingress_stage(
        payload={"raw": True, "aux_operation": {"domain": "x"}},
        root=Path("."),
        normalize_payload=lambda *, payload: {"normalized": True, **payload},
        parse_options=lambda *, payload, root: {"root": str(root), "keys": sorted(payload)},
        select_mode=default_mode_selector,
    )

    assert stage.payload["normalized"] is True
    assert stage.mode == "aux_operation"
    assert stage.options["root"] == "."


def test_analysis_stage_returns_structured_contract() -> None:
    stage = run_analysis_stage(
        context=object(),
        state=object(),
        collection_resume_payload={"resume": True},
        run_analysis_with_progress=lambda **_kwargs: _Outcome(),
    )

    assert stage.semantic_progress_cumulative == {"done": 1}
    assert stage.latest_collection_progress["total"] == 1


def test_output_stage_calls_primary_and_auxiliary_emitters() -> None:
    called: list[str] = []

    def _primary(*_args: object, **_kwargs: object) -> dict[str, object]:
        called.append("primary")
        return {"analysis_state": "succeeded", "phase_checkpoint_state": {"phase": "emit"}}

    def _aux(*_args: object, **_kwargs: object) -> None:
        called.append("aux")

    stage = run_output_stage(primary_output_emitter=_primary, auxiliary_output_emitter=_aux)

    assert called == ["primary", "aux"]
    assert stage.phase_checkpoint_state == {"phase": "emit"}


def test_timeout_stage_contracts() -> None:
    classification = timeout_classification_decision(progress_payload={})
    assert classification == "timed_out_no_progress"

    payload = render_timeout_payload(base_payload={"timeout": True}, classification=classification)
    assert payload["classification"] == "timed_out_no_progress"

    stage = run_timeout_stage(
        exc=TimeoutError("timed out"),
        context=object(),
        cleanup_handler=lambda **_kwargs: {"timeout": True},
    )
    assert stage.response == {"timeout": True}
