from __future__ import annotations

from dataclasses import dataclass

from gabion.server_core.command_contract import (
    ExecutionPayloadOptionsContract,
    IngressStageMode,
    ProgressTraceStateContract,
)
from gabion.server_core.ingress_stage import default_mode_selector
from gabion.server_core.output_stage import run_output_stage
from gabion.server_core.stage_contracts import AuxiliaryOutputRequest, PrimaryOutputRequest


@dataclass(frozen=True)
class _Options:
    emit_phase_timeline: bool = False
    progress_heartbeat_seconds: float = 1.0


@dataclass(frozen=True)
class _TraceState:
    trace_id: str


# gabion:behavior primary=desired
def test_default_mode_selector_returns_stage_mode_enum() -> None:
    mode = default_mode_selector(payload={"aux_operation": {"domain": "x"}}, options=_Options())
    assert mode is IngressStageMode.AUX_OPERATION


# gabion:behavior primary=desired
def test_output_stage_uses_typed_emission_requests() -> None:
    events: list[str] = []

    def _emit_primary() -> dict[str, object]:
        events.append("primary")
        return {"phase_checkpoint_state": {"phase": "emit"}}

    def _emit_aux() -> None:
        events.append("aux")

    result = run_output_stage(
        primary_request=PrimaryOutputRequest(emit=_emit_primary),
        auxiliary_request=AuxiliaryOutputRequest(emit=_emit_aux),
    )

    assert result.phase_checkpoint_state == {"phase": "emit"}
    assert events == ["primary", "aux"]


# gabion:behavior primary=desired
def test_contract_marker_protocols_accept_dataclass_carriers() -> None:
    options: ExecutionPayloadOptionsContract = _Options()
    assert options.progress_heartbeat_seconds == 1.0

    trace: ProgressTraceStateContract = _TraceState(trace_id="t-1")
    assert isinstance(trace, _TraceState)
