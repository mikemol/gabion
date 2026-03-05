from __future__ import annotations

from gabion.server_core.stage_contracts import (
    AuxiliaryOutputRequest,
    StageOutputResult,
    PrimaryOutputRequest,
)


def run_output_stage(
    *,
    primary_request: PrimaryOutputRequest,
    auxiliary_request: AuxiliaryOutputRequest,
) -> StageOutputResult:
    rendered = primary_request.emit()
    auxiliary_request.emit()
    checkpoint_state = rendered.get("phase_checkpoint_state", {})
    return StageOutputResult(response=rendered, phase_checkpoint_state=checkpoint_state)
