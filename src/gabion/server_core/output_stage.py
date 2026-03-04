from __future__ import annotations

from gabion.server_core.stage_contracts import (
    AuxiliaryOutputEmitter,
    PrimaryOutputEmitter,
    StageOutputResult,
)


def run_output_stage(
    *,
    primary_output_emitter: PrimaryOutputEmitter,
    auxiliary_output_emitter: AuxiliaryOutputEmitter,
    primary_args: tuple[object, ...] = (),
    primary_kwargs: dict[str, object] | None = None,
    auxiliary_args: tuple[object, ...] = (),
    auxiliary_kwargs: dict[str, object] | None = None,
) -> StageOutputResult:
    rendered = primary_output_emitter(
        *primary_args,
        **(primary_kwargs or {}),
    )
    auxiliary_output_emitter(
        *auxiliary_args,
        **(auxiliary_kwargs or {}),
    )
    checkpoint_state = rendered.get("phase_checkpoint_state", {})
    return StageOutputResult(response=rendered, phase_checkpoint_state=checkpoint_state)
