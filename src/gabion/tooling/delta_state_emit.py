# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Callable, Literal, Mapping

from gabion.commands import progress_contract as progress_timeline
from gabion.lsp_client import run_command, run_command_direct
from gabion.tooling import delta_emit_runtime

EmitterId = Literal[
    "delta_state_emit",
    "obsolescence_delta_emit",
    "annotation_drift_delta_emit",
    "ambiguity_delta_emit",
]

_DEFAULT_TIMEOUT_TICKS = delta_emit_runtime.DEFAULT_TIMEOUT_TICKS
_DEFAULT_TIMEOUT_TICK_NS = delta_emit_runtime.DEFAULT_TIMEOUT_TICK_NS
_TIMELINE_MIN_INTERVAL_SECONDS = (
    progress_timeline.DEFAULT_TIMELINE_MIN_INTERVAL_SECONDS
)
_LSP_PROGRESS_NOTIFICATION_METHOD = progress_timeline.LSP_PROGRESS_NOTIFICATION_METHOD
_LSP_PROGRESS_TOKEN = progress_timeline.LSP_PROGRESS_TOKEN
_DEFAULT_RESUME_CHECKPOINT_PATH = delta_emit_runtime.DEFAULT_RESUME_CHECKPOINT_PATH
_EXPECTED_STATE_PATHS = (
    Path("artifacts/out/test_obsolescence_state.json"),
    Path("artifacts/out/test_annotation_drift.json"),
    Path("artifacts/out/ambiguity_state.json"),
)
_OBSOLESCENCE_STATE_PATH = Path("artifacts/out/test_obsolescence_state.json")
_OBSOLESCENCE_DELTA_PATH = Path("artifacts/out/test_obsolescence_delta.json")
_ANNOTATION_DRIFT_STATE_PATH = Path("artifacts/out/test_annotation_drift.json")
_ANNOTATION_DRIFT_DELTA_PATH = Path("artifacts/out/test_annotation_drift_delta.json")
_AMBIGUITY_STATE_PATH = Path("artifacts/out/ambiguity_state.json")
_AMBIGUITY_DELTA_PATH = Path("artifacts/out/ambiguity_delta.json")


@dataclass(frozen=True)
class DeltaEmitterConfig:
    payload_spec: delta_emit_runtime.DeltaEmitPayloadSpec
    run_spec: delta_emit_runtime.DeltaEmitRunSpec


_EMITTER_CONFIGS: dict[EmitterId, DeltaEmitterConfig] = {
    "delta_state_emit": DeltaEmitterConfig(
        payload_spec=delta_emit_runtime.DeltaEmitPayloadSpec(
            emit_payload_keys=(
                "emit_test_obsolescence_state",
                "emit_test_annotation_drift",
                "emit_ambiguity_state",
            ),
            default_resume_checkpoint_path=_DEFAULT_RESUME_CHECKPOINT_PATH,
        ),
        run_spec=delta_emit_runtime.DeltaEmitRunSpec(
            script_name="delta_state_emit",
            failure_label="Delta state emit",
            expected_outputs=_EXPECTED_STATE_PATHS,
            success_mode="state_artifacts",
        ),
    ),
    "obsolescence_delta_emit": DeltaEmitterConfig(
        payload_spec=delta_emit_runtime.DeltaEmitPayloadSpec(
            emit_payload_keys=("emit_test_obsolescence_delta",),
            state_inputs=(
                delta_emit_runtime.StateInputSpec(
                    payload_key="test_obsolescence_state",
                    path=_OBSOLESCENCE_STATE_PATH,
                ),
            ),
            default_resume_checkpoint_path=_DEFAULT_RESUME_CHECKPOINT_PATH,
        ),
        run_spec=delta_emit_runtime.DeltaEmitRunSpec(
            script_name="obsolescence_delta_emit",
            failure_label="Test obsolescence delta emit",
            expected_outputs=(_OBSOLESCENCE_DELTA_PATH,),
            success_mode="output",
        ),
    ),
    "annotation_drift_delta_emit": DeltaEmitterConfig(
        payload_spec=delta_emit_runtime.DeltaEmitPayloadSpec(
            emit_payload_keys=("emit_test_annotation_drift_delta",),
            state_inputs=(
                delta_emit_runtime.StateInputSpec(
                    payload_key="test_annotation_drift_state",
                    path=_ANNOTATION_DRIFT_STATE_PATH,
                ),
            ),
            default_resume_checkpoint_path=_DEFAULT_RESUME_CHECKPOINT_PATH,
        ),
        run_spec=delta_emit_runtime.DeltaEmitRunSpec(
            script_name="annotation_drift_delta_emit",
            failure_label="Annotation drift delta emit",
            expected_outputs=(_ANNOTATION_DRIFT_DELTA_PATH,),
            success_mode="output",
        ),
    ),
    "ambiguity_delta_emit": DeltaEmitterConfig(
        payload_spec=delta_emit_runtime.DeltaEmitPayloadSpec(
            emit_payload_keys=("emit_ambiguity_delta",),
            state_inputs=(
                delta_emit_runtime.StateInputSpec(
                    payload_key="ambiguity_state",
                    path=_AMBIGUITY_STATE_PATH,
                ),
            ),
            default_resume_checkpoint_path=_DEFAULT_RESUME_CHECKPOINT_PATH,
        ),
        run_spec=delta_emit_runtime.DeltaEmitRunSpec(
            script_name="ambiguity_delta_emit",
            failure_label="Ambiguity delta emit",
            expected_outputs=(_AMBIGUITY_DELTA_PATH,),
            success_mode="output",
        ),
    ),
}


def _phase_timeline_header_columns() -> list[str]:
    return progress_timeline.phase_timeline_header_columns()


def _phase_timeline_header_block() -> str:
    return progress_timeline.phase_timeline_header_block()


def _phase_progress_dimensions_summary(
    phase_progress_v2: Mapping[str, object] | None,
) -> str:
    return progress_timeline.phase_progress_dimensions_summary(phase_progress_v2)


def _phase_timeline_row_from_phase_progress(
    phase_progress: Mapping[str, object],
) -> str:
    return progress_timeline.phase_timeline_row_from_phase_progress(phase_progress)


def _timeout_ticks() -> int:
    return delta_emit_runtime.timeout_ticks()


def _timeout_tick_ns() -> int:
    return delta_emit_runtime.timeout_tick_ns()


def _build_payload() -> dict[str, object]:
    return _build_payload_for_emitter("delta_state_emit")


def _build_payload_for_emitter(
    emitter_id: EmitterId,
    *,
    resume_checkpoint: Path | bool | None = None,
) -> dict[str, object]:
    config = _EMITTER_CONFIGS[emitter_id]
    return delta_emit_runtime.build_payload(
        config.payload_spec,
        resume_checkpoint=resume_checkpoint,
    )


def _supports_notification_callback(
    run_command_direct_fn: Callable[..., Mapping[str, object]],
) -> bool:
    return delta_emit_runtime.supports_notification_callback(run_command_direct_fn)


def _phase_progress_from_notification(
    notification: Mapping[str, object],
) -> dict[str, object] | None:
    phase_progress = progress_timeline.phase_progress_from_progress_notification(
        notification
    )
    if isinstance(phase_progress, Mapping):
        return {str(key): phase_progress[key] for key in phase_progress}
    return None


def _emit_phase_progress_line(
    phase_progress: Mapping[str, object],
    *,
    print_fn: Callable[[str], None] = print,
) -> None:
    phase = str(phase_progress.get("phase", "") or "")
    if not phase:
        return
    timeline = progress_timeline.phase_timeline_from_phase_progress(phase_progress)
    print_fn("delta_state_emit timeline:")
    print_fn(str(timeline["header"]))
    print_fn(str(timeline["row"]))


def main_for_emitter(
    emitter_id: EmitterId,
    *,
    run_command_direct_fn: Callable[..., Mapping[str, object]] = run_command_direct,
    run_command_fn: Callable[..., Mapping[str, object]] = run_command,
    root_path: Path = Path("."),
    output_path: Path | None = None,
    expected_outputs: tuple[Path, ...] | None = None,
    resume_checkpoint: Path | bool | None = None,
    print_fn: Callable[[str], None] = print,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> int:
    config = _EMITTER_CONFIGS[emitter_id]
    run_spec = config.run_spec
    if isinstance(expected_outputs, tuple):
        run_spec = replace(run_spec, expected_outputs=expected_outputs)
    elif isinstance(output_path, Path):
        run_spec = replace(run_spec, expected_outputs=(output_path,))
    payload = _build_payload_for_emitter(
        emitter_id,
        resume_checkpoint=resume_checkpoint,
    )
    return delta_emit_runtime.run_delta_emit(
        run_spec=run_spec,
        payload=payload,
        run_command_direct_fn=run_command_direct_fn,
        run_command_fn=run_command_fn,
        root_path=root_path,
        print_fn=print_fn,
        monotonic_fn=monotonic_fn,
        min_interval_seconds=_TIMELINE_MIN_INTERVAL_SECONDS,
    )


def main(
    *,
    run_command_direct_fn: Callable[..., Mapping[str, object]] = run_command_direct,
    run_command_fn: Callable[..., Mapping[str, object]] = run_command,
    print_fn: Callable[[str], None] = print,
    monotonic_fn: Callable[[], float] = time.monotonic,
    expected_state_paths: tuple[Path, ...] = _EXPECTED_STATE_PATHS,
    root_path: Path = Path("."),
) -> int:
    return main_for_emitter(
        "delta_state_emit",
        run_command_direct_fn=run_command_direct_fn,
        run_command_fn=run_command_fn,
        root_path=root_path,
        expected_outputs=expected_state_paths,
        print_fn=print_fn,
        monotonic_fn=monotonic_fn,
    )


def obsolescence_main(
    *,
    run_command_direct_fn: Callable[..., Mapping[str, object]] = run_command_direct,
    run_command_fn: Callable[..., Mapping[str, object]] = run_command,
    root_path: Path = Path("."),
    delta_path: Path = _OBSOLESCENCE_DELTA_PATH,
    resume_checkpoint: Path | bool | None = None,
    print_fn: Callable[[str], None] = print,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> int:
    return main_for_emitter(
        "obsolescence_delta_emit",
        run_command_direct_fn=run_command_direct_fn,
        run_command_fn=run_command_fn,
        root_path=root_path,
        output_path=delta_path,
        resume_checkpoint=resume_checkpoint,
        print_fn=print_fn,
        monotonic_fn=monotonic_fn,
    )


def annotation_drift_main(
    *,
    run_command_direct_fn: Callable[..., Mapping[str, object]] = run_command_direct,
    run_command_fn: Callable[..., Mapping[str, object]] = run_command,
    root_path: Path = Path("."),
    delta_path: Path = _ANNOTATION_DRIFT_DELTA_PATH,
    resume_checkpoint: Path | bool | None = None,
    print_fn: Callable[[str], None] = print,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> int:
    return main_for_emitter(
        "annotation_drift_delta_emit",
        run_command_direct_fn=run_command_direct_fn,
        run_command_fn=run_command_fn,
        root_path=root_path,
        output_path=delta_path,
        resume_checkpoint=resume_checkpoint,
        print_fn=print_fn,
        monotonic_fn=monotonic_fn,
    )


def ambiguity_main(
    *,
    run_command_direct_fn: Callable[..., Mapping[str, object]] = run_command_direct,
    run_command_fn: Callable[..., Mapping[str, object]] = run_command,
    root_path: Path = Path("."),
    delta_path: Path = _AMBIGUITY_DELTA_PATH,
    resume_checkpoint: Path | bool | None = None,
    print_fn: Callable[[str], None] = print,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> int:
    return main_for_emitter(
        "ambiguity_delta_emit",
        run_command_direct_fn=run_command_direct_fn,
        run_command_fn=run_command_fn,
        root_path=root_path,
        output_path=delta_path,
        resume_checkpoint=resume_checkpoint,
        print_fn=print_fn,
        monotonic_fn=monotonic_fn,
    )
