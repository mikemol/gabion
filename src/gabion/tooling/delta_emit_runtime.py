# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import inspect
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

from gabion import server
from gabion.commands import progress_contract as progress_timeline
from gabion.lsp_client import CommandRequest, run_command_direct
from gabion.runtime import env_policy

DEFAULT_TIMEOUT_TICKS = "65000000"
DEFAULT_TIMEOUT_TICK_NS = "1000000"
DEFAULT_RESUME_CHECKPOINT_PATH = Path(
    "artifacts/audit_reports/dataflow_resume_checkpoint_ci.json"
)


@dataclass(frozen=True)
class StateInputSpec:
    payload_key: str
    path: Path


@dataclass(frozen=True)
class DeltaEmitPayloadSpec:
    emit_payload_keys: tuple[str, ...]
    state_inputs: tuple[StateInputSpec, ...] = ()
    default_resume_checkpoint_path: Path = DEFAULT_RESUME_CHECKPOINT_PATH


@dataclass(frozen=True)
class DeltaEmitRunSpec:
    script_name: str
    failure_label: str
    expected_outputs: tuple[Path, ...]
    success_mode: Literal["output", "state_artifacts"] = "output"


@dataclass
class TimelineEmitState:
    last_signature: tuple[object, ...] | None = None
    timeline_header_emitted: bool = False
    last_timeline_emit_monotonic: float | None = None
    last_timeline_phase: str | None = None
    pending_phase_progress: dict[str, object] | None = None
    pending_signature: tuple[object, ...] | None = None


def _emit_timeline_row(
    *,
    state: TimelineEmitState,
    phase_progress: Mapping[str, object],
    signature: tuple[object, ...],
    script_name: str,
    print_fn: Callable[[str], None],
    now_monotonic: float,
) -> None:
    phase = str(phase_progress.get("phase", "") or "")
    if not phase:
        return
    timeline = progress_timeline.phase_timeline_from_phase_progress(phase_progress)
    print_fn(f"{script_name} timeline:")
    if not state.timeline_header_emitted:
        print_fn(str(timeline["header"]))
        state.timeline_header_emitted = True
    print_fn(str(timeline["row"]))
    state.last_signature = signature
    state.last_timeline_emit_monotonic = now_monotonic
    state.last_timeline_phase = phase
    state.pending_phase_progress = None
    state.pending_signature = None


def timeout_ticks() -> int:
    raw = env_policy.env_text("GABION_LSP_TIMEOUT_TICKS", default=DEFAULT_TIMEOUT_TICKS)
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return int(DEFAULT_TIMEOUT_TICKS)
    return parsed if parsed > 0 else int(DEFAULT_TIMEOUT_TICKS)


def timeout_tick_ns() -> int:
    raw = env_policy.env_text("GABION_LSP_TIMEOUT_TICK_NS", default=DEFAULT_TIMEOUT_TICK_NS)
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return int(DEFAULT_TIMEOUT_TICK_NS)
    return parsed if parsed > 0 else int(DEFAULT_TIMEOUT_TICK_NS)


def supports_notification_callback(
    run_command_direct_fn: Callable[..., Mapping[str, object]],
) -> bool:
    try:
        signature = inspect.signature(run_command_direct_fn)
    except (TypeError, ValueError):
        return True
    parameters = signature.parameters
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return True
    callback = parameters.get("notification_callback")
    if callback is None:
        return False
    return callback.kind in {
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }


def build_payload(
    payload_spec: DeltaEmitPayloadSpec,
    *,
    resume_checkpoint: Path | bool | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "analysis_timeout_ticks": timeout_ticks(),
        "analysis_timeout_tick_ns": timeout_tick_ns(),
        "fail_on_violations": False,
        "fail_on_type_ambiguities": False,
        "resume_on_timeout": 1,
        "emit_timeout_progress_report": True,
    }
    for key in payload_spec.emit_payload_keys:
        payload[key] = True
    for state_input in payload_spec.state_inputs:
        if state_input.path.exists():
            payload[state_input.payload_key] = str(state_input.path)
    payload["resume_checkpoint"] = _resolve_resume_checkpoint(
        resume_checkpoint=resume_checkpoint,
        default_resume_checkpoint_path=payload_spec.default_resume_checkpoint_path,
    )
    return payload


def _resolve_resume_checkpoint(
    *,
    resume_checkpoint: Path | bool | None,
    default_resume_checkpoint_path: Path,
) -> str | bool:
    if resume_checkpoint is False:
        return False
    if isinstance(resume_checkpoint, Path):
        return str(resume_checkpoint)
    if default_resume_checkpoint_path.exists():
        return str(default_resume_checkpoint_path)
    return False


def maybe_emit_timeline_row(
    notification: Mapping[str, object],
    *,
    state: TimelineEmitState,
    script_name: str,
    print_fn: Callable[[str], None],
    monotonic_fn: Callable[[], float],
    min_interval_seconds: float = progress_timeline.DEFAULT_TIMELINE_MIN_INTERVAL_SECONDS,
) -> None:
    phase_progress = progress_timeline.phase_progress_from_progress_notification(
        notification
    )
    if not isinstance(phase_progress, Mapping):
        return
    normalized_phase_progress = {str(key): phase_progress[key] for key in phase_progress}
    signature = progress_timeline.phase_progress_signature(phase_progress)
    if signature == state.last_signature or signature == state.pending_signature:
        return
    now = monotonic_fn()
    if not progress_timeline.phase_progress_emit_due(
        phase_progress=phase_progress,
        timeline_header_emitted=state.timeline_header_emitted,
        last_emitted_phase=state.last_timeline_phase,
        last_emitted_monotonic=state.last_timeline_emit_monotonic,
        now_monotonic=now,
        min_interval_seconds=min_interval_seconds,
    ):
        state.pending_phase_progress = normalized_phase_progress
        state.pending_signature = signature
        return
    _emit_timeline_row(
        state=state,
        phase_progress=normalized_phase_progress,
        signature=signature,
        script_name=script_name,
        print_fn=print_fn,
        now_monotonic=now,
    )


def flush_pending_timeline_row_if_due(
    *,
    state: TimelineEmitState,
    script_name: str,
    print_fn: Callable[[str], None],
    monotonic_fn: Callable[[], float],
    min_interval_seconds: float = progress_timeline.DEFAULT_TIMELINE_MIN_INTERVAL_SECONDS,
    force: bool = False,
) -> None:
    pending_phase_progress = state.pending_phase_progress
    if not isinstance(pending_phase_progress, Mapping):
        return
    pending_signature = state.pending_signature or progress_timeline.phase_progress_signature(
        pending_phase_progress
    )
    now = monotonic_fn()
    if not force and not progress_timeline.phase_progress_emit_due(
        phase_progress=pending_phase_progress,
        timeline_header_emitted=state.timeline_header_emitted,
        last_emitted_phase=state.last_timeline_phase,
        last_emitted_monotonic=state.last_timeline_emit_monotonic,
        now_monotonic=now,
        min_interval_seconds=min_interval_seconds,
    ):
        return
    normalized_phase_progress = {
        str(key): pending_phase_progress[key] for key in pending_phase_progress
    }
    _emit_timeline_row(
        state=state,
        phase_progress=normalized_phase_progress,
        signature=pending_signature,
        script_name=script_name,
        print_fn=print_fn,
        now_monotonic=now,
    )


def run_delta_emit(
    *,
    run_spec: DeltaEmitRunSpec,
    payload: Mapping[str, object],
    run_command_direct_fn: Callable[..., Mapping[str, object]] = run_command_direct,
    root_path: Path = Path("."),
    print_fn: Callable[[str], None] = print,
    monotonic_fn: Callable[[], float] = time.monotonic,
    min_interval_seconds: float = progress_timeline.DEFAULT_TIMELINE_MIN_INTERVAL_SECONDS,
) -> int:
    started = monotonic_fn()
    print_fn(
        f"{run_spec.script_name}: start "
        f"timeout_ticks={payload.get('analysis_timeout_ticks')} "
        f"timeout_tick_ns={payload.get('analysis_timeout_tick_ns')}"
    )
    timeline_state = TimelineEmitState()
    timeline_state_lock = threading.Lock()
    flush_stop_event = threading.Event()
    flush_interval_seconds = max(0.05, min(1.0, float(min_interval_seconds) / 4.0))

    def _on_notification(notification: dict[str, object]) -> None:
        with timeline_state_lock:
            maybe_emit_timeline_row(
                notification,
                state=timeline_state,
                script_name=run_spec.script_name,
                print_fn=print_fn,
                monotonic_fn=monotonic_fn,
                min_interval_seconds=min_interval_seconds,
            )

    def _flush_pending_loop() -> None:
        while not flush_stop_event.wait(flush_interval_seconds):
            with timeline_state_lock:
                flush_pending_timeline_row_if_due(
                    state=timeline_state,
                    script_name=run_spec.script_name,
                    print_fn=print_fn,
                    monotonic_fn=monotonic_fn,
                    min_interval_seconds=min_interval_seconds,
                )

    flush_thread = threading.Thread(
        target=_flush_pending_loop,
        name=f"{run_spec.script_name}-timeline-flush",
        daemon=True,
    )
    flush_thread.start()

    request = CommandRequest(server.DATAFLOW_COMMAND, [dict(payload)])
    try:
        if supports_notification_callback(run_command_direct_fn):
            result = run_command_direct_fn(
                request,
                root=root_path,
                notification_callback=_on_notification,
            )
        else:
            result = run_command_direct_fn(request, root=root_path)
    finally:
        flush_stop_event.set()
        flush_thread.join(timeout=1.0)
    with timeline_state_lock:
        flush_pending_timeline_row_if_due(
            state=timeline_state,
            script_name=run_spec.script_name,
            print_fn=print_fn,
            monotonic_fn=monotonic_fn,
            min_interval_seconds=min_interval_seconds,
            force=True,
        )
    exit_code = int(result.get("exit_code", 0))
    elapsed_seconds = max(0.0, monotonic_fn() - started)
    print_fn(
        f"{run_spec.script_name}: complete exit={exit_code} elapsed_s={elapsed_seconds:.2f}"
    )
    if exit_code != 0:
        print_fn(f"{run_spec.failure_label} failed (exit {exit_code}).")
        return exit_code
    missing_outputs = [
        str(path) for path in run_spec.expected_outputs if not path.exists()
    ]
    if missing_outputs:
        print_fn(
            _missing_output_message(
                run_spec=run_spec,
                missing_outputs=missing_outputs,
            )
        )
        return 1
    print_fn(_success_message(run_spec=run_spec))
    return 0


def _missing_output_message(
    *,
    run_spec: DeltaEmitRunSpec,
    missing_outputs: Sequence[str],
) -> str:
    if run_spec.success_mode == "state_artifacts":
        return (
            f"{run_spec.failure_label} failed: missing expected state artifacts: "
            + ", ".join(missing_outputs)
        )
    output_path = missing_outputs[0] if missing_outputs else "<missing>"
    return f"{run_spec.failure_label} failed: missing output {output_path}."


def _success_message(*, run_spec: DeltaEmitRunSpec) -> str:
    if run_spec.success_mode == "state_artifacts":
        return (
            f"{run_spec.script_name}: state artifacts ready "
            + ", ".join(str(path) for path in run_spec.expected_outputs)
        )
    output_path = run_spec.expected_outputs[0] if run_spec.expected_outputs else Path(".")
    return f"{run_spec.script_name}: output ready {output_path}"
