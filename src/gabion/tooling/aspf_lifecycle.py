# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from gabion.analysis.timeout_context import check_deadline
from gabion.invariants import never
from gabion.tooling import aspf_handoff

ResumeImportPolicy = Literal["success_only", "success_or_resumable_timeout"]


@dataclass(frozen=True)
class AspfLifecycleConfig:
    enabled: bool
    root: Path
    session_id: str
    manifest_path: Path
    state_root: Path
    write_manifest_projection: bool = True
    resume_import_policy: ResumeImportPolicy = "success_or_resumable_timeout"


@dataclass(frozen=True)
class AspfStepResult:
    command_with_aspf: tuple[str, ...]
    exit_code: int
    status: str
    analysis_state: str
    session_id: str | None
    sequence: int | None
    manifest_path: Path | None
    state_path: Path | None
    import_state_paths: tuple[Path, ...]


# gabion:decision_protocol
def resume_import_policy(
    *,
    config: AspfLifecycleConfig,
) -> ResumeImportPolicy:
    if not config.enabled:
        return "success_only"
    policy = str(config.resume_import_policy)
    if policy in {"success_only", "success_or_resumable_timeout"}:
        return policy  # type: ignore[return-value]
    never("invalid resume import policy", policy=policy)
    return "success_only"  # pragma: no cover


# gabion:decision_protocol
def run_with_aspf_lifecycle(
    *,
    config: AspfLifecycleConfig | None,
    step_id: str,
    command_profile: str,
    command: Sequence[str],
    run_command_fn: Callable[[Sequence[str]], int],
    analysis_state_from_state_path_fn: Callable[[Path], str],
    prepare_step_fn: Callable[..., Any] = aspf_handoff.prepare_step,
    record_step_fn: Callable[..., bool] = aspf_handoff.record_step,
    aspf_cli_args_fn: Callable[[Any], list[str]] = aspf_handoff.aspf_cli_args,
) -> AspfStepResult:
    if config is None or not config.enabled:
        exit_code = int(run_command_fn(command))
        analysis_state = "succeeded" if exit_code == 0 else "failed"
        return AspfStepResult(
            command_with_aspf=tuple(str(token) for token in command),
            exit_code=exit_code,
            status="success" if exit_code == 0 else "failed",
            analysis_state=analysis_state,
            session_id=None,
            sequence=None,
            manifest_path=None,
            state_path=None,
            import_state_paths=(),
        )

    prepared = prepare_step_fn(
        root=config.root,
        session_id=config.session_id,
        step_id=step_id,
        command_profile=command_profile,
        manifest_path=config.manifest_path,
        state_root=config.state_root,
        write_manifest_projection=config.write_manifest_projection,
        resume_import_policy=resume_import_policy(config=config),
    )
    command_with_aspf = tuple([
        *(str(token) for token in command),
        *aspf_cli_args_fn(prepared),
    ])
    exit_code = int(run_command_fn(command_with_aspf))
    status = "success" if exit_code == 0 else "failed"
    analysis_state = str(analysis_state_from_state_path_fn(prepared.state_path) or "").strip()
    if not analysis_state or analysis_state == "none":
        analysis_state = "succeeded" if exit_code == 0 else "failed"
    recorded = record_step_fn(
        manifest_path=prepared.manifest_path,
        session_id=prepared.session_id,
        sequence=prepared.sequence,
        status=status,
        exit_code=exit_code,
        analysis_state=analysis_state,
        write_manifest_projection=config.write_manifest_projection,
    )
    if not recorded:
        never(
            "aspf lifecycle failed to record prepared step",
            session_id=prepared.session_id,
            sequence=prepared.sequence,
            step_id=prepared.step_id,
        )
    return AspfStepResult(
        command_with_aspf=command_with_aspf,
        exit_code=exit_code,
        status=status,
        analysis_state=analysis_state,
        session_id=prepared.session_id,
        sequence=prepared.sequence,
        manifest_path=prepared.manifest_path,
        state_path=prepared.state_path,
        import_state_paths=tuple(prepared.import_state_paths),
    )


__all__ = [
    "AspfLifecycleConfig",
    "AspfStepResult",
    "ResumeImportPolicy",
    "resume_import_policy",
    "run_with_aspf_lifecycle",
]
