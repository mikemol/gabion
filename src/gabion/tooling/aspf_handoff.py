from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence

from gabion.json_types import JSONObject

_HANDOFF_FORMAT_VERSION = 1
_DEFAULT_MANIFEST_PATH = Path("artifacts/out/aspf_handoff_manifest.json")
_DEFAULT_STATE_ROOT = Path("artifacts/out/aspf_state")


@dataclass(frozen=True)
class PreparedHandoffStep:
    sequence: int
    session_id: str
    step_id: str
    command_profile: str
    state_path: Path
    delta_path: Path
    action_plan_json_path: Path
    action_plan_md_path: Path
    import_state_paths: tuple[Path, ...]
    manifest_path: Path
    started_at_utc: str


@dataclass(frozen=True)
class AspfHandoffRunSpec:
    root: Path
    session_id: str | None
    step_id: str
    command_profile: str
    command: tuple[str, ...]
    manifest_path: Path | None = None
    state_root: Path | None = None
    aspf_cli_mode: str = "full"
    infer_analysis_state_from_state_file: bool = True


@dataclass(frozen=True)
class AspfHandoffRunResult:
    ok: bool
    exit_code: int
    status: str
    analysis_state: str | None
    prepared: PreparedHandoffStep
    command: tuple[str, ...]
    command_with_aspf: tuple[str, ...]


def default_manifest_path(root: Path) -> Path:
    return root / _DEFAULT_MANIFEST_PATH


def default_state_root(root: Path) -> Path:
    return root / _DEFAULT_STATE_ROOT


def new_session_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"session-{stamp}-{os.getpid()}"


def prepare_step(
    *,
    root: Path,
    session_id: str | None,
    step_id: str,
    command_profile: str,
    resume_checkpoint_path: Path | None = None,
    manifest_path: Path | None = None,
    state_root: Path | None = None,
) -> PreparedHandoffStep:
    resolved_root = root.resolve()
    resolved_manifest_path = (
        (resolved_root / manifest_path).resolve()
        if manifest_path is not None and not manifest_path.is_absolute()
        else (manifest_path if manifest_path is not None else default_manifest_path(resolved_root))
    )
    resolved_state_root = (
        (resolved_root / state_root).resolve()
        if state_root is not None and not state_root.is_absolute()
        else (state_root if state_root is not None else default_state_root(resolved_root))
    )
    resolved_session_id = (session_id or "").strip() or new_session_id()
    manifest = load_manifest(resolved_manifest_path)
    if manifest.get("session_id") != resolved_session_id:
        manifest = {
            "format_version": _HANDOFF_FORMAT_VERSION,
            "session_id": resolved_session_id,
            "root": str(resolved_root),
            "entries": [],
        }
    entries_raw = manifest.get("entries")
    entries = list(entries_raw) if isinstance(entries_raw, list) else []
    import_state_paths = tuple(_successful_state_paths(entries, root=resolved_root))
    sequence = len(entries) + 1
    safe_step_id = _step_slug(step_id)
    state_path = (
        resolved_state_root
        / resolved_session_id
        / f"{sequence:04d}_{safe_step_id}.snapshot.json"
    )
    delta_path = (
        resolved_state_root
        / resolved_session_id
        / f"{sequence:04d}_{safe_step_id}.delta.jsonl"
    )
    action_plan_json_path = (
        resolved_state_root
        / resolved_session_id
        / f"{sequence:04d}_{safe_step_id}.action_plan.json"
    )
    action_plan_md_path = (
        resolved_state_root
        / resolved_session_id
        / f"{sequence:04d}_{safe_step_id}.action_plan.md"
    )
    started_at_utc = _now_utc()
    entry: JSONObject = {
        "sequence": sequence,
        "step_id": step_id,
        "command_profile": command_profile,
        "state_path": _path_to_manifest_ref(state_path, root=resolved_root),
        "delta_path": _path_to_manifest_ref(delta_path, root=resolved_root),
        "action_plan_json_path": _path_to_manifest_ref(
            action_plan_json_path, root=resolved_root
        ),
        "action_plan_md_path": _path_to_manifest_ref(
            action_plan_md_path, root=resolved_root
        ),
        "import_state_paths": [
            _path_to_manifest_ref(path, root=resolved_root)
            for path in import_state_paths
        ],
        "status": "started",
        "exit_code": None,
        "analysis_state": None,
        "started_at_utc": started_at_utc,
        "completed_at_utc": None,
    }
    entries.append(entry)
    manifest["entries"] = entries
    _write_manifest(resolved_manifest_path, manifest)
    return PreparedHandoffStep(
        sequence=sequence,
        session_id=resolved_session_id,
        step_id=step_id,
        command_profile=command_profile,
        state_path=state_path,
        delta_path=delta_path,
        action_plan_json_path=action_plan_json_path,
        action_plan_md_path=action_plan_md_path,
        import_state_paths=import_state_paths,
        manifest_path=resolved_manifest_path,
        started_at_utc=started_at_utc,
    )


def record_step(
    *,
    manifest_path: Path,
    session_id: str,
    sequence: int,
    status: str,
    exit_code: int | None,
    analysis_state: str | None,
) -> bool:
    manifest = load_manifest(manifest_path)
    if manifest.get("session_id") != session_id:
        return False
    entries_raw = manifest.get("entries")
    if not isinstance(entries_raw, list):
        return False
    target: JSONObject | None = None
    for raw_entry in entries_raw:
        if not isinstance(raw_entry, Mapping):
            continue
        seq_value = raw_entry.get("sequence")
        if not isinstance(seq_value, int):
            continue
        if seq_value != sequence:
            continue
        target = {str(key): raw_entry[key] for key in raw_entry}
        break
    if target is None:
        return False
    target["status"] = str(status)
    target["exit_code"] = int(exit_code) if exit_code is not None else None
    target["analysis_state"] = str(analysis_state) if analysis_state is not None else None
    target["completed_at_utc"] = _now_utc()
    normalized_entries: list[JSONObject] = []
    for raw_entry in entries_raw:
        if not isinstance(raw_entry, Mapping):
            continue
        seq_value = raw_entry.get("sequence")
        if isinstance(seq_value, int) and seq_value == sequence:
            normalized_entries.append(target)
            continue
        normalized_entries.append({str(key): raw_entry[key] for key in raw_entry})
    manifest["entries"] = normalized_entries
    _write_manifest(manifest_path, manifest)
    return True


def aspf_cli_args(
    step: PreparedHandoffStep,
    *,
    mode: str = "full",
) -> list[str]:
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"full", "state_only"}:
        raise ValueError(f"unsupported ASPF CLI mode: {mode}")
    args = [
        "--aspf-state-json",
        str(step.state_path),
    ]
    if normalized_mode == "full":
        args.extend(
            [
                "--aspf-delta-jsonl",
                str(step.delta_path),
                "--aspf-action-plan-json",
                str(step.action_plan_json_path),
                "--aspf-action-plan-md",
                str(step.action_plan_md_path),
            ]
        )
    for import_path in step.import_state_paths:
        args.extend(["--aspf-import-state", str(import_path)])
    return args


def run_with_handoff(
    *,
    spec: AspfHandoffRunSpec,
    run_command_fn: Callable[[Sequence[str]], int],
    derive_analysis_state_fn: Callable[[PreparedHandoffStep, int], str | None]
    | None = None,
) -> AspfHandoffRunResult:
    prepared = prepare_step(
        root=spec.root,
        session_id=spec.session_id,
        step_id=spec.step_id,
        command_profile=spec.command_profile,
        manifest_path=spec.manifest_path,
        state_root=spec.state_root,
    )
    aspf_args = aspf_cli_args(prepared, mode=spec.aspf_cli_mode)
    command_with_aspf = tuple([*spec.command, *aspf_args])
    exit_code = int(run_command_fn(command_with_aspf))
    status = "success" if exit_code == 0 else "failed"
    if derive_analysis_state_fn is not None:
        analysis_state = derive_analysis_state_fn(prepared, exit_code)
    elif spec.infer_analysis_state_from_state_file:
        analysis_state = _analysis_state_from_state_file(prepared.state_path)
    else:
        analysis_state = None
    if not analysis_state:
        analysis_state = "succeeded" if exit_code == 0 else "failed"
    ok = record_step(
        manifest_path=prepared.manifest_path,
        session_id=prepared.session_id,
        sequence=prepared.sequence,
        status=status,
        exit_code=exit_code,
        analysis_state=analysis_state,
    )
    return AspfHandoffRunResult(
        ok=ok,
        exit_code=exit_code,
        status=status,
        analysis_state=analysis_state,
        prepared=prepared,
        command=tuple(spec.command),
        command_with_aspf=command_with_aspf,
    )


def load_manifest(path: Path) -> JSONObject:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    return {str(key): payload[key] for key in payload}


def _successful_state_paths(entries: list[object], *, root: Path) -> list[Path]:
    paths: list[Path] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            continue
        status = str(raw_entry.get("status", "")).strip().lower()
        if status != "success":
            continue
        state_path = raw_entry.get("state_path")
        if not isinstance(state_path, str) or not state_path.strip():
            continue
        paths.append(_path_from_manifest_ref(state_path, root=root))
    return paths


def _write_manifest(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _step_slug(step_id: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", step_id.strip())
    trimmed = normalized.strip("-.")
    return trimmed or "step"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _analysis_state_from_state_file(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    analysis_state = payload.get("analysis_state")
    if isinstance(analysis_state, str) and analysis_state.strip():
        return analysis_state.strip()
    resume_projection = payload.get("resume_projection")
    if isinstance(resume_projection, Mapping):
        projected_state = resume_projection.get("analysis_state")
        if isinstance(projected_state, str) and projected_state.strip():
            return projected_state.strip()
    return None


def _path_to_manifest_ref(path: Path, *, root: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError:
        return str(resolved_path)
    return relative.as_posix()


def _path_from_manifest_ref(value: str, *, root: Path) -> Path:
    candidate = Path(value.strip())
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()
