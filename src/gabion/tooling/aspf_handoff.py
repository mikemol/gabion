from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Mapping

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
    import_state_paths: tuple[Path, ...]
    manifest_path: Path
    started_at_utc: str


def new_session_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"session-{stamp}-{os.getpid()}"


# gabion:decision_protocol
def prepare_step(
    *,
    root: Path,
    session_id: str | None,
    step_id: str,
    command_profile: str,
    manifest_path: Path | None = None,
    state_root: Path | None = None,
) -> PreparedHandoffStep:
    resolved_root = root.resolve()
    resolved_manifest_path = _resolve_under_root(
        root=resolved_root,
        value=manifest_path if manifest_path is not None else _DEFAULT_MANIFEST_PATH,
    )
    resolved_state_root = _resolve_under_root(
        root=resolved_root,
        value=state_root if state_root is not None else _DEFAULT_STATE_ROOT,
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
    entries = _manifest_entries(manifest)
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
    started_at_utc = _now_utc()
    entry: JSONObject = {
        "sequence": sequence,
        "step_id": step_id,
        "command_profile": command_profile,
        "state_path": _path_to_manifest_ref(state_path, root=resolved_root),
        "delta_path": _path_to_manifest_ref(delta_path, root=resolved_root),
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
        import_state_paths=import_state_paths,
        manifest_path=resolved_manifest_path,
        started_at_utc=started_at_utc,
    )


# gabion:decision_protocol gabion:boundary_normalization
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
    assert manifest.get("session_id") == session_id
    entries_raw = _manifest_entries(manifest)
    target: JSONObject | None = None
    for raw_entry in entries_raw:
        seq_value = raw_entry.get("sequence")
        assert isinstance(seq_value, int)
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
        seq_value = raw_entry.get("sequence")
        assert isinstance(seq_value, int)
        if isinstance(seq_value, int) and seq_value == sequence:
            normalized_entries.append(target)
            continue
        normalized_entries.append({str(key): raw_entry[key] for key in raw_entry})
    manifest["entries"] = normalized_entries
    _write_manifest(manifest_path, manifest)
    return True


# gabion:decision_protocol
def aspf_cli_args(
    step: PreparedHandoffStep,
) -> list[str]:
    args = [
        "--aspf-state-json",
        str(step.state_path),
        "--aspf-delta-jsonl",
        str(step.delta_path),
    ]
    for import_path in step.import_state_paths:
        args.extend(["--aspf-import-state", str(import_path)])
    return args


# gabion:decision_protocol gabion:boundary_normalization
def load_manifest(path: Path) -> JSONObject:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, Mapping)
    return {str(key): payload[key] for key in payload}


# gabion:decision_protocol gabion:boundary_normalization
def _successful_state_paths(entries: list[object], *, root: Path) -> list[Path]:
    paths: list[Path] = []
    for raw_entry in entries:
        status = str(raw_entry.get("status", "")).strip().lower()
        if status != "success":
            continue
        state_path = str(raw_entry["state_path"]).strip()
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


# gabion:decision_protocol gabion:boundary_normalization
def _resolve_under_root(*, root: Path, value: Path) -> Path:
    if value.is_absolute():
        return value
    return (root / value).resolve()


def _manifest_entries(payload: Mapping[str, object]) -> list[JSONObject]:
    entries_raw = payload.get("entries", [])
    assert isinstance(entries_raw, list)
    return [{str(key): raw_entry[key] for key in raw_entry} for raw_entry in entries_raw]


# gabion:decision_protocol
def _path_to_manifest_ref(path: Path, *, root: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    relative = resolved_path.relative_to(resolved_root)
    return relative.as_posix()


# gabion:decision_protocol
def _path_from_manifest_ref(value: str, *, root: Path) -> Path:
    candidate = Path(value.strip())
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
