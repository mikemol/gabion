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
_JOURNAL_SUFFIX = ".journal.jsonl"


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
    generated_session_id = (
        f"session-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    )
    resolved_session_id = (session_id or "").strip() or generated_session_id

    manifest = _normalize_manifest_for_session(
        load_manifest(resolved_manifest_path),
        session_id=resolved_session_id,
        root=resolved_root,
    )
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
            _path_to_manifest_ref(path, root=resolved_root) for path in import_state_paths
        ],
        "status": "started",
        "exit_code": None,
        "analysis_state": None,
        "started_at_utc": started_at_utc,
        "completed_at_utc": None,
    }

    journal_path = _journal_path_for_manifest(resolved_manifest_path)
    _append_event(
        journal_path,
        {
            "event": "prepare_step",
            "session_id": resolved_session_id,
            "root": str(resolved_root),
            "entry": entry,
        },
    )
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
    if str(manifest.get("session_id")) != session_id:
        return False

    entries = _manifest_entries(manifest)
    completed_at_utc = _now_utc()
    found = False
    normalized_entries: list[JSONObject] = []
    for raw_entry in entries:
        seq_value = raw_entry.get("sequence")
        assert isinstance(seq_value, int)
        normalized_entry = {str(key): raw_entry[key] for key in raw_entry}
        if seq_value == sequence:
            normalized_entry["status"] = str(status)
            normalized_entry["exit_code"] = int(exit_code) if exit_code is not None else None
            normalized_entry["analysis_state"] = (
                str(analysis_state) if analysis_state is not None else None
            )
            normalized_entry["completed_at_utc"] = completed_at_utc
            found = True
        normalized_entries.append(normalized_entry)
    if not found:
        return False

    manifest["entries"] = normalized_entries
    journal_path = _journal_path_for_manifest(manifest_path)
    _append_event(
        journal_path,
        {
            "event": "record_step",
            "session_id": session_id,
            "sequence": sequence,
            "status": str(status),
            "exit_code": int(exit_code) if exit_code is not None else None,
            "analysis_state": str(analysis_state) if analysis_state is not None else None,
            "completed_at_utc": completed_at_utc,
        },
    )
    _write_manifest(manifest_path, manifest)
    return True


# gabion:decision_protocol
def aspf_cli_args(
    step: PreparedHandoffStep,
) -> list[str]:
    args = ["--aspf-state-json", str(step.state_path), "--aspf-delta-jsonl", str(step.delta_path)]
    for import_path in step.import_state_paths:
        args.extend(["--aspf-import-state", str(import_path)])
    return args


# gabion:decision_protocol gabion:boundary_normalization
def load_manifest(path: Path) -> JSONObject:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, Mapping)
        return {str(key): payload[key] for key in payload}
    journal_path = _journal_path_for_manifest(path)
    if not journal_path.exists():
        return {}
    payload = _fold_journal(journal_path)
    if payload:
        _write_manifest(path, payload)
    return payload


# gabion:decision_protocol gabion:boundary_normalization
def _successful_state_paths(entries: list[object], *, root: Path) -> list[Path]:
    paths: list[Path] = []
    for raw_entry in entries:
        status = str(raw_entry.get("status", "")).strip().lower()
        if status != "success":
            continue
        state_path = str(raw_entry["state_path"]).strip()
        resolved_state_path = _path_from_manifest_ref(state_path, root=root)
        if not resolved_state_path.exists():
            continue
        paths.append(resolved_state_path)
    return paths


def _write_manifest(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _append_event(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event_payload = dict(payload)
    event_payload["recorded_at_utc"] = _now_utc()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event_payload, sort_keys=False) + "\n")


def _fold_journal(path: Path) -> JSONObject:
    manifest: JSONObject = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            assert isinstance(payload, Mapping)
            event_name = str(payload.get("event", "")).strip()
            if event_name == "prepare_step":
                raw_entry = payload.get("entry")
                assert isinstance(raw_entry, Mapping)
                session_id = str(payload.get("session_id", "")).strip()
                root = Path(str(payload.get("root", "")).strip() or ".").resolve()
                manifest = _normalize_manifest_for_session(
                    manifest,
                    session_id=session_id,
                    root=root,
                )
                entries = _manifest_entries(manifest)
                entries.append({str(key): raw_entry[key] for key in raw_entry})
                manifest["entries"] = entries
                continue
            if event_name == "record_step":
                if str(manifest.get("session_id")) != str(payload.get("session_id", "")):
                    continue
                target_sequence = int(payload["sequence"])
                entries = _manifest_entries(manifest)
                normalized_entries: list[JSONObject] = []
                for raw_entry in entries:
                    seq_value = raw_entry.get("sequence")
                    assert isinstance(seq_value, int)
                    normalized_entry = {str(key): raw_entry[key] for key in raw_entry}
                    if seq_value == target_sequence:
                        normalized_entry["status"] = str(payload["status"])
                        normalized_entry["exit_code"] = payload.get("exit_code")
                        normalized_entry["analysis_state"] = payload.get("analysis_state")
                        normalized_entry["completed_at_utc"] = payload.get("completed_at_utc")
                    normalized_entries.append(normalized_entry)
                manifest["entries"] = normalized_entries
    return manifest


def _normalize_manifest_for_session(
    manifest: Mapping[str, object],
    *,
    session_id: str,
    root: Path,
) -> JSONObject:
    current = {str(key): manifest[key] for key in manifest}
    if current.get("session_id") != session_id:
        return {
            "format_version": _HANDOFF_FORMAT_VERSION,
            "session_id": session_id,
            "root": str(root),
            "entries": [],
        }
    current.setdefault("format_version", _HANDOFF_FORMAT_VERSION)
    current.setdefault("root", str(root))
    current.setdefault("entries", [])
    return current


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


def _journal_path_for_manifest(path: Path) -> Path:
    return path.with_name(f"{path.stem}{_JOURNAL_SUFFIX}")
