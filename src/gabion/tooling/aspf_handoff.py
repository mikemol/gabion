# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Literal, Mapping

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


@dataclass(frozen=True)
class PrepareStepEvent:
    event: Literal["prepare_step"]
    session_id: str
    root: str
    entry: JSONObject


@dataclass(frozen=True)
class RecordStepEvent:
    event: Literal["record_step"]
    session_id: str
    sequence: int
    status: str
    exit_code: int | None
    analysis_state: str | None
    completed_at_utc: str | None


HandoffEvent = PrepareStepEvent | RecordStepEvent


@dataclass
class HandoffProjectionState:
    manifest: JSONObject

    @classmethod
    def empty(cls) -> HandoffProjectionState:
        return cls(manifest={})


class HandoffEventReducer:
    @staticmethod
    def apply(state: HandoffProjectionState, event: HandoffEvent) -> HandoffProjectionState:
        manifest = {str(key): state.manifest[key] for key in state.manifest}
        if isinstance(event, PrepareStepEvent):
            normalized = _normalize_manifest_for_session(
                manifest,
                session_id=event.session_id,
                root=Path(event.root).resolve(),
            )
            entries = _manifest_entries(normalized)
            entries.append({str(key): event.entry[key] for key in event.entry})
            normalized["entries"] = entries
            return HandoffProjectionState(manifest=normalized)

        if str(manifest.get("session_id")) != event.session_id:
            return HandoffProjectionState(manifest=manifest)

        entries = _manifest_entries(manifest)
        found = False
        normalized_entries: list[JSONObject] = []
        for raw_entry in entries:
            seq_value = raw_entry.get("sequence")
            assert isinstance(seq_value, int)
            normalized_entry = {str(key): raw_entry[key] for key in raw_entry}
            if seq_value == event.sequence:
                normalized_entry["status"] = event.status
                normalized_entry["exit_code"] = event.exit_code
                normalized_entry["analysis_state"] = event.analysis_state
                normalized_entry["completed_at_utc"] = event.completed_at_utc
                found = True
            normalized_entries.append(normalized_entry)
        if not found:
            return HandoffProjectionState(manifest=manifest)
        manifest["entries"] = normalized_entries
        return HandoffProjectionState(manifest=manifest)


# gabion:decision_protocol
def prepare_step(
    *,
    root: Path,
    session_id: str | None,
    step_id: str,
    command_profile: str,
    manifest_path: Path | None = None,
    state_root: Path | None = None,
    write_manifest_projection: bool = True,
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
        load_manifest(resolved_manifest_path, cache_projection=False),
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
    event = PrepareStepEvent(
        event="prepare_step",
        session_id=resolved_session_id,
        root=str(resolved_root),
        entry=entry,
    )
    _append_event(journal_path, _event_to_payload(event))
    projected = HandoffEventReducer.apply(HandoffProjectionState(manifest=manifest), event).manifest
    if write_manifest_projection:
        _write_manifest(resolved_manifest_path, projected)

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
    write_manifest_projection: bool = True,
) -> bool:
    manifest = load_manifest(manifest_path, cache_projection=False)
    if str(manifest.get("session_id")) != session_id:
        return False

    entries = _manifest_entries(manifest)
    completed_at_utc = _now_utc()
    if not any(raw_entry.get("sequence") == sequence for raw_entry in entries):
        return False
    journal_path = _journal_path_for_manifest(manifest_path)
    event = RecordStepEvent(
        event="record_step",
        session_id=session_id,
        sequence=sequence,
        status=str(status),
        exit_code=int(exit_code) if exit_code is not None else None,
        analysis_state=str(analysis_state) if analysis_state is not None else None,
        completed_at_utc=completed_at_utc,
    )
    _append_event(journal_path, _event_to_payload(event))
    projected = HandoffEventReducer.apply(HandoffProjectionState(manifest=manifest), event).manifest
    if write_manifest_projection:
        _write_manifest(manifest_path, projected)
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
def load_manifest(path: Path, *, cache_projection: bool = True) -> JSONObject:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, Mapping)
        return _project_manifest_payload(payload)
    journal_path = _journal_path_for_manifest(path)
    if not journal_path.exists():
        return {}
    payload = _fold_journal(journal_path)
    if payload and cache_projection:
        _write_manifest(path, payload)
    return payload


def _project_manifest_payload(payload: Mapping[str, object]) -> JSONObject:
    state = HandoffProjectionState.empty()
    for event in _events_from_manifest_payload(payload):
        state = HandoffEventReducer.apply(state, event)
    return state.manifest


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


# gabion:boundary_normalization
def _fold_journal(path: Path) -> JSONObject:
    state = HandoffProjectionState.empty()
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            assert isinstance(payload, Mapping)
            event = _event_from_payload(payload)
            if event is None:
                continue
            state = HandoffEventReducer.apply(state, event)
    return state.manifest


def _events_from_manifest_payload(payload: Mapping[str, object]) -> list[HandoffEvent]:
    manifest = {str(key): payload[key] for key in payload}
    root = Path(str(manifest.get("root", "."))).resolve()
    session_id = str(manifest.get("session_id", "")).strip()
    if not session_id:
        return []
    events: list[HandoffEvent] = []
    for raw_entry in _manifest_entries(manifest):
        events.append(
            PrepareStepEvent(
                event="prepare_step",
                session_id=session_id,
                root=str(root),
                entry={str(key): raw_entry[key] for key in raw_entry},
            )
        )
        status = str(raw_entry.get("status", "")).strip()
        if status == "started":
            continue
        sequence = raw_entry.get("sequence")
        assert isinstance(sequence, int)
        events.append(
            RecordStepEvent(
                event="record_step",
                session_id=session_id,
                sequence=sequence,
                status=status,
                exit_code=_optional_int(raw_entry.get("exit_code")),
                analysis_state=_optional_str(raw_entry.get("analysis_state")),
                completed_at_utc=_optional_str(raw_entry.get("completed_at_utc")),
            )
        )
    return events


def _event_from_payload(payload: Mapping[str, object]) -> HandoffEvent | None:
    event_name = str(payload.get("event", "")).strip()
    if event_name == "prepare_step":
        raw_entry = payload.get("entry")
        assert isinstance(raw_entry, Mapping)
        return PrepareStepEvent(
            event="prepare_step",
            session_id=str(payload.get("session_id", "")).strip(),
            root=str(payload.get("root", "")).strip() or ".",
            entry={str(key): raw_entry[key] for key in raw_entry},
        )
    if event_name == "record_step":
        return RecordStepEvent(
            event="record_step",
            session_id=str(payload.get("session_id", "")).strip(),
            sequence=int(payload["sequence"]),
            status=str(payload["status"]),
            exit_code=_optional_int(payload.get("exit_code")),
            analysis_state=_optional_str(payload.get("analysis_state")),
            completed_at_utc=_optional_str(payload.get("completed_at_utc")),
        )
    return None


def _event_to_payload(event: HandoffEvent) -> JSONObject:
    if isinstance(event, PrepareStepEvent):
        return {
            "event": event.event,
            "session_id": event.session_id,
            "root": event.root,
            "entry": event.entry,
        }
    return {
        "event": event.event,
        "session_id": event.session_id,
        "sequence": event.sequence,
        "status": event.status,
        "exit_code": event.exit_code,
        "analysis_state": event.analysis_state,
        "completed_at_utc": event.completed_at_utc,
    }


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


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
