#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Sequence

from gabion.tooling import aspf_lifecycle
from gabion.tooling import aspf_handoff
from gabion.tooling import dataflow_invocation_runner
from gabion.tooling import execution_envelope


def _load_state_object(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items()}


def _analysis_state_from_state_path(path: Path) -> str:
    payload = _load_state_object(path)
    state = payload.get("analysis_state")
    if isinstance(state, str) and state:
        return state
    resume_projection = payload.get("resume_projection")
    if isinstance(resume_projection, dict):
        projection_state = resume_projection.get("analysis_state")
        if isinstance(projection_state, str) and projection_state:
            return projection_state
    return "none"


def _prepare(args: argparse.Namespace) -> int:
    prepared = aspf_handoff.prepare_step(
        root=Path(args.root),
        session_id=args.session_id,
        step_id=args.step_id,
        command_profile=args.command_profile,
        manifest_path=Path(args.manifest) if args.manifest else None,
        state_root=Path(args.state_root) if args.state_root else None,
        write_manifest_projection=not args.no_manifest_projection,
    )
    payload = {
        "sequence": prepared.sequence,
        "session_id": prepared.session_id,
        "step_id": prepared.step_id,
        "command_profile": prepared.command_profile,
        "state_path": str(prepared.state_path),
        "delta_path": str(prepared.delta_path),
        "import_state_paths": [str(path) for path in prepared.import_state_paths],
        "manifest_path": str(prepared.manifest_path),
        "started_at_utc": prepared.started_at_utc,
        "aspf_cli_args": aspf_handoff.aspf_cli_args(prepared),
    }
    print(json.dumps(payload, indent=2, sort_keys=False))
    return 0


def _record(args: argparse.Namespace) -> int:
    ok = aspf_handoff.record_step(
        manifest_path=Path(args.manifest),
        session_id=args.session_id,
        sequence=int(args.sequence),
        status=args.status,
        exit_code=args.exit_code,
        analysis_state=args.analysis_state,
        write_manifest_projection=not args.no_manifest_projection,
    )
    print(json.dumps({"ok": bool(ok)}, indent=2, sort_keys=False))
    return 0 if ok else 1


def _run(args: argparse.Namespace) -> int:
    raw_command = [str(token) for token in (args.command or [])]
    if raw_command and raw_command[0] == "--":
        raw_command = raw_command[1:]
    if not raw_command:
        print(
            json.dumps(
                {"ok": False, "error": "missing command after `run --`"},
                indent=2,
                sort_keys=False,
            )
        )
        return 2

    lifecycle = aspf_lifecycle.AspfLifecycleConfig(
        enabled=True,
        root=Path(args.root),
        session_id=str(args.session_id or ""),
        manifest_path=Path(args.manifest) if args.manifest else Path("artifacts/out/aspf_handoff_manifest.json"),
        state_root=Path(args.state_root) if args.state_root else Path("artifacts/out/aspf_state"),
        write_manifest_projection=not args.no_manifest_projection,
        resume_import_policy="success_or_resumable_timeout",
    )
    invocation_runner = dataflow_invocation_runner.DataflowInvocationRunner()

    def _run_with_best_path(command: Sequence[str]) -> int:
        envelope = _parse_delta_bundle_envelope(command)
        if envelope is not None:
            return int(invocation_runner.run_delta_bundle(envelope).exit_code)
        raw_check = _parse_raw_check_args(command)
        if raw_check is not None:
            envelope_raw = execution_envelope.ExecutionEnvelope.for_raw(
                root=Path(args.root),
                aspf_state_json=raw_check.aspf_state_json,
                aspf_delta_jsonl=raw_check.aspf_delta_jsonl,
                aspf_import_state=tuple(raw_check.aspf_import_state),
            )
            return int(
                invocation_runner.run_raw(envelope_raw, raw_check.raw_args).exit_code
            )
        completed = subprocess.run(list(command), check=False)
        return int(completed.returncode)

    lifecycle_result = aspf_lifecycle.run_with_aspf_lifecycle(
        config=lifecycle,
        step_id=str(args.step_id),
        command_profile=str(args.command_profile),
        command=raw_command,
        run_command_fn=_run_with_best_path,
        analysis_state_from_state_path_fn=_analysis_state_from_state_path,
    )
    payload = {
        "ok": True,
        "exit_code": lifecycle_result.exit_code,
        "status": lifecycle_result.status,
        "analysis_state": lifecycle_result.analysis_state,
        "sequence": lifecycle_result.sequence,
        "session_id": lifecycle_result.session_id,
        "step_id": str(args.step_id),
        "command_profile": str(args.command_profile),
        "state_path": str(lifecycle_result.state_path) if lifecycle_result.state_path is not None else None,
        "delta_path": _option_value(lifecycle_result.command_with_aspf, "--aspf-delta-jsonl"),
        "import_state_paths": [str(path) for path in lifecycle_result.import_state_paths],
        "manifest_path": str(lifecycle_result.manifest_path) if lifecycle_result.manifest_path is not None else str(lifecycle.manifest_path),
        "command": list(raw_command),
        "command_with_aspf": list(lifecycle_result.command_with_aspf),
    }
    print(json.dumps(payload, indent=2, sort_keys=False))
    return int(lifecycle_result.exit_code)


class _RawCheckArgs:
    def __init__(
        self,
        *,
        raw_args: list[str],
        aspf_state_json: Path | None,
        aspf_delta_jsonl: Path | None,
        aspf_import_state: list[Path],
    ) -> None:
        self.raw_args = raw_args
        self.aspf_state_json = aspf_state_json
        self.aspf_delta_jsonl = aspf_delta_jsonl
        self.aspf_import_state = aspf_import_state


def _option_value(command: Sequence[str], option: str) -> str | None:
    tokens = list(command)
    for idx, token in enumerate(tokens):
        if token != option:
            continue
        if idx + 1 >= len(tokens):
            return None
        return str(tokens[idx + 1])
    return None


def _parse_delta_bundle_envelope(
    command: Sequence[str],
) -> execution_envelope.ExecutionEnvelope | None:
    tokens = [str(token) for token in command]
    try:
        check_index = tokens.index("check")
    except ValueError:
        return None
    if check_index + 1 >= len(tokens) or tokens[check_index + 1] != "delta-bundle":
        return None
    report_path: Path | None = None
    strictness: str | None = None
    aspf_state_json: Path | None = None
    aspf_delta_jsonl: Path | None = None
    aspf_import_state: list[Path] = []
    options = tokens[check_index + 2 :]
    idx = 0
    while idx < len(options):
        token = options[idx]
        if token in {"--report", "--strictness", "--aspf-state-json", "--aspf-delta-jsonl", "--aspf-import-state"}:
            if idx + 1 >= len(options):
                break
            value = options[idx + 1]
            if token == "--report":
                report_path = Path(value)
            elif token == "--strictness":
                strictness = value
            elif token == "--aspf-state-json":
                aspf_state_json = Path(value)
            elif token == "--aspf-delta-jsonl":
                aspf_delta_jsonl = Path(value)
            elif token == "--aspf-import-state":
                aspf_import_state.append(Path(value))
            idx += 2
            continue
        idx += 1
    if report_path is None:
        return None
    return execution_envelope.ExecutionEnvelope.for_delta_bundle(
        root=Path("."),
        report_path=report_path,
        strictness=strictness,
        allow_external=None,
        aspf_state_json=aspf_state_json,
        aspf_delta_jsonl=aspf_delta_jsonl,
        aspf_import_state=tuple(aspf_import_state),
    )


def _parse_raw_check_args(command: Sequence[str]) -> _RawCheckArgs | None:
    tokens = [str(token) for token in command]
    try:
        check_index = tokens.index("check")
    except ValueError:
        return None
    if check_index + 1 >= len(tokens) or tokens[check_index + 1] != "raw":
        return None
    options = tokens[check_index + 2 :]
    if options and options[0] == "--":
        options = options[1:]
    aspf_state_json: Path | None = None
    aspf_delta_jsonl: Path | None = None
    aspf_import_state: list[Path] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token in {"--aspf-state-json", "--aspf-delta-jsonl", "--aspf-import-state"} and idx + 1 < len(tokens):
            value = tokens[idx + 1]
            if token == "--aspf-state-json":
                aspf_state_json = Path(value)
            elif token == "--aspf-delta-jsonl":
                aspf_delta_jsonl = Path(value)
            elif token == "--aspf-import-state":
                aspf_import_state.append(Path(value))
            idx += 2
            continue
        idx += 1
    return _RawCheckArgs(
        raw_args=options,
        aspf_state_json=aspf_state_json,
        aspf_delta_jsonl=aspf_delta_jsonl,
        aspf_import_state=aspf_import_state,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ASPF cross-script handoff helper.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Reserve next handoff step and emit CLI args.",
    )
    prepare.add_argument("--root", default=".")
    prepare.add_argument("--session-id", default="")
    prepare.add_argument("--step-id", required=True)
    prepare.add_argument("--command-profile", required=True)
    prepare.add_argument(
        "--manifest",
        default="artifacts/out/aspf_handoff_manifest.json",
    )
    prepare.add_argument(
        "--state-root",
        default="artifacts/out/aspf_state",
    )
    prepare.add_argument(
        "--no-manifest-projection",
        action="store_true",
        help="Skip writing manifest projection cache (journal remains canonical).",
    )

    record = subparsers.add_parser(
        "record",
        help="Record completion for a previously prepared step.",
    )
    record.add_argument(
        "--manifest",
        default="artifacts/out/aspf_handoff_manifest.json",
    )
    record.add_argument("--session-id", required=True)
    record.add_argument("--sequence", type=int, required=True)
    record.add_argument("--status", required=True)
    record.add_argument("--exit-code", type=int, default=None)
    record.add_argument("--analysis-state", default=None)
    record.add_argument(
        "--no-manifest-projection",
        action="store_true",
        help="Skip writing manifest projection cache (journal remains canonical).",
    )

    run = subparsers.add_parser(
        "run",
        help=(
            "Prepare step, execute command with ASPF args, record completion, and emit JSON."
        ),
    )
    run.add_argument("--root", default=".")
    run.add_argument("--session-id", default="")
    run.add_argument("--step-id", required=True)
    run.add_argument("--command-profile", required=True)
    run.add_argument(
        "--manifest",
        default="artifacts/out/aspf_handoff_manifest.json",
    )
    run.add_argument(
        "--state-root",
        default="artifacts/out/aspf_state",
    )
    run.add_argument(
        "--no-manifest-projection",
        action="store_true",
        help="Skip writing manifest projection cache (journal remains canonical).",
    )
    run.add_argument("command", nargs=argparse.REMAINDER)

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "prepare":
        return _prepare(args)
    if args.cmd == "record":
        return _record(args)
    if args.cmd == "run":
        return _run(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
