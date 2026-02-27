#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Sequence

from gabion.tooling import aspf_handoff


def _prepare(args: argparse.Namespace) -> int:
    prepared = aspf_handoff.prepare_step(
        root=Path(args.root),
        session_id=args.session_id,
        step_id=args.step_id,
        command_profile=args.command_profile,
        manifest_path=Path(args.manifest) if args.manifest else None,
        state_root=Path(args.state_root) if args.state_root else None,
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

    prepared = aspf_handoff.prepare_step(
        root=Path(args.root),
        session_id=args.session_id,
        step_id=args.step_id,
        command_profile=args.command_profile,
        manifest_path=Path(args.manifest) if args.manifest else None,
        state_root=Path(args.state_root) if args.state_root else None,
    )
    command_with_aspf = tuple([*raw_command, *aspf_handoff.aspf_cli_args(prepared)])
    completed = subprocess.run(list(command_with_aspf), check=False)
    exit_code = int(completed.returncode)
    analysis_state = "succeeded" if exit_code == 0 else "failed"
    status = "success" if exit_code == 0 else "failed"
    ok = aspf_handoff.record_step(
        manifest_path=prepared.manifest_path,
        session_id=prepared.session_id,
        sequence=prepared.sequence,
        status=status,
        exit_code=exit_code,
        analysis_state=analysis_state,
    )
    payload = {
        "ok": bool(ok),
        "exit_code": exit_code,
        "status": status,
        "analysis_state": analysis_state,
        "sequence": prepared.sequence,
        "session_id": prepared.session_id,
        "step_id": prepared.step_id,
        "command_profile": prepared.command_profile,
        "state_path": str(prepared.state_path),
        "delta_path": str(prepared.delta_path),
        "import_state_paths": [str(path) for path in prepared.import_state_paths],
        "manifest_path": str(prepared.manifest_path),
        "command": list(raw_command),
        "command_with_aspf": list(command_with_aspf),
    }
    print(json.dumps(payload, indent=2, sort_keys=False))
    if exit_code != 0:
        return exit_code
    return 0 if ok else 1


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
