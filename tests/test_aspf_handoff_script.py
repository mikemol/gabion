from __future__ import annotations

import json
from pathlib import Path

from scripts import aspf_handoff as aspf_handoff_script


# gabion:evidence E:call_footprint::tests/test_aspf_handoff_script.py::test_script_prepare_and_record_roundtrip::aspf_handoff.py::scripts.aspf_handoff.main
def test_script_prepare_and_record_roundtrip(
    tmp_path: Path,
    capsys,
) -> None:
    manifest = tmp_path / "artifacts/out/aspf_handoff_manifest.json"
    state_root = tmp_path / "artifacts/out/aspf_state"
    root = tmp_path

    prepare_exit = aspf_handoff_script.main(
        [
            "prepare",
            "--root",
            str(root),
            "--session-id",
            "session-script",
            "--step-id",
            "script.check.run",
            "--command-profile",
            "check.run",
            "--manifest",
            str(manifest),
            "--state-root",
            str(state_root),
        ]
    )
    assert prepare_exit == 0
    prepare_payload = json.loads(capsys.readouterr().out)
    assert prepare_payload["session_id"] == "session-script"
    assert prepare_payload["sequence"] == 1
    assert prepare_payload["aspf_cli_args"][0] == "--aspf-state-json"

    record_exit = aspf_handoff_script.main(
        [
            "record",
            "--manifest",
            str(manifest),
            "--session-id",
            "session-script",
            "--sequence",
            str(prepare_payload["sequence"]),
            "--status",
            "success",
            "--exit-code",
            "0",
            "--analysis-state",
            "succeeded",
        ]
    )
    assert record_exit == 0
    record_payload = json.loads(capsys.readouterr().out)
    assert record_payload == {"ok": True}


# gabion:evidence E:call_footprint::tests/test_aspf_handoff_script.py::test_script_record_returns_nonzero_when_entry_missing::aspf_handoff.py::scripts.aspf_handoff.main
def test_script_record_returns_nonzero_when_entry_missing(
    tmp_path: Path,
    capsys,
) -> None:
    manifest = tmp_path / "artifacts/out/aspf_handoff_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "session_id": "session-script",
                "root": str(tmp_path),
                "entries": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = aspf_handoff_script.main(
        [
            "record",
            "--manifest",
            str(manifest),
            "--session-id",
            "session-script",
            "--sequence",
            "9",
            "--status",
            "failed",
            "--exit-code",
            "1",
            "--analysis-state",
            "failed",
        ]
    )
    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"ok": False}
