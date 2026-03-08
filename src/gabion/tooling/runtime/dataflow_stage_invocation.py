from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from gabion.tooling.runtime.execution_envelope import ExecutionEnvelope


@dataclass(frozen=True)
class DeltaBundleCommandInvocation:
    root: Path
    report_path: Path
    strictness: str | None
    aspf_state_json: Path | None
    aspf_delta_jsonl: Path | None
    aspf_import_state: tuple[Path, ...]

    def to_execution_envelope(self) -> ExecutionEnvelope:
        return ExecutionEnvelope.for_delta_bundle(
            root=self.root,
            report_path=self.report_path,
            strictness=self.strictness,
            allow_external=None,
            aspf_state_json=self.aspf_state_json,
            aspf_delta_jsonl=self.aspf_delta_jsonl,
            aspf_import_state=self.aspf_import_state,
        )


def _delta_bundle_option_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--root", default=".")
    parser.add_argument("--report")
    parser.add_argument("--strictness")
    parser.add_argument("--aspf-state-json")
    parser.add_argument("--aspf-delta-jsonl")
    parser.add_argument("--aspf-import-state", action="append", default=[])
    return parser


def parse_delta_bundle_command_invocation(
    command: Sequence[str],
) -> DeltaBundleCommandInvocation | None:
    tokens = [str(token) for token in command]
    if not tokens:
        return None
    try:
        check_index = tokens.index("check")
    except ValueError:
        return None
    if check_index + 1 >= len(tokens):
        return None
    if tokens[check_index + 1] != "delta-bundle":
        return None
    option_tokens = tokens[check_index + 2 :]
    parser = _delta_bundle_option_parser()
    try:
        parsed, _ = parser.parse_known_args(option_tokens)
    except SystemExit:
        return None
    if not parsed.report:
        return None
    strictness = str(parsed.strictness) if parsed.strictness else None
    aspf_state_json = (
        Path(str(parsed.aspf_state_json))
        if parsed.aspf_state_json is not None
        else None
    )
    aspf_delta_jsonl = (
        Path(str(parsed.aspf_delta_jsonl))
        if parsed.aspf_delta_jsonl is not None
        else None
    )
    import_state = tuple(Path(str(path)) for path in list(parsed.aspf_import_state))
    return DeltaBundleCommandInvocation(
        root=Path(str(parsed.root)),
        report_path=Path(str(parsed.report)),
        strictness=strictness,
        aspf_state_json=aspf_state_json,
        aspf_delta_jsonl=aspf_delta_jsonl,
        aspf_import_state=import_state,
    )


__all__ = [
    "DeltaBundleCommandInvocation",
    "parse_delta_bundle_command_invocation",
]

