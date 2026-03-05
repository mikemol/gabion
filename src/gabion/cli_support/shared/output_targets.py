from __future__ import annotations

from pathlib import Path
from typing import Callable
import json

from gabion.cli_support.shared import output_emitters, result_emitters


def parse_lint_line(line: str) -> dict[str, object] | None:
    return result_emitters.parse_lint_line_entry(line)


def collect_lint_entries(
    lines: list[str],
    *,
    check_deadline_fn: Callable[[], None],
) -> list[dict[str, object]]:
    return result_emitters.collect_lint_entries(lines, check_deadline_fn=check_deadline_fn)


def is_stdout_target(
    target: object,
    *,
    stdout_alias: str,
    stdout_path: str,
) -> bool:
    return output_emitters.is_stdout_target(
        target,
        stdout_alias=stdout_alias,
        stdout_path=stdout_path,
    )


def normalize_output_target(
    target: str | Path,
    *,
    stdout_alias: str,
    stdout_path: str,
) -> str:
    return output_emitters.normalize_output_target(
        target,
        stdout_alias=stdout_alias,
        stdout_path=stdout_path,
    )


def write_text_to_target(
    target: str | Path,
    payload: str,
    *,
    ensure_trailing_newline: bool = False,
    encoding: str = "utf-8",
    stdout_alias: str,
    stdout_path: str,
) -> None:
    output_emitters.write_text_to_target(
        normalize_output_target(target, stdout_alias=stdout_alias, stdout_path=stdout_path),
        payload,
        ensure_trailing_newline=ensure_trailing_newline,
        encoding=encoding,
        stdout_alias=stdout_alias,
        stdout_path=stdout_path,
    )


def emit_result_json_to_stdout(
    *,
    payload: object,
    write_text_to_target_fn: Callable[..., None],
    stdout_path: str,
) -> None:
    write_text_to_target_fn(
        stdout_path,
        json.dumps(payload, indent=2, sort_keys=False),
        ensure_trailing_newline=True,
    )
