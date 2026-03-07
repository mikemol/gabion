# gabion:decision_protocol_module
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping

import typer

from gabion.commands.lint_parser import parse_lint_line
from gabion.json_types import JSONObject

CheckDeadlineFn = Callable[[], None]
WriteTextToTargetFn = Callable[..., None]
WriteLintSarifFn = Callable[[str, list[dict[str, object]]], None]
RenderDeadlineProfileMarkdownFn = Callable[[Mapping[str, object]], str]




class TargetStreamRouter:
    def __init__(self, *, max_open_streams: int = 8) -> None:
        self._max_open_streams = max(1, int(max_open_streams))
        self._streams: dict[str, tuple[object, str]] = {}

    def _stream_for_target(self, *, target: str, encoding: str):
        existing = self._streams.get(target)
        if existing is not None:
            stream, current_encoding = existing
            if current_encoding == encoding:
                return stream
            stream.close()
            self._streams.pop(target, None)
        if len(self._streams) >= self._max_open_streams:
            oldest_target = next(iter(self._streams))
            oldest_stream, _ = self._streams.pop(oldest_target)
            oldest_stream.close()
        stream = Path(target).open("w", encoding=encoding)
        self._streams[target] = (stream, encoding)
        return stream

    def write(self, *, target: str, payload: str, encoding: str = "utf-8") -> None:
        stream = self._stream_for_target(target=target, encoding=encoding)
        stream.seek(0)
        stream.truncate(0)
        stream.write(payload)
        stream.flush()

    def close(self) -> None:
        for target in list(self._streams):
            stream, _ = self._streams.pop(target)
            stream.close()


def parse_lint_line_entry(line: str) -> dict[str, object] | None:
    entry = parse_lint_line(line)
    if entry is None:
        return None
    return {
        **entry.model_dump(),
        "severity": "warning",
    }


def collect_lint_entries(
    lines: list[str],
    *,
    check_deadline_fn: CheckDeadlineFn,
) -> list[dict[str, object]]:
    check_deadline_fn()
    entries: list[dict[str, object]] = []
    for line in lines:
        check_deadline_fn()
        parsed = parse_lint_line_entry(line)
        if parsed is not None:
            entries.append(parsed)
    return entries


def write_lint_jsonl(
    target: str,
    entries: list[dict[str, object]],
    *,
    write_text_to_target_fn: WriteTextToTargetFn,
) -> None:
    payload = "\n".join(json.dumps(entry, sort_keys=False) for entry in entries)
    write_text_to_target_fn(
        target,
        payload,
        ensure_trailing_newline=bool(payload),
    )


def emit_lint_outputs(
    lint_lines: list[str],
    *,
    lint: bool,
    lint_jsonl: Path | None,
    lint_sarif: Path | None,
    check_deadline_fn: CheckDeadlineFn,
    write_lint_jsonl_fn: Callable[[str, list[dict[str, object]]], None],
    write_lint_sarif_fn: WriteLintSarifFn,
    lint_entries: list[dict[str, object]] | None = None,
) -> None:
    check_deadline_fn()
    if lint:
        for line in lint_lines:
            check_deadline_fn()
            typer.echo(line)
    if lint_jsonl or lint_sarif:
        entries = (
            lint_entries
            if lint_entries is not None
            else collect_lint_entries(lint_lines, check_deadline_fn=check_deadline_fn)
        )
        if lint_jsonl is not None:
            write_lint_jsonl_fn(str(lint_jsonl), entries)
        if lint_sarif is not None:
            write_lint_sarif_fn(str(lint_sarif), entries)


def emit_timeout_profile_artifacts(
    result: Mapping[str, object],
    *,
    root: Path,
    render_deadline_profile_markdown_fn: RenderDeadlineProfileMarkdownFn,
) -> None:
    timeout_context = result.get("timeout_context")
    if not isinstance(timeout_context, Mapping):
        return
    profile = timeout_context.get("deadline_profile")
    if not isinstance(profile, Mapping):
        return
    artifact_dir = root / "artifacts" / "out"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    profile_json_path = artifact_dir / "deadline_profile.json"
    profile_md_path = artifact_dir / "deadline_profile.md"
    profile_json_path.write_text(json.dumps(profile, indent=2, sort_keys=False) + "\n")
    profile_md_path.write_text(
        render_deadline_profile_markdown_fn(profile) + "\n",
        encoding="utf-8",
    )
    typer.echo(f"Wrote deadline profile JSON: {profile_json_path}")
    typer.echo(f"Wrote deadline profile markdown: {profile_md_path}")


def nonzero_exit_causes(result: JSONObject) -> list[str]:
    causes: list[str] = []
    if bool(result.get("timeout", False)):
        analysis_state = str(result.get("analysis_state") or "unknown")
        causes.append(f"timeout (analysis_state={analysis_state})")
    violations = int(result.get("violations", 0) or 0)
    if violations > 0:
        causes.append(f"policy violations={violations}")
    type_ambiguities_raw = result.get("type_ambiguities")
    if isinstance(type_ambiguities_raw, list) and type_ambiguities_raw:
        causes.append(f"type ambiguities={len(type_ambiguities_raw)}")
    errors_raw = result.get("errors")
    if isinstance(errors_raw, list) and errors_raw:
        first_error = str(errors_raw[0])
        if len(errors_raw) > 1:
            causes.append(f"errors={len(errors_raw)} (first: {first_error})")
        else:
            causes.append(f"error: {first_error}")
    if not causes:
        analysis_state = str(result.get("analysis_state") or "unknown")
        causes.append(
            "no explicit violations/type ambiguities/errors were returned; "
            f"analysis_state={analysis_state}"
        )
    return causes


def emit_nonzero_exit_causes(result: JSONObject) -> None:
    exit_code = int(result.get("exit_code", 0) or 0)
    if exit_code == 0:
        return
    causes = "; ".join(nonzero_exit_causes(result))
    typer.echo(f"Non-zero exit ({exit_code}) cause(s): {causes}", err=True)


def emit_analysis_resume_summary(result: JSONObject) -> None:
    resume = result.get("analysis_resume")
    if not isinstance(resume, Mapping):
        return
    path = str(resume.get("state_path", "") or "")
    status = str(resume.get("status", "") or "")
    reused_files = int(resume.get("reused_files", 0) or 0)
    total_files = int(resume.get("total_files", 0) or 0)
    remaining_files = int(resume.get("remaining_files", 0) or 0)
    cache_verdict = str(resume.get("cache_verdict", "") or "")
    status_suffix = f" status={status}" if status else ""
    verdict_suffix = f" cache_verdict={cache_verdict}" if cache_verdict else ""
    typer.echo(
        "Resume state: "
        f"path={path or '<none>'} reused_files={reused_files}/{total_files} "
        f"remaining_files={remaining_files}{status_suffix}{verdict_suffix}"
    )
